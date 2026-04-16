import math

import torch
import torch.nn as nn

from training.networks import SongUNet

from .utils import batch_scalar_like


def noise_features(c_noise: torch.Tensor) -> torch.Tensor:
    """Small sinusoidal embedding used for noise level conditioning."""

    c_noise = c_noise.unsqueeze(1)
    return torch.cat([c_noise, torch.sin(c_noise), torch.cos(c_noise)], dim=1)


def time_features(t: torch.Tensor) -> torch.Tensor:
    """Sinusoidal embedding for normalized scalar time (unused in current toy flow)."""

    t = t.unsqueeze(1)
    return torch.cat([t, torch.sin(2 * math.pi * t), torch.cos(2 * math.pi * t)], dim=1)


class ToyEDMDenoiser(nn.Module):
    """2D MLP denoiser with EDM preconditioning.

    Predicts x0 via:
      x0_hat = c_skip(sigma) * x + c_out(sigma) * F_theta(c_in(sigma) * x, c_noise)
    with the same coefficient definitions as EDM.
    """

    def __init__(self, hidden_dim: int = 128, sigma_data: float = 0.5):
        super().__init__()
        self.generative_family = "ve"
        self.sigma_data = float(sigma_data)
        self.model = nn.Sequential(
            nn.Linear(2 + 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        sigma = sigma.clamp_min(1e-6)
        sigma2 = sigma.square()
        sigma_data2 = self.sigma_data ** 2

        c_skip = sigma_data2 / (sigma2 + sigma_data2)
        c_out = sigma * self.sigma_data / torch.sqrt(sigma2 + sigma_data2)
        c_in = 1.0 / torch.sqrt(sigma2 + sigma_data2)
        c_noise = torch.log(sigma) / 4.0

        h = torch.cat([batch_scalar_like(c_in, x) * x, noise_features(c_noise)], dim=1)
        f_x = self.model(h)
        return batch_scalar_like(c_skip, x) * x + batch_scalar_like(c_out, x) * f_x


class ToyScoreModel(nn.Module):
    """2D MLP score model for VE corruption x_sigma = x0 + sigma * eps.

    `forward()` returns an x0 estimate to stay API-compatible with reverse/eval code:
      x0_hat = x + sigma^2 * s_theta(x, sigma).
    """

    def __init__(self, hidden_dim: int = 128, sigma_data: float = 0.5):
        super().__init__()
        self.generative_family = "ve"
        self.sigma_data = float(sigma_data)
        self.model = nn.Sequential(
            nn.Linear(2 + 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )

    def predict_score(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        sigma = sigma.clamp_min(1e-6)
        sigma2 = sigma.square()
        sigma_data2 = self.sigma_data ** 2
        c_in = 1.0 / torch.sqrt(sigma2 + sigma_data2)
        c_noise = torch.log(sigma) / 4.0
        h = torch.cat([batch_scalar_like(c_in, x) * x, noise_features(c_noise)], dim=1)
        return self.model(h)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        sigma = sigma.clamp_min(1e-6)
        score = self.predict_score(x, sigma)
        return x + batch_scalar_like(sigma.square(), x) * score


class ToyRectifiedFlowModel(nn.Module):
    """2D MLP rectified-flow velocity model with endpoint-compatible forward output."""

    def __init__(self, hidden_dim: int = 128, sigma_max: float = 1.0):
        super().__init__()
        self.generative_family = "rectified_flow"
        self.sigma_max = max(float(sigma_max), 1e-8)
        self.model = nn.Sequential(
            nn.Linear(2 + 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )

    def _time(self, sigma: torch.Tensor) -> torch.Tensor:
        return (sigma / self.sigma_max).clamp(0.0, 1.0)

    def predict_velocity(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        t = self._time(sigma)
        h = torch.cat([x, time_features(t)], dim=1)
        return self.model(h)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        t = self._time(sigma).clamp(0.0, 1.0)
        velocity = self.predict_velocity(x, sigma)
        return x + batch_scalar_like(1.0 - t, x) * velocity


class ControlNet(nn.Module):
    """Control policy phi for constrained forward perturbations delta_k.

    Input = [x_ref_k, gap_k, embed(log(sigma_k)/4)] where
      gap_k = x_ctrl_k - x_ref_k.
    Output is projected by the rollout to satisfy ||delta_k|| <= kappa_k * Delta_sigma_k.
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 + 2 + 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x_ref: torch.Tensor, gap: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        sigma = sigma.clamp_min(1e-6)
        sigma_feat = noise_features(torch.log(sigma) / 4.0)
        h = torch.cat([x_ref, gap, sigma_feat], dim=1)
        return self.net(h)


def _num_groups(num_channels: int) -> int:
    """Pick a valid GroupNorm group count for the given channel size."""

    for groups in [32, 16, 8, 4, 2, 1]:
        if num_channels % groups == 0:
            return groups
    return 1


class _ConvResidualBlock(nn.Module):
    """Simple FiLM-conditioned residual block for image backends."""

    def __init__(self, channels: int):
        super().__init__()
        groups = _num_groups(channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.film = nn.Linear(channels, channels * 2)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.film(cond).chunk(2, dim=1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]
        h = self.norm1(x)
        h = h * (1.0 + scale) + shift
        h = self.act(h)
        h = self.conv1(h)
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return x + h


def _small_songunet_channel_mult(img_resolution: int) -> list[int]:
    """Return a shallow DDPM++ pyramid sized for toy-scale images."""

    if img_resolution >= 24:
        return [1, 2, 2]
    if img_resolution >= 12:
        return [1, 2]
    return [1]


def _small_songunet_attn_resolutions(img_resolution: int, channel_mult: list[int]) -> list[int]:
    """Place a single attention block at the deepest resolution when possible."""

    if len(channel_mult) <= 1:
        return []
    return [max(int(img_resolution) >> (len(channel_mult) - 1), 1)]


class _SmallSongUNetBackbone(nn.Module):
    """Small DDPM++-style SongUNet for low-resolution toy image experiments."""

    def __init__(self, img_resolution: int, in_channels: int = 3, hidden_dim: int = 64, num_blocks: int = 2):
        super().__init__()
        channel_mult = _small_songunet_channel_mult(int(img_resolution))
        attn_resolutions = _small_songunet_attn_resolutions(int(img_resolution), channel_mult)
        self.model = SongUNet(
            img_resolution=int(img_resolution),
            in_channels=int(in_channels),
            out_channels=int(in_channels),
            label_dim=0,
            model_channels=int(hidden_dim),
            channel_mult=channel_mult,
            num_blocks=int(num_blocks),
            attn_resolutions=attn_resolutions,
            dropout=0.0,
            embedding_type="positional",
            channel_mult_noise=1,
            encoder_type="standard",
            decoder_type="standard",
            resample_filter=[1, 1],
        )

    def forward(self, x: torch.Tensor, noise_labels: torch.Tensor) -> torch.Tensor:
        return self.model(x, noise_labels.to(torch.float32).flatten(), class_labels=None)


class ImageEDMDenoiser(nn.Module):
    """Small convolutional EDM denoiser for image-shaped inputs."""

    def __init__(self, in_channels: int = 3, hidden_dim: int = 64, sigma_data: float = 0.5, num_blocks: int = 4):
        super().__init__()
        self.generative_family = "ve"
        self.sigma_data = float(sigma_data)
        self.in_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.noise_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.blocks = nn.ModuleList([_ConvResidualBlock(hidden_dim) for _ in range(num_blocks)])
        self.out_norm = nn.GroupNorm(_num_groups(hidden_dim), hidden_dim)
        self.out_conv = nn.Conv2d(hidden_dim, in_channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        sigma = sigma.clamp_min(1e-6)
        sigma2 = sigma.square()
        sigma_data2 = self.sigma_data ** 2

        c_skip = sigma_data2 / (sigma2 + sigma_data2)
        c_out = sigma * self.sigma_data / torch.sqrt(sigma2 + sigma_data2)
        c_in = 1.0 / torch.sqrt(sigma2 + sigma_data2)
        c_noise = torch.log(sigma) / 4.0

        cond = self.noise_mlp(noise_features(c_noise))
        h = self.in_conv(batch_scalar_like(c_in, x) * x)
        for block in self.blocks:
            h = block(h, cond)
        f_x = self.out_conv(self.act(self.out_norm(h)))
        return batch_scalar_like(c_skip, x) * x + batch_scalar_like(c_out, x) * f_x


class ImageSongUNetDenoiser(nn.Module):
    """Small SongUNet/DDPM++-style EDM denoiser for toy image experiments."""

    def __init__(self, img_resolution: int, in_channels: int = 3, hidden_dim: int = 64, sigma_data: float = 0.5):
        super().__init__()
        self.generative_family = "ve"
        self.sigma_data = float(sigma_data)
        self.model = _SmallSongUNetBackbone(
            img_resolution=int(img_resolution),
            in_channels=int(in_channels),
            hidden_dim=int(hidden_dim),
        )

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        sigma = sigma.clamp_min(1e-6)
        sigma2 = sigma.square()
        sigma_data2 = self.sigma_data ** 2

        c_skip = sigma_data2 / (sigma2 + sigma_data2)
        c_out = sigma * self.sigma_data / torch.sqrt(sigma2 + sigma_data2)
        c_in = 1.0 / torch.sqrt(sigma2 + sigma_data2)
        c_noise = torch.log(sigma) / 4.0

        f_x = self.model(batch_scalar_like(c_in, x) * x, c_noise)
        return batch_scalar_like(c_skip, x) * x + batch_scalar_like(c_out, x) * f_x


class ImageScoreModel(nn.Module):
    """Small convolutional score model with x0-compatible forward output."""

    def __init__(self, in_channels: int = 3, hidden_dim: int = 64, sigma_data: float = 0.5, num_blocks: int = 4):
        super().__init__()
        self.generative_family = "ve"
        self.sigma_data = float(sigma_data)
        self.in_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.noise_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.blocks = nn.ModuleList([_ConvResidualBlock(hidden_dim) for _ in range(num_blocks)])
        self.out_norm = nn.GroupNorm(_num_groups(hidden_dim), hidden_dim)
        self.out_conv = nn.Conv2d(hidden_dim, in_channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()

    def predict_score(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        sigma = sigma.clamp_min(1e-6)
        sigma2 = sigma.square()
        sigma_data2 = self.sigma_data ** 2
        c_in = 1.0 / torch.sqrt(sigma2 + sigma_data2)
        c_noise = torch.log(sigma) / 4.0
        cond = self.noise_mlp(noise_features(c_noise))
        h = self.in_conv(batch_scalar_like(c_in, x) * x)
        for block in self.blocks:
            h = block(h, cond)
        return self.out_conv(self.act(self.out_norm(h)))

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        sigma = sigma.clamp_min(1e-6)
        score = self.predict_score(x, sigma)
        return x + batch_scalar_like(sigma.square(), x) * score


class ImageSongUNetScoreModel(nn.Module):
    """Small SongUNet-based score model with x0-compatible forward output."""

    def __init__(self, img_resolution: int, in_channels: int = 3, hidden_dim: int = 64, sigma_data: float = 0.5):
        super().__init__()
        self.generative_family = "ve"
        self.sigma_data = float(sigma_data)
        self.model = _SmallSongUNetBackbone(
            img_resolution=int(img_resolution),
            in_channels=int(in_channels),
            hidden_dim=int(hidden_dim),
        )

    def predict_score(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        sigma = sigma.clamp_min(1e-6)
        sigma2 = sigma.square()
        sigma_data2 = self.sigma_data ** 2
        c_in = 1.0 / torch.sqrt(sigma2 + sigma_data2)
        c_noise = torch.log(sigma) / 4.0
        return self.model(batch_scalar_like(c_in, x) * x, c_noise)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        sigma = sigma.clamp_min(1e-6)
        score = self.predict_score(x, sigma)
        return x + batch_scalar_like(sigma.square(), x) * score


class ImageRectifiedFlowModel(nn.Module):
    """Small convolutional rectified-flow model with endpoint-compatible forward output."""

    def __init__(self, in_channels: int = 3, hidden_dim: int = 64, sigma_max: float = 1.0, num_blocks: int = 4):
        super().__init__()
        self.generative_family = "rectified_flow"
        self.sigma_max = max(float(sigma_max), 1e-8)
        self.in_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.time_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.blocks = nn.ModuleList([_ConvResidualBlock(hidden_dim) for _ in range(num_blocks)])
        self.out_norm = nn.GroupNorm(_num_groups(hidden_dim), hidden_dim)
        self.out_conv = nn.Conv2d(hidden_dim, in_channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()

    def _time(self, sigma: torch.Tensor) -> torch.Tensor:
        return (sigma / self.sigma_max).clamp(0.0, 1.0)

    def predict_velocity(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        t = self._time(sigma)
        cond = self.time_mlp(time_features(t))
        h = self.in_conv(x)
        for block in self.blocks:
            h = block(h, cond)
        return self.out_conv(self.act(self.out_norm(h)))

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        t = self._time(sigma).clamp(0.0, 1.0)
        velocity = self.predict_velocity(x, sigma)
        return x + batch_scalar_like(1.0 - t, x) * velocity


class ImageSongUNetRectifiedFlowModel(nn.Module):
    """Small SongUNet-based rectified-flow model with endpoint-compatible output."""

    def __init__(self, img_resolution: int, in_channels: int = 3, hidden_dim: int = 64, sigma_max: float = 1.0):
        super().__init__()
        self.generative_family = "rectified_flow"
        self.sigma_max = max(float(sigma_max), 1e-8)
        self.model = _SmallSongUNetBackbone(
            img_resolution=int(img_resolution),
            in_channels=int(in_channels),
            hidden_dim=int(hidden_dim),
        )

    def _time(self, sigma: torch.Tensor) -> torch.Tensor:
        return (sigma / self.sigma_max).clamp(0.0, 1.0)

    def predict_velocity(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        return self.model(x, self._time(sigma))

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        t = self._time(sigma).clamp(0.0, 1.0)
        velocity = self.predict_velocity(x, sigma)
        return x + batch_scalar_like(1.0 - t, x) * velocity


class ImageControlNet(nn.Module):
    """Convolutional control network for image trajectory perturbations."""

    def __init__(self, in_channels: int = 3, hidden_dim: int = 64, num_blocks: int = 3):
        super().__init__()
        self.in_conv = nn.Conv2d(in_channels * 2, hidden_dim, kernel_size=3, padding=1)
        self.noise_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.blocks = nn.ModuleList([_ConvResidualBlock(hidden_dim) for _ in range(num_blocks)])
        self.out_norm = nn.GroupNorm(_num_groups(hidden_dim), hidden_dim)
        self.out_conv = nn.Conv2d(hidden_dim, in_channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()
        # Start from delta ~= 0 so image controls do not immediately saturate the hard constraint.
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x_ref: torch.Tensor, gap: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        sigma = sigma.clamp_min(1e-6)
        cond = self.noise_mlp(noise_features(torch.log(sigma) / 4.0))
        h = self.in_conv(torch.cat([x_ref, gap], dim=1))
        for block in self.blocks:
            h = block(h, cond)
        return self.out_conv(self.act(self.out_norm(h)))


def set_requires_grad(module: nn.Module, flag: bool) -> None:
    """Enable/disable autograd for all parameters in `module`."""

    for p in module.parameters():
        p.requires_grad_(flag)
