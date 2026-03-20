import math

import torch
import torch.nn as nn


def noise_features(c_noise: torch.Tensor) -> torch.Tensor:
    c_noise = c_noise.unsqueeze(1)
    return torch.cat([c_noise, torch.sin(c_noise), torch.cos(c_noise)], dim=1)


def time_features(t: torch.Tensor) -> torch.Tensor:
    t = t.unsqueeze(1)
    return torch.cat([t, torch.sin(2 * math.pi * t), torch.cos(2 * math.pi * t)], dim=1)


class ToyEDMDenoiser(nn.Module):
    def __init__(self, hidden_dim: int = 128, sigma_data: float = 0.5):
        super().__init__()
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

        h = torch.cat([c_in.unsqueeze(1) * x, noise_features(c_noise)], dim=1)
        f_x = self.model(h)
        return c_skip.unsqueeze(1) * x + c_out.unsqueeze(1) * f_x


class ControlNet(nn.Module):
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 + 3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = torch.cat([x, time_features(t)], dim=1)
        return self.net(h)


def set_requires_grad(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad_(flag)
