import math
from dataclasses import dataclass

import torch


@dataclass
class ImageMarkovSchedule:
    family: str
    total_time: float
    dt: torch.Tensor
    drift_coeff: torch.Tensor
    beta: torch.Tensor
    g: torch.Tensor
    step_sigma: torch.Tensor
    weights: torch.Tensor

    def to_snapshot_dict(self):
        return {
            "family": self.family,
            "total_time": float(self.total_time),
            "dt": self.dt.detach().cpu(),
            "drift_coeff": self.drift_coeff.detach().cpu(),
            "beta": self.beta.detach().cpu(),
            "g": self.g.detach().cpu(),
            "step_sigma": self.step_sigma.detach().cpu(),
            "weights": self.weights.detach().cpu(),
        }

    @classmethod
    def from_snapshot_dict(cls, payload, device):
        return cls(
            family=str(payload["family"]),
            total_time=float(payload["total_time"]),
            dt=torch.as_tensor(payload["dt"], device=device, dtype=torch.float32),
            drift_coeff=torch.as_tensor(payload["drift_coeff"], device=device, dtype=torch.float32),
            beta=torch.as_tensor(payload["beta"], device=device, dtype=torch.float32),
            g=torch.as_tensor(payload["g"], device=device, dtype=torch.float32),
            step_sigma=torch.as_tensor(payload["step_sigma"], device=device, dtype=torch.float32),
            weights=torch.as_tensor(payload["weights"], device=device, dtype=torch.float32),
        )


def cosine_alpha_bar(t: torch.Tensor, *, s: float) -> torch.Tensor:
    scale = math.pi / 2.0
    numerator = torch.cos(((t + s) / (1.0 + s)) * scale).square()
    denominator = math.cos((s / (1.0 + s)) * scale) ** 2
    return numerator / denominator


def build_step_weights(*, step_sigma: torch.Tensor, schedule: str) -> torch.Tensor:
    if schedule == "uniform":
        return torch.ones_like(step_sigma)
    if schedule == "sigma_sq":
        return step_sigma.square()
    if schedule == "inv_sigma_sq":
        return step_sigma.square().clamp(min=1e-8).reciprocal()
    raise ValueError(f"Unsupported markov weight schedule: {schedule}")


def build_image_markov_schedule(
    *,
    num_steps: int,
    total_time: float,
    beta_min: float,
    beta_max: float,
    sde_family: str,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    weight_schedule: str = "uniform",
    cosine_s: float = 0.008,
):
    if num_steps < 1:
        raise ValueError("num_steps must be at least 1.")
    dt = torch.full((num_steps,), float(total_time) / float(num_steps), device=device, dtype=dtype)

    if sde_family == "vp_linear":
        beta = torch.linspace(beta_min, beta_max, steps=num_steps, device=device, dtype=dtype)
        mean_scale = torch.exp(-0.5 * beta * dt)
        step_sigma = (1.0 - torch.exp(-beta * dt)).clamp(min=1e-8).sqrt()
        drift_coeff = (mean_scale - 1.0) / dt
        g = beta.clamp(min=1e-8).sqrt()
    elif sde_family == "vp_cosine":
        time_edges = torch.linspace(0.0, 1.0, steps=num_steps + 1, device=device, dtype=dtype)
        alpha_bar = cosine_alpha_bar(time_edges, s=cosine_s).clamp(min=1e-8)
        alpha_ratio = (alpha_bar[1:] / alpha_bar[:-1]).clamp(min=1e-8, max=0.999999)
        mean_scale = alpha_ratio.sqrt()
        step_sigma = (1.0 - alpha_ratio).clamp(min=1e-8).sqrt()
        drift_coeff = (mean_scale - 1.0) / dt
        beta = (-alpha_ratio.log()) / dt
        g = beta.clamp(min=1e-8).sqrt()
    else:
        raise ValueError(f"Unsupported markov sde_family: {sde_family}")

    weights = build_step_weights(step_sigma=step_sigma, schedule=weight_schedule)
    return ImageMarkovSchedule(
        family=sde_family,
        total_time=float(total_time),
        dt=dt,
        drift_coeff=drift_coeff,
        beta=beta,
        g=g,
        step_sigma=step_sigma,
        weights=weights,
    )


def forward_drift(x: torch.Tensor, *, drift_coeff: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
    return drift_coeff * x + control


def initialize_terminal_stats(*, example: torch.Tensor):
    mean = torch.zeros_like(example[0]).detach()
    var = torch.ones_like(example[0]).detach()
    return {"mean": mean, "var": var}


def update_terminal_stats(*, terminal_states: torch.Tensor, current: dict | None, momentum: float, min_var: float = 1e-6):
    batch_mean = terminal_states.mean(dim=0).detach()
    batch_var = terminal_states.var(dim=0, unbiased=False).clamp(min=min_var).detach()
    if current is None:
        return {"mean": batch_mean, "var": batch_var}
    mean = current["mean"] * momentum + batch_mean * (1.0 - momentum)
    var = current["var"] * momentum + batch_var * (1.0 - momentum)
    return {"mean": mean, "var": var.clamp(min=min_var)}


def rollout_markov_forward_images(
    *,
    clean_images: torch.Tensor,
    net,
    schedule: ImageMarkovSchedule,
    class_labels=None,
    augment_labels=None,
    zero_control: bool = False,
    control_scale: float = 1.0,
):
    current = clean_images
    states = [current]
    means = []
    noises = []
    controls = []
    for step_idx in range(schedule.dt.shape[0]):
        sigma_step = schedule.step_sigma[step_idx].expand(clean_images.shape[0])
        if zero_control:
            control = torch.zeros_like(current)
        else:
            control = net.control(current, sigma_step, class_labels, augment_labels=augment_labels) * float(control_scale)
        mean = current + forward_drift(current, drift_coeff=schedule.drift_coeff[step_idx], control=control) * schedule.dt[step_idx]
        noise = torch.randn_like(current)
        current = mean + schedule.step_sigma[step_idx] * noise
        states.append(current)
        means.append(mean)
        noises.append(noise)
        controls.append(control)
    return {
        "states": torch.stack(states, dim=1),
        "means": torch.stack(means, dim=1),
        "noises": torch.stack(noises, dim=1),
        "controls": torch.stack(controls, dim=1),
    }


def local_denoise_loss(
    *,
    net,
    rollout: dict,
    schedule: ImageMarkovSchedule,
    sigma_data: float,
    class_labels=None,
    augment_labels=None,
):
    batch_size, num_steps = rollout["means"].shape[:2]
    observations = rollout["states"][:, 1:, ...].reshape(batch_size * num_steps, *rollout["states"].shape[2:])
    targets = rollout["means"].reshape(batch_size * num_steps, *rollout["means"].shape[2:])
    sigma = schedule.step_sigma.view(1, num_steps, 1, 1, 1).expand(batch_size, num_steps, 1, 1, 1).reshape(batch_size * num_steps)
    if class_labels is not None:
        class_labels = class_labels.unsqueeze(1).expand(batch_size, num_steps, class_labels.shape[1]).reshape(batch_size * num_steps, class_labels.shape[1])
    if augment_labels is not None:
        augment_labels = augment_labels.unsqueeze(1).expand(batch_size, num_steps, augment_labels.shape[1]).reshape(batch_size * num_steps, augment_labels.shape[1])
    weight = (sigma.view(-1, 1, 1, 1) ** 2 + sigma_data ** 2) / (sigma.view(-1, 1, 1, 1) * sigma_data) ** 2
    pred = net(observations, sigma, class_labels, augment_labels=augment_labels)
    sq_error = weight * ((pred - targets) ** 2)
    step_weights = schedule.weights.view(1, num_steps, 1, 1, 1).expand(batch_size, num_steps, *sq_error.shape[1:]).reshape_as(sq_error)
    return (sq_error * step_weights).mean()


def compute_control_cost(*, rollout: dict, schedule: ImageMarkovSchedule):
    per_step = rollout["controls"].square().mean(dim=[2, 3, 4])
    return (per_step * schedule.dt.view(1, -1)).mean()


@torch.no_grad()
def sample_markov_reverse_images(
    *,
    net,
    schedule: ImageMarkovSchedule,
    terminal_stats: dict,
    latents: torch.Tensor,
    class_labels=None,
    reverse_control_scale: float = 1.0,
    noise_scale: float = 1.0,
    base_control_scale: float = 1.0,
):
    current = terminal_stats["mean"].to(device=latents.device, dtype=latents.dtype).unsqueeze(0)
    current = current + latents * terminal_stats["var"].to(device=latents.device, dtype=latents.dtype).clamp(min=1e-6).sqrt().unsqueeze(0)

    for step_idx in range(schedule.dt.shape[0] - 1, -1, -1):
        sigma_step = schedule.step_sigma[step_idx].expand(current.shape[0])
        denoised = net(current, sigma_step, class_labels)
        score = (denoised - current) / schedule.step_sigma[step_idx].square().clamp(min=1e-8)
        control = net.control(current, sigma_step, class_labels) * float(base_control_scale) * float(reverse_control_scale)
        dt_step = schedule.dt[step_idx]
        score_scale = schedule.step_sigma[step_idx].square() / dt_step
        reverse_drift = -forward_drift(current, drift_coeff=schedule.drift_coeff[step_idx], control=control) + score_scale * score
        current = current + reverse_drift * dt_step
        if step_idx > 0:
            current = current + schedule.step_sigma[step_idx] * float(noise_scale) * torch.randn_like(current)
    return current
