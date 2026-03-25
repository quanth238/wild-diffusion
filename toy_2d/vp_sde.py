from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class VPSchedule:
    times: torch.Tensor
    dt: torch.Tensor
    noise_rates: torch.Tensor
    weights: torch.Tensor
    reverse_noise_scales: torch.Tensor

    @property
    def num_steps(self) -> int:
        return int(self.noise_rates.shape[0])

    def to(self, *, device: torch.device, dtype: torch.dtype) -> "VPSchedule":
        return VPSchedule(
            times=self.times.to(device=device, dtype=dtype),
            dt=self.dt.to(device=device, dtype=dtype),
            noise_rates=self.noise_rates.to(device=device, dtype=dtype),
            weights=self.weights.to(device=device, dtype=dtype),
            reverse_noise_scales=self.reverse_noise_scales.to(device=device, dtype=dtype),
        )


def build_vp_schedule(
    *,
    num_steps: int,
    total_time: float,
    beta_start: float,
    beta_end: float,
    reverse_noise_scale: float,
    weight_mode: str = "uniform",
) -> VPSchedule:
    if num_steps < 1:
        raise ValueError("num_steps must be at least 1.")
    if total_time <= 0.0:
        raise ValueError("total_time must be positive.")
    if beta_start <= 0.0 or beta_end <= 0.0:
        raise ValueError("beta_start and beta_end must be positive.")
    if reverse_noise_scale <= 0.0:
        raise ValueError("reverse_noise_scale must be positive.")

    times = torch.linspace(0.0, float(total_time), steps=num_steps + 1, dtype=torch.float32)
    dt = times[1:] - times[:-1]
    noise_rates = torch.linspace(float(beta_start), float(beta_end), steps=num_steps, dtype=torch.float32)
    weights = build_loss_weights(noise_rates=noise_rates, mode=weight_mode)
    reverse_noise_scales = reverse_noise_scale * torch.sqrt(dt)
    return VPSchedule(
        times=times,
        dt=dt,
        noise_rates=noise_rates,
        weights=weights,
        reverse_noise_scales=reverse_noise_scales,
    )


def deserialize_vp_schedule(
    *,
    payload: dict,
    device: torch.device,
    dtype: torch.dtype,
) -> VPSchedule:
    return VPSchedule(
        times=_to_tensor(payload["times"], device=device, dtype=dtype),
        dt=_to_tensor(payload["dt"], device=device, dtype=dtype),
        noise_rates=_to_tensor(payload["noise_rates"], device=device, dtype=dtype),
        weights=_to_tensor(payload["weights"], device=device, dtype=dtype),
        reverse_noise_scales=_to_tensor(payload["reverse_noise_scales"], device=device, dtype=dtype),
    )


def build_loss_weights(*, noise_rates: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "uniform":
        return torch.ones_like(noise_rates)
    if mode == "noise_rate":
        return noise_rates / noise_rates.mean().clamp_min(1e-12)
    if mode == "sqrt_noise_rate":
        return torch.sqrt(noise_rates / noise_rates.mean().clamp_min(1e-12))
    raise ValueError(f"Unsupported weight_mode: {mode}")


def sample_reference_paths(
    *,
    x0: torch.Tensor,
    schedule: VPSchedule,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, data_dim = x0.shape
    path = torch.empty(batch_size, schedule.num_steps + 1, data_dim, device=x0.device, dtype=x0.dtype)
    reference_noise = torch.randn(batch_size, schedule.num_steps, data_dim, device=x0.device, dtype=x0.dtype)
    path[:, 0, :] = x0

    for step_idx in range(schedule.num_steps):
        state = path[:, step_idx, :]
        noise_rate = schedule.noise_rates[step_idx]
        dt = schedule.dt[step_idx]
        drift = -0.5 * noise_rate * state * dt
        diffusion = torch.sqrt(noise_rate * dt) * reference_noise[:, step_idx, :]
        path[:, step_idx + 1, :] = state + drift + diffusion

    return path, reference_noise


def compute_vp_reference_target(
    *,
    reference_noise: torch.Tensor,
    controls: torch.Tensor,
    schedule: VPSchedule,
) -> torch.Tensor:
    control_scale = torch.sqrt(schedule.dt / schedule.noise_rates).view(1, -1, 1)
    return reference_noise + control_scale * controls


def compute_control_cost(*, controls: torch.Tensor, schedule: VPSchedule) -> torch.Tensor:
    per_step_cost = controls.square().sum(dim=2)
    return (per_step_cost * schedule.dt.view(1, -1)).sum(dim=1)


def compute_weighted_path_mse(
    *,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    schedule: VPSchedule,
) -> torch.Tensor:
    if predictions.shape != targets.shape:
        raise ValueError(
            f"predictions and targets must have the same shape, got {tuple(predictions.shape)} vs {tuple(targets.shape)}"
        )
    per_step_sq_error = (predictions - targets).square().sum(dim=2)
    weighted = per_step_sq_error * schedule.weights.view(1, -1) * schedule.dt.view(1, -1)
    return weighted.sum(dim=1)


def _to_tensor(value, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)
