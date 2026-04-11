from typing import Optional

import torch

from .objective import predict_velocity, rf_time_levels_from_sigma_levels, terminal_prior_scale_from_family
from ..utils import batch_scalar_like


def reverse_posterior_mean(
    x: torch.Tensor,
    x0_pred: torch.Tensor,
    sigma: torch.Tensor,
    sigma_prev: torch.Tensor,
) -> torch.Tensor:
    """Posterior mean for VE step k->k-1 under x = x0 + sigma * eps."""

    sigma2 = sigma.square().clamp_min(1e-12)
    alpha = sigma_prev.square() / sigma2
    alpha = batch_scalar_like(alpha, x)
    return alpha * x + (1.0 - alpha) * x0_pred


def reverse_posterior_std(sigma: torch.Tensor, sigma_prev: torch.Tensor) -> torch.Tensor:
    """Posterior std for VE reverse transition k->k-1."""

    sigma2 = sigma.square().clamp_min(1e-12)
    var = sigma_prev.square() * (1.0 - sigma_prev.square() / sigma2)
    return torch.sqrt(var.clamp_min(0.0))


def _is_rectified_flow_model(denoiser) -> bool:
    return str(getattr(denoiser, "generative_family", "")).lower() == "rectified_flow"


@torch.no_grad()
def sample_rectified_flow_paths_from_source(
    denoiser,
    x_source: torch.Tensor,
    sigma_levels: torch.Tensor,
) -> torch.Tensor:
    """Deterministic RF integration from source noise at t=0 to the data endpoint at t=1."""

    n_steps = int(sigma_levels.numel() - 1)
    x = x_source.clone()
    states = [x]
    t_levels = rf_time_levels_from_sigma_levels(sigma_levels).to(device=x.device, dtype=x.dtype)

    for k in range(n_steps):
        sigma = torch.full((x.shape[0],), float(sigma_levels[k].item()), device=x.device, dtype=x.dtype)
        t_cur = float(t_levels[k].item())
        t_next = float(t_levels[k + 1].item())
        dt = max(t_next - t_cur, 0.0)
        velocity = predict_velocity(
            denoiser,
            x,
            sigma,
            sigma_max=float(sigma_levels[-1].item()),
        )
        x = x + dt * velocity
        states.append(x)
    return torch.stack(states, dim=1)


@torch.no_grad()
def _reverse_rectified_flow_paths_from_terminal(
    denoiser,
    x_terminal: torch.Tensor,
    sigma_levels: torch.Tensor,
) -> torch.Tensor:
    """Deterministic reverse-time RF integration from the data endpoint back to source noise."""

    n_steps = int(sigma_levels.numel() - 1)
    x = x_terminal.clone()
    states = [None for _ in range(n_steps + 1)]
    states[n_steps] = x
    t_levels = rf_time_levels_from_sigma_levels(sigma_levels).to(device=x.device, dtype=x.dtype)

    for k in range(n_steps, 0, -1):
        sigma = torch.full((x.shape[0],), float(sigma_levels[k].item()), device=x.device, dtype=x.dtype)
        t_cur = float(t_levels[k].item())
        t_prev = float(t_levels[k - 1].item())
        dt = max(t_cur - t_prev, 0.0)
        velocity = predict_velocity(
            denoiser,
            x,
            sigma,
            sigma_max=float(sigma_levels[-1].item()),
        )
        x = x - dt * velocity
        states[k - 1] = x
    return torch.stack(states, dim=1)


@torch.no_grad()
def sample_reverse_paths(
    denoiser,
    sigma_levels: torch.Tensor,
    n_samples: int,
    device: torch.device,
    stochastic: bool = True,
    noise_schedule: Optional[torch.Tensor] = None,
    sample_terminal_batch_fn=None,
) -> torch.Tensor:
    """Generate family-aware sample paths from the model prior/source state."""

    terminal_scale = terminal_prior_scale_from_family(
        getattr(denoiser, "generative_family", ""),
        float(sigma_levels[-1].item()),
    )
    if sample_terminal_batch_fn is None:
        x = torch.randn(n_samples, 2, device=device, dtype=sigma_levels.dtype) * terminal_scale
    else:
        x = sample_terminal_batch_fn(n_samples, terminal_scale).to(device=device, dtype=sigma_levels.dtype)
    if _is_rectified_flow_model(denoiser):
        return sample_rectified_flow_paths_from_source(
            denoiser=denoiser,
            x_source=x,
            sigma_levels=sigma_levels,
        )
    n_steps = sigma_levels.numel() - 1
    states = [None for _ in range(n_steps + 1)]
    states[n_steps] = x

    if noise_schedule is not None:
        if noise_schedule.shape[0] < (n_steps + 1):
            raise ValueError(f"noise_schedule first dim must be >= {n_steps + 1}, got {noise_schedule.shape[0]}")
        if tuple(noise_schedule.shape[1:]) != tuple(x.shape):
            raise ValueError(
                "noise_schedule must have shape [>=N+1, *x.shape], "
                f"got {tuple(noise_schedule.shape)}"
            )

    for k in range(n_steps, 0, -1):
        sigma = torch.full((n_samples,), sigma_levels[k], device=device, dtype=x.dtype)
        sigma_next = torch.full((n_samples,), sigma_levels[k - 1], device=device, dtype=x.dtype)
        x0_pred = denoiser(x, sigma)
        mean = reverse_posterior_mean(x, x0_pred, sigma, sigma_next)
        if stochastic and k > 1:
            std = reverse_posterior_std(sigma, sigma_next)
            if noise_schedule is None:
                eps = torch.randn_like(x)
            else:
                eps = noise_schedule[k].to(device=x.device, dtype=x.dtype)
            x = mean + batch_scalar_like(std, x) * eps
        else:
            x = mean
        states[k - 1] = x
    return torch.stack(states, dim=1)


@torch.no_grad()
def reverse_paths_from_terminal(
    denoiser,
    x_terminal: torch.Tensor,
    sigma_levels: torch.Tensor,
    stochastic: bool = False,
    noise_schedule: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Reverse trajectories from provided terminal states x_k at the final index."""

    if _is_rectified_flow_model(denoiser):
        return _reverse_rectified_flow_paths_from_terminal(
            denoiser=denoiser,
            x_terminal=x_terminal,
            sigma_levels=sigma_levels,
        )

    n_steps = sigma_levels.numel() - 1
    x = x_terminal.clone()
    states = [None for _ in range(n_steps + 1)]
    states[n_steps] = x

    if noise_schedule is not None:
        if noise_schedule.shape[0] < (n_steps + 1):
            raise ValueError(f"noise_schedule first dim must be >= {n_steps + 1}, got {noise_schedule.shape[0]}")
        if tuple(noise_schedule.shape[1:]) != tuple(x.shape):
            raise ValueError(
                "noise_schedule must have shape [>=N+1, *x.shape], "
                f"got {tuple(noise_schedule.shape)}"
            )

    for k in range(n_steps, 0, -1):
        sigma = torch.full((x.shape[0],), sigma_levels[k], device=x.device, dtype=x.dtype)
        sigma_prev = torch.full((x.shape[0],), sigma_levels[k - 1], device=x.device, dtype=x.dtype)
        x0_pred = denoiser(x, sigma)
        mean = reverse_posterior_mean(x, x0_pred, sigma, sigma_prev)
        if stochastic and k > 1:
            std = reverse_posterior_std(sigma, sigma_prev)
            if noise_schedule is None:
                eps = torch.randn_like(x)
            else:
                eps = noise_schedule[k].to(device=x.device, dtype=x.dtype)
            x = mean + batch_scalar_like(std, x) * eps
        else:
            x = mean
        states[k - 1] = x

    return torch.stack(states, dim=1)


def generated_data_path_index_from_family(generative_family: str) -> int:
    """Return the path index that corresponds to the generated data endpoint."""

    return -1 if str(generative_family).strip().lower() == "rectified_flow" else 0


def generated_data_path_index_from_denoiser(denoiser) -> int:
    """Return the path index of generated samples for the provided model family."""

    return generated_data_path_index_from_family(getattr(denoiser, "generative_family", ""))
