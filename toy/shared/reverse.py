from typing import Optional

import torch

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
    """Generate full reverse trajectories from sigma_N to sigma_0."""

    if sample_terminal_batch_fn is None:
        x = torch.randn(n_samples, 2, device=device) * sigma_levels[-1]
    else:
        x = sample_terminal_batch_fn(n_samples, sigma_levels[-1]).to(device=device)
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

