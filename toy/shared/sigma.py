import math

import torch


def build_sigma_levels(sigma_min: float, sigma_max: float, n_steps: int, device: torch.device) -> torch.Tensor:
    """Build VE noise ladder [sigma_0, ..., sigma_N] with sigma_0 = 0."""

    return torch.cat(
        [
            torch.zeros(1, device=device),
            torch.logspace(math.log10(sigma_min), math.log10(sigma_max), n_steps, device="cpu").to(device),
        ]
    )


def sample_target_indices(batch_size: int, sigma_levels: torch.Tensor) -> torch.Tensor:
    """Uniformly sample training target steps i in {1, ..., N}."""

    return torch.randint(1, sigma_levels.numel(), (batch_size,), device=sigma_levels.device)


def sample_target_indices_log_normal(
    batch_size: int,
    sigma_levels: torch.Tensor,
    p_mean: float = -1.2,
    p_std: float = 1.2,
) -> torch.Tensor:
    """EDM-style sigma sampling with discretization to nearest ladder step."""

    rnd = torch.randn(batch_size, device=sigma_levels.device)
    sigma = torch.exp(rnd * p_std + p_mean).clamp(min=float(sigma_levels[1]), max=float(sigma_levels[-1]))
    d = (sigma.unsqueeze(1) - sigma_levels[1:].unsqueeze(0)).abs()
    nearest = torch.argmin(d, dim=1)
    return nearest + 1

