import math
from typing import Optional

import torch


def build_sigma_levels(sigma_min: float, sigma_max: float, n_steps: int, device: torch.device) -> torch.Tensor:
    """Build VE noise ladder [sigma_0, ..., sigma_N] with sigma_0 = 0."""

    return torch.cat(
        [
            torch.zeros(1, device=device),
            torch.logspace(math.log10(sigma_min), math.log10(sigma_max), n_steps, device="cpu").to(device),
        ]
    )


def _truncated_log_sigma_cdf_bounds(
    *,
    sigma_min: float,
    sigma_max: float,
    p_mean: float,
    p_std: float,
) -> tuple[torch.distributions.Normal, torch.Tensor, torch.Tensor]:
    """Return the warmup-law normal and its truncated CDF bounds in log-sigma space."""

    if sigma_min <= 0.0 or sigma_max <= 0.0:
        raise ValueError(f"sigma_min and sigma_max must be positive, got {sigma_min}, {sigma_max}")
    if sigma_min >= sigma_max:
        raise ValueError(f"sigma_min must be < sigma_max, got {sigma_min} >= {sigma_max}")
    if p_std <= 0.0:
        raise ValueError(f"p_std must be > 0, got {p_std}")

    normal = torch.distributions.Normal(
        loc=torch.tensor(float(p_mean), dtype=torch.float64),
        scale=torch.tensor(float(p_std), dtype=torch.float64),
    )
    z_min = torch.tensor(math.log(float(sigma_min)), dtype=torch.float64)
    z_max = torch.tensor(math.log(float(sigma_max)), dtype=torch.float64)
    cdf_min = normal.cdf(z_min)
    cdf_max = normal.cdf(z_max)
    if float((cdf_max - cdf_min).item()) <= 1e-12:
        raise ValueError("Truncated warmup-law support has zero mass on [sigma_min, sigma_max].")
    return normal, cdf_min, cdf_max


def sample_sigmas_log_normal(
    batch_size: int,
    *,
    sigma_min: float,
    sigma_max: float,
    device: torch.device,
    p_mean: float = -1.2,
    p_std: float = 1.2,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Sample continuous EDM warmup sigmas from the clamped log-normal law."""

    out_dtype = torch.float32 if dtype is None else dtype
    rnd = torch.randn(batch_size, device=device, dtype=out_dtype)
    sigma = torch.exp(rnd * float(p_std) + float(p_mean))
    return sigma.clamp(min=float(sigma_min), max=float(sigma_max))


def assign_sigmas_to_nearest_levels(sigmas: torch.Tensor, sigma_levels: torch.Tensor) -> torch.Tensor:
    """Map continuous sigmas to the nearest positive ladder point for diagnostics only."""

    if sigma_levels.numel() < 2:
        raise ValueError(f"sigma_levels must contain at least 2 values, got {sigma_levels.numel()}")
    d = (sigmas.unsqueeze(1) - sigma_levels[1:].unsqueeze(0)).abs()
    nearest = torch.argmin(d, dim=1)
    return nearest + 1


def build_sigma_levels_from_warmup_quantiles(
    sigma_min: float,
    sigma_max: float,
    n_steps: int,
    device: torch.device,
    p_mean: float = -1.2,
    p_std: float = 1.2,
) -> torch.Tensor:
    """Build the positive continuation ladder from warmup-law midpoint quantiles."""

    if n_steps <= 0:
        raise ValueError(f"n_steps must be > 0, got {n_steps}")
    normal, cdf_min, cdf_max = _truncated_log_sigma_cdf_bounds(
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        p_mean=p_mean,
        p_std=p_std,
    )
    probs = (torch.arange(1, n_steps + 1, dtype=torch.float64) - 0.5) / float(n_steps)
    trunc_cdf = cdf_min + probs * (cdf_max - cdf_min)
    z = normal.icdf(trunc_cdf.clamp(min=1e-12, max=1.0 - 1e-12))
    sigma_positive = torch.exp(z).to(device=device, dtype=torch.float32)
    if not torch.all(sigma_positive[1:] > sigma_positive[:-1]):
        raise ValueError("Warmup-quantile continuation ladder must be strictly increasing.")
    return torch.cat([torch.zeros(1, device=device, dtype=sigma_positive.dtype), sigma_positive])


def build_log_sigma_quantile_cell_edges(
    sigma_min: float,
    sigma_max: float,
    n_steps: int,
    *,
    p_mean: float = -1.2,
    p_std: float = 1.2,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Build exact log-sigma cell edges for the midpoint-quantile continuation grid."""

    if n_steps <= 0:
        raise ValueError(f"n_steps must be > 0, got {n_steps}")
    normal, cdf_min, cdf_max = _truncated_log_sigma_cdf_bounds(
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        p_mean=p_mean,
        p_std=p_std,
    )
    out_device = torch.device("cpu") if device is None else device
    out_dtype = torch.float32 if dtype is None else dtype

    z_min = torch.tensor(math.log(float(sigma_min)), dtype=torch.float64)
    z_max = torch.tensor(math.log(float(sigma_max)), dtype=torch.float64)
    if n_steps == 1:
        z_edges = torch.stack([z_min, z_max])
    else:
        probs = torch.arange(1, n_steps, dtype=torch.float64) / float(n_steps)
        trunc_cdf = cdf_min + probs * (cdf_max - cdf_min)
        z_inner = normal.icdf(trunc_cdf.clamp(min=1e-12, max=1.0 - 1e-12))
        z_edges = torch.cat([z_min.view(1), z_inner, z_max.view(1)])
    if not torch.all(z_edges[1:] > z_edges[:-1]):
        raise ValueError("Warmup-quantile cell edges must be strictly increasing.")
    return z_edges.to(device=out_device, dtype=out_dtype)


def sample_target_indices(batch_size: int, sigma_levels: torch.Tensor) -> torch.Tensor:
    """Uniformly sample training target steps i in {1, ..., N}."""

    return torch.randint(1, sigma_levels.numel(), (batch_size,), device=sigma_levels.device)


def sample_target_indices_log_normal(
    batch_size: int,
    sigma_levels: torch.Tensor,
    p_mean: float = -1.2,
    p_std: float = 1.2,
) -> torch.Tensor:
    """EDM-style sigma sampling snapped to the nearest continuation ladder point."""

    sigma = sample_sigmas_log_normal(
        batch_size,
        sigma_min=float(sigma_levels[1].item()),
        sigma_max=float(sigma_levels[-1].item()),
        device=sigma_levels.device,
        p_mean=p_mean,
        p_std=p_std,
        dtype=sigma_levels.dtype,
    )
    return assign_sigmas_to_nearest_levels(sigma, sigma_levels)
