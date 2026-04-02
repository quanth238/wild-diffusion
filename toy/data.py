import math
from typing import Optional, Tuple

import torch


def build_circle_gmm_centers(n_modes: int, radius: float, device: torch.device) -> torch.Tensor:
    """Place `n_modes` Gaussian centers uniformly on a circle of given radius."""

    angles = torch.linspace(0, 2 * math.pi, n_modes + 1, device=device)[:-1]
    centers = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1) * radius
    return centers


def sample_gmm(batch_size: int, centers: torch.Tensor, std: float) -> torch.Tensor:
    """Sample x0 from an equally-weighted isotropic Gaussian mixture."""

    mode_ids = torch.randint(0, centers.shape[0], (batch_size,), device=centers.device)
    noise = torch.randn(batch_size, 2, device=centers.device) * std
    return centers[mode_ids] + noise


def build_fixed_gmm_dataset_per_mode(
    centers: torch.Tensor,
    std: float,
    points_per_mode: int,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create limited-data pool with exactly `points_per_mode` samples per mode."""

    if points_per_mode <= 0:
        raise ValueError(f"points_per_mode must be > 0, got {points_per_mode}")
    n_modes = int(centers.shape[0])
    mode_ids = torch.arange(n_modes, device=centers.device).repeat_interleave(points_per_mode)
    generator = None
    if seed is not None:
        generator = torch.Generator(device=centers.device)
        generator.manual_seed(int(seed))
    noise = torch.randn(mode_ids.numel(), centers.shape[1], device=centers.device, generator=generator) * std
    x = centers[mode_ids] + noise
    return x, mode_ids


def build_random_gmm_dataset(
    total_samples: int,
    centers: torch.Tensor,
    std: float,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create finite random dataset from the full GMM population."""

    if total_samples <= 0:
        raise ValueError(f"total_samples must be > 0, got {total_samples}")
    generator = None
    if seed is not None:
        generator = torch.Generator(device=centers.device)
        generator.manual_seed(int(seed))
    mode_ids = torch.randint(0, centers.shape[0], (total_samples,), device=centers.device, generator=generator)
    noise = torch.randn(total_samples, centers.shape[1], device=centers.device, generator=generator) * std
    x = centers[mode_ids] + noise
    return x, mode_ids


def sample_from_pool(pool: torch.Tensor, batch_size: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Uniformly sample mini-batch rows from a finite pool."""

    if pool.ndim != 2:
        raise ValueError(f"pool must be 2D [N, D], got shape {tuple(pool.shape)}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    idx = torch.randint(0, pool.shape[0], (batch_size,), device=pool.device, generator=generator)
    return pool[idx]


def sample_with_labels(batch_size: int, centers: torch.Tensor, std: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample GMM data and return mode ids alongside points."""

    mode_ids = torch.randint(0, centers.shape[0], (batch_size,), device=centers.device)
    noise = torch.randn(batch_size, 2, device=centers.device) * std
    x = centers[mode_ids] + noise
    return x, mode_ids
