from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class DatasetBundle:
    name: str
    raw_points: np.ndarray
    train_points: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor
    bounds_min: torch.Tensor
    bounds_max: torch.Tensor

    def destandardize(self, points: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(device=points.device, dtype=points.dtype)
        std = self.std.to(device=points.device, dtype=points.dtype)
        return points * std + mean


def available_datasets() -> tuple[str, ...]:
    return ("eight_gaussians", "two_moons", "spiral")


def _sample_eight_gaussians(num_samples: int, rng: np.random.Generator, noise: float) -> np.ndarray:
    radius = 2.0
    centers = []
    for idx in range(8):
        theta = 2 * math.pi * idx / 8
        centers.append([radius * math.cos(theta), radius * math.sin(theta)])
    centers = np.asarray(centers, dtype=np.float32)
    assignments = rng.integers(0, len(centers), size=num_samples)
    points = centers[assignments] + noise * rng.standard_normal((num_samples, 2), dtype=np.float32)
    return points.astype(np.float32)


def _sample_two_moons(num_samples: int, rng: np.random.Generator, noise: float) -> np.ndarray:
    first_count = num_samples // 2
    second_count = num_samples - first_count
    theta_first = rng.uniform(0.0, math.pi, size=first_count).astype(np.float32)
    theta_second = rng.uniform(0.0, math.pi, size=second_count).astype(np.float32)

    first = np.stack([np.cos(theta_first), np.sin(theta_first)], axis=1)
    second = np.stack([1.0 - np.cos(theta_second), 1.0 - np.sin(theta_second) - 0.5], axis=1)
    points = np.concatenate([first, second], axis=0)
    points += noise * rng.standard_normal(points.shape, dtype=np.float32)
    points *= 1.5
    return points.astype(np.float32)


def _sample_spiral(num_samples: int, rng: np.random.Generator, noise: float) -> np.ndarray:
    half = num_samples // 2
    remain = num_samples - half

    def make_arm(count: int, sign: float) -> np.ndarray:
        t = np.sqrt(rng.uniform(0.0, 1.0, size=count)).astype(np.float32) * 4.0 * math.pi
        radius = 0.35 * t
        x = sign * radius * np.cos(t)
        y = sign * radius * np.sin(t)
        arm = np.stack([x, y], axis=1)
        arm += noise * rng.standard_normal(arm.shape, dtype=np.float32)
        return arm.astype(np.float32)

    points = np.concatenate([make_arm(half, 1.0), make_arm(remain, -1.0)], axis=0)
    points *= 1.1
    return points.astype(np.float32)


def _generate_points(name: str, num_samples: int, rng: np.random.Generator, noise: float) -> np.ndarray:
    if name == "eight_gaussians":
        return _sample_eight_gaussians(num_samples=num_samples, rng=rng, noise=noise)
    if name == "two_moons":
        return _sample_two_moons(num_samples=num_samples, rng=rng, noise=noise)
    if name == "spiral":
        return _sample_spiral(num_samples=num_samples, rng=rng, noise=noise)
    raise ValueError(f"Unknown dataset '{name}'. Available: {', '.join(available_datasets())}")


def build_dataset(
    *,
    name: str,
    num_samples: int,
    seed: int,
    noise: float = 0.08,
    standardize: bool = True,
    clamp_margin: float = 1.0,
) -> DatasetBundle:
    rng = np.random.default_rng(seed)
    raw_points = _generate_points(name=name, num_samples=num_samples, rng=rng, noise=noise)

    if standardize:
        mean_np = raw_points.mean(axis=0, keepdims=True).astype(np.float32)
        std_np = raw_points.std(axis=0, keepdims=True).astype(np.float32)
        std_np = np.where(std_np < 1e-6, 1.0, std_np)
        train_points = (raw_points - mean_np) / std_np
        mean = torch.from_numpy(mean_np.squeeze(0))
        std = torch.from_numpy(std_np.squeeze(0))
    else:
        train_points = raw_points.copy()
        mean = torch.zeros(2, dtype=torch.float32)
        std = torch.ones(2, dtype=torch.float32)

    train_tensor = torch.from_numpy(train_points.astype(np.float32))
    bounds_min = train_tensor.amin(dim=0) - clamp_margin
    bounds_max = train_tensor.amax(dim=0) + clamp_margin

    return DatasetBundle(
        name=name,
        raw_points=raw_points.astype(np.float32),
        train_points=train_tensor,
        mean=mean.to(torch.float32),
        std=std.to(torch.float32),
        bounds_min=bounds_min.to(torch.float32),
        bounds_max=bounds_max.to(torch.float32),
    )
