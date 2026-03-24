from dataclasses import dataclass
from typing import Callable, Optional

import torch

from ..data import (
    build_circle_gmm_centers,
    build_fixed_gmm_dataset_per_mode,
    build_random_gmm_dataset,
    sample_from_pool,
    sample_gmm,
)


@dataclass
class DatasetBundle:
    """Dataset contract consumed by trainer/experiment.

    This keeps toy/image backends interchangeable as long as they expose:
    - `sample_train_batch(batch_size) -> x0`
    - `sample_val_batch(batch_size) -> x0`
    """

    name: str
    centers: torch.Tensor
    train_pool: Optional[torch.Tensor]
    train_pool_labels: Optional[torch.Tensor]
    val_pool: torch.Tensor
    sample_train_batch: Callable[[int], torch.Tensor]
    sample_val_batch: Callable[[int], torch.Tensor]


def build_dataset_bundle(cfg, device: torch.device) -> DatasetBundle:
    """Factory for dataset backends.

    Current implementation supports `toy_gmm`; new image datasets should add a
    branch here that returns the same `DatasetBundle` interface.
    """

    if cfg.dataset_kind != "toy_gmm":
        raise NotImplementedError(
            f"Unsupported dataset_kind='{cfg.dataset_kind}'. "
            "Template extension point: add a new builder in toy/data_backends/provider.py for your image dataset."
        )

    centers = build_circle_gmm_centers(cfg.n_modes, cfg.mode_radius, device=device)
    train_pool = None
    train_pool_labels = None
    if cfg.limited_data_enabled:
        train_pool, train_pool_labels = build_fixed_gmm_dataset_per_mode(
            centers=centers,
            std=cfg.data_std,
            points_per_mode=cfg.train_points_per_mode,
            seed=cfg.seed,
        )

    val_pool, _ = build_random_gmm_dataset(
        total_samples=cfg.val_size,
        centers=centers,
        std=cfg.data_std,
        seed=cfg.seed + 1001,
    )

    def sample_train(batch_size: int) -> torch.Tensor:
        if train_pool is not None:
            return sample_from_pool(train_pool, batch_size)
        return sample_gmm(batch_size, centers, cfg.data_std)

    def sample_val(batch_size: int) -> torch.Tensor:
        return sample_from_pool(val_pool, batch_size)

    return DatasetBundle(
        name="toy_gmm",
        centers=centers,
        train_pool=train_pool,
        train_pool_labels=train_pool_labels,
        val_pool=val_pool,
        sample_train_batch=sample_train,
        sample_val_batch=sample_val,
    )
