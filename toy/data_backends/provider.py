from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch

from ..data import (
    build_circle_gmm_centers,
    build_fixed_gmm_dataset_per_mode,
    build_random_gmm_dataset,
    sample_from_pool,
    sample_gmm,
)
from ..metrics import evaluate_mode_coverage


@dataclass
class DatasetBundle:
    """Dataset contract consumed by trainer/experiment.

    This keeps toy/image backends interchangeable as long as they expose:
    - `sample_train_batch(batch_size) -> x0`
    - `sample_val_batch(batch_size) -> x0`
    - `sample_population_batch(batch_size) -> x0`
    - `sample_terminal_batch(batch_size, sigma) -> x_N`
    - `evaluate_sample_metrics(samples_np) -> dict`
    - `estimate_sigma_data() -> float`
    """

    name: str
    data_shape: Tuple[int, ...]
    centers: Optional[torch.Tensor]
    train_pool: Optional[torch.Tensor]
    train_pool_labels: Optional[torch.Tensor]
    val_pool: torch.Tensor
    sample_train_batch: Callable[[int], torch.Tensor]
    sample_val_batch: Callable[[int], torch.Tensor]
    sample_population_batch: Callable[[int], torch.Tensor]
    sample_terminal_batch: Callable[[int, Union[float, torch.Tensor]], torch.Tensor]
    evaluate_sample_metrics: Callable[[object], Dict[str, float]]
    estimate_sigma_data: Callable[[], float]
    metadata: Dict[str, object] = field(default_factory=dict)


def build_dataset_bundle(cfg, device: torch.device) -> DatasetBundle:
    """Factory for dataset backends.

    Current implementation supports `toy_gmm`; new image datasets should add a
    branch here that returns the same `DatasetBundle` interface.
    """

    if cfg.dataset_kind == "toy_gmm":
        return _build_toy_gmm_bundle(cfg, device)
    if cfg.dataset_kind == "image_folder":
        return _build_image_folder_bundle(cfg, device)
    raise NotImplementedError(
        f"Unsupported dataset_kind='{cfg.dataset_kind}'. "
        "Template extension point: add a new builder in toy/data_backends/provider.py for your image dataset."
    )


def _build_toy_gmm_bundle(cfg, device: torch.device) -> DatasetBundle:
    """Build the current toy GMM dataset backend."""

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

    data_shape = tuple(int(v) for v in val_pool.shape[1:])

    def sample_train(batch_size: int) -> torch.Tensor:
        if train_pool is not None:
            return sample_from_pool(train_pool, batch_size)
        return sample_gmm(batch_size, centers, cfg.data_std)

    def sample_val(batch_size: int) -> torch.Tensor:
        return sample_from_pool(val_pool, batch_size)

    def sample_population(batch_size: int) -> torch.Tensor:
        return sample_gmm(batch_size, centers, cfg.data_std)

    def sample_terminal(batch_size: int, sigma: Union[float, torch.Tensor]) -> torch.Tensor:
        sigma_value = float(sigma.item()) if torch.is_tensor(sigma) else float(sigma)
        return torch.randn((batch_size, *data_shape), device=device) * sigma_value

    centers_np = centers.detach().cpu().numpy()

    def evaluate_samples(samples_np) -> Dict[str, float]:
        return evaluate_mode_coverage(samples_np, centers_np)

    def estimate_sigma() -> float:
        center_var = centers.var(dim=0, unbiased=False).mean()
        return float(torch.sqrt(center_var + cfg.data_std ** 2).item())

    return DatasetBundle(
        name="toy_gmm",
        data_shape=data_shape,
        centers=centers,
        train_pool=train_pool,
        train_pool_labels=train_pool_labels,
        val_pool=val_pool,
        sample_train_batch=sample_train,
        sample_val_batch=sample_val,
        sample_population_batch=sample_population,
        sample_terminal_batch=sample_terminal,
        evaluate_sample_metrics=evaluate_samples,
        estimate_sigma_data=estimate_sigma,
        metadata={"centers_np": centers_np, "enable_nearest_reference_distance": True, "num_classes": int(cfg.n_modes)},
    )


def _load_image_folder_tensors(root: str, image_size: int, image_channels: int):
    """Load an ImageFolder dataset into CPU tensors normalized to [-1, 1]."""

    if not root:
        raise ValueError("dataset_path must be provided for dataset_kind='image_folder'.")

    from pathlib import Path

    import PIL.Image

    if image_channels not in (1, 3):
        raise ValueError(f"image_channels must be 1 or 3, got {image_channels}")

    color_mode = "L" if image_channels == 1 else "RGB"
    root_path = Path(root)
    if not root_path.exists():
        raise ValueError(f"dataset_path does not exist: {root}")

    class_dirs = sorted([p for p in root_path.iterdir() if p.is_dir()])
    if len(class_dirs) == 0:
        raise ValueError(f"No class subdirectories found under dataset_path='{root}'")

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    images = []
    labels = []
    classes = [p.name for p in class_dirs]
    for label, class_dir in enumerate(class_dirs):
        files = sorted([p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts])
        for file_path in files:
            img = PIL.Image.open(file_path).convert(color_mode).resize((image_size, image_size))
            arr = np.asarray(img, dtype=np.float32)
            if image_channels == 1:
                arr = arr[None, :, :]
            else:
                arr = np.transpose(arr, (2, 0, 1))
            tensor = torch.from_numpy(arr / 255.0) * 2.0 - 1.0
            images.append(tensor)
            labels.append(label)

    if len(images) == 0:
        raise ValueError(f"No image files found under dataset_path='{root}'")

    images = torch.stack(images, dim=0).to(dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)
    return images, labels, classes


def _split_indices(total: int, train_size: int, val_size: int, seed: int):
    """Deterministically split one image pool into limited-train / full-train / val subsets."""

    if train_size <= 0 or val_size <= 0:
        raise ValueError(f"image_train_size and image_val_size must be > 0, got {train_size}, {val_size}")
    if train_size + val_size > total:
        raise ValueError(
            f"Requested image_train_size + image_val_size = {train_size + val_size} exceeds dataset size {total}"
        )
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    perm = torch.randperm(total, generator=g)
    val_idx = perm[:val_size]
    train_idx = perm[val_size : val_size + train_size]
    full_train_idx = perm[val_size:]
    return train_idx, full_train_idx, val_idx


def _subsample_indices(total: int, subset_size: int, seed: int) -> torch.Tensor:
    """Pick a deterministic random subset of indices without replacement."""

    if subset_size <= 0:
        raise ValueError(f"subset_size must be > 0, got {subset_size}")
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    subset_size = min(int(subset_size), int(total))
    return torch.randperm(total, generator=g)[:subset_size]


def _sample_pool_to_device(pool: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    """Randomly sample a CPU pool and move the batch to target device."""

    idx = torch.randint(0, pool.shape[0], (batch_size,), device=pool.device)
    return pool[idx].to(device=device)


def _build_image_folder_bundle(cfg, device: torch.device) -> DatasetBundle:
    """Build a small image-folder backend for the toy protocol."""

    train_root_resolved = None
    val_root_resolved = None
    images_cpu, labels_cpu, classes = _load_image_folder_tensors(
        root=cfg.dataset_path,
        image_size=cfg.image_size,
        image_channels=cfg.image_channels,
    )
    from pathlib import Path

    train_root_resolved = str(Path(cfg.dataset_path).resolve())
    num_classes = len(classes)

    if cfg.dataset_val_path:
        val_root_resolved = str(Path(cfg.dataset_val_path).resolve())
        if val_root_resolved == train_root_resolved:
            raise ValueError(
                "dataset_path and dataset_val_path resolve to the same directory. "
                "For a held-out split from one root, leave dataset_val_path empty and use "
                "image_train_size/image_val_size so the backend creates a disjoint split."
            )
        val_images_cpu, val_labels_cpu, val_classes = _load_image_folder_tensors(
            root=cfg.dataset_val_path,
            image_size=cfg.image_size,
            image_channels=cfg.image_channels,
        )
        if val_classes != classes:
            raise ValueError(
                "dataset_path and dataset_val_path must have the same sorted class names, "
                f"got train={classes} vs val={val_classes}"
            )
        train_idx = _subsample_indices(images_cpu.shape[0], cfg.image_train_size, cfg.image_split_seed)
        val_idx = _subsample_indices(val_images_cpu.shape[0], cfg.image_val_size, cfg.image_split_seed + 1001)
        train_pool_limited_cpu = images_cpu[train_idx].clone()
        train_labels_limited_cpu = labels_cpu[train_idx].clone()
        val_pool_cpu = val_images_cpu[val_idx].clone()
        val_labels_cpu = val_labels_cpu[val_idx].clone()
        population_pool_cpu = images_cpu
    else:
        train_idx, full_train_idx, val_idx = _split_indices(
            total=images_cpu.shape[0],
            train_size=cfg.image_train_size,
            val_size=cfg.image_val_size,
            seed=cfg.image_split_seed,
        )
        train_pool_limited_cpu = images_cpu[train_idx].clone()
        train_labels_limited_cpu = labels_cpu[train_idx].clone()
        full_train_pool_cpu = images_cpu[full_train_idx].clone()
        val_pool_cpu = images_cpu[val_idx].clone()
        val_labels_cpu = labels_cpu[val_idx].clone()
        population_pool_cpu = full_train_pool_cpu

    data_shape = tuple(int(v) for v in population_pool_cpu.shape[1:])

    def sample_train(batch_size: int) -> torch.Tensor:
        if cfg.limited_data_enabled:
            return _sample_pool_to_device(train_pool_limited_cpu, batch_size, device)
        return _sample_pool_to_device(population_pool_cpu, batch_size, device)

    def sample_val(batch_size: int) -> torch.Tensor:
        return _sample_pool_to_device(val_pool_cpu, batch_size, device)

    def sample_population(batch_size: int) -> torch.Tensor:
        return _sample_pool_to_device(population_pool_cpu, batch_size, device)

    def sample_terminal(batch_size: int, sigma: Union[float, torch.Tensor]) -> torch.Tensor:
        sigma_value = float(sigma.item()) if torch.is_tensor(sigma) else float(sigma)
        return torch.randn((batch_size, *data_shape), device=device) * sigma_value

    def evaluate_samples(samples_np) -> Dict[str, float]:
        samples_np = samples_np.astype("float32", copy=False)
        flat = samples_np.reshape(samples_np.shape[0], -1)
        per_sample_std = flat.std(axis=1)
        return {
            "global_mean": float(flat.mean()),
            "global_std": float(flat.std()),
            "per_sample_std_mean": float(per_sample_std.mean()),
            "per_sample_std_p10": float(np.quantile(per_sample_std, 0.10)),
            "frac_saturated_abs_gt_095": float((np.abs(samples_np) > 0.95).mean()),
            "value_min": float(samples_np.min()),
            "value_max": float(samples_np.max()),
        }

    def estimate_sigma() -> float:
        source = train_pool_limited_cpu if cfg.limited_data_enabled else population_pool_cpu
        return float(source.reshape(source.shape[0], -1).std(unbiased=False).item())

    train_pool_field = train_pool_limited_cpu if cfg.limited_data_enabled else None
    train_labels_field = train_labels_limited_cpu if cfg.limited_data_enabled else None

    return DatasetBundle(
        name="image_folder",
        data_shape=data_shape,
        centers=None,
        train_pool=train_pool_field,
        train_pool_labels=train_labels_field,
        val_pool=val_pool_cpu,
        sample_train_batch=sample_train,
        sample_val_batch=sample_val,
        sample_population_batch=sample_population,
        sample_terminal_batch=sample_terminal,
        evaluate_sample_metrics=evaluate_samples,
        estimate_sigma_data=estimate_sigma,
        metadata={
            "num_classes": int(num_classes),
            "class_names": classes,
            "val_pool_labels": val_labels_cpu,
            "population_size": int(population_pool_cpu.shape[0]),
            "enable_nearest_reference_distance": False,
            "train_root": train_root_resolved,
            "val_root": val_root_resolved,
            "train_val_disjoint_guarantee": bool(not cfg.dataset_val_path),
        },
    )
