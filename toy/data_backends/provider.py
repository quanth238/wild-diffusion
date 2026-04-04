from dataclasses import dataclass, field
import gzip
import os
from pathlib import Path
import struct
from typing import Callable, Dict, Optional, Tuple, Union
from urllib.request import urlretrieve

import numpy as np
import torch
import torch.nn.functional as F

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
    if cfg.dataset_kind == "mnist":
        return _build_mnist_bundle(cfg, device)
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
    """Pick a deterministic random subset of indices without replacement.

    This function keeps backward compatibility for global random subsampling.
    New code paths may call it with additional keyword arguments.
    """

    return _subsample_indices_with_policy(
        total=total,
        subset_size=subset_size,
        seed=seed,
        labels=None,
        strategy="global",
    )


def _resolve_subset_size(total: int, fallback_size: int, percent: float) -> int:
    """Resolve subset size from percent if provided, else from fallback size."""

    total = int(total)
    if total <= 0:
        raise ValueError(f"total must be > 0, got {total}")
    if float(percent) > 0:
        resolved = int(round(total * (float(percent) / 100.0)))
        return min(max(resolved, 1), total)
    if int(fallback_size) <= 0:
        raise ValueError(f"fallback_size must be > 0 when percent <= 0, got {fallback_size}")
    return min(int(fallback_size), total)


def _subsample_indices_stratified(labels: torch.Tensor, subset_size: int, seed: int) -> torch.Tensor:
    """Deterministic stratified subset preserving class proportions as closely as possible."""

    if labels.ndim != 1:
        raise ValueError(f"labels must be rank-1, got shape={tuple(labels.shape)}")
    if labels.numel() == 0:
        raise ValueError("labels must be non-empty for stratified sampling.")

    labels = labels.to(dtype=torch.long, device="cpu")
    total = int(labels.shape[0])
    if subset_size <= 0:
        raise ValueError(f"subset_size must be > 0, got {subset_size}")
    subset_size = min(int(subset_size), total)

    max_label = int(labels.max().item())
    counts = torch.bincount(labels, minlength=max_label + 1).to(dtype=torch.long)
    raw = counts.to(dtype=torch.float64) * (float(subset_size) / float(total))
    quotas = torch.floor(raw).to(dtype=torch.long)

    remainder = subset_size - int(quotas.sum().item())
    if remainder > 0:
        frac = raw - quotas.to(dtype=torch.float64)
        order = torch.argsort(frac, descending=True)
        capacity = counts - quotas
        for cls in order.tolist():
            if remainder <= 0:
                break
            if capacity[cls] > 0:
                quotas[cls] += 1
                capacity[cls] -= 1
                remainder -= 1
        if remainder > 0:
            # Fallback: fill leftover quota from classes with remaining capacity.
            order = torch.argsort(counts - quotas, descending=True)
            for cls in order.tolist():
                while remainder > 0 and quotas[cls] < counts[cls]:
                    quotas[cls] += 1
                    remainder -= 1
                if remainder <= 0:
                    break
    if int(quotas.sum().item()) != subset_size:
        raise RuntimeError(
            f"Stratified allocation failed: expected {subset_size}, got {int(quotas.sum().item())}"
        )

    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    perm = torch.randperm(total, generator=g)
    taken = torch.zeros_like(quotas)
    selected = []
    for idx in perm.tolist():
        cls = int(labels[idx].item())
        if taken[cls] < quotas[cls]:
            selected.append(idx)
            taken[cls] += 1
            if len(selected) == subset_size:
                break

    if len(selected) != subset_size:
        raise RuntimeError(
            f"Stratified sampling selected {len(selected)} samples, expected {subset_size}."
        )
    return torch.tensor(selected, dtype=torch.long)


def _subsample_indices_with_policy(
    total: int,
    subset_size: int,
    seed: int,
    labels: Optional[torch.Tensor] = None,
    strategy: str = "global",
) -> torch.Tensor:
    """Pick a deterministic subset with policy: global random or stratified by label."""

    if subset_size <= 0:
        raise ValueError(f"subset_size must be > 0, got {subset_size}")
    total = int(total)
    subset_size = min(int(subset_size), total)
    strategy = str(strategy).lower()
    if strategy == "stratified":
        if labels is None:
            raise ValueError("labels are required when strategy='stratified'.")
        return _subsample_indices_stratified(labels=labels, subset_size=subset_size, seed=seed)
    if strategy != "global":
        raise ValueError(f"Unknown subset strategy '{strategy}'. Expected 'global' or 'stratified'.")
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    return torch.randperm(total, generator=g)[:subset_size]


def _sample_pool_to_device(pool: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    """Randomly sample a CPU pool and move the batch to target device."""

    idx = torch.randint(0, pool.shape[0], (batch_size,), device=pool.device)
    batch = pool[idx]
    non_blocking = bool(device.type == "cuda" and batch.device.type == "cpu" and batch.is_pinned())
    return batch.to(device=device, non_blocking=non_blocking)


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _resolve_image_pool_storage(
    *,
    pool_cpu: torch.Tensor,
    device: torch.device,
    gpu_cache_budget_bytes: int,
) -> torch.Tensor:
    if device.type != "cuda":
        return pool_cpu
    if _tensor_nbytes(pool_cpu) <= int(gpu_cache_budget_bytes):
        return pool_cpu.to(device=device, non_blocking=True)
    return pool_cpu.pin_memory()


def _evaluate_image_samples(samples_np) -> Dict[str, float]:
    """Backend-agnostic image quality summary used by image-like datasets."""

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

    gpu_cache_budget_mb = int(os.environ.get("TOY_IMAGE_FOLDER_GPU_CACHE_MAX_MB", "512"))
    gpu_cache_budget_bytes = int(max(gpu_cache_budget_mb, 0)) * 1024 * 1024
    train_pool_limited_sample = _resolve_image_pool_storage(
        pool_cpu=train_pool_limited_cpu,
        device=device,
        gpu_cache_budget_bytes=gpu_cache_budget_bytes,
    )
    val_pool_sample = _resolve_image_pool_storage(
        pool_cpu=val_pool_cpu,
        device=device,
        gpu_cache_budget_bytes=gpu_cache_budget_bytes,
    )
    population_pool_sample = _resolve_image_pool_storage(
        pool_cpu=population_pool_cpu,
        device=device,
        gpu_cache_budget_bytes=gpu_cache_budget_bytes,
    )
    data_shape = tuple(int(v) for v in population_pool_cpu.shape[1:])

    def sample_train(batch_size: int) -> torch.Tensor:
        if cfg.limited_data_enabled:
            return _sample_pool_to_device(train_pool_limited_sample, batch_size, device)
        return _sample_pool_to_device(population_pool_sample, batch_size, device)

    def sample_val(batch_size: int) -> torch.Tensor:
        return _sample_pool_to_device(val_pool_sample, batch_size, device)

    def sample_population(batch_size: int) -> torch.Tensor:
        return _sample_pool_to_device(population_pool_sample, batch_size, device)

    def sample_terminal(batch_size: int, sigma: Union[float, torch.Tensor]) -> torch.Tensor:
        sigma_value = float(sigma.item()) if torch.is_tensor(sigma) else float(sigma)
        return torch.randn((batch_size, *data_shape), device=device) * sigma_value

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
        evaluate_sample_metrics=_evaluate_image_samples,
        estimate_sigma_data=estimate_sigma,
        metadata={
            "num_classes": int(num_classes),
            "class_names": classes,
            "train_subset_size_resolved": int(train_pool_limited_cpu.shape[0]),
            "val_subset_size_resolved": int(val_pool_cpu.shape[0]),
            "train_subset_fraction_resolved": float(train_pool_limited_cpu.shape[0]) / float(images_cpu.shape[0]),
            "val_pool_labels": val_labels_cpu,
            "population_size": int(population_pool_cpu.shape[0]),
            "enable_nearest_reference_distance": True,
            "sample_storage": {
                "device": str(device),
                "gpu_cache_budget_mb": int(gpu_cache_budget_mb),
                "train_pool_device": str(train_pool_limited_sample.device),
                "val_pool_device": str(val_pool_sample.device),
                "population_pool_device": str(population_pool_sample.device),
            },
            "train_root": train_root_resolved,
            "val_root": val_root_resolved,
            "train_val_disjoint_guarantee": bool(not cfg.dataset_val_path),
        },
    )


def _download_if_missing(url: str, path: Path) -> None:
    """Download URL to `path` if absent (atomic rename on success)."""

    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    urlretrieve(url, tmp_path)
    tmp_path.replace(path)


def _read_idx_images(path: Path) -> torch.Tensor:
    """Read IDX image file (gzip) into uint8 tensor [N,1,H,W]."""

    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid image magic for {path}: {magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    data = data.reshape(n, 1, rows, cols)
    return torch.from_numpy(data.copy())


def _read_idx_labels(path: Path) -> torch.Tensor:
    """Read IDX label file (gzip) into uint8 tensor [N]."""

    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid label magic for {path}: {magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    if data.shape[0] != n:
        raise ValueError(f"Label size mismatch for {path}: header={n}, actual={data.shape[0]}")
    return torch.from_numpy(data.copy())


def _load_mnist_tensors(cache_root: Path):
    """Download/load MNIST train/test splits as float32 in [-1,1]."""

    base_urls = [
        "https://storage.googleapis.com/cvdf-datasets/mnist/",
        "http://yann.lecun.com/exdb/mnist/",
    ]
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }

    for name, filename in files.items():
        dst = cache_root / filename
        if dst.exists():
            continue
        last_err = None
        for base in base_urls:
            try:
                _download_if_missing(base + filename, dst)
                last_err = None
                break
            except Exception as err:  # pragma: no cover - network fallback path
                last_err = err
                if dst.exists():
                    dst.unlink(missing_ok=True)
        if last_err is not None:
            raise RuntimeError(f"Failed to download MNIST file '{filename}': {last_err}")

    train_images_u8 = _read_idx_images(cache_root / files["train_images"])
    train_labels = _read_idx_labels(cache_root / files["train_labels"]).to(dtype=torch.long)
    test_images_u8 = _read_idx_images(cache_root / files["test_images"])
    test_labels = _read_idx_labels(cache_root / files["test_labels"]).to(dtype=torch.long)

    train_images = train_images_u8.to(dtype=torch.float32) / 255.0 * 2.0 - 1.0
    test_images = test_images_u8.to(dtype=torch.float32) / 255.0 * 2.0 - 1.0
    return train_images, train_labels, test_images, test_labels


def _build_mnist_bundle(cfg, device: torch.device) -> DatasetBundle:
    """Build MNIST backend (download IDX files if needed, no torchvision dependency)."""

    if cfg.image_channels != 1:
        raise ValueError(
            f"dataset_kind='mnist' requires --image-channels=1, got image_channels={cfg.image_channels}"
        )
    cache_root = Path(cfg.dataset_path).expanduser() if cfg.dataset_path else Path.home() / ".cache" / "wild_diffusion" / "mnist"
    train_images, train_labels, test_images, test_labels = _load_mnist_tensors(cache_root)

    if cfg.image_size > 0 and cfg.image_size != train_images.shape[-1]:
        size = (int(cfg.image_size), int(cfg.image_size))
        train_images = F.interpolate(train_images, size=size, mode="bilinear", align_corners=False)
        test_images = F.interpolate(test_images, size=size, mode="bilinear", align_corners=False)

    if cfg.mnist_use_percent_split:
        train_subset_size = _resolve_subset_size(
            total=int(train_images.shape[0]),
            fallback_size=cfg.image_train_size,
            percent=float(cfg.mnist_train_percent),
        )
        val_subset_size = _resolve_subset_size(
            total=int(test_images.shape[0]),
            fallback_size=cfg.image_val_size,
            percent=float(cfg.mnist_val_percent),
        )
    else:
        train_subset_size = _resolve_subset_size(
            total=int(train_images.shape[0]),
            fallback_size=cfg.image_train_size,
            percent=-1.0,
        )
        val_subset_size = _resolve_subset_size(
            total=int(test_images.shape[0]),
            fallback_size=cfg.image_val_size,
            percent=-1.0,
        )

    train_idx = _subsample_indices_with_policy(
        total=int(train_images.shape[0]),
        subset_size=int(train_subset_size),
        seed=int(cfg.image_split_seed),
        labels=train_labels,
        strategy="stratified",
    )
    val_idx = _subsample_indices_with_policy(
        total=int(test_images.shape[0]),
        subset_size=int(val_subset_size),
        seed=int(cfg.image_split_seed + 1001),
        labels=test_labels,
        strategy="stratified",
    )

    train_pool_limited_cpu = train_images[train_idx].clone()
    train_labels_limited_cpu = train_labels[train_idx].clone()
    val_pool_cpu = test_images[val_idx].clone()
    val_labels_cpu = test_labels[val_idx].clone()
    population_pool_cpu = train_images
    population_labels_cpu = train_labels
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

    def estimate_sigma() -> float:
        source = train_pool_limited_cpu if cfg.limited_data_enabled else population_pool_cpu
        return float(source.reshape(source.shape[0], -1).std(unbiased=False).item())

    train_pool_field = train_pool_limited_cpu if cfg.limited_data_enabled else None
    train_labels_field = train_labels_limited_cpu if cfg.limited_data_enabled else None

    return DatasetBundle(
        name="mnist",
        data_shape=data_shape,
        centers=None,
        train_pool=train_pool_field,
        train_pool_labels=train_labels_field,
        val_pool=val_pool_cpu,
        sample_train_batch=sample_train,
        sample_val_batch=sample_val,
        sample_population_batch=sample_population,
        sample_terminal_batch=sample_terminal,
        evaluate_sample_metrics=_evaluate_image_samples,
        estimate_sigma_data=estimate_sigma,
        metadata={
            "num_classes": 10,
            "class_names": [str(i) for i in range(10)],
            "train_selection_policy": {
                "limited_data_enabled": bool(cfg.limited_data_enabled),
                "mnist_use_percent_split": bool(cfg.mnist_use_percent_split),
                "mnist_train_percent": float(cfg.mnist_train_percent),
                "mnist_val_percent": float(cfg.mnist_val_percent),
                "mnist_subset_sampling": "stratified",
                "image_split_seed": int(cfg.image_split_seed),
            },
            "train_subset_size_resolved": int(train_pool_limited_cpu.shape[0]),
            "val_subset_size_resolved": int(val_pool_cpu.shape[0]),
            "train_subset_fraction_resolved": float(train_pool_limited_cpu.shape[0]) / float(train_images.shape[0]),
            "val_pool_labels": val_labels_cpu,
            "population_size": int(population_pool_cpu.shape[0]),
            "enable_nearest_reference_distance": False,
            "train_root": str(cache_root),
            "val_root": str(cache_root),
            "train_val_disjoint_guarantee": True,
            "population_labels": population_labels_cpu,
        },
    )
