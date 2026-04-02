#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from PIL import Image

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from toy.data_backends.provider import _load_mnist_tensors, _subsample_indices_with_policy
else:
    from .data_backends.provider import _load_mnist_tensors, _subsample_indices_with_policy


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _format_token(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def _resolve_subset_size(total: int, subset_percent: float, subset_size: Optional[int]) -> int:
    total = int(total)
    if subset_size is not None and int(subset_size) > 0:
        return min(max(int(subset_size), 1), total)
    if float(subset_percent) <= 0.0:
        return total
    resolved = int(round(total * (float(subset_percent) / 100.0)))
    return min(max(resolved, 1), total)


def default_mnist_fid_policy_name(
    *,
    split: str,
    image_size: int,
    subset_percent: float,
    subset_sampling: str,
    subset_seed: int,
    max_images: int,
    subset_size: Optional[int] = None,
) -> str:
    scope_token = (
        f"size{int(subset_size)}"
        if subset_size is not None and int(subset_size) > 0
        else f"pct{_format_token(subset_percent)}"
    )
    return (
        f"mnist_{str(split).lower()}_img{int(image_size)}_{scope_token}_"
        f"{str(subset_sampling).lower()}_seed{int(subset_seed)}_n{int(max_images)}"
    )


def _select_indices(
    *,
    total: int,
    labels: torch.Tensor,
    subset_size: int,
    subset_sampling: str,
    subset_seed: int,
) -> torch.Tensor:
    subset_sampling = str(subset_sampling).lower()
    if subset_sampling == "first":
        return torch.arange(int(subset_size), dtype=torch.long)
    if subset_sampling not in ("global", "stratified"):
        raise ValueError(
            f"Unsupported subset_sampling='{subset_sampling}'. Expected one of: first, global, stratified."
        )
    return _subsample_indices_with_policy(
        total=int(total),
        subset_size=int(subset_size),
        seed=int(subset_seed),
        labels=labels,
        strategy=subset_sampling,
    )


def build_mnist_fid_reference(
    *,
    split: str = "test",
    image_size: int = 32,
    subset_percent: float = 100.0,
    subset_size: Optional[int] = None,
    subset_sampling: str = "stratified",
    subset_seed: int = 0,
    max_images: int = 5000,
    dest: Optional[Path] = None,
    images_dir: Optional[Path] = None,
    policy_name: Optional[str] = None,
    dataset_root: str = "",
    force: bool = False,
) -> Dict[str, Any]:
    repo_root = _repo_root()
    split = str(split).lower()
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split}")

    policy_name = policy_name or default_mnist_fid_policy_name(
        split=split,
        image_size=int(image_size),
        subset_percent=float(subset_percent),
        subset_sampling=str(subset_sampling),
        subset_seed=int(subset_seed),
        max_images=int(max_images),
        subset_size=subset_size,
    )
    ref_dir = repo_root / "toy_outputs" / "fid_refs"
    dest = Path(dest) if dest is not None else ref_dir / f"{policy_name}.npz"
    images_dir = Path(images_dir) if images_dir is not None else dest.parent / f"{dest.stem}_images"
    dest = dest.resolve()
    images_dir = images_dir.resolve()
    meta_path = Path(f"{dest}.json")

    if dest.is_file() and meta_path.is_file() and not force:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if isinstance(metadata, dict):
            metadata["reused_existing"] = True
            print(f"[ref] using existing MNIST FID reference: {dest}", flush=True)
            return metadata

    cache_root = Path(dataset_root).expanduser() if dataset_root else Path.home() / ".cache" / "wild_diffusion" / "mnist"
    train_images, train_labels, test_images, test_labels = _load_mnist_tensors(cache_root)
    split_images = train_images if split == "train" else test_images
    split_labels = train_labels if split == "train" else test_labels

    if int(image_size) > 0 and int(image_size) != int(split_images.shape[-1]):
        split_images = F.interpolate(
            split_images,
            size=(int(image_size), int(image_size)),
            mode="bilinear",
            align_corners=False,
        )

    total_available = int(split_images.shape[0])
    subset_size_resolved = _resolve_subset_size(
        total=total_available,
        subset_percent=float(subset_percent),
        subset_size=subset_size,
    )
    subset_indices = _select_indices(
        total=total_available,
        labels=split_labels,
        subset_size=subset_size_resolved,
        subset_sampling=str(subset_sampling),
        subset_seed=int(subset_seed),
    )
    subset_images = split_images[subset_indices]

    num_to_export = min(int(max_images), int(subset_images.shape[0]))
    if num_to_export < 2:
        raise RuntimeError(f"Need at least 2 images for FID reference, got {num_to_export}.")

    if images_dir.is_dir():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    dest.parent.mkdir(parents=True, exist_ok=True)

    print(
        "[ref] building MNIST FID reference:"
        f" policy={policy_name}"
        f" split={split}"
        f" image_size={int(image_size)}"
        f" subset_size={subset_size_resolved}"
        f" exported={num_to_export}"
        f" dest={dest}",
        flush=True,
    )

    for i in range(num_to_export):
        arr = subset_images[i, 0].detach().cpu().numpy()
        arr = ((arr + 1.0) * 0.5 * 255.0).clip(0.0, 255.0).astype("uint8")
        Image.fromarray(arr, mode="L").save(images_dir / f"{i:05d}.png")

    ref_env = os.environ.copy()
    ref_env.setdefault("MASTER_ADDR", "127.0.0.1")
    ref_env.setdefault("MASTER_PORT", "29501")
    ref_env.setdefault("RANK", "0")
    ref_env.setdefault("LOCAL_RANK", "0")
    ref_env.setdefault("WORLD_SIZE", "1")

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "fid.py"),
            "ref",
            "--data",
            str(images_dir),
            "--dest",
            str(dest),
        ],
        check=True,
        cwd=str(repo_root),
        env=ref_env,
    )

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "policy_name": policy_name,
        "split": split,
        "image_size": int(image_size),
        "subset_percent_requested": float(subset_percent),
        "subset_size_requested": None if subset_size is None else int(subset_size),
        "subset_size_resolved": int(subset_size_resolved),
        "subset_sampling": str(subset_sampling).lower(),
        "subset_seed": int(subset_seed),
        "max_images_requested": int(max_images),
        "num_images_exported": int(num_to_export),
        "source_total_images": int(total_available),
        "cache_root": str(cache_root),
        "dest": str(dest),
        "images_dir": str(images_dir),
        "metadata_path": str(meta_path),
        "reused_existing": False,
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[ref] saved MNIST FID reference to {dest}", flush=True)
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an explicit MNIST FID reference NPZ with recorded policy metadata.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--subset-percent", type=float, default=100.0)
    parser.add_argument("--subset-size", type=int, default=None)
    parser.add_argument("--subset-sampling", type=str, default="stratified", choices=["first", "global", "stratified"])
    parser.add_argument("--subset-seed", type=int, default=0)
    parser.add_argument("--max-images", type=int, default=5000)
    parser.add_argument("--dataset-root", type=str, default="")
    parser.add_argument("--dest", type=Path, default=None)
    parser.add_argument("--images-dir", type=Path, default=None)
    parser.add_argument("--policy-name", type=str, default="")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> Dict[str, Any]:
    args = build_parser().parse_args(argv)
    metadata = build_mnist_fid_reference(
        split=str(args.split),
        image_size=int(args.image_size),
        subset_percent=float(args.subset_percent),
        subset_size=args.subset_size,
        subset_sampling=str(args.subset_sampling),
        subset_seed=int(args.subset_seed),
        max_images=int(args.max_images),
        dest=args.dest,
        images_dir=args.images_dir,
        policy_name=str(args.policy_name).strip() or None,
        dataset_root=str(args.dataset_root),
        force=bool(args.force),
    )
    print(json.dumps(metadata, indent=2), flush=True)
    return metadata


if __name__ == "__main__":
    main()
