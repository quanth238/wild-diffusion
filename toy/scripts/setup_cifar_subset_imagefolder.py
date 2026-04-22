#!/usr/bin/env python3
"""Prepare a balanced CIFAR-10 class subset in image_folder format for toy runs."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List

from PIL import Image
from torchvision.datasets import CIFAR10


DEFAULT_CLASSES = ("airplane", "automobile")
DEFAULT_TRAIN_PER_CLASS = 1000
DEFAULT_TEST_PER_CLASS = 0  # 0 => keep all held-out examples from the official test split.


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _flatten_class_tokens(tokens: Iterable[str]) -> List[str]:
    out: List[str] = []
    for token in tokens:
        for part in str(token).split(","):
            value = part.strip().lower()
            if value:
                out.append(value)
    if not out:
        raise SystemExit("[error] at least one class must be provided via --classes")
    return out


def _resolve_selected_classes(requested: List[str], available: List[str]) -> list[tuple[int, str]]:
    name_to_idx = {name.lower(): idx for idx, name in enumerate(available)}
    resolved: list[tuple[int, str]] = []
    seen = set()
    for token in requested:
        if token.isdigit():
            idx = int(token)
            if idx < 0 or idx >= len(available):
                raise SystemExit(
                    f"[error] class index {idx} is out of range for CIFAR-10 (expected 0..{len(available) - 1})"
                )
            name = available[idx]
        else:
            if token not in name_to_idx:
                raise SystemExit(
                    f"[error] unknown CIFAR-10 class '{token}'. "
                    f"Available classes: {', '.join(available)}"
                )
            idx = name_to_idx[token]
            name = available[idx]
        if idx in seen:
            continue
        seen.add(idx)
        resolved.append((idx, name))
    if not resolved:
        raise SystemExit("[error] resolved class list is empty")
    return resolved


def _select_indices_by_class(
    labels: List[int],
    selected: list[tuple[int, str]],
    *,
    per_class: int,
    seed: int,
) -> dict[int, list[int]]:
    indices_by_class = {class_idx: [] for class_idx, _name in selected}
    for idx, label in enumerate(labels):
        if label in indices_by_class:
            indices_by_class[int(label)].append(idx)

    rng = random.Random(int(seed))
    selected_indices: dict[int, list[int]] = {}
    for class_idx, _name in selected:
        class_indices = list(indices_by_class[int(class_idx)])
        if not class_indices:
            raise SystemExit(f"[error] no examples found for class index {class_idx}")
        rng.shuffle(class_indices)
        if int(per_class) > 0:
            if len(class_indices) < int(per_class):
                raise SystemExit(
                    f"[error] requested {int(per_class)} examples for class index {class_idx}, "
                    f"but only found {len(class_indices)}"
                )
            class_indices = class_indices[: int(per_class)]
        selected_indices[int(class_idx)] = sorted(class_indices)
    return selected_indices


def _write_split(
    *,
    dataset: CIFAR10,
    selected: list[tuple[int, str]],
    selected_indices: dict[int, list[int]],
    split_root: Path,
    image_size: int,
) -> dict:
    if split_root.exists():
        shutil.rmtree(split_root)
    split_root.mkdir(parents=True, exist_ok=True)

    remap = {orig_idx: new_idx for new_idx, (orig_idx, _name) in enumerate(selected)}
    labels_payload = []
    class_counts = {}
    total_count = 0

    for orig_idx, class_name in selected:
        new_idx = remap[int(orig_idx)]
        class_root = split_root / class_name
        class_root.mkdir(parents=True, exist_ok=True)
        class_indices = list(selected_indices[int(orig_idx)])
        class_counts[class_name] = len(class_indices)
        total_count += len(class_indices)
        for write_idx, src_idx in enumerate(class_indices):
            image, label = dataset[int(src_idx)]
            if int(label) != int(orig_idx):
                raise RuntimeError(
                    f"Label mismatch while materializing split: expected {orig_idx}, got {int(label)} at idx={src_idx}"
                )
            image = image.convert("RGB")
            if image.size != (int(image_size), int(image_size)):
                image = image.resize((int(image_size), int(image_size)), Image.Resampling.BILINEAR)
            rel_path = Path(class_name) / f"{write_idx:05d}.png"
            image.save(split_root / rel_path)
            labels_payload.append([str(rel_path).replace("\\", "/"), int(new_idx)])

    labels_path = split_root / "dataset.json"
    labels_path.write_text(json.dumps({"labels": labels_payload}) + "\n", encoding="utf-8")

    return {
        "root": str(split_root.resolve()),
        "total": int(total_count),
        "class_counts": {k: int(v) for k, v in class_counts.items()},
        "dataset_json": str(labels_path.resolve()),
    }


def _build_fid_ref(repo_root: Path, data_dir: Path, dest_path: Path) -> None:
    cmd = [
        sys.executable,
        str(repo_root / "fid.py"),
        "ref",
        "--data",
        str(data_dir),
        "--dest",
        str(dest_path),
    ]
    subprocess.run(cmd, check=True, cwd=str(repo_root))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a balanced CIFAR-10 class subset in image_folder format for toy image experiments."
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=list(DEFAULT_CLASSES),
        help=(
            "Class names or indices to keep. Accepts space-separated values and/or comma-separated tokens. "
            f"Default: {' '.join(DEFAULT_CLASSES)}"
        ),
    )
    parser.add_argument(
        "--train-per-class",
        type=int,
        default=DEFAULT_TRAIN_PER_CLASS,
        help="Number of official CIFAR-10 train images to keep per selected class.",
    )
    parser.add_argument(
        "--test-per-class",
        type=int,
        default=DEFAULT_TEST_PER_CLASS,
        help="Number of official CIFAR-10 test images to keep per selected class. Use 0 to keep all held-out images.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed used for per-class subset selection.")
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--data-root", type=Path, default=Path("datasets"))
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--download", action="store_true", help="Allow torchvision to download CIFAR-10 if missing.")
    parser.add_argument("--skip-fid-ref", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = _repo_root()
    data_root = (repo_root / args.data_root).resolve()
    class_tokens = _flatten_class_tokens(args.classes)

    preview = CIFAR10(root=str(data_root), train=True, download=bool(args.download))
    selected = _resolve_selected_classes(class_tokens, preview.classes)
    selected_names = [name for _idx, name in selected]
    selected_original_indices = {name: int(idx) for idx, name in selected}

    out_root = args.out_root
    if out_root is None:
        tag = "_".join(selected_names)
        out_root = Path("toy_data") / f"cifar10_subset_{tag}"
    out_root = (repo_root / out_root).resolve()
    imagefolder_root = out_root / "imagefolder"
    train_root = imagefolder_root / "train"
    test_root = imagefolder_root / "test"
    fid_ref_root = out_root / "fid_refs"
    fid_ref_root.mkdir(parents=True, exist_ok=True)
    fid_ref_name = "cifar10_" + "_".join(selected_names) + f"_test_{int(args.image_size)}x{int(args.image_size)}.npz"
    fid_ref_path = fid_ref_root / fid_ref_name
    summary_path = out_root / "summary.json"

    print(f"[setup] data_root={data_root}", flush=True)
    print(f"[setup] classes={selected_names}", flush=True)
    print(f"[setup] out_root={out_root}", flush=True)

    train_ds = CIFAR10(root=str(data_root), train=True, download=bool(args.download))
    test_ds = CIFAR10(root=str(data_root), train=False, download=bool(args.download))

    train_indices = _select_indices_by_class(
        labels=train_ds.targets,
        selected=selected,
        per_class=int(args.train_per_class),
        seed=int(args.seed),
    )
    test_indices = _select_indices_by_class(
        labels=test_ds.targets,
        selected=selected,
        per_class=int(args.test_per_class),
        seed=int(args.seed + 1001),
    )

    train_summary = _write_split(
        dataset=train_ds,
        selected=selected,
        selected_indices=train_indices,
        split_root=train_root,
        image_size=int(args.image_size),
    )
    test_summary = _write_split(
        dataset=test_ds,
        selected=selected,
        selected_indices=test_indices,
        split_root=test_root,
        image_size=int(args.image_size),
    )

    if not args.skip_fid_ref:
        print(f"[fid-ref] building from {test_root} -> {fid_ref_path}", flush=True)
        _build_fid_ref(repo_root=repo_root, data_dir=test_root, dest_path=fid_ref_path)

    payload = {
        "source": {
            "dataset": "torchvision.datasets.CIFAR10",
            "data_root": str(data_root),
            "download_enabled": bool(args.download),
            "seed": int(args.seed),
        },
        "subset": {
            "class_names": selected_names,
            "class_original_indices": selected_original_indices,
            "num_classes": int(len(selected_names)),
            "train_per_class_requested": int(args.train_per_class),
            "test_per_class_requested": int(args.test_per_class),
            "test_per_class_mode": "full_official_test_split" if int(args.test_per_class) <= 0 else "fixed_count",
        },
        "dataset": {
            "image_size": int(args.image_size),
            "image_channels": 3,
            "train_total": int(train_summary["total"]),
            "test_total": int(test_summary["total"]),
            "train_class_counts": train_summary["class_counts"],
            "test_class_counts": test_summary["class_counts"],
        },
        "imagefolder_root": str(imagefolder_root.resolve()),
        "train_root": train_summary["root"],
        "test_root": test_summary["root"],
        "fid_ref_path": None if args.skip_fid_ref else str(fid_ref_path.resolve()),
    }
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"[done] summary={summary_path}", flush=True)
    print(f"[done] train_root={train_summary['root']} ({train_summary['total']} images)", flush=True)
    print(f"[done] test_root={test_summary['root']} ({test_summary['total']} images)", flush=True)
    if not args.skip_fid_ref:
        print(f"[done] fid_ref={fid_ref_path}", flush=True)


if __name__ == "__main__":
    main()
