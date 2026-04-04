#!/usr/bin/env python3
"""Download a small public Simpsons dataset and prepare toy image_folder splits.

The current default source is the public Hugging Face dataset:
  macadeliccc/simpsons-images

It contains five family-member classes with a small number of labeled images.
This script:
1. lists source image files from the Hub API,
2. creates deterministic train/val splits per class,
3. downloads the images into image_folder layout,
4. optionally builds an FID reference `.npz` from the val split.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from PIL import Image


DEFAULT_REPO_ID = "macadeliccc/simpsons-images"
DEFAULT_SOURCE_PREFIX = "data/train"
DEFAULT_VAL_FRACTION = 0.30


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _hf_tree(repo_id: str) -> List[dict]:
    url = f"https://huggingface.co/api/datasets/{repo_id}/tree/main?recursive=true"
    req = urllib.request.Request(url, headers={"User-Agent": "wild-diffusion-simpsons-setup/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def _hf_resolve_url(repo_id: str, path: str) -> str:
    quoted = urllib.parse.quote(path, safe="/")
    return f"https://huggingface.co/datasets/{repo_id}/resolve/main/{quoted}"


def _download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    req = urllib.request.Request(url, headers={"User-Agent": "wild-diffusion-simpsons-setup/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as fh:
        shutil.copyfileobj(resp, fh)
    tmp.replace(dest)


def _collect_source_images(tree: List[dict], source_prefix: str) -> Dict[str, List[str]]:
    classes: Dict[str, List[str]] = defaultdict(list)
    prefix = source_prefix.rstrip("/") + "/"
    for row in tree:
        path = str(row.get("path", ""))
        if row.get("type") != "file":
            continue
        if not path.startswith(prefix):
            continue
        if not path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
            continue
        parts = path.split("/")
        if len(parts) < 3:
            continue
        cls = parts[2]
        classes[cls].append(path)
    if not classes:
        raise RuntimeError(f"No image files found under source prefix '{source_prefix}'.")
    for cls in classes:
        classes[cls].sort()
    return dict(classes)


def _split_class_paths(paths: List[str], *, val_fraction: float, rng: random.Random) -> tuple[List[str], List[str]]:
    if len(paths) < 2:
        raise RuntimeError("Need at least 2 images per class to create train/val splits.")
    shuffled = list(paths)
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_fraction)))
    n_val = min(n_val, len(shuffled) - 1)
    val_paths = sorted(shuffled[:n_val])
    train_paths = sorted(shuffled[n_val:])
    return train_paths, val_paths


def _build_fid_ref(repo_root: Path, data_dir: Path, dest_path: Path) -> None:
    cmd = [
        str(repo_root / ".venv" / "bin" / "python"),
        str(repo_root / "fid.py"),
        "ref",
        "--data",
        str(data_dir),
        "--dest",
        str(dest_path),
    ]
    subprocess.run(cmd, check=True, cwd=str(repo_root))


def _write_resized_rgb_copy(src: Path, dest: Path, image_size: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        img = img.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
        img.save(dest)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prepare Simpsons image_folder dataset for toy image experiments.")
    p.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID)
    p.add_argument("--source-prefix", type=str, default=DEFAULT_SOURCE_PREFIX)
    p.add_argument("--out-root", type=Path, default=Path("toy_data/simpsons_images"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--force-redownload", action="store_true")
    p.add_argument("--skip-fid-ref", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    repo_root = _repo_root()
    out_root = (repo_root / args.out_root).resolve()
    imagefolder_root = out_root / "imagefolder"
    train_root = imagefolder_root / "train"
    val_root = imagefolder_root / "val"
    fid_ref_root = out_root / "fid_refs"
    fid_ref_images_root = out_root / "fid_ref_images" / "val"
    fid_ref_root.mkdir(parents=True, exist_ok=True)
    fid_ref_path = fid_ref_root / "simpsons_val_seed0.npz"
    summary_path = out_root / "summary.json"

    print(f"[setup] repo_id={args.repo_id}")
    print(f"[setup] out_root={out_root}")
    tree = _hf_tree(args.repo_id)
    classes = _collect_source_images(tree, args.source_prefix)
    rng = random.Random(int(args.seed))

    split_summary = {}
    for cls, paths in sorted(classes.items()):
        train_paths, val_paths = _split_class_paths(paths, val_fraction=float(args.val_fraction), rng=rng)
        split_summary[cls] = {
            "total": len(paths),
            "train": len(train_paths),
            "val": len(val_paths),
        }
        for split_name, split_paths, split_root in [
            ("train", train_paths, train_root),
            ("val", val_paths, val_root),
        ]:
            class_root = split_root / cls
            class_root.mkdir(parents=True, exist_ok=True)
            for src_path in split_paths:
                dest = class_root / Path(src_path).name
                if dest.is_file() and not args.force_redownload:
                    pass
                else:
                    url = _hf_resolve_url(args.repo_id, src_path)
                    print(f"[download] {split_name} {cls} {Path(src_path).name}", flush=True)
                    _download_file(url, dest)
                if split_name == "val":
                    fid_ref_img_dest = fid_ref_images_root / cls / (dest.stem + ".png")
                    if args.force_redownload or not fid_ref_img_dest.is_file():
                        _write_resized_rgb_copy(dest, fid_ref_img_dest, int(args.image_size))

    if not args.skip_fid_ref:
        print(f"[fid-ref] building from {fid_ref_images_root} -> {fid_ref_path}", flush=True)
        _build_fid_ref(repo_root, fid_ref_images_root, fid_ref_path)

    payload = {
        "source": {
            "repo_id": args.repo_id,
            "source_prefix": args.source_prefix,
        },
        "seed": int(args.seed),
        "val_fraction": float(args.val_fraction),
        "imagefolder_root": str(imagefolder_root),
        "train_root": str(train_root),
        "val_root": str(val_root),
        "fid_ref_images_root": str(fid_ref_images_root),
        "fid_ref_path": None if args.skip_fid_ref else str(fid_ref_path),
        "classes": split_summary,
    }
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[done] summary={summary_path}")
    print(f"[done] train_root={train_root}")
    print(f"[done] val_root={val_root}")
    if not args.skip_fid_ref:
        print(f"[done] fid_ref={fid_ref_path}")


if __name__ == "__main__":
    main()
