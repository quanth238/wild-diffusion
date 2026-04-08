#!/usr/bin/env python3
"""Build a deterministic CIFAR-10 subset from an ImageFolder-style dataset."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import random
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="Source CIFAR dataset directory")
    parser.add_argument("--dst", required=True, help="Destination subset directory")
    parser.add_argument("--percent", type=int, required=True, help="Subset percent in [1, 100]")
    parser.add_argument("--seed", type=int, default=0, help="Shuffle seed")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild destination even if it already looks complete",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.percent < 1 or args.percent > 100:
        raise SystemExit("--percent must be in [1, 100]")

    src_dir = Path(args.src).resolve()
    dst_dir = Path(args.dst).resolve()
    src_labels_path = src_dir / "dataset.json"
    if not src_labels_path.exists():
        raise SystemExit(f"Missing source dataset.json: {src_labels_path}")

    with src_labels_path.open("r", encoding="utf-8") as f:
        src_labels = json.load(f).get("labels", [])
    if not src_labels:
        raise SystemExit(f"Source dataset has no labels: {src_labels_path}")

    if args.percent >= 100:
        print(f"[INFO] percent=100 -> use full dataset directly: {src_dir}")
        return

    target_count = max(1, int(len(src_labels) * args.percent / 100))
    dst_labels_path = dst_dir / "dataset.json"
    manifest_path = dst_dir / "subset_manifest.json"
    dst_png_count = sum(1 for _ in dst_dir.rglob("*.png")) if dst_dir.exists() else 0

    if not args.force and dst_labels_path.exists() and dst_png_count == target_count:
        print(f"[INFO] Reusing subset: {dst_dir} ({dst_png_count} images)")
        with dst_labels_path.open("r", encoding="utf-8") as f:
            subset_labels = json.load(f).get("labels", [])
    else:
        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        dst_dir.mkdir(parents=True, exist_ok=True)

        rng = random.Random(args.seed)
        indices = list(range(len(src_labels)))
        rng.shuffle(indices)
        keep_set = set(indices[:target_count])

        subset_labels = []
        for idx, (rel_path, label) in enumerate(src_labels):
            if idx not in keep_set:
                continue
            src_img = src_dir / rel_path
            dst_img = dst_dir / rel_path
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_img, dst_img)
            subset_labels.append([rel_path, int(label)])

    with dst_labels_path.open("w", encoding="utf-8") as f:
        json.dump({"labels": subset_labels}, f)

    class_counts = Counter(int(label) for _rel_path, label in subset_labels)
    manifest = {
        "source_dir": str(src_dir),
        "dest_dir": str(dst_dir),
        "percent": int(args.percent),
        "seed": int(args.seed),
        "source_image_count": int(len(src_labels)),
        "subset_image_count": int(len(subset_labels)),
        "class_counts": {str(k): int(v) for k, v in sorted(class_counts.items())},
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    print(
        f"[INFO] Built CIFAR subset: {dst_dir} "
        f"({len(subset_labels)} / {len(src_labels)} images, pct={args.percent}, seed={args.seed})"
    )


if __name__ == "__main__":
    main()
