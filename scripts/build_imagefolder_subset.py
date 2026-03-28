from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a deterministic subset of an ImageFolderDataset.")
    parser.add_argument("--src", type=Path, required=True, help="Source ImageFolderDataset directory.")
    parser.add_argument("--dest", type=Path, required=True, help="Destination directory for the subset.")
    parser.add_argument("--percent", type=int, required=True, help="Subset percent in [1, 100].")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for subset selection.")
    parser.add_argument("--force", action="store_true", help="Rebuild even if destination already matches.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.percent < 1 or args.percent > 100:
        raise SystemExit("[ERROR] --percent must be in [1, 100].")
    if not args.src.is_dir():
        raise SystemExit(f"[ERROR] Source must be a directory: {args.src}")

    src_labels_path = args.src / "dataset.json"
    if not src_labels_path.is_file():
        raise SystemExit(f"[ERROR] Missing dataset.json in source dataset: {src_labels_path}")

    payload = json.loads(src_labels_path.read_text(encoding="utf-8"))
    labels = payload.get("labels")
    if not labels:
        raise SystemExit(f"[ERROR] Source dataset has no labels: {src_labels_path}")

    if args.percent >= 100:
        print(f"[INFO] percent=100 -> no subset needed, source dataset is unchanged: {args.src}")
        return

    rng = random.Random(args.seed)
    target_count = max(1, int(len(labels) * args.percent / 100))
    selected = choose_subset(labels=labels, target_count=target_count, rng=rng)

    dest_labels_path = args.dest / "dataset.json"
    existing_count = sum(1 for _ in args.dest.rglob("*.png")) if args.dest.exists() else 0
    if not args.force and dest_labels_path.is_file() and existing_count == len(selected):
        print(f"[INFO] Reusing existing subset: {args.dest} ({existing_count} images)")
        return

    if args.dest.exists():
        shutil.rmtree(args.dest)
    args.dest.mkdir(parents=True, exist_ok=True)

    subset_labels = []
    for rel_path, label in selected:
        src_img = args.src / rel_path
        dst_img = args.dest / rel_path
        if not src_img.is_file():
            raise SystemExit(f"[ERROR] Missing source image referenced by dataset.json: {src_img}")
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_img, dst_img)
        subset_labels.append([rel_path, label])

    dest_labels_path.write_text(json.dumps({"labels": subset_labels}), encoding="utf-8")
    print(
        f"[INFO] Built subset: {args.dest} "
        f"({len(subset_labels)} / {len(labels)} images, percent={args.percent}, seed={args.seed})"
    )


def choose_subset(*, labels: list[list], target_count: int, rng: random.Random) -> list[list]:
    by_label: dict[int, list[list]] = defaultdict(list)
    total_count = len(labels)
    for rel_path, label in labels:
        by_label[int(label)].append([str(rel_path), int(label)])

    # Preserve approximate class balance when labels are available.
    picks: list[list] = []
    leftovers: list[list] = []
    for label, entries in sorted(by_label.items()):
        shuffled = entries[:]
        rng.shuffle(shuffled)
        take = max(1, round(target_count * len(entries) / total_count))
        take = min(take, len(shuffled))
        picks.extend(shuffled[:take])
        leftovers.extend(shuffled[take:])

    if len(picks) > target_count:
        rng.shuffle(picks)
        picks = picks[:target_count]
    elif len(picks) < target_count:
        rng.shuffle(leftovers)
        picks.extend(leftovers[: target_count - len(picks)])

    return picks


if __name__ == "__main__":
    main()
