#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a class-balanced MNIST comparison sheet.")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Path to MNIST ImageFolder directory.")
    parser.add_argument("--baseline-samples", type=Path, required=True, help="Directory containing baseline PNG samples.")
    parser.add_argument("--wdro-samples", type=Path, required=True, help="Directory containing WDRO PNG samples.")
    parser.add_argument("--baseline-eval", type=Path, default=None, help="Optional baseline evaluation_result.json.")
    parser.add_argument("--wdro-eval", type=Path, default=None, help="Optional WDRO evaluation_result.json.")
    parser.add_argument("--output", type=Path, required=True, help="Output PNG path.")
    parser.add_argument("--per-class", type=int, default=8, help="Number of real MNIST examples per class.")
    parser.add_argument("--generated-count", type=int, default=80, help="Number of generated examples per method.")
    parser.add_argument("--cols", type=int, default=10, help="Number of columns in each panel.")
    parser.add_argument("--scale", type=int, default=6, help="Nearest-neighbor upsample factor.")
    parser.add_argument("--pad", type=int, default=2, help="Padding between images.")
    return parser.parse_args()


def load_eval_fid(path: Path | None) -> float | None:
    if path is None or not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    value = payload.get("fid")
    return float(value) if value is not None else None


def make_grid(paths: list[Path], cols: int, scale: int, pad: int) -> Image.Image:
    images = [Image.open(path).convert("L") for path in paths]
    if not images:
        raise ValueError("No images provided for grid.")
    width, height = images[0].size
    rows = (len(images) + cols - 1) // cols
    grid = Image.new(
        "L",
        (cols * (width * scale + pad) + pad, rows * (height * scale + pad) + pad),
        color=255,
    )
    for idx, image in enumerate(images):
        image = ImageOps.autocontrast(image).resize(
            (width * scale, height * scale), Image.Resampling.NEAREST
        )
        row, col = divmod(idx, cols)
        x = pad + col * (width * scale + pad)
        y = pad + row * (height * scale + pad)
        grid.paste(image, (x, y))
    return grid


def collect_real_examples(dataset_dir: Path, per_class: int) -> list[Path]:
    selected: list[Path] = []
    for label in range(10):
        paths = sorted((dataset_dir / str(label)).glob("*.png"))[:per_class]
        if len(paths) < per_class:
            raise ValueError(f"Class {label} only has {len(paths)} images, expected {per_class}.")
        selected.extend(paths)
    return selected


def collect_generated_examples(sample_dir: Path, count: int) -> list[Path]:
    paths = sorted(sample_dir.rglob("*.png"))[:count]
    if len(paths) < count:
        raise ValueError(f"Only found {len(paths)} generated images in {sample_dir}, expected {count}.")
    return paths


def draw_panel(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    label: str,
    image: Image.Image,
    x: int,
    y: int,
    label_height: int,
) -> int:
    draw.text((x, y), label, fill="black")
    canvas.paste(Image.merge("RGB", (image, image, image)), (x, y + label_height))
    return y + label_height + image.height


def main() -> None:
    args = parse_args()

    real_paths = collect_real_examples(args.dataset_dir, args.per_class)
    baseline_paths = collect_generated_examples(args.baseline_samples, args.generated_count)
    wdro_paths = collect_generated_examples(args.wdro_samples, args.generated_count)

    real_grid = make_grid(real_paths, cols=args.cols, scale=args.scale, pad=args.pad)
    baseline_grid = make_grid(baseline_paths, cols=args.cols, scale=args.scale, pad=args.pad)
    wdro_grid = make_grid(wdro_paths, cols=args.cols, scale=args.scale, pad=args.pad)

    baseline_fid = load_eval_fid(args.baseline_eval)
    wdro_fid = load_eval_fid(args.wdro_eval)

    panels: list[tuple[str, Image.Image]] = [
        (f"Real MNIST ({args.per_class} per class)", real_grid),
        (
            f"Baseline generated"
            + (f"  |  FID {baseline_fid:.3f}" if baseline_fid is not None else ""),
            baseline_grid,
        ),
        (
            f"WDRO generated"
            + (f"  |  FID {wdro_fid:.3f}" if wdro_fid is not None else ""),
            wdro_grid,
        ),
    ]

    label_height = 28
    margin = 12
    width = max(image.width for _, image in panels) + margin * 2
    height = sum(image.height + label_height + margin for _, image in panels) + margin

    canvas = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(canvas)

    y = margin
    for label, image in panels:
        y = draw_panel(canvas, draw, label, image, margin, y, label_height) + margin

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
