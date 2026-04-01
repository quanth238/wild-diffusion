#!/usr/bin/env python3
"""Build a visual panel: Real | Baseline EDM | (optional v1.1) | v2 | v2.1."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Sequence

import numpy as np
from PIL import Image, ImageDraw

try:
    _RESAMPLE_NEAREST = Image.Resampling.NEAREST
except AttributeError:  # Pillow < 10
    _RESAMPLE_NEAREST = Image.NEAREST


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _pngs_in_dir(directory: Path) -> List[Path]:
    files = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".png")
    if not files:
        raise FileNotFoundError(f"No PNG files found in: {directory}")
    return files


def _pick_indices(max_count: int, n_pick: int, seed: int) -> List[int]:
    if max_count < n_pick:
        raise ValueError(f"Not enough images: need {n_pick}, but only {max_count} are available.")
    idx = list(range(max_count))
    rng = random.Random(seed)
    rng.shuffle(idx)
    return idx[:n_pick]


def _build_grid(
    files: Sequence[Path],
    indices: Sequence[int],
    rows: int,
    cols: int,
    tile_size: int,
    tile_pad: int,
) -> Image.Image:
    width = cols * tile_size + (cols - 1) * tile_pad
    height = rows * tile_size + (rows - 1) * tile_pad
    canvas = np.zeros((height, width), dtype=np.uint8)

    for i, idx in enumerate(indices):
        if i >= rows * cols:
            break
        row = i // cols
        col = i % cols
        img = Image.open(files[idx]).convert("L").resize((tile_size, tile_size), _RESAMPLE_NEAREST)
        tile = np.asarray(img, dtype=np.uint8)
        y0 = row * (tile_size + tile_pad)
        x0 = col * (tile_size + tile_pad)
        canvas[y0 : y0 + tile_size, x0 : x0 + tile_size] = tile

    return Image.fromarray(canvas, mode="L").convert("RGB")


def _draw_panel(columns: Sequence[Image.Image], titles: Sequence[str], out_path: Path) -> None:
    if len(columns) != len(titles):
        raise ValueError("columns and titles must have the same length.")

    col_w = max(im.width for im in columns)
    col_h = max(im.height for im in columns)
    n = len(columns)

    outer_pad = 16
    col_gap = 18
    title_h = 28
    total_w = outer_pad * 2 + n * col_w + (n - 1) * col_gap
    total_h = outer_pad * 2 + title_h + col_h

    panel = Image.new("RGB", (total_w, total_h), color=(232, 232, 232))
    draw = ImageDraw.Draw(panel)

    for i, (title, col_img) in enumerate(zip(titles, columns)):
        x = outer_pad + i * (col_w + col_gap)
        y = outer_pad + title_h
        panel.paste(col_img, (x, y))

        bbox = draw.textbbox((0, 0), title)
        text_w = bbox[2] - bbox[0]
        text_x = x + (col_w - text_w) // 2
        draw.text((text_x, outer_pad), title, fill=(16, 16, 16))

        draw.rectangle([x - 1, y - 1, x + col_img.width, y + col_img.height], outline=(80, 80, 80), width=1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(out_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create side-by-side MNIST panel for Real/Baseline/(optional v1.1)/v2/v2.1."
    )
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--prefix", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--real-dir", type=Path, default=None)
    parser.add_argument("--baseline-dir", type=Path, default=None)
    parser.add_argument("--v11-dir", type=Path, default=None)
    parser.add_argument("--v2-dir", type=Path, default=None)
    parser.add_argument("--v21-dir", type=Path, default=None)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--cols", type=int, default=8)
    parser.add_argument("--tile-size", type=int, default=32)
    parser.add_argument("--tile-pad", type=int, default=2)
    parser.add_argument("--pick-seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.rows <= 0 or args.cols <= 0:
        raise ValueError("rows/cols must be positive.")
    if args.tile_size <= 0:
        raise ValueError("tile-size must be positive.")
    if args.tile_pad < 0:
        raise ValueError("tile-pad must be >= 0.")

    exp_v2 = args.outdir / f"{args.prefix}_v2_s{args.seed}"
    exp_v21 = args.outdir / f"{args.prefix}_2_1_s{args.seed}"
    real_dir = args.real_dir or (_repo_root() / "toy_outputs" / "fid_ref_images")
    baseline_dir = args.baseline_dir or (exp_v2 / "fid_baseline")
    v11_dir = args.v11_dir
    v2_dir = args.v2_dir or (exp_v2 / "fid_robust")
    v21_dir = args.v21_dir or (exp_v21 / "fid_robust")

    real_pngs = _pngs_in_dir(real_dir)
    baseline_pngs = _pngs_in_dir(baseline_dir)
    v11_pngs = _pngs_in_dir(v11_dir) if v11_dir is not None else None
    v2_pngs = _pngs_in_dir(v2_dir)
    v21_pngs = _pngs_in_dir(v21_dir)

    n_show = args.rows * args.cols
    n_common = min(
        len(real_pngs),
        len(baseline_pngs),
        len(v2_pngs),
        len(v21_pngs),
        len(v11_pngs) if v11_pngs is not None else 10**12,
    )
    picked = _pick_indices(n_common, n_show, seed=args.pick_seed)

    columns = [
        _build_grid(real_pngs, picked, args.rows, args.cols, args.tile_size, args.tile_pad),
        _build_grid(baseline_pngs, picked, args.rows, args.cols, args.tile_size, args.tile_pad),
    ]
    titles = ["Real", "Baseline EDM"]
    if v11_pngs is not None:
        columns.append(_build_grid(v11_pngs, picked, args.rows, args.cols, args.tile_size, args.tile_pad))
        titles.append("v1.1 Robust")
    columns.extend(
        [
            _build_grid(v2_pngs, picked, args.rows, args.cols, args.tile_size, args.tile_pad),
            _build_grid(v21_pngs, picked, args.rows, args.cols, args.tile_size, args.tile_pad),
        ]
    )
    titles.extend(["v2 Robust", "v2.1 Robust"])

    default_name = (
        f"{args.prefix}_panel_real_baseline_v11_v2_v21_s{args.seed}.png"
        if v11_pngs is not None
        else f"{args.prefix}_panel_real_baseline_v2_v21_s{args.seed}.png"
    )
    output = args.output or (args.outdir / default_name)
    _draw_panel(columns=columns, titles=titles, out_path=output)

    print(f"[done] panel: {output}")
    print(f"[info] real_dir={real_dir}")
    print(f"[info] baseline_dir={baseline_dir}")
    if v11_dir is not None:
        print(f"[info] v11_dir={v11_dir}")
    print(f"[info] v2_dir={v2_dir}")
    print(f"[info] v21_dir={v21_dir}")
    print(f"[info] picked_images={n_show} (common_pool={n_common}, pick_seed={args.pick_seed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
