#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import torch
from pytorch_fid.fid_score import (
    IMAGE_EXTENSIONS,
    calculate_activation_statistics,
    calculate_frechet_distance,
)
from pytorch_fid.inception import InceptionV3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recursive pytorch-fid wrapper that supports nested image directories."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    save_parser = subparsers.add_parser("save-stats")
    save_parser.add_argument("--images", type=str, required=True)
    save_parser.add_argument("--dest", type=str, required=True)
    save_parser.add_argument("--batch-size", type=int, default=64)
    save_parser.add_argument("--device", type=str, default="cuda")
    save_parser.add_argument("--dims", type=int, choices=(64, 192, 768, 2048), default=2048)
    save_parser.add_argument("--num-workers", type=int, default=4)

    calc_parser = subparsers.add_parser("calc")
    calc_parser.add_argument("--images", type=str, required=True)
    calc_parser.add_argument("--ref", type=str, required=True)
    calc_parser.add_argument("--batch-size", type=int, default=64)
    calc_parser.add_argument("--device", type=str, default="cuda")
    calc_parser.add_argument("--dims", type=int, choices=(64, 192, 768, 2048), default=2048)
    calc_parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def collect_image_files(root: Path) -> list[str]:
    files = []
    for ext in IMAGE_EXTENSIONS:
        files.extend(root.rglob(f"*.{ext}"))
        files.extend(root.rglob(f"*.{ext.upper()}"))
    unique = sorted({str(path) for path in files if path.is_file()})
    if not unique:
        raise RuntimeError(f"No image files found under {root}")
    return unique


def build_model(*, dims: int, device: str) -> InceptionV3:
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[int(dims)]
    model = InceptionV3([block_idx]).to(device)
    model.eval()
    return model


def compute_stats(*, image_root: Path, model: InceptionV3, batch_size: int, dims: int, device: str, num_workers: int):
    files = collect_image_files(image_root)
    return calculate_activation_statistics(
        files,
        model,
        batch_size=int(batch_size),
        dims=int(dims),
        device=device,
        num_workers=int(num_workers),
    )


def main() -> None:
    args = parse_args()
    device = args.device
    model = build_model(dims=args.dims, device=device)

    if args.command == "save-stats":
        image_root = Path(args.images).resolve()
        dest = Path(args.dest).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        mu, sigma = compute_stats(
            image_root=image_root,
            model=model,
            batch_size=args.batch_size,
            dims=args.dims,
            device=device,
            num_workers=args.num_workers,
        )
        np.savez(dest, mu=mu, sigma=sigma)
        print(f"[OK] Saved recursive pytorch-fid stats: {dest}")
        return

    if args.command == "calc":
        image_root = Path(args.images).resolve()
        ref = Path(args.ref).resolve()
        mu, sigma = compute_stats(
            image_root=image_root,
            model=model,
            batch_size=args.batch_size,
            dims=args.dims,
            device=device,
            num_workers=args.num_workers,
        )
        if ref.suffix == ".npz":
            with np.load(ref) as payload:
                mu_ref = payload["mu"][:]
                sigma_ref = payload["sigma"][:]
        else:
            mu_ref, sigma_ref = compute_stats(
                image_root=ref,
                model=model,
                batch_size=args.batch_size,
                dims=args.dims,
                device=device,
                num_workers=args.num_workers,
            )
        fid = calculate_frechet_distance(mu, sigma, mu_ref, sigma_ref)
        print(f"FID: {float(fid):g}")
        return

    raise RuntimeError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
