from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

from PIL import ImageOps
from torchvision.datasets import MNIST


SPLIT_SIZES = {
    "train": 60_000,
    "test": 10_000,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare MNIST in ImageFolderDataset format.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("datasets"),
        help="Directory where converted MNIST datasets will be stored.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("datasets") / "raw",
        help="Directory where torchvision stores the raw MNIST files.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "test"),
        default=("train", "test"),
        help="Which MNIST splits to prepare.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=32,
        help="Output resolution. The default pads 28x28 MNIST to 32x32.",
    )
    parser.add_argument(
        "--zip",
        action="store_true",
        help="Also create zip archives alongside the directory datasets.",
    )
    parser.add_argument(
        "--rgb",
        action="store_true",
        help="Convert MNIST to RGB instead of keeping it grayscale.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild outputs even if they already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.data_root.mkdir(parents=True, exist_ok=True)
    args.raw_root.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        prepare_split(
            split=split,
            data_root=args.data_root,
            raw_root=args.raw_root,
            resolution=args.resolution,
            make_zip=args.zip,
            rgb=args.rgb,
            force=args.force,
        )


def prepare_split(
    *,
    split: str,
    data_root: Path,
    raw_root: Path,
    resolution: int,
    make_zip: bool,
    rgb: bool,
    force: bool,
) -> None:
    train = split == "train"
    dataset_name = f"mnist-{resolution}x{resolution}-{split}"
    out_dir = data_root / dataset_name
    out_zip = data_root / f"{dataset_name}.zip"
    expected_count = SPLIT_SIZES[split]

    if force:
        if out_dir.exists():
            shutil.rmtree(out_dir)
        if out_zip.exists():
            out_zip.unlink()

    if dataset_ready(out_dir=out_dir, expected_count=expected_count) and (not make_zip or out_zip.exists()):
        print(f"[INFO] Reusing existing MNIST {split} split: {out_dir}")
        return

    ds = MNIST(root=str(raw_root), train=train, download=True)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels: list[list[object]] = []
    for idx, (img, label) in enumerate(ds):
        converted = convert_image(img=img, resolution=resolution, rgb=rgb)
        class_dir = out_dir / str(label)
        class_dir.mkdir(parents=True, exist_ok=True)
        rel_path = Path(str(label)) / f"{idx:05d}.png"
        converted.save(out_dir / rel_path)
        labels.append([rel_path.as_posix(), int(label)])

    labels_path = out_dir / "dataset.json"
    labels_path.write_text(json.dumps({"labels": labels}, indent=2), encoding="utf-8")
    print(f"[INFO] Wrote {len(labels)} images to {out_dir}")

    if make_zip:
        create_zip(dataset_dir=out_dir, zip_path=out_zip)
        print(f"[INFO] Wrote {out_zip}")


def dataset_ready(*, out_dir: Path, expected_count: int) -> bool:
    labels_path = out_dir / "dataset.json"
    png_count = sum(1 for _ in out_dir.rglob("*.png")) if out_dir.exists() else 0
    return out_dir.exists() and labels_path.exists() and png_count == expected_count


def convert_image(*, img, resolution: int, rgb: bool):
    if img.size[0] > resolution or img.size[1] > resolution:
        raise ValueError(f"MNIST source image is larger than requested resolution {resolution}: {img.size}")

    pad_x = resolution - img.size[0]
    pad_y = resolution - img.size[1]
    if pad_x < 0 or pad_y < 0:
        raise ValueError(f"Invalid resolution {resolution} for source image size {img.size}")

    left = pad_x // 2
    right = pad_x - left
    top = pad_y // 2
    bottom = pad_y - top
    converted = ImageOps.expand(img, border=(left, top, right, bottom), fill=0)
    if rgb:
        converted = converted.convert("RGB")
    return converted


def create_zip(*, dataset_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(dataset_dir.rglob("*")):
            if path.is_dir():
                continue
            archive.write(path, arcname=path.relative_to(dataset_dir).as_posix())


if __name__ == "__main__":
    main()
