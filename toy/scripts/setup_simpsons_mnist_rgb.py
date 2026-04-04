#!/usr/bin/env python3
"""Stage the exact RGB split from alvarobartt/simpsons-mnist for toy image runs."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


DEFAULT_REPO_URL = "https://github.com/alvarobartt/simpsons-mnist"
DEFAULT_CLONE_DIR = ".tmp/simpsons-mnist"
DEFAULT_OUT_ROOT = "toy_data/simpsons_mnist_rgb"
DEFAULT_FID_REF_NAME = "simpsons_mnist_rgb_test_28x28.npz"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _clone_or_update(repo_url: str, clone_dir: Path) -> str:
    clone_dir.parent.mkdir(parents=True, exist_ok=True)
    if not clone_dir.exists():
        subprocess.run(["git", "clone", "--depth", "1", repo_url, str(clone_dir)], check=True)
    else:
        subprocess.run(["git", "-C", str(clone_dir), "fetch", "--depth", "1", "origin", "master"], check=True)
        subprocess.run(["git", "-C", str(clone_dir), "reset", "--hard", "FETCH_HEAD"], check=True)
    commit = subprocess.check_output(["git", "-C", str(clone_dir), "rev-parse", "HEAD"], text=True).strip()
    return commit


def _replace_symlink(path: Path, target: Path) -> None:
    if path.is_symlink() or path.exists():
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(target)


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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prepare the exact RGB simpsons-mnist dataset for toy image runs.")
    p.add_argument("--repo-url", type=str, default=DEFAULT_REPO_URL)
    p.add_argument("--clone-dir", type=Path, default=Path(DEFAULT_CLONE_DIR))
    p.add_argument("--out-root", type=Path, default=Path(DEFAULT_OUT_ROOT))
    p.add_argument("--skip-fid-ref", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    repo_root = _repo_root()
    clone_dir = (repo_root / args.clone_dir).resolve()
    out_root = (repo_root / args.out_root).resolve()
    imagefolder_root = out_root / "imagefolder"
    train_root = imagefolder_root / "train"
    test_root = imagefolder_root / "test"
    fid_ref_root = out_root / "fid_refs"
    fid_ref_root.mkdir(parents=True, exist_ok=True)
    fid_ref_path = fid_ref_root / DEFAULT_FID_REF_NAME
    summary_path = out_root / "summary.json"

    print(f"[setup] repo_url={args.repo_url}", flush=True)
    commit = _clone_or_update(args.repo_url, clone_dir)
    rgb_root = clone_dir / "dataset" / "rgb"
    source_train_root = (rgb_root / "train").resolve()
    source_test_root = (rgb_root / "test").resolve()
    if not source_train_root.is_dir() or not source_test_root.is_dir():
        raise SystemExit(f"[error] expected rgb train/test under {rgb_root}")

    _replace_symlink(train_root, source_train_root)
    _replace_symlink(test_root, source_test_root)

    class_counts = {}
    total_train = 0
    total_test = 0
    for cls_dir in sorted([p for p in source_train_root.iterdir() if p.is_dir()]):
        train_count = len([p for p in cls_dir.rglob("*") if p.is_file()])
        test_cls_dir = source_test_root / cls_dir.name
        test_count = len([p for p in test_cls_dir.rglob("*") if p.is_file()]) if test_cls_dir.is_dir() else 0
        class_counts[cls_dir.name] = {"train": train_count, "test": test_count}
        total_train += train_count
        total_test += test_count

    if not args.skip_fid_ref:
        print(f"[fid-ref] building from {test_root} -> {fid_ref_path}", flush=True)
        _build_fid_ref(repo_root, test_root, fid_ref_path)

    payload = {
        "source": {
            "repo_url": args.repo_url,
            "clone_dir": str(clone_dir),
            "commit": commit,
            "rgb_root": str(rgb_root.resolve()),
        },
        "dataset": {
            "image_size": 28,
            "image_channels": 3,
            "num_classes": len(class_counts),
            "train_total": total_train,
            "test_total": total_test,
            "class_counts": class_counts,
        },
        "imagefolder_root": str(imagefolder_root),
        "train_root": str(train_root),
        "test_root": str(test_root),
        "fid_ref_path": None if args.skip_fid_ref else str(fid_ref_path),
    }
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[done] summary={summary_path}", flush=True)
    print(f"[done] train_root={train_root}", flush=True)
    print(f"[done] test_root={test_root}", flush=True)
    if not args.skip_fid_ref:
        print(f"[done] fid_ref={fid_ref_path}", flush=True)


if __name__ == "__main__":
    main()
