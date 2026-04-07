#!/usr/bin/env python3
"""Replace symlinks under toy_pngs/ with real copied files."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize toy_pngs symlinks into real files.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("toy_pngs"),
        help="Directory to scan relative to the repo root.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned replacements without modifying files.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = _repo_root()
    root = (repo_root / args.root).resolve()
    if not root.exists():
        raise SystemExit(f"[error] root does not exist: {root}")

    replaced = 0
    skipped = 0

    for path in sorted(root.rglob("*")):
        if not path.is_symlink():
            continue

        target = path.resolve(strict=True)
        if not target.is_file():
            raise SystemExit(f"[error] symlink target is not a file: {path} -> {target}")

        print(f"[materialize] {path} <- {target}")
        if args.dry_run:
            replaced += 1
            continue

        tmp_path = path.with_name(f".{path.name}.tmp")
        if tmp_path.exists() or tmp_path.is_symlink():
            tmp_path.unlink()
        shutil.copy2(target, tmp_path)
        path.unlink()
        tmp_path.replace(path)
        replaced += 1

    if replaced == 0:
        skipped += 1
        print(f"[done] no symlinks found under {root}")
    else:
        print(f"[done] materialized={replaced} root={root}")
    if skipped:
        print(f"[done] skipped={skipped}")


if __name__ == "__main__":
    main()
