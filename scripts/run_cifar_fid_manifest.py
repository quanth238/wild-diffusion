#!/usr/bin/env python3
import argparse
import csv
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List


ROOT_DIR = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run posthoc CIFAR-10 FID evaluation for rows in a manifest CSV."
    )
    parser.add_argument("--manifest-csv", type=str, required=True)
    parser.add_argument("--eval-root", type=str, default="")
    parser.add_argument("--methods", type=str, default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num-images", type=int, default=50000)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--gen-batch", type=int, default=128)
    parser.add_argument("--fid-batch", type=int, default=64)
    parser.add_argument("--gen-steps", type=int, default=18)
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--env-mode", type=str, default="venv")
    parser.add_argument("--venv-dir", type=str, default="/home/bachlc/.venvs/wild-diffusion-h100")
    parser.add_argument("--install-deps", type=str, default="0")
    parser.add_argument("--prepare-dataset", type=str, default="0")
    parser.add_argument("--ref-mode", type=str, default="path")
    parser.add_argument("--ref-path", type=str, default="/home/bachlc/GM-CDRO/datasets/fid-refs/cifar10-32x32.npz")
    parser.add_argument("--only-pending", action="store_true", default=False)
    parser.add_argument("--force", action="store_true", default=False)
    return parser.parse_args()


def load_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_manifest(path: Path, rows: List[Dict[str, str]]) -> None:
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _safe_float(value: object):
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def should_run_row(row: Dict[str, str], *, allowed_methods: set, only_pending: bool, force: bool) -> bool:
    if allowed_methods and str(row.get("robust_method", "")).strip() not in allowed_methods:
        return False
    if not _truthy(row.get("fid_eval_selected", "")):
        return False
    if force:
        return True
    result_json = Path(str(row.get("eval_result_json", "")).strip())
    if result_json.is_file():
        return True
    if only_pending and _truthy(row.get("fid_evaluated", "")):
        return False
    return not _truthy(row.get("fid_evaluated", ""))


def update_row_from_result(row: Dict[str, str], result_json: Path) -> None:
    payload = json.loads(result_json.read_text(encoding="utf-8"))
    fid_value = payload.get("fid")
    row["fid"] = "" if fid_value is None else str(float(fid_value))
    row["fid_evaluated"] = "True"
    row["fid_source"] = "posthoc_cifar_eval"
    row["eval_status"] = "done"
    row["eval_result_json"] = str(result_json)
    row["eval_result_txt"] = str(result_json.with_name("evaluation_result.txt"))
    row["eval_dir"] = str(result_json.parent)


def run_single_eval(row: Dict[str, str], *, args: argparse.Namespace, eval_root: Path) -> None:
    eval_tag = str(row.get("eval_tag", "")).strip()
    if not eval_tag:
        raise RuntimeError(f"Row is missing eval_tag: {row}")
    network_pkl = str(row.get("network_pkl", "")).strip()
    if not network_pkl:
        raise RuntimeError(f"Row is missing network_pkl: {row}")
    run_dir = str(row.get("run_dir", "")).strip() or str(Path(network_pkl).resolve().parent)
    env = os.environ.copy()
    env.update(
        {
            "ENV_MODE": str(args.env_mode),
            "VENV_DIR": str(args.venv_dir),
            "INSTALL_DEPS": str(args.install_deps),
            "PREPARE_DATASET": str(args.prepare_dataset),
            "RUN_DIR": run_dir,
            "NETWORK_PKL": network_pkl,
            "EVAL_ROOT": str(eval_root),
            "EVAL_TAG": eval_tag,
            "NPROC_PER_NODE": str(args.nproc_per_node),
            "NUM_IMAGES": str(args.num_images),
            "SEED_START": str(args.seed_start),
            "GEN_BATCH": str(args.gen_batch),
            "FID_BATCH": str(args.fid_batch),
            "GEN_STEPS": str(args.gen_steps),
            "REF_MODE": str(args.ref_mode),
            "REF_PATH": str(args.ref_path),
            "PYTORCH_CUDA_ALLOC_CONF": env.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
        }
    )
    cmd = ["bash", str(ROOT_DIR / "scripts" / "setup_and_eval_cifar10.sh")]
    subprocess.run(
        cmd,
        cwd=str(ROOT_DIR),
        env=env,
        check=True,
    )


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest_csv).resolve()
    rows = load_manifest(manifest_path)
    if not rows:
        raise SystemExit(f"[ERROR] Manifest is empty: {manifest_path}")
    eval_root = Path(args.eval_root).resolve() if args.eval_root else Path(rows[0]["eval_root"]).resolve()
    eval_root.mkdir(parents=True, exist_ok=True)
    allowed_methods = {item.strip() for item in str(args.methods).split(",") if item.strip()}

    selected_rows = [
        row
        for row in rows
        if should_run_row(
            row,
            allowed_methods=allowed_methods,
            only_pending=bool(args.only_pending),
            force=bool(args.force),
        )
    ]
    if args.limit > 0:
        selected_rows = selected_rows[: int(args.limit)]
    if not selected_rows:
        print(f"[OK] Nothing to do for manifest: {manifest_path}")
        return

    print(f"[INFO] Running {len(selected_rows)} eval(s) from {manifest_path}")
    for row in selected_rows:
        result_json = eval_root / str(row["eval_tag"]) / "evaluation_result.json"
        row["eval_root"] = str(eval_root)
        row["eval_result_json"] = str(result_json)
        row["eval_result_txt"] = str(result_json.with_name("evaluation_result.txt"))
        if result_json.is_file() and not args.force:
            update_row_from_result(row, result_json)
            write_manifest(manifest_path, rows)
            print(
                f"[SKIP] Reused existing result for {row['robust_method']} "
                f"kimg={row['snapshot_kimg']} fid={row['fid']}"
            )
            continue
        row["eval_status"] = "running"
        write_manifest(manifest_path, rows)
        print(f"[RUN ] {row['robust_method']} kimg={row['snapshot_kimg']} tag={row['eval_tag']}")
        run_single_eval(row, args=args, eval_root=eval_root)
        if not result_json.is_file():
            raise RuntimeError(f"Expected evaluation result JSON was not created: {result_json}")
        update_row_from_result(row, result_json)
        write_manifest(manifest_path, rows)
        print(f"[DONE] {row['robust_method']} kimg={row['snapshot_kimg']} fid={row['fid']}")

    print(f"[OK] Manifest updated in place: {manifest_path}")


if __name__ == "__main__":
    main()
