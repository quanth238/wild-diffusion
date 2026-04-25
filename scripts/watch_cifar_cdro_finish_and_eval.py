#!/usr/bin/env python3
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = (
    "/home/bachlc/GM-CDRO/training-runs/paper-cifar10-cdro-fp16/"
    "00000-cifar10-32x32-train20pct-seed0-uncond-ddpmpp-cdroedm-gpus1-batch1024-fp16-"
    "paper-cifar10-uncond-ddpmpp-cdro-20pct-n032-rho32p0-i1-aw0p30-cw0p00-bg1024-"
    "resume040000-wcu537488"
)
DEFAULT_JOB_ID = "10076"
DEFAULT_BASE_COMPARE_CSV = (
    "/home/bachlc/GM-CDRO/training-runs/fid-sweeps/cifar10_baseline_vs_wdro_coarse_20260414/"
    "cifar10_baseline_vs_wdro_coarse_20260414_three_method_compare.csv"
)
DEFAULT_MERGED_COMPARE_CSV = (
    "/home/bachlc/GM-CDRO/training-runs/fid-sweeps/cifar10_baseline_vs_wdro_coarse_20260414/"
    "cifar10_baseline_vs_wdro_cdro_budgeted_compare.csv"
)
DEFAULT_MERGED_SUMMARY_JSON = (
    "/home/bachlc/GM-CDRO/training-runs/fid-sweeps/cifar10_baseline_vs_wdro_coarse_20260414/"
    "cifar10_baseline_vs_wdro_cdro_budgeted_compare_summary.json"
)
DEFAULT_PYTORCH_FID_REF = (
    "/home/bachlc/GM-CDRO/training-runs/fid-sweeps/cifar10_baseline_vs_wdro_coarse_20260414/"
    "pytorch_fid_cifar10_train_ref_stats.npz"
)
DEFAULT_VENV_DIR = "/home/bachlc/.venvs/wild-diffusion-h100"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for a live CIFAR CDRO run to finish, then build its FID manifest, "
            "run pending posthoc FID evaluations inside the existing Slurm allocation, "
            "and emit a merged baseline+WDRO+CDRO comparison CSV."
        )
    )
    parser.add_argument("--run-dir", type=str, default=DEFAULT_RUN_DIR)
    parser.add_argument("--job-id", type=str, default=DEFAULT_JOB_ID)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--skip-wait", action="store_true", default=False)
    parser.add_argument("--manifest-outdir", type=str, default="")
    parser.add_argument("--manifest-name", type=str, default="cifar10_cdro_budget_manifest.csv")
    parser.add_argument("--manifest-summary-name", type=str, default="cifar10_cdro_budget_manifest_summary.json")
    parser.add_argument("--base-compare-csv", type=str, default=DEFAULT_BASE_COMPARE_CSV)
    parser.add_argument("--merged-compare-csv", type=str, default=DEFAULT_MERGED_COMPARE_CSV)
    parser.add_argument("--merged-summary-json", type=str, default=DEFAULT_MERGED_SUMMARY_JSON)
    parser.add_argument("--flop-calibration-json", type=str, default="")
    parser.add_argument("--env-mode", type=str, default="venv")
    parser.add_argument("--venv-dir", type=str, default=DEFAULT_VENV_DIR)
    parser.add_argument("--install-deps", type=str, default="0")
    parser.add_argument("--prepare-dataset", type=str, default="0")
    parser.add_argument("--fid-backend", type=str, choices=["edm", "pytorch_fid"], default="pytorch_fid")
    parser.add_argument("--ref-mode", type=str, default="path")
    parser.add_argument("--ref-path", type=str, default=DEFAULT_PYTORCH_FID_REF)
    parser.add_argument("--num-images", type=int, default=50000)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--gen-batch", type=int, default=128)
    parser.add_argument("--fid-batch", type=int, default=64)
    parser.add_argument("--gen-steps", type=int, default=18)
    parser.add_argument("--nproc-per-node", type=int, default=1)
    return parser.parse_args()


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


def load_training_options(run_dir: Path) -> Dict:
    return json.loads((run_dir / "training_options.json").read_text(encoding="utf-8"))


def load_last_stats_payload(run_dir: Path):
    stats_path = run_dir / "stats.jsonl"
    if not stats_path.is_file():
        return None
    lines = [line for line in stats_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return None
    return json.loads(lines[-1])


def mean_field(payload: Dict, key: str):
    value = payload.get(key)
    if isinstance(value, dict):
        value = value.get("mean")
    if value is None:
        return None
    return float(value)


def is_run_done(run_dir: Path, total_kimg: int) -> bool:
    final_snapshot = run_dir / f"network-snapshot-{int(total_kimg):06d}.pkl"
    if final_snapshot.is_file():
        return True
    payload = load_last_stats_payload(run_dir)
    if payload is not None:
        kimg = mean_field(payload, "Progress/kimg")
        if kimg is not None and kimg >= float(total_kimg):
            return True
    log_path = run_dir / "log.txt"
    if log_path.is_file():
        try:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        except Exception:
            tail = ""
        if "Exiting..." in tail:
            return True
    return False


def format_progress(run_dir: Path, total_kimg: int) -> str:
    payload = load_last_stats_payload(run_dir)
    if payload is None:
        return f"{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())} waiting: no stats yet"
    last_kimg = mean_field(payload, "Progress/kimg") or 0.0
    tick = int(round(mean_field(payload, "Progress/tick") or 0.0))
    sec_per_kimg = mean_field(payload, "Timing/sec_per_kimg")
    stats_age = max(time.time() - os.path.getmtime(run_dir / "stats.jsonl"), 0.0)
    est_kimg = float(last_kimg)
    eta_hr = None
    if sec_per_kimg is not None and sec_per_kimg > 0:
        est_kimg = min(last_kimg + stats_age / sec_per_kimg, float(total_kimg))
        eta_hr = max((float(total_kimg) - est_kimg) * sec_per_kimg / 3600.0, 0.0)
    pct = 100.0 * est_kimg / max(float(total_kimg), 1e-8)
    eta_text = "unknown" if eta_hr is None else f"{eta_hr:.2f}h"
    return (
        f"{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())} "
        f"tick={tick} est_kimg={est_kimg:.3f}/{int(total_kimg)} ({pct:.2f}%) "
        f"stats_age={int(stats_age)}s eta_finish={eta_text}"
    )


def wait_for_completion(run_dir: Path, total_kimg: int, poll_seconds: float) -> None:
    while True:
        if is_run_done(run_dir, total_kimg):
            print(f"[watch] detected completion for {run_dir}", flush=True)
            return
        print(f"[watch] {format_progress(run_dir, total_kimg)}", flush=True)
        time.sleep(max(float(poll_seconds), 5.0))


def run_subprocess(cmd: List[str], *, cwd: Path | None = None) -> None:
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd or ROOT_DIR), check=True)


def manifest_outdir_for_run(run_dir: Path, explicit_outdir: str) -> Path:
    if explicit_outdir:
        return Path(explicit_outdir).resolve()
    return (ROOT_DIR / "training-runs" / "fid-sweeps" / run_dir.name).resolve()


def build_cdro_manifest(args: argparse.Namespace, run_dir: Path, outdir: Path) -> Path:
    cmd = [
        sys.executable,
        str(ROOT_DIR / "scripts" / "build_cifar_cdro_fid_manifest.py"),
        "--cdro-run-dir",
        str(run_dir),
        "--outdir",
        str(outdir),
        "--manifest-name",
        str(args.manifest_name),
        "--summary-name",
        str(args.manifest_summary_name),
    ]
    if str(args.flop_calibration_json).strip():
        cmd.extend(["--flop-calibration-json", str(args.flop_calibration_json)])
    run_subprocess(cmd)
    return outdir / str(args.manifest_name)


def run_cdro_eval_in_job(args: argparse.Namespace, manifest_csv: Path, eval_root: Path) -> None:
    inner_cmd = [
        "cd",
        str(ROOT_DIR),
        "&&",
        "source",
        str(Path(args.venv_dir) / "bin" / "activate"),
        "&&",
        "python",
        str(ROOT_DIR / "scripts" / "run_cifar_fid_manifest.py"),
        "--manifest-csv",
        str(manifest_csv),
        "--eval-root",
        str(eval_root),
        "--methods",
        "cdro",
        "--only-pending",
        "--env-mode",
        str(args.env_mode),
        "--venv-dir",
        str(args.venv_dir),
        "--install-deps",
        str(args.install_deps),
        "--prepare-dataset",
        str(args.prepare_dataset),
        "--fid-backend",
        str(args.fid_backend),
        "--ref-mode",
        str(args.ref_mode),
        "--ref-path",
        str(args.ref_path),
        "--num-images",
        str(int(args.num_images)),
        "--seed-start",
        str(int(args.seed_start)),
        "--gen-batch",
        str(int(args.gen_batch)),
        "--fid-batch",
        str(int(args.fid_batch)),
        "--gen-steps",
        str(int(args.gen_steps)),
        "--nproc-per-node",
        str(int(args.nproc_per_node)),
    ]
    cmd = [
        "srun",
        f"--jobid={args.job_id}",
        "--overlap",
        "--quiet",
        "bash",
        "-lc",
        " ".join(inner_cmd),
    ]
    run_subprocess(cmd)


def load_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: List[Dict[str, object]]) -> None:
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def method_sort_key(method: str) -> tuple[int, str]:
    order = {"baseline": 0, "wild_diffusion": 1, "cdro": 2}
    return (order.get(str(method), 99), str(method))


def sort_rows(rows: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    return sorted(
        rows,
        key=lambda row: (
            method_sort_key(str(row.get("robust_method", row.get("method", "")))),
            int(float(row.get("step", 0) or 0)),
        ),
    )


def summarize_rows(rows: List[Dict[str, object]]) -> Dict[str, object]:
    methods = sorted(
        {
            str(row.get("robust_method", row.get("method", ""))).strip()
            for row in rows
            if str(row.get("robust_method", row.get("method", ""))).strip()
        },
        key=method_sort_key,
    )
    per_method: Dict[str, Dict[str, object]] = {}
    for method in methods:
        method_rows = [
            row for row in rows
            if str(row.get("robust_method", row.get("method", ""))).strip() == method
        ]
        scored_rows = [row for row in method_rows if _safe_float(row.get("fid")) is not None]
        if not scored_rows:
            continue
        best_row = min(scored_rows, key=lambda row: float(row["fid"]))
        final_row = max(scored_rows, key=lambda row: float(row["step"]))
        per_method[method] = {
            "num_rows": len(scored_rows),
            "best_fid": float(best_row["fid"]),
            "best_step": int(float(best_row["step"])),
            "best_weighted_compute_units": _safe_float(best_row.get("weighted_compute_units")),
            "best_total_train_pflops": _safe_float(best_row.get("total_train_pflops")),
            "best_train_wall_clock_sec": _safe_float(best_row.get("train_wall_clock_sec")),
            "final_step": int(float(final_row["step"])),
            "final_fid": float(final_row["fid"]),
            "final_weighted_compute_units": _safe_float(final_row.get("weighted_compute_units")),
            "final_total_train_pflops": _safe_float(final_row.get("total_train_pflops")),
            "final_train_wall_clock_sec": _safe_float(final_row.get("train_wall_clock_sec")),
        }
    return {"methods": per_method, "num_rows": len(rows)}


def merge_compare(args: argparse.Namespace, manifest_csv: Path) -> None:
    base_rows = load_csv_rows(Path(args.base_compare_csv).resolve())
    cdro_rows = [
        row for row in load_csv_rows(manifest_csv)
        if str(row.get("robust_method", row.get("method", ""))).strip() == "cdro"
        and _truthy(row.get("fid_evaluated"))
        and _safe_float(row.get("fid")) is not None
    ]
    merged_rows = sort_rows([*base_rows, *cdro_rows])
    merged_csv = Path(args.merged_compare_csv).resolve()
    write_csv_rows(merged_csv, merged_rows)
    summary_payload = {
        "created_at_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "base_compare_csv": str(Path(args.base_compare_csv).resolve()),
        "cdro_manifest_csv": str(manifest_csv.resolve()),
        "merged_compare_csv": str(merged_csv),
        "summary": summarize_rows(merged_rows),
    }
    summary_path = Path(args.merged_summary_json).resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(f"[merge] wrote {merged_csv}", flush=True)
    print(f"[merge] wrote {summary_path}", flush=True)
    methods = summary_payload["summary"]["methods"]
    for method in ("baseline", "wild_diffusion", "cdro"):
        if method in methods:
            record = methods[method]
            print(
                f"[merge] {method}: best_fid={record['best_fid']:.5f} at step={record['best_step']} "
                f"final_fid={record['final_fid']:.5f} at step={record['final_step']}",
                flush=True,
            )


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")
    opts = load_training_options(run_dir)
    total_kimg = int(opts["total_kimg"])
    outdir = manifest_outdir_for_run(run_dir, args.manifest_outdir)
    manifest_csv = outdir / str(args.manifest_name)
    eval_root = outdir / "evals"

    print(f"[watch] run_dir={run_dir}", flush=True)
    print(f"[watch] total_kimg={total_kimg}", flush=True)
    print(f"[watch] job_id={args.job_id}", flush=True)
    print(f"[watch] manifest_outdir={outdir}", flush=True)
    if not args.skip_wait:
        wait_for_completion(run_dir, total_kimg=total_kimg, poll_seconds=float(args.poll_seconds))
    else:
        print("[watch] skip-wait enabled; proceeding immediately.", flush=True)

    manifest_csv = build_cdro_manifest(args, run_dir=run_dir, outdir=outdir)
    run_cdro_eval_in_job(args, manifest_csv=manifest_csv, eval_root=eval_root)
    merge_compare(args, manifest_csv=manifest_csv)
    print("[watch] post-finish CDRO evaluation pipeline completed.", flush=True)


if __name__ == "__main__":
    main()
