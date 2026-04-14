#!/usr/bin/env python3
import argparse
import csv
import json
import math
import sys
from bisect import bisect_left
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.cifar_cdro_budget_utils import (  # noqa: E402
    DEFAULT_CALIBRATION_JSON,
    DEFAULT_PYTORCH_FID_REF,
    DEFAULT_WDRO_SUMMARY_JSON,
    advedm_robust_step_compute_be,
    advedm_robust_step_weighted_compute_units,
    calibration_from_path,
    load_warmup_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a posthoc FID manifest for a CIFAR-10 CDRO-style image run using "
            "the same weighted-compute and wall-clock accounting scheme as the existing "
            "Baseline-vs-WDRO sweep."
        )
    )
    parser.add_argument("--cdro-run-dir", type=str, required=True)
    parser.add_argument("--summary-json", type=str, default=DEFAULT_WDRO_SUMMARY_JSON)
    parser.add_argument("--calibration-json", type=str, default=DEFAULT_CALIBRATION_JSON)
    parser.add_argument("--outdir", type=str, default="")
    parser.add_argument("--manifest-name", type=str, default="cifar10_cdro_budget_manifest.csv")
    parser.add_argument("--summary-name", type=str, default="cifar10_cdro_budget_manifest_summary.json")
    parser.add_argument("--kimg", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--adv-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-percent-label", type=str, default="20%")
    parser.add_argument("--ref-path", type=str, default=DEFAULT_PYTORCH_FID_REF)
    return parser.parse_args()


def _safe_mean_field(payload: Dict, key: str) -> float:
    value = payload[key]
    if isinstance(value, dict):
        value = value.get("mean")
    return float(value)


def load_stats_trace(path: Path) -> List[Tuple[float, float]]:
    trace: List[Tuple[float, float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            trace.append(
                (
                    _safe_mean_field(payload, "Progress/kimg"),
                    _safe_mean_field(payload, "Timing/total_sec"),
                )
            )
    if not trace:
        raise RuntimeError(f"No stats rows found in {path}")
    trace.sort(key=lambda item: item[0])
    return trace


def interpolate_total_sec(trace: Sequence[Tuple[float, float]], target_kimg: float) -> float:
    xs = [item[0] for item in trace]
    ys = [item[1] for item in trace]
    if target_kimg <= xs[0]:
        if xs[0] == target_kimg:
            return float(ys[0])
        raise RuntimeError(
            f"Target kimg {target_kimg} is before the first recorded stats point {xs[0]:.4f}."
        )
    if target_kimg >= xs[-1]:
        if xs[-1] == target_kimg:
            return float(ys[-1])
        raise RuntimeError(
            f"Target kimg {target_kimg} is after the last recorded stats point {xs[-1]:.4f}."
        )
    idx = bisect_left(xs, target_kimg)
    if idx < len(xs) and xs[idx] == target_kimg:
        return float(ys[idx])
    x0, y0 = xs[idx - 1], ys[idx - 1]
    x1, y1 = xs[idx], ys[idx]
    frac = (float(target_kimg) - float(x0)) / max(float(x1) - float(x0), 1e-12)
    return float(y0 + frac * (y1 - y0))


def parse_kimg_list(text: str) -> List[int]:
    out: List[int] = []
    for chunk in str(text).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        out.append(int(chunk))
    return sorted(set(out))


def discover_snapshot_kimg(run_dir: Path) -> List[int]:
    out = []
    for snapshot in sorted(run_dir.glob("network-snapshot-*.pkl")):
        suffix = snapshot.stem.replace("network-snapshot-", "")
        if suffix.isdigit():
            out.append(int(suffix))
    return sorted(set(out))


def _series_fields(*, method_version_used: str) -> Dict[str, str]:
    return {
        "method": "cdro",
        "robust_method": "cdro",
        "robust_label": "CDRO",
        "training_objective": "edm",
        "backbone_family": "edm",
        "backbone_label": "EDM",
        "series_key": "cdro_edm",
        "series_label": "CDRO EDM",
        "method_version_used": method_version_used,
    }


def build_rows(args: argparse.Namespace) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    cdro_run_dir = Path(args.cdro_run_dir).resolve()
    budget_plan_path = cdro_run_dir / "cdro_budget_plan.json"
    if not budget_plan_path.is_file():
        raise FileNotFoundError(f"Missing launch metadata: {budget_plan_path}")
    budget_plan = json.loads(budget_plan_path.read_text(encoding="utf-8"))

    calibration = calibration_from_path(args.calibration_json)
    warmup_summary = load_warmup_summary(args.summary_json)
    trace = load_stats_trace(cdro_run_dir / "stats.jsonl")
    requested_kimg = parse_kimg_list(args.kimg)
    snapshot_kimg = requested_kimg if requested_kimg else discover_snapshot_kimg(cdro_run_dir)
    warmup_kimg = int(round(float(warmup_summary["warmup_kimg"])))
    snapshot_kimg = [value for value in snapshot_kimg if int(value) > warmup_kimg]
    if not snapshot_kimg:
        raise RuntimeError(f"No robust-phase snapshots found in {cdro_run_dir}")

    robust_step_wcu = advedm_robust_step_weighted_compute_units(
        calibration=calibration,
        adv_steps=args.adv_steps,
    )
    robust_step_compute_be = advedm_robust_step_compute_be(adv_steps=args.adv_steps)
    warmup_wcu = float(warmup_summary["warmup_weighted_compute_units"])
    warmup_sec = float(warmup_summary["warmup_train_wall_clock_sec"])
    warmup_compute_be = float(warmup_summary["warmup_compute_be"])

    if args.outdir:
        outdir = Path(args.outdir).resolve()
    else:
        outdir = (ROOT_DIR / "training-runs" / "fid-sweeps" / cdro_run_dir.name).resolve()
    eval_root = outdir / "evals"

    rows: List[Dict[str, object]] = []
    for kimg in snapshot_kimg:
        snapshot = cdro_run_dir / f"network-snapshot-{int(kimg):06d}.pkl"
        if not snapshot.is_file():
            raise FileNotFoundError(f"Missing snapshot: {snapshot}")
        robust_steps = max((float(kimg) - float(warmup_kimg)) * 1000.0 / float(args.batch_size), 0.0)
        robust_wcu = float(robust_steps) * float(robust_step_wcu)
        robust_compute_be = float(robust_steps) * float(robust_step_compute_be)
        robust_sec = interpolate_total_sec(trace, float(kimg))
        total_sec = warmup_sec + robust_sec
        eval_tag = f"cdro_kimg{int(kimg):06d}"
        rows.append(
            {
                **_series_fields(method_version_used="advedm"),
                "seed": int(args.seed),
                "step": int(kimg),
                "snapshot_kimg": int(kimg),
                "images_shown_m": float(kimg) / 1000.0,
                "row_origin": "trajectory_robust_phase",
                "run_dir": str(cdro_run_dir),
                "network_pkl": str(snapshot),
                "metrics_path": str(cdro_run_dir / "stats.jsonl"),
                "train_wall_clock_sec": float(total_sec),
                "train_elapsed_sec": float(total_sec),
                "train_gpu_hours": float(total_sec) / 3600.0,
                "train_wall_clock_source": "normalized_baseline_warmup_plus_observed_cdro_stats_jsonl",
                "baseline_train_wall_clock_sec_effective": float(warmup_sec),
                "robust_train_wall_clock_sec_effective": float(robust_sec),
                "baseline_train_wall_clock_source": "normalized_baseline_stats_jsonl",
                "weighted_compute_units": float(warmup_wcu + robust_wcu),
                "baseline_weighted_compute_units": float(warmup_wcu),
                "robust_weighted_compute_units": float(robust_wcu),
                "compute_budget_be": float(warmup_compute_be + robust_compute_be),
                "baseline_compute_be": float(warmup_compute_be),
                "robust_compute_be": float(robust_compute_be),
                "weighted_compute_source": "cifar_calibration_advedm_cdro",
                "warmup_steps_fixed": float(warmup_compute_be),
                "robust_steps_observed": float(robust_steps),
                "train_wall_clock_complete": True,
                "fid": "",
                "fid_evaluated": False,
                "fid_eval_selected": True,
                "fid_source": "",
                "eval_root": str(eval_root),
                "eval_tag": eval_tag,
                "eval_result_json": str(eval_root / eval_tag / "evaluation_result.json"),
                "eval_result_txt": str(eval_root / eval_tag / "evaluation_result.txt"),
                "eval_status": "pending",
                "ref_path": str(Path(args.ref_path).resolve()),
                "train_percent_label": str(args.train_percent_label),
                "eval_dir": str(eval_root / eval_tag),
                "loss_final": "",
                "loss_mean_last": "",
                "fixed_warmup_steps": float(warmup_compute_be),
            }
        )

    summary = {
        "manifest_version": "cifar_cdro_fid_manifest_v1",
        "cdro_run_dir": str(cdro_run_dir),
        "calibration_json": str(Path(args.calibration_json).resolve()),
        "summary_json": str(Path(args.summary_json).resolve()),
        "warmup_summary": warmup_summary,
        "adv_steps": int(args.adv_steps),
        "robust_step_weighted_compute_units": float(robust_step_wcu),
        "robust_step_compute_be": float(robust_step_compute_be),
        "snapshot_kimg": [int(value) for value in snapshot_kimg],
        "budget_plan_path": str(budget_plan_path),
        "budget_plan": budget_plan,
        "num_rows": int(len(rows)),
    }
    return rows, summary


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
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


def main() -> None:
    args = parse_args()
    rows, summary = build_rows(args)
    if args.outdir:
        outdir = Path(args.outdir).resolve()
    else:
        outdir = (ROOT_DIR / "training-runs" / "fid-sweeps" / Path(args.cdro_run_dir).resolve().name).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    manifest_path = outdir / args.manifest_name
    summary_path = outdir / args.summary_name
    write_csv(manifest_path, rows)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[OK] Wrote CDRO manifest: {manifest_path}")
    print(f"[OK] Wrote CDRO summary:  {summary_path}")


if __name__ == "__main__":
    main()
