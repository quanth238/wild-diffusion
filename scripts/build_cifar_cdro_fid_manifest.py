#!/usr/bin/env python3
import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.cifar_cdro_budget_utils import (  # noqa: E402
    DEFAULT_CALIBRATION_JSON,
    DEFAULT_PYTORCH_FID_REF,
    DEFAULT_WDRO_SUMMARY_JSON,
    calibration_from_path,
    cdro_robust_step_compute_be,
    cdro_robust_step_weighted_compute_units,
    load_warmup_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a posthoc FID manifest for a real CIFAR-10 CDRO run using "
            "the same weighted-compute and wall-clock accounting scheme as the "
            "existing Baseline-vs-WDRO sweep."
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-percent-label", type=str, default="20%")
    parser.add_argument("--ref-path", type=str, default=DEFAULT_PYTORCH_FID_REF)
    return parser.parse_args()


def _safe_float(value):
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return float(parsed)


def _safe_mean_field(payload: Dict, key: str):
    value = payload.get(key)
    if isinstance(value, dict):
        value = value.get("mean")
    return _safe_float(value)


def load_stats_trace(path: Path) -> List[Dict[str, float]]:
    trace: List[Dict[str, float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            kimg = _safe_mean_field(payload, "Progress/kimg")
            total_sec = _safe_mean_field(payload, "Timing/total_sec")
            if kimg is None or total_sec is None:
                continue
            trace.append(
                {
                    "kimg": float(kimg),
                    "total_sec": float(total_sec),
                    "loss_loss": _safe_mean_field(payload, "Loss/loss"),
                    "cdro_outer_loss": _safe_mean_field(payload, "CDRO/outer_loss"),
                    "cdro_outer_loss_attack": _safe_mean_field(payload, "CDRO/outer_loss_attack"),
                    "cdro_outer_loss_clean": _safe_mean_field(payload, "CDRO/outer_loss_clean"),
                    "cdro_delta_ratio_mean": _safe_mean_field(payload, "CDRO/delta_ratio_mean"),
                }
            )
    if not trace:
        raise RuntimeError(f"No stats rows found in {path}")
    trace.sort(key=lambda item: item["kimg"])
    return trace


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


def match_trace_row_for_snapshot(trace: List[Dict[str, float]], snapshot_kimg: int) -> Dict[str, float]:
    exact = [row for row in trace if int(math.floor(float(row["kimg"]))) == int(snapshot_kimg)]
    if exact:
        return exact[-1]
    nearest = min(trace, key=lambda row: abs(float(row["kimg"]) - float(snapshot_kimg)))
    if abs(float(nearest["kimg"]) - float(snapshot_kimg)) > 2.0:
        raise RuntimeError(
            f"Could not align snapshot kimg={snapshot_kimg} with stats trace; nearest row is {nearest['kimg']:.3f}."
        )
    return nearest


def _series_fields() -> Dict[str, str]:
    return {
        "method": "cdro",
        "robust_method": "cdro",
        "robust_label": "CDRO",
        "training_objective": "edm",
        "backbone_family": "edm",
        "backbone_label": "EDM",
        "series_key": "cdro_edm",
        "series_label": "CDRO EDM",
        "method_version_used": "cdroedm",
    }


def build_rows(args: argparse.Namespace):
    cdro_run_dir = Path(args.cdro_run_dir).resolve()
    budget_plan_path = cdro_run_dir / "cdro_budget_plan.json"
    if not budget_plan_path.is_file():
        raise FileNotFoundError(f"Missing launch metadata: {budget_plan_path}")
    budget_plan = json.loads(budget_plan_path.read_text(encoding="utf-8"))
    cdro_config = dict(budget_plan.get("cdro_config", {}))
    calibration = calibration_from_path(args.calibration_json)
    warmup_summary = load_warmup_summary(args.summary_json)
    trace = load_stats_trace(cdro_run_dir / "stats.jsonl")
    requested_kimg = parse_kimg_list(args.kimg)
    snapshot_kimg = requested_kimg if requested_kimg else discover_snapshot_kimg(cdro_run_dir)
    warmup_kimg = int(round(float(warmup_summary["warmup_kimg"])))
    snapshot_kimg = [value for value in snapshot_kimg if int(value) > warmup_kimg]
    if not snapshot_kimg:
        raise RuntimeError(f"No robust-phase snapshots found in {cdro_run_dir}")

    robust_step_wcu = cdro_robust_step_weighted_compute_units(
        calibration=calibration,
        n_steps_path=int(cdro_config["cdro_n_steps_path"]),
        attack_num_steps=int(cdro_config["attack_num_steps"]),
        outer_attack_weight=float(cdro_config["outer_attack_weight"]),
        outer_clean_weight=float(cdro_config["outer_clean_weight"]),
    )
    robust_step_compute_be = cdro_robust_step_compute_be(
        n_steps_path=int(cdro_config["cdro_n_steps_path"]),
        attack_num_steps=int(cdro_config["attack_num_steps"]),
        outer_attack_weight=float(cdro_config["outer_attack_weight"]),
        outer_clean_weight=float(cdro_config["outer_clean_weight"]),
    )

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
        trace_row = match_trace_row_for_snapshot(trace, int(kimg))
        actual_progress_kimg = float(trace_row["kimg"])
        robust_steps = max(int(round((actual_progress_kimg - float(warmup_kimg)) * 1000.0 / float(args.batch_size))), 0)
        robust_wcu = float(robust_steps) * float(robust_step_wcu)
        robust_compute_be = float(robust_steps) * float(robust_step_compute_be)
        total_sec = float(warmup_sec + trace_row["total_sec"])
        eval_tag = f"cdro_kimg{int(kimg):06d}"
        loss_final = trace_row["cdro_outer_loss"]
        if loss_final is None:
            loss_final = trace_row["loss_loss"]
        rows.append(
            {
                **_series_fields(),
                "seed": int(args.seed),
                "step": int(kimg),
                "snapshot_kimg": int(kimg),
                "images_shown_m": float(actual_progress_kimg) / 1000.0,
                "row_origin": "trajectory_robust_phase",
                "run_dir": str(cdro_run_dir),
                "network_pkl": str(snapshot),
                "metrics_path": str(cdro_run_dir / "stats.jsonl"),
                "train_wall_clock_sec": float(total_sec),
                "train_elapsed_sec": float(total_sec),
                "train_gpu_hours": float(total_sec) / 3600.0,
                "train_wall_clock_source": "normalized_baseline_warmup_plus_observed_cdro_stats_jsonl",
                "baseline_train_wall_clock_sec_effective": float(warmup_sec),
                "robust_train_wall_clock_sec_effective": float(trace_row["total_sec"]),
                "baseline_train_wall_clock_source": "normalized_baseline_stats_jsonl",
                "weighted_compute_units": float(warmup_wcu + robust_wcu),
                "baseline_weighted_compute_units": float(warmup_wcu),
                "robust_weighted_compute_units": float(robust_wcu),
                "compute_budget_be": float(warmup_compute_be + robust_compute_be),
                "baseline_compute_be": float(warmup_compute_be),
                "robust_compute_be": float(robust_compute_be),
                "weighted_compute_source": "cifar_calibration_cdro_path_primitive_counts",
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
                "loss_final": "" if loss_final is None else float(loss_final),
                "loss_mean_last": "" if trace_row["loss_loss"] is None else float(trace_row["loss_loss"]),
                "fixed_warmup_steps": float(warmup_compute_be),
            }
        )

    summary = {
        "manifest_version": "cifar_cdro_fid_manifest_v2",
        "cdro_run_dir": str(cdro_run_dir),
        "calibration_json": str(Path(args.calibration_json).resolve()),
        "summary_json": str(Path(args.summary_json).resolve()),
        "warmup_summary": warmup_summary,
        "cdro_config": cdro_config,
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
