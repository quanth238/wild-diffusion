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
    DEFAULT_FLOP_CALIBRATION_JSON,
    DEFAULT_PYTORCH_FID_REF,
    DEFAULT_WDRO_SUMMARY_JSON,
    baseline_step_flops,
    calibration_from_path,
    cdro_robust_step_flops,
    cdro_robust_step_compute_be,
    cdro_robust_step_weighted_compute_units,
    flop_calibration_from_path,
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
    parser.add_argument("--flop-calibration-json", type=str, default=DEFAULT_FLOP_CALIBRATION_JSON)
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


def _safe_mean_field_any(payload: Dict, *keys: str):
    for key in keys:
        value = _safe_mean_field(payload, key)
        if value is not None:
            return value
    return None


def _blank_if_none(value: object):
    return "" if value is None else value


def _interpolate_trace_field(trace: List[Dict[str, float]], *, target_kimg: float, field: str) -> float:
    if not trace:
        raise RuntimeError(f"Cannot interpolate {field!r} without any trace rows.")
    rows = sorted(trace, key=lambda item: item["kimg"])
    exact = [row for row in rows if abs(float(row["kimg"]) - float(target_kimg)) <= 1e-9]
    if exact:
        return float(exact[-1][field])
    if float(target_kimg) <= float(rows[0]["kimg"]):
        return float(rows[0][field])
    if float(target_kimg) >= float(rows[-1]["kimg"]):
        return float(rows[-1][field])
    for idx in range(1, len(rows)):
        prev_row = rows[idx - 1]
        row = rows[idx]
        x0 = float(prev_row["kimg"])
        x1 = float(row["kimg"])
        if x0 <= float(target_kimg) <= x1:
            if x1 <= x0:
                return float(row[field])
            frac = (float(target_kimg) - x0) / (x1 - x0)
            y0 = float(prev_row[field])
            y1 = float(row[field])
            return float(y0 + frac * (y1 - y0))
    return float(rows[-1][field])


def _stitch_effective_total_sec(raw_trace: List[Dict[str, float]]) -> List[Dict[str, float]]:
    if not raw_trace:
        return []
    segments: List[List[Dict[str, float]]] = []
    current_segment: List[Dict[str, float]] = []
    prev_kimg = None
    prev_total_sec = None
    for row in raw_trace:
        row_kimg = float(row["kimg"])
        row_total_sec = float(row["total_sec"])
        if current_segment and (
            (prev_kimg is not None and row_kimg + 1e-9 < float(prev_kimg))
            or (prev_total_sec is not None and row_total_sec + 1e-9 < float(prev_total_sec))
        ):
            segments.append(current_segment)
            current_segment = []
        current_segment.append(dict(row))
        prev_kimg = row_kimg
        prev_total_sec = row_total_sec
    if current_segment:
        segments.append(current_segment)

    stitched: List[Dict[str, float]] = []
    for segment in segments:
        start_kimg = float(segment[0]["kimg"])
        base_effective_sec = 0.0
        if stitched:
            base_effective_sec = _interpolate_trace_field(
                stitched,
                target_kimg=start_kimg,
                field="effective_total_sec",
            )
        for row in segment:
            row = dict(row)
            row["effective_total_sec"] = float(base_effective_sec + float(row["total_sec"]))
            stitched.append(row)
    stitched.sort(key=lambda item: item["kimg"])
    return stitched


def load_stats_trace(path: Path) -> List[Dict[str, float]]:
    raw_trace: List[Dict[str, float]] = []
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
            raw_trace.append(
                {
                    "kimg": float(kimg),
                    "total_sec": float(total_sec),
                    "loss": _safe_mean_field_any(payload, "Loss", "Loss/loss"),
                    "cdro_edm_clean_probe": _safe_mean_field(payload, "CDRO/edm_clean_probe"),
                    "cdro_outer_loss": _safe_mean_field(payload, "CDRO/outer_loss"),
                    "cdro_outer_loss_attack": _safe_mean_field(payload, "CDRO/outer_loss_attack"),
                    "cdro_outer_loss_clean": _safe_mean_field(payload, "CDRO/outer_loss_clean"),
                    "cdro_delta_ratio_mean": _safe_mean_field(payload, "CDRO/delta_ratio_mean"),
                }
            )
    if not raw_trace:
        raise RuntimeError(f"No stats rows found in {path}")
    return _stitch_effective_total_sec(raw_trace)


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
    flop_calibration = flop_calibration_from_path(args.flop_calibration_json)
    warmup_summary = load_warmup_summary(args.summary_json)
    trace = load_stats_trace(cdro_run_dir / "stats.jsonl")
    requested_kimg = parse_kimg_list(args.kimg)
    snapshot_kimg = requested_kimg if requested_kimg else discover_snapshot_kimg(cdro_run_dir)
    warmup_kimg = int(round(float(warmup_summary["warmup_kimg"])))
    if requested_kimg:
        snapshot_kimg = [
            value for value in snapshot_kimg if int(value) > warmup_kimg or int(value) == warmup_kimg
        ]
    else:
        snapshot_kimg = [value for value in snapshot_kimg if int(value) > warmup_kimg]
    if not snapshot_kimg:
        raise RuntimeError(f"No robust-phase snapshots found in {cdro_run_dir}")

    robust_step_wcu = cdro_robust_step_weighted_compute_units(
        calibration=calibration,
        n_steps_path=int(cdro_config["cdro_n_steps_path"]),
        attack_num_steps=int(cdro_config["attack_num_steps"]),
        outer_attack_weight=float(cdro_config["outer_attack_weight"]),
        outer_clean_weight=float(cdro_config["outer_clean_weight"]),
        total_budget_rho=float(cdro_config["cdro_total_budget_rho"]),
    )
    robust_step_compute_be = cdro_robust_step_compute_be(
        n_steps_path=int(cdro_config["cdro_n_steps_path"]),
        attack_num_steps=int(cdro_config["attack_num_steps"]),
        outer_attack_weight=float(cdro_config["outer_attack_weight"]),
        outer_clean_weight=float(cdro_config["outer_clean_weight"]),
        total_budget_rho=float(cdro_config["cdro_total_budget_rho"]),
    )
    robust_step_flops = cdro_robust_step_flops(
        calibration=flop_calibration,
        n_steps_path=int(cdro_config["cdro_n_steps_path"]),
        attack_num_steps=int(cdro_config["attack_num_steps"]),
        outer_attack_weight=float(cdro_config["outer_attack_weight"]),
        outer_clean_weight=float(cdro_config["outer_clean_weight"]),
        total_budget_rho=float(cdro_config["cdro_total_budget_rho"]),
    )

    warmup_wcu = float(warmup_summary["warmup_weighted_compute_units"])
    warmup_sec = float(warmup_summary["warmup_train_wall_clock_sec"])
    warmup_compute_be = float(warmup_summary["warmup_compute_be"])
    baseline_step_train_flops = baseline_step_flops(calibration=flop_calibration)
    warmup_train_flops = (
        None if baseline_step_train_flops is None else float(warmup_compute_be) * float(baseline_step_train_flops)
    )

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
        if int(kimg) == warmup_kimg:
            trace_row = {
                "kimg": float(warmup_kimg),
                "total_sec": 0.0,
                "effective_total_sec": 0.0,
                "loss": None,
                "cdro_edm_clean_probe": None,
                "cdro_outer_loss": None,
                "cdro_outer_loss_attack": None,
                "cdro_outer_loss_clean": None,
                "cdro_delta_ratio_mean": None,
            }
            actual_progress_kimg = float(warmup_kimg)
            robust_steps = 0
            robust_wcu = 0.0
            robust_compute_be = 0.0
            total_sec = float(warmup_sec)
            row_origin = "baseline_warmup_boundary"
        else:
            trace_row = match_trace_row_for_snapshot(trace, int(kimg))
            actual_progress_kimg = float(trace_row["kimg"])
            robust_steps = max(
                int(round((actual_progress_kimg - float(warmup_kimg)) * 1000.0 / float(args.batch_size))),
                0,
            )
            robust_wcu = float(robust_steps) * float(robust_step_wcu)
            robust_compute_be = float(robust_steps) * float(robust_step_compute_be)
            total_sec = float(warmup_sec + trace_row["effective_total_sec"])
            row_origin = "trajectory_robust_phase"
        robust_train_flops = (
            None if robust_step_flops is None else float(robust_steps) * float(robust_step_flops)
        )
        total_train_flops = (
            None if warmup_train_flops is None or robust_train_flops is None else float(warmup_train_flops + robust_train_flops)
        )
        eval_tag = f"cdro_kimg{int(kimg):06d}"
        loss_final = trace_row["cdro_outer_loss"]
        if loss_final is None:
            loss_final = trace_row["loss"]
        rows.append(
            {
                **_series_fields(),
                "seed": int(args.seed),
                "step": int(kimg),
                "snapshot_kimg": int(kimg),
                "images_shown_m": float(actual_progress_kimg) / 1000.0,
                "row_origin": row_origin,
                "run_dir": str(cdro_run_dir),
                "network_pkl": str(snapshot),
                "metrics_path": str(cdro_run_dir / "stats.jsonl"),
                "train_wall_clock_sec": float(total_sec),
                "train_elapsed_sec": float(total_sec),
                "train_gpu_hours": float(total_sec) / 3600.0,
                "train_wall_clock_source": "normalized_baseline_warmup_plus_stitched_cdro_stats_jsonl",
                "baseline_train_wall_clock_sec_effective": float(warmup_sec),
                "robust_train_wall_clock_sec_effective": float(trace_row["effective_total_sec"]),
                "baseline_train_wall_clock_source": "normalized_baseline_stats_jsonl",
                "weighted_compute_units": float(warmup_wcu + robust_wcu),
                "baseline_weighted_compute_units": float(warmup_wcu),
                "robust_weighted_compute_units": float(robust_wcu),
                "compute_budget_be": float(warmup_compute_be + robust_compute_be),
                "baseline_compute_be": float(warmup_compute_be),
                "robust_compute_be": float(robust_compute_be),
                "total_train_flops": _blank_if_none(None if total_train_flops is None else float(total_train_flops)),
                "total_train_tflops": _blank_if_none(
                    None if total_train_flops is None else float(total_train_flops) / 1e12
                ),
                "total_train_pflops": _blank_if_none(
                    None if total_train_flops is None else float(total_train_flops) / 1e15
                ),
                "baseline_train_flops": _blank_if_none(
                    None if warmup_train_flops is None else float(warmup_train_flops)
                ),
                "robust_train_flops": _blank_if_none(
                    None if robust_train_flops is None else float(robust_train_flops)
                ),
                "train_flop_source": (
                    "flop_calibration_cdro_path_primitive_counts"
                    if total_train_flops is not None
                    else ""
                ),
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
                "loss_mean_last": "" if trace_row["loss"] is None else float(trace_row["loss"]),
                "loss_comparable_final": (
                    ""
                    if trace_row["cdro_edm_clean_probe"] is None
                    else float(trace_row["cdro_edm_clean_probe"])
                ),
                "loss_comparable_key": (
                    ""
                    if trace_row["cdro_edm_clean_probe"] is None
                    else "CDRO/edm_clean_probe"
                ),
                "cdro_edm_clean_probe": (
                    ""
                    if trace_row["cdro_edm_clean_probe"] is None
                    else float(trace_row["cdro_edm_clean_probe"])
                ),
                "cdro_outer_loss": "" if trace_row["cdro_outer_loss"] is None else float(trace_row["cdro_outer_loss"]),
                "cdro_outer_loss_attack": (
                    ""
                    if trace_row["cdro_outer_loss_attack"] is None
                    else float(trace_row["cdro_outer_loss_attack"])
                ),
                "cdro_outer_loss_clean": (
                    ""
                    if trace_row["cdro_outer_loss_clean"] is None
                    else float(trace_row["cdro_outer_loss_clean"])
                ),
                "fixed_warmup_steps": float(warmup_compute_be),
            }
        )

    summary = {
        "manifest_version": "cifar_cdro_fid_manifest_v2",
        "cdro_run_dir": str(cdro_run_dir),
        "calibration_json": str(Path(args.calibration_json).resolve()),
        "flop_calibration_json": (
            str(Path(args.flop_calibration_json).resolve()) if str(args.flop_calibration_json).strip() else None
        ),
        "summary_json": str(Path(args.summary_json).resolve()),
        "warmup_summary": warmup_summary,
        "cdro_config": cdro_config,
        "robust_step_weighted_compute_units": float(robust_step_wcu),
        "robust_step_compute_be": float(robust_step_compute_be),
        "warmup_train_flops": None if warmup_train_flops is None else float(warmup_train_flops),
        "baseline_step_train_flops": (
            None if baseline_step_train_flops is None else float(baseline_step_train_flops)
        ),
        "robust_step_train_flops": None if robust_step_flops is None else float(robust_step_flops),
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
