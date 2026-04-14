#!/usr/bin/env python3
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from toy.compute_accounting import load_weighted_compute_calibration, weighted_compute_units  # noqa: E402


DEFAULT_BASELINE_RUN_DIR = (
    "/home/bachlc/GM-CDRO/training-runs/paper-cifar10-baseline-fp16/"
    "00001-cifar10-32x32-train20pct-seed0-uncond-ddpmpp-wdroedm-gpus1-batch1024-fp16-"
    "paper-cifar10-uncond-ddpmpp-baseline-20pct-fp16-bg512"
)
DEFAULT_BASELINE_RESUME_KIMG = 40000
DEFAULT_WDRO_COMPARE_CSV = (
    "/home/bachlc/GM-CDRO/training-runs/fid-sweeps/cifar10_baseline_vs_wdro_coarse_20260414/"
    "cifar10_baseline_vs_wdro_coarse_three_method_compare.csv"
)
DEFAULT_WDRO_SUMMARY_JSON = (
    "/home/bachlc/GM-CDRO/training-runs/fid-sweeps/cifar10_baseline_vs_wdro_coarse_20260414/"
    "cifar10_baseline_vs_wdro_coarse_manifest_summary.json"
)
DEFAULT_CALIBRATION_JSON = (
    "/home/bachlc/GM-CDRO/training-runs/compute_calibration/"
    "cifar10_32x32_ddpmpp_wdroedm_fp16_b1024_h100_20260414.json"
)
DEFAULT_PYTORCH_FID_REF = (
    "/home/bachlc/GM-CDRO/training-runs/fid-sweeps/cifar10_baseline_vs_wdro_coarse_20260414/"
    "pytorch_fid_cifar10_train_ref_stats.npz"
)


def _safe_float(value) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return float(parsed)


def load_compare_rows(compare_csv: str) -> List[Dict[str, str]]:
    path = Path(compare_csv).resolve()
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def best_row_for_method(rows: List[Dict[str, str]], method: str) -> Dict[str, str]:
    method_rows = [row for row in rows if str(row.get("method", "")).strip() == str(method)]
    scored_rows = [row for row in method_rows if _safe_float(row.get("fid")) is not None]
    if not scored_rows:
        raise RuntimeError(f"No scored rows found for method={method}")
    return min(scored_rows, key=lambda row: float(row["fid"]))


def load_warmup_summary(summary_json: str) -> Dict[str, float]:
    payload = json.loads(Path(summary_json).resolve().read_text(encoding="utf-8"))
    return {
        "warmup_kimg": float(payload["warmup_boundary_kimg"]),
        "warmup_train_wall_clock_sec": float(payload["warmup_boundary_train_wall_clock_sec"]),
        "warmup_weighted_compute_units": float(payload["warmup_boundary_weighted_compute_units"]),
        "warmup_compute_be": float(payload["warmup_boundary_compute_be"]),
    }


def resolve_budget_target(compare_csv: str, target_mode: str) -> Dict[str, float]:
    rows = load_compare_rows(compare_csv)
    baseline_best = best_row_for_method(rows, "baseline")
    wdro_best = best_row_for_method(rows, "wild_diffusion")
    wdro_rows = [row for row in rows if str(row.get("method", "")).strip() == "wild_diffusion"]
    wdro_final = max(
        [row for row in wdro_rows if _safe_float(row.get("weighted_compute_units")) is not None],
        key=lambda row: float(row["step"]),
    )
    baseline_best_wcu = float(baseline_best["weighted_compute_units"])
    wdro_best_wcu = float(wdro_best["weighted_compute_units"])
    if target_mode == "wdro_best":
        target_wcu = wdro_best_wcu
    elif target_mode == "wdro_final":
        target_wcu = float(wdro_final["weighted_compute_units"])
    elif target_mode == "midpoint":
        target_wcu = 0.5 * (baseline_best_wcu + wdro_best_wcu)
    elif target_mode == "baseline_best":
        target_wcu = baseline_best_wcu
    else:
        raise ValueError(f"Unsupported target mode: {target_mode}")
    return {
        "target_wcu": float(target_wcu),
        "baseline_best_wcu": float(baseline_best_wcu),
        "wdro_best_wcu": float(wdro_best_wcu),
        "wdro_final_wcu": float(wdro_final["weighted_compute_units"]),
        "baseline_best_fid": float(baseline_best["fid"]),
        "wdro_best_fid": float(wdro_best["fid"]),
        "wdro_final_fid": float(wdro_final["fid"]),
        "baseline_best_step": float(baseline_best["step"]),
        "wdro_best_step": float(wdro_best["step"]),
        "wdro_final_step": float(wdro_final["step"]),
        "baseline_best_wall_clock_sec": float(baseline_best["train_wall_clock_sec"]),
        "wdro_best_wall_clock_sec": float(wdro_best["train_wall_clock_sec"]),
        "wdro_final_wall_clock_sec": float(wdro_final["train_wall_clock_sec"]),
    }


def advedm_robust_step_weighted_compute_units(*, calibration: Dict, adv_steps: int) -> float:
    adv_steps_value = max(int(adv_steps), 0)
    value = weighted_compute_units(
        n_fwd=0.0,
        n_fwd_inputgrad=float(adv_steps_value),
        n_fwd_parambackward=2.0,
        calibration=calibration,
    )
    if value is None:
        raise RuntimeError("Weighted-compute calibration is unavailable.")
    return float(value)


def advedm_robust_step_compute_be(*, adv_steps: int) -> float:
    return float(max(int(adv_steps), 0) + 2.0)


def build_budget_plan(
    *,
    target_wcu: float,
    warmup_kimg: float,
    warmup_wcu: float,
    warmup_compute_be: float,
    batch_size: int,
    robust_step_wcu: float,
    robust_step_compute_be: float,
) -> Dict[str, float]:
    robust_steps = max((float(target_wcu) - float(warmup_wcu)) / float(robust_step_wcu), 0.0)
    robust_kimg = float(robust_steps) * float(batch_size) / 1000.0
    robust_compute_be = float(robust_steps) * float(robust_step_compute_be)
    total_compute_be = float(warmup_compute_be) + float(robust_compute_be)
    total_kimg = float(warmup_kimg) + float(robust_kimg)
    return {
        "robust_steps_float": float(robust_steps),
        "robust_kimg_float": float(robust_kimg),
        "total_kimg_float": float(total_kimg),
        "robust_compute_be": float(robust_compute_be),
        "total_compute_be": float(total_compute_be),
    }


def calibration_from_path(calibration_json: str) -> Dict:
    calibration = load_weighted_compute_calibration(calibration_path=calibration_json)
    if not calibration.get("available", False):
        raise RuntimeError(f"Weighted-compute calibration is unavailable: {calibration_json}")
    return calibration
