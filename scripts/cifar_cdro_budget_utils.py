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
    "cifar10_baseline_vs_wdro_coarse_20260414_three_method_compare.csv"
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


def cdro_robust_step_weighted_compute_units(
    *,
    calibration: Dict,
    n_steps_path: int,
    attack_num_steps: int,
    outer_attack_weight: float,
    outer_clean_weight: float,
) -> float:
    path_steps = max(int(n_steps_path), 0)
    attack_enabled = bool(float(outer_attack_weight) > 0.0 and int(attack_num_steps) > 0)
    clean_enabled = bool(float(outer_clean_weight) > 0.0)
    active_outer_branches = int(float(outer_attack_weight) > 0.0) + int(clean_enabled)
    # The image-port CDRO rollout constructs the nominal noisy state analytically and only
    # spends denoiser work on the input-gradient attack construction plus the outer backward pass.
    value = weighted_compute_units(
        n_fwd=0.0,
        n_fwd_inputgrad=float(path_steps * max(int(attack_num_steps), 0)) if attack_enabled else 0.0,
        n_fwd_parambackward=float(path_steps * active_outer_branches),
        calibration=calibration,
    )
    if value is None:
        raise RuntimeError("Weighted-compute calibration is unavailable.")
    return float(value)


def cdro_robust_step_compute_be(
    *,
    n_steps_path: int,
    attack_num_steps: int,
    outer_attack_weight: float,
    outer_clean_weight: float,
) -> float:
    path_steps = max(int(n_steps_path), 0)
    attack_enabled = bool(float(outer_attack_weight) > 0.0 and int(attack_num_steps) > 0)
    clean_enabled = bool(float(outer_clean_weight) > 0.0)
    active_outer_branches = int(float(outer_attack_weight) > 0.0) + int(clean_enabled)
    return float(path_steps * ((max(int(attack_num_steps), 0) if attack_enabled else 0) + active_outer_branches))


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
    target_wcu_value = max(float(target_wcu), float(warmup_wcu))
    required_robust_steps = max((target_wcu_value - float(warmup_wcu)) / float(robust_step_wcu), 0.0)
    floor_steps = max(int(math.floor(required_robust_steps + 1e-12)), 0)
    ceil_steps = max(int(math.ceil(required_robust_steps - 1e-12)), 0)
    candidates = sorted(set([floor_steps, ceil_steps]))
    robust_steps_int = min(
        candidates,
        key=lambda steps: (
            abs(float(warmup_wcu) + float(steps) * float(robust_step_wcu) - float(target_wcu_value)),
            steps,
        ),
    )
    warmup_nimg = int(round(float(warmup_kimg) * 1000.0))
    actual_total_nimg = warmup_nimg + int(robust_steps_int) * int(batch_size)
    actual_total_kimg_float = float(actual_total_nimg) / 1000.0
    if robust_steps_int <= 0:
        total_kimg_int = int(round(float(warmup_kimg)))
    else:
        prev_nimg = warmup_nimg + int(robust_steps_int - 1) * int(batch_size)
        lower_total_kimg_int = int(prev_nimg // 1000) + 1
        upper_total_kimg_int = int(actual_total_nimg // 1000)
        if lower_total_kimg_int > upper_total_kimg_int:
            raise RuntimeError(
                "Could not map the desired CDRO robust-step count to an exact train.py total_kimg stop. "
                f"prev_nimg={prev_nimg} actual_total_nimg={actual_total_nimg}"
            )
        total_kimg_int = int(upper_total_kimg_int)
    actual_robust_kimg = max(float(actual_total_nimg - warmup_nimg) / 1000.0, 0.0)
    actual_robust_wcu = float(robust_steps_int) * float(robust_step_wcu)
    actual_robust_compute_be = float(robust_steps_int) * float(robust_step_compute_be)
    return {
        "required_robust_steps_float": float(required_robust_steps),
        "robust_steps_int": int(robust_steps_int),
        "warmup_nimg": int(warmup_nimg),
        "actual_total_nimg": int(actual_total_nimg),
        "actual_total_kimg_float": float(actual_total_kimg_float),
        "total_kimg_int": int(total_kimg_int),
        "actual_robust_kimg": float(actual_robust_kimg),
        "actual_robust_wcu": float(actual_robust_wcu),
        "actual_total_wcu": float(warmup_wcu + actual_robust_wcu),
        "actual_robust_compute_be": float(actual_robust_compute_be),
        "actual_total_compute_be": float(warmup_compute_be + actual_robust_compute_be),
    }


def estimate_total_wall_clock_sec(
    *,
    warmup_wall_clock_sec: float,
    robust_steps_int: int,
    robust_step_wall_clock_sec: float,
) -> float:
    return float(warmup_wall_clock_sec) + float(int(robust_steps_int)) * float(robust_step_wall_clock_sec)


def calibration_from_path(calibration_json: str) -> Dict:
    calibration = load_weighted_compute_calibration(calibration_path=calibration_json)
    if not calibration.get("available", False):
        raise RuntimeError(f"Weighted-compute calibration is unavailable: {calibration_json}")
    return calibration
