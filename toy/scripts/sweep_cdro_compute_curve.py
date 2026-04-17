#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from toy.compute_accounting import (
    baseline_weighted_compute_units_for_steps,
    cdro_adjusted_weighted_compute_from_compute_accounting,
    cdro_robust_step_weighted_compute_units,
    load_weighted_compute_calibration,
    solve_warmup_steps_for_target_compute_fraction,
    wdro_robust_step_weighted_compute_units,
    weighted_compute_units,
)
from toy.config import ToyConfig


DEFAULT_TRAIN_ROOT = os.path.join(ROOT_DIR, "toy_data", "simpsons_mnist_rgb", "imagefolder", "train")
DEFAULT_VAL_ROOT = os.path.join(ROOT_DIR, "toy_data", "simpsons_mnist_rgb", "imagefolder", "test")
DEFAULT_FID_REF = os.path.join(
    ROOT_DIR,
    "toy_data",
    "simpsons_mnist_rgb",
    "fid_refs",
    "simpsons_mnist_rgb_test_28x28.npz",
)
DEFAULT_BASELINE_AGGREGATE = os.path.join(
    ROOT_DIR,
    "toy_outputs",
    "simpsons_mnist_rgb_convergence_ckpt_1pct_from0",
    "simpsons_mnist_rgb_baseline_curve_ckpt_1pct_from0_aggregate.csv",
)
DEFAULT_BASELINE_CHECKPOINT_DIR = os.path.join(
    ROOT_DIR,
    "toy_outputs",
    "simpsons_mnist_rgb_convergence_ckpt_1pct_from0",
    "simpsons_mnist_rgb_baseline_curve_ckpt_1pct_from0_1pct_s0",
    "checkpoints",
)
DEFAULT_STEPS_LIST = "0,10,25,50,100,250,500,1000,2500,5000,10000,15000,20000,25000,27500,30000,32500,35000,37500,40000,45000,50000,60000,80000"
DEFAULT_FID_DETECTOR = (
    "/root/.cache/dnnlib/downloads/"
    "18d9c1159d16cd4cc6adf7db0f2dd2a9_https___api.ngc.nvidia.com_v2_models_nvidia_research_"
    "stylegan3_versions_1_files_metrics_inception-2015-12-05.pkl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a CDRO compute curve on Simpsons-MNIST RGB and compare against a baseline EDM curve."
    )
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--prefix", type=str, required=True)
    parser.add_argument("--python-bin", type=str, default=os.path.join(ROOT_DIR, ".venv", "bin", "python"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dataset-path", type=str, default=DEFAULT_TRAIN_ROOT)
    parser.add_argument("--dataset-val-path", type=str, default=DEFAULT_VAL_ROOT)
    parser.add_argument("--fid-ref-path", type=str, default=DEFAULT_FID_REF)
    parser.add_argument("--baseline-aggregate", action="append", default=[])
    parser.add_argument("--skip-baseline-overlay", action="store_true")
    parser.add_argument("--baseline-checkpoint-dir", type=str, default=DEFAULT_BASELINE_CHECKPOINT_DIR)
    parser.add_argument("--disable-baseline-ckpt", action="store_true")
    parser.add_argument("--image-size", type=int, default=28)
    parser.add_argument("--image-channels", type=int, default=3)
    parser.add_argument("--image-train-size", type=int, default=80)
    parser.add_argument("--image-val-size", type=int, default=2000)
    parser.add_argument("--train-percent-label", type=str, default="1%")
    parser.add_argument("--image-split-seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--eval-samples", type=int, default=2000)
    parser.add_argument("--debug-eval-batch", type=int, default=64)
    parser.add_argument("--debug-terminal-step", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--n-steps-path", type=int, default=64)
    parser.add_argument("--sigma-min", type=float, default=0.002)
    parser.add_argument("--sigma-max", type=float, default=2.0)
    parser.add_argument("--steps-list", type=str, default=DEFAULT_STEPS_LIST)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--inner-steps", type=int, default=1)
    parser.add_argument("--outer-attack-weight", type=float, default=1.0)
    parser.add_argument("--outer-clean-weight", type=float, default=0.0)
    parser.add_argument("--cdro-step-size", type=float, default=0.02)
    parser.add_argument("--cdro-total-budget-rho", type=float, default=4.0)
    parser.add_argument("--cdro-time-horizon", type=float, default=1.0)
    parser.add_argument(
        "--cdro-edm-ladder-mode",
        type=str,
        default="stochastic_stratified_quantile",
        choices=["deterministic_midpoint_quantile", "stochastic_stratified_quantile"],
    )
    parser.add_argument("--cdro-warmup-fraction", type=float, default=0.05)
    parser.add_argument("--match-reference-csv", type=str, default="")
    parser.add_argument("--match-reference-method", type=str, default="wdro")
    parser.add_argument(
        "--match-reference-metric",
        type=str,
        default="weighted_compute_units",
        choices=["weighted_compute_units"],
    )
    parser.add_argument("--match-reference-steps-list", type=str, default="")
    parser.add_argument(
        "--reference-match-pilot-robust-steps",
        type=int,
        default=0,
        help=(
            "If > 0, run a short CDRO pilot from each shared warmup checkpoint and use the observed "
            "robust weighted-compute-per-step to refine the matched total steps before the main run."
        ),
    )
    parser.add_argument("--weighted-compute-calibration-path", type=str, default="")
    parser.add_argument("--weighted-inputgrad-alpha", type=float, default=0.0)
    parser.add_argument("--weighted-parambackward-beta", type=float, default=0.0)
    parser.add_argument("--train-accelerator-count", type=int, default=1)
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_steps_list(value: str) -> List[int]:
    steps = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    return [step for step in steps if step >= 0]


def parse_optional_steps_list(value: str) -> Optional[set[int]]:
    if not str(value).strip():
        return None
    return set(parse_steps_list(value))


def safe_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def maybe_set_fid_detector_env(env: Dict[str, str]) -> Dict[str, str]:
    env = dict(env)
    if "FID_DETECTOR_PATH" not in env and os.path.isfile(DEFAULT_FID_DETECTOR):
        env["FID_DETECTOR_PATH"] = DEFAULT_FID_DETECTOR
    return env


def find_baseline_checkpoint(checkpoint_dir: str, step: int) -> Optional[str]:
    if not checkpoint_dir:
        return None
    path = os.path.join(checkpoint_dir, f"baseline_step{int(step):05d}.pt")
    return path if os.path.isfile(path) else None


def resolve_requested_baseline_checkpoint(args: argparse.Namespace, warmup_steps: int) -> Optional[str]:
    """Return a baseline checkpoint only when cache reuse is enabled for this sweep."""

    if bool(args.disable_baseline_ckpt):
        return None
    return find_baseline_checkpoint(args.baseline_checkpoint_dir, warmup_steps)


def resolve_weighted_calibration(args: argparse.Namespace) -> Dict:
    return load_weighted_compute_calibration(
        calibration_path=str(args.weighted_compute_calibration_path).strip(),
        inputgrad_alpha=float(args.weighted_inputgrad_alpha),
        parambackward_beta=float(args.weighted_parambackward_beta),
    )


def estimate_wdro_reference_warmup_weighted_compute_fraction(
    *,
    total_steps: int,
    warmup_fraction: float,
    batch_size: int,
    train_pool_size: Optional[int],
    calibration: Dict[str, Any],
) -> Optional[Dict[str, float]]:
    total_steps_value = max(int(total_steps), 0)
    if total_steps_value <= 0:
        return {
            "warmup_steps": 0.0,
            "robust_steps": 0.0,
            "warmup_weighted_compute_fraction": 0.0,
            "wdro_robust_step_weighted_compute_units": 0.0,
        }

    warmup_steps = max(0, min(int(total_steps_value * float(warmup_fraction)), total_steps_value))
    robust_steps = max(total_steps_value - warmup_steps, 0)
    baseline_weighted_units = baseline_weighted_compute_units_for_steps(
        steps=int(warmup_steps),
        calibration=calibration,
    )
    wdro_robust_step_weighted_units = wdro_robust_step_weighted_compute_units(
        batch_size=int(batch_size),
        train_pool_size=train_pool_size,
        refresh_epochs=float(ToyConfig.wdro_refresh_epochs),
        adv_prob=float(ToyConfig.wdro_adv_prob),
        attack_steps=int(ToyConfig.wdro_attack_steps),
        calibration=calibration,
    )
    if baseline_weighted_units is None or wdro_robust_step_weighted_units is None:
        return None
    total_weighted_units = float(baseline_weighted_units) + float(robust_steps) * float(wdro_robust_step_weighted_units)
    warmup_weighted_fraction = 0.0 if total_weighted_units <= 0.0 else float(baseline_weighted_units) / float(total_weighted_units)
    return {
        "warmup_steps": float(warmup_steps),
        "robust_steps": float(robust_steps),
        "warmup_weighted_compute_fraction": float(warmup_weighted_fraction),
        "wdro_robust_step_weighted_compute_units": float(wdro_robust_step_weighted_units),
    }


def load_reference_rows_for_matching(
    *,
    csv_path: str,
    reference_method: str,
    metric_name: str,
    reference_steps_filter: Optional[set[int]],
) -> List[Dict[str, Any]]:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Reference curve CSV not found: {csv_path}")
    rows: List[Dict[str, Any]] = []
    with open(csv_path, "r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("method", "")).strip().lower() != str(reference_method).strip().lower():
                continue
            step = int(float(row["step"]))
            if reference_steps_filter is not None and step not in reference_steps_filter:
                continue
            warmup_steps_value = row.get("warmup_steps_fixed")
            if warmup_steps_value in (None, ""):
                raise RuntimeError(
                    f"Reference row step={step} in {csv_path} is missing warmup_steps_fixed; "
                    "use a method curve CSV that exports absolute warmup checkpoints."
                )
            target_metric_value = safe_float(row.get(metric_name))
            if target_metric_value is None:
                raise RuntimeError(
                    f"Reference row step={step} in {csv_path} is missing {metric_name}; "
                    "cannot derive a matched CDRO budget from it."
                )
            rows.append(
                {
                    "reference_method": str(reference_method).strip().lower(),
                    "reference_step": int(step),
                    "reference_metric_name": str(metric_name),
                    "reference_metric_value": float(target_metric_value),
                    "warmup_steps_fixed": int(float(warmup_steps_value)),
                    "reference_row": row,
                }
            )
    rows.sort(key=lambda row: row["reference_step"])
    if not rows:
        raise RuntimeError(
            f"No reference rows found in {csv_path} for method='{reference_method}' "
            f"and step filter={sorted(reference_steps_filter) if reference_steps_filter is not None else 'all'}."
        )
    return rows


def match_weighted_budget_with_linear_robust_cost(
    *,
    warmup_steps: int,
    target_weighted_compute_units: float,
    baseline_weighted_units: float,
    robust_step_weighted_units: float,
    estimate_source: str,
) -> Dict[str, Any]:
    target_robust_budget = max(float(target_weighted_compute_units) - float(baseline_weighted_units), 0.0)
    if float(robust_step_weighted_units) <= 0.0:
        robust_steps = 0
    else:
        robust_steps_float = target_robust_budget / float(robust_step_weighted_units)
        robust_steps_floor = max(int(robust_steps_float), 0)
        candidate_steps = {0, robust_steps_floor, robust_steps_floor + 1}
        robust_steps = min(
            candidate_steps,
            key=lambda candidate: abs(
                float(baseline_weighted_units)
                + float(candidate) * float(robust_step_weighted_units)
                - float(target_weighted_compute_units)
            ),
        )
    matched_total_weighted = float(
        float(baseline_weighted_units) + float(robust_steps) * float(robust_step_weighted_units)
    )
    abs_error = abs(float(target_weighted_compute_units) - matched_total_weighted)
    rel_error = abs_error / float(target_weighted_compute_units) if float(target_weighted_compute_units) > 0.0 else 0.0
    return {
        "warmup_steps_fixed": int(warmup_steps),
        "robust_steps_target": int(robust_steps),
        "target_total_steps": int(warmup_steps + robust_steps),
        "matched_metric_name": "weighted_compute_units",
        "matched_metric_target": float(target_weighted_compute_units),
        "matched_metric_value_estimate": float(matched_total_weighted),
        "matched_metric_abs_error_estimate": float(abs_error),
        "matched_metric_rel_error_estimate": float(rel_error),
        "baseline_weighted_compute_units_target": float(baseline_weighted_units),
        "cdro_robust_step_weighted_compute_units": float(robust_step_weighted_units),
        "cdro_robust_step_weighted_compute_units_estimate_source": str(estimate_source),
    }


def match_cdro_steps_to_reference_weighted_budget(
    *,
    warmup_steps: int,
    target_weighted_compute_units: float,
    args: argparse.Namespace,
    calibration: Dict[str, Any],
) -> Dict[str, Any]:
    baseline_weighted_units = baseline_weighted_compute_units_for_steps(
        steps=int(warmup_steps),
        calibration=calibration,
    )
    robust_step_weighted_units = cdro_robust_step_weighted_compute_units(
        n_steps_path=int(args.n_steps_path),
        inner_steps=int(args.inner_steps),
        total_budget_rho=float(args.cdro_total_budget_rho),
        outer_attack_weight=float(args.outer_attack_weight),
        outer_clean_weight=float(args.outer_clean_weight),
        calibration=calibration,
    )
    if baseline_weighted_units is None or robust_step_weighted_units is None:
        raise RuntimeError("Weighted compute calibration is required for CDRO reference-budget matching.")
    return match_weighted_budget_with_linear_robust_cost(
        warmup_steps=int(warmup_steps),
        target_weighted_compute_units=float(target_weighted_compute_units),
        baseline_weighted_units=float(baseline_weighted_units),
        robust_step_weighted_units=float(robust_step_weighted_units),
        estimate_source="analytic_formula",
    )


def maybe_refine_reference_match_plan_entry_with_pilot(
    *,
    args: argparse.Namespace,
    plan_entry: Dict[str, Any],
    logs_dir: str,
    calibration: Dict[str, Any],
) -> Dict[str, Any]:
    pilot_robust_steps_limit = max(int(args.reference_match_pilot_robust_steps), 0)
    if pilot_robust_steps_limit <= 0:
        return plan_entry
    if int(plan_entry.get("robust_steps_target", 0)) <= 0:
        return plan_entry

    warmup_steps = int(plan_entry["warmup_steps_fixed"])
    pilot_robust_steps = max(1, min(pilot_robust_steps_limit, int(plan_entry["robust_steps_target"])))
    baseline_ckpt_path = resolve_requested_baseline_checkpoint(args, warmup_steps)
    reference_step = int(plan_entry["reference_step"])
    pilot_exp_name = (
        f"{args.prefix}_pilot_refst{reference_step}_warm{warmup_steps}_rob{pilot_robust_steps}_s{args.seed}"
    )
    pilot_metrics_path = os.path.join(args.outdir, pilot_exp_name, "metrics.json")
    pilot_log_path = os.path.join(logs_dir, f"{pilot_exp_name}.log")
    if args.skip_existing and os.path.isfile(pilot_metrics_path):
        print(
            f"[cdro-curve] reuse pilot metrics ref_step={reference_step} "
            f"warmup_steps={warmup_steps} metrics={pilot_metrics_path}",
            flush=True,
        )
    else:
        print(
            f"[cdro-curve] pilot ref_step={reference_step} warmup_steps={warmup_steps} "
            f"robust_steps={pilot_robust_steps} baseline_ckpt={baseline_ckpt_path or 'none'} "
            f"log={pilot_log_path}",
            flush=True,
        )
        run_point(
            args=args,
            target_total_steps=int(warmup_steps + pilot_robust_steps),
            warmup_steps=warmup_steps,
            exp_name=pilot_exp_name,
            log_path=pilot_log_path,
            baseline_ckpt_path=baseline_ckpt_path,
            use_baseline_steps_override=True,
        )

    pilot_row = extract_cdro_row(
        pilot_metrics_path,
        baseline_ckpt_requested=baseline_ckpt_path,
        calibration=calibration,
        train_accelerator_count=int(args.train_accelerator_count),
    )
    pilot_robust_steps_observed = int(pilot_row["robust_steps_target"])
    pilot_robust_weighted_compute_units = safe_float(pilot_row.get("robust_weighted_compute_units"))
    if pilot_robust_steps_observed <= 0 or pilot_robust_weighted_compute_units is None:
        raise RuntimeError(
            "Pilot-based CDRO matching requires a completed robust phase with direct weighted-compute accounting. "
            f"Got robust_steps={pilot_robust_steps_observed} and "
            f"robust_weighted_compute_units={pilot_robust_weighted_compute_units} for {pilot_metrics_path}."
        )
    pilot_baseline_weighted_compute_units = safe_float(pilot_row.get("baseline_weighted_compute_units"))
    if pilot_baseline_weighted_compute_units is None:
        pilot_baseline_weighted_compute_units = float(plan_entry["baseline_weighted_compute_units_target"])
    pilot_robust_weighted_per_step = (
        float(pilot_robust_weighted_compute_units) / float(pilot_robust_steps_observed)
    )
    matched = match_weighted_budget_with_linear_robust_cost(
        warmup_steps=int(warmup_steps),
        target_weighted_compute_units=float(plan_entry["reference_metric_value"]),
        baseline_weighted_units=float(pilot_baseline_weighted_compute_units),
        robust_step_weighted_units=float(pilot_robust_weighted_per_step),
        estimate_source="pilot_observed_robust_step",
    )
    robust_train_wall_clock_sec = safe_float(pilot_row.get("robust_train_wall_clock_sec_effective"))
    refined_entry = dict(plan_entry)
    refined_entry.update(matched)
    refined_entry.update(
        {
            "pilot_enabled": True,
            "pilot_exp_name": pilot_exp_name,
            "pilot_metrics_path": pilot_metrics_path,
            "pilot_log_path": pilot_log_path,
            "pilot_target_total_steps": int(warmup_steps + pilot_robust_steps),
            "pilot_robust_steps_requested": int(pilot_robust_steps),
            "pilot_robust_steps_observed": int(pilot_robust_steps_observed),
            "pilot_robust_weighted_compute_units": float(pilot_robust_weighted_compute_units),
            "pilot_robust_weighted_compute_units_per_step": float(pilot_robust_weighted_per_step),
            "pilot_baseline_weighted_compute_units": float(pilot_baseline_weighted_compute_units),
            "pilot_baseline_ckpt_loaded": bool(pilot_row["baseline_ckpt_loaded"]),
            "pilot_robust_train_wall_clock_sec": (
                None if robust_train_wall_clock_sec is None else float(robust_train_wall_clock_sec)
            ),
            "pilot_robust_train_wall_clock_sec_per_step": (
                None
                if robust_train_wall_clock_sec is None
                else float(robust_train_wall_clock_sec) / float(pilot_robust_steps_observed)
            ),
        }
    )
    print(
        f"[cdro-curve] pilot_refined ref_step={reference_step} "
        f"old_total_steps={plan_entry['target_total_steps']} "
        f"new_total_steps={refined_entry['target_total_steps']} "
        f"pilot_robust_weighted_per_step={pilot_robust_weighted_per_step:.6f}",
        flush=True,
    )
    return refined_entry


def run_point(
    *,
    args: argparse.Namespace,
    target_total_steps: int,
    warmup_steps: int,
    exp_name: str,
    log_path: str,
    baseline_ckpt_path: Optional[str],
    use_baseline_steps_override: bool,
) -> str:
    cmd = [
        args.python_bin,
        os.path.join(ROOT_DIR, "toy", "run_toy.py"),
        "--exp-name",
        exp_name,
        "--outdir",
        args.outdir,
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--steps",
        str(target_total_steps),
        "--batch-size",
        str(args.batch_size),
        "--log-every",
        str(args.log_every),
        "--eval-samples",
        str(args.eval_samples),
        "--debug-eval-batch",
        str(args.debug_eval_batch),
        "--debug-terminal-step",
        str(args.debug_terminal_step),
        "--n-steps-path",
        str(args.n_steps_path),
        "--hidden-dim",
        str(args.hidden_dim),
        "--dataset-kind",
        "image_folder",
        "--model-kind",
        "image_conv",
        "--diagnostics-kind",
        "image_basic",
        "--dataset-path",
        args.dataset_path,
        "--dataset-val-path",
        args.dataset_val_path,
        "--image-size",
        str(args.image_size),
        "--image-channels",
        str(args.image_channels),
        "--image-train-size",
        str(args.image_train_size),
        "--image-val-size",
        str(args.image_val_size),
        "--image-split-seed",
        str(args.image_split_seed),
        "--sigma-min",
        str(args.sigma_min),
        "--sigma-max",
        str(args.sigma_max),
        "--compute-fid",
        "--fid-ref-path",
        args.fid_ref_path,
        "--method-version",
        "cdro",
        "--disable-baseline-gate",
        "--skip-checks",
        "--inner-steps",
        str(args.inner_steps),
        "--outer-attack-weight",
        str(args.outer_attack_weight),
        "--outer-clean-weight",
        str(args.outer_clean_weight),
        "--cdro-step-size",
        str(args.cdro_step_size),
        "--cdro-total-budget-rho",
        str(args.cdro_total_budget_rho),
        "--cdro-time-horizon",
        str(args.cdro_time_horizon),
        "--cdro-edm-ladder-mode",
        str(args.cdro_edm_ladder_mode),
        "--cdro-warmup-fraction",
        str(args.cdro_warmup_fraction),
    ]
    if use_baseline_steps_override:
        cmd.extend(
            [
                "--baseline-steps-override",
                str(warmup_steps),
            ]
        )
    if baseline_ckpt_path:
        cmd.extend(
            [
                "--baseline-ckpt-path",
                baseline_ckpt_path,
                "--disable-baseline-ckpt-strict-meta",
            ]
        )
    if bool(args.disable_baseline_ckpt):
        cmd.append("--disable-baseline-ckpt")
    if str(args.weighted_compute_calibration_path).strip():
        cmd.extend(
            [
                "--weighted-compute-calibration-path",
                str(args.weighted_compute_calibration_path).strip(),
            ]
        )
    if float(args.weighted_inputgrad_alpha) > 0.0 and float(args.weighted_parambackward_beta) > 0.0:
        cmd.extend(
            [
                "--weighted-inputgrad-alpha",
                str(args.weighted_inputgrad_alpha),
                "--weighted-parambackward-beta",
                str(args.weighted_parambackward_beta),
            ]
        )
    env = maybe_set_fid_detector_env(os.environ)
    with open(log_path, "w", encoding="utf-8") as handle:
        subprocess.run(cmd, cwd=ROOT_DIR, env=env, check=True, stdout=handle, stderr=subprocess.STDOUT)
    return os.path.join(args.outdir, exp_name, "metrics.json")


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_baseline_rows(
    paths: List[str],
    *,
    calibration: Dict,
    train_accelerator_count: int,
) -> List[Dict]:
    by_step: Dict[int, Dict] = {}
    for path in paths:
        if not path or not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                step = int(float(row["step"]))
                train_wall_clock_sec = float(
                    row.get("train_wall_clock_sec", row.get("train_wall_clock_median_sec", row["train_elapsed_median_sec"]))
                )
                weighted_units = row.get("weighted_compute_units_median")
                if weighted_units not in (None, ""):
                    weighted_units = float(weighted_units)
                else:
                    weighted_units = weighted_compute_units(
                        n_fwd=0.0,
                        n_fwd_inputgrad=0.0,
                        n_fwd_parambackward=float(step),
                        calibration=calibration,
                    )
                by_step[step] = {
                    "method": "baseline_edm",
                    "step": step,
                    "compute_budget_be": float(step),
                    "baseline_compute_be": float(step),
                    "robust_compute_be": 0.0,
                    "weighted_compute_units": weighted_units,
                    "baseline_weighted_compute_units": weighted_units,
                    "robust_weighted_compute_units": 0.0
                    if calibration.get("available", False)
                    else None,
                    "images_shown_m": float(row["images_shown_m"]),
                    "fid": float(row["fid_median"]),
                    "train_wall_clock_sec": float(train_wall_clock_sec),
                    "train_elapsed_sec": float(train_wall_clock_sec),
                    "train_gpu_hours": float(train_wall_clock_sec)
                    * float(max(int(train_accelerator_count), 0))
                    / 3600.0,
                    "train_wall_clock_complete": True,
                    "source": path,
                }
    return [by_step[step] for step in sorted(by_step.keys())]


def extract_cdro_row(
    metrics_path: str,
    *,
    baseline_ckpt_requested: Optional[str],
    calibration: Dict,
    train_accelerator_count: int,
) -> Dict:
    payload = load_json(metrics_path)
    metrics = payload["metrics"]
    config = payload.get("config", {})
    flow = metrics["flow_debug"]
    runtime = flow["runtime"]
    sample_quality = metrics.get("sample_quality_debug", {})
    compute_accounting = flow["compute_accounting"]
    budget_accounting = flow["budget_accounting"]
    baseline_phase_steps = int(flow["baseline_phase_steps"])
    robust_phase_steps = int(flow["robust_phase_steps"])
    total_steps_requested = int(flow["total_steps_requested"])
    baseline_compute_be = float(compute_accounting["baseline_batch_equiv_denoiser_evals_total"])
    robust_compute_be = float(compute_accounting["robust_batch_equiv_denoiser_evals_total"])
    total_compute_be = float(
        compute_accounting.get(
            "effective_train_batch_equiv_denoiser_evals_total",
            baseline_compute_be + robust_compute_be,
        )
    )
    baseline_weighted_compute = compute_accounting.get("baseline_weighted_compute_units")
    robust_weighted_compute = compute_accounting.get("robust_weighted_compute_units")
    total_weighted_compute = compute_accounting.get("weighted_compute_units", runtime.get("weighted_compute_units"))
    adjusted_cdro_weighted = cdro_adjusted_weighted_compute_from_compute_accounting(
        compute_accounting=compute_accounting,
        calibration=calibration,
    )
    weighted_compute_source = "metrics_payload"
    robust_logging_only_forward_count_excluded = None
    if adjusted_cdro_weighted is not None:
        baseline_weighted_compute = adjusted_cdro_weighted["baseline_weighted_compute_units"]
        robust_weighted_compute = adjusted_cdro_weighted["robust_weighted_compute_units"]
        total_weighted_compute = adjusted_cdro_weighted["weighted_compute_units"]
        robust_logging_only_forward_count_excluded = adjusted_cdro_weighted[
            "robust_logging_only_forward_count_excluded"
        ]
        weighted_compute_source = "cdro_adjusted_excluding_logging_forward"
    if total_weighted_compute is None and calibration.get("available", False):
        path_steps = int(config.get("n_steps_path", 24))
        attack_weight = float(flow.get("outer_attack_weight", config.get("outer_attack_weight", 0.0)))
        clean_weight = float(flow.get("outer_clean_weight", config.get("outer_clean_weight", 0.0)))
        inner_steps = int(flow.get("inner_steps", config.get("inner_steps", 0)))
        attack_enabled = bool(
            float(flow.get("cdro_total_budget_rho", config.get("cdro_total_budget_rho", 0.0))) > 0.0
            and attack_weight > 0.0
            and inner_steps > 0
        )
        baseline_weighted_compute = weighted_compute_units(
            n_fwd=0.0,
            n_fwd_inputgrad=0.0,
            n_fwd_parambackward=float(baseline_phase_steps),
            calibration=calibration,
        )
        robust_weighted_compute = weighted_compute_units(
            n_fwd=0.0,
            n_fwd_inputgrad=float(path_steps * inner_steps * robust_phase_steps) if attack_enabled else 0.0,
            n_fwd_parambackward=float(
                path_steps
                * robust_phase_steps
                * (
                    int((attack_weight > 0.0) + (clean_weight > 0.0))
                    if attack_enabled
                    else int((attack_weight > 0.0) or (clean_weight > 0.0))
                )
            ),
            calibration=calibration,
        )
        if baseline_weighted_compute is not None and robust_weighted_compute is not None:
            total_weighted_compute = float(baseline_weighted_compute + robust_weighted_compute)
    train_wall_clock_sec = compute_accounting.get("train_wall_clock_sec", runtime.get("train_wall_clock_sec"))
    if train_wall_clock_sec is None:
        train_wall_clock_sec = runtime.get("effective_train_total")
    train_gpu_hours = compute_accounting.get("train_gpu_hours", runtime.get("train_gpu_hours"))
    if train_gpu_hours is None and train_wall_clock_sec is not None:
        train_gpu_hours = float(train_wall_clock_sec) * float(max(int(train_accelerator_count), 0)) / 3600.0
    baseline_train_wall_clock_sec = runtime.get("baseline_train_wall_clock_sec_effective")
    robust_train_wall_clock_sec = runtime.get("robust_phase")
    total_images_shown_m = float(
        budget_accounting.get(
            "effective_train_images_seen_total",
            total_steps_requested * int(budget_accounting["batch_size"]),
        )
    ) / 1_000_000.0
    warmup_only = bool(robust_phase_steps <= 0)
    fid_value = (
        float(sample_quality["baseline_fid"])
        if warmup_only and sample_quality.get("baseline_fid") is not None
        else (
            float(sample_quality["robust_fid"])
            if sample_quality.get("robust_fid") is not None
            else float("nan")
        )
    )
    return {
        "method": "cdro",
        "step": int(total_steps_requested),
        "compute_budget_be": float(total_compute_be),
        "baseline_compute_be": float(baseline_compute_be),
        "robust_compute_be": float(robust_compute_be),
        "weighted_compute_units": None if total_weighted_compute is None else float(total_weighted_compute),
        "baseline_weighted_compute_units": (
            None if baseline_weighted_compute is None else float(baseline_weighted_compute)
        ),
        "robust_weighted_compute_units": None if robust_weighted_compute is None else float(robust_weighted_compute),
        "weighted_compute_source": str(weighted_compute_source),
        "robust_logging_only_forward_count_excluded": (
            None
            if robust_logging_only_forward_count_excluded is None
            else float(robust_logging_only_forward_count_excluded)
        ),
        "images_shown_m": float(total_images_shown_m),
        "fid": float(fid_value),
        "baseline_fid_same_run": (
            float(sample_quality["baseline_fid"]) if sample_quality.get("baseline_fid") is not None else float("nan")
        ),
        "train_wall_clock_sec": None if train_wall_clock_sec is None else float(train_wall_clock_sec),
        "train_gpu_hours": None if train_gpu_hours is None else float(train_gpu_hours),
        "train_elapsed_sec": float(runtime["effective_train_total"]),
        "runtime_total_sec": float(runtime["total"]),
        "baseline_train_wall_clock_sec_effective": (
            None if baseline_train_wall_clock_sec is None else float(baseline_train_wall_clock_sec)
        ),
        "robust_train_wall_clock_sec_effective": (
            None if robust_train_wall_clock_sec is None else float(robust_train_wall_clock_sec)
        ),
        "baseline_train_wall_clock_source": runtime.get("baseline_train_wall_clock_sec_effective_source"),
        "warmup_steps_fixed": int(baseline_phase_steps),
        "robust_steps_target": int(robust_phase_steps),
        "warmup_only": bool(warmup_only),
        "baseline_ckpt_requested": baseline_ckpt_requested,
        "baseline_ckpt_loaded": bool(flow["baseline_ckpt_loaded"]),
        "phase_step_split_mode": str(flow["phase_step_split_mode"]),
        "cdro_edm_ladder_mode": str(flow.get("cdro_edm_ladder_mode", "deterministic_midpoint_quantile")),
        "train_wall_clock_complete": bool(
            compute_accounting.get("train_wall_clock_complete", train_wall_clock_sec is not None)
        ),
        "metrics_path": metrics_path,
    }


def write_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_curve_outputs(
    *,
    args: argparse.Namespace,
    run_plan: Dict[str, Any],
    baseline_paths: List[str],
    baseline_rows: List[Dict],
    cdro_rows: List[Dict],
    weighted_calibration: Dict[str, Any],
    announce: bool,
) -> None:
    combined_rows = baseline_rows + cdro_rows
    combined_rows.sort(
        key=lambda row: (
            row["method"],
            float(row["train_wall_clock_sec"]) if row.get("train_wall_clock_sec") is not None else float("inf"),
            float(row["weighted_compute_units"]) if row.get("weighted_compute_units") is not None else float("inf"),
            float(row["compute_budget_be"]),
        )
    )

    cdro_csv = os.path.join(args.outdir, f"{args.prefix}_cdro_curve.csv")
    combined_csv = os.path.join(args.outdir, f"{args.prefix}_compare_curve.csv")
    summary_json = os.path.join(args.outdir, f"{args.prefix}_compare_curve_summary.json")
    plot_wall_clock = os.path.join(args.outdir, f"{args.prefix}_fid_vs_train_wall_clock.png")
    plot_weighted = os.path.join(args.outdir, f"{args.prefix}_fid_vs_weighted_compute.png")
    plot_legacy = os.path.join(args.outdir, f"{args.prefix}_fid_vs_batch_equiv.png")

    write_csv(cdro_csv, cdro_rows)
    write_csv(combined_csv, combined_rows)
    make_plot(
        plot_wall_clock,
        baseline_rows=baseline_rows,
        cdro_rows=cdro_rows,
        train_percent_label=str(args.train_percent_label),
        x_key="train_wall_clock_sec",
        x_label="Train Wall-Clock (sec)",
        title_suffix="Train Wall-Clock",
    )
    make_plot(
        plot_weighted,
        baseline_rows=baseline_rows,
        cdro_rows=cdro_rows,
        train_percent_label=str(args.train_percent_label),
        x_key="weighted_compute_units",
        x_label="Weighted Compute Units",
        title_suffix="Weighted Compute",
    )
    make_plot(
        plot_legacy,
        baseline_rows=baseline_rows,
        cdro_rows=cdro_rows,
        train_percent_label=str(args.train_percent_label),
        x_key="compute_budget_be",
        x_label="Legacy Batch-Equivalent Denoiser Evals",
        title_suffix="Legacy Batch-Equivalent Compute",
    )

    summary = {
        "protocol": {
            **run_plan["summary_protocol"],
            "primary_metric": "train_wall_clock_sec",
            "secondary_metric": "weighted_compute_units",
            "legacy_metric": "batch_equiv_denoiser_evals",
            "baseline_overlay_included": bool(baseline_rows),
            "primary_definition": (
                "Effective training-only wall-clock on fixed hardware, including reused baseline warmup time "
                "when the reference checkpoint has recoverable runtime metadata."
            ),
            "secondary_definition": (
                "Weighted-op budget 1*N_fwd + alpha*N_fwd_plus_inputgrad + beta*N_fwd_plus_parambackward, "
                "with alpha/beta from the supplied calibration."
            ),
            "legacy_definition": (
                "One denoiser forward over one training batch counts as 1 unit; "
                "a forward over B*T path states counts as T units."
            ),
            "diagnostics_included": False,
            "train_accelerator_count": int(args.train_accelerator_count),
            "weighted_compute_calibration": weighted_calibration,
        },
        "steps_list": [int(v) for v in run_plan["steps_list"]],
        "baseline_aggregate_paths": baseline_paths,
        "baseline_checkpoint_dir": str(args.baseline_checkpoint_dir),
        "baseline_checkpoint_reuse_enabled": bool(not args.disable_baseline_ckpt),
        "cdro_rows": cdro_rows,
        "baseline_rows": baseline_rows,
        "train_percent_label": str(args.train_percent_label),
        "plot_paths": {
            "train_wall_clock_sec": plot_wall_clock,
            "weighted_compute_units": plot_weighted,
            "batch_equiv_denoiser_evals": plot_legacy,
        },
        "combined_csv": combined_csv,
    }
    with open(summary_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    if announce:
        print(f"[cdro-curve] wrote {cdro_csv}", flush=True)
        print(f"[cdro-curve] wrote {combined_csv}", flush=True)
        print(f"[cdro-curve] wrote {summary_json}", flush=True)
        print(f"[cdro-curve] wrote {plot_wall_clock}", flush=True)
        print(f"[cdro-curve] wrote {plot_weighted}", flush=True)
        print(f"[cdro-curve] wrote {plot_legacy}", flush=True)


def make_plot(
    path: str,
    baseline_rows: List[Dict],
    cdro_rows: List[Dict],
    *,
    train_percent_label: str,
    x_key: str,
    x_label: str,
    title_suffix: str,
) -> None:
    plt.figure(figsize=(8, 5))
    baseline_pairs = [(row[x_key], row["fid"]) for row in baseline_rows if row.get(x_key) is not None]
    cdro_pairs = [(row[x_key], row["fid"]) for row in cdro_rows if row.get(x_key) is not None]
    if baseline_pairs:
        plt.plot(
            [value for value, _ in baseline_pairs],
            [fid for _, fid in baseline_pairs],
            marker="o",
            linewidth=2.0,
            label="Baseline EDM",
        )
    if cdro_pairs:
        plt.plot(
            [value for value, _ in cdro_pairs],
            [fid for _, fid in cdro_pairs],
            marker="o",
            linewidth=2.0,
            label="CDRO",
        )
    plt.xlabel(x_label)
    plt.ylabel("FID")
    plt.title(f"Simpsons-MNIST RGB {train_percent_label}: FID vs {title_suffix}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def build_run_plan(args: argparse.Namespace, calibration: Dict[str, Any]) -> Dict[str, Any]:
    reference_csv = str(args.match_reference_csv).strip()
    if not reference_csv:
        steps_list = parse_steps_list(args.steps_list)
        entries: List[Dict[str, Any]] = []
        for target_total_steps in steps_list:
            protocol_name = "independent_per_point_fixed_fraction_warmup_unweighted_fallback"
            protocol_detail = "weighted calibration unavailable; fallback to fractional warmup derived from cdro_warmup_fraction"
            warmup_steps = int(target_total_steps * float(args.cdro_warmup_fraction))
            target_warmup_weighted_compute_fraction = None
            wdro_reference_robust_step_weighted_compute_units = None
            cdro_robust_step_weighted_units = None
            if calibration.get("available", False):
                wdro_reference = estimate_wdro_reference_warmup_weighted_compute_fraction(
                    total_steps=int(target_total_steps),
                    warmup_fraction=float(args.cdro_warmup_fraction),
                    batch_size=int(args.batch_size),
                    train_pool_size=int(args.image_train_size),
                    calibration=calibration,
                )
                cdro_robust_step_weighted_units = cdro_robust_step_weighted_compute_units(
                    n_steps_path=int(args.n_steps_path),
                    inner_steps=int(args.inner_steps),
                    total_budget_rho=float(args.cdro_total_budget_rho),
                    outer_attack_weight=float(args.outer_attack_weight),
                    outer_clean_weight=float(args.outer_clean_weight),
                    calibration=calibration,
                )
                baseline_step_weighted_units = baseline_weighted_compute_units_for_steps(
                    steps=1,
                    calibration=calibration,
                )
                if (
                    wdro_reference is not None
                    and cdro_robust_step_weighted_units is not None
                    and baseline_step_weighted_units is not None
                ):
                    target_warmup_weighted_compute_fraction = float(
                        wdro_reference["warmup_weighted_compute_fraction"]
                    )
                    wdro_reference_robust_step_weighted_compute_units = float(
                        wdro_reference["wdro_robust_step_weighted_compute_units"]
                    )
                    warmup_steps = solve_warmup_steps_for_target_compute_fraction(
                        total_steps=int(target_total_steps),
                        target_warmup_compute_fraction=float(target_warmup_weighted_compute_fraction),
                        baseline_step_compute_units=float(baseline_step_weighted_units),
                        robust_step_compute_units=float(cdro_robust_step_weighted_units),
                    )
                    protocol_name = "independent_per_point_weighted_compute_matched_warmup"
                    protocol_detail = (
                        "warmup steps chosen so CDRO baseline weighted-compute share matches the "
                        "WDRO-style weighted-compute share implied by cdro_warmup_fraction"
                    )
            entries.append(
                {
                    "exp_name": f"{args.prefix}_st{target_total_steps}_s{args.seed}",
                    "target_total_steps": int(target_total_steps),
                    "warmup_steps_fixed": int(warmup_steps),
                    "robust_steps_target": int(max(target_total_steps - warmup_steps, 0)),
                    "protocol_name": str(protocol_name),
                    "protocol_detail": str(protocol_detail),
                    "reference_warmup_step_fraction": float(args.cdro_warmup_fraction),
                    "target_warmup_weighted_compute_fraction": target_warmup_weighted_compute_fraction,
                    "wdro_reference_robust_step_weighted_compute_units": (
                        wdro_reference_robust_step_weighted_compute_units
                    ),
                    "cdro_robust_step_weighted_compute_units": (
                        None if cdro_robust_step_weighted_units is None else float(cdro_robust_step_weighted_units)
                    ),
                }
            )
        return {
            "mode": "weighted_compute_matched_warmup" if calibration.get("available", False) else "fixed_fraction_fallback",
            "entries": entries,
            "steps_list": steps_list,
            "summary_protocol": {
                "name": (
                    "independent_per_point_weighted_compute_matched_warmup"
                    if calibration.get("available", False)
                    else "independent_per_point_fixed_fraction_warmup_unweighted_fallback"
                ),
                "reference_warmup_step_fraction": float(args.cdro_warmup_fraction),
            },
        }

    reference_steps_filter = parse_optional_steps_list(args.match_reference_steps_list)
    reference_rows = load_reference_rows_for_matching(
        csv_path=reference_csv,
        reference_method=str(args.match_reference_method),
        metric_name=str(args.match_reference_metric),
        reference_steps_filter=reference_steps_filter,
    )
    entries = []
    for reference in reference_rows:
        matched = match_cdro_steps_to_reference_weighted_budget(
            warmup_steps=int(reference["warmup_steps_fixed"]),
            target_weighted_compute_units=float(reference["reference_metric_value"]),
            args=args,
            calibration=calibration,
        )
        reference_step = int(reference["reference_step"])
        target_total_steps = int(matched["target_total_steps"])
        entries.append(
            {
                "exp_name": f"{args.prefix}_refst{reference_step}_cdrost{target_total_steps}_s{args.seed}",
                "protocol_name": "shared_warmup_reference_weighted_budget_match",
                "protocol_detail": (
                    f"shared baseline prefix from {reference['reference_method']} step={reference_step} "
                    f"and matched {reference['reference_metric_name']}"
                ),
                **reference,
                **matched,
            }
        )
    return {
        "mode": "reference_weighted_budget_match",
        "entries": entries,
        "steps_list": [int(entry["target_total_steps"]) for entry in entries],
        "summary_protocol": {
            "name": "shared_warmup_reference_weighted_budget_match",
            "reference_curve_csv": os.path.abspath(reference_csv),
            "reference_method": str(args.match_reference_method).strip().lower(),
            "reference_metric": str(args.match_reference_metric),
            "reference_match_pilot_robust_steps": int(max(args.reference_match_pilot_robust_steps, 0)),
            "reference_steps_filter": (
                None if reference_steps_filter is None else sorted(int(step) for step in reference_steps_filter)
            ),
        },
    }


def main() -> None:
    args = parse_args()
    ensure_dir(args.outdir)
    logs_dir = os.path.join(args.outdir, "logs")
    ensure_dir(logs_dir)
    weighted_calibration = resolve_weighted_calibration(args)
    print(
        "[cdro-curve] weighted_compute "
        f"available={weighted_calibration['available']} "
        f"source={weighted_calibration['source']} "
        f"alpha={weighted_calibration['inputgrad_alpha']} "
        f"beta={weighted_calibration['parambackward_beta']}",
        flush=True,
    )
    run_plan = build_run_plan(args, weighted_calibration)
    print(
        f"[cdro-curve] protocol={run_plan['summary_protocol']['name']} "
        f"checkpoints={run_plan['steps_list']}",
        flush=True,
    )
    if run_plan["mode"] == "reference_weighted_budget_match":
        print(
            "[cdro-curve] reference_match "
            f"csv={run_plan['summary_protocol']['reference_curve_csv']} "
            f"method={run_plan['summary_protocol']['reference_method']} "
            f"metric={run_plan['summary_protocol']['reference_metric']}",
            flush=True,
        )

    baseline_paths = [] if bool(args.skip_baseline_overlay) else (args.baseline_aggregate or [DEFAULT_BASELINE_AGGREGATE])
    baseline_rows = load_baseline_rows(
        baseline_paths,
        calibration=weighted_calibration,
        train_accelerator_count=int(args.train_accelerator_count),
    )

    cdro_rows: List[Dict] = []
    for raw_plan_entry in run_plan["entries"]:
        plan_entry = dict(raw_plan_entry)
        if run_plan["mode"] == "reference_weighted_budget_match":
            plan_entry = maybe_refine_reference_match_plan_entry_with_pilot(
                args=args,
                plan_entry=plan_entry,
                logs_dir=logs_dir,
                calibration=weighted_calibration,
            )
        target_total_steps = int(plan_entry["target_total_steps"])
        warmup_steps = int(plan_entry["warmup_steps_fixed"])
        baseline_ckpt_path = resolve_requested_baseline_checkpoint(args, warmup_steps)
        exp_name = str(plan_entry["exp_name"])
        metrics_path = os.path.join(args.outdir, exp_name, "metrics.json")
        log_path = os.path.join(logs_dir, f"{exp_name}.log")
        if args.skip_existing and os.path.isfile(metrics_path):
            print(f"[cdro-curve] reuse existing metrics for step={target_total_steps}: {metrics_path}", flush=True)
        else:
            log_message = (
                f"[cdro-curve] run step={target_total_steps} warmup_steps={warmup_steps} "
                f"baseline_ckpt={baseline_ckpt_path or 'none'} log={log_path}"
            )
            if run_plan["mode"] == "reference_weighted_budget_match":
                log_message += (
                    f" ref_step={plan_entry['reference_step']}"
                    f" target_weighted={plan_entry['matched_metric_target']:.4f}"
                    f" est_weighted={plan_entry['matched_metric_value_estimate']:.4f}"
                    f" est_abs_err={plan_entry['matched_metric_abs_error_estimate']:.4f}"
                )
            print(log_message, flush=True)
            run_point(
                args=args,
                target_total_steps=target_total_steps,
                warmup_steps=warmup_steps,
                exp_name=exp_name,
                log_path=log_path,
                baseline_ckpt_path=baseline_ckpt_path,
                use_baseline_steps_override=(run_plan["mode"] == "reference_weighted_budget_match"),
            )
        row = extract_cdro_row(
            metrics_path,
            baseline_ckpt_requested=baseline_ckpt_path,
            calibration=weighted_calibration,
            train_accelerator_count=int(args.train_accelerator_count),
        )
        row.update(
            {
                "protocol_name": str(plan_entry["protocol_name"]),
                "protocol_detail": str(plan_entry["protocol_detail"]),
            }
        )
        for optional_key in (
            "reference_warmup_step_fraction",
            "target_warmup_weighted_compute_fraction",
            "wdro_reference_robust_step_weighted_compute_units",
            "cdro_robust_step_weighted_compute_units",
        ):
            if optional_key in plan_entry:
                row[optional_key] = plan_entry[optional_key]
        if run_plan["mode"] == "reference_weighted_budget_match":
            row.update(
                {
                    "reference_method": str(plan_entry["reference_method"]),
                    "reference_step": int(plan_entry["reference_step"]),
                    "reference_metric_name": str(plan_entry["reference_metric_name"]),
                    "reference_metric_target": float(plan_entry["reference_metric_value"]),
                    "reference_warmup_steps_fixed": int(plan_entry["warmup_steps_fixed"]),
                    "reference_weighted_compute_units": float(plan_entry["reference_metric_value"]),
                    "matched_metric_name": str(plan_entry["matched_metric_name"]),
                    "matched_metric_target": float(plan_entry["matched_metric_target"]),
                    "matched_metric_value_estimate": float(plan_entry["matched_metric_value_estimate"]),
                    "matched_metric_abs_error_estimate": float(plan_entry["matched_metric_abs_error_estimate"]),
                    "matched_metric_rel_error_estimate": float(plan_entry["matched_metric_rel_error_estimate"]),
                    "baseline_weighted_compute_units_target": float(plan_entry["baseline_weighted_compute_units_target"]),
                    "cdro_robust_step_weighted_compute_units": float(
                        plan_entry["cdro_robust_step_weighted_compute_units"]
                    ),
                    "cdro_robust_step_weighted_compute_units_estimate_source": str(
                        plan_entry.get("cdro_robust_step_weighted_compute_units_estimate_source", "analytic_formula")
                    ),
                }
            )
            if bool(plan_entry.get("pilot_enabled", False)):
                row.update(
                    {
                        "pilot_exp_name": str(plan_entry["pilot_exp_name"]),
                        "pilot_metrics_path": str(plan_entry["pilot_metrics_path"]),
                        "pilot_log_path": str(plan_entry["pilot_log_path"]),
                        "pilot_target_total_steps": int(plan_entry["pilot_target_total_steps"]),
                        "pilot_robust_steps_requested": int(plan_entry["pilot_robust_steps_requested"]),
                        "pilot_robust_steps_observed": int(plan_entry["pilot_robust_steps_observed"]),
                        "pilot_robust_weighted_compute_units": float(
                            plan_entry["pilot_robust_weighted_compute_units"]
                        ),
                        "pilot_robust_weighted_compute_units_per_step": float(
                            plan_entry["pilot_robust_weighted_compute_units_per_step"]
                        ),
                        "pilot_baseline_weighted_compute_units": float(
                            plan_entry["pilot_baseline_weighted_compute_units"]
                        ),
                        "pilot_baseline_ckpt_loaded": bool(plan_entry["pilot_baseline_ckpt_loaded"]),
                        "pilot_robust_train_wall_clock_sec": plan_entry["pilot_robust_train_wall_clock_sec"],
                        "pilot_robust_train_wall_clock_sec_per_step": plan_entry[
                            "pilot_robust_train_wall_clock_sec_per_step"
                        ],
                    }
                )
        cdro_rows.append(row)
        completion_message = (
            f"[cdro-curve] step={row['step']} train_wall_clock_sec={row['train_wall_clock_sec']} "
            f"weighted_compute_units={row['weighted_compute_units']} fid={row['fid']:.4f} "
            f"warmup={row['warmup_steps_fixed']} baseline_ckpt_loaded={row['baseline_ckpt_loaded']}"
        )
        if run_plan["mode"] == "reference_weighted_budget_match":
            actual_weighted = safe_float(row.get("weighted_compute_units"))
            target_weighted = float(plan_entry["matched_metric_target"])
            actual_abs_error = None if actual_weighted is None else abs(float(actual_weighted) - target_weighted)
            completion_message += (
                f" ref_step={plan_entry['reference_step']}"
                f" target_weighted={target_weighted:.4f}"
                f" actual_abs_err={actual_abs_error}"
            )
            row["matched_metric_value_actual"] = actual_weighted
            row["matched_metric_abs_error_actual"] = actual_abs_error
            row["matched_metric_rel_error_actual"] = (
                None
                if actual_abs_error is None or target_weighted <= 0.0
                else float(actual_abs_error) / float(target_weighted)
            )
        print(completion_message, flush=True)
        write_curve_outputs(
            args=args,
            run_plan=run_plan,
            baseline_paths=baseline_paths,
            baseline_rows=baseline_rows,
            cdro_rows=cdro_rows,
            weighted_calibration=weighted_calibration,
            announce=False,
        )
    write_curve_outputs(
        args=args,
        run_plan=run_plan,
        baseline_paths=baseline_paths,
        baseline_rows=baseline_rows,
        cdro_rows=cdro_rows,
        weighted_calibration=weighted_calibration,
        announce=True,
    )


if __name__ == "__main__":
    main()
