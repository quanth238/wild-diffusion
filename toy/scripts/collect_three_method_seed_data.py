#!/usr/bin/env python3
import argparse
import csv
import json
import os
import subprocess
import sys
from typing import Dict, Iterable, List, Optional, Tuple


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


from toy.compute_accounting import (  # noqa: E402
    baseline_weighted_compute_units_for_steps,
    cdro_robust_step_weighted_compute_units,
    load_weighted_compute_calibration,
    solve_warmup_steps_for_target_compute_fraction,
    wdro_robust_step_weighted_compute_units,
)
from toy.config import ToyConfig  # noqa: E402


DEFAULT_TRAIN_ROOT = os.path.join(ROOT_DIR, "toy_data", "simpsons_mnist_rgb", "imagefolder", "train")
DEFAULT_VAL_ROOT = os.path.join(ROOT_DIR, "toy_data", "simpsons_mnist_rgb", "imagefolder", "test")
DEFAULT_FID_REF = os.path.join(
    ROOT_DIR,
    "toy_data",
    "simpsons_mnist_rgb",
    "fid_refs",
    "simpsons_mnist_rgb_test_28x28.npz",
)
DEFAULT_CALIBRATION = os.path.join(
    ROOT_DIR,
    "toy_outputs",
    "compute_calibration",
    "simpsons_mnist_rgb_image_conv_edm_b256_h64_cuda.json",
)

# Reuse the previous 25-knot relative template, but apply it in weighted-compute space.
GRID_TEMPLATE_STEPS = [
    0,
    10,
    25,
    50,
    100,
    250,
    500,
    1000,
    2500,
    5000,
    10000,
    15000,
    20000,
    25000,
    27500,
    30000,
    32500,
    35000,
    37500,
    40000,
    45000,
    50000,
    60000,
    70000,
    76970,
]
GRID_TEMPLATE_MAX = GRID_TEMPLATE_STEPS[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect raw per-seed baseline, WDRO, and CDRO data on a shared weighted-compute grid. "
            "All three methods use single fixed-horizon trajectories; aggregation is deferred to later analysis."
        )
    )
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--prefix", type=str, required=True)
    parser.add_argument("--python-bin", type=str, default=os.path.join(ROOT_DIR, ".venv", "bin", "python"))
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dataset-path", type=str, default=DEFAULT_TRAIN_ROOT)
    parser.add_argument("--dataset-val-path", type=str, default=DEFAULT_VAL_ROOT)
    parser.add_argument("--fid-ref-path", type=str, default=DEFAULT_FID_REF)
    parser.add_argument("--weighted-compute-calibration-path", type=str, default=DEFAULT_CALIBRATION)
    parser.add_argument("--weighted-inputgrad-alpha", type=float, default=0.0)
    parser.add_argument("--weighted-parambackward-beta", type=float, default=0.0)
    parser.add_argument("--train-accelerator-count", type=int, default=1)
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
    parser.add_argument("--n-steps-path", type=int, default=24)
    parser.add_argument("--sigma-min", type=float, default=0.002)
    parser.add_argument("--sigma-max", type=float, default=2.0)
    parser.add_argument("--shared-weighted-cap", type=float, default=0.0)
    parser.add_argument("--baseline-max-steps", type=int, default=80000)
    parser.add_argument("--wdro-max-total-steps", type=int, default=76970)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--wdro-warmup-fraction", type=float, default=0.2)
    parser.add_argument("--wdro-refresh-epochs", type=float, default=100.0)
    parser.add_argument("--wdro-adv-prob", type=float, default=0.3)
    parser.add_argument("--wdro-attack-steps", type=int, default=2)
    parser.add_argument("--wdro-attack-step-size", type=float, default=1e-3)
    parser.add_argument("--wdro-gamma", type=float, default=1.0)
    parser.add_argument("--inner-steps", type=int, default=1)
    parser.add_argument("--outer-attack-weight", type=float, default=0.5)
    parser.add_argument("--outer-clean-weight", type=float, default=1.0)
    parser.add_argument("--cdro-step-size", type=float, default=0.02)
    parser.add_argument("--cdro-total-budget-rho", type=float, default=0.02)
    parser.add_argument("--cdro-time-horizon", type=float, default=1.0)
    parser.add_argument("--cdro-warmup-fraction", type=float, default=0.2)
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_int_list(text: str) -> List[int]:
    values = sorted({int(tok.strip()) for tok in text.split(",") if tok.strip()})
    if not values:
        raise ValueError("Expected at least one seed.")
    return values


def format_steps_list(steps: Iterable[int]) -> str:
    return ",".join(str(int(step)) for step in steps)


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        return
    ensure_dir(os.path.dirname(path))
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_command(*, cmd: List[str], log_path: str) -> None:
    ensure_dir(os.path.dirname(log_path))
    with open(log_path, "w", encoding="utf-8") as handle:
        subprocess.run(cmd, cwd=ROOT_DIR, check=True, stdout=handle, stderr=subprocess.STDOUT)


def realize_shared_weighted_grid(shared_weighted_cap: float) -> List[float]:
    cap = max(float(shared_weighted_cap), 0.0)
    if cap <= 0.0:
        return [0.0]
    out: List[float] = []
    previous = -1.0
    for base_step in GRID_TEMPLATE_STEPS:
        scaled = float(base_step) * cap / float(GRID_TEMPLATE_MAX)
        if scaled <= previous:
            scaled = previous + 1e-6
        out.append(float(scaled))
        previous = float(scaled)
    if out[-1] != cap:
        out[-1] = cap
    return out


def _baseline_step_weighted_units(calibration: Dict) -> float:
    units = baseline_weighted_compute_units_for_steps(steps=1, calibration=calibration)
    if units is None:
        raise RuntimeError("Missing weighted-compute calibration for baseline steps.")
    return float(units)


def _wdro_expected_robust_step_weighted_units(args: argparse.Namespace, calibration: Dict) -> float:
    units = wdro_robust_step_weighted_compute_units(
        batch_size=int(args.batch_size),
        train_pool_size=int(args.image_train_size),
        refresh_epochs=float(args.wdro_refresh_epochs),
        adv_prob=float(args.wdro_adv_prob),
        attack_steps=int(args.wdro_attack_steps),
        calibration=calibration,
    )
    if units is None:
        raise RuntimeError("Missing weighted-compute calibration for WDRO robust steps.")
    return float(units)


def _cdro_expected_robust_step_weighted_units(args: argparse.Namespace, calibration: Dict) -> float:
    units = cdro_robust_step_weighted_compute_units(
        n_steps_path=int(args.n_steps_path),
        inner_steps=int(args.inner_steps),
        outer_attack_weight=float(args.outer_attack_weight),
        outer_clean_weight=float(args.outer_clean_weight),
        calibration=calibration,
    )
    if units is None:
        raise RuntimeError("Missing weighted-compute calibration for CDRO robust steps.")
    return float(units)


def _wdro_reference_warmup_weighted_fraction(
    *,
    total_steps: int,
    warmup_fraction: float,
    baseline_step_weighted_units: float,
    wdro_robust_step_weighted_units: float,
) -> float:
    total_steps_value = max(int(total_steps), 0)
    if total_steps_value <= 0:
        return 0.0
    warmup_steps = max(0, min(int(total_steps_value * float(warmup_fraction)), total_steps_value))
    robust_steps = max(total_steps_value - warmup_steps, 0)
    baseline_weighted = float(warmup_steps) * float(baseline_step_weighted_units)
    robust_weighted = float(robust_steps) * float(wdro_robust_step_weighted_units)
    total_weighted = baseline_weighted + robust_weighted
    return 0.0 if total_weighted <= 0.0 else float(baseline_weighted / total_weighted)


def _cdro_fixed_warmup_steps(
    *,
    total_steps: int,
    args: argparse.Namespace,
    baseline_step_weighted_units: float,
    wdro_robust_step_weighted_units: float,
    cdro_robust_step_weighted_units: float,
) -> int:
    target_fraction = _wdro_reference_warmup_weighted_fraction(
        total_steps=int(total_steps),
        warmup_fraction=float(args.cdro_warmup_fraction),
        baseline_step_weighted_units=float(baseline_step_weighted_units),
        wdro_robust_step_weighted_units=float(wdro_robust_step_weighted_units),
    )
    return solve_warmup_steps_for_target_compute_fraction(
        total_steps=int(total_steps),
        target_warmup_compute_fraction=float(target_fraction),
        baseline_step_compute_units=float(baseline_step_weighted_units),
        robust_step_compute_units=float(cdro_robust_step_weighted_units),
    )


def _piecewise_weighted_prefix(
    *,
    total_steps_prefix: int,
    fixed_warmup_steps: int,
    baseline_step_weighted_units: float,
    robust_step_weighted_units: float,
) -> float:
    prefix = max(int(total_steps_prefix), 0)
    warmup = max(0, min(int(fixed_warmup_steps), prefix))
    robust = max(prefix - warmup, 0)
    return float(warmup) * float(baseline_step_weighted_units) + float(robust) * float(robust_step_weighted_units)


def _solve_prefix_steps_for_target_weighted(
    *,
    target_weighted_units: float,
    max_total_steps: int,
    fixed_warmup_steps: int,
    baseline_step_weighted_units: float,
    robust_step_weighted_units: float,
) -> int:
    target = max(float(target_weighted_units), 0.0)
    lo, hi = 0, max(int(max_total_steps), 0)
    while lo < hi:
        mid = (lo + hi) // 2
        value = _piecewise_weighted_prefix(
            total_steps_prefix=int(mid),
            fixed_warmup_steps=int(fixed_warmup_steps),
            baseline_step_weighted_units=float(baseline_step_weighted_units),
            robust_step_weighted_units=float(robust_step_weighted_units),
        )
        if value < target:
            lo = mid + 1
        else:
            hi = mid
    return int(lo)


def _solve_baseline_steps_for_target_weighted(
    *,
    target_weighted_units: float,
    baseline_step_weighted_units: float,
    max_total_steps: int,
) -> int:
    if baseline_step_weighted_units <= 0.0:
        return 0
    step = int(round(float(target_weighted_units) / float(baseline_step_weighted_units)))
    return max(0, min(step, int(max_total_steps)))


def _solve_cdro_total_steps_for_shared_cap(
    *,
    shared_weighted_cap: float,
    args: argparse.Namespace,
    baseline_step_weighted_units: float,
    wdro_robust_step_weighted_units: float,
    cdro_robust_step_weighted_units: float,
) -> Tuple[int, int, float]:
    target = max(float(shared_weighted_cap), 0.0)
    lo, hi = 0, max(int(args.wdro_max_total_steps), 1)
    while True:
        warmup = _cdro_fixed_warmup_steps(
            total_steps=int(hi),
            args=args,
            baseline_step_weighted_units=float(baseline_step_weighted_units),
            wdro_robust_step_weighted_units=float(wdro_robust_step_weighted_units),
            cdro_robust_step_weighted_units=float(cdro_robust_step_weighted_units),
        )
        weighted = _piecewise_weighted_prefix(
            total_steps_prefix=int(hi),
            fixed_warmup_steps=int(warmup),
            baseline_step_weighted_units=float(baseline_step_weighted_units),
            robust_step_weighted_units=float(cdro_robust_step_weighted_units),
        )
        if weighted >= target:
            break
        lo = hi
        hi *= 2
        if hi > 1_000_000:
            raise RuntimeError("Could not bracket CDRO total steps for the shared weighted cap.")
    while lo < hi:
        mid = (lo + hi) // 2
        warmup = _cdro_fixed_warmup_steps(
            total_steps=int(mid),
            args=args,
            baseline_step_weighted_units=float(baseline_step_weighted_units),
            wdro_robust_step_weighted_units=float(wdro_robust_step_weighted_units),
            cdro_robust_step_weighted_units=float(cdro_robust_step_weighted_units),
        )
        weighted = _piecewise_weighted_prefix(
            total_steps_prefix=int(mid),
            fixed_warmup_steps=int(warmup),
            baseline_step_weighted_units=float(baseline_step_weighted_units),
            robust_step_weighted_units=float(cdro_robust_step_weighted_units),
        )
        if weighted < target:
            lo = mid + 1
        else:
            hi = mid
    total_steps = int(lo)
    warmup_steps = _cdro_fixed_warmup_steps(
        total_steps=int(total_steps),
        args=args,
        baseline_step_weighted_units=float(baseline_step_weighted_units),
        wdro_robust_step_weighted_units=float(wdro_robust_step_weighted_units),
        cdro_robust_step_weighted_units=float(cdro_robust_step_weighted_units),
    )
    weighted_total = _piecewise_weighted_prefix(
        total_steps_prefix=int(total_steps),
        fixed_warmup_steps=int(warmup_steps),
        baseline_step_weighted_units=float(baseline_step_weighted_units),
        robust_step_weighted_units=float(cdro_robust_step_weighted_units),
    )
    return int(total_steps), int(warmup_steps), float(weighted_total)


def _normalize_baseline_runs(*, runs_csv: str) -> List[Dict]:
    rows = load_csv_rows(runs_csv)
    out: List[Dict] = []
    for row in rows:
        out.append(
            {
                "method": "baseline_edm",
                "seed": int(row["seed"]),
                "step": int(row["step"]),
                "fid": float(row["baseline_fid"]),
                "train_wall_clock_sec": float(row["train_wall_clock_sec"]),
                "weighted_compute_units": float(row["weighted_compute_units"]),
                "batch_equiv_denoiser_evals": float(row["batch_equiv_denoiser_evals"]),
                "images_shown_m": float(row["images_shown_m"]),
                "row_origin": "baseline_seed_run",
                "source_csv": runs_csv,
                "exp_name": row["exp_name"],
                "exp_dir": row["exp_dir"],
                "checkpoint_path": row["checkpoint_path"],
            }
        )
    return out


def _extract_wdro_row(
    *,
    metrics_path: str,
    calibration: Dict,
    train_accelerator_count: int,
) -> Dict:
    payload = load_json(metrics_path)
    metrics = payload["metrics"]
    flow = metrics["flow_debug"]
    runtime = flow["runtime"]
    sample_quality = metrics.get("sample_quality_debug", {})
    objective = metrics["objective_debug"]
    batch_size = int(flow["budget_accounting"]["batch_size"])
    baseline_phase_steps = int(flow["baseline_phase_steps"])
    robust_phase_steps = int(flow["robust_phase_steps"])
    total_steps_requested = int(flow["total_steps_requested"])
    compute_accounting = flow["compute_accounting"]
    budget_accounting = flow["budget_accounting"]
    robust_compute_be_raw = float(compute_accounting["robust_batch_equiv_denoiser_evals_total"])
    baseline_compute_be_raw = float(compute_accounting["baseline_batch_equiv_denoiser_evals_total"])
    total_compute_be_effective = compute_accounting.get("effective_train_batch_equiv_denoiser_evals_total")
    if total_compute_be_effective is None:
        baseline_compute_be_effective = float(max(baseline_compute_be_raw, float(baseline_phase_steps)))
        total_compute_be_effective = float(baseline_compute_be_effective + robust_compute_be_raw)
    else:
        total_compute_be_effective = float(total_compute_be_effective)
        baseline_compute_be_effective = float(max(total_compute_be_effective - robust_compute_be_raw, 0.0))
    baseline_weighted_compute = compute_accounting.get("baseline_weighted_compute_units")
    robust_weighted_compute = compute_accounting.get("robust_weighted_compute_units")
    total_weighted_compute = compute_accounting.get("weighted_compute_units", runtime.get("weighted_compute_units"))
    train_wall_clock_sec = compute_accounting.get("train_wall_clock_sec", runtime.get("train_wall_clock_sec"))
    if train_wall_clock_sec is None:
        train_wall_clock_sec = runtime.get("effective_train_total")
    train_gpu_hours = compute_accounting.get("train_gpu_hours", runtime.get("train_gpu_hours"))
    if train_gpu_hours is None and train_wall_clock_sec is not None:
        train_gpu_hours = float(train_wall_clock_sec) * float(max(int(train_accelerator_count), 0)) / 3600.0
    baseline_train_wall_clock_sec = runtime.get("baseline_train_wall_clock_sec_effective")
    robust_train_wall_clock_sec = runtime.get("robust_phase")
    effective_images_seen_total = budget_accounting.get("effective_train_images_seen_total")
    if effective_images_seen_total is None:
        total_images_shown_m_effective = float(total_steps_requested * batch_size) / 1_000_000.0
    else:
        total_images_shown_m_effective = float(effective_images_seen_total) / 1_000_000.0
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
        "method": "wdro",
        "step": int(total_steps_requested),
        "compute_budget_be": float(total_compute_be_effective),
        "baseline_compute_be": float(baseline_compute_be_effective),
        "robust_compute_be": float(robust_compute_be_raw),
        "weighted_compute_units": None if total_weighted_compute is None else float(total_weighted_compute),
        "baseline_weighted_compute_units": (
            None if baseline_weighted_compute is None else float(baseline_weighted_compute)
        ),
        "robust_weighted_compute_units": None if robust_weighted_compute is None else float(robust_weighted_compute),
        "images_shown_m": float(total_images_shown_m_effective),
        "fid": float(fid_value),
        "metrics_path": metrics_path,
        "train_wall_clock_sec": None if train_wall_clock_sec is None else float(train_wall_clock_sec),
        "train_elapsed_sec": None if train_wall_clock_sec is None else float(train_wall_clock_sec),
        "train_gpu_hours": None if train_gpu_hours is None else float(train_gpu_hours),
        "baseline_train_wall_clock_sec_effective": (
            None if baseline_train_wall_clock_sec is None else float(baseline_train_wall_clock_sec)
        ),
        "robust_train_wall_clock_sec_effective": (
            None if robust_train_wall_clock_sec is None else float(robust_train_wall_clock_sec)
        ),
        "baseline_train_wall_clock_source": runtime.get("baseline_train_wall_clock_sec_effective_source"),
        "warmup_steps_fixed": int(baseline_phase_steps),
        "robust_steps_observed": int(robust_phase_steps),
        "warmup_only": bool(warmup_only),
        "wdro_refresh_steps": objective.get("wdro_refresh_steps", []),
        "wdro_dataset_sizes": objective.get("wdro_dataset_sizes", []),
        "wdro_final_dataset_size": (
            int(objective["wdro_dataset_sizes"][-1]) if objective.get("wdro_dataset_sizes") else 0
        ),
        "baseline_ckpt_enabled": bool(flow.get("baseline_ckpt_enabled", False)),
        "baseline_ckpt_loaded": bool(flow.get("baseline_ckpt_loaded", False)),
        "robust_resume_loaded": bool(flow.get("robust_resume_loaded", False)),
        "phase_step_split_mode": str(flow["phase_step_split_mode"]),
        "train_wall_clock_complete": bool(
            compute_accounting.get("train_wall_clock_complete", train_wall_clock_sec is not None)
        ),
    }


def _extract_cdro_row(
    *,
    metrics_path: str,
    baseline_ckpt_requested: Optional[str],
    calibration: Dict,
    train_accelerator_count: int,
) -> Dict:
    payload = load_json(metrics_path)
    metrics = payload["metrics"]
    flow = metrics["flow_debug"]
    runtime = flow["runtime"]
    sample_quality = metrics.get("sample_quality_debug", {})
    batch_size = int(flow["budget_accounting"]["batch_size"])
    baseline_phase_steps = int(flow["baseline_phase_steps"])
    robust_phase_steps = int(flow["robust_phase_steps"])
    total_steps_requested = int(flow["total_steps_requested"])
    compute_accounting = flow["compute_accounting"]
    budget_accounting = flow["budget_accounting"]
    robust_compute_be_raw = float(compute_accounting["robust_batch_equiv_denoiser_evals_total"])
    baseline_compute_be_raw = float(compute_accounting["baseline_batch_equiv_denoiser_evals_total"])
    total_compute_be_effective = compute_accounting.get("effective_train_batch_equiv_denoiser_evals_total")
    if total_compute_be_effective is None:
        baseline_compute_be_effective = float(max(baseline_compute_be_raw, float(baseline_phase_steps)))
        total_compute_be_effective = float(baseline_compute_be_effective + robust_compute_be_raw)
    else:
        total_compute_be_effective = float(total_compute_be_effective)
        baseline_compute_be_effective = float(max(total_compute_be_effective - robust_compute_be_raw, 0.0))
    baseline_weighted_compute = compute_accounting.get("baseline_weighted_compute_units")
    robust_weighted_compute = compute_accounting.get("robust_weighted_compute_units")
    total_weighted_compute = compute_accounting.get("weighted_compute_units", runtime.get("weighted_compute_units"))
    train_wall_clock_sec = compute_accounting.get("train_wall_clock_sec", runtime.get("train_wall_clock_sec"))
    if train_wall_clock_sec is None:
        train_wall_clock_sec = runtime.get("effective_train_total")
    train_gpu_hours = compute_accounting.get("train_gpu_hours", runtime.get("train_gpu_hours"))
    if train_gpu_hours is None and train_wall_clock_sec is not None:
        train_gpu_hours = float(train_wall_clock_sec) * float(max(int(train_accelerator_count), 0)) / 3600.0
    baseline_train_wall_clock_sec = runtime.get("baseline_train_wall_clock_sec_effective")
    robust_train_wall_clock_sec = runtime.get("robust_phase")
    effective_images_seen_total = budget_accounting.get("effective_train_images_seen_total")
    if effective_images_seen_total is None:
        total_images_shown_m_effective = float(total_steps_requested * batch_size) / 1_000_000.0
    else:
        total_images_shown_m_effective = float(effective_images_seen_total) / 1_000_000.0
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
        "compute_budget_be": float(total_compute_be_effective),
        "baseline_compute_be": float(baseline_compute_be_effective),
        "robust_compute_be": float(robust_compute_be_raw),
        "weighted_compute_units": None if total_weighted_compute is None else float(total_weighted_compute),
        "baseline_weighted_compute_units": (
            None if baseline_weighted_compute is None else float(baseline_weighted_compute)
        ),
        "robust_weighted_compute_units": None if robust_weighted_compute is None else float(robust_weighted_compute),
        "images_shown_m": float(total_images_shown_m_effective),
        "fid": float(fid_value),
        "metrics_path": metrics_path,
        "train_wall_clock_sec": None if train_wall_clock_sec is None else float(train_wall_clock_sec),
        "train_elapsed_sec": None if train_wall_clock_sec is None else float(train_wall_clock_sec),
        "train_gpu_hours": None if train_gpu_hours is None else float(train_gpu_hours),
        "baseline_train_wall_clock_sec_effective": (
            None if baseline_train_wall_clock_sec is None else float(baseline_train_wall_clock_sec)
        ),
        "robust_train_wall_clock_sec_effective": (
            None if robust_train_wall_clock_sec is None else float(robust_train_wall_clock_sec)
        ),
        "baseline_train_wall_clock_source": runtime.get("baseline_train_wall_clock_sec_effective_source"),
        "warmup_steps_fixed": int(baseline_phase_steps),
        "robust_steps_observed": int(robust_phase_steps),
        "warmup_only": bool(warmup_only),
        "baseline_ckpt_requested": baseline_ckpt_requested,
        "baseline_ckpt_enabled": bool(flow.get("baseline_ckpt_enabled", False)),
        "baseline_ckpt_loaded": bool(flow.get("baseline_ckpt_loaded", False)),
        "robust_resume_loaded": bool(flow.get("robust_resume_loaded", False)),
        "phase_step_split_mode": str(flow["phase_step_split_mode"]),
        "train_wall_clock_complete": bool(
            compute_accounting.get("train_wall_clock_complete", train_wall_clock_sec is not None)
        ),
    }


def _normalize_method_metrics_row(
    *,
    row: Dict,
    seed: int,
    method_name: str,
    row_origin: str,
    weighted_grid_target: float,
    trajectory_total_steps_max: int,
    fixed_warmup_steps: int,
) -> Dict:
    out = dict(row)
    out["method"] = str(method_name)
    out["seed"] = int(seed)
    out["row_origin"] = str(row_origin)
    out["weighted_grid_target"] = float(weighted_grid_target)
    out["trajectory_total_steps_max"] = int(trajectory_total_steps_max)
    out["fixed_warmup_steps"] = int(fixed_warmup_steps)
    return out


def _transfer_baseline_row_to_method(
    *,
    baseline_row: Dict,
    method_name: str,
    weighted_grid_target: float,
    trajectory_total_steps_max: int,
    fixed_warmup_steps: int,
) -> Dict:
    row = dict(baseline_row)
    row["method"] = str(method_name)
    row["row_origin"] = "baseline_warmup_transfer"
    row["weighted_grid_target"] = float(weighted_grid_target)
    row["trajectory_total_steps_max"] = int(trajectory_total_steps_max)
    row["fixed_warmup_steps"] = int(fixed_warmup_steps)
    row["warmup_only"] = True
    row["baseline_ckpt_loaded"] = True
    row["robust_resume_loaded"] = False
    row["phase_step_split_mode"] = "single_trajectory_fixed_warmup_baseline_transfer"
    return row


def _build_run_toy_cmd(
    *,
    args: argparse.Namespace,
    method_name: str,
    outdir: str,
    exp_name: str,
    seed: int,
    total_steps: int,
    fixed_warmup_steps: int,
    resume_path: str,
    baseline_ckpt_path: Optional[str],
    use_resume: bool,
) -> List[str]:
    cmd = [
        args.python_bin,
        os.path.join(ROOT_DIR, "toy", "run_toy.py"),
        "--exp-name",
        exp_name,
        "--outdir",
        outdir,
        "--device",
        args.device,
        "--seed",
        str(seed),
        "--steps",
        str(int(total_steps)),
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
        method_name,
        "--disable-baseline-gate",
        "--skip-checks",
        "--baseline-steps-override",
        str(int(fixed_warmup_steps)),
        "--disable-baseline-ckpt",
        "--weighted-compute-calibration-path",
        str(args.weighted_compute_calibration_path).strip(),
        "--train-accelerator-count",
        str(args.train_accelerator_count),
    ]
    if float(args.weighted_inputgrad_alpha) > 0.0 and float(args.weighted_parambackward_beta) > 0.0:
        cmd.extend(
            [
                "--weighted-inputgrad-alpha",
                str(args.weighted_inputgrad_alpha),
                "--weighted-parambackward-beta",
                str(args.weighted_parambackward_beta),
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
    if use_resume and os.path.exists(resume_path):
        cmd.extend(["--robust-resume-ckpt-path", resume_path])
    cmd.extend(["--robust-save-ckpt-path", resume_path])
    if method_name == "wdro":
        cmd.extend(
            [
                "--wdro-warmup-fraction",
                str(args.wdro_warmup_fraction),
                "--wdro-refresh-epochs",
                str(args.wdro_refresh_epochs),
                "--wdro-adv-prob",
                str(args.wdro_adv_prob),
                "--wdro-attack-steps",
                str(args.wdro_attack_steps),
                "--wdro-attack-step-size",
                str(args.wdro_attack_step_size),
                "--wdro-gamma",
                str(args.wdro_gamma),
            ]
        )
    elif method_name == "cdro":
        cmd.extend(
            [
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
                "--cdro-warmup-fraction",
                str(args.cdro_warmup_fraction),
                "--disable-collapse-diagnostics",
            ]
        )
    else:
        raise ValueError(f"Unsupported method_name: {method_name}")
    return cmd


def main() -> None:
    args = parse_args()
    seeds = parse_int_list(args.seeds)
    ensure_dir(args.outdir)
    logs_dir = os.path.join(args.outdir, "logs")
    ensure_dir(logs_dir)

    calibration = load_weighted_compute_calibration(
        calibration_path=str(args.weighted_compute_calibration_path).strip(),
        inputgrad_alpha=float(args.weighted_inputgrad_alpha),
        parambackward_beta=float(args.weighted_parambackward_beta),
    )
    if not calibration.get("available", False):
        raise RuntimeError("Weighted-compute calibration is required for this weighted-grid collector.")

    baseline_step_weighted_units = _baseline_step_weighted_units(calibration)
    wdro_robust_step_weighted_units = _wdro_expected_robust_step_weighted_units(args, calibration)
    cdro_robust_step_weighted_units = _cdro_expected_robust_step_weighted_units(args, calibration)

    wdro_fixed_warmup_steps = int(int(args.wdro_max_total_steps) * float(args.wdro_warmup_fraction))
    wdro_shared_cap = _piecewise_weighted_prefix(
        total_steps_prefix=int(args.wdro_max_total_steps),
        fixed_warmup_steps=int(wdro_fixed_warmup_steps),
        baseline_step_weighted_units=float(baseline_step_weighted_units),
        robust_step_weighted_units=float(wdro_robust_step_weighted_units),
    )
    baseline_cap_from_limit = float(int(args.baseline_max_steps)) * float(baseline_step_weighted_units)
    shared_weighted_cap = (
        min(baseline_cap_from_limit, wdro_shared_cap)
        if float(args.shared_weighted_cap) <= 0.0
        else min(float(args.shared_weighted_cap), baseline_cap_from_limit, wdro_shared_cap)
    )

    cdro_max_total_steps, cdro_fixed_warmup_steps, cdro_shared_cap = _solve_cdro_total_steps_for_shared_cap(
        shared_weighted_cap=float(shared_weighted_cap),
        args=args,
        baseline_step_weighted_units=float(baseline_step_weighted_units),
        wdro_robust_step_weighted_units=float(wdro_robust_step_weighted_units),
        cdro_robust_step_weighted_units=float(cdro_robust_step_weighted_units),
    )
    shared_weighted_cap = min(float(shared_weighted_cap), float(cdro_shared_cap))

    weighted_grid_targets = realize_shared_weighted_grid(float(shared_weighted_cap))
    baseline_curve_steps = sorted(
        {
            _solve_baseline_steps_for_target_weighted(
                target_weighted_units=float(target),
                baseline_step_weighted_units=float(baseline_step_weighted_units),
                max_total_steps=int(args.baseline_max_steps),
            )
            for target in weighted_grid_targets
        }
    )
    wdro_curve_steps = sorted(
        {
            _solve_prefix_steps_for_target_weighted(
                target_weighted_units=float(target),
                max_total_steps=int(args.wdro_max_total_steps),
                fixed_warmup_steps=int(wdro_fixed_warmup_steps),
                baseline_step_weighted_units=float(baseline_step_weighted_units),
                robust_step_weighted_units=float(wdro_robust_step_weighted_units),
            )
            for target in weighted_grid_targets
        }
    )
    cdro_curve_steps = sorted(
        {
            _solve_prefix_steps_for_target_weighted(
                target_weighted_units=float(target),
                max_total_steps=int(cdro_max_total_steps),
                fixed_warmup_steps=int(cdro_fixed_warmup_steps),
                baseline_step_weighted_units=float(baseline_step_weighted_units),
                robust_step_weighted_units=float(cdro_robust_step_weighted_units),
            )
            for target in weighted_grid_targets
        }
    )
    baseline_run_steps = sorted(
        {
            step
            for step in baseline_curve_steps
            + [int(wdro_fixed_warmup_steps), int(cdro_fixed_warmup_steps)]
            + [step for step in wdro_curve_steps if step <= int(wdro_fixed_warmup_steps)]
            + [step for step in cdro_curve_steps if step <= int(cdro_fixed_warmup_steps)]
            if step > 0
        }
    )

    print(
        "[collect-weighted] shared_cap="
        f"{shared_weighted_cap:.4f} baseline_cap_limit={baseline_cap_from_limit:.4f} "
        f"wdro_cap={wdro_shared_cap:.4f} cdro_cap={cdro_shared_cap:.4f}",
        flush=True,
    )
    print(
        "[collect-weighted] wdro "
        f"max_total_steps={int(args.wdro_max_total_steps)} fixed_warmup_steps={int(wdro_fixed_warmup_steps)}",
        flush=True,
    )
    print(
        "[collect-weighted] cdro "
        f"max_total_steps={int(cdro_max_total_steps)} fixed_warmup_steps={int(cdro_fixed_warmup_steps)}",
        flush=True,
    )
    print(f"[collect-weighted] weighted_targets={weighted_grid_targets}", flush=True)
    print(f"[collect-weighted] baseline_steps={baseline_run_steps}", flush=True)
    print(f"[collect-weighted] wdro_steps={wdro_curve_steps}", flush=True)
    print(f"[collect-weighted] cdro_steps={cdro_curve_steps}", flush=True)

    baseline_outdir = os.path.join(args.outdir, "baseline")
    wdro_root = os.path.join(args.outdir, "wdro")
    cdro_root = os.path.join(args.outdir, "cdro")
    ensure_dir(baseline_outdir)
    ensure_dir(wdro_root)
    ensure_dir(cdro_root)

    baseline_prefix = f"{args.prefix}_baseline"
    baseline_runs_csv = os.path.join(baseline_outdir, f"{baseline_prefix}_runs.csv")
    baseline_agg_csv = os.path.join(baseline_outdir, f"{baseline_prefix}_aggregate.csv")
    baseline_log = os.path.join(logs_dir, f"{baseline_prefix}.log")
    baseline_cmd = [
        args.python_bin,
        os.path.join(ROOT_DIR, "toy", "scripts", "sweep_mnist_convergence_checkpointed.py"),
        "--outdir",
        baseline_outdir,
        "--prefix",
        baseline_prefix,
        "--seeds",
        args.seeds,
        "--train-percents",
        "1",
        "--steps-list",
        format_steps_list(baseline_run_steps),
        "--device",
        args.device,
        "--require-cuda",
        "--dataset-kind",
        "image_folder",
        "--dataset-path",
        args.dataset_path,
        "--dataset-val-path",
        args.dataset_val_path,
        "--image-size",
        str(args.image_size),
        "--image-channels",
        str(args.image_channels),
        "--batch-size",
        str(args.batch_size),
        "--hidden-dim",
        str(args.hidden_dim),
        "--n-steps-path",
        str(args.n_steps_path),
        "--sigma-min",
        str(args.sigma_min),
        "--sigma-max",
        str(args.sigma_max),
        "--fid-ref-path",
        args.fid_ref_path,
        "--fid-samples",
        str(args.eval_samples),
        "--weighted-compute-calibration-path",
        str(args.weighted_compute_calibration_path).strip(),
        "--train-accelerator-count",
        str(args.train_accelerator_count),
    ]
    if float(args.weighted_inputgrad_alpha) > 0.0 and float(args.weighted_parambackward_beta) > 0.0:
        baseline_cmd.extend(
            [
                "--weighted-inputgrad-alpha",
                str(args.weighted_inputgrad_alpha),
                "--weighted-parambackward-beta",
                str(args.weighted_parambackward_beta),
            ]
        )
    if args.skip_existing and os.path.isfile(baseline_runs_csv) and os.path.isfile(baseline_agg_csv):
        print(f"[collect-weighted] reuse baseline outputs: {baseline_outdir}", flush=True)
    else:
        print(f"[collect-weighted] baseline seeds={seeds} steps={baseline_run_steps}", flush=True)
        run_command(cmd=baseline_cmd, log_path=baseline_log)

    baseline_raw = _normalize_baseline_runs(runs_csv=baseline_runs_csv)
    baseline_raw.sort(key=lambda row: (int(row["seed"]), int(row["step"])))
    baseline_by_seed_step = {(int(row["seed"]), int(row["step"])): row for row in baseline_raw}
    baseline_ckpt_by_seed_step = {
        (int(row["seed"]), int(row["step"])): row.get("checkpoint_path", "") for row in baseline_raw if row.get("checkpoint_path")
    }
    weighted_target_by_baseline_step = {
        int(step): float(target) for step, target in zip(baseline_curve_steps, weighted_grid_targets)
    }
    for row in baseline_raw:
        row["weighted_grid_target"] = weighted_target_by_baseline_step.get(int(row["step"]))
        row["trajectory_total_steps_max"] = max((int(step) for step in baseline_curve_steps), default=0)
        row["fixed_warmup_steps"] = 0

    wdro_rows: List[Dict] = []
    cdro_rows: List[Dict] = []
    wdro_manifest_rows: List[Dict] = []
    cdro_manifest_rows: List[Dict] = []

    for seed in seeds:
        seed_baseline_warmup_ckpt = baseline_ckpt_by_seed_step.get((int(seed), int(wdro_fixed_warmup_steps)))
        if not seed_baseline_warmup_ckpt:
            raise RuntimeError(f"Missing baseline checkpoint for seed={seed} step={wdro_fixed_warmup_steps}")
        wdro_outdir = os.path.join(wdro_root, f"s{seed}")
        ensure_dir(wdro_outdir)
        wdro_resume_path = os.path.join(wdro_outdir, f"{args.prefix}_wdro_s{seed}_resume.pt")
        wdro_started_resume = False
        for target_weighted, total_steps in zip(weighted_grid_targets, wdro_curve_steps):
            if int(total_steps) <= int(wdro_fixed_warmup_steps):
                baseline_row = baseline_by_seed_step.get((int(seed), int(total_steps)))
                if baseline_row is None:
                    raise RuntimeError(f"Missing baseline raw row for seed={seed} step={total_steps}")
                wdro_rows.append(
                    _transfer_baseline_row_to_method(
                        baseline_row=baseline_row,
                        method_name="wdro",
                        weighted_grid_target=float(target_weighted),
                        trajectory_total_steps_max=int(args.wdro_max_total_steps),
                        fixed_warmup_steps=int(wdro_fixed_warmup_steps),
                    )
                )
                wdro_manifest_rows.append(
                    {
                        "seed": int(seed),
                        "step": int(total_steps),
                        "weighted_grid_target": float(target_weighted),
                        "row_origin": "baseline_warmup_transfer",
                    }
                )
                continue
            exp_name = f"{args.prefix}_wdro_s{seed}_st{int(total_steps)}"
            metrics_path = os.path.join(wdro_outdir, exp_name, "metrics.json")
            log_path = os.path.join(logs_dir, f"{exp_name}.log")
            if args.skip_existing and os.path.isfile(metrics_path):
                print(f"[collect-weighted] reuse wdro seed={seed} step={total_steps}: {metrics_path}", flush=True)
            else:
                print(
                    f"[collect-weighted] wdro seed={seed} step={total_steps} "
                    f"target_weighted={target_weighted:.4f} use_resume={wdro_started_resume}",
                    flush=True,
                )
                cmd = _build_run_toy_cmd(
                    args=args,
                    method_name="wdro",
                    outdir=wdro_outdir,
                    exp_name=exp_name,
                    seed=int(seed),
                    total_steps=int(total_steps),
                    fixed_warmup_steps=int(wdro_fixed_warmup_steps),
                    resume_path=wdro_resume_path,
                    baseline_ckpt_path=None if wdro_started_resume else seed_baseline_warmup_ckpt,
                    use_resume=wdro_started_resume,
                )
                run_command(cmd=cmd, log_path=log_path)
            row = _extract_wdro_row(
                metrics_path=metrics_path,
                calibration=calibration,
                train_accelerator_count=int(args.train_accelerator_count),
            )
            wdro_rows.append(
                _normalize_method_metrics_row(
                    row=row,
                    seed=int(seed),
                    method_name="wdro",
                    row_origin="method_metrics",
                    weighted_grid_target=float(target_weighted),
                    trajectory_total_steps_max=int(args.wdro_max_total_steps),
                    fixed_warmup_steps=int(wdro_fixed_warmup_steps),
                )
            )
            wdro_manifest_rows.append(
                {
                    "seed": int(seed),
                    "step": int(total_steps),
                    "weighted_grid_target": float(target_weighted),
                    "row_origin": "method_metrics",
                    "metrics_path": metrics_path,
                }
            )
            wdro_started_resume = True

        seed_baseline_warmup_ckpt = baseline_ckpt_by_seed_step.get((int(seed), int(cdro_fixed_warmup_steps)))
        if not seed_baseline_warmup_ckpt:
            raise RuntimeError(f"Missing baseline checkpoint for seed={seed} step={cdro_fixed_warmup_steps}")
        cdro_outdir = os.path.join(cdro_root, f"s{seed}")
        ensure_dir(cdro_outdir)
        cdro_resume_path = os.path.join(cdro_outdir, f"{args.prefix}_cdro_s{seed}_resume.pt")
        cdro_started_resume = False
        for target_weighted, total_steps in zip(weighted_grid_targets, cdro_curve_steps):
            if int(total_steps) <= int(cdro_fixed_warmup_steps):
                baseline_row = baseline_by_seed_step.get((int(seed), int(total_steps)))
                if baseline_row is None:
                    raise RuntimeError(f"Missing baseline raw row for seed={seed} step={total_steps}")
                cdro_rows.append(
                    _transfer_baseline_row_to_method(
                        baseline_row=baseline_row,
                        method_name="cdro",
                        weighted_grid_target=float(target_weighted),
                        trajectory_total_steps_max=int(cdro_max_total_steps),
                        fixed_warmup_steps=int(cdro_fixed_warmup_steps),
                    )
                )
                cdro_manifest_rows.append(
                    {
                        "seed": int(seed),
                        "step": int(total_steps),
                        "weighted_grid_target": float(target_weighted),
                        "row_origin": "baseline_warmup_transfer",
                    }
                )
                continue
            exp_name = f"{args.prefix}_cdro_s{seed}_st{int(total_steps)}"
            metrics_path = os.path.join(cdro_outdir, exp_name, "metrics.json")
            log_path = os.path.join(logs_dir, f"{exp_name}.log")
            if args.skip_existing and os.path.isfile(metrics_path):
                print(f"[collect-weighted] reuse cdro seed={seed} step={total_steps}: {metrics_path}", flush=True)
            else:
                print(
                    f"[collect-weighted] cdro seed={seed} step={total_steps} "
                    f"target_weighted={target_weighted:.4f} use_resume={cdro_started_resume}",
                    flush=True,
                )
                cmd = _build_run_toy_cmd(
                    args=args,
                    method_name="cdro",
                    outdir=cdro_outdir,
                    exp_name=exp_name,
                    seed=int(seed),
                    total_steps=int(total_steps),
                    fixed_warmup_steps=int(cdro_fixed_warmup_steps),
                    resume_path=cdro_resume_path,
                    baseline_ckpt_path=None if cdro_started_resume else seed_baseline_warmup_ckpt,
                    use_resume=cdro_started_resume,
                )
                run_command(cmd=cmd, log_path=log_path)
            row = _extract_cdro_row(
                metrics_path=metrics_path,
                baseline_ckpt_requested=None if cdro_started_resume else seed_baseline_warmup_ckpt,
                calibration=calibration,
                train_accelerator_count=int(args.train_accelerator_count),
            )
            cdro_rows.append(
                _normalize_method_metrics_row(
                    row=row,
                    seed=int(seed),
                    method_name="cdro",
                    row_origin="method_metrics",
                    weighted_grid_target=float(target_weighted),
                    trajectory_total_steps_max=int(cdro_max_total_steps),
                    fixed_warmup_steps=int(cdro_fixed_warmup_steps),
                )
            )
            cdro_manifest_rows.append(
                {
                    "seed": int(seed),
                    "step": int(total_steps),
                    "weighted_grid_target": float(target_weighted),
                    "row_origin": "method_metrics",
                    "metrics_path": metrics_path,
                }
            )
            cdro_started_resume = True

    wdro_rows.sort(key=lambda row: (int(row["seed"]), int(row["step"])))
    cdro_rows.sort(key=lambda row: (int(row["seed"]), int(row["step"])))
    combined_raw = baseline_raw + wdro_rows + cdro_rows
    combined_raw.sort(key=lambda row: (str(row["method"]), int(row["seed"]), int(row["step"])))

    baseline_raw_csv = os.path.join(args.outdir, f"{args.prefix}_baseline_raw_seed_rows.csv")
    wdro_raw_csv = os.path.join(args.outdir, f"{args.prefix}_wdro_raw_seed_rows.csv")
    cdro_raw_csv = os.path.join(args.outdir, f"{args.prefix}_cdro_raw_seed_rows.csv")
    combined_raw_csv = os.path.join(args.outdir, f"{args.prefix}_all_methods_raw_seed_rows.csv")
    write_csv(baseline_raw_csv, baseline_raw)
    write_csv(wdro_raw_csv, wdro_rows)
    write_csv(cdro_raw_csv, cdro_rows)
    write_csv(combined_raw_csv, combined_raw)

    manifest = {
        "protocol": {
            "name": "three_method_raw_seed_collection_weighted_grid",
            "description": (
                "Raw per-seed data collection on a shared weighted-compute grid. Baseline is a native "
                "checkpointed single trajectory. WDRO and CDRO use fixed-horizon trajectories with exact "
                "same-seed baseline checkpoint transfer for warmup bootstrap, then resume-chain only their "
                "own robust phase. Aggregation is intentionally deferred to later analysis."
            ),
            "seeds": seeds,
            "shared_grid_template_name": "wdro_dense_25_relative",
            "shared_grid_template_steps": GRID_TEMPLATE_STEPS,
            "grid_strategy": "25-knot relative template applied in weighted-compute space",
            "primary_metric": "weighted_compute_units",
            "secondary_metric": "train_wall_clock_sec",
            "legacy_metric": "batch_equiv_denoiser_evals",
            "baseline_aggregate_role": "summary_only_not_used_for_wdro_cdro_raw_collection",
            "same_seed_baseline_checkpoint_transfer": True,
            "cross_seed_checkpoint_reuse": False,
            "cross_method_baseline_cache_reuse": False,
            "cdro_collapse_diagnostics_disabled": True,
        },
        "weighted_compute": {
            "shared_cap": float(shared_weighted_cap),
            "grid_targets": weighted_grid_targets,
            "baseline_step_weighted_units": float(baseline_step_weighted_units),
            "wdro_expected_robust_step_weighted_units": float(wdro_robust_step_weighted_units),
            "cdro_expected_robust_step_weighted_units": float(cdro_robust_step_weighted_units),
            "calibration": calibration,
        },
        "trajectories": {
            "baseline": {
                "max_total_steps": max(baseline_run_steps) if baseline_run_steps else 0,
                "checkpoint_steps": baseline_run_steps,
            },
            "wdro": {
                "max_total_steps": int(args.wdro_max_total_steps),
                "fixed_warmup_steps": int(wdro_fixed_warmup_steps),
                "checkpoint_steps": wdro_curve_steps,
            },
            "cdro": {
                "max_total_steps": int(cdro_max_total_steps),
                "fixed_warmup_steps": int(cdro_fixed_warmup_steps),
                "checkpoint_steps": cdro_curve_steps,
            },
        },
        "artifacts": {
            "baseline_runs_csv": baseline_runs_csv,
            "baseline_aggregate_csv": baseline_agg_csv,
            "baseline_raw_csv": baseline_raw_csv,
            "wdro_raw_csv": wdro_raw_csv,
            "cdro_raw_csv": cdro_raw_csv,
            "combined_raw_csv": combined_raw_csv,
        },
        "per_seed_runs": {
            "wdro": wdro_manifest_rows,
            "cdro": cdro_manifest_rows,
        },
    }
    manifest_path = os.path.join(args.outdir, f"{args.prefix}_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"[collect-weighted] wrote {baseline_raw_csv}", flush=True)
    print(f"[collect-weighted] wrote {wdro_raw_csv}", flush=True)
    print(f"[collect-weighted] wrote {cdro_raw_csv}", flush=True)
    print(f"[collect-weighted] wrote {combined_raw_csv}", flush=True)
    print(f"[collect-weighted] wrote {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
