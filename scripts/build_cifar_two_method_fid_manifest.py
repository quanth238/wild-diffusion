#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import sys
from bisect import bisect_left
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from toy.compute_accounting import (  # noqa: E402
    load_weighted_compute_calibration,
    wdro_expected_attack_construction_units_per_step,
    weighted_compute_units,
)
from scripts.cifar_cdro_budget_utils import (  # noqa: E402
    DEFAULT_FLOP_CALIBRATION_JSON,
    baseline_step_flops,
    flop_calibration_from_path,
    flop_metadata_fields,
    hardware_flop_diagnostic_metadata_fields,
    wdro_robust_step_flops,
)


DEFAULT_BASELINE_RUN_DIR = (
    "/home/bachlc/GM-CDRO/training-runs/paper-cifar10-baseline-fp16/"
    "00001-cifar10-32x32-train20pct-seed0-uncond-ddpmpp-wdroedm-gpus1-batch1024-fp16-"
    "paper-cifar10-uncond-ddpmpp-baseline-20pct-fp16-bg512"
)
DEFAULT_WDRO_RUN_DIR = (
    "/home/bachlc/GM-CDRO/training-runs/paper-cifar10-wdro-fp16/"
    "00000-cifar10-32x32-train20pct-seed0-uncond-ddpmpp-wdroedm-gpus1-batch1024-fp16-"
    "paper-cifar10-uncond-ddpmpp-wdro-20pct-fp16-bg1024-resume040000"
)
DEFAULT_CALIBRATION_JSON = (
    "/home/bachlc/GM-CDRO/training-runs/compute_calibration/"
    "cifar10_32x32_ddpmpp_wdroedm_fp16_b1024_h100_20260414.json"
)
DEFAULT_OUTDIR = (
    "/home/bachlc/GM-CDRO/training-runs/fid-sweeps/cifar10_baseline_vs_wdro_coarse_20260414"
)
DEFAULT_BASELINE_KIMG = "1000,2000,5000,10000,20000,30000,40000,60000,80000,100000,120000,160000,200000"
DEFAULT_WDRO_KIMG = "40000,41000,45000,50000,60000,80000,100000,120000,140000,160000,180000,200000"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a coarse posthoc FID manifest for the CIFAR-10 baseline-vs-WDRO "
            "paper runs, including normalized baseline wall-clock and calibrated WCU."
        )
    )
    parser.add_argument("--baseline-run-dir", type=str, default=DEFAULT_BASELINE_RUN_DIR)
    parser.add_argument("--wdro-run-dir", type=str, default=DEFAULT_WDRO_RUN_DIR)
    parser.add_argument("--calibration-json", type=str, default=DEFAULT_CALIBRATION_JSON)
    parser.add_argument("--flop-calibration-json", type=str, default=DEFAULT_FLOP_CALIBRATION_JSON)
    parser.add_argument("--hardware-flop-diagnostic-json", type=str, default="")
    parser.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR)
    parser.add_argument("--manifest-name", type=str, default="cifar10_baseline_vs_wdro_coarse_manifest.csv")
    parser.add_argument("--summary-name", type=str, default="cifar10_baseline_vs_wdro_coarse_manifest_summary.json")
    parser.add_argument("--baseline-kimg", type=str, default=DEFAULT_BASELINE_KIMG)
    parser.add_argument("--wdro-kimg", type=str, default=DEFAULT_WDRO_KIMG)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--train-pool-size", type=int, default=10000)
    parser.add_argument("--warmup-kimg", type=int, default=40000)
    parser.add_argument("--wdro-refresh-epochs", type=float, default=20.0)
    parser.add_argument("--wdro-attack-steps", type=int, default=5)
    parser.add_argument("--wdro-adv-prob", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-percent-label", type=str, default="20%")
    parser.add_argument("--ref-path", type=str, default="/home/bachlc/GM-CDRO/datasets/fid-refs/cifar10-32x32.npz")
    return parser.parse_args()


def parse_kimg_list(text: str) -> List[int]:
    out: List[int] = []
    for chunk in str(text).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        value = int(chunk)
        if value <= 0:
            raise ValueError(f"kimg values must be positive, got {value}")
        out.append(value)
    if not out:
        raise ValueError("Expected at least one kimg value")
    return out


def _safe_mean_field(payload: Dict, key: str) -> float:
    value = payload[key]
    if isinstance(value, dict):
        value = value.get("mean")
    return float(value)


def _blank_if_none(value: object):
    return "" if value is None else value


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
    if x1 <= x0:
        return float(y1)
    frac = (float(target_kimg) - float(x0)) / (float(x1) - float(x0))
    return float(y0 + frac * (y1 - y0))


def ensure_snapshot_exists(run_dir: Path, kimg: int) -> Path:
    snapshot = run_dir / f"network-snapshot-{kimg:06d}.pkl"
    if not snapshot.is_file():
        raise FileNotFoundError(f"Missing checkpoint snapshot: {snapshot}")
    return snapshot


def _series_fields(*, robust_method: str, method_version_used: str) -> Dict[str, str]:
    robust_label = "Baseline" if robust_method == "baseline" else "Wild-Diffusion"
    return {
        "method": robust_method,
        "robust_method": robust_method,
        "robust_label": robust_label,
        "training_objective": "edm",
        "backbone_family": "edm",
        "backbone_label": "EDM",
        "series_key": f"{robust_method}_edm",
        "series_label": f"{robust_label} EDM",
        "method_version_used": method_version_used,
    }


def _baseline_weighted_units_for_kimg(*, kimg: int, batch_size: int, calibration: Dict) -> float:
    steps = float(kimg) * 1000.0 / float(batch_size)
    value = weighted_compute_units(
        n_fwd=0.0,
        n_fwd_inputgrad=0.0,
        n_fwd_parambackward=steps,
        calibration=calibration,
    )
    assert value is not None
    return float(value)


def _baseline_compute_be_for_kimg(*, kimg: int, batch_size: int) -> float:
    return float(kimg) * 1000.0 / float(batch_size)


def build_manifest_rows(args: argparse.Namespace) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    baseline_run_dir = Path(args.baseline_run_dir).resolve()
    wdro_run_dir = Path(args.wdro_run_dir).resolve()
    calibration = load_weighted_compute_calibration(calibration_path=args.calibration_json)
    if not calibration.get("available", False):
        raise RuntimeError(f"Weighted-compute calibration is unavailable: {args.calibration_json}")
    flop_calibration = flop_calibration_from_path(
        args.flop_calibration_json,
        batch_size=args.batch_size,
    )
    flop_metadata = flop_metadata_fields(flop_calibration)
    hardware_flop_metadata = hardware_flop_diagnostic_metadata_fields(args.hardware_flop_diagnostic_json)

    baseline_trace = load_stats_trace(baseline_run_dir / "stats.jsonl")
    wdro_trace = load_stats_trace(wdro_run_dir / "stats.jsonl")
    baseline_kimg_grid = parse_kimg_list(args.baseline_kimg)
    wdro_kimg_grid = parse_kimg_list(args.wdro_kimg)

    warmup_boundary_kimg = int(args.warmup_kimg)
    warmup_boundary_sec = interpolate_total_sec(baseline_trace, warmup_boundary_kimg)
    warmup_boundary_wcu = _baseline_weighted_units_for_kimg(
        kimg=warmup_boundary_kimg,
        batch_size=args.batch_size,
        calibration=calibration,
    )
    warmup_boundary_be = _baseline_compute_be_for_kimg(
        kimg=warmup_boundary_kimg,
        batch_size=args.batch_size,
    )
    baseline_step_train_flops = baseline_step_flops(calibration=flop_calibration)
    warmup_boundary_train_flops = (
        None if baseline_step_train_flops is None else float(warmup_boundary_be) * float(baseline_step_train_flops)
    )

    robust_attack_units_per_step = wdro_expected_attack_construction_units_per_step(
        batch_size=args.batch_size,
        train_pool_size=args.train_pool_size,
        refresh_epochs=args.wdro_refresh_epochs,
        adv_prob=args.wdro_adv_prob,
        attack_steps=args.wdro_attack_steps,
    )
    robust_units_per_step = weighted_compute_units(
        n_fwd=0.0,
        n_fwd_inputgrad=robust_attack_units_per_step,
        n_fwd_parambackward=1.0,
        calibration=calibration,
    )
    assert robust_units_per_step is not None
    robust_be_per_step = 1.0 + float(robust_attack_units_per_step)
    robust_flops_per_step = wdro_robust_step_flops(
        calibration=flop_calibration,
        attack_construction_units_per_step=robust_attack_units_per_step,
    )

    outdir = Path(args.outdir).resolve()
    eval_root = outdir / "evals"
    rows: List[Dict[str, object]] = []

    for kimg in baseline_kimg_grid:
        snapshot = ensure_snapshot_exists(baseline_run_dir, kimg)
        train_wall_clock_sec = interpolate_total_sec(baseline_trace, float(kimg))
        weighted_compute = _baseline_weighted_units_for_kimg(
            kimg=kimg,
            batch_size=args.batch_size,
            calibration=calibration,
        )
        compute_be = _baseline_compute_be_for_kimg(kimg=kimg, batch_size=args.batch_size)
        total_train_flops = (
            None if baseline_step_train_flops is None else float(compute_be) * float(baseline_step_train_flops)
        )
        eval_tag = f"baseline_kimg{kimg:06d}"
        rows.append(
            {
                **_series_fields(robust_method="baseline", method_version_used="baseline"),
                "seed": int(args.seed),
                "step": int(kimg),
                "snapshot_kimg": int(kimg),
                "images_shown_m": float(kimg) / 1000.0,
                "row_origin": "trajectory_warmup_phase",
                "run_dir": str(baseline_run_dir),
                "network_pkl": str(snapshot),
                "metrics_path": str(baseline_run_dir / "stats.jsonl"),
                "train_wall_clock_sec": float(train_wall_clock_sec),
                "train_elapsed_sec": float(train_wall_clock_sec),
                "train_gpu_hours": float(train_wall_clock_sec) / 3600.0,
                "train_wall_clock_source": "normalized_baseline_stats_jsonl",
                "baseline_train_wall_clock_sec_effective": "",
                "robust_train_wall_clock_sec_effective": "",
                "baseline_train_wall_clock_source": "",
                "weighted_compute_units": float(weighted_compute),
                "baseline_weighted_compute_units": float(weighted_compute),
                "robust_weighted_compute_units": 0.0,
                "compute_budget_be": float(compute_be),
                "baseline_compute_be": float(compute_be),
                "robust_compute_be": 0.0,
                "total_train_flops": _blank_if_none(None if total_train_flops is None else float(total_train_flops)),
                "total_train_tflops": _blank_if_none(
                    None if total_train_flops is None else float(total_train_flops) / 1e12
                ),
                "total_train_pflops": _blank_if_none(
                    None if total_train_flops is None else float(total_train_flops) / 1e15
                ),
                "baseline_train_flops": _blank_if_none(
                    None if total_train_flops is None else float(total_train_flops)
                ),
                "robust_train_flops": 0.0 if total_train_flops is not None else "",
                "train_flop_source": (
                    "flop_calibration_baseline"
                    if total_train_flops is not None
                    else ""
                ),
                **flop_metadata,
                **hardware_flop_metadata,
                "weighted_compute_source": "cifar_calibration_baseline",
                "warmup_steps_fixed": float(compute_be),
                "robust_steps_observed": 0.0,
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
            }
        )

    for kimg in wdro_kimg_grid:
        snapshot = ensure_snapshot_exists(wdro_run_dir, kimg)
        robust_kimg = max(int(kimg) - warmup_boundary_kimg, 0)
        robust_wall_clock_sec = 0.0 if robust_kimg <= 0 else interpolate_total_sec(wdro_trace, float(kimg))
        robust_steps = float(robust_kimg) * 1000.0 / float(args.batch_size)
        robust_weighted_compute = float(robust_units_per_step) * robust_steps
        robust_compute_be = float(robust_be_per_step) * robust_steps
        total_wall_clock_sec = float(warmup_boundary_sec + robust_wall_clock_sec)
        total_weighted_compute = float(warmup_boundary_wcu + robust_weighted_compute)
        total_compute_be = float(warmup_boundary_be + robust_compute_be)
        robust_train_flops = (
            None if robust_flops_per_step is None else float(robust_steps) * float(robust_flops_per_step)
        )
        total_train_flops = (
            None
            if warmup_boundary_train_flops is None or robust_train_flops is None
            else float(warmup_boundary_train_flops + robust_train_flops)
        )
        eval_tag = f"wild_diffusion_kimg{kimg:06d}"
        rows.append(
            {
                **_series_fields(robust_method="wild_diffusion", method_version_used="wdro"),
                "seed": int(args.seed),
                "step": int(kimg),
                "snapshot_kimg": int(kimg),
                "images_shown_m": float(kimg) / 1000.0,
                "row_origin": "trajectory_robust_phase",
                "run_dir": str(wdro_run_dir),
                "network_pkl": str(snapshot),
                "metrics_path": str(wdro_run_dir / "stats.jsonl"),
                "train_wall_clock_sec": float(total_wall_clock_sec),
                "train_elapsed_sec": float(total_wall_clock_sec),
                "train_gpu_hours": float(total_wall_clock_sec) / 3600.0,
                "train_wall_clock_source": "normalized_baseline_warmup_plus_observed_wdro_stats_jsonl",
                "baseline_train_wall_clock_sec_effective": float(warmup_boundary_sec),
                "robust_train_wall_clock_sec_effective": float(robust_wall_clock_sec),
                "baseline_train_wall_clock_source": "normalized_baseline_stats_jsonl",
                "weighted_compute_units": float(total_weighted_compute),
                "baseline_weighted_compute_units": float(warmup_boundary_wcu),
                "robust_weighted_compute_units": float(robust_weighted_compute),
                "compute_budget_be": float(total_compute_be),
                "baseline_compute_be": float(warmup_boundary_be),
                "robust_compute_be": float(robust_compute_be),
                "total_train_flops": _blank_if_none(None if total_train_flops is None else float(total_train_flops)),
                "total_train_tflops": _blank_if_none(
                    None if total_train_flops is None else float(total_train_flops) / 1e12
                ),
                "total_train_pflops": _blank_if_none(
                    None if total_train_flops is None else float(total_train_flops) / 1e15
                ),
                "baseline_train_flops": _blank_if_none(
                    None if warmup_boundary_train_flops is None else float(warmup_boundary_train_flops)
                ),
                "robust_train_flops": _blank_if_none(
                    None if robust_train_flops is None else float(robust_train_flops)
                ),
                "train_flop_source": (
                    "flop_calibration_expected_wdro"
                    if total_train_flops is not None
                    else ""
                ),
                **flop_metadata,
                **hardware_flop_metadata,
                "weighted_compute_source": "cifar_calibration_expected_wdro",
                "warmup_steps_fixed": float(warmup_boundary_be),
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
            }
        )

    rows.sort(key=lambda row: (str(row["series_key"]), int(row["step"])))
    summary = {
        "manifest_version": "cifar_two_method_fid_manifest_v1",
        "baseline_run_dir": str(baseline_run_dir),
        "wdro_run_dir": str(wdro_run_dir),
        "calibration_json": str(Path(args.calibration_json).resolve()),
        "flop_calibration_json": (
            str(Path(args.flop_calibration_json).resolve()) if str(args.flop_calibration_json).strip() else None
        ),
        "hardware_flop_diagnostic_json": (
            str(Path(args.hardware_flop_diagnostic_json).resolve())
            if str(args.hardware_flop_diagnostic_json).strip()
            else None
        ),
        "train_percent_label": str(args.train_percent_label),
        "seed": int(args.seed),
        "warmup_boundary_kimg": warmup_boundary_kimg,
        "warmup_boundary_train_wall_clock_sec": float(warmup_boundary_sec),
        "warmup_boundary_weighted_compute_units": float(warmup_boundary_wcu),
        "warmup_boundary_compute_be": float(warmup_boundary_be),
        "warmup_boundary_train_flops": (
            None if warmup_boundary_train_flops is None else float(warmup_boundary_train_flops)
        ),
        "wdro_attack_units_per_step": float(robust_attack_units_per_step),
        "wdro_weighted_units_per_step": float(robust_units_per_step),
        "wdro_compute_be_per_step": float(robust_be_per_step),
        "baseline_step_train_flops": (
            None if baseline_step_train_flops is None else float(baseline_step_train_flops)
        ),
        "wdro_train_flops_per_step": (
            None if robust_flops_per_step is None else float(robust_flops_per_step)
        ),
        "flop_metadata": flop_metadata,
        "hardware_flop_metadata": hardware_flop_metadata,
        "baseline_kimg_grid": baseline_kimg_grid,
        "wdro_kimg_grid": wdro_kimg_grid,
        "num_rows": len(rows),
    }
    return rows, summary


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    row_list = list(rows)
    if not row_list:
        raise RuntimeError("Cannot write an empty manifest")
    fieldnames: List[str] = []
    for row in row_list:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(row_list)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "evals").mkdir(parents=True, exist_ok=True)
    rows, summary = build_manifest_rows(args)
    manifest_path = outdir / args.manifest_name
    summary_path = outdir / args.summary_name
    write_csv(manifest_path, rows)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"[OK] Wrote manifest CSV: {manifest_path}")
    print(f"[OK] Wrote manifest summary: {summary_path}")
    print(
        "[OK] Warmup boundary: "
        f"kimg={summary['warmup_boundary_kimg']} "
        f"wall_clock={summary['warmup_boundary_train_wall_clock_sec']:.3f}s "
        f"weighted_compute={summary['warmup_boundary_weighted_compute_units']:.3f}"
    )


if __name__ == "__main__":
    main()
