#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import subprocess
import sys
from typing import Dict, Iterable, List, Optional, Tuple


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


from toy.compute_accounting import (  # noqa: E402
    baseline_weighted_compute_units_for_steps,
    cdro_adjusted_weighted_compute_from_compute_accounting,
    cdro_robust_step_weighted_compute_units,
    estimate_wall_clock_sec_from_batch_equiv,
    load_weighted_compute_calibration,
    solve_warmup_steps_for_target_compute_fraction,
    wdro_robust_step_weighted_compute_units,
)
from toy.process_title import apply_process_title, build_process_title, child_process_env  # noqa: E402
from toy.shared.ema import append_ema_cli_args, ema_config_dict  # noqa: E402


_APPLIED_PROCESS_TITLE = apply_process_title()


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
GRID_TEMPLATE_STEPS_STANDARD = [
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
GRID_TEMPLATE_STEPS_DENSER = [
    0,
    5,
    10,
    15,
    25,
    35,
    50,
    75,
    100,
    150,
    250,
    350,
    500,
    750,
    1000,
    1500,
    2000,
    2500,
    3500,
    5000,
    7500,
    10000,
    12500,
    15000,
    17500,
    20000,
    22500,
    25000,
    27500,
    30000,
    32500,
    35000,
    37500,
    40000,
    45000,
    50000,
    55000,
    60000,
    65000,
    70000,
    73500,
    76970,
]
FID_EVAL_TEMPLATE_STEPS_BALANCED = [
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
    30000,
    35000,
    40000,
    50000,
    60000,
    70000,
    76970,
]
GRID_TEMPLATES = {
    "standard": {
        "name": "wdro_dense_25_relative",
        "steps": GRID_TEMPLATE_STEPS_STANDARD,
    },
    "denser": {
        "name": "wdro_dense_41_relative",
        "steps": GRID_TEMPLATE_STEPS_DENSER,
    },
}
FID_EVAL_TEMPLATES = {
    "all": {
        "name": "every_comparison_knot",
        "steps": None,
    },
    "balanced": {
        "name": "weighted_balanced_20_relative",
        "steps": FID_EVAL_TEMPLATE_STEPS_BALANCED,
    },
}
TRANSITION_SENTINEL_ROBUST_COUNT = 3
DEFAULT_WALL_CLOCK_MODE = "observed"
DEFAULT_WALL_CLOCK_SEC_PER_KIMG = 0.629646
_OBSERVED_SOURCE_CSV_CACHE: Dict[str, Dict[Tuple[int, int], Dict[str, str]]] = {}
_OBSERVED_METRICS_RUNTIME_CACHE: Dict[str, Dict] = {}


def _grid_template_names() -> List[str]:
    return sorted(GRID_TEMPLATES.keys())


def _fid_eval_template_names() -> List[str]:
    return sorted(FID_EVAL_TEMPLATES.keys())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect raw per-seed baseline, WDRO, and CDRO data on a shared weighted-compute grid. "
            "Each method/seed is evaluated along a single checkpointed trajectory: baseline is trained as one "
            "continuous baseline trajectory, while WDRO/CDRO use a method-local warmup baseline trajectory up "
            "to one fixed warmup checkpoint and then continue the robust phase via same-seed same-method "
            "resume checkpoints. Aggregation is deferred to later analysis."
        )
    )
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--prefix", type=str, required=True)
    parser.add_argument("--python-bin", type=str, default=os.path.join(ROOT_DIR, ".venv", "bin", "python"))
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--amp-dtype",
        type=str,
        default="auto",
        choices=["auto", "off", "bfloat16", "float16", "bf16", "fp16", "half"],
    )
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="GM-CDRO")
    parser.add_argument("--wandb-entity", type=str, default="lechibachh")
    parser.add_argument("--use-ema-eval", action="store_true", default=True)
    parser.add_argument("--disable-ema-eval", action="store_false", dest="use_ema_eval")
    parser.add_argument("--ema-mode", type=str, default="official", choices=["official", "fixed"])
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--ema-halflife-kimg", type=float, default=500.0)
    parser.add_argument("--ema-rampup-ratio", type=float, default=0.05)
    parser.add_argument("--disable-ema-rampup", action="store_true")
    parser.add_argument("--dataset-path", type=str, default=DEFAULT_TRAIN_ROOT)
    parser.add_argument("--dataset-val-path", type=str, default=DEFAULT_VAL_ROOT)
    parser.add_argument("--fid-ref-path", type=str, default=DEFAULT_FID_REF)
    parser.add_argument("--weighted-compute-calibration-path", type=str, default=DEFAULT_CALIBRATION)
    parser.add_argument("--weighted-inputgrad-alpha", type=float, default=0.0)
    parser.add_argument("--weighted-parambackward-beta", type=float, default=0.0)
    parser.add_argument("--train-accelerator-count", type=int, default=1)
    parser.add_argument(
        "--wall-clock-mode",
        type=str,
        choices=["observed", "current_sec_per_kimg"],
        default=DEFAULT_WALL_CLOCK_MODE,
    )
    parser.add_argument("--wall-clock-sec-per-kimg", type=float, default=DEFAULT_WALL_CLOCK_SEC_PER_KIMG)
    parser.add_argument("--image-size", type=int, default=28)
    parser.add_argument("--image-channels", type=int, default=3)
    parser.add_argument("--image-train-size", type=int, default=400)
    parser.add_argument("--image-val-size", type=int, default=2000)
    parser.add_argument("--train-percent-label", type=str, default="5%")
    parser.add_argument("--image-split-seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--baseline-train-batch-gpu", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--image-backbone", type=str, default="conv", choices=["conv", "songunet", "ddpmpp"])
    parser.add_argument("--baseline-train-backend", type=str, default="toy", choices=["toy", "mainline"])
    parser.add_argument("--training-objective", type=str, default="edm", choices=["edm", "score", "rf"])
    parser.add_argument("--rf-baseline-mode", type=str, default="strong", choices=["strong", "plain"])
    parser.add_argument("--rf-stage1-fraction", type=float, default=0.5)
    parser.add_argument("--rf-reflow-t-distribution", type=str, default="u_shaped", choices=["u_shaped", "uniform"])
    parser.add_argument("--rf-loss", type=str, default="pseudo_huber", choices=["pseudo_huber", "mse"])
    parser.add_argument("--rf-pseudo-huber-delta", type=float, default=0.1)
    parser.add_argument("--rf-edm-init-ckpt-path", type=str, default="")
    parser.add_argument("--rf-cdro-pair-source", type=str, default="auto", choices=["auto", "reflow", "data_noise"])
    parser.add_argument("--eval-samples", type=int, default=2000)
    parser.add_argument("--fid-samples", type=int, default=2000)
    parser.add_argument("--fid-gen-batch", type=int, default=64)
    parser.add_argument("--debug-eval-batch", type=int, default=64)
    parser.add_argument("--debug-terminal-step", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--n-steps-path", type=int, default=64)
    parser.add_argument("--sigma-min", type=float, default=0.002)
    parser.add_argument("--sigma-max", type=float, default=2.0)
    parser.add_argument("--shared-weighted-cap", type=float, default=400000.0)
    parser.add_argument("--grid-template", type=str, choices=_grid_template_names(), default="denser")
    parser.add_argument("--fid-eval-template", type=str, choices=_fid_eval_template_names(), default="balanced")
    parser.add_argument(
        "--transition-sentinel-robust-count",
        type=int,
        default=TRANSITION_SENTINEL_ROBUST_COUNT,
    )
    parser.add_argument(
        "--baseline-fid-mode",
        type=str,
        choices=["in_run", "posthoc_from_checkpoints"],
        default="posthoc_from_checkpoints",
    )
    parser.add_argument(
        "--robust-fid-mode",
        type=str,
        choices=["in_run", "posthoc_from_checkpoints"],
        default="posthoc_from_checkpoints",
    )
    parser.add_argument(
        "--reeval-script",
        type=str,
        default=os.path.join(ROOT_DIR, "toy", "scripts", "reevaluate_three_method_fids_from_checkpoints.py"),
    )
    parser.add_argument("--posthoc-fid-batch-size", type=int, default=512)
    parser.add_argument("--baseline-max-steps", type=int, default=160000)
    parser.add_argument("--wdro-max-total-steps", type=int, default=160000)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--reuse-baseline-runs-csv", type=str, default="")
    parser.add_argument("--reuse-baseline-aggregate-csv", type=str, default="")
    parser.add_argument("--reuse-wdro-raw-csv", type=str, default="")
    parser.add_argument("--reuse-cdro-warmup-runs-csv", type=str, default="")
    parser.add_argument("--reuse-cdro-warmup-aggregate-csv", type=str, default="")
    parser.add_argument("--wdro-warmup-fraction", type=float, default=0.2)
    parser.add_argument("--wdro-refresh-epochs", type=float, default=100.0)
    parser.add_argument("--wdro-adv-prob", type=float, default=0.3)
    parser.add_argument("--wdro-attack-steps", type=int, default=2)
    parser.add_argument("--wdro-attack-step-size", type=float, default=1e-3)
    parser.add_argument("--wdro-gamma", type=float, default=1.0)
    parser.add_argument("--inner-steps", type=int, default=1)
    parser.add_argument("--outer-attack-weight", type=float, default=1.0)
    parser.add_argument("--outer-clean-weight", type=float, default=0.0)
    parser.add_argument("--cdro-step-size", type=float, default=0.02)
    parser.add_argument("--cdro-total-budget-rho", type=float, default=32.0)
    parser.add_argument("--cdro-time-horizon", type=float, default=1.0)
    parser.add_argument(
        "--cdro-edm-ladder-mode",
        type=str,
        default="stochastic_stratified_quantile",
        choices=["deterministic_midpoint_quantile", "stochastic_stratified_quantile"],
    )
    parser.add_argument("--cdro-warmup-fraction", type=float, default=0.2)
    parser.add_argument("--cdro-n-steps-path", type=int, default=32)
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _canonical_amp_dtype(value: str) -> str:
    mode = str(value).strip().lower()
    if mode in {"auto", "off", "bfloat16", "float16"}:
        return mode
    if mode == "bf16":
        return "bfloat16"
    if mode in {"fp16", "half"}:
        return "float16"
    raise ValueError(f"Unsupported amp dtype: {value}")


def _train_percent_cli_value(label: str) -> str:
    text = str(label).strip()
    if text.endswith("%"):
        text = text[:-1].strip()
    if not text:
        raise ValueError(f"Invalid train percent label: {label!r}")
    return text


def _append_rf_cli_args(cmd: List[str], args: argparse.Namespace) -> None:
    cmd.extend(
        [
            "--rf-baseline-mode",
            str(args.rf_baseline_mode),
            "--rf-stage1-fraction",
            str(args.rf_stage1_fraction),
            "--rf-reflow-t-distribution",
            str(args.rf_reflow_t_distribution),
            "--rf-loss",
            str(args.rf_loss),
            "--rf-pseudo-huber-delta",
            str(args.rf_pseudo_huber_delta),
            "--rf-cdro-pair-source",
            str(args.rf_cdro_pair_source),
        ]
    )
    rf_edm_init_ckpt_path = str(getattr(args, "rf_edm_init_ckpt_path", "")).strip()
    if rf_edm_init_ckpt_path:
        cmd.extend(["--rf-edm-init-ckpt-path", rf_edm_init_ckpt_path])


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


def _encode_float_list(values: Iterable[float]) -> str:
    return ",".join(f"{float(value):.12g}" for value in values)


def _encode_str_list(values: Iterable[str]) -> str:
    return ",".join(str(value) for value in values)


def _optional_float(value) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return float(parsed)


def _optional_int(value) -> Optional[int]:
    parsed = _optional_float(value)
    if parsed is None:
        return None
    return int(parsed)


def _optional_bool(value, *, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off", ""}:
        return False
    return bool(default)


def _source_csv_rows_by_seed_step(path: str) -> Dict[Tuple[int, int], Dict[str, str]]:
    indexed = _OBSERVED_SOURCE_CSV_CACHE.get(path)
    if indexed is not None:
        return indexed
    indexed = {}
    for row in load_csv_rows(path):
        seed = _optional_int(row.get("seed"))
        step = _optional_int(row.get("step"))
        if seed is None or step is None:
            continue
        indexed[(int(seed), int(step))] = row
    _OBSERVED_SOURCE_CSV_CACHE[path] = indexed
    return indexed


def _recover_observed_wall_clock_from_source_csv(
    *,
    source_csv: str,
    seed: Optional[int],
    step: Optional[int],
) -> Optional[Dict]:
    if seed is None or step is None:
        return None
    if not source_csv or not os.path.isfile(source_csv):
        return None
    source_row = _source_csv_rows_by_seed_step(source_csv).get((int(seed), int(step)))
    if not isinstance(source_row, dict):
        return None
    train_wall_clock_sec = _optional_float(source_row.get("train_wall_clock_sec", source_row.get("train_elapsed_sec")))
    if train_wall_clock_sec is None:
        return None
    return {
        "train_wall_clock_sec": float(train_wall_clock_sec),
        "train_elapsed_sec": float(train_wall_clock_sec),
        "train_gpu_hours": _optional_float(source_row.get("train_gpu_hours")),
        "train_wall_clock_complete": _optional_bool(
            source_row.get("train_wall_clock_complete"),
            default=(train_wall_clock_sec is not None),
        ),
        "train_wall_clock_source": str(source_row.get("train_wall_clock_source") or "observed_source_csv"),
    }


def _recover_observed_wall_clock_from_metrics_path(metrics_path: str) -> Optional[Dict]:
    cached = _OBSERVED_METRICS_RUNTIME_CACHE.get(metrics_path)
    if cached is not None:
        return cached
    if not metrics_path or not os.path.isfile(metrics_path):
        return None
    try:
        payload = load_json(metrics_path)
    except Exception:
        return None
    metrics = payload.get("metrics", {})
    flow = metrics.get("flow_debug", {})
    runtime = flow.get("runtime", {})
    compute_accounting = flow.get("compute_accounting", {})
    train_wall_clock_sec = _optional_float(
        compute_accounting.get("train_wall_clock_sec", runtime.get("train_wall_clock_sec", runtime.get("effective_train_total")))
    )
    if train_wall_clock_sec is None:
        return None
    recovered = {
        "train_wall_clock_sec": float(train_wall_clock_sec),
        "train_elapsed_sec": float(train_wall_clock_sec),
        "train_gpu_hours": _optional_float(compute_accounting.get("train_gpu_hours", runtime.get("train_gpu_hours"))),
        "train_wall_clock_complete": _optional_bool(
            compute_accounting.get("train_wall_clock_complete", runtime.get("train_wall_clock_complete")),
            default=(train_wall_clock_sec is not None),
        ),
        "train_wall_clock_source": "observed_metrics_runtime",
        "baseline_train_wall_clock_sec_effective": _optional_float(runtime.get("baseline_train_wall_clock_sec_effective")),
        "robust_train_wall_clock_sec_effective": _optional_float(runtime.get("robust_phase")),
        "baseline_train_wall_clock_source": str(
            runtime.get("baseline_train_wall_clock_sec_effective_source")
            or runtime.get("baseline_reference_train_wall_clock_source")
            or "observed_metrics_runtime"
        ),
    }
    _OBSERVED_METRICS_RUNTIME_CACHE[metrics_path] = recovered
    return recovered


def _is_synthetic_wall_clock_source(value) -> bool:
    source = str(value or "").strip()
    return source.startswith("current_sec_per_kimg:")


def _clear_unrecovered_observed_wall_clock(row: Dict) -> Dict:
    out = dict(row)
    if _is_synthetic_wall_clock_source(out.get("train_wall_clock_source")):
        out["train_wall_clock_sec"] = None
        out["train_elapsed_sec"] = None
        out["train_gpu_hours"] = None
        out["train_wall_clock_complete"] = False
        out["train_wall_clock_source"] = "observed_runtime_missing"
        out["baseline_train_wall_clock_sec_effective"] = None
        out["robust_train_wall_clock_sec_effective"] = None
        if not str(out.get("baseline_train_wall_clock_source", "")).strip() or _is_synthetic_wall_clock_source(
            out.get("baseline_train_wall_clock_source")
        ):
            out["baseline_train_wall_clock_source"] = "observed_runtime_missing"
        if not str(out.get("robust_train_wall_clock_source", "")).strip() or _is_synthetic_wall_clock_source(
            out.get("robust_train_wall_clock_source")
        ):
            out["robust_train_wall_clock_source"] = "observed_runtime_missing"
        return out
    if _is_synthetic_wall_clock_source(out.get("baseline_train_wall_clock_source")):
        out["baseline_train_wall_clock_sec_effective"] = None
        out["baseline_train_wall_clock_source"] = "observed_runtime_missing"
    if _is_synthetic_wall_clock_source(out.get("robust_train_wall_clock_source")):
        out["robust_train_wall_clock_sec_effective"] = None
        out["robust_train_wall_clock_source"] = "observed_runtime_missing"
    return out


def _recover_observed_wall_clock_accounting(*, row: Dict, args: argparse.Namespace) -> Dict:
    out = dict(row)
    if str(args.wall_clock_mode) == "current_sec_per_kimg":
        return out

    recovered = None
    for metrics_key in ("metrics_path", "reeval_metrics_path"):
        metrics_path = str(out.get(metrics_key, "")).strip()
        if not metrics_path:
            continue
        recovered = _recover_observed_wall_clock_from_metrics_path(metrics_path)
        if recovered is not None:
            break

    if recovered is None:
        source_csv = str(out.get("source_csv", "")).strip()
        recovered = _recover_observed_wall_clock_from_source_csv(
            source_csv=source_csv,
            seed=_optional_int(out.get("seed")),
            step=_optional_int(out.get("step")),
        )

    if recovered is None:
        return _clear_unrecovered_observed_wall_clock(out)

    train_wall_clock_sec = _optional_float(recovered.get("train_wall_clock_sec"))
    if train_wall_clock_sec is not None:
        out["train_wall_clock_sec"] = float(train_wall_clock_sec)
        out["train_elapsed_sec"] = float(train_wall_clock_sec)
    train_gpu_hours = _optional_float(recovered.get("train_gpu_hours"))
    if train_gpu_hours is None and train_wall_clock_sec is not None:
        train_gpu_hours = float(train_wall_clock_sec) * float(max(int(args.train_accelerator_count), 0)) / 3600.0
    if train_gpu_hours is not None:
        out["train_gpu_hours"] = float(train_gpu_hours)
    out["train_wall_clock_complete"] = bool(
        recovered.get("train_wall_clock_complete", train_wall_clock_sec is not None)
    )
    out["train_wall_clock_source"] = str(recovered.get("train_wall_clock_source") or "observed_artifact_runtime")

    baseline_train_wall_clock_sec = _optional_float(recovered.get("baseline_train_wall_clock_sec_effective"))
    if baseline_train_wall_clock_sec is not None:
        out["baseline_train_wall_clock_sec_effective"] = float(baseline_train_wall_clock_sec)
    robust_train_wall_clock_sec = _optional_float(recovered.get("robust_train_wall_clock_sec_effective"))
    if robust_train_wall_clock_sec is not None:
        out["robust_train_wall_clock_sec_effective"] = float(robust_train_wall_clock_sec)
    if baseline_train_wall_clock_sec is not None:
        out["baseline_train_wall_clock_source"] = str(
            recovered.get("baseline_train_wall_clock_source") or "observed_artifact_runtime"
        )
    return out


def _validate_wall_clock_args(args: argparse.Namespace) -> None:
    if str(args.wall_clock_mode) == "current_sec_per_kimg" and float(args.wall_clock_sec_per_kimg) <= 0.0:
        raise ValueError("--wall-clock-sec-per-kimg must be positive when --wall-clock-mode=current_sec_per_kimg")


def _wall_clock_source_label(args: argparse.Namespace) -> str:
    if str(args.wall_clock_mode) == "current_sec_per_kimg":
        return f"current_sec_per_kimg:{float(args.wall_clock_sec_per_kimg):.6f}"
    return "observed_train_runtime"


def _batch_equiv_wall_clock_sec(
    *,
    batch_equiv_denoiser_evals,
    args: argparse.Namespace,
) -> Optional[float]:
    batch_equiv_value = _optional_float(batch_equiv_denoiser_evals)
    if batch_equiv_value is None:
        return None
    if str(args.wall_clock_mode) != "current_sec_per_kimg":
        return None
    return float(
        estimate_wall_clock_sec_from_batch_equiv(
            batch_equiv_denoiser_evals=float(batch_equiv_value),
            batch_size=int(args.batch_size),
            sec_per_kimg=float(args.wall_clock_sec_per_kimg),
        )
    )


def _apply_wall_clock_accounting(*, row: Dict, args: argparse.Namespace) -> Dict:
    out = _recover_observed_wall_clock_accounting(row=row, args=args)
    source_label = _wall_clock_source_label(args)
    if str(args.wall_clock_mode) != "current_sec_per_kimg":
        if not str(out.get("train_wall_clock_source", "")).strip():
            out["train_wall_clock_source"] = source_label
        if (
            _optional_float(out.get("baseline_train_wall_clock_sec_effective")) is not None
            and not str(out.get("baseline_train_wall_clock_source", "")).strip()
        ):
            out["baseline_train_wall_clock_source"] = source_label
        return out

    total_batch_equiv = _optional_float(out.get("compute_budget_be", out.get("batch_equiv_denoiser_evals")))
    total_wall_clock_sec = _batch_equiv_wall_clock_sec(batch_equiv_denoiser_evals=total_batch_equiv, args=args)
    if total_wall_clock_sec is None:
        if not str(out.get("train_wall_clock_source", "")).strip():
            out["train_wall_clock_source"] = source_label
        return out

    baseline_wall_clock_sec = _batch_equiv_wall_clock_sec(
        batch_equiv_denoiser_evals=out.get("baseline_compute_be"),
        args=args,
    )
    robust_wall_clock_sec = _batch_equiv_wall_clock_sec(
        batch_equiv_denoiser_evals=out.get("robust_compute_be"),
        args=args,
    )
    out["train_wall_clock_sec"] = float(total_wall_clock_sec)
    out["train_elapsed_sec"] = float(total_wall_clock_sec)
    out["train_gpu_hours"] = (
        float(total_wall_clock_sec) * float(max(int(args.train_accelerator_count), 0)) / 3600.0
    )
    out["train_wall_clock_complete"] = True
    out["train_wall_clock_source"] = source_label
    if "baseline_compute_be" in out or baseline_wall_clock_sec is not None:
        out["baseline_train_wall_clock_sec_effective"] = (
            None if baseline_wall_clock_sec is None else float(baseline_wall_clock_sec)
        )
        out["baseline_train_wall_clock_source"] = source_label
    if "robust_compute_be" in out or robust_wall_clock_sec is not None:
        out["robust_train_wall_clock_sec_effective"] = (
            None if robust_wall_clock_sec is None else float(robust_wall_clock_sec)
        )
    return out


def _resolve_fid_annotation(
    *,
    fid_value: Optional[float],
    fid_selected: bool,
    source_if_evaluated: str = "in_run_metrics",
) -> Dict:
    fid_evaluated = fid_value is not None
    if fid_evaluated:
        missing_reason = ""
        fid_source = str(source_if_evaluated)
    elif fid_selected:
        missing_reason = "selected_but_missing"
        fid_source = "missing_in_run_metrics"
    else:
        missing_reason = "not_selected_by_schedule"
        fid_source = "not_evaluated"
    return {
        "fid": None if fid_value is None else float(fid_value),
        "fid_eval_selected": bool(fid_selected),
        "fid_evaluated": bool(fid_evaluated),
        "fid_missing_reason": str(missing_reason),
        "fid_source": str(fid_source),
    }


def _ensure_row_fid_fields(row: Dict) -> Dict:
    out = dict(row)
    fid_value = _optional_float(out.get("fid"))
    fid_selected = _optional_bool(out.get("fid_eval_selected"), default=(fid_value is not None))
    fid_evaluated = _optional_bool(out.get("fid_evaluated"), default=(fid_value is not None))
    fid_missing_reason = str(out.get("fid_missing_reason", "")).strip()
    if not fid_missing_reason and not fid_evaluated:
        fid_missing_reason = "not_selected_by_schedule" if not fid_selected else "selected_but_missing"
    fid_source = str(out.get("fid_source", "")).strip()
    if not fid_source:
        fid_source = "in_run_metrics" if fid_evaluated else "not_evaluated"
    out.update(
        {
            "fid": fid_value,
            "fid_eval_selected": bool(fid_selected),
            "fid_evaluated": bool(fid_evaluated),
            "fid_missing_reason": str(fid_missing_reason),
            "fid_source": str(fid_source),
        }
    )
    return out


def _backbone_family(training_objective: str) -> str:
    objective = str(training_objective).strip().lower()
    if objective == "rf":
        return "rf"
    if objective == "score":
        return "score"
    return "edm"


def _backbone_label(training_objective: str) -> str:
    family = _backbone_family(training_objective)
    if family == "rf":
        return "RF"
    if family == "score":
        return "Score VE"
    return "EDM"


def _canonical_robust_method(method_name: str) -> str:
    method = str(method_name).strip().lower()
    if method in {"baseline", "baseline_edm", "baseline_rf", "baseline_score"}:
        return "baseline"
    if method in {"wdro", "wild", "wild_diffusion"}:
        return "wild_diffusion"
    if method == "cdro":
        return "cdro"
    return method


def _robust_label(method_name: str) -> str:
    robust_method = _canonical_robust_method(method_name)
    if robust_method == "baseline":
        return "Baseline"
    if robust_method == "wild_diffusion":
        return "Wild-Diffusion"
    if robust_method == "cdro":
        return "CDRO"
    return robust_method.replace("_", " ").title()


def _series_fields(*, method_name: str, training_objective: str, method_version_used: Optional[str] = None) -> Dict[str, str]:
    robust_method = _canonical_robust_method(method_name)
    backbone_family = _backbone_family(training_objective)
    backbone_label = _backbone_label(training_objective)
    robust_label = _robust_label(method_name)
    return {
        "method": robust_method,
        "method_version_used": str(method_version_used or method_name),
        "robust_method": robust_method,
        "robust_label": robust_label,
        "backbone_family": backbone_family,
        "backbone_label": backbone_label,
        "training_objective": str(training_objective),
        "series_key": f"{robust_method}_{backbone_family}",
        "series_label": f"{robust_label} {backbone_label}",
    }


def _summary_stat(series, stat_name: str) -> Optional[float]:
    if not isinstance(series, dict):
        return None
    return _optional_float(series.get(stat_name))


def _pick_objective_series(objective: Dict, keys: Iterable[str]) -> Tuple[Optional[str], Optional[Dict]]:
    for key in keys:
        series = objective.get(str(key))
        if not isinstance(series, dict):
            continue
        if any(_summary_stat(series, stat_name) is not None for stat_name in ("final", "mean_last", "min", "max")):
            return str(key), series
    return None, None


def _empty_objective_debug_fields() -> Dict:
    return {
        "outer_loss_final": None,
        "outer_loss_mean_last": None,
        "outer_loss_attack_final": None,
        "outer_loss_attack_mean_last": None,
        "outer_loss_clean_final": None,
        "outer_loss_clean_mean_last": None,
        "attack_metric_kind": None,
        "attack_metric_final": None,
        "attack_metric_mean_last": None,
        "transport_cost_kind": None,
        "transport_cost_final": None,
        "transport_cost_mean_last": None,
        "delta_norm_mean_final": None,
        "delta_norm_max_final": None,
        "delta_norm_ratio_mean_final": None,
        "delta_norm_ratio_max_final": None,
        "sched_attack_weight_final": None,
        "sched_clean_weight_final": None,
        "diag_delta_norm_ratio_mean_final": None,
        "diag_inner_obj_gap_ratio_final": None,
    }


def _objective_debug_fields(*, objective: Dict) -> Dict:
    transport_kind, transport_series = _pick_objective_series(
        objective,
        (
            "cdro_transport_cost",
            "robust_energy",
            "wdro_transport_cost",
            "v11_path_transport_cost",
            "wild_sample_transport_cost",
            "wild_inner_transport_cost",
        ),
    )
    attack_metric_kind, attack_metric_series = _pick_objective_series(
        objective,
        (
            "robust_inner_obj",
            "wdro_attack_loss",
            "wild_inner_attack_loss",
        ),
    )
    fields = _empty_objective_debug_fields()
    fields.update(
        {
            "outer_loss_final": _summary_stat(objective.get("robust_outer_loss"), "final"),
            "outer_loss_mean_last": _summary_stat(objective.get("robust_outer_loss"), "mean_last"),
            "outer_loss_attack_final": _summary_stat(objective.get("robust_outer_loss_attack"), "final"),
            "outer_loss_attack_mean_last": _summary_stat(objective.get("robust_outer_loss_attack"), "mean_last"),
            "outer_loss_clean_final": _summary_stat(objective.get("robust_outer_loss_clean"), "final"),
            "outer_loss_clean_mean_last": _summary_stat(objective.get("robust_outer_loss_clean"), "mean_last"),
            "attack_metric_kind": attack_metric_kind,
            "attack_metric_final": _summary_stat(attack_metric_series, "final"),
            "attack_metric_mean_last": _summary_stat(attack_metric_series, "mean_last"),
            "transport_cost_kind": transport_kind,
            "transport_cost_final": _summary_stat(transport_series, "final"),
            "transport_cost_mean_last": _summary_stat(transport_series, "mean_last"),
            "delta_norm_mean_final": _summary_stat(objective.get("robust_delta_norm_mean"), "final"),
            "delta_norm_max_final": _summary_stat(objective.get("robust_delta_norm_max"), "final"),
            "delta_norm_ratio_mean_final": _summary_stat(objective.get("robust_delta_norm_ratio_mean"), "final"),
            "delta_norm_ratio_max_final": _summary_stat(objective.get("robust_delta_norm_ratio_max"), "final"),
            "sched_attack_weight_final": _summary_stat(objective.get("robust_sched_attack_weight"), "final"),
            "sched_clean_weight_final": _summary_stat(objective.get("robust_sched_clean_weight"), "final"),
            "diag_delta_norm_ratio_mean_final": _summary_stat(
                objective.get("diag_delta_norm_ratio_mean"),
                "final",
            ),
            "diag_inner_obj_gap_ratio_final": _summary_stat(objective.get("diag_inner_obj_gap_ratio"), "final"),
        }
    )
    return fields


def _row_role_label(*, is_comparison_knot: bool, is_aux_warmup_support: bool) -> str:
    if is_comparison_knot and is_aux_warmup_support:
        return "comparison_knot+aux_warmup_support"
    if is_comparison_knot:
        return "comparison_knot"
    if is_aux_warmup_support:
        return "aux_warmup_support"
    return "unlabeled"


def run_command(*, cmd: List[str], log_path: str, proc_title: Optional[str] = None) -> None:
    ensure_dir(os.path.dirname(log_path))
    with open(log_path, "w", encoding="utf-8") as handle:
        subprocess.run(
            cmd,
            cwd=ROOT_DIR,
            check=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=child_process_env(proc_title=proc_title),
        )


def _build_posthoc_reeval_cmd(
    *,
    args: argparse.Namespace,
    combined_csv: str,
    methods: List[str],
) -> List[str]:
    methods_text = ",".join(str(method).strip() for method in methods if str(method).strip())
    if not methods_text:
        raise ValueError("Expected at least one method for posthoc FID reevaluation.")
    return [
        args.python_bin,
        str(args.reeval_script),
        "--combined-csv",
        combined_csv,
        "--outdir",
        args.outdir,
        "--prefix",
        args.prefix,
        "--python-bin",
        args.python_bin,
        "--device",
        args.device,
        "--amp-dtype",
        _canonical_amp_dtype(args.amp_dtype),
        "--dataset-path",
        args.dataset_path,
        "--dataset-val-path",
        args.dataset_val_path,
        "--fid-ref-path",
        args.fid_ref_path,
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
        "--batch-size",
        str(args.batch_size),
        "--baseline-train-batch-gpu",
        str(args.baseline_train_batch_gpu),
        "--hidden-dim",
        str(args.hidden_dim),
        "--image-backbone",
        str(args.image_backbone),
        "--eval-samples",
        str(args.eval_samples),
        "--fid-samples",
        str(args.fid_samples),
        "--fid-batch-size",
        str(max(int(args.posthoc_fid_batch_size), 1)),
        "--debug-eval-batch",
        str(args.debug_eval_batch),
        "--debug-terminal-step",
        str(args.debug_terminal_step),
        "--log-every",
        str(args.log_every),
        "--n-steps-path-default",
        str(args.n_steps_path),
        "--sigma-min",
        str(args.sigma_min),
        "--sigma-max",
        str(args.sigma_max),
        "--train-percent-label",
        str(args.train_percent_label),
        "--methods",
        methods_text,
        "--only-missing-fid",
        "--respect-fid-selection",
        "--skip-plot",
    ]


def realize_shared_weighted_grid(*, shared_weighted_cap: float, template_steps: List[int]) -> List[float]:
    cap = max(float(shared_weighted_cap), 0.0)
    if cap <= 0.0:
        return [0.0]
    if not template_steps:
        raise ValueError("Expected at least one grid template step.")
    template_max = max(int(step) for step in template_steps)
    if template_max <= 0:
        raise ValueError("Grid template must have a positive max step.")
    out: List[float] = []
    previous = -1.0
    for base_step in template_steps:
        scaled = float(base_step) * cap / float(template_max)
        if scaled <= previous:
            scaled = previous + 1e-6
        out.append(float(scaled))
        previous = float(scaled)
    if out[-1] != cap:
        out[-1] = cap
    return out


def _select_fid_eval_indices(
    *,
    checkpoint_targets: List[float],
    fid_eval_template_key: str,
    shared_weighted_cap: float,
) -> Dict:
    if not checkpoint_targets:
        return {
            "template_key": str(fid_eval_template_key),
            "template_name": str(FID_EVAL_TEMPLATES[str(fid_eval_template_key)]["name"]),
            "requested_relative_steps": [],
            "requested_targets": [],
            "selected_indices": [],
            "selected_targets": [],
            "selection_strategy": "empty_checkpoint_grid",
        }
    template = FID_EVAL_TEMPLATES[str(fid_eval_template_key)]
    if template["steps"] is None:
        selected_indices = list(range(len(checkpoint_targets)))
        return {
            "template_key": str(fid_eval_template_key),
            "template_name": str(template["name"]),
            "requested_relative_steps": [],
            "requested_targets": [float(target) for target in checkpoint_targets],
            "selected_indices": [int(idx) for idx in selected_indices],
            "selected_targets": [float(target) for target in checkpoint_targets],
            "selection_strategy": "all_comparison_knots",
        }

    requested_targets = [
        float(target)
        for target in realize_shared_weighted_grid(
            shared_weighted_cap=float(shared_weighted_cap),
            template_steps=list(template["steps"]),
        )
        if float(target) > 0.0
    ]
    selected_indices: List[int] = []
    for target in requested_targets:
        best_index = min(
            range(len(checkpoint_targets)),
            key=lambda idx: (abs(float(checkpoint_targets[idx]) - float(target)), idx),
        )
        if best_index not in selected_indices:
            selected_indices.append(int(best_index))
    if (len(checkpoint_targets) - 1) not in selected_indices:
        selected_indices.append(len(checkpoint_targets) - 1)
    selected_indices = sorted(set(int(idx) for idx in selected_indices))
    return {
        "template_key": str(fid_eval_template_key),
        "template_name": str(template["name"]),
        "requested_relative_steps": [int(step) for step in template["steps"]],
        "requested_targets": [float(target) for target in requested_targets],
        "selected_indices": [int(idx) for idx in selected_indices],
        "selected_targets": [float(checkpoint_targets[idx]) for idx in selected_indices],
        "selection_strategy": "nearest_shared_weighted_checkpoint_knot",
    }


def _augment_fid_eval_steps_with_transition_sentinels(
    *,
    comparison_steps: List[int],
    fid_eval_steps: List[int],
    fixed_warmup_steps: int,
    robust_sentinel_count: int,
) -> List[int]:
    selected = {int(step) for step in fid_eval_steps if int(step) > 0}
    warmup_support_step = int(fixed_warmup_steps)
    if warmup_support_step > 0:
        selected.add(warmup_support_step)
    robust_steps = [
        int(step)
        for step in comparison_steps
        if int(step) > warmup_support_step
    ]
    for step in robust_steps[: max(int(robust_sentinel_count), 0)]:
        selected.add(int(step))
    return sorted(selected)


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


def _effective_cdro_n_steps_path(args: argparse.Namespace) -> int:
    override_value = max(int(getattr(args, "cdro_n_steps_path", 0)), 0)
    if override_value > 0:
        return int(override_value)
    return int(args.n_steps_path)


def _cdro_expected_robust_step_weighted_units(args: argparse.Namespace, calibration: Dict) -> float:
    units = cdro_robust_step_weighted_compute_units(
        n_steps_path=int(_effective_cdro_n_steps_path(args)),
        inner_steps=int(args.inner_steps),
        total_budget_rho=float(args.cdro_total_budget_rho),
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


def _wdro_warmup_steps_for_total(*, total_steps: int, args: argparse.Namespace) -> int:
    total_steps_value = max(int(total_steps), 0)
    warmup_steps = int(total_steps_value * float(args.wdro_warmup_fraction))
    return max(0, min(warmup_steps, total_steps_value))


def _solve_total_steps_for_target_weighted(
    *,
    target_weighted_units: float,
    max_total_steps: int,
    warmup_steps_fn,
    baseline_step_weighted_units: float,
    robust_step_weighted_units: float,
) -> int:
    target = max(float(target_weighted_units), 0.0)
    lo, hi = 0, max(int(max_total_steps), 0)
    while lo < hi:
        mid = (lo + hi) // 2
        warmup_steps = int(warmup_steps_fn(int(mid)))
        value = _piecewise_weighted_prefix(
            total_steps_prefix=int(mid),
            fixed_warmup_steps=int(warmup_steps),
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


def _normalize_baseline_runs(*, runs_csv: str, args: argparse.Namespace) -> List[Dict]:
    rows = load_csv_rows(runs_csv)
    out: List[Dict] = []
    for row in rows:
        fid_value = _optional_float(row.get("baseline_fid"))
        fid_selected = _optional_bool(row.get("fid_eval_selected"), default=(fid_value is not None))
        fid_evaluated = _optional_bool(row.get("fid_evaluated"), default=(fid_value is not None))
        fid_missing_reason = str(row.get("fid_missing_reason", "")).strip()
        if not fid_missing_reason and not fid_evaluated:
            fid_missing_reason = "not_selected_by_schedule" if not fid_selected else "selected_but_missing"
        fid_source = str(row.get("fid_source", "")).strip()
        if not fid_source:
            fid_source = "in_run_metrics" if fid_evaluated else "not_evaluated"
        out.append(
            _apply_wall_clock_accounting(
                row={
                    "seed": int(row["seed"]),
                    "step": int(row["step"]),
                    "fid": fid_value,
                    "train_wall_clock_sec": float(row["train_wall_clock_sec"]),
                    "weighted_compute_units": float(row["weighted_compute_units"]),
                    "batch_equiv_denoiser_evals": float(row["batch_equiv_denoiser_evals"]),
                    "loss_kind": "baseline_loss",
                    "loss_final": float(row["baseline_loss_final"]),
                    "loss_mean_last": float(row["baseline_loss_mean_last"]),
                    "images_shown_m": float(row["images_shown_m"]),
                    "image_backbone": str(row.get("image_backbone", getattr(args, "image_backbone", "conv"))),
                    "row_origin": "baseline_seed_run",
                    "source_csv": runs_csv,
                    "exp_name": row["exp_name"],
                    "exp_dir": row["exp_dir"],
                    "checkpoint_path": row["checkpoint_path"],
                    "fid_eval_selected": bool(fid_selected),
                    "fid_evaluated": bool(fid_evaluated),
                    "fid_missing_reason": str(fid_missing_reason),
                    "fid_source": str(fid_source),
                    **_series_fields(
                        method_name="baseline",
                        training_objective=str(args.training_objective),
                        method_version_used="baseline",
                    ),
                    **_empty_objective_debug_fields(),
                },
                args=args,
            )
        )
    return out


def _extract_wdro_row(
    *,
    metrics_path: str,
    calibration: Dict,
    train_accelerator_count: int,
    fid_selected: bool,
    args: argparse.Namespace,
) -> Dict:
    payload = load_json(metrics_path)
    source_cfg = payload.get("config", {})
    if not isinstance(source_cfg, dict):
        source_cfg = {}
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
        _optional_float(sample_quality.get("baseline_fid"))
        if warmup_only
        else _optional_float(sample_quality.get("robust_fid"))
    )
    row = {
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
        "image_backbone": str(source_cfg.get("image_backbone", getattr(args, "image_backbone", "conv"))),
        "loss_kind": "robust_outer_loss",
        "loss_final": _optional_float(objective.get("robust_outer_loss", {}).get("final")),
        "loss_mean_last": _optional_float(objective.get("robust_outer_loss", {}).get("mean_last")),
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
        **_series_fields(
            method_name="wdro",
            training_objective=str(flow.get("training_objective", getattr(args, "training_objective", "edm"))),
            method_version_used="wdro",
        ),
        **_objective_debug_fields(objective=objective),
    }
    row.update(
        _resolve_fid_annotation(
            fid_value=fid_value,
            fid_selected=bool(fid_selected),
            source_if_evaluated="in_run_metrics",
        )
    )
    return _apply_wall_clock_accounting(row=row, args=args)


def _extract_cdro_row(
    *,
    metrics_path: str,
    baseline_ckpt_requested: Optional[str],
    calibration: Dict,
    train_accelerator_count: int,
    fid_selected: bool,
    args: argparse.Namespace,
) -> Dict:
    payload = load_json(metrics_path)
    source_cfg = payload.get("config", {})
    if not isinstance(source_cfg, dict):
        source_cfg = {}
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
        _optional_float(sample_quality.get("baseline_fid"))
        if warmup_only
        else _optional_float(sample_quality.get("robust_fid"))
    )
    row = {
        "step": int(total_steps_requested),
        "compute_budget_be": float(total_compute_be_effective),
        "baseline_compute_be": float(baseline_compute_be_effective),
        "robust_compute_be": float(robust_compute_be_raw),
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
        "images_shown_m": float(total_images_shown_m_effective),
        "image_backbone": str(source_cfg.get("image_backbone", getattr(args, "image_backbone", "conv"))),
        "loss_kind": "robust_outer_loss",
        "loss_final": _optional_float(objective.get("robust_outer_loss", {}).get("final")),
        "loss_mean_last": _optional_float(objective.get("robust_outer_loss", {}).get("mean_last")),
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
        "cdro_edm_ladder_mode": str(flow.get("cdro_edm_ladder_mode", "deterministic_midpoint_quantile")),
        "train_wall_clock_complete": bool(
            compute_accounting.get("train_wall_clock_complete", train_wall_clock_sec is not None)
        ),
        **_series_fields(
            method_name="cdro",
            training_objective=str(flow.get("training_objective", getattr(args, "training_objective", "edm"))),
            method_version_used="cdro",
        ),
        **_objective_debug_fields(objective=objective),
    }
    row.update(
        _resolve_fid_annotation(
            fid_value=fid_value,
            fid_selected=bool(fid_selected),
            source_if_evaluated="in_run_metrics",
        )
    )
    return _apply_wall_clock_accounting(row=row, args=args)


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
    out["method_version_used"] = str(out.get("method_version_used", method_name))
    out["seed"] = int(seed)
    out["row_origin"] = str(row_origin)
    out["row_role"] = "comparison_knot"
    out["is_comparison_knot"] = True
    out["is_aux_warmup_support"] = False
    out["weighted_grid_target"] = float(weighted_grid_target)
    out["comparison_weighted_targets"] = _encode_float_list([float(weighted_grid_target)])
    out["warmup_support_methods"] = ""
    out["trajectory_total_steps_max"] = int(trajectory_total_steps_max)
    out["fixed_warmup_steps"] = int(fixed_warmup_steps)
    return out


def _transfer_baseline_row_to_method(
    *,
    baseline_row: Dict,
    method_name: str,
    weighted_grid_target: float,
    fid_selected: bool,
    trajectory_total_steps_max: int,
    fixed_warmup_steps: int,
    row_origin: str = "trajectory_warmup_phase",
) -> Dict:
    row = dict(baseline_row)
    row.update(
        _series_fields(
            method_name=method_name,
            training_objective=str(row.get("training_objective", "")),
            method_version_used=method_name,
        )
    )
    row["row_origin"] = str(row_origin)
    row["weighted_grid_target"] = float(weighted_grid_target)
    row["comparison_weighted_targets"] = _encode_float_list([float(weighted_grid_target)])
    row["warmup_support_methods"] = ""
    row["row_role"] = _row_role_label(is_comparison_knot=True, is_aux_warmup_support=False)
    row["is_comparison_knot"] = True
    row["is_aux_warmup_support"] = False
    row["trajectory_total_steps_max"] = int(trajectory_total_steps_max)
    row["fixed_warmup_steps"] = int(fixed_warmup_steps)
    row["warmup_only"] = True
    row["baseline_ckpt_loaded"] = False
    row["robust_resume_loaded"] = False
    row["phase_step_split_mode"] = "single_trajectory_warmup_phase"
    row.update(
        _resolve_fid_annotation(
            fid_value=_optional_float(baseline_row.get("fid")),
            fid_selected=bool(fid_selected),
            source_if_evaluated=str(baseline_row.get("fid_source", "in_run_metrics") or "in_run_metrics"),
        )
    )
    return row


def _transfer_baseline_row_to_aux_warmup_support(
    *,
    baseline_row: Dict,
    method_name: str,
    trajectory_total_steps_max: int,
    fixed_warmup_steps: int,
) -> Dict:
    row = dict(baseline_row)
    row.update(
        _series_fields(
            method_name=method_name,
            training_objective=str(row.get("training_objective", "")),
            method_version_used=method_name,
        )
    )
    row["row_origin"] = "trajectory_warmup_phase"
    weighted_compute_units = _optional_float(baseline_row.get("weighted_compute_units"))
    row["weighted_grid_target"] = 0.0 if weighted_compute_units is None else float(weighted_compute_units)
    row["comparison_weighted_targets"] = ""
    row["warmup_support_methods"] = str(method_name)
    row["row_role"] = _row_role_label(is_comparison_knot=False, is_aux_warmup_support=True)
    row["is_comparison_knot"] = False
    row["is_aux_warmup_support"] = True
    row["trajectory_total_steps_max"] = int(trajectory_total_steps_max)
    row["fixed_warmup_steps"] = int(fixed_warmup_steps)
    row["warmup_only"] = True
    row["baseline_ckpt_loaded"] = False
    row["robust_resume_loaded"] = False
    row["phase_step_split_mode"] = "single_trajectory_warmup_phase"
    return row


def _build_baseline_sweep_cmd(
    *,
    args: argparse.Namespace,
    outdir: str,
    prefix: str,
    seeds_text: str,
    steps: List[int],
    fid_eval_steps: List[int],
) -> List[str]:
    cmd = [
        args.python_bin,
        os.path.join(ROOT_DIR, "toy", "scripts", "sweep_mnist_convergence_checkpointed.py"),
        "--outdir",
        outdir,
        "--prefix",
        prefix,
        "--seeds",
        seeds_text,
        "--train-percents",
        _train_percent_cli_value(args.train_percent_label),
        "--steps-list",
        format_steps_list(steps),
        "--fid-eval-steps-list",
        ("none" if not fid_eval_steps else format_steps_list(fid_eval_steps)),
        "--device",
        args.device,
        "--amp-dtype",
        _canonical_amp_dtype(args.amp_dtype),
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
        "--image-backbone",
        str(args.image_backbone),
        "--baseline-train-backend",
        str(args.baseline_train_backend),
        "--training-objective",
        str(args.training_objective),
        "--n-steps-path",
        str(args.n_steps_path),
        "--sigma-min",
        str(args.sigma_min),
        "--sigma-max",
        str(args.sigma_max),
        "--fid-ref-path",
        args.fid_ref_path,
        "--fid-samples",
        str(args.fid_samples),
        "--gen-batch",
        str(args.fid_gen_batch),
        "--weighted-compute-calibration-path",
        str(args.weighted_compute_calibration_path).strip(),
        "--train-accelerator-count",
        str(args.train_accelerator_count),
    ]
    _append_rf_cli_args(cmd, args)
    append_ema_cli_args(cmd, args)
    if float(args.weighted_inputgrad_alpha) > 0.0 and float(args.weighted_parambackward_beta) > 0.0:
        cmd.extend(
            [
                "--weighted-inputgrad-alpha",
                str(args.weighted_inputgrad_alpha),
                "--weighted-parambackward-beta",
                str(args.weighted_parambackward_beta),
            ]
        )
    return cmd


def _build_run_toy_cmd(
    *,
    args: argparse.Namespace,
    method_name: str,
    outdir: str,
    exp_name: str,
    seed: int,
    total_steps: int,
    fixed_warmup_steps: Optional[int],
    resume_path: Optional[str],
    save_path: Optional[str],
    baseline_ckpt_path: Optional[str],
    compute_fid: bool,
) -> List[str]:
    n_steps_path_value = int(_effective_cdro_n_steps_path(args)) if method_name == "cdro" else int(args.n_steps_path)
    cmd = [
        args.python_bin,
        os.path.join(ROOT_DIR, "toy", "run_toy.py"),
        "--exp-name",
        exp_name,
        "--outdir",
        outdir,
        "--device",
        args.device,
        "--amp-dtype",
        _canonical_amp_dtype(args.amp_dtype),
        "--seed",
        str(seed),
        "--steps",
        str(int(total_steps)),
        "--batch-size",
        str(args.batch_size),
        "--baseline-train-batch-gpu",
        str(args.baseline_train_batch_gpu),
        "--log-every",
        str(args.log_every),
        "--eval-samples",
        str(args.eval_samples),
        "--fid-samples",
        str(args.fid_samples),
        "--debug-eval-batch",
        str(args.debug_eval_batch),
        "--debug-terminal-step",
        str(args.debug_terminal_step),
        "--n-steps-path",
        str(n_steps_path_value),
        "--hidden-dim",
        str(args.hidden_dim),
        "--image-backbone",
        str(args.image_backbone),
        "--baseline-train-backend",
        str(args.baseline_train_backend),
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
        "--fid-ref-path",
        args.fid_ref_path,
        "--training-objective",
        str(args.training_objective),
        "--method-version",
        method_name,
        "--disable-baseline-gate",
        "--skip-checks",
        "--weighted-compute-calibration-path",
        str(args.weighted_compute_calibration_path).strip(),
    ]
    if compute_fid:
        cmd.append("--compute-fid")
    if bool(getattr(args, "wandb", False)):
        cmd.extend(
            [
                "--wandb",
                "--wandb-project",
                str(args.wandb_project),
                "--wandb-entity",
                str(args.wandb_entity),
            ]
        )
    _append_rf_cli_args(cmd, args)
    append_ema_cli_args(cmd, args)
    if fixed_warmup_steps is not None and int(fixed_warmup_steps) > 0:
        cmd.extend(
            [
                "--baseline-steps-override",
                str(int(fixed_warmup_steps)),
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
    if baseline_ckpt_path:
        cmd.extend(
            [
                "--baseline-ckpt-path",
                baseline_ckpt_path,
                "--disable-baseline-ckpt-strict-meta",
            ]
        )
    else:
        # Keep checkpoint lineage explicit: only the provided warmup/support checkpoint
        # or same-method resume checkpoint may seed a later run.
        cmd.append("--disable-baseline-ckpt")
    if resume_path and os.path.exists(resume_path):
        cmd.extend(["--robust-resume-ckpt-path", resume_path])
    if save_path:
        cmd.extend(["--robust-save-ckpt-path", save_path])
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
                "--cdro-edm-ladder-mode",
                str(args.cdro_edm_ladder_mode),
                "--cdro-warmup-fraction",
                str(args.cdro_warmup_fraction),
                "--disable-collapse-diagnostics",
            ]
        )
    else:
        raise ValueError(f"Unsupported method_name: {method_name}")
    return cmd


def _trajectory_checkpoint_path(*, trajectory_dir: str, method_name: str, seed: int, total_steps: int) -> str:
    ckpt_dir = os.path.join(trajectory_dir, "_trajectory_checkpoints")
    ensure_dir(ckpt_dir)
    return os.path.join(ckpt_dir, f"{method_name}_s{int(seed)}_step{int(total_steps):06d}.pt")


def _run_method_local_warmup_trajectory(
    *,
    args: argparse.Namespace,
    method_name: str,
    method_root: str,
    seed: int,
    weighted_grid_targets: List[float],
    comparison_steps: List[int],
    fid_eval_steps: List[int],
    skip_fid_eval: bool,
    fixed_warmup_steps: int,
    trajectory_total_steps_max: int,
    reuse_runs_csv: str = "",
    reuse_aggregate_csv: str = "",
) -> Tuple[List[Dict], Optional[str], Dict]:
    warmup_checkpoint_steps = sorted(
        {
            int(step)
            for step in list(comparison_steps) + [int(fixed_warmup_steps)]
            if int(step) > 0 and int(step) <= int(fixed_warmup_steps)
        }
    )
    if bool(skip_fid_eval):
        warmup_fid_eval_steps = []
    elif str(args.fid_eval_template) == "all":
        warmup_fid_eval_steps = list(warmup_checkpoint_steps)
    else:
        warmup_fid_eval_steps = sorted(
            {
                int(step)
                for step in fid_eval_steps
                if int(step) > 0 and int(step) in set(warmup_checkpoint_steps)
            }
        )
    if not warmup_checkpoint_steps:
        return (
            [],
            None,
            {
                "runs_csv": None,
                "warmup_eval_steps": [],
                "warmup_checkpoint_steps": [],
                "warmup_fid_eval_steps": [],
                "support_checkpoint_path": None,
            },
        )

    seed_root = os.path.join(method_root, f"s{seed}")
    warmup_outdir = os.path.join(seed_root, "_warmup_baseline")
    ensure_dir(warmup_outdir)
    warmup_prefix = f"{args.prefix}_{method_name}_s{seed}_warmup"
    warmup_runs_csv = str(reuse_runs_csv).strip() or os.path.join(warmup_outdir, f"{warmup_prefix}_runs.csv")
    warmup_agg_csv = str(reuse_aggregate_csv).strip() or os.path.join(warmup_outdir, f"{warmup_prefix}_aggregate.csv")
    warmup_log = os.path.join(args.outdir, "logs", f"{warmup_prefix}.log")
    warmup_cmd = _build_baseline_sweep_cmd(
        args=args,
        outdir=warmup_outdir,
        prefix=warmup_prefix,
        seeds_text=str(int(seed)),
        steps=warmup_checkpoint_steps,
        fid_eval_steps=warmup_fid_eval_steps,
    )
    if str(reuse_runs_csv).strip():
        print(
            f"[collect-weighted] reuse {method_name} warmup runs csv: {warmup_runs_csv}",
            flush=True,
        )
    elif args.skip_existing and os.path.isfile(warmup_runs_csv) and os.path.isfile(warmup_agg_csv):
        print(
            f"[collect-weighted] reuse {method_name} warmup seed={seed}: {warmup_outdir}",
            flush=True,
        )
    else:
        print(
            f"[collect-weighted] {method_name} warmup seed={seed}"
            f" checkpoint_steps={warmup_checkpoint_steps}"
            f" fid_eval_steps={warmup_fid_eval_steps}",
            flush=True,
        )
        run_command(
            cmd=warmup_cmd,
            log_path=warmup_log,
            proc_title=build_process_title("wdiff", "warmup", method_name, f"s{int(seed)}"),
        )

    warmup_rows = _normalize_baseline_runs(runs_csv=warmup_runs_csv, args=args)
    warmup_by_step = {(int(row["seed"]), int(row["step"])): row for row in warmup_rows}
    support_row = warmup_by_step.get((int(seed), int(fixed_warmup_steps)))
    if support_row is None:
        raise RuntimeError(
            f"Missing {method_name} warmup support checkpoint for seed={seed} step={fixed_warmup_steps} "
            f"in {warmup_runs_csv}"
        )
    support_checkpoint_path = support_row.get("checkpoint_path")
    if not support_checkpoint_path:
        raise RuntimeError(
            f"Warmup support checkpoint path missing for {method_name} seed={seed} step={fixed_warmup_steps}"
        )
    if not os.path.isfile(support_checkpoint_path):
        raise RuntimeError(
            f"Warmup support checkpoint does not exist for {method_name} seed={seed}: {support_checkpoint_path}"
        )

    comparison_rows: List[Dict] = []
    warmup_fid_eval_step_set = set(int(step) for step in fid_eval_steps)
    for target_weighted, total_steps in zip(weighted_grid_targets, comparison_steps):
        if int(total_steps) <= 0 or int(total_steps) > int(fixed_warmup_steps):
            continue
        baseline_row = warmup_by_step.get((int(seed), int(total_steps)))
        if baseline_row is None:
            raise RuntimeError(
                f"Missing {method_name} warmup comparison row for seed={seed} step={total_steps} in {warmup_runs_csv}"
            )
        fid_selected = bool(int(total_steps) in warmup_fid_eval_step_set)
        comparison_rows.append(
            _transfer_baseline_row_to_method(
                baseline_row=baseline_row,
                method_name=method_name,
                weighted_grid_target=float(target_weighted),
                fid_selected=bool(fid_selected),
                trajectory_total_steps_max=int(trajectory_total_steps_max),
                fixed_warmup_steps=int(fixed_warmup_steps),
                row_origin="trajectory_warmup_phase",
            )
        )
    if int(fixed_warmup_steps) not in {int(row["step"]) for row in comparison_rows}:
        comparison_rows.append(
            _transfer_baseline_row_to_aux_warmup_support(
                baseline_row=support_row,
                method_name=method_name,
                trajectory_total_steps_max=int(trajectory_total_steps_max),
                fixed_warmup_steps=int(fixed_warmup_steps),
            )
        )
        comparison_rows.sort(key=lambda row: int(row["step"]))

    manifest_entry = {
        "outdir": warmup_outdir,
        "runs_csv": warmup_runs_csv,
        "aggregate_csv": warmup_agg_csv,
        "warmup_eval_steps": [int(step) for step in warmup_checkpoint_steps],
        "warmup_checkpoint_steps": [int(step) for step in warmup_checkpoint_steps],
        "warmup_fid_eval_steps": [int(step) for step in warmup_fid_eval_steps],
        "support_checkpoint_step": int(fixed_warmup_steps),
        "support_checkpoint_path": support_checkpoint_path,
        "reused_runs_csv": str(reuse_runs_csv).strip() or None,
        "reused_aggregate_csv": str(reuse_aggregate_csv).strip() or None,
    }
    return comparison_rows, support_checkpoint_path, manifest_entry


def main() -> None:
    args = parse_args()
    _validate_wall_clock_args(args)
    if _APPLIED_PROCESS_TITLE is None:
        apply_process_title(build_process_title("wdiff", "collect", args.prefix))
    baseline_fid_posthoc = str(args.baseline_fid_mode) == "posthoc_from_checkpoints"
    robust_fid_posthoc = str(args.robust_fid_mode) == "posthoc_from_checkpoints"
    any_posthoc_fid = bool(baseline_fid_posthoc or robust_fid_posthoc)
    if any_posthoc_fid and not os.path.isfile(str(args.reeval_script)):
        raise FileNotFoundError(f"Missing reevaluation script for posthoc FID: {args.reeval_script}")
    seeds = parse_int_list(args.seeds)
    ensure_dir(args.outdir)
    logs_dir = os.path.join(args.outdir, "logs")
    ensure_dir(logs_dir)
    grid_template = GRID_TEMPLATES[str(args.grid_template)]

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

    baseline_cap_from_limit = float(int(args.baseline_max_steps)) * float(baseline_step_weighted_units)
    requested_shared_cap = (
        float(baseline_cap_from_limit)
        if float(args.shared_weighted_cap) <= 0.0
        else min(float(args.shared_weighted_cap), float(baseline_cap_from_limit))
    )
    wdro_max_total_steps = _solve_total_steps_for_target_weighted(
        target_weighted_units=float(requested_shared_cap),
        max_total_steps=int(args.wdro_max_total_steps),
        warmup_steps_fn=lambda total: _wdro_warmup_steps_for_total(total_steps=int(total), args=args),
        baseline_step_weighted_units=float(baseline_step_weighted_units),
        robust_step_weighted_units=float(wdro_robust_step_weighted_units),
    )
    wdro_fixed_warmup_steps = _wdro_warmup_steps_for_total(
        total_steps=int(wdro_max_total_steps),
        args=args,
    )
    wdro_shared_cap = _piecewise_weighted_prefix(
        total_steps_prefix=int(wdro_max_total_steps),
        fixed_warmup_steps=int(wdro_fixed_warmup_steps),
        baseline_step_weighted_units=float(baseline_step_weighted_units),
        robust_step_weighted_units=float(wdro_robust_step_weighted_units),
    )
    shared_weighted_cap = min(float(requested_shared_cap), float(wdro_shared_cap))

    cdro_max_total_steps, cdro_fixed_warmup_steps, cdro_shared_cap = _solve_cdro_total_steps_for_shared_cap(
        shared_weighted_cap=float(shared_weighted_cap),
        args=args,
        baseline_step_weighted_units=float(baseline_step_weighted_units),
        wdro_robust_step_weighted_units=float(wdro_robust_step_weighted_units),
        cdro_robust_step_weighted_units=float(cdro_robust_step_weighted_units),
    )
    shared_weighted_cap = min(float(shared_weighted_cap), float(cdro_shared_cap))
    wdro_max_total_steps = _solve_total_steps_for_target_weighted(
        target_weighted_units=float(shared_weighted_cap),
        max_total_steps=int(args.wdro_max_total_steps),
        warmup_steps_fn=lambda total: _wdro_warmup_steps_for_total(total_steps=int(total), args=args),
        baseline_step_weighted_units=float(baseline_step_weighted_units),
        robust_step_weighted_units=float(wdro_robust_step_weighted_units),
    )
    wdro_fixed_warmup_steps = _wdro_warmup_steps_for_total(
        total_steps=int(wdro_max_total_steps),
        args=args,
    )
    wdro_shared_cap = _piecewise_weighted_prefix(
        total_steps_prefix=int(wdro_max_total_steps),
        fixed_warmup_steps=int(wdro_fixed_warmup_steps),
        baseline_step_weighted_units=float(baseline_step_weighted_units),
        robust_step_weighted_units=float(wdro_robust_step_weighted_units),
    )

    weighted_grid_targets = [
        float(target)
        for target in realize_shared_weighted_grid(
            shared_weighted_cap=float(shared_weighted_cap),
            template_steps=list(grid_template["steps"]),
        )
        # Raw collection should only request measured checkpoints, not a synthetic step-0 origin.
        if float(target) > 0.0
    ]
    baseline_curve_steps = [
        _solve_baseline_steps_for_target_weighted(
            target_weighted_units=float(target),
            baseline_step_weighted_units=float(baseline_step_weighted_units),
            max_total_steps=int(args.baseline_max_steps),
        )
        for target in weighted_grid_targets
    ]
    wdro_curve_steps = [
        _solve_prefix_steps_for_target_weighted(
            target_weighted_units=float(target),
            max_total_steps=int(wdro_max_total_steps),
            fixed_warmup_steps=int(wdro_fixed_warmup_steps),
            baseline_step_weighted_units=float(baseline_step_weighted_units),
            robust_step_weighted_units=float(wdro_robust_step_weighted_units),
        )
        for target in weighted_grid_targets
    ]
    cdro_curve_steps = [
        _solve_prefix_steps_for_target_weighted(
            target_weighted_units=float(target),
            max_total_steps=int(cdro_max_total_steps),
            fixed_warmup_steps=int(cdro_fixed_warmup_steps),
            baseline_step_weighted_units=float(baseline_step_weighted_units),
            robust_step_weighted_units=float(cdro_robust_step_weighted_units),
        )
        for target in weighted_grid_targets
    ]
    fid_eval_schedule = _select_fid_eval_indices(
        checkpoint_targets=weighted_grid_targets,
        fid_eval_template_key=str(args.fid_eval_template),
        shared_weighted_cap=float(shared_weighted_cap),
    )
    fid_eval_indices = [int(idx) for idx in fid_eval_schedule["selected_indices"]]
    baseline_fid_eval_steps = sorted(
        {
            int(baseline_curve_steps[idx])
            for idx in fid_eval_indices
            if 0 <= int(idx) < len(baseline_curve_steps) and int(baseline_curve_steps[idx]) > 0
        }
    )
    wdro_fid_eval_steps = sorted(
        {
            int(wdro_curve_steps[idx])
            for idx in fid_eval_indices
            if 0 <= int(idx) < len(wdro_curve_steps) and int(wdro_curve_steps[idx]) > 0
        }
    )
    cdro_fid_eval_steps = sorted(
        {
            int(cdro_curve_steps[idx])
            for idx in fid_eval_indices
            if 0 <= int(idx) < len(cdro_curve_steps) and int(cdro_curve_steps[idx]) > 0
        }
    )
    wdro_fid_eval_steps = _augment_fid_eval_steps_with_transition_sentinels(
        comparison_steps=wdro_curve_steps,
        fid_eval_steps=wdro_fid_eval_steps,
        fixed_warmup_steps=int(wdro_fixed_warmup_steps),
        robust_sentinel_count=int(args.transition_sentinel_robust_count),
    )
    cdro_fid_eval_steps = _augment_fid_eval_steps_with_transition_sentinels(
        comparison_steps=cdro_curve_steps,
        fid_eval_steps=cdro_fid_eval_steps,
        fixed_warmup_steps=int(cdro_fixed_warmup_steps),
        robust_sentinel_count=int(args.transition_sentinel_robust_count),
    )
    wdro_fid_eval_step_set = set(int(step) for step in wdro_fid_eval_steps)
    cdro_fid_eval_step_set = set(int(step) for step in cdro_fid_eval_steps)
    baseline_fid_eval_step_set = set(int(step) for step in baseline_fid_eval_steps)
    baseline_run_steps = sorted({int(step) for step in baseline_curve_steps if int(step) > 0})
    baseline_trajectory_total_steps_max = max(baseline_run_steps) if baseline_run_steps else 0

    print(
        "[collect-weighted] shared_cap="
        f"{shared_weighted_cap:.4f} requested_cap={requested_shared_cap:.4f} "
        f"baseline_cap_limit={baseline_cap_from_limit:.4f} "
        f"wdro_cap={wdro_shared_cap:.4f} cdro_cap={cdro_shared_cap:.4f}",
        flush=True,
    )
    print(
        "[collect-weighted] wdro "
        f"max_total_steps={int(wdro_max_total_steps)} "
        f"warmup_mode=fixed_from_max_budget fixed_warmup_steps={int(wdro_fixed_warmup_steps)}",
        flush=True,
    )
    print(
        "[collect-weighted] cdro "
        f"max_total_steps={int(cdro_max_total_steps)} "
        f"warmup_mode=fixed_from_max_budget fixed_warmup_steps={int(cdro_fixed_warmup_steps)}",
        flush=True,
    )
    print(f"[collect-weighted] weighted_targets={weighted_grid_targets}", flush=True)
    print(
        "[collect-weighted] fid_eval_schedule="
        f"{fid_eval_schedule['template_key']} selected_targets={fid_eval_schedule['selected_targets']}",
        flush=True,
    )
    print(f"[collect-weighted] baseline_fid_mode={args.baseline_fid_mode}", flush=True)
    print(f"[collect-weighted] robust_fid_mode={args.robust_fid_mode}", flush=True)
    print(
        "[collect-weighted] wall_clock_mode="
        f"{args.wall_clock_mode} source={_wall_clock_source_label(args)}",
        flush=True,
    )
    print(f"[collect-weighted] baseline_steps={baseline_curve_steps}", flush=True)
    print(f"[collect-weighted] baseline_fid_eval_steps={baseline_fid_eval_steps}", flush=True)
    print(f"[collect-weighted] wdro_steps={wdro_curve_steps}", flush=True)
    print(f"[collect-weighted] wdro_fid_eval_steps={wdro_fid_eval_steps}", flush=True)
    print(f"[collect-weighted] cdro_steps={cdro_curve_steps}", flush=True)
    print(f"[collect-weighted] cdro_fid_eval_steps={cdro_fid_eval_steps}", flush=True)

    reused_baseline_runs_csv = str(args.reuse_baseline_runs_csv).strip()
    reused_baseline_aggregate_csv = str(args.reuse_baseline_aggregate_csv).strip()
    reused_wdro_raw_csv = str(args.reuse_wdro_raw_csv).strip()
    wdro_enabled = bool(reused_wdro_raw_csv) or str(args.training_objective).strip().lower() != "rf"

    baseline_outdir = os.path.join(args.outdir, "baseline")
    wdro_root = os.path.join(args.outdir, "wdro")
    cdro_root = os.path.join(args.outdir, "cdro")
    ensure_dir(baseline_outdir)
    ensure_dir(wdro_root)
    ensure_dir(cdro_root)

    baseline_prefix = f"{args.prefix}_baseline"
    if reused_baseline_runs_csv:
        baseline_runs_csv = reused_baseline_runs_csv
        baseline_agg_csv = reused_baseline_aggregate_csv
        if not os.path.isfile(baseline_runs_csv):
            raise FileNotFoundError(f"Requested reused baseline runs CSV not found: {baseline_runs_csv}")
        if baseline_agg_csv and not os.path.isfile(baseline_agg_csv):
            raise FileNotFoundError(f"Requested reused baseline aggregate CSV not found: {baseline_agg_csv}")
        print(f"[collect-weighted] reuse baseline runs csv: {baseline_runs_csv}", flush=True)
    else:
        baseline_runs_csv = os.path.join(baseline_outdir, f"{baseline_prefix}_runs.csv")
        baseline_agg_csv = os.path.join(baseline_outdir, f"{baseline_prefix}_aggregate.csv")
        baseline_log = os.path.join(logs_dir, f"{baseline_prefix}.log")
        baseline_cmd = _build_baseline_sweep_cmd(
            args=args,
            outdir=baseline_outdir,
            prefix=baseline_prefix,
            seeds_text=args.seeds,
            steps=baseline_run_steps,
            fid_eval_steps=[] if baseline_fid_posthoc else baseline_fid_eval_steps,
        )
        if args.skip_existing and os.path.isfile(baseline_runs_csv) and os.path.isfile(baseline_agg_csv):
            print(f"[collect-weighted] reuse baseline outputs: {baseline_outdir}", flush=True)
        else:
            print(f"[collect-weighted] baseline seeds={seeds} steps={baseline_run_steps}", flush=True)
            run_command(
                cmd=baseline_cmd,
                log_path=baseline_log,
                proc_title=build_process_title("wdiff", "baseline", args.prefix),
            )

    baseline_raw = _normalize_baseline_runs(runs_csv=baseline_runs_csv, args=args)
    baseline_raw.sort(key=lambda row: (int(row["seed"]), int(row["step"])))
    baseline_by_seed_step = {(int(row["seed"]), int(row["step"])): row for row in baseline_raw}
    baseline_target_rows: List[Dict] = []
    for seed in seeds:
        for target_weighted, total_steps in zip(weighted_grid_targets, baseline_curve_steps):
            baseline_row = baseline_by_seed_step.get((int(seed), int(total_steps)))
            if baseline_row is None:
                raise RuntimeError(f"Missing baseline raw row for seed={seed} step={total_steps}")
            row = dict(baseline_row)
            row["weighted_grid_target"] = float(target_weighted)
            row["comparison_weighted_targets"] = _encode_float_list([float(target_weighted)])
            row["warmup_support_methods"] = ""
            row["row_role"] = _row_role_label(is_comparison_knot=True, is_aux_warmup_support=False)
            row["is_comparison_knot"] = True
            row["is_aux_warmup_support"] = False
            row["trajectory_total_steps_max"] = int(baseline_trajectory_total_steps_max)
            row["fixed_warmup_steps"] = 0
            row.update(
                _resolve_fid_annotation(
                    fid_value=_optional_float(baseline_row.get("fid")),
                    fid_selected=bool(int(total_steps) in baseline_fid_eval_step_set),
                    source_if_evaluated=str(baseline_row.get("fid_source", "in_run_metrics") or "in_run_metrics"),
                )
            )
            baseline_target_rows.append(row)
    baseline_all_eval_rows = list(baseline_target_rows)

    wdro_rows: List[Dict] = []
    cdro_rows: List[Dict] = []
    wdro_seed_manifests: List[Dict] = []
    cdro_seed_manifests: List[Dict] = []

    if not wdro_enabled:
        print(
            "[collect-weighted] skip wdro for training_objective=rf "
            "(Wild-Diffusion-RF is optional and not implemented in this milestone)",
            flush=True,
        )
    elif reused_wdro_raw_csv:
        if not os.path.isfile(reused_wdro_raw_csv):
            raise FileNotFoundError(f"Requested reused WDRO raw CSV not found: {reused_wdro_raw_csv}")
        reused_rows = [
            _apply_wall_clock_accounting(row=_ensure_row_fid_fields(row), args=args)
            for row in load_csv_rows(reused_wdro_raw_csv)
            if _canonical_robust_method(str(row.get("method", ""))) == "wild_diffusion" and int(row["seed"]) in seeds
        ]
        if not reused_rows:
            raise RuntimeError(f"No WDRO rows found in reused raw CSV for seeds={seeds}: {reused_wdro_raw_csv}")
        wdro_rows.extend(reused_rows)
        wdro_rows.sort(key=lambda row: (int(row["seed"]), int(row["step"])))
        wdro_seed_manifests = [
            {
                "seed": int(seed),
                "reused_raw_csv": reused_wdro_raw_csv,
            }
            for seed in seeds
        ]
        print(f"[collect-weighted] reuse wdro raw csv: {reused_wdro_raw_csv}", flush=True)

    for seed in seeds:
        if wdro_enabled and not reused_wdro_raw_csv:
            wdro_outdir = os.path.join(wdro_root, f"s{seed}")
            ensure_dir(wdro_outdir)
            wdro_warmup_rows, wdro_support_checkpoint, wdro_warmup_manifest = _run_method_local_warmup_trajectory(
                args=args,
                method_name="wdro",
                method_root=wdro_root,
                seed=int(seed),
                weighted_grid_targets=weighted_grid_targets,
                comparison_steps=wdro_curve_steps,
                fid_eval_steps=wdro_fid_eval_steps,
                skip_fid_eval=bool(robust_fid_posthoc),
                fixed_warmup_steps=int(wdro_fixed_warmup_steps),
                trajectory_total_steps_max=int(wdro_max_total_steps),
            )
            wdro_rows.extend(wdro_warmup_rows)
            wdro_seed_manifest = {
                "seed": int(seed),
                "trajectory_total_steps_max": int(wdro_max_total_steps),
                "fixed_warmup_steps": int(wdro_fixed_warmup_steps),
                "warmup": wdro_warmup_manifest,
                "robust_checkpoints": [],
            }
            wdro_resume_checkpoint = None
            for target_weighted, total_steps in zip(weighted_grid_targets, wdro_curve_steps):
                if int(total_steps) <= int(wdro_fixed_warmup_steps):
                    continue
                fid_selected = bool(int(total_steps) in wdro_fid_eval_step_set)
                checkpoint_path = _trajectory_checkpoint_path(
                    trajectory_dir=wdro_outdir,
                    method_name="wdro",
                    seed=int(seed),
                    total_steps=int(total_steps),
                )
                baseline_ckpt_requested = wdro_support_checkpoint if wdro_resume_checkpoint is None else None
                exp_name = f"{args.prefix}_wdro_s{seed}_st{int(total_steps)}"
                metrics_path = os.path.join(wdro_outdir, exp_name, "metrics.json")
                log_path = os.path.join(logs_dir, f"{exp_name}.log")
                if args.skip_existing and os.path.isfile(metrics_path) and os.path.isfile(checkpoint_path):
                    print(
                        f"[collect-weighted] reuse wdro seed={seed} step={total_steps}: {metrics_path}",
                        flush=True,
                    )
                else:
                    print(
                        f"[collect-weighted] wdro seed={seed} step={total_steps} "
                        f"target_weighted={target_weighted:.4f} fixed_warmup_steps={int(wdro_fixed_warmup_steps)} "
                        f"baseline_ckpt={'yes' if baseline_ckpt_requested else 'no'} "
                        f"resume_ckpt={'yes' if wdro_resume_checkpoint else 'no'} "
                        f"fid={'yes' if fid_selected else 'no'}",
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
                        resume_path=wdro_resume_checkpoint,
                        save_path=checkpoint_path,
                        baseline_ckpt_path=baseline_ckpt_requested,
                        compute_fid=(bool(fid_selected) and not bool(robust_fid_posthoc)),
                    )
                    run_command(
                        cmd=cmd,
                        log_path=log_path,
                        proc_title=build_process_title("wdiff", "wdro", f"s{int(seed)}", f"st{int(total_steps)}"),
                    )
                row = _extract_wdro_row(
                    metrics_path=metrics_path,
                    calibration=calibration,
                    train_accelerator_count=int(args.train_accelerator_count),
                    fid_selected=bool(fid_selected),
                    args=args,
                )
                row["checkpoint_path"] = checkpoint_path
                row["baseline_ckpt_requested"] = baseline_ckpt_requested
                row["robust_resume_ckpt_requested"] = wdro_resume_checkpoint
                wdro_rows.append(
                    _normalize_method_metrics_row(
                        row=row,
                        seed=int(seed),
                        method_name="wdro",
                        row_origin="trajectory_robust_phase",
                        weighted_grid_target=float(target_weighted),
                        trajectory_total_steps_max=int(wdro_max_total_steps),
                        fixed_warmup_steps=int(wdro_fixed_warmup_steps),
                    )
                )
                wdro_seed_manifest["robust_checkpoints"].append(
                    {
                        "seed": int(seed),
                        "step": int(total_steps),
                        "fixed_warmup_steps": int(wdro_fixed_warmup_steps),
                        "weighted_grid_target": float(target_weighted),
                        "row_origin": "trajectory_robust_phase",
                        "metrics_path": metrics_path,
                        "checkpoint_path": checkpoint_path,
                        "fid_eval_selected": bool(fid_selected),
                        "fid_evaluated": bool(row.get("fid_evaluated", False)),
                        "fid_source": str(row.get("fid_source", "")),
                        "baseline_ckpt_path": baseline_ckpt_requested,
                        "resume_ckpt_path": wdro_resume_checkpoint,
                    }
                )
                wdro_resume_checkpoint = checkpoint_path
            wdro_seed_manifests.append(wdro_seed_manifest)

        cdro_outdir = os.path.join(cdro_root, f"s{seed}")
        ensure_dir(cdro_outdir)
        cdro_warmup_rows, cdro_support_checkpoint, cdro_warmup_manifest = _run_method_local_warmup_trajectory(
            args=args,
            method_name="cdro",
            method_root=cdro_root,
            seed=int(seed),
            weighted_grid_targets=weighted_grid_targets,
            comparison_steps=cdro_curve_steps,
            fid_eval_steps=cdro_fid_eval_steps,
            skip_fid_eval=bool(robust_fid_posthoc),
            fixed_warmup_steps=int(cdro_fixed_warmup_steps),
            trajectory_total_steps_max=int(cdro_max_total_steps),
            reuse_runs_csv=str(args.reuse_cdro_warmup_runs_csv).strip(),
            reuse_aggregate_csv=str(args.reuse_cdro_warmup_aggregate_csv).strip(),
        )
        cdro_rows.extend(cdro_warmup_rows)
        cdro_seed_manifest = {
            "seed": int(seed),
            "trajectory_total_steps_max": int(cdro_max_total_steps),
            "fixed_warmup_steps": int(cdro_fixed_warmup_steps),
            "warmup": cdro_warmup_manifest,
            "robust_checkpoints": [],
        }
        cdro_resume_checkpoint = None
        for target_weighted, total_steps in zip(weighted_grid_targets, cdro_curve_steps):
            if int(total_steps) <= int(cdro_fixed_warmup_steps):
                continue
            fid_selected = bool(int(total_steps) in cdro_fid_eval_step_set)
            checkpoint_path = _trajectory_checkpoint_path(
                trajectory_dir=cdro_outdir,
                method_name="cdro",
                seed=int(seed),
                total_steps=int(total_steps),
            )
            baseline_ckpt_requested = cdro_support_checkpoint if cdro_resume_checkpoint is None else None
            exp_name = f"{args.prefix}_cdro_s{seed}_st{int(total_steps)}"
            metrics_path = os.path.join(cdro_outdir, exp_name, "metrics.json")
            log_path = os.path.join(logs_dir, f"{exp_name}.log")
            if args.skip_existing and os.path.isfile(metrics_path) and os.path.isfile(checkpoint_path):
                print(
                    f"[collect-weighted] reuse cdro seed={seed} step={total_steps}: {metrics_path}",
                    flush=True,
                )
            else:
                print(
                    f"[collect-weighted] cdro seed={seed} step={total_steps} "
                    f"target_weighted={target_weighted:.4f} fixed_warmup_steps={int(cdro_fixed_warmup_steps)} "
                    f"baseline_ckpt={'yes' if baseline_ckpt_requested else 'no'} "
                    f"resume_ckpt={'yes' if cdro_resume_checkpoint else 'no'} "
                    f"fid={'yes' if fid_selected else 'no'}",
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
                    resume_path=cdro_resume_checkpoint,
                    save_path=checkpoint_path,
                    baseline_ckpt_path=baseline_ckpt_requested,
                    compute_fid=(bool(fid_selected) and not bool(robust_fid_posthoc)),
                )
                run_command(
                    cmd=cmd,
                    log_path=log_path,
                    proc_title=build_process_title("wdiff", "cdro", f"s{int(seed)}", f"st{int(total_steps)}"),
                )
            row = _extract_cdro_row(
                metrics_path=metrics_path,
                baseline_ckpt_requested=baseline_ckpt_requested,
                calibration=calibration,
                train_accelerator_count=int(args.train_accelerator_count),
                fid_selected=bool(fid_selected),
                args=args,
            )
            row["checkpoint_path"] = checkpoint_path
            row["robust_resume_ckpt_requested"] = cdro_resume_checkpoint
            cdro_rows.append(
                _normalize_method_metrics_row(
                    row=row,
                    seed=int(seed),
                    method_name="cdro",
                    row_origin="trajectory_robust_phase",
                    weighted_grid_target=float(target_weighted),
                    trajectory_total_steps_max=int(cdro_max_total_steps),
                    fixed_warmup_steps=int(cdro_fixed_warmup_steps),
                )
            )
            cdro_seed_manifest["robust_checkpoints"].append(
                {
                    "seed": int(seed),
                    "step": int(total_steps),
                    "fixed_warmup_steps": int(cdro_fixed_warmup_steps),
                    "weighted_grid_target": float(target_weighted),
                    "row_origin": "trajectory_robust_phase",
                    "metrics_path": metrics_path,
                    "checkpoint_path": checkpoint_path,
                    "fid_eval_selected": bool(fid_selected),
                    "fid_evaluated": bool(row.get("fid_evaluated", False)),
                    "fid_source": str(row.get("fid_source", "")),
                    "baseline_ckpt_path": baseline_ckpt_requested,
                    "resume_ckpt_path": cdro_resume_checkpoint,
                }
            )
            cdro_resume_checkpoint = checkpoint_path
        cdro_seed_manifests.append(cdro_seed_manifest)

    wdro_rows.sort(key=lambda row: (int(row["seed"]), int(row["step"])))
    cdro_rows.sort(key=lambda row: (int(row["seed"]), int(row["step"])))
    baseline_all_eval_rows = [_ensure_row_fid_fields(row) for row in baseline_all_eval_rows]
    baseline_target_rows = [_ensure_row_fid_fields(row) for row in baseline_target_rows]
    wdro_rows = [_ensure_row_fid_fields(row) for row in wdro_rows]
    cdro_rows = [_ensure_row_fid_fields(row) for row in cdro_rows]
    combined_raw = baseline_target_rows + wdro_rows + cdro_rows
    combined_raw.sort(key=lambda row: (str(row["method"]), int(row["seed"]), int(row["step"])))

    baseline_all_eval_raw_csv = os.path.join(args.outdir, f"{args.prefix}_baseline_all_eval_raw_seed_rows.csv")
    baseline_raw_csv = os.path.join(args.outdir, f"{args.prefix}_baseline_raw_seed_rows.csv")
    wdro_raw_csv = os.path.join(args.outdir, f"{args.prefix}_wdro_raw_seed_rows.csv")
    cdro_raw_csv = os.path.join(args.outdir, f"{args.prefix}_cdro_raw_seed_rows.csv")
    combined_raw_csv = os.path.join(args.outdir, f"{args.prefix}_all_methods_raw_seed_rows.csv")
    write_csv(baseline_all_eval_raw_csv, baseline_all_eval_rows)
    write_csv(baseline_raw_csv, baseline_target_rows)
    write_csv(wdro_raw_csv, wdro_rows)
    write_csv(cdro_raw_csv, cdro_rows)
    write_csv(combined_raw_csv, combined_raw)

    posthoc_reeval_manifest_path = None
    posthoc_reeval_log_path = None
    posthoc_methods: List[str] = []
    if baseline_fid_posthoc:
        posthoc_methods.append("baseline")
    if robust_fid_posthoc:
        if wdro_rows:
            posthoc_methods.append("wild_diffusion")
        if cdro_rows:
            posthoc_methods.append("cdro")
    if posthoc_methods:
        reeval_cmd = _build_posthoc_reeval_cmd(
            args=args,
            combined_csv=combined_raw_csv,
            methods=posthoc_methods,
        )
        if args.skip_existing:
            reeval_cmd.append("--skip-existing")
        posthoc_reeval_log_path = os.path.join(logs_dir, f"{args.prefix}_posthoc_fid.log")
        print(
            "[collect-weighted] posthoc FID"
            f" methods={','.join(posthoc_methods)} combined_csv={combined_raw_csv}",
            flush=True,
        )
        run_command(
            cmd=reeval_cmd,
            log_path=posthoc_reeval_log_path,
            proc_title=build_process_title("wdiff", "reeval", args.prefix),
        )
        posthoc_reeval_manifest_path = os.path.join(args.outdir, f"{args.prefix}_reeval_manifest.json")
        combined_raw = [_ensure_row_fid_fields(row) for row in load_csv_rows(combined_raw_csv)]
        combined_raw.sort(key=lambda row: (str(row["method"]), int(row["seed"]), int(row["step"])))
        baseline_target_rows = [row for row in combined_raw if str(row.get("method", "")) == "baseline"]
        baseline_all_eval_rows = list(baseline_target_rows)
        wdro_rows = [row for row in combined_raw if str(row.get("method", "")) == "wild_diffusion"]
        cdro_rows = [row for row in combined_raw if str(row.get("method", "")) == "cdro"]
        combined_by_key = {
            (
                str(row.get("method", "")),
                int(row.get("seed", 0)),
                int(row.get("step", 0)),
                str(row.get("row_origin", "")),
            ): row
            for row in combined_raw
        }
        for method_name, row_method_name, seed_manifests in (
            ("wdro", "wild_diffusion", wdro_seed_manifests),
            ("cdro", "cdro", cdro_seed_manifests),
        ):
            for seed_manifest in seed_manifests:
                for checkpoint_entry in seed_manifest.get("robust_checkpoints", []):
                    row = combined_by_key.get(
                        (
                            str(row_method_name),
                            int(checkpoint_entry.get("seed", 0)),
                            int(checkpoint_entry.get("step", 0)),
                            "trajectory_robust_phase",
                        )
                    )
                    if row is None:
                        continue
                    checkpoint_entry["fid_eval_selected"] = bool(_optional_bool(row.get("fid_eval_selected")))
                    checkpoint_entry["fid_evaluated"] = bool(_optional_bool(row.get("fid_evaluated")))
                    checkpoint_entry["fid_missing_reason"] = str(row.get("fid_missing_reason", ""))
                    checkpoint_entry["fid_source"] = str(row.get("fid_source", ""))
                    if row.get("reeval_metrics_path"):
                        checkpoint_entry["reeval_metrics_path"] = str(row.get("reeval_metrics_path"))
                    if row.get("reeval_log_path"):
                        checkpoint_entry["reeval_log_path"] = str(row.get("reeval_log_path"))
        write_csv(baseline_all_eval_raw_csv, baseline_all_eval_rows)
        write_csv(baseline_raw_csv, baseline_target_rows)
        write_csv(wdro_raw_csv, wdro_rows)
        write_csv(cdro_raw_csv, cdro_rows)
        write_csv(combined_raw_csv, combined_raw)

    manifest = {
        "protocol": {
            "name": "three_method_single_trajectory_weighted_grid",
            "description": (
                "Raw per-seed data collection on a shared weighted-compute grid. Baseline runs as one native "
                "checkpointed trajectory. WDRO and CDRO each use one same-seed same-method trajectory: a "
                "method-local checkpointed warmup baseline trajectory up to a fixed warmup checkpoint, then a "
                "continued robust trajectory resumed only from that method's own checkpoints. Comparison rows are "
                "emitted only at shared weighted-grid checkpoints, while FID evaluation may run on a coarser "
                "schedule over those saved checkpoints; exact warmup checkpoints are auxiliary support artifacts. "
                "Aggregation is intentionally deferred to later analysis."
            ),
            "seeds": seeds,
            "shared_grid_template_name": str(grid_template["name"]),
            "shared_grid_template_steps": list(grid_template["steps"]),
            "grid_strategy": (
                f"{len(list(grid_template['steps'])) - 1}-knot relative template applied in weighted-compute space"
            ),
            "primary_metric": "weighted_compute_units",
            "secondary_metric": "train_wall_clock_sec",
            "legacy_metric": "batch_equiv_denoiser_evals",
            "wall_clock_accounting_mode": str(args.wall_clock_mode),
            "wall_clock_accounting_source": _wall_clock_source_label(args),
            "wall_clock_sec_per_kimg": (
                None
                if str(args.wall_clock_mode) != "current_sec_per_kimg"
                else float(args.wall_clock_sec_per_kimg)
            ),
            "baseline_aggregate_role": "summary_only_not_used_for_wdro_cdro_raw_collection",
            "baseline_auxiliary_eval_points_enabled": False,
            "method_local_warmup_support_checkpoints_enabled": True,
            "baseline_fid_mode": str(args.baseline_fid_mode),
            "robust_fid_mode": str(args.robust_fid_mode),
            "fid_evaluation_schedule": {
                "template_key": str(fid_eval_schedule["template_key"]),
                "template_name": str(fid_eval_schedule["template_name"]),
                "selection_strategy": str(fid_eval_schedule["selection_strategy"]),
                "requested_relative_steps": list(fid_eval_schedule["requested_relative_steps"]),
                "requested_weighted_targets": list(fid_eval_schedule["requested_targets"]),
                "selected_comparison_indices": list(fid_eval_schedule["selected_indices"]),
                "selected_weighted_targets": list(fid_eval_schedule["selected_targets"]),
                "transition_sentinels_enabled": True,
                "transition_sentinel_support_checkpoint_enabled": True,
                "transition_sentinel_robust_checkpoints_per_method": int(args.transition_sentinel_robust_count),
            },
            "checkpoint_schedule": "every_comparison_checkpoint_plus_exact_warmup_support",
            "implicit_outdir_baseline_cache_reuse": False,
            "cross_seed_checkpoint_reuse": False,
            "cross_method_checkpoint_reuse": False,
            "cross_method_baseline_cache_reuse": False,
            "cross_knot_robust_resume_reuse": "same_seed_same_method_only",
            "cdro_collapse_diagnostics_disabled": True,
            **ema_config_dict(args),
            "baseline_reused_from_existing_runs_csv": bool(reused_baseline_runs_csv),
            "wdro_reused_from_existing_raw_csv": bool(reused_wdro_raw_csv),
            "wdro_enabled": bool(wdro_enabled),
            "wdro_optional_for_rf_first_milestone": bool(str(args.training_objective).strip().lower() == "rf"),
        },
        "weighted_compute": {
            "shared_cap": float(shared_weighted_cap),
            "grid_targets": weighted_grid_targets,
            "baseline_step_weighted_units": float(baseline_step_weighted_units),
            "wdro_expected_robust_step_weighted_units": float(wdro_robust_step_weighted_units),
            "cdro_expected_robust_step_weighted_units": float(cdro_robust_step_weighted_units),
            "cdro_effective_n_steps_path": int(_effective_cdro_n_steps_path(args)),
            "cdro_edm_ladder_mode": str(args.cdro_edm_ladder_mode),
            "calibration": calibration,
        },
        "trajectories": {
            "baseline": {
                "max_total_steps": int(baseline_trajectory_total_steps_max),
                "comparison_steps": baseline_curve_steps,
                "checkpoint_steps": baseline_run_steps,
                "fid_eval_steps": baseline_fid_eval_steps,
            },
            "wdro": {
                "enabled": bool(wdro_enabled),
                "max_total_steps": int(wdro_max_total_steps),
                "warmup_mode": "fixed_from_max_budget_trajectory",
                "fixed_warmup_steps": int(wdro_fixed_warmup_steps),
                "warmup_support_steps": [int(wdro_fixed_warmup_steps)] if int(wdro_fixed_warmup_steps) > 0 else [],
                "checkpoint_steps": wdro_curve_steps,
                "fid_eval_steps": wdro_fid_eval_steps,
            },
            "cdro": {
                "max_total_steps": int(cdro_max_total_steps),
                "warmup_mode": "fixed_from_max_budget_trajectory",
                "fixed_warmup_steps": int(cdro_fixed_warmup_steps),
                "warmup_support_steps": [int(cdro_fixed_warmup_steps)] if int(cdro_fixed_warmup_steps) > 0 else [],
                "checkpoint_steps": cdro_curve_steps,
                "fid_eval_steps": cdro_fid_eval_steps,
            },
        },
        "artifacts": {
            "baseline_runs_csv": baseline_runs_csv,
            "baseline_aggregate_csv": baseline_agg_csv,
            "reused_baseline_runs_csv": reused_baseline_runs_csv or None,
            "reused_baseline_aggregate_csv": reused_baseline_aggregate_csv or None,
            "reused_wdro_raw_csv": reused_wdro_raw_csv or None,
            "baseline_all_eval_raw_csv": baseline_all_eval_raw_csv,
            "baseline_raw_csv": baseline_raw_csv,
            "wdro_raw_csv": wdro_raw_csv,
            "cdro_raw_csv": cdro_raw_csv,
            "combined_raw_csv": combined_raw_csv,
            "posthoc_reeval_manifest": posthoc_reeval_manifest_path,
            "posthoc_reeval_log": posthoc_reeval_log_path,
        },
        "per_seed_runs": {
            "wdro": wdro_seed_manifests,
            "cdro": cdro_seed_manifests,
        },
    }
    manifest_path = os.path.join(args.outdir, f"{args.prefix}_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"[collect-weighted] wrote {baseline_all_eval_raw_csv}", flush=True)
    print(f"[collect-weighted] wrote {baseline_raw_csv}", flush=True)
    print(f"[collect-weighted] wrote {wdro_raw_csv}", flush=True)
    print(f"[collect-weighted] wrote {cdro_raw_csv}", flush=True)
    print(f"[collect-weighted] wrote {combined_raw_csv}", flush=True)
    print(f"[collect-weighted] wrote {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
