#!/usr/bin/env python3
import argparse
import csv
import json
import os
import subprocess
import sys
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


from toy.process_title import apply_process_title, build_process_title, child_process_env


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

METHOD_COLORS = {
    "baseline_edm": "tab:blue",
    "wdro": "tab:orange",
    "cdro": "tab:green",
}
METHOD_LABELS = {
    "baseline_edm": "Baseline EDM",
    "wdro": "WDRO",
    "cdro": "CDRO",
}
LINESTYLES = ["-", "--", "-.", ":"]

CASE_LIBRARY: Dict[str, Dict] = {
    "warm05_default": {
        "label": "warm=5%, aw=0.50, rho=0.02, N=24",
        "wdro_warmup_fraction": 0.05,
        "cdro_warmup_fraction": 0.05,
        "outer_attack_weight": 0.50,
        "cdro_total_budget_rho": 0.02,
        "cdro_n_steps_path": 24,
        "reuse_baseline_from": None,
        "reuse_wdro_from": None,
    },
    "warm10_default": {
        "label": "warm=10%, aw=0.50, rho=0.02, N=24",
        "wdro_warmup_fraction": 0.10,
        "cdro_warmup_fraction": 0.10,
        "outer_attack_weight": 0.50,
        "cdro_total_budget_rho": 0.02,
        "cdro_n_steps_path": 24,
        "reuse_baseline_from": "warm05_default",
        "reuse_wdro_from": None,
    },
    "warm20_default": {
        "label": "warm=20%, aw=0.50, rho=0.02, N=24",
        "wdro_warmup_fraction": 0.20,
        "cdro_warmup_fraction": 0.20,
        "outer_attack_weight": 0.50,
        "cdro_total_budget_rho": 0.02,
        "cdro_n_steps_path": 24,
        "reuse_baseline_from": "warm05_default",
        "reuse_wdro_from": None,
    },
    "warm05_aw1p00": {
        "label": "warm=5%, aw=1.00, rho=0.02, N=24",
        "wdro_warmup_fraction": 0.05,
        "cdro_warmup_fraction": 0.05,
        "outer_attack_weight": 1.00,
        "cdro_total_budget_rho": 0.02,
        "cdro_n_steps_path": 24,
        "reuse_baseline_from": "warm05_default",
        "reuse_wdro_from": "warm05_default",
    },
    "warm05_aw0p25": {
        "label": "warm=5%, aw=0.25, rho=0.02, N=24",
        "wdro_warmup_fraction": 0.05,
        "cdro_warmup_fraction": 0.05,
        "outer_attack_weight": 0.25,
        "cdro_total_budget_rho": 0.02,
        "cdro_n_steps_path": 24,
        "reuse_baseline_from": "warm05_default",
        "reuse_wdro_from": "warm05_default",
    },
    "warm05_rho0p05": {
        "label": "warm=5%, aw=0.50, rho=0.05, N=24",
        "wdro_warmup_fraction": 0.05,
        "cdro_warmup_fraction": 0.05,
        "outer_attack_weight": 0.50,
        "cdro_total_budget_rho": 0.05,
        "cdro_n_steps_path": 24,
        "reuse_baseline_from": "warm05_default",
        "reuse_wdro_from": "warm05_default",
    },
    "warm05_rho0p01": {
        "label": "warm=5%, aw=0.50, rho=0.01, N=24",
        "wdro_warmup_fraction": 0.05,
        "cdro_warmup_fraction": 0.05,
        "outer_attack_weight": 0.50,
        "cdro_total_budget_rho": 0.01,
        "cdro_n_steps_path": 24,
        "reuse_baseline_from": "warm05_default",
        "reuse_wdro_from": "warm05_default",
    },
    "warm05_n12": {
        "label": "warm=5%, aw=0.50, rho=0.02, N=12",
        "wdro_warmup_fraction": 0.05,
        "cdro_warmup_fraction": 0.05,
        "outer_attack_weight": 0.50,
        "cdro_total_budget_rho": 0.02,
        "cdro_n_steps_path": 12,
        "reuse_baseline_from": "warm05_default",
        "reuse_wdro_from": "warm05_default",
    },
    "warm05_n32": {
        "label": "warm=5%, aw=0.50, rho=0.02, N=32",
        "wdro_warmup_fraction": 0.05,
        "cdro_warmup_fraction": 0.05,
        "outer_attack_weight": 0.50,
        "cdro_total_budget_rho": 0.02,
        "cdro_n_steps_path": 32,
        "reuse_baseline_from": "warm05_default",
        "reuse_wdro_from": "warm05_default",
    },
}

PROFILE_CASES = {
    "overnight": [
        "warm05_default",
        "warm10_default",
        "warm20_default",
        "warm05_aw1p00",
        "warm05_rho0p05",
        "warm05_n12",
        "warm05_n32",
    ],
    "extended": [
        "warm05_default",
        "warm10_default",
        "warm20_default",
        "warm05_aw1p00",
        "warm05_aw0p25",
        "warm05_rho0p05",
        "warm05_rho0p01",
        "warm05_n12",
        "warm05_n32",
    ],
    "warmup_only": [
        "warm05_default",
        "warm10_default",
        "warm20_default",
    ],
}

SUMMARY_GROUPS = {
    "warmup": ["warm05_default", "warm10_default", "warm20_default"],
    "cdro_attack_weight": ["warm05_default", "warm05_aw1p00", "warm05_aw0p25"],
    "cdro_rho": ["warm05_default", "warm05_rho0p05", "warm05_rho0p01"],
    "cdro_n_steps": ["warm05_default", "warm05_n12", "warm05_n32"],
}


def _format_token(value: float, digits: int) -> str:
    return f"{float(value):.{digits}f}".replace("-", "m").replace(".", "p")


def _format_percent_label(fraction: float) -> str:
    percent = 100.0 * float(fraction)
    rounded = round(percent)
    if abs(percent - rounded) < 1e-9:
        return f"{int(rounded)}%"
    return f"{percent:.1f}".rstrip("0").rstrip(".") + "%"


def _generated_case_id(
    *,
    warmup_fraction: float,
    outer_attack_weight: float,
    outer_clean_weight: float,
    total_budget_rho: float,
    n_steps_path: int,
) -> str:
    return (
        f"warm{_format_token(warmup_fraction, 3)}"
        f"_aw{_format_token(outer_attack_weight, 2)}"
        f"_cw{_format_token(outer_clean_weight, 2)}"
        f"_rho{_format_token(total_budget_rho, 2)}"
        f"_n{int(n_steps_path)}"
    )


def _generated_case_label(
    *,
    warmup_fraction: float,
    outer_attack_weight: float,
    outer_clean_weight: float,
    total_budget_rho: float,
    n_steps_path: int,
) -> str:
    return (
        f"warm={_format_percent_label(warmup_fraction)}, "
        f"aw={float(outer_attack_weight):.2f}, "
        f"cw={float(outer_clean_weight):.2f}, "
        f"rho={float(total_budget_rho):.2f}, "
        f"N={int(n_steps_path)}"
    )


def _append_unique(values: List[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _register_round2_case(
    *,
    warmup_fraction: float,
    outer_attack_weight: float,
    outer_clean_weight: float,
    total_budget_rho: float,
    n_steps_path: int,
    baseline_anchor: Optional[str],
    wdro_anchor_by_warm: Dict[str, str],
) -> Tuple[str, Optional[str]]:
    case_id = _generated_case_id(
        warmup_fraction=warmup_fraction,
        outer_attack_weight=outer_attack_weight,
        outer_clean_weight=outer_clean_weight,
        total_budget_rho=total_budget_rho,
        n_steps_path=n_steps_path,
    )
    warm_key = _format_token(warmup_fraction, 3)
    is_new = case_id not in CASE_LIBRARY
    if is_new:
        CASE_LIBRARY[case_id] = {
            "label": _generated_case_label(
                warmup_fraction=warmup_fraction,
                outer_attack_weight=outer_attack_weight,
                outer_clean_weight=outer_clean_weight,
                total_budget_rho=total_budget_rho,
                n_steps_path=n_steps_path,
            ),
            "wdro_warmup_fraction": float(warmup_fraction),
            "cdro_warmup_fraction": float(warmup_fraction),
            "outer_attack_weight": float(outer_attack_weight),
            "outer_clean_weight": float(outer_clean_weight),
            "cdro_total_budget_rho": float(total_budget_rho),
            "cdro_n_steps_path": int(n_steps_path),
            "reuse_baseline_from": baseline_anchor,
            "reuse_wdro_from": wdro_anchor_by_warm.get(warm_key),
        }

    if baseline_anchor is None:
        CASE_LIBRARY[case_id]["reuse_baseline_from"] = None
        baseline_anchor = case_id
    if warm_key not in wdro_anchor_by_warm:
        CASE_LIBRARY[case_id]["reuse_wdro_from"] = None
        wdro_anchor_by_warm[warm_key] = case_id
    return case_id, baseline_anchor


def _extend_case_library_for_round2() -> None:
    warmups = [0.0, 0.025, 0.05]
    rhos = [0.01, 0.02, 0.05, 0.10, 0.20]
    n_steps_values = [8, 12, 24, 32, 48]
    attack_weights = [0.25, 0.30, 0.50, 1.0, 2.0, 4.0]
    clean_weights = [1.0, 0.5, 0.0]
    weight_anchors = [
        (0.05, 0.02, 24),
        (0.05, 0.01, 32),
        (0.00, 0.01, 32),
    ]

    baseline_anchor: Optional[str] = None
    wdro_anchor_by_warm: Dict[str, str] = {}
    structural_case_ids: Dict[Tuple[float, float, int], str] = {}
    round2_structural: List[str] = []
    round2_weights: List[str] = []
    round2_full: List[str] = []

    for warmup_fraction in warmups:
        for total_budget_rho in rhos:
            for n_steps_path in n_steps_values:
                case_id, baseline_anchor = _register_round2_case(
                    warmup_fraction=warmup_fraction,
                    outer_attack_weight=0.50,
                    outer_clean_weight=1.0,
                    total_budget_rho=total_budget_rho,
                    n_steps_path=n_steps_path,
                    baseline_anchor=baseline_anchor,
                    wdro_anchor_by_warm=wdro_anchor_by_warm,
                )
                structural_case_ids[(warmup_fraction, total_budget_rho, n_steps_path)] = case_id
                _append_unique(round2_structural, case_id)
                _append_unique(round2_full, case_id)

    for case_id in wdro_anchor_by_warm.values():
        _append_unique(round2_weights, case_id)

    for warmup_fraction, total_budget_rho, n_steps_path in weight_anchors:
        for outer_attack_weight in attack_weights:
            for outer_clean_weight in clean_weights:
                case_id, baseline_anchor = _register_round2_case(
                    warmup_fraction=warmup_fraction,
                    outer_attack_weight=outer_attack_weight,
                    outer_clean_weight=outer_clean_weight,
                    total_budget_rho=total_budget_rho,
                    n_steps_path=n_steps_path,
                    baseline_anchor=baseline_anchor,
                    wdro_anchor_by_warm=wdro_anchor_by_warm,
                )
                _append_unique(round2_weights, case_id)
                _append_unique(round2_full, case_id)

    PROFILE_CASES["round2_structural"] = round2_structural
    PROFILE_CASES["round2_weights"] = round2_weights
    PROFILE_CASES["round2_full"] = round2_full

    SUMMARY_GROUPS.update(
        {
            "round2_warmup_rho0p01_n32": [
                structural_case_ids[(warmup_fraction, 0.01, 32)]
                for warmup_fraction in warmups
            ],
            "round2_rho_warm0p05_n32": [
                structural_case_ids[(0.05, total_budget_rho, 32)]
                for total_budget_rho in rhos
            ],
            "round2_n_warm0p05_rho0p01": [
                structural_case_ids[(0.05, 0.01, n_steps_path)]
                for n_steps_path in n_steps_values
            ],
            "round2_aw_warm0p05_rho0p01_n32": [
                _generated_case_id(
                    warmup_fraction=0.05,
                    outer_attack_weight=outer_attack_weight,
                    outer_clean_weight=1.0,
                    total_budget_rho=0.01,
                    n_steps_path=32,
                )
                for outer_attack_weight in attack_weights
            ],
            "round2_cw_aw0p30_warm0p05_rho0p01_n32": [
                _generated_case_id(
                    warmup_fraction=0.05,
                    outer_attack_weight=0.30,
                    outer_clean_weight=outer_clean_weight,
                    total_budget_rho=0.01,
                    n_steps_path=32,
                )
                for outer_clean_weight in clean_weights
            ],
            "round2_cw_aw4p00_warm0p05_rho0p01_n32": [
                _generated_case_id(
                    warmup_fraction=0.05,
                    outer_attack_weight=4.0,
                    outer_clean_weight=outer_clean_weight,
                    total_budget_rho=0.01,
                    n_steps_path=32,
                )
                for outer_clean_weight in clean_weights
            ],
        }
    )


_extend_case_library_for_round2()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Queue a small overnight single-trajectory ablation matrix for Simpsons-MNIST RGB 1%, "
            "generate per-case figures, and emit compact family comparison figures."
        )
    )
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--prefix", type=str, required=True)
    parser.add_argument("--profile", type=str, choices=sorted(PROFILE_CASES.keys()), default="overnight")
    parser.add_argument("--case-ids", type=str, default="")
    parser.add_argument("--python-bin", type=str, default=os.path.join(ROOT_DIR, ".venv", "bin", "python"))
    parser.add_argument("--collector-script", type=str, default=os.path.join(ROOT_DIR, "toy", "scripts", "collect_three_method_seed_data.py"))
    parser.add_argument("--plot-script", type=str, default=os.path.join(ROOT_DIR, "toy", "scripts", "plot_three_method_fid_curves.py"))
    parser.add_argument("--seeds", type=str, default="0")
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
    parser.add_argument("--image-split-seed", type=int, default=0)
    parser.add_argument("--train-percent-label", type=str, default="1%")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--eval-samples", type=int, default=2000)
    parser.add_argument("--fid-samples", type=int, default=2000)
    parser.add_argument("--debug-eval-batch", type=int, default=64)
    parser.add_argument("--debug-terminal-step", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--n-steps-path", type=int, default=24)
    parser.add_argument("--sigma-min", type=float, default=0.002)
    parser.add_argument("--sigma-max", type=float, default=2.0)
    parser.add_argument("--shared-weighted-cap", type=float, default=200000.0)
    parser.add_argument("--grid-template", type=str, choices=["standard", "denser"], default="denser")
    parser.add_argument("--baseline-max-steps", type=int, default=80000)
    parser.add_argument("--wdro-max-total-steps", type=int, default=76970)
    parser.add_argument("--wdro-refresh-epochs", type=float, default=100.0)
    parser.add_argument("--wdro-adv-prob", type=float, default=0.3)
    parser.add_argument("--wdro-attack-steps", type=int, default=2)
    parser.add_argument("--wdro-attack-step-size", type=float, default=1e-3)
    parser.add_argument("--wdro-gamma", type=float, default=1.0)
    parser.add_argument("--inner-steps", type=int, default=1)
    parser.add_argument("--outer-clean-weight", type=float, default=1.0)
    parser.add_argument("--cdro-step-size", type=float, default=0.02)
    parser.add_argument("--cdro-time-horizon", type=float, default=1.0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _safe_float(value) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_case_ids(args: argparse.Namespace) -> List[str]:
    if str(args.case_ids).strip():
        case_ids = [token.strip() for token in str(args.case_ids).split(",") if token.strip()]
    else:
        case_ids = list(PROFILE_CASES[str(args.profile)])
    unknown = [case_id for case_id in case_ids if case_id not in CASE_LIBRARY]
    if unknown:
        raise ValueError(f"Unknown case ids: {unknown}")
    return case_ids


def _case_prefix(*, base_prefix: str, case_id: str) -> str:
    return f"{base_prefix}_{case_id}"


def _case_outdir(*, base_outdir: str, case_id: str) -> str:
    return os.path.join(base_outdir, case_id)


def _case_artifact_paths(*, base_outdir: str, base_prefix: str, case_id: str) -> Dict[str, str]:
    case_outdir = _case_outdir(base_outdir=base_outdir, case_id=case_id)
    case_prefix = _case_prefix(base_prefix=base_prefix, case_id=case_id)
    return {
        "case_outdir": case_outdir,
        "case_prefix": case_prefix,
        "combined_csv": os.path.join(case_outdir, f"{case_prefix}_all_methods_raw_seed_rows.csv"),
        "manifest_json": os.path.join(case_outdir, f"{case_prefix}_manifest.json"),
        "baseline_runs_csv": os.path.join(case_outdir, "baseline", f"{case_prefix}_baseline_runs.csv"),
        "baseline_aggregate_csv": os.path.join(case_outdir, "baseline", f"{case_prefix}_baseline_aggregate.csv"),
        "wdro_raw_csv": os.path.join(case_outdir, f"{case_prefix}_wdro_raw_seed_rows.csv"),
        "plots_dir": os.path.join(case_outdir, "plots"),
        "plots_manifest": os.path.join(case_outdir, "plots", f"{case_prefix}_three_method_compare_summary.json"),
    }


def _build_collector_cmd(*, args: argparse.Namespace, case_id: str, case_cfg: Dict) -> List[str]:
    paths = _case_artifact_paths(base_outdir=args.outdir, base_prefix=args.prefix, case_id=case_id)
    cmd = [
        args.python_bin,
        args.collector_script,
        "--outdir",
        paths["case_outdir"],
        "--prefix",
        paths["case_prefix"],
        "--seeds",
        args.seeds,
        "--device",
        args.device,
        "--dataset-path",
        args.dataset_path,
        "--dataset-val-path",
        args.dataset_val_path,
        "--fid-ref-path",
        args.fid_ref_path,
        "--weighted-compute-calibration-path",
        args.weighted_compute_calibration_path,
        "--weighted-inputgrad-alpha",
        str(args.weighted_inputgrad_alpha),
        "--weighted-parambackward-beta",
        str(args.weighted_parambackward_beta),
        "--train-accelerator-count",
        str(args.train_accelerator_count),
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
        "--train-percent-label",
        args.train_percent_label,
        "--batch-size",
        str(args.batch_size),
        "--hidden-dim",
        str(args.hidden_dim),
        "--eval-samples",
        str(args.eval_samples),
        "--fid-samples",
        str(args.fid_samples),
        "--debug-eval-batch",
        str(args.debug_eval_batch),
        "--debug-terminal-step",
        str(args.debug_terminal_step),
        "--log-every",
        str(args.log_every),
        "--n-steps-path",
        str(args.n_steps_path),
        "--sigma-min",
        str(args.sigma_min),
        "--sigma-max",
        str(args.sigma_max),
        "--shared-weighted-cap",
        str(args.shared_weighted_cap),
        "--grid-template",
        str(args.grid_template),
        "--baseline-max-steps",
        str(args.baseline_max_steps),
        "--wdro-max-total-steps",
        str(args.wdro_max_total_steps),
        "--wdro-warmup-fraction",
        str(case_cfg["wdro_warmup_fraction"]),
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
        "--inner-steps",
        str(args.inner_steps),
        "--outer-attack-weight",
        str(case_cfg["outer_attack_weight"]),
        "--outer-clean-weight",
        str(case_cfg.get("outer_clean_weight", args.outer_clean_weight)),
        "--cdro-step-size",
        str(args.cdro_step_size),
        "--cdro-total-budget-rho",
        str(case_cfg["cdro_total_budget_rho"]),
        "--cdro-time-horizon",
        str(args.cdro_time_horizon),
        "--cdro-warmup-fraction",
        str(case_cfg["cdro_warmup_fraction"]),
        "--cdro-n-steps-path",
        str(case_cfg["cdro_n_steps_path"]),
    ]
    if args.skip_existing:
        cmd.append("--skip-existing")

    reuse_baseline_from = case_cfg.get("reuse_baseline_from")
    if reuse_baseline_from:
        ref_paths = _case_artifact_paths(base_outdir=args.outdir, base_prefix=args.prefix, case_id=str(reuse_baseline_from))
        cmd.extend(
            [
                "--reuse-baseline-runs-csv",
                ref_paths["baseline_runs_csv"],
                "--reuse-baseline-aggregate-csv",
                ref_paths["baseline_aggregate_csv"],
            ]
        )

    reuse_wdro_from = case_cfg.get("reuse_wdro_from")
    if reuse_wdro_from:
        ref_paths = _case_artifact_paths(base_outdir=args.outdir, base_prefix=args.prefix, case_id=str(reuse_wdro_from))
        cmd.extend(["--reuse-wdro-raw-csv", ref_paths["wdro_raw_csv"]])

    return cmd


def _build_plot_cmd(*, args: argparse.Namespace, case_id: str) -> List[str]:
    paths = _case_artifact_paths(base_outdir=args.outdir, base_prefix=args.prefix, case_id=case_id)
    return [
        args.python_bin,
        args.plot_script,
        "--combined-csv",
        paths["combined_csv"],
        "--outdir",
        paths["plots_dir"],
        "--prefix",
        paths["case_prefix"],
        "--train-percent-label",
        args.train_percent_label,
    ]


def _run_logged_command(*, cmd: List[str], log_path: str, dry_run: bool, proc_title: Optional[str] = None) -> None:
    ensure_dir(os.path.dirname(log_path))
    if dry_run:
        print("[dry-run]", " ".join(cmd), flush=True)
        return
    with open(log_path, "w", encoding="utf-8") as handle:
        subprocess.run(
            cmd,
            cwd=ROOT_DIR,
            check=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=child_process_env(proc_title=proc_title),
        )


def _load_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: str, rows: List[Dict]) -> None:
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


def _method_rows(rows: List[Dict[str, str]], method_name: str) -> List[Dict[str, str]]:
    out = [row for row in rows if str(row.get("method", "")) == method_name]
    out.sort(key=lambda row: (_safe_float(row.get("weighted_compute_units")) or float("inf"), int(row.get("step", 0))))
    return out


def _best_row(rows: List[Dict[str, str]]) -> Optional[Dict[str, str]]:
    finite_rows = [row for row in rows if _safe_float(row.get("fid")) is not None]
    if not finite_rows:
        return None
    return min(finite_rows, key=lambda row: float(row["fid"]))


def _final_row(rows: List[Dict[str, str]]) -> Optional[Dict[str, str]]:
    finite_rows = [row for row in rows if _safe_float(row.get("weighted_compute_units")) is not None]
    if not finite_rows:
        return None
    return max(finite_rows, key=lambda row: float(row["weighted_compute_units"]))


def _first_robust_row(rows: List[Dict[str, str]]) -> Optional[Dict[str, str]]:
    robust_rows = [row for row in rows if str(row.get("row_origin", "")) == "trajectory_robust_phase"]
    if not robust_rows:
        return None
    return min(robust_rows, key=lambda row: float(row["weighted_compute_units"]))


def _best_robust_row(rows: List[Dict[str, str]]) -> Optional[Dict[str, str]]:
    robust_rows = [row for row in rows if str(row.get("row_origin", "")) == "trajectory_robust_phase"]
    return _best_row(robust_rows)


ROW_DIAGNOSTIC_FIELDS = [
    "outer_loss_final",
    "outer_loss_mean_last",
    "outer_loss_attack_final",
    "outer_loss_attack_mean_last",
    "outer_loss_clean_final",
    "outer_loss_clean_mean_last",
    "attack_metric_kind",
    "attack_metric_final",
    "attack_metric_mean_last",
    "transport_cost_kind",
    "transport_cost_final",
    "transport_cost_mean_last",
    "delta_norm_mean_final",
    "delta_norm_max_final",
    "delta_norm_ratio_mean_final",
    "delta_norm_ratio_max_final",
    "sched_attack_weight_final",
    "sched_clean_weight_final",
    "diag_delta_norm_ratio_mean_final",
    "diag_inner_obj_gap_ratio_final",
]
ROW_DIAGNOSTIC_STRING_FIELDS = {"attack_metric_kind", "transport_cost_kind"}


def _row_diagnostic_fields(*, prefix: str, row: Optional[Dict[str, str]]) -> Dict:
    out: Dict[str, Optional[float]] = {}
    for key in ROW_DIAGNOSTIC_FIELDS:
        field_name = f"{prefix}_{key}"
        if row is None:
            out[field_name] = None
        elif key in ROW_DIAGNOSTIC_STRING_FIELDS:
            value = row.get(key)
            out[field_name] = None if value in (None, "") else str(value)
        else:
            out[field_name] = _safe_float(row.get(key))
    return out


def _summarize_case(*, case_id: str, case_cfg: Dict, combined_csv: str) -> List[Dict]:
    rows = _load_rows(combined_csv)
    summary_rows: List[Dict] = []
    for method_name in ("baseline_edm", "wdro", "cdro"):
        method_rows = _method_rows(rows, method_name)
        if not method_rows:
            continue
        best_row = _best_row(method_rows)
        final_row = _final_row(method_rows)
        first_robust_row = _first_robust_row(method_rows)
        best_robust_row = _best_robust_row(method_rows)
        summary_rows.append(
            {
                "case_id": case_id,
                "case_label": case_cfg["label"],
                "method": method_name,
                "wdro_warmup_fraction": case_cfg["wdro_warmup_fraction"],
                "cdro_warmup_fraction": case_cfg["cdro_warmup_fraction"],
                "cdro_outer_attack_weight": case_cfg["outer_attack_weight"],
                "cdro_outer_clean_weight": case_cfg.get("outer_clean_weight", 1.0),
                "cdro_total_budget_rho": case_cfg["cdro_total_budget_rho"],
                "cdro_n_steps_path": case_cfg["cdro_n_steps_path"],
                "best_fid": None if best_row is None else float(best_row["fid"]),
                "best_fid_weighted_compute": None if best_row is None else float(best_row["weighted_compute_units"]),
                "best_fid_train_wall_clock_sec": None if best_row is None else _safe_float(best_row.get("train_wall_clock_sec")),
                "final_fid": None if final_row is None else float(final_row["fid"]),
                "final_weighted_compute": None if final_row is None else float(final_row["weighted_compute_units"]),
                "final_train_wall_clock_sec": None if final_row is None else _safe_float(final_row.get("train_wall_clock_sec")),
                "warmup_end_weighted_compute": (
                    None
                    if first_robust_row is None
                    else _safe_float(first_robust_row.get("baseline_weighted_compute_units"))
                ),
                "best_robust_fid": None if best_robust_row is None else float(best_robust_row["fid"]),
                "best_robust_fid_weighted_compute": (
                    None if best_robust_row is None else float(best_robust_row["weighted_compute_units"])
                ),
                **_row_diagnostic_fields(prefix="best_fid", row=best_row),
                **_row_diagnostic_fields(prefix="final", row=final_row),
            }
        )
    return summary_rows


def _plot_curve(*, ax, rows: List[Dict[str, str]], method_name: str, label: str, color: str, linestyle: str) -> None:
    method_rows = _method_rows(rows, method_name)
    xs = [_safe_float(row.get("weighted_compute_units")) for row in method_rows]
    ys = [_safe_float(row.get("fid")) for row in method_rows]
    pairs = [(x_value, y_value) for x_value, y_value in zip(xs, ys) if x_value is not None and y_value is not None]
    if not pairs:
        return
    ax.plot(
        [x_value for x_value, _ in pairs],
        [y_value for _, y_value in pairs],
        marker="o",
        linewidth=2.0,
        markersize=4.0,
        linestyle=linestyle,
        color=color,
        label=label,
    )


def _make_family_plot(
    *,
    group_name: str,
    case_ids: List[str],
    case_rows: Dict[str, List[Dict[str, str]]],
    case_cfgs: Dict[str, Dict],
    outdir: str,
) -> Optional[str]:
    available_case_ids = [case_id for case_id in case_ids if case_id in case_rows]
    if len(available_case_ids) < 2:
        return None

    ref_case_id = available_case_ids[0]
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    _plot_curve(
        ax=ax,
        rows=case_rows[ref_case_id],
        method_name="baseline_edm",
        label=METHOD_LABELS["baseline_edm"],
        color=METHOD_COLORS["baseline_edm"],
        linestyle="-",
    )

    wdro_per_case = len({case_cfgs[case_id]["wdro_warmup_fraction"] for case_id in available_case_ids}) > 1
    if wdro_per_case:
        for case_index, case_id in enumerate(available_case_ids):
            linestyle = LINESTYLES[case_index % len(LINESTYLES)]
            case_label = case_cfgs[case_id]["label"]
            _plot_curve(
                ax=ax,
                rows=case_rows[case_id],
                method_name="wdro",
                label=f"WDRO {case_label}",
                color=METHOD_COLORS["wdro"],
                linestyle=linestyle,
            )
            _plot_curve(
                ax=ax,
                rows=case_rows[case_id],
                method_name="cdro",
                label=f"CDRO {case_label}",
                color=METHOD_COLORS["cdro"],
                linestyle=linestyle,
            )
    else:
        _plot_curve(
            ax=ax,
            rows=case_rows[ref_case_id],
            method_name="wdro",
            label=METHOD_LABELS["wdro"],
            color=METHOD_COLORS["wdro"],
            linestyle="-",
        )
        for case_index, case_id in enumerate(available_case_ids):
            linestyle = LINESTYLES[case_index % len(LINESTYLES)]
            case_label = case_cfgs[case_id]["label"]
            _plot_curve(
                ax=ax,
                rows=case_rows[case_id],
                method_name="cdro",
                label=f"CDRO {case_label}",
                color=METHOD_COLORS["cdro"],
                linestyle=linestyle,
            )

    ax.set_xlabel("Weighted Compute Units")
    ax.set_ylabel("FID")
    ax.set_title(f"Simpsons-MNIST RGB 1%: {group_name.replace('_', ' ').title()} Sweep")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()

    ensure_dir(outdir)
    plot_path = os.path.join(outdir, f"{group_name}_fid_vs_weighted_compute.png")
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    return plot_path


def main() -> None:
    args = parse_args()
    if _APPLIED_PROCESS_TITLE is None:
        apply_process_title(build_process_title("wdiff", "queue", args.prefix))
    case_ids = _parse_case_ids(args)
    ensure_dir(args.outdir)
    logs_dir = os.path.join(args.outdir, "logs")
    ensure_dir(logs_dir)

    case_statuses: List[Dict] = []
    case_rows: Dict[str, List[Dict[str, str]]] = {}
    summary_rows: List[Dict] = []

    for case_id in case_ids:
        case_cfg = dict(CASE_LIBRARY[case_id])
        paths = _case_artifact_paths(base_outdir=args.outdir, base_prefix=args.prefix, case_id=case_id)
        collector_log = os.path.join(logs_dir, f"{case_id}_collector.log")
        plot_log = os.path.join(logs_dir, f"{case_id}_plot.log")

        print(f"[matrix] case={case_id} label={case_cfg['label']}", flush=True)
        collector_cmd = _build_collector_cmd(args=args, case_id=case_id, case_cfg=case_cfg)
        _run_logged_command(
            cmd=collector_cmd,
            log_path=collector_log,
            dry_run=args.dry_run,
            proc_title=build_process_title("wdiff", "collect", case_id),
        )

        if not args.dry_run:
            if not os.path.isfile(paths["combined_csv"]):
                raise RuntimeError(f"Missing combined csv after collector run: {paths['combined_csv']}")
            plot_cmd = _build_plot_cmd(args=args, case_id=case_id)
            _run_logged_command(
                cmd=plot_cmd,
                log_path=plot_log,
                dry_run=False,
                proc_title=build_process_title("wdiff", "plot", case_id),
            )
            case_rows[case_id] = _load_rows(paths["combined_csv"])
            summary_rows.extend(_summarize_case(case_id=case_id, case_cfg=case_cfg, combined_csv=paths["combined_csv"]))

        case_statuses.append(
            {
                "case_id": case_id,
                "case_label": case_cfg["label"],
                "collector_log": collector_log,
                "plot_log": plot_log,
                "combined_csv": paths["combined_csv"],
                "manifest_json": paths["manifest_json"],
                "plots_dir": paths["plots_dir"],
                "reuse_baseline_from": case_cfg.get("reuse_baseline_from"),
                "reuse_wdro_from": case_cfg.get("reuse_wdro_from"),
            }
        )

    manifest = {
        "protocol": {
            "name": "three_method_single_trajectory_ablation_matrix",
            "description": (
                "Sequential overnight matrix around the single-trajectory three-method collector. "
                "Baseline is reused across cases when the base trajectory config is identical; WDRO is reused "
                "across CDRO-only ablations that keep the WDRO config fixed."
            ),
            "profile": args.profile,
            "case_ids": case_ids,
            "summary_groups": {name: [case_id for case_id in group if case_id in case_ids] for name, group in SUMMARY_GROUPS.items()},
            "primary_metric": "weighted_compute_units",
            "secondary_metric": "train_wall_clock_sec",
        },
        "base_config": {
            "seeds": args.seeds,
            "device": args.device,
            "dataset_path": args.dataset_path,
            "dataset_val_path": args.dataset_val_path,
            "fid_ref_path": args.fid_ref_path,
            "weighted_compute_calibration_path": args.weighted_compute_calibration_path,
            "shared_weighted_cap": args.shared_weighted_cap,
            "grid_template": args.grid_template,
            "n_steps_path": args.n_steps_path,
            "batch_size": args.batch_size,
            "hidden_dim": args.hidden_dim,
            "eval_samples": args.eval_samples,
            "fid_samples": args.fid_samples,
            "cdro_step_size": args.cdro_step_size,
            "cdro_time_horizon": args.cdro_time_horizon,
            "default_outer_clean_weight": args.outer_clean_weight,
        },
        "cases": {
            case_id: dict(CASE_LIBRARY[case_id])
            for case_id in case_ids
        },
        "artifacts": case_statuses,
    }
    manifest_path = os.path.join(args.outdir, f"{args.prefix}_matrix_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"[matrix] wrote {manifest_path}", flush=True)

    if args.dry_run:
        return

    summary_dir = os.path.join(args.outdir, "summary")
    ensure_dir(summary_dir)
    summary_csv = os.path.join(summary_dir, f"{args.prefix}_case_method_summary.csv")
    _write_csv(summary_csv, summary_rows)
    print(f"[matrix] wrote {summary_csv}", flush=True)

    family_plot_paths: List[str] = []
    for group_name, group_case_ids in SUMMARY_GROUPS.items():
        plot_path = _make_family_plot(
            group_name=group_name,
            case_ids=[case_id for case_id in group_case_ids if case_id in case_ids],
            case_rows=case_rows,
            case_cfgs=CASE_LIBRARY,
            outdir=summary_dir,
        )
        if plot_path:
            family_plot_paths.append(plot_path)
            print(f"[matrix] wrote {plot_path}", flush=True)

    summary_json = os.path.join(summary_dir, f"{args.prefix}_summary.json")
    with open(summary_json, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "summary_csv": summary_csv,
                "family_plots": family_plot_paths,
                "cases": case_statuses,
            },
            handle,
            indent=2,
        )
    print(f"[matrix] wrote {summary_json}", flush=True)


if __name__ == "__main__":
    main()
