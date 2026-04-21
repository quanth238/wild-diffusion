#!/usr/bin/env python3
"""Queue an RF-family CDRO sweep over rho and N with baseline/WDRO reuse.

This helper is intentionally narrow:
- baseline RF raw rows are reused from one existing shared-checkpoint run
- WDRO-RF raw rows are reused from one existing shared-checkpoint run
- each `(rho, N)` case reruns only the CDRO-RF lane
- each case gets the standard three-method compare plots
- the sweep root also gets a compact summary CSV plus CDRO-focused heatmaps
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


from toy.compute_accounting import resolve_default_weighted_compute_calibration_path
from toy.config import ToyConfig
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

METHOD_LABELS = {
    "baseline": "Baseline RF",
    "wild_diffusion": "WDRO-RF",
    "cdro": "CDRO-RF",
}


def _parse_float_list(text: str) -> List[float]:
    values: List[float] = []
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        value = float(token)
        if value < 0.0:
            raise ValueError(f"Expected non-negative rho value, got {value}")
        values.append(float(value))
    if not values:
        raise ValueError("Expected at least one rho value.")
    return sorted(set(values))


def _parse_int_list(text: str) -> List[int]:
    values: List[int] = []
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise ValueError(f"Expected positive integer N value, got {value}")
        values.append(int(value))
    if not values:
        raise ValueError("Expected at least one N value.")
    return sorted(set(values))


def _safe_float(value) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return float(parsed)


def _format_token(value: float, digits: int = 6) -> str:
    text = f"{float(value):.{digits}f}".rstrip("0").rstrip(".")
    if not text:
        text = "0"
    return text.replace("-", "m").replace(".", "p")


def _case_id(*, rho: float, n_steps: int) -> str:
    return f"rho{_format_token(rho)}_n{int(n_steps)}"


def _case_label(*, rho: float, n_steps: int) -> str:
    return f"rho={float(rho):.6g}, N={int(n_steps)}"


def _resolve_calibration_path(args: argparse.Namespace) -> str:
    return resolve_default_weighted_compute_calibration_path(
        calibration_path=str(args.weighted_compute_calibration_path).strip(),
        training_objective="rf",
        image_backbone=str(args.image_backbone),
        batch_size=int(args.batch_size),
        hidden_dim=int(args.hidden_dim),
        device=str(args.device),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Queue an RF-family CDRO sweep over `(rho, cdro_n_steps_path)` while reusing one baseline RF raw CSV "
            "and one WDRO-RF raw CSV. Each case reruns only CDRO-RF on the shared checkpoint branch."
        )
    )
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--prefix", type=str, required=True)
    parser.add_argument("--rho-values", type=str, required=True)
    parser.add_argument("--cdro-n-values", type=str, required=True)
    parser.add_argument("--python-bin", type=str, default=os.path.join(ROOT_DIR, ".venv", "bin", "python"))
    parser.add_argument(
        "--collector-script",
        type=str,
        default=os.path.join(ROOT_DIR, "toy", "scripts", "collect_three_method_seed_data.py"),
    )
    parser.add_argument(
        "--plot-script",
        type=str,
        default=os.path.join(ROOT_DIR, "toy", "scripts", "plot_three_method_fid_curves.py"),
    )
    parser.add_argument("--reuse-baseline-raw-csv", type=str, required=True)
    parser.add_argument("--reuse-wdro-raw-csv", type=str, required=True)
    parser.add_argument("--rf-edm-init-ckpt-path", type=str, required=True)
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp-dtype", type=str, default="auto")
    parser.add_argument("--dataset-path", type=str, default=DEFAULT_TRAIN_ROOT)
    parser.add_argument("--dataset-val-path", type=str, default=DEFAULT_VAL_ROOT)
    parser.add_argument("--fid-ref-path", type=str, default=DEFAULT_FID_REF)
    parser.add_argument("--weighted-compute-calibration-path", type=str, default="")
    parser.add_argument("--weighted-inputgrad-alpha", type=float, default=0.0)
    parser.add_argument("--weighted-parambackward-beta", type=float, default=0.0)
    parser.add_argument("--train-accelerator-count", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=28)
    parser.add_argument("--image-channels", type=int, default=3)
    parser.add_argument("--image-train-size", type=int, default=400)
    parser.add_argument("--image-val-size", type=int, default=2000)
    parser.add_argument("--image-split-seed", type=int, default=0)
    parser.add_argument("--train-percent-label", type=str, default="100%")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--image-backbone", type=str, default="conv")
    parser.add_argument("--eval-samples", type=int, default=2000)
    parser.add_argument("--fid-samples", type=int, default=2000)
    parser.add_argument("--fid-gen-batch", type=int, default=64)
    parser.add_argument("--debug-eval-batch", type=int, default=64)
    parser.add_argument("--debug-terminal-step", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--n-steps-path", type=int, default=32)
    parser.add_argument("--sigma-min", type=float, default=0.002)
    parser.add_argument("--sigma-max", type=float, default=2.0)
    parser.add_argument("--shared-weighted-cap", type=float, default=400000.0)
    parser.add_argument("--grid-template", type=str, default="denser")
    parser.add_argument("--fid-eval-template", type=str, default="balanced")
    parser.add_argument("--transition-sentinel-robust-count", type=int, default=3)
    parser.add_argument("--baseline-fid-mode", type=str, default="posthoc_from_checkpoints")
    parser.add_argument("--robust-fid-mode", type=str, default="posthoc_from_checkpoints")
    parser.add_argument("--posthoc-fid-batch-size", type=int, default=512)
    parser.add_argument("--baseline-max-steps", type=int, default=160000)
    parser.add_argument("--wdro-max-total-steps", type=int, default=160000)
    parser.add_argument("--robust-warmup-mode", type=str, default="shared_exact")
    parser.add_argument("--wdro-refresh-epochs", type=float, default=100.0)
    parser.add_argument("--wdro-adv-prob", type=float, default=0.3)
    parser.add_argument("--wdro-attack-steps", type=int, default=2)
    parser.add_argument("--wdro-attack-step-size", type=float, default=1e-3)
    parser.add_argument("--wdro-gamma", type=float, default=1.0)
    parser.add_argument("--inner-steps", type=int, default=1)
    parser.add_argument("--outer-attack-weight", type=float, default=1.0)
    parser.add_argument("--outer-clean-weight", type=float, default=0.0)
    parser.add_argument("--cdro-step-size", type=float, default=0.02)
    parser.add_argument("--cdro-time-horizon", type=float, default=1.0)
    parser.add_argument("--cdro-edm-ladder-mode", type=str, default="stochastic_stratified_quantile")
    parser.add_argument("--rf-baseline-mode", type=str, default="strong", choices=["strong"])
    parser.add_argument("--rf-reflow-t-distribution", type=str, default=ToyConfig.rf_reflow_t_distribution)
    parser.add_argument("--rf-loss", type=str, default=ToyConfig.rf_loss)
    parser.add_argument("--rf-pseudo-huber-delta", type=float, default=ToyConfig.rf_pseudo_huber_delta)
    parser.add_argument("--rf-cdro-pair-source", type=str, default="reflow", choices=["auto", "reflow"])
    parser.add_argument(
        "--rf-edm-teacher-sampler",
        type=str,
        default=ToyConfig.rf_edm_teacher_sampler,
        choices=["ancestral_stochastic", "ancestral_mean_only", "edm_euler", "edm_heun"],
    )
    parser.add_argument("--rf-teacher-n-steps-path", type=int, default=ToyConfig.rf_teacher_n_steps_path)
    parser.add_argument("--rf-eval-n-steps-path", type=int, default=ToyConfig.rf_eval_n_steps_path)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _case_paths(*, base_outdir: str, base_prefix: str, case_id: str) -> Dict[str, str]:
    case_outdir = os.path.join(base_outdir, case_id)
    case_prefix = f"{base_prefix}_{case_id}"
    return {
        "case_outdir": case_outdir,
        "case_prefix": case_prefix,
        "combined_csv": os.path.join(case_outdir, f"{case_prefix}_all_methods_raw_seed_rows.csv"),
        "manifest_json": os.path.join(case_outdir, f"{case_prefix}_manifest.json"),
        "plots_dir": os.path.join(case_outdir, "plots"),
        "plots_manifest": os.path.join(case_outdir, "plots", f"{case_prefix}_three_method_compare_summary.json"),
    }


def _build_collector_cmd(
    *,
    args: argparse.Namespace,
    case_id: str,
    rho: float,
    n_steps: int,
) -> List[str]:
    paths = _case_paths(base_outdir=args.outdir, base_prefix=args.prefix, case_id=case_id)
    cmd = [
        args.python_bin,
        args.collector_script,
        "--outdir",
        paths["case_outdir"],
        "--prefix",
        paths["case_prefix"],
        "--seeds",
        str(args.seeds),
        "--device",
        str(args.device),
        "--amp-dtype",
        str(args.amp_dtype),
        "--dataset-path",
        str(args.dataset_path),
        "--dataset-val-path",
        str(args.dataset_val_path),
        "--fid-ref-path",
        str(args.fid_ref_path),
        "--weighted-compute-calibration-path",
        str(args.weighted_compute_calibration_path),
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
        str(args.train_percent_label),
        "--batch-size",
        str(args.batch_size),
        "--hidden-dim",
        str(args.hidden_dim),
        "--image-backbone",
        str(args.image_backbone),
        "--training-objective",
        "rf",
        "--eval-samples",
        str(args.eval_samples),
        "--fid-samples",
        str(args.fid_samples),
        "--fid-gen-batch",
        str(args.fid_gen_batch),
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
        "--fid-eval-template",
        str(args.fid_eval_template),
        "--transition-sentinel-robust-count",
        str(args.transition_sentinel_robust_count),
        "--baseline-fid-mode",
        str(args.baseline_fid_mode),
        "--robust-fid-mode",
        str(args.robust_fid_mode),
        "--posthoc-fid-batch-size",
        str(args.posthoc_fid_batch_size),
        "--baseline-max-steps",
        str(args.baseline_max_steps),
        "--wdro-max-total-steps",
        str(args.wdro_max_total_steps),
        "--robust-warmup-mode",
        str(args.robust_warmup_mode),
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
        str(args.outer_attack_weight),
        "--outer-clean-weight",
        str(args.outer_clean_weight),
        "--cdro-step-size",
        str(args.cdro_step_size),
        "--cdro-total-budget-rho",
        str(rho),
        "--cdro-time-horizon",
        str(args.cdro_time_horizon),
        "--cdro-edm-ladder-mode",
        str(args.cdro_edm_ladder_mode),
        "--cdro-n-steps-path",
        str(int(n_steps)),
        "--rf-baseline-mode",
        str(args.rf_baseline_mode),
        "--rf-reflow-t-distribution",
        str(args.rf_reflow_t_distribution),
        "--rf-loss",
        str(args.rf_loss),
        "--rf-pseudo-huber-delta",
        str(args.rf_pseudo_huber_delta),
        "--rf-cdro-pair-source",
        str(args.rf_cdro_pair_source),
        "--rf-edm-teacher-sampler",
        str(args.rf_edm_teacher_sampler),
        "--rf-teacher-n-steps-path",
        str(args.rf_teacher_n_steps_path),
        "--rf-eval-n-steps-path",
        str(args.rf_eval_n_steps_path),
        "--rf-edm-init-ckpt-path",
        str(args.rf_edm_init_ckpt_path),
        "--reuse-baseline-raw-csv",
        str(args.reuse_baseline_raw_csv),
        "--reuse-wdro-raw-csv",
        str(args.reuse_wdro_raw_csv),
    ]
    if args.skip_existing:
        cmd.append("--skip-existing")
    return cmd


def _build_plot_cmd(*, args: argparse.Namespace, case_id: str) -> List[str]:
    paths = _case_paths(base_outdir=args.outdir, base_prefix=args.prefix, case_id=case_id)
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
        str(args.train_percent_label),
    ]


def _run_logged_command(*, cmd: List[str], log_path: str, dry_run: bool, proc_title: Optional[str]) -> None:
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


def _load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: str, rows: Sequence[Dict]) -> None:
    ensure_dir(os.path.dirname(path))
    if not rows:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("")
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


def _method_rows(rows: Sequence[Dict[str, str]], method_name: str) -> List[Dict[str, str]]:
    out = [row for row in rows if str(row.get("method", "")) == method_name]
    out.sort(
        key=lambda row: (
            _safe_float(row.get("weighted_compute_units")) if _safe_float(row.get("weighted_compute_units")) is not None else float("inf"),
            int(row.get("step", 0) or 0),
        )
    )
    return out


def _best_robust_row(rows: Sequence[Dict[str, str]], method_name: str) -> Optional[Dict[str, str]]:
    robust_rows = [
        row
        for row in _method_rows(rows, method_name)
        if str(row.get("row_origin", "")) == "trajectory_robust_phase" and _safe_float(row.get("fid")) is not None
    ]
    if not robust_rows:
        return None
    return min(robust_rows, key=lambda row: float(row["fid"]))


def _first_robust_row(rows: Sequence[Dict[str, str]], method_name: str) -> Optional[Dict[str, str]]:
    robust_rows = [
        row
        for row in _method_rows(rows, method_name)
        if str(row.get("row_origin", "")) == "trajectory_robust_phase"
        and _safe_float(row.get("weighted_compute_units")) is not None
    ]
    if not robust_rows:
        return None
    return min(robust_rows, key=lambda row: float(row["weighted_compute_units"]))


def _final_row(rows: Sequence[Dict[str, str]], method_name: str) -> Optional[Dict[str, str]]:
    finite_rows = [
        row for row in _method_rows(rows, method_name) if _safe_float(row.get("weighted_compute_units")) is not None
    ]
    if not finite_rows:
        return None
    return max(finite_rows, key=lambda row: float(row["weighted_compute_units"]))


def _row_value(row: Optional[Dict[str, str]], field: str) -> Optional[float]:
    if row is None:
        return None
    return _safe_float(row.get(field))


def _summarize_case(
    *,
    case_id: str,
    rho: float,
    n_steps: int,
    manifest_path: str,
    combined_csv: str,
) -> Dict:
    manifest = _load_json(manifest_path)
    rows = _load_rows(combined_csv)
    shared_cap = _safe_float(manifest.get("weighted_compute", {}).get("shared_cap"))
    baseline_final = _final_row(rows, "baseline")
    wdro_final = _final_row(rows, "wild_diffusion")
    cdro_final = _final_row(rows, "cdro")
    cdro_best_robust = _best_robust_row(rows, "cdro")
    cdro_first_robust = _first_robust_row(rows, "cdro")

    cdro_final_wcu = _row_value(cdro_final, "weighted_compute_units")
    shared_gap = None if shared_cap is None or cdro_final_wcu is None else float(cdro_final_wcu - shared_cap)
    shared_ratio = None if shared_cap in (None, 0.0) or cdro_final_wcu is None else float(cdro_final_wcu / shared_cap)

    return {
        "case_id": str(case_id),
        "case_label": _case_label(rho=rho, n_steps=n_steps),
        "cdro_total_budget_rho": float(rho),
        "cdro_n_steps_path": int(n_steps),
        "shared_weighted_cap": shared_cap,
        "shared_fixed_warmup_steps": manifest.get("protocol", {}).get("shared_robust_fixed_warmup_steps"),
        "cdro_expected_robust_step_weighted_units": _safe_float(
            manifest.get("weighted_compute", {}).get("cdro_expected_robust_step_weighted_units")
        ),
        "cdro_max_total_steps": manifest.get("trajectories", {}).get("cdro", {}).get("max_total_steps"),
        "baseline_final_fid": _row_value(baseline_final, "fid"),
        "baseline_final_weighted_compute": _row_value(baseline_final, "weighted_compute_units"),
        "wdro_final_fid": _row_value(wdro_final, "fid"),
        "wdro_final_weighted_compute": _row_value(wdro_final, "weighted_compute_units"),
        "cdro_first_robust_fid": _row_value(cdro_first_robust, "fid"),
        "cdro_first_robust_weighted_compute": _row_value(cdro_first_robust, "weighted_compute_units"),
        "cdro_best_robust_fid": _row_value(cdro_best_robust, "fid"),
        "cdro_best_robust_weighted_compute": _row_value(cdro_best_robust, "weighted_compute_units"),
        "cdro_final_fid": _row_value(cdro_final, "fid"),
        "cdro_final_weighted_compute": cdro_final_wcu,
        "cdro_final_wcu_gap_vs_shared_cap": shared_gap,
        "cdro_final_wcu_ratio_vs_shared_cap": shared_ratio,
        "combined_csv": str(combined_csv),
        "manifest_json": str(manifest_path),
    }


def _format_rho_label(rho: float) -> str:
    return f"{float(rho):.6g}"


def _heatmap_matrix(
    *,
    summary_rows: Sequence[Dict],
    rho_values: Sequence[float],
    n_values: Sequence[int],
    field: str,
) -> np.ndarray:
    matrix = np.full((len(n_values), len(rho_values)), np.nan, dtype=float)
    for row in summary_rows:
        rho = float(row["cdro_total_budget_rho"])
        n_steps = int(row["cdro_n_steps_path"])
        value = _safe_float(row.get(field))
        if value is None:
            continue
        if rho not in rho_values or n_steps not in n_values:
            continue
        matrix[n_values.index(n_steps), rho_values.index(rho)] = float(value)
    return matrix


def _plot_heatmap(
    *,
    summary_rows: Sequence[Dict],
    rho_values: Sequence[float],
    n_values: Sequence[int],
    field: str,
    title: str,
    colorbar_label: str,
    outpath: str,
    cmap: str,
    annotation_scale: float = 1.0,
    annotation_fmt: str = "{:.2f}",
) -> Optional[str]:
    matrix = _heatmap_matrix(summary_rows=summary_rows, rho_values=rho_values, n_values=n_values, field=field)
    if not np.isfinite(matrix).any():
        return None
    ensure_dir(os.path.dirname(outpath))
    masked = np.ma.masked_invalid(matrix)
    fig, ax = plt.subplots(figsize=(1.6 * len(rho_values) + 2.4, 1.0 * len(n_values) + 2.6))
    im = ax.imshow(masked, aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(rho_values)))
    ax.set_xticklabels([_format_rho_label(rho) for rho in rho_values], rotation=30, ha="right")
    ax.set_yticks(range(len(n_values)))
    ax.set_yticklabels([str(int(n_steps)) for n_steps in n_values])
    ax.set_xlabel("rho")
    ax.set_ylabel("CDRO N")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label)
    for y_index, n_steps in enumerate(n_values):
        for x_index, rho in enumerate(rho_values):
            value = matrix[y_index, x_index]
            if not math.isfinite(float(value)):
                ax.text(x_index, y_index, "NA", ha="center", va="center", fontsize=8, color="gray")
                continue
            display_value = float(value) / float(annotation_scale)
            ax.text(
                x_index,
                y_index,
                annotation_fmt.format(display_value),
                ha="center",
                va="center",
                fontsize=8,
                color="white" if float(value) > float(np.nanmedian(matrix)) else "black",
            )
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)
    return outpath


def main() -> None:
    args = parse_args()
    if _APPLIED_PROCESS_TITLE is None:
        apply_process_title(build_process_title("wdiff", "rf-sweep", args.prefix))
    args.weighted_compute_calibration_path = _resolve_calibration_path(args)
    rho_values = _parse_float_list(args.rho_values)
    n_values = _parse_int_list(args.cdro_n_values)

    ensure_dir(args.outdir)
    logs_dir = os.path.join(args.outdir, "logs")
    ensure_dir(logs_dir)

    print(
        "[rf-rho-n-sweep] weighted_compute_calibration_path="
        f"{args.weighted_compute_calibration_path or 'unresolved'}",
        flush=True,
    )
    print(
        "[rf-rho-n-sweep] rho_values="
        f"{','.join(_format_rho_label(value) for value in rho_values)} "
        f"cdro_n_values={','.join(str(value) for value in n_values)}",
        flush=True,
    )

    case_statuses: List[Dict] = []
    summary_rows: List[Dict] = []
    for rho in rho_values:
        for n_steps in n_values:
            case_id = _case_id(rho=rho, n_steps=n_steps)
            paths = _case_paths(base_outdir=args.outdir, base_prefix=args.prefix, case_id=case_id)
            collector_log = os.path.join(logs_dir, f"{case_id}_collector.log")
            plot_log = os.path.join(logs_dir, f"{case_id}_plot.log")
            print(f"[rf-rho-n-sweep] case={case_id} label={_case_label(rho=rho, n_steps=n_steps)}", flush=True)

            collector_cmd = _build_collector_cmd(args=args, case_id=case_id, rho=rho, n_steps=n_steps)
            _run_logged_command(
                cmd=collector_cmd,
                log_path=collector_log,
                dry_run=args.dry_run,
                proc_title=build_process_title("wdiff", "collect", case_id),
            )

            if not args.dry_run:
                if not os.path.isfile(paths["combined_csv"]):
                    raise RuntimeError(f"Missing combined CSV after collector run: {paths['combined_csv']}")
                plot_cmd = _build_plot_cmd(args=args, case_id=case_id)
                _run_logged_command(
                    cmd=plot_cmd,
                    log_path=plot_log,
                    dry_run=False,
                    proc_title=build_process_title("wdiff", "plot", case_id),
                )
                summary_rows.append(
                    _summarize_case(
                        case_id=case_id,
                        rho=rho,
                        n_steps=n_steps,
                        manifest_path=paths["manifest_json"],
                        combined_csv=paths["combined_csv"],
                    )
                )

            case_statuses.append(
                {
                    "case_id": case_id,
                    "case_label": _case_label(rho=rho, n_steps=n_steps),
                    "rho": float(rho),
                    "cdro_n_steps_path": int(n_steps),
                    "collector_log": collector_log,
                    "plot_log": plot_log,
                    "combined_csv": paths["combined_csv"],
                    "manifest_json": paths["manifest_json"],
                    "plots_dir": paths["plots_dir"],
                }
            )

    summary_csv = os.path.join(args.outdir, f"{args.prefix}_rf_rho_n_sweep_summary.csv")
    manifest_json = os.path.join(args.outdir, f"{args.prefix}_rf_rho_n_sweep_manifest.json")
    summary_plots_dir = os.path.join(args.outdir, "summary_plots")

    heatmap_paths: Dict[str, Optional[str]] = {}
    if not args.dry_run:
        summary_rows.sort(key=lambda row: (float(row["cdro_total_budget_rho"]), int(row["cdro_n_steps_path"])))
        _write_csv(summary_csv, summary_rows)
        heatmap_paths["cdro_best_robust_fid"] = _plot_heatmap(
            summary_rows=summary_rows,
            rho_values=rho_values,
            n_values=n_values,
            field="cdro_best_robust_fid",
            title="CDRO-RF Best Robust FID by rho and N",
            colorbar_label="Best robust FID",
            outpath=os.path.join(summary_plots_dir, f"{args.prefix}_cdro_best_robust_fid_heatmap.png"),
            cmap="viridis_r",
        )
        heatmap_paths["cdro_final_fid"] = _plot_heatmap(
            summary_rows=summary_rows,
            rho_values=rho_values,
            n_values=n_values,
            field="cdro_final_fid",
            title="CDRO-RF Final FID by rho and N",
            colorbar_label="Final FID",
            outpath=os.path.join(summary_plots_dir, f"{args.prefix}_cdro_final_fid_heatmap.png"),
            cmap="viridis_r",
        )
        heatmap_paths["cdro_final_weighted_compute_k"] = _plot_heatmap(
            summary_rows=summary_rows,
            rho_values=rho_values,
            n_values=n_values,
            field="cdro_final_weighted_compute",
            title="CDRO-RF Final Weighted Compute by rho and N",
            colorbar_label="Final weighted compute",
            outpath=os.path.join(summary_plots_dir, f"{args.prefix}_cdro_final_weighted_compute_heatmap.png"),
            cmap="magma",
            annotation_scale=1000.0,
            annotation_fmt="{:.1f}k",
        )

    manifest = {
        "protocol": {
            "name": "rf_cdro_rho_n_grid",
            "description": (
                "Sequential RF-family CDRO sweep over `(rho, cdro_n_steps_path)` using one reused baseline RF raw CSV "
                "and one reused WDRO-RF raw CSV. Each case runs a fresh CDRO-RF lane and then renders the standard "
                "three-method comparison plots."
            ),
            "rho_values": [float(value) for value in rho_values],
            "cdro_n_values": [int(value) for value in n_values],
            "primary_metric": "weighted_compute_units",
            "secondary_metric": "train_wall_clock_sec",
        },
        "base_config": {
            "reuse_baseline_raw_csv": str(args.reuse_baseline_raw_csv),
            "reuse_wdro_raw_csv": str(args.reuse_wdro_raw_csv),
            "rf_edm_init_ckpt_path": str(args.rf_edm_init_ckpt_path),
            "weighted_compute_calibration_path": str(args.weighted_compute_calibration_path),
            "shared_weighted_cap": float(args.shared_weighted_cap),
            "rf_teacher_n_steps_path": int(args.rf_teacher_n_steps_path),
            "rf_eval_n_steps_path": int(args.rf_eval_n_steps_path),
            "rf_edm_teacher_sampler": str(args.rf_edm_teacher_sampler),
            "seeds": str(args.seeds),
            "device": str(args.device),
            "dataset_path": str(args.dataset_path),
            "dataset_val_path": str(args.dataset_val_path),
            "fid_ref_path": str(args.fid_ref_path),
            "train_percent_label": str(args.train_percent_label),
        },
        "cases": case_statuses,
        "summary_csv": None if args.dry_run else summary_csv,
        "summary_plots_dir": None if args.dry_run else summary_plots_dir,
        "summary_heatmaps": heatmap_paths,
    }
    with open(manifest_json, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


if __name__ == "__main__":
    main()
