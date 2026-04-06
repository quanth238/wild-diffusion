#!/usr/bin/env python3
import argparse
import csv
import json
import os
import subprocess
import sys
from typing import Dict, List, Optional, Tuple


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


from toy.process_title import apply_process_title, build_process_title, child_process_env  # noqa: E402


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-evaluate existing three-method trajectory checkpoints with a larger FID sample count "
            "without retraining, then regenerate per-case plots from the refreshed combined CSV."
        )
    )
    parser.add_argument("--combined-csv", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--prefix", type=str, required=True)
    parser.add_argument("--python-bin", type=str, default=os.path.join(ROOT_DIR, ".venv", "bin", "python"))
    parser.add_argument(
        "--plot-script",
        type=str,
        default=os.path.join(ROOT_DIR, "toy", "scripts", "plot_three_method_fid_curves.py"),
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dataset-path", type=str, default=DEFAULT_TRAIN_ROOT)
    parser.add_argument("--dataset-val-path", type=str, default=DEFAULT_VAL_ROOT)
    parser.add_argument("--fid-ref-path", type=str, default=DEFAULT_FID_REF)
    parser.add_argument("--weighted-compute-calibration-path", type=str, default=DEFAULT_CALIBRATION)
    parser.add_argument("--weighted-inputgrad-alpha", type=float, default=0.0)
    parser.add_argument("--weighted-parambackward-beta", type=float, default=0.0)
    parser.add_argument("--image-size", type=int, default=28)
    parser.add_argument("--image-channels", type=int, default=3)
    parser.add_argument("--image-train-size", type=int, default=80)
    parser.add_argument("--image-val-size", type=int, default=2000)
    parser.add_argument("--image-split-seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--eval-samples", type=int, default=2000)
    parser.add_argument("--fid-samples", type=int, default=50000)
    parser.add_argument("--debug-eval-batch", type=int, default=64)
    parser.add_argument("--debug-terminal-step", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--n-steps-path-default", type=int, default=24)
    parser.add_argument("--sigma-min", type=float, default=0.002)
    parser.add_argument("--sigma-max", type=float, default=2.0)
    parser.add_argument("--train-percent-label", type=str, default="1%")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str, rows: List[Dict[str, str]]) -> None:
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


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _safe_float(value, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _row_seed(row: Dict[str, str]) -> int:
    return _safe_int(row.get("seed"), 0)


def _row_step(row: Dict[str, str]) -> int:
    return _safe_int(row.get("step"), 0)


def _is_baseline_style_row(row: Dict[str, str]) -> bool:
    method = str(row.get("method", "")).strip()
    row_origin = str(row.get("row_origin", "")).strip()
    return method == "baseline_edm" or row_origin == "trajectory_warmup_phase"


def _load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _common_run_toy_prefix(
    *,
    args: argparse.Namespace,
    exp_name: str,
    eval_outdir: str,
    seed: int,
    n_steps_path: int,
    batch_size: int,
    hidden_dim: int,
    sigma_min: float,
    sigma_max: float,
) -> List[str]:
    cmd = [
        args.python_bin,
        os.path.join(ROOT_DIR, "toy", "run_toy.py"),
        "--exp-name",
        exp_name,
        "--outdir",
        eval_outdir,
        "--device",
        args.device,
        "--seed",
        str(seed),
        "--batch-size",
        str(batch_size),
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
        str(n_steps_path),
        "--hidden-dim",
        str(hidden_dim),
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
        str(sigma_min),
        "--sigma-max",
        str(sigma_max),
        "--compute-fid",
        "--fid-ref-path",
        args.fid_ref_path,
        "--skip-checks",
        "--disable-baseline-gate",
        "--weighted-compute-calibration-path",
        args.weighted_compute_calibration_path,
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
    return cmd


def _baseline_eval_command(
    *,
    args: argparse.Namespace,
    row: Dict[str, str],
    exp_name: str,
    eval_outdir: str,
) -> List[str]:
    step = _row_step(row)
    cmd = _common_run_toy_prefix(
        args=args,
        exp_name=exp_name,
        eval_outdir=eval_outdir,
        seed=_row_seed(row),
        n_steps_path=int(args.n_steps_path_default),
        batch_size=int(args.batch_size),
        hidden_dim=int(args.hidden_dim),
        sigma_min=float(args.sigma_min),
        sigma_max=float(args.sigma_max),
    )
    cmd.extend(
        [
            "--steps",
            str(step),
            "--method-version",
            "clean",
            "--baseline-only",
            "--baseline-ckpt-path",
            str(row["checkpoint_path"]),
            "--disable-baseline-ckpt-strict-meta",
        ]
    )
    return cmd


def _robust_eval_command(
    *,
    args: argparse.Namespace,
    row: Dict[str, str],
    exp_name: str,
    eval_outdir: str,
) -> List[str]:
    metrics_path = str(row.get("metrics_path", "")).strip()
    if not metrics_path:
        raise RuntimeError(f"Missing metrics_path for robust row: method={row.get('method')} step={row.get('step')}")
    payload = _load_json(metrics_path)
    cfg = payload.get("config", {})
    method_name = str(row["method"]).strip()
    cmd = _common_run_toy_prefix(
        args=args,
        exp_name=exp_name,
        eval_outdir=eval_outdir,
        seed=_row_seed(row),
        n_steps_path=_safe_int(cfg.get("n_steps_path"), int(args.n_steps_path_default)),
        batch_size=_safe_int(cfg.get("batch_size"), int(args.batch_size)),
        hidden_dim=_safe_int(cfg.get("hidden_dim"), int(args.hidden_dim)),
        sigma_min=float(_safe_float(cfg.get("sigma_min"), float(args.sigma_min))),
        sigma_max=float(_safe_float(cfg.get("sigma_max"), float(args.sigma_max))),
    )
    cmd.extend(
        [
            "--steps",
            str(_safe_int(cfg.get("steps"), _row_step(row))),
            "--method-version",
            method_name,
            "--disable-baseline-ckpt",
            "--robust-resume-ckpt-path",
            str(row["checkpoint_path"]),
        ]
    )
    baseline_steps_override = _safe_int(cfg.get("baseline_steps_override"), 0)
    if baseline_steps_override > 0:
        cmd.extend(["--baseline-steps-override", str(baseline_steps_override)])

    if method_name == "wdro":
        cmd.extend(
            [
                "--wdro-warmup-fraction",
                str(float(cfg.get("wdro_warmup_fraction", 0.0))),
                "--wdro-refresh-epochs",
                str(float(cfg.get("wdro_refresh_epochs", 100.0))),
                "--wdro-adv-prob",
                str(float(cfg.get("wdro_adv_prob", 0.3))),
                "--wdro-attack-steps",
                str(_safe_int(cfg.get("wdro_attack_steps"), 2)),
                "--wdro-attack-step-size",
                str(float(cfg.get("wdro_attack_step_size", 1e-3))),
                "--wdro-gamma",
                str(float(cfg.get("wdro_gamma", 1.0))),
            ]
        )
    elif method_name == "cdro":
        cmd.extend(
            [
                "--inner-steps",
                str(_safe_int(cfg.get("inner_steps"), 1)),
                "--outer-attack-weight",
                str(float(cfg.get("outer_attack_weight", 0.5))),
                "--outer-clean-weight",
                str(float(cfg.get("outer_clean_weight", 1.0))),
                "--cdro-step-size",
                str(float(cfg.get("cdro_step_size", 0.02))),
                "--cdro-total-budget-rho",
                str(float(cfg.get("cdro_total_budget_rho", 0.02))),
                "--cdro-time-horizon",
                str(float(cfg.get("cdro_time_horizon", 1.0))),
                "--cdro-warmup-fraction",
                str(float(cfg.get("cdro_warmup_fraction", 0.0))),
                "--disable-collapse-diagnostics",
            ]
        )
    else:
        raise RuntimeError(f"Unsupported robust method for reevaluation: {method_name}")
    return cmd


def _reeval_cache_key(row: Dict[str, str]) -> Tuple[str, str]:
    checkpoint_path = str(row.get("checkpoint_path", "")).strip()
    if not checkpoint_path:
        raise RuntimeError(f"Missing checkpoint_path in row: method={row.get('method')} step={row.get('step')}")
    if _is_baseline_style_row(row):
        return ("baseline", checkpoint_path)
    return (str(row.get("method", "")).strip(), checkpoint_path)


def _exp_name_from_row(prefix: str, row: Dict[str, str], occurrence_index: int) -> str:
    method = str(row.get("method", "")).strip()
    row_origin = str(row.get("row_origin", "")).strip() or "row"
    seed = _row_seed(row)
    step = _row_step(row)
    return f"{prefix}_{method}_{row_origin}_st{step}_s{seed}_k{occurrence_index:03d}"


def _extract_fid(metrics_path: str, row: Dict[str, str]) -> float:
    payload = _load_json(metrics_path)
    sample_quality = payload.get("metrics", {}).get("sample_quality_debug", {})
    key = "baseline_fid" if _is_baseline_style_row(row) else "robust_fid"
    fid_value = _safe_float(sample_quality.get(key))
    if fid_value is None:
        raise RuntimeError(f"Missing {key} in reevaluated metrics: {metrics_path}")
    return float(fid_value)


def _run_command(*, cmd: List[str], log_path: str, proc_title: str, skip_existing: bool, expected_metrics_path: str) -> None:
    ensure_dir(os.path.dirname(log_path))
    if skip_existing and os.path.isfile(expected_metrics_path):
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


def main() -> None:
    args = parse_args()
    if _APPLIED_PROCESS_TITLE is None:
        apply_process_title(build_process_title("wdiff", "reeval", args.prefix))
    ensure_dir(args.outdir)
    eval_runs_dir = os.path.join(args.outdir, "reeval_runs")
    logs_dir = os.path.join(args.outdir, "logs")
    plots_dir = os.path.join(args.outdir, "plots")
    ensure_dir(eval_runs_dir)
    ensure_dir(logs_dir)
    ensure_dir(plots_dir)

    input_rows = load_rows(args.combined_csv)
    reevaluated_rows: List[Dict[str, str]] = []
    cache: Dict[Tuple[str, str], Dict[str, str]] = {}

    for row_index, row in enumerate(input_rows):
        cache_key = _reeval_cache_key(row)
        cached = cache.get(cache_key)
        if cached is None:
            exp_name = _exp_name_from_row(args.prefix, row, row_index)
            if _is_baseline_style_row(row):
                cmd = _baseline_eval_command(args=args, row=row, exp_name=exp_name, eval_outdir=eval_runs_dir)
            else:
                cmd = _robust_eval_command(args=args, row=row, exp_name=exp_name, eval_outdir=eval_runs_dir)
            metrics_path = os.path.join(eval_runs_dir, exp_name, "metrics.json")
            log_path = os.path.join(logs_dir, f"{exp_name}.log")
            proc_title = build_process_title("wdiff", "reeval", row.get("method", ""), f"st{_row_step(row)}")
            _run_command(
                cmd=cmd,
                log_path=log_path,
                proc_title=proc_title or "wdiff:reeval",
                skip_existing=bool(args.skip_existing),
                expected_metrics_path=metrics_path,
            )
            cached = {
                "fid": str(_extract_fid(metrics_path, row)),
                "reeval_metrics_path": metrics_path,
                "reeval_log_path": log_path,
                "reeval_exp_name": exp_name,
            }
            cache[cache_key] = cached

        updated = dict(row)
        updated["fid_original"] = str(row.get("fid", ""))
        updated["fid"] = str(cached["fid"])
        updated["reeval_metrics_path"] = str(cached["reeval_metrics_path"])
        updated["reeval_log_path"] = str(cached["reeval_log_path"])
        updated["reeval_exp_name"] = str(cached["reeval_exp_name"])
        updated["reeval_fid_samples"] = str(int(args.fid_samples))
        reevaluated_rows.append(updated)

    combined_out = os.path.join(args.outdir, f"{args.prefix}_all_methods_raw_seed_rows.csv")
    write_csv(combined_out, reevaluated_rows)

    plot_cmd = [
        args.python_bin,
        args.plot_script,
        "--combined-csv",
        combined_out,
        "--outdir",
        plots_dir,
        "--prefix",
        args.prefix,
        "--train-percent-label",
        args.train_percent_label,
    ]
    plot_log = os.path.join(logs_dir, f"{args.prefix}_plot.log")
    _run_command(
        cmd=plot_cmd,
        log_path=plot_log,
        proc_title=build_process_title("wdiff", "plot", args.prefix) or "wdiff:plot",
        skip_existing=False,
        expected_metrics_path=os.path.join(plots_dir, f"{args.prefix}_three_method_compare_summary.json"),
    )

    manifest = {
        "combined_csv_in": args.combined_csv,
        "combined_csv_out": combined_out,
        "fid_samples": int(args.fid_samples),
        "plots_dir": plots_dir,
        "plot_log": plot_log,
        "unique_reevaluations": len(cache),
    }
    with open(os.path.join(args.outdir, f"{args.prefix}_reeval_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"[reeval] wrote {combined_out}", flush=True)
    print(f"[reeval] unique_reevaluations={len(cache)} fid_samples={int(args.fid_samples)}", flush=True)


if __name__ == "__main__":
    main()
