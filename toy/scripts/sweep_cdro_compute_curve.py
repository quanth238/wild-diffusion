#!/usr/bin/env python3
import argparse
import csv
import json
import os
import subprocess
from typing import Dict, List, Optional

import matplotlib.pyplot as plt


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
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
    parser.add_argument("--baseline-checkpoint-dir", type=str, default=DEFAULT_BASELINE_CHECKPOINT_DIR)
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
    parser.add_argument("--steps-list", type=str, default=DEFAULT_STEPS_LIST)
    parser.add_argument("--skip-existing", action="store_true")
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


def parse_steps_list(value: str) -> List[int]:
    steps = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    return [step for step in steps if step >= 0]


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


def run_point(
    *,
    args: argparse.Namespace,
    target_total_steps: int,
    warmup_steps: int,
    exp_name: str,
    log_path: str,
    baseline_ckpt_path: Optional[str],
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
        "--baseline-steps-override",
        str(warmup_steps),
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
        "--cdro-warmup-fraction",
        str(args.cdro_warmup_fraction),
    ]
    if baseline_ckpt_path:
        cmd.extend(
            [
                "--baseline-ckpt-path",
                baseline_ckpt_path,
                "--disable-baseline-ckpt-strict-meta",
            ]
        )
    env = maybe_set_fid_detector_env(os.environ)
    with open(log_path, "w", encoding="utf-8") as handle:
        subprocess.run(cmd, cwd=ROOT_DIR, env=env, check=True, stdout=handle, stderr=subprocess.STDOUT)
    return os.path.join(args.outdir, exp_name, "metrics.json")


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_baseline_rows(paths: List[str]) -> List[Dict]:
    by_step: Dict[int, Dict] = {}
    for path in paths:
        if not path or not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                step = int(float(row["step"]))
                by_step[step] = {
                    "method": "baseline_edm",
                    "step": step,
                    "compute_budget_be": float(step),
                    "images_shown_m": float(row["images_shown_m"]),
                    "fid": float(row["fid_median"]),
                    "train_elapsed_sec": float(row["train_elapsed_median_sec"]),
                    "source": path,
                }
    return [by_step[step] for step in sorted(by_step.keys())]


def extract_cdro_row(metrics_path: str, *, baseline_ckpt_requested: Optional[str]) -> Dict:
    payload = load_json(metrics_path)
    metrics = payload["metrics"]
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
        "images_shown_m": float(total_images_shown_m),
        "fid": float(fid_value),
        "baseline_fid_same_run": (
            float(sample_quality["baseline_fid"]) if sample_quality.get("baseline_fid") is not None else float("nan")
        ),
        "train_elapsed_sec": float(runtime["effective_train_total"]),
        "runtime_total_sec": float(runtime["total"]),
        "warmup_steps_fixed": int(baseline_phase_steps),
        "robust_steps_target": int(robust_phase_steps),
        "warmup_only": bool(warmup_only),
        "baseline_ckpt_requested": baseline_ckpt_requested,
        "baseline_ckpt_loaded": bool(flow["baseline_ckpt_loaded"]),
        "phase_step_split_mode": str(flow["phase_step_split_mode"]),
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


def make_plot(
    path: str,
    baseline_rows: List[Dict],
    cdro_rows: List[Dict],
    train_percent_label: str,
) -> None:
    plt.figure(figsize=(8, 5))
    if baseline_rows:
        plt.plot(
            [row["compute_budget_be"] for row in baseline_rows],
            [row["fid"] for row in baseline_rows],
            marker="o",
            linewidth=2.0,
            label="Baseline EDM",
        )
    if cdro_rows:
        plt.plot(
            [row["compute_budget_be"] for row in cdro_rows],
            [row["fid"] for row in cdro_rows],
            marker="o",
            linewidth=2.0,
            label="CDRO",
        )
    plt.xlabel("Compute Budget (batch-equivalent denoiser evals)")
    plt.ylabel("FID")
    plt.title(f"Simpsons-MNIST RGB {train_percent_label}: FID vs Compute Budget")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    ensure_dir(args.outdir)
    logs_dir = os.path.join(args.outdir, "logs")
    ensure_dir(logs_dir)

    steps_list = parse_steps_list(args.steps_list)
    print(f"[cdro-curve] checkpoints={steps_list}", flush=True)

    cdro_rows: List[Dict] = []
    for target_total_steps in steps_list:
        warmup_steps = int(target_total_steps * float(args.cdro_warmup_fraction))
        baseline_ckpt_path = find_baseline_checkpoint(args.baseline_checkpoint_dir, warmup_steps)
        exp_name = f"{args.prefix}_st{target_total_steps}_s{args.seed}"
        metrics_path = os.path.join(args.outdir, exp_name, "metrics.json")
        log_path = os.path.join(logs_dir, f"{exp_name}.log")
        if args.skip_existing and os.path.isfile(metrics_path):
            print(f"[cdro-curve] reuse existing metrics for step={target_total_steps}: {metrics_path}", flush=True)
        else:
            print(
                f"[cdro-curve] run step={target_total_steps} warmup_steps={warmup_steps} "
                f"baseline_ckpt={baseline_ckpt_path or 'none'} log={log_path}",
                flush=True,
            )
            run_point(
                args=args,
                target_total_steps=target_total_steps,
                warmup_steps=warmup_steps,
                exp_name=exp_name,
                log_path=log_path,
                baseline_ckpt_path=baseline_ckpt_path,
            )
        row = extract_cdro_row(metrics_path, baseline_ckpt_requested=baseline_ckpt_path)
        cdro_rows.append(row)
        print(
            f"[cdro-curve] step={row['step']} compute_be={row['compute_budget_be']:.1f} "
            f"fid={row['fid']:.4f} warmup={row['warmup_steps_fixed']} "
            f"baseline_ckpt_loaded={row['baseline_ckpt_loaded']}",
            flush=True,
        )

    baseline_paths = args.baseline_aggregate or [DEFAULT_BASELINE_AGGREGATE]
    baseline_rows = load_baseline_rows(baseline_paths)
    combined_rows = baseline_rows + cdro_rows
    combined_rows.sort(key=lambda row: (row["method"], float(row["compute_budget_be"])))

    cdro_csv = os.path.join(args.outdir, f"{args.prefix}_cdro_curve.csv")
    combined_csv = os.path.join(args.outdir, f"{args.prefix}_compare_curve.csv")
    summary_json = os.path.join(args.outdir, f"{args.prefix}_compare_curve_summary.json")
    plot_path = os.path.join(args.outdir, f"{args.prefix}_fid_vs_compute.png")
    write_csv(cdro_csv, cdro_rows)
    write_csv(combined_csv, combined_rows)
    make_plot(
        plot_path,
        baseline_rows=baseline_rows,
        cdro_rows=cdro_rows,
        train_percent_label=str(args.train_percent_label),
    )

    summary = {
        "protocol": {
            "name": "independent_per_point_fixed_fraction_warmup",
            "compute_unit": "batch_equiv_denoiser_evals",
            "compute_definition": (
                "One denoiser forward over one training batch counts as 1 unit; "
                "a forward over B*T path states counts as T units. Baseline warmup compute is counted "
                "in the total budget even when an exact warmup checkpoint is reused."
            ),
            "baseline_warmup_fraction": float(args.cdro_warmup_fraction),
            "diagnostics_included": False,
        },
        "steps_list": [int(v) for v in steps_list],
        "baseline_aggregate_paths": baseline_paths,
        "baseline_checkpoint_dir": str(args.baseline_checkpoint_dir),
        "cdro_rows": cdro_rows,
        "baseline_rows": baseline_rows,
        "train_percent_label": str(args.train_percent_label),
        "plot_path": plot_path,
        "combined_csv": combined_csv,
    }
    with open(summary_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"[cdro-curve] wrote {cdro_csv}", flush=True)
    print(f"[cdro-curve] wrote {combined_csv}", flush=True)
    print(f"[cdro-curve] wrote {summary_json}", flush=True)
    print(f"[cdro-curve] wrote {plot_path}", flush=True)


if __name__ == "__main__":
    main()
