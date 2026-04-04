#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import subprocess
import sys
from typing import Dict, List

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
DEFAULT_BASELINE_AGGREGATES = [
    os.path.join(
        ROOT_DIR,
        "toy_outputs",
        "simpsons_mnist_rgb_convergence_ckpt_20pct",
        "simpsons_mnist_rgb_baseline_curve_ckpt_20pct_aggregate.csv",
    ),
    os.path.join(
        ROOT_DIR,
        "toy_outputs",
        "simpsons_mnist_rgb_convergence_ckpt_20pct_focus",
        "simpsons_mnist_rgb_baseline_curve_ckpt_20pct_focus_aggregate.csv",
    ),
]
DEFAULT_FID_DETECTOR = (
    "/root/.cache/dnnlib/downloads/"
    "18d9c1159d16cd4cc6adf7db0f2dd2a9_https___api.ngc.nvidia.com_v2_models_nvidia_research_"
    "stylegan3_versions_1_files_metrics_inception-2015-12-05.pkl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a paper-style WDRO curve on Simpsons-MNIST RGB and compare against baseline EDM."
    )
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--prefix", type=str, required=True)
    parser.add_argument("--python-bin", type=str, default=os.path.join(ROOT_DIR, ".venv", "bin", "python"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dataset-path", type=str, default=DEFAULT_TRAIN_ROOT)
    parser.add_argument("--dataset-val-path", type=str, default=DEFAULT_VAL_ROOT)
    parser.add_argument("--fid-ref-path", type=str, default=DEFAULT_FID_REF)
    parser.add_argument("--image-size", type=int, default=28)
    parser.add_argument("--image-channels", type=int, default=3)
    parser.add_argument("--image-train-size", type=int, default=1600)
    parser.add_argument("--image-val-size", type=int, default=2000)
    parser.add_argument("--train-percent-label", type=str, default="20%")
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
    parser.add_argument("--target-baseline-mimg", type=float, default=20.0)
    parser.add_argument(
        "--curve-protocol",
        type=str,
        default="single_trajectory",
        choices=["single_trajectory", "independent_per_point"],
    )
    parser.add_argument("--wdro-warmup-fraction", type=float, default=0.2)
    parser.add_argument("--wdro-refresh-epochs", type=float, default=100.0)
    parser.add_argument("--wdro-adv-prob", type=float, default=0.3)
    parser.add_argument("--wdro-attack-steps", type=int, default=2)
    parser.add_argument("--wdro-attack-step-size", type=float, default=1e-3)
    parser.add_argument("--wdro-gamma", type=float, default=1.0)
    parser.add_argument("--steps-list", type=str, default="")
    parser.add_argument("--max-total-steps", type=int, default=0)
    parser.add_argument("--baseline-aggregate", action="append", default=[])
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def estimate_wdro_total_be(
    *,
    total_steps: int,
    warmup_fraction: float,
    train_size: int,
    batch_size: int,
    refresh_epochs: float,
    adv_prob: float,
    attack_steps: int,
) -> float:
    warmup_steps = max(0, min(int(total_steps * warmup_fraction), total_steps))
    robust_steps = max(total_steps - warmup_steps, 0)
    if robust_steps <= 0:
        return float(warmup_steps)
    refresh_interval = max(int(math.ceil(refresh_epochs * train_size / max(batch_size, 1))), 1)
    batches_per_refresh = max(int(math.ceil(train_size / max(batch_size, 1))), 1)
    refresh_count = 1 + ((robust_steps - 1) // refresh_interval)
    extra_be = float(refresh_count) * float(batches_per_refresh) * float(adv_prob) * float(max(attack_steps, 0))
    return float(total_steps) + extra_be


def choose_total_steps_for_target_be(args: argparse.Namespace) -> int:
    if args.max_total_steps > 0:
        return int(args.max_total_steps)
    target_be = int(round(float(args.target_baseline_mimg) * 1_000_000.0 / float(args.batch_size)))
    lo = max(1, target_be - 8000)
    hi = max(lo, target_be)
    best_steps = lo
    best_gap = None
    for total_steps in range(lo, hi + 1):
        be_est = estimate_wdro_total_be(
            total_steps=total_steps,
            warmup_fraction=float(args.wdro_warmup_fraction),
            train_size=int(args.image_train_size),
            batch_size=int(args.batch_size),
            refresh_epochs=float(args.wdro_refresh_epochs),
            adv_prob=float(args.wdro_adv_prob),
            attack_steps=int(args.wdro_attack_steps),
        )
        gap = abs(be_est - float(target_be))
        if best_gap is None or gap < best_gap:
            best_gap = gap
            best_steps = total_steps
    return int(best_steps)


def default_steps_list(max_total_steps: int, warmup_steps: int) -> List[int]:
    candidates = [0, warmup_steps, 20000, 25000, 30000, 35000, 40000, 50000, 60000, 70000, max_total_steps]
    values = sorted({int(v) for v in candidates if int(v) >= 0 and int(v) <= int(max_total_steps)})
    if not values or values[-1] != int(max_total_steps):
        values.append(int(max_total_steps))
    return values


def parse_steps_list(value: str, max_total_steps: int, warmup_steps: int) -> List[int]:
    if not value.strip():
        return default_steps_list(max_total_steps, warmup_steps)
    steps = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    return [step for step in steps if step >= 0 and step <= max_total_steps]


def maybe_set_fid_detector_env(env: Dict[str, str]) -> Dict[str, str]:
    env = dict(env)
    if "FID_DETECTOR_PATH" not in env and os.path.isfile(DEFAULT_FID_DETECTOR):
        env["FID_DETECTOR_PATH"] = DEFAULT_FID_DETECTOR
    return env


def run_point(
    *,
    args: argparse.Namespace,
    target_total_steps: int,
    warmup_steps: int,
    resume_path: str,
    exp_name: str,
    log_path: str,
    use_resume: bool,
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
        "wdro",
        "--baseline-steps-override",
        str(warmup_steps),
        "--disable-baseline-gate",
        "--skip-checks",
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
    if use_resume:
        cmd.extend([
            "--robust-save-ckpt-path",
            resume_path,
        ])
    if use_resume and os.path.exists(resume_path):
        cmd.extend(["--robust-resume-ckpt-path", resume_path])
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


def extract_wdro_row(metrics_path: str) -> Dict:
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
        "baseline_compute_be_raw": float(baseline_compute_be_raw),
        "robust_compute_be": float(robust_compute_be_raw),
        "images_shown_m": float(total_images_shown_m_effective),
        "images_shown_m_raw": float(
            budget_accounting.get("effective_train_images_seen_total", total_steps_requested * batch_size)
        ) / 1_000_000.0,
        "fid": float(fid_value),
        "baseline_fid_same_run": (
            float(sample_quality["baseline_fid"]) if sample_quality.get("baseline_fid") is not None else float("nan")
        ),
        "train_elapsed_sec": float(runtime["effective_train_total"]),
        "runtime_total_sec": float(runtime["total"]),
        "warmup_steps_fixed": int(flow["baseline_phase_steps"]),
        "robust_steps_target": int(flow["robust_phase_steps"]),
        "warmup_only": bool(warmup_only),
        "wdro_refreshes": len(objective.get("wdro_refresh_steps", [])),
        "wdro_final_dataset_size": (
            int(objective["wdro_dataset_sizes"][-1]) if objective.get("wdro_dataset_sizes") else 0
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


def make_plot(
    path: str,
    baseline_rows: List[Dict],
    wdro_rows: List[Dict],
    target_be: float,
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
    if wdro_rows:
        plt.plot(
            [row["compute_budget_be"] for row in wdro_rows],
            [row["fid"] for row in wdro_rows],
            marker="o",
            linewidth=2.0,
            label="WDRO",
        )
    plt.axvline(float(target_be), color="gray", linestyle="--", linewidth=1.5, label="20 MIMG baseline compute")
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

    target_be = int(round(float(args.target_baseline_mimg) * 1_000_000.0 / float(args.batch_size)))
    max_total_steps = choose_total_steps_for_target_be(args)
    warmup_steps_fixed = int(max_total_steps * float(args.wdro_warmup_fraction))
    steps_list = parse_steps_list(args.steps_list, max_total_steps=max_total_steps, warmup_steps=warmup_steps_fixed)
    resume_path = os.path.join(args.outdir, f"{args.prefix}_resume.pt")
    use_resume_chain = bool(str(args.curve_protocol) == "single_trajectory")

    print(
        "[wdro-curve] target_baseline_mimg="
        f"{args.target_baseline_mimg} target_compute_be={target_be} "
        f"chosen_max_total_steps={max_total_steps} warmup_steps_fixed={warmup_steps_fixed} "
        f"curve_protocol={args.curve_protocol}",
        flush=True,
    )
    print(f"[wdro-curve] checkpoints={steps_list}", flush=True)

    wdro_rows: List[Dict] = []
    for index, target_total_steps in enumerate(steps_list):
        exp_name = f"{args.prefix}_st{target_total_steps}_s{args.seed}"
        metrics_path = os.path.join(args.outdir, exp_name, "metrics.json")
        log_path = os.path.join(logs_dir, f"{exp_name}.log")
        if args.skip_existing and os.path.isfile(metrics_path):
            print(f"[wdro-curve] reuse existing metrics for step={target_total_steps}: {metrics_path}", flush=True)
        else:
            warmup_steps_point = int(target_total_steps * float(args.wdro_warmup_fraction))
            print(
                f"[wdro-curve] run step={target_total_steps} warmup_steps={warmup_steps_point} "
                f"use_resume={use_resume_chain} log={log_path}",
                flush=True,
            )
            run_point(
                args=args,
                target_total_steps=target_total_steps,
                warmup_steps=warmup_steps_point,
                resume_path=resume_path,
                exp_name=exp_name,
                log_path=log_path,
                use_resume=use_resume_chain,
            )
        row = extract_wdro_row(metrics_path)
        wdro_rows.append(row)
        print(
            f"[wdro-curve] step={row['step']} compute_be={row['compute_budget_be']:.1f} "
            f"fid={row['fid']:.4f} images_m={row['images_shown_m']:.4f}",
            flush=True,
        )

    baseline_paths = args.baseline_aggregate or DEFAULT_BASELINE_AGGREGATES
    baseline_rows = load_baseline_rows(baseline_paths)
    combined_rows = baseline_rows + wdro_rows
    combined_rows.sort(key=lambda row: (row["method"], float(row["compute_budget_be"])))

    wdro_csv = os.path.join(args.outdir, f"{args.prefix}_wdro_curve.csv")
    combined_csv = os.path.join(args.outdir, f"{args.prefix}_compare_curve.csv")
    summary_json = os.path.join(args.outdir, f"{args.prefix}_compare_curve_summary.json")
    plot_path = os.path.join(args.outdir, f"{args.prefix}_fid_vs_compute.png")
    write_csv(wdro_csv, wdro_rows)
    write_csv(combined_csv, combined_rows)
    make_plot(
        plot_path,
        baseline_rows=baseline_rows,
        wdro_rows=wdro_rows,
        target_be=float(target_be),
        train_percent_label=str(args.train_percent_label),
    )

    summary = {
        "target_baseline_mimg": float(args.target_baseline_mimg),
        "target_compute_be": int(target_be),
        "curve_protocol": str(args.curve_protocol),
        "chosen_max_total_steps": int(max_total_steps),
        "warmup_steps_fixed": int(warmup_steps_fixed),
        "steps_list": [int(v) for v in steps_list],
        "baseline_aggregate_paths": baseline_paths,
        "wdro_resume_path": resume_path,
        "wdro_rows": wdro_rows,
        "baseline_rows": baseline_rows,
        "train_percent_label": str(args.train_percent_label),
        "plot_path": plot_path,
        "combined_csv": combined_csv,
    }
    with open(summary_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"[wdro-curve] wrote {wdro_csv}", flush=True)
    print(f"[wdro-curve] wrote {combined_csv}", flush=True)
    print(f"[wdro-curve] wrote {summary_json}", flush=True)
    print(f"[wdro-curve] wrote {plot_path}", flush=True)


if __name__ == "__main__":
    main()
