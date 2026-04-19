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
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from toy.compute_accounting import load_weighted_compute_calibration, weighted_compute_units


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
    parser.add_argument("--skip-baseline-overlay", action="store_true")
    parser.add_argument("--disable-baseline-ckpt", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--weighted-compute-calibration-path", type=str, default="")
    parser.add_argument("--weighted-inputgrad-alpha", type=float, default=0.0)
    parser.add_argument("--weighted-parambackward-beta", type=float, default=0.0)
    parser.add_argument("--train-accelerator-count", type=int, default=1)
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


def resolve_weighted_calibration(args: argparse.Namespace) -> Dict:
    return load_weighted_compute_calibration(
        calibration_path=str(args.weighted_compute_calibration_path).strip(),
        inputgrad_alpha=float(args.weighted_inputgrad_alpha),
        parambackward_beta=float(args.weighted_parambackward_beta),
        training_objective="edm",
        image_backbone="conv",
        batch_size=int(args.batch_size),
        hidden_dim=int(args.hidden_dim),
        image_size=int(args.image_size),
        image_channels=int(args.image_channels),
        device=str(args.device),
    )


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


def extract_wdro_row(
    metrics_path: str,
    *,
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
    if total_weighted_compute is None and calibration.get("available", False):
        baseline_weighted_compute = weighted_compute_units(
            n_fwd=0.0,
            n_fwd_inputgrad=0.0,
            n_fwd_parambackward=float(baseline_phase_steps),
            calibration=calibration,
        )
        attack_inputgrad_units = max(float(robust_compute_be_raw) - float(robust_phase_steps), 0.0)
        robust_weighted_compute = weighted_compute_units(
            n_fwd=0.0,
            n_fwd_inputgrad=float(attack_inputgrad_units),
            n_fwd_parambackward=float(robust_phase_steps),
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
        "weighted_compute_units": None if total_weighted_compute is None else float(total_weighted_compute),
        "baseline_weighted_compute_units": (
            None if baseline_weighted_compute is None else float(baseline_weighted_compute)
        ),
        "robust_weighted_compute_units": None if robust_weighted_compute is None else float(robust_weighted_compute),
        "images_shown_m": float(total_images_shown_m_effective),
        "images_shown_m_raw": float(
            budget_accounting.get("effective_train_images_seen_total", total_steps_requested * batch_size)
        ) / 1_000_000.0,
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
        "warmup_steps_fixed": int(flow["baseline_phase_steps"]),
        "robust_steps_target": int(flow["robust_phase_steps"]),
        "warmup_only": bool(warmup_only),
        "wdro_refreshes": len(objective.get("wdro_refresh_steps", [])),
        "wdro_final_dataset_size": (
            int(objective["wdro_dataset_sizes"][-1]) if objective.get("wdro_dataset_sizes") else 0
        ),
        "baseline_ckpt_enabled": bool(flow.get("baseline_ckpt_enabled", False)),
        "baseline_ckpt_loaded": bool(flow.get("baseline_ckpt_loaded", False)),
        "robust_resume_loaded": bool(flow.get("robust_resume_loaded", False)),
        "phase_step_split_mode": str(flow.get("phase_step_split_mode", "")),
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


def make_plot(
    path: str,
    baseline_rows: List[Dict],
    wdro_rows: List[Dict],
    *,
    train_percent_label: str,
    x_key: str,
    x_label: str,
    title_suffix: str,
    target_x: float | None = None,
    target_label: str | None = None,
) -> None:
    plt.figure(figsize=(8, 5))
    baseline_pairs = [(row[x_key], row["fid"]) for row in baseline_rows if row.get(x_key) is not None]
    wdro_pairs = [(row[x_key], row["fid"]) for row in wdro_rows if row.get(x_key) is not None]
    if baseline_pairs:
        plt.plot(
            [value for value, _ in baseline_pairs],
            [fid for _, fid in baseline_pairs],
            marker="o",
            linewidth=2.0,
            label="Baseline EDM",
        )
    if wdro_pairs:
        plt.plot(
            [value for value, _ in wdro_pairs],
            [fid for _, fid in wdro_pairs],
            marker="o",
            linewidth=2.0,
            label="WDRO",
        )
    if target_x is not None:
        plt.axvline(
            float(target_x),
            color="gray",
            linestyle="--",
            linewidth=1.5,
            label=(target_label or "Target"),
        )
    plt.xlabel(x_label)
    plt.ylabel("FID")
    plt.title(f"Simpsons-MNIST RGB {train_percent_label}: FID vs {title_suffix}")
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
    weighted_calibration = resolve_weighted_calibration(args)

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
    print(
        "[wdro-curve] weighted_compute "
        f"available={weighted_calibration['available']} "
        f"source={weighted_calibration['source']} "
        f"alpha={weighted_calibration['inputgrad_alpha']} "
        f"beta={weighted_calibration['parambackward_beta']}",
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
        row = extract_wdro_row(
            metrics_path,
            calibration=weighted_calibration,
            train_accelerator_count=int(args.train_accelerator_count),
        )
        wdro_rows.append(row)
        print(
            f"[wdro-curve] step={row['step']} train_wall_clock_sec={row['train_wall_clock_sec']} "
            f"weighted_compute_units={row['weighted_compute_units']} "
            f"fid={row['fid']:.4f} images_m={row['images_shown_m']:.4f}",
            flush=True,
        )

    baseline_paths = [] if bool(args.skip_baseline_overlay) else (args.baseline_aggregate or DEFAULT_BASELINE_AGGREGATES)
    baseline_rows = load_baseline_rows(
        baseline_paths,
        calibration=weighted_calibration,
        train_accelerator_count=int(args.train_accelerator_count),
    )
    combined_rows = baseline_rows + wdro_rows
    combined_rows.sort(
        key=lambda row: (
            row["method"],
            float(row["train_wall_clock_sec"]) if row.get("train_wall_clock_sec") is not None else float("inf"),
            float(row["weighted_compute_units"]) if row.get("weighted_compute_units") is not None else float("inf"),
            float(row["compute_budget_be"]),
        )
    )

    wdro_csv = os.path.join(args.outdir, f"{args.prefix}_wdro_curve.csv")
    combined_csv = os.path.join(args.outdir, f"{args.prefix}_compare_curve.csv")
    summary_json = os.path.join(args.outdir, f"{args.prefix}_compare_curve_summary.json")
    plot_wall_clock = os.path.join(args.outdir, f"{args.prefix}_fid_vs_train_wall_clock.png")
    plot_weighted = os.path.join(args.outdir, f"{args.prefix}_fid_vs_weighted_compute.png")
    plot_legacy = os.path.join(args.outdir, f"{args.prefix}_fid_vs_batch_equiv.png")
    write_csv(wdro_csv, wdro_rows)
    write_csv(combined_csv, combined_rows)
    make_plot(
        plot_wall_clock,
        baseline_rows=baseline_rows,
        wdro_rows=wdro_rows,
        train_percent_label=str(args.train_percent_label),
        x_key="train_wall_clock_sec",
        x_label="Train Wall-Clock (sec)",
        title_suffix="Train Wall-Clock",
    )
    make_plot(
        plot_weighted,
        baseline_rows=baseline_rows,
        wdro_rows=wdro_rows,
        train_percent_label=str(args.train_percent_label),
        x_key="weighted_compute_units",
        x_label="Weighted Compute Units",
        title_suffix="Weighted Compute",
    )
    make_plot(
        plot_legacy,
        baseline_rows=baseline_rows,
        wdro_rows=wdro_rows,
        train_percent_label=str(args.train_percent_label),
        x_key="compute_budget_be",
        x_label="Legacy Batch-Equivalent Denoiser Evals",
        title_suffix="Legacy Batch-Equivalent Compute",
        target_x=float(target_be),
        target_label="20 MIMG baseline compute",
    )

    summary = {
        "protocol": {
            "target_baseline_mimg": float(args.target_baseline_mimg),
            "target_compute_be": int(target_be),
            "curve_protocol": str(args.curve_protocol),
            "chosen_max_total_steps": int(max_total_steps),
            "warmup_steps_fixed": int(warmup_steps_fixed),
            "baseline_checkpoint_reuse_enabled": bool(not args.disable_baseline_ckpt),
            "baseline_overlay_included": bool(baseline_rows),
            "primary_metric": "train_wall_clock_sec",
            "secondary_metric": "weighted_compute_units",
            "legacy_metric": "batch_equiv_denoiser_evals",
            "weighted_compute_calibration": weighted_calibration,
            "train_accelerator_count": int(args.train_accelerator_count),
        },
        "steps_list": [int(v) for v in steps_list],
        "baseline_aggregate_paths": baseline_paths,
        "wdro_resume_path": resume_path,
        "wdro_rows": wdro_rows,
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
    print(f"[wdro-curve] wrote {wdro_csv}", flush=True)
    print(f"[wdro-curve] wrote {combined_csv}", flush=True)
    print(f"[wdro-curve] wrote {summary_json}", flush=True)
    print(f"[wdro-curve] wrote {plot_wall_clock}", flush=True)
    print(f"[wdro-curve] wrote {plot_weighted}", flush=True)
    print(f"[wdro-curve] wrote {plot_legacy}", flush=True)


if __name__ == "__main__":
    main()
