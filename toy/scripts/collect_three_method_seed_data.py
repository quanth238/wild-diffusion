#!/usr/bin/env python3
import argparse
import csv
import json
import os
import subprocess
import sys
from typing import Dict, Iterable, List


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


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

# Reuse the WDRO 25-point curve as a shared relative-density template.
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
            "Collect raw per-seed baseline, WDRO, and CDRO curve data with a shared relative-density "
            "checkpoint template. Aggregation is intentionally deferred to later analysis."
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
    parser.add_argument("--baseline-max-steps", type=int, default=80000)
    parser.add_argument("--wdro-max-total-steps", type=int, default=76970)
    parser.add_argument("--cdro-max-total-steps", type=int, default=1700)
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


def realize_shared_grid(max_step: int, *, include_zero: bool) -> List[int]:
    max_step_value = max(int(max_step), 0)
    if max_step_value <= 0:
        return [0] if include_zero else []

    out: List[int] = []
    previous = -1
    template = GRID_TEMPLATE_STEPS if include_zero else GRID_TEMPLATE_STEPS[1:]
    for base_step in template:
        scaled = int(round(float(base_step) * float(max_step_value) / float(GRID_TEMPLATE_MAX)))
        if not include_zero and scaled <= 0:
            scaled = 1
        if scaled <= previous:
            scaled = previous + 1
        if scaled > max_step_value:
            scaled = max_step_value
        if scaled > previous:
            out.append(int(scaled))
            previous = int(scaled)
        if previous >= max_step_value:
            break
    if include_zero and (not out or out[0] != 0):
        out.insert(0, 0)
    if out[-1] != max_step_value:
        out.append(max_step_value)
    return out


def run_command(*, cmd: List[str], log_path: str) -> None:
    ensure_dir(os.path.dirname(log_path))
    with open(log_path, "w", encoding="utf-8") as handle:
        subprocess.run(cmd, cwd=ROOT_DIR, check=True, stdout=handle, stderr=subprocess.STDOUT)


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


def normalize_baseline_runs(*, runs_csv: str) -> List[Dict]:
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
                "source_csv": runs_csv,
                "exp_name": row["exp_name"],
                "exp_dir": row["exp_dir"],
                "checkpoint_path": row["checkpoint_path"],
            }
        )
    return out


def normalize_method_curve(*, csv_path: str, method_name: str, seed: int) -> List[Dict]:
    rows = load_csv_rows(csv_path)
    out: List[Dict] = []
    for row in rows:
        if row.get("method") != method_name:
            continue
        out.append(
            {
                "method": str(method_name),
                "seed": int(seed),
                "step": int(float(row["step"])),
                "fid": float(row["fid"]),
                "train_wall_clock_sec": float(row["train_wall_clock_sec"]),
                "weighted_compute_units": float(row["weighted_compute_units"]),
                "batch_equiv_denoiser_evals": float(row["compute_budget_be"]),
                "images_shown_m": float(row["images_shown_m"]),
                "source_csv": csv_path,
                "metrics_path": row.get("metrics_path", ""),
            }
        )
    return out


def main() -> None:
    args = parse_args()
    seeds = parse_int_list(args.seeds)
    ensure_dir(args.outdir)
    logs_dir = os.path.join(args.outdir, "logs")
    ensure_dir(logs_dir)

    baseline_steps = realize_shared_grid(int(args.baseline_max_steps), include_zero=False)
    wdro_steps = realize_shared_grid(int(args.wdro_max_total_steps), include_zero=True)
    cdro_steps = realize_shared_grid(int(args.cdro_max_total_steps), include_zero=True)

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
        format_steps_list(baseline_steps),
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
        print(f"[collect] reuse baseline outputs: {baseline_outdir}", flush=True)
    else:
        print(f"[collect] baseline seeds={seeds} steps={baseline_steps}", flush=True)
        run_command(cmd=baseline_cmd, log_path=baseline_log)

    wdro_curves: List[Dict[str, str]] = []
    cdro_curves: List[Dict[str, str]] = []
    for seed in seeds:
        wdro_outdir = os.path.join(wdro_root, f"s{seed}")
        wdro_prefix = f"{args.prefix}_wdro_s{seed}"
        wdro_compare_csv = os.path.join(wdro_outdir, f"{wdro_prefix}_compare_curve.csv")
        wdro_log = os.path.join(logs_dir, f"{wdro_prefix}.log")
        wdro_cmd = [
            args.python_bin,
            os.path.join(ROOT_DIR, "toy", "scripts", "sweep_wdro_compute_curve.py"),
            "--outdir",
            wdro_outdir,
            "--prefix",
            wdro_prefix,
            "--seed",
            str(seed),
            "--device",
            args.device,
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
            "--train-percent-label",
            str(args.train_percent_label),
            "--image-split-seed",
            str(args.image_split_seed),
            "--batch-size",
            str(args.batch_size),
            "--hidden-dim",
            str(args.hidden_dim),
            "--eval-samples",
            str(args.eval_samples),
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
            "--max-total-steps",
            str(args.wdro_max_total_steps),
            "--curve-protocol",
            "independent_per_point",
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
            "--steps-list",
            format_steps_list(wdro_steps),
            "--skip-baseline-overlay",
            "--disable-baseline-ckpt",
            "--weighted-compute-calibration-path",
            str(args.weighted_compute_calibration_path).strip(),
            "--train-accelerator-count",
            str(args.train_accelerator_count),
        ]
        if float(args.weighted_inputgrad_alpha) > 0.0 and float(args.weighted_parambackward_beta) > 0.0:
            wdro_cmd.extend(
                [
                    "--weighted-inputgrad-alpha",
                    str(args.weighted_inputgrad_alpha),
                    "--weighted-parambackward-beta",
                    str(args.weighted_parambackward_beta),
                ]
            )
        if args.skip_existing and os.path.isfile(wdro_compare_csv):
            print(f"[collect] reuse wdro seed={seed}: {wdro_compare_csv}", flush=True)
        else:
            print(f"[collect] wdro seed={seed} steps={wdro_steps}", flush=True)
            run_command(cmd=wdro_cmd, log_path=wdro_log)
        wdro_curves.append({"seed": str(seed), "compare_csv": wdro_compare_csv, "outdir": wdro_outdir})

        cdro_outdir = os.path.join(cdro_root, f"s{seed}")
        cdro_prefix = f"{args.prefix}_cdro_s{seed}"
        cdro_compare_csv = os.path.join(cdro_outdir, f"{cdro_prefix}_compare_curve.csv")
        cdro_log = os.path.join(logs_dir, f"{cdro_prefix}.log")
        cdro_cmd = [
            args.python_bin,
            os.path.join(ROOT_DIR, "toy", "scripts", "sweep_cdro_compute_curve.py"),
            "--outdir",
            cdro_outdir,
            "--prefix",
            cdro_prefix,
            "--seed",
            str(seed),
            "--device",
            args.device,
            "--dataset-path",
            args.dataset_path,
            "--dataset-val-path",
            args.dataset_val_path,
            "--fid-ref-path",
            args.fid_ref_path,
            "--skip-baseline-overlay",
            "--disable-baseline-ckpt",
            "--image-size",
            str(args.image_size),
            "--image-channels",
            str(args.image_channels),
            "--image-train-size",
            str(args.image_train_size),
            "--image-val-size",
            str(args.image_val_size),
            "--train-percent-label",
            str(args.train_percent_label),
            "--image-split-seed",
            str(args.image_split_seed),
            "--batch-size",
            str(args.batch_size),
            "--hidden-dim",
            str(args.hidden_dim),
            "--eval-samples",
            str(args.eval_samples),
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
            "--steps-list",
            format_steps_list(cdro_steps),
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
            "--weighted-compute-calibration-path",
            str(args.weighted_compute_calibration_path).strip(),
            "--train-accelerator-count",
            str(args.train_accelerator_count),
        ]
        if float(args.weighted_inputgrad_alpha) > 0.0 and float(args.weighted_parambackward_beta) > 0.0:
            cdro_cmd.extend(
                [
                    "--weighted-inputgrad-alpha",
                    str(args.weighted_inputgrad_alpha),
                    "--weighted-parambackward-beta",
                    str(args.weighted_parambackward_beta),
                ]
            )
        if args.skip_existing and os.path.isfile(cdro_compare_csv):
            print(f"[collect] reuse cdro seed={seed}: {cdro_compare_csv}", flush=True)
        else:
            print(f"[collect] cdro seed={seed} steps={cdro_steps}", flush=True)
            run_command(cmd=cdro_cmd, log_path=cdro_log)
        cdro_curves.append({"seed": str(seed), "compare_csv": cdro_compare_csv, "outdir": cdro_outdir})

    baseline_raw = normalize_baseline_runs(runs_csv=baseline_runs_csv)
    wdro_raw = []
    for item in wdro_curves:
        wdro_raw.extend(normalize_method_curve(csv_path=item["compare_csv"], method_name="wdro", seed=int(item["seed"])))
    cdro_raw = []
    for item in cdro_curves:
        cdro_raw.extend(normalize_method_curve(csv_path=item["compare_csv"], method_name="cdro", seed=int(item["seed"])))

    combined_raw = baseline_raw + wdro_raw + cdro_raw
    combined_raw.sort(key=lambda row: (str(row["method"]), int(row["seed"]), int(row["step"])))
    baseline_raw.sort(key=lambda row: (int(row["seed"]), int(row["step"])))
    wdro_raw.sort(key=lambda row: (int(row["seed"]), int(row["step"])))
    cdro_raw.sort(key=lambda row: (int(row["seed"]), int(row["step"])))

    baseline_raw_csv = os.path.join(args.outdir, f"{args.prefix}_baseline_raw_seed_rows.csv")
    wdro_raw_csv = os.path.join(args.outdir, f"{args.prefix}_wdro_raw_seed_rows.csv")
    cdro_raw_csv = os.path.join(args.outdir, f"{args.prefix}_cdro_raw_seed_rows.csv")
    combined_raw_csv = os.path.join(args.outdir, f"{args.prefix}_all_methods_raw_seed_rows.csv")
    write_csv(baseline_raw_csv, baseline_raw)
    write_csv(wdro_raw_csv, wdro_raw)
    write_csv(cdro_raw_csv, cdro_raw)
    write_csv(combined_raw_csv, combined_raw)

    manifest = {
        "protocol": {
            "name": "three_method_raw_seed_collection",
            "description": (
                "Raw per-seed data collection only. Baseline, WDRO, and CDRO are run independently; "
                "aggregation is intentionally deferred to later analysis."
            ),
            "seeds": seeds,
            "shared_grid_template_name": "wdro_dense_25_relative",
            "shared_grid_template_steps": GRID_TEMPLATE_STEPS,
            "grid_strategy": (
                "Use the same 25-knot relative-density template for each method, scaled to that method's "
                "max-step horizon rather than forcing exact step alignment."
            ),
            "primary_metric": "train_wall_clock_sec",
            "secondary_metric": "weighted_compute_units",
            "legacy_metric": "batch_equiv_denoiser_evals",
            "baseline_aggregate_role": "summary_only_not_used_for_wdro_cdro_raw_collection",
        },
        "grids": {
            "baseline_steps": baseline_steps,
            "wdro_steps": wdro_steps,
            "cdro_steps": cdro_steps,
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
            "wdro": wdro_curves,
            "cdro": cdro_curves,
        },
    }
    manifest_path = os.path.join(args.outdir, f"{args.prefix}_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"[collect] wrote {baseline_raw_csv}", flush=True)
    print(f"[collect] wrote {wdro_raw_csv}", flush=True)
    print(f"[collect] wrote {cdro_raw_csv}", flush=True)
    print(f"[collect] wrote {combined_raw_csv}", flush=True)
    print(f"[collect] wrote {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
