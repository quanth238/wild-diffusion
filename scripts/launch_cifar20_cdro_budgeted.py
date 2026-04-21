#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.cifar_cdro_budget_utils import (  # noqa: E402
    DEFAULT_BASELINE_RESUME_KIMG,
    DEFAULT_BASELINE_RUN_DIR,
    DEFAULT_CALIBRATION_JSON,
    DEFAULT_WDRO_COMPARE_CSV,
    DEFAULT_WDRO_SUMMARY_JSON,
    build_budget_plan,
    calibration_from_path,
    cdro_robust_step_compute_be,
    cdro_robust_step_weighted_compute_units,
    estimate_total_wall_clock_sec,
    load_warmup_summary,
    resolve_budget_target,
)


def parse_bool_arg(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare and optionally launch a real CIFAR-10 20% CDRO run that matches "
            "the existing Baseline/WDRO comparison by weighted compute budget."
        )
    )
    parser.add_argument(
        "--outdir-root",
        type=str,
        default="/home/bachlc/GM-CDRO/training-runs/paper-cifar10-cdro-fp16",
    )
    parser.add_argument("--baseline-run-dir", type=str, default=DEFAULT_BASELINE_RUN_DIR)
    parser.add_argument("--baseline-resume-kimg", type=int, default=DEFAULT_BASELINE_RESUME_KIMG)
    parser.add_argument("--compare-csv", type=str, default=DEFAULT_WDRO_COMPARE_CSV)
    parser.add_argument("--summary-json", type=str, default=DEFAULT_WDRO_SUMMARY_JSON)
    parser.add_argument("--calibration-json", type=str, default=DEFAULT_CALIBRATION_JSON)
    parser.add_argument(
        "--target-mode",
        type=str,
        choices=["wdro_best", "wdro_final", "midpoint", "baseline_best"],
        default="wdro_final",
    )
    parser.add_argument("--target-wcu", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--batch-gpu", type=int, default=1024)
    parser.add_argument("--arch", type=str, choices=["ddpmpp", "ncsnpp", "adm"], default="ddpmpp")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cifar-train-percent", type=int, default=20)
    parser.add_argument("--cifar-train-seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--augment", type=float, default=0.12)
    parser.add_argument("--fp16", type=int, choices=[0, 1], default=1)
    parser.add_argument("--cdro-n-steps-path", type=int, default=32)
    parser.add_argument("--cdro-step-size", type=float, default=0.02)
    parser.add_argument("--cdro-total-budget-rho", type=float, default=32.0)
    parser.add_argument("--cdro-time-horizon", type=float, default=1.0)
    parser.add_argument("--cdro-sigma-min", type=float, default=0.002)
    parser.add_argument("--cdro-sigma-max", type=float, default=80.0)
    parser.add_argument(
        "--cdro-edm-ladder-mode",
        type=str,
        choices=["deterministic_midpoint_quantile", "stochastic_stratified_quantile"],
        default="stochastic_stratified_quantile",
    )
    parser.add_argument("--cdro-per-example-sigma-ladders", type=parse_bool_arg, default=True)
    parser.add_argument("--attack-num-steps", type=int, default=1)
    parser.add_argument("--outer-attack-weight", type=float, default=0.3)
    parser.add_argument("--outer-clean-weight", type=float, default=0.0)
    parser.add_argument("--tick-kimg", type=int, default=128)
    parser.add_argument("--snap-ticks", type=int, default=1)
    parser.add_argument("--dump-ticks", type=int, default=8)
    parser.add_argument("--env-mode", type=str, default="venv")
    parser.add_argument("--venv-dir", type=str, default="/home/bachlc/.venvs/wild-diffusion-h100")
    parser.add_argument("--install-deps", type=str, default="0")
    parser.add_argument("--robust-step-wall-clock-sec", type=float, default=None)
    parser.add_argument("--launch", action="store_true", default=False)
    return parser.parse_args()


def next_run_dir(outdir_root: Path, desc: str) -> Path:
    outdir_root.mkdir(parents=True, exist_ok=True)
    prev_ids = []
    for child in outdir_root.iterdir():
        if not child.is_dir():
            continue
        prefix = child.name.split("-", 1)[0]
        if prefix.isdigit():
            prev_ids.append(int(prefix))
    run_id = max(prev_ids, default=-1) + 1
    return outdir_root / f"{run_id:05d}-{desc}"


def format_run_desc(args: argparse.Namespace, target_wcu: float) -> str:
    wcu_tag = f"{int(round(float(target_wcu))):06d}"
    desc = (
        f"cifar10-32x32-train{int(args.cifar_train_percent)}pct-seed{int(args.seed)}-"
        f"uncond-{args.arch}-cdroedm-gpus1-batch{int(args.batch_size)}-"
        f"{'fp16' if int(args.fp16) else 'fp32'}-paper-cifar10-uncond-{args.arch}-cdro-"
        f"{int(args.cifar_train_percent)}pct-n{int(args.cdro_n_steps_path):03d}-"
        f"rho{float(args.cdro_total_budget_rho):.1f}-i{int(args.attack_num_steps)}-"
        f"aw{float(args.outer_attack_weight):.2f}-cw{float(args.outer_clean_weight):.2f}-"
        f"pel{int(bool(args.cdro_per_example_sigma_ladders))}-"
        f"bg{int(args.batch_gpu)}-resume{int(args.baseline_resume_kimg):06d}-wcu{wcu_tag}"
    )
    return desc.replace(".", "p")


def build_launch_env(args: argparse.Namespace, *, resume_state: Path, total_kimg_int: int) -> dict:
    extra_args = [
        f"--cdro-n-steps-path={int(args.cdro_n_steps_path)}",
        f"--cdro-step-size={float(args.cdro_step_size)}",
        f"--cdro-total-budget-rho={float(args.cdro_total_budget_rho)}",
        f"--cdro-time-horizon={float(args.cdro_time_horizon)}",
        f"--cdro-sigma-min={float(args.cdro_sigma_min)}",
        f"--cdro-sigma-max={float(args.cdro_sigma_max)}",
        f"--cdro-edm-ladder-mode={str(args.cdro_edm_ladder_mode)}",
        f"--cdro-per-example-sigma-ladders={'True' if bool(args.cdro_per_example_sigma_ladders) else 'False'}",
        f"--attack-num-steps={int(args.attack_num_steps)}",
        f"--outer-attack-weight={float(args.outer_attack_weight)}",
        f"--outer-clean-weight={float(args.outer_clean_weight)}",
    ]
    env = os.environ.copy()
    env.update(
        {
            "PYTORCH_CUDA_ALLOC_CONF": env.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
            "ENV_MODE": str(args.env_mode),
            "VENV_DIR": str(args.venv_dir),
            "INSTALL_DEPS": str(args.install_deps),
            "RESUME": str(resume_state),
            "DURATION_MIMG": f"{float(total_kimg_int) / 1000.0:.6f}",
            "BATCH": str(int(args.batch_size)),
            "BATCH_GPU": str(int(args.batch_gpu)),
            "LR": str(float(args.lr)),
            "WORKERS": str(int(args.workers)),
            "ARCH": str(args.arch),
            "PRECOND": "cdroedm",
            "COND": "0",
            "FP16": str(int(args.fp16)),
            "AUGMENT": str(float(args.augment)),
            "DEBUG_EVAL": "0",
            "DEBUG_ADV_VISUAL": "0",
            "CIFAR_TRAIN_PERCENT": str(int(args.cifar_train_percent)),
            "CIFAR_TRAIN_SEED": str(int(args.cifar_train_seed)),
            "TICK_KIMG": str(int(args.tick_kimg)),
            "SNAP_TICKS": str(int(args.snap_ticks)),
            "DUMP_TICKS": str(int(args.dump_ticks)),
            "SEED": str(int(args.seed)),
            "EXTRA_TRAIN_ARGS": " ".join(extra_args),
        }
    )
    return env


def main() -> None:
    args = parse_args()
    calibration = calibration_from_path(args.calibration_json)
    budget_target = resolve_budget_target(args.compare_csv, args.target_mode)
    warmup_summary = load_warmup_summary(args.summary_json)
    target_wcu = float(args.target_wcu) if args.target_wcu is not None else float(budget_target["target_wcu"])
    robust_step_wcu = cdro_robust_step_weighted_compute_units(
        calibration=calibration,
        n_steps_path=args.cdro_n_steps_path,
        attack_num_steps=args.attack_num_steps,
        outer_attack_weight=args.outer_attack_weight,
        outer_clean_weight=args.outer_clean_weight,
        total_budget_rho=args.cdro_total_budget_rho,
    )
    robust_step_compute_be = cdro_robust_step_compute_be(
        n_steps_path=args.cdro_n_steps_path,
        attack_num_steps=args.attack_num_steps,
        outer_attack_weight=args.outer_attack_weight,
        outer_clean_weight=args.outer_clean_weight,
        total_budget_rho=args.cdro_total_budget_rho,
    )
    plan = build_budget_plan(
        target_wcu=target_wcu,
        warmup_kimg=warmup_summary["warmup_kimg"],
        warmup_wcu=warmup_summary["warmup_weighted_compute_units"],
        warmup_compute_be=warmup_summary["warmup_compute_be"],
        batch_size=args.batch_size,
        robust_step_wcu=robust_step_wcu,
        robust_step_compute_be=robust_step_compute_be,
    )

    baseline_run_dir = Path(args.baseline_run_dir).resolve()
    src_state = baseline_run_dir / f"training-state-{int(args.baseline_resume_kimg):06d}.pt"
    src_snapshot = baseline_run_dir / f"network-snapshot-{int(args.baseline_resume_kimg):06d}.pkl"
    if not src_state.is_file():
        raise FileNotFoundError(f"Missing baseline resume state: {src_state}")
    if not src_snapshot.is_file():
        raise FileNotFoundError(f"Missing baseline resume snapshot: {src_snapshot}")

    run_desc = format_run_desc(args, target_wcu=target_wcu)
    run_dir = next_run_dir(Path(args.outdir_root).resolve(), run_desc)
    dst_state = run_dir / src_state.name
    dst_snapshot = run_dir / src_snapshot.name
    launch_env = build_launch_env(args, resume_state=dst_state, total_kimg_int=int(plan["total_kimg_int"]))
    launch_cmd = ["bash", str(ROOT_DIR / "scripts" / "setup_and_train_cifar10.sh")]

    estimated_total_wall_clock_sec = None
    if args.robust_step_wall_clock_sec is not None:
        estimated_total_wall_clock_sec = estimate_total_wall_clock_sec(
            warmup_wall_clock_sec=warmup_summary["warmup_train_wall_clock_sec"],
            robust_steps_int=int(plan["robust_steps_int"]),
            robust_step_wall_clock_sec=float(args.robust_step_wall_clock_sec),
        )

    plan_payload = {
        "launcher": Path(__file__).resolve().as_posix(),
        "run_dir": str(run_dir),
        "baseline_run_dir": str(baseline_run_dir),
        "bootstrap_resume_state_src": str(src_state),
        "bootstrap_resume_snapshot_src": str(src_snapshot),
        "bootstrap_resume_state_dst": str(dst_state),
        "bootstrap_resume_snapshot_dst": str(dst_snapshot),
        "compare_csv": str(Path(args.compare_csv).resolve()),
        "summary_json": str(Path(args.summary_json).resolve()),
        "calibration_json": str(Path(args.calibration_json).resolve()),
        "target_mode": str(args.target_mode),
        "target_wcu_requested": float(target_wcu),
        "target_envelope": budget_target,
        "warmup_summary": warmup_summary,
        "cdro_config": {
            "cdro_n_steps_path": int(args.cdro_n_steps_path),
            "cdro_step_size": float(args.cdro_step_size),
            "cdro_total_budget_rho": float(args.cdro_total_budget_rho),
            "cdro_time_horizon": float(args.cdro_time_horizon),
            "cdro_sigma_min": float(args.cdro_sigma_min),
            "cdro_sigma_max": float(args.cdro_sigma_max),
            "cdro_edm_ladder_mode": str(args.cdro_edm_ladder_mode),
            "cdro_per_example_sigma_ladders": bool(args.cdro_per_example_sigma_ladders),
            "attack_num_steps": int(args.attack_num_steps),
            "outer_attack_weight": float(args.outer_attack_weight),
            "outer_clean_weight": float(args.outer_clean_weight),
        },
        "robust_step_weighted_compute_units": float(robust_step_wcu),
        "robust_step_compute_be": float(robust_step_compute_be),
        "budget_plan": plan,
        "estimated_total_wall_clock_sec": estimated_total_wall_clock_sec,
        "launch_env_subset": {
            key: launch_env[key]
            for key in (
                "RESUME",
                "DURATION_MIMG",
                "BATCH",
                "BATCH_GPU",
                "PRECOND",
                "FP16",
                "CIFAR_TRAIN_PERCENT",
                "CIFAR_TRAIN_SEED",
                "SEED",
                "TICK_KIMG",
                "SNAP_TICKS",
                "DUMP_TICKS",
                "EXTRA_TRAIN_ARGS",
            )
        },
        "launch_cmd": launch_cmd,
    }
    print(json.dumps(plan_payload, indent=2))
    if not args.launch:
        print("[DRY-RUN] Launch plan computed; no files were created.")
        return

    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(src_state, dst_state)
    shutil.copy2(src_snapshot, dst_snapshot)
    (run_dir / "cdro_budget_plan.json").write_text(json.dumps(plan_payload, indent=2), encoding="utf-8")

    subprocess.run(
        launch_cmd,
        cwd=str(ROOT_DIR),
        env=launch_env,
        check=True,
    )


if __name__ == "__main__":
    main()
