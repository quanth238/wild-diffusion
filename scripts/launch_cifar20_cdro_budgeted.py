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
    advedm_robust_step_compute_be,
    advedm_robust_step_weighted_compute_units,
    build_budget_plan,
    calibration_from_path,
    load_warmup_summary,
    resolve_budget_target,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare and optionally launch a CIFAR-10 20% CDRO-style image run that "
            "matches the existing Baseline/WDRO comparison by weighted compute budget."
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
    parser.add_argument("--target-mode", type=str, choices=["wdro_best", "wdro_final", "midpoint", "baseline_best"], default="baseline_best")
    parser.add_argument("--target-wcu", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--batch-gpu", type=int, default=512)
    parser.add_argument("--adv-steps", type=int, default=1)
    parser.add_argument("--adv-step-size", type=float, default=0.02)
    parser.add_argument("--adv-eps", type=float, default=None)
    parser.add_argument("--adv-mix", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--augment", type=float, default=0.12)
    parser.add_argument("--fp16", type=int, choices=[0, 1], default=1)
    parser.add_argument("--arch", type=str, choices=["ddpmpp", "ncsnpp", "adm"], default="ddpmpp")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cifar-train-percent", type=int, default=20)
    parser.add_argument("--cifar-train-seed", type=int, default=0)
    parser.add_argument("--tick-kimg", type=int, default=50)
    parser.add_argument("--snap-mimg", type=float, default=1.0)
    parser.add_argument("--dump-mimg", type=float, default=10.0)
    parser.add_argument("--env-mode", type=str, default="venv")
    parser.add_argument("--venv-dir", type=str, default="/home/bachlc/.venvs/wild-diffusion-h100")
    parser.add_argument("--install-deps", type=str, default="0")
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
    return (
        f"cifar10-32x32-train{int(args.cifar_train_percent)}pct-seed{int(args.seed)}-"
        f"uncond-{args.arch}-advedm-gpus1-batch{int(args.batch_size)}-"
        f"{'fp16' if int(args.fp16) else 'fp32'}-paper-cifar10-uncond-{args.arch}-cdro-"
        f"{int(args.cifar_train_percent)}pct-adv{int(args.adv_steps)}-mix{float(args.adv_mix):.2f}-"
        f"bg{int(args.batch_gpu)}-resume{int(args.baseline_resume_kimg):06d}-wcu{wcu_tag}"
    ).replace(".", "p")


def build_launch_env(args: argparse.Namespace, *, resume_state: Path, total_kimg: float) -> dict:
    extra_args = [
        f"--adv-steps={int(args.adv_steps)}",
        f"--adv-step-size={float(args.adv_step_size)}",
        f"--adv-mix={float(args.adv_mix)}",
    ]
    if args.adv_eps is not None:
        extra_args.append(f"--adv-eps={float(args.adv_eps)}")
    env = os.environ.copy()
    env.update(
        {
            "PYTORCH_CUDA_ALLOC_CONF": env.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
            "ENV_MODE": str(args.env_mode),
            "VENV_DIR": str(args.venv_dir),
            "INSTALL_DEPS": str(args.install_deps),
            "RESUME": str(resume_state),
            "DURATION_MIMG": f"{float(total_kimg) / 1000.0:.6f}",
            "BATCH": str(int(args.batch_size)),
            "BATCH_GPU": str(int(args.batch_gpu)),
            "LR": str(float(args.lr)),
            "WORKERS": str(int(args.workers)),
            "ARCH": str(args.arch),
            "PRECOND": "advedm",
            "COND": "0",
            "FP16": str(int(args.fp16)),
            "AUGMENT": str(float(args.augment)),
            "WDRO_WARMUP_RATIO": "1.0",
            "WDRO_P_ADV": "0.0",
            "DEBUG_EVAL": "0",
            "DEBUG_ADV_VISUAL": "0",
            "CIFAR_TRAIN_PERCENT": str(int(args.cifar_train_percent)),
            "CIFAR_TRAIN_SEED": str(int(args.cifar_train_seed)),
            "TICK_KIMG": str(int(args.tick_kimg)),
            "SNAP_MIMG": str(float(args.snap_mimg)),
            "DUMP_MIMG": str(float(args.dump_mimg)),
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
    robust_step_wcu = advedm_robust_step_weighted_compute_units(
        calibration=calibration,
        adv_steps=args.adv_steps,
    )
    robust_step_compute_be = advedm_robust_step_compute_be(adv_steps=args.adv_steps)
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

    launch_env = build_launch_env(args, resume_state=dst_state, total_kimg=plan["total_kimg_float"])
    launch_cmd = ["bash", str(ROOT_DIR / "scripts" / "setup_and_train_cifar10.sh")]

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
        "target_wcu": float(target_wcu),
        "target_envelope": budget_target,
        "warmup_summary": warmup_summary,
        "adv_config": {
            "adv_steps": int(args.adv_steps),
            "adv_step_size": float(args.adv_step_size),
            "adv_eps": None if args.adv_eps is None else float(args.adv_eps),
            "adv_mix": float(args.adv_mix),
        },
        "robust_step_weighted_compute_units": float(robust_step_wcu),
        "robust_step_compute_be": float(robust_step_compute_be),
        "budget_plan": plan,
        "launch_env_subset": {
            key: launch_env[key]
            for key in (
                "RESUME",
                "DURATION_MIMG",
                "BATCH",
                "BATCH_GPU",
                "PRECOND",
                "FP16",
                "WDRO_WARMUP_RATIO",
                "CIFAR_TRAIN_PERCENT",
                "CIFAR_TRAIN_SEED",
                "SEED",
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
