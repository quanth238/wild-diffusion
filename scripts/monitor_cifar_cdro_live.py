#!/usr/bin/env python3
import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path


DEFAULT_RUN_DIR = (
    "/home/bachlc/GM-CDRO/training-runs/paper-cifar10-cdro-fp16/"
    "00000-cifar10-32x32-train20pct-seed0-uncond-ddpmpp-cdroedm-gpus1-batch1024-fp16-"
    "paper-cifar10-uncond-ddpmpp-cdro-20pct-n032-rho32p0-i1-aw0p30-cw0p00-bg1024-"
    "resume040000-wcu537488"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monitor a CIFAR CDRO training run between sparse tick writes."
    )
    parser.add_argument("--run-dir", type=str, default=DEFAULT_RUN_DIR)
    parser.add_argument("--job-id", type=str, default="10076")
    parser.add_argument("--interval", type=float, default=15.0)
    parser.add_argument("--watch", action="store_true", default=False)
    return parser.parse_args()


def load_last_stats(stats_path: Path) -> dict:
    lines = stats_path.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        raise RuntimeError(f"No stats rows found in {stats_path}")
    return json.loads(lines[-1])


def mean_field(payload: dict, key: str) -> float:
    value = payload[key]
    if isinstance(value, dict):
        value = value["mean"]
    return float(value)


def gpu_status(job_id: str) -> str:
    cmd = [
        "srun",
        f"--jobid={job_id}",
        "--overlap",
        "--quiet",
        "bash",
        "-lc",
        "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw "
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "gpu=unavailable"
    if not out:
        return "gpu=unavailable"
    parts = [chunk.strip() for chunk in out.split(",")]
    if len(parts) < 5:
        return f"gpu_raw={out}"
    util, mem_used, mem_total, temp_c, power_w = parts[:5]
    return f"gpu={util}% mem={mem_used}/{mem_total}MiB temp={temp_c}C power={power_w}W"


def format_status(run_dir: Path, job_id: str) -> str:
    stats_path = run_dir / "stats.jsonl"
    opts_path = run_dir / "training_options.json"
    stats_payload = load_last_stats(stats_path)
    opts = json.loads(opts_path.read_text(encoding="utf-8"))

    last_tick = int(round(mean_field(stats_payload, "Progress/tick")))
    last_kimg = mean_field(stats_payload, "Progress/kimg")
    total_kimg = float(opts["total_kimg"])
    sec_per_kimg = mean_field(stats_payload, "Timing/sec_per_kimg")
    total_sec = mean_field(stats_payload, "Timing/total_sec")
    stats_age_sec = max(time.time() - os.path.getmtime(stats_path), 0.0)
    est_kimg = min(last_kimg + stats_age_sec / max(sec_per_kimg, 1e-8), total_kimg)
    kimg_per_tick = float(opts["kimg_per_tick"])
    next_tick_kimg = min(last_kimg + kimg_per_tick, total_kimg)
    eta_next_tick_min = max((next_tick_kimg - est_kimg) * sec_per_kimg / 60.0, 0.0)
    eta_finish_hr = max((total_kimg - est_kimg) * sec_per_kimg / 3600.0, 0.0)
    pct = 100.0 * est_kimg / max(total_kimg, 1e-8)
    gpu = gpu_status(job_id)
    return (
        f"{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())} "
        f"tick={last_tick} last_kimg={last_kimg:.3f} est_kimg={est_kimg:.3f}/{total_kimg:.0f} ({pct:.2f}%) "
        f"sec_per_kimg={sec_per_kimg:.2f} total_time={total_sec/3600.0:.2f}h "
        f"next_tick={next_tick_kimg:.3f} eta_next_tick={eta_next_tick_min:.1f}m "
        f"eta_finish={eta_finish_hr:.2f}h stats_age={stats_age_sec:.0f}s {gpu}"
    )


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    if args.watch:
        while True:
            print(format_status(run_dir, args.job_id), flush=True)
            time.sleep(max(float(args.interval), 1.0))
    else:
        print(format_status(run_dir, args.job_id))


if __name__ == "__main__":
    main()
