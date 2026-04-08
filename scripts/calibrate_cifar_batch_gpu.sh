#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

OUTROOT="${OUTROOT:-${ROOT_DIR}/training-runs/cifar10_batch_gpu_calibration}"
BATCH_GPU_CANDIDATES="${BATCH_GPU_CANDIDATES:-128 256 512}"

# Keep the paper-facing training setup fixed except for the per-GPU microbatch.
DURATION_MIMG="${DURATION_MIMG:-0.12}"
BATCH="${BATCH:-1024}"
LR="${LR:-1e-5}"
CIFAR_TRAIN_PERCENT="${CIFAR_TRAIN_PERCENT:-20}"
CIFAR_TRAIN_SEED="${CIFAR_TRAIN_SEED:-0}"
TICK_KIMG="${TICK_KIMG:-20}"
SNAP_TICKS="${SNAP_TICKS:-1000000}"
DUMP_TICKS="${DUMP_TICKS:-1000000}"
SEED="${SEED:-0}"

mkdir -p "${OUTROOT}"
SUMMARY_CSV="${OUTROOT}/summary.csv"
echo "batch_gpu,status,last_tick,last_kimg,last_sec_per_kimg,avg_sec_per_kimg,peak_gpu_mem_gb,run_dir" > "${SUMMARY_CSV}"

for bg in ${BATCH_GPU_CANDIDATES}; do
  RUN_GROUP="${OUTROOT}/bg${bg}"
  mkdir -p "${RUN_GROUP}"
  if (( BATCH % bg != 0 )); then
    echo "${bg},skipped,invalid_divisor,,,,," >> "${SUMMARY_CSV}"
    echo "[calib] skipping batch_gpu=${bg} because BATCH=${BATCH} is not divisible by batch_gpu"
    continue
  fi
  echo "[calib] starting batch_gpu=${bg}"
  set +e
  ENV_MODE="${ENV_MODE:-venv}" \
  VENV_DIR="${VENV_DIR:-${HOME}/.venvs/wild-diffusion-h100}" \
  INSTALL_DEPS="${INSTALL_DEPS:-0}" \
  OUTDIR="${RUN_GROUP}" \
  DURATION_MIMG="${DURATION_MIMG}" \
  BATCH="${BATCH}" \
  BATCH_GPU="${bg}" \
  LR="${LR}" \
  ARCH="ddpmpp" \
  PRECOND="wdroedm" \
  COND="0" \
  FP16="0" \
  WDRO_WARMUP_RATIO="1.0" \
  WDRO_P_ADV="0.0" \
  DEBUG_EVAL="0" \
  CIFAR_TRAIN_PERCENT="${CIFAR_TRAIN_PERCENT}" \
  CIFAR_TRAIN_SEED="${CIFAR_TRAIN_SEED}" \
  TICK_KIMG="${TICK_KIMG}" \
  SNAP_TICKS="${SNAP_TICKS}" \
  DUMP_TICKS="${DUMP_TICKS}" \
  DESC="calib-bg${bg}" \
  SEED="${SEED}" \
  ./scripts/setup_and_train_cifar10.sh
  status=$?
  set -e

  latest_run="$(find "${RUN_GROUP}" -maxdepth 1 -mindepth 1 -type d | sort | tail -n 1 || true)"
  if [[ -z "${latest_run}" ]]; then
    echo "${bg},failed,no_run_dir,,,,," >> "${SUMMARY_CSV}"
    echo "[calib] batch_gpu=${bg} failed before run dir creation"
    continue
  fi

  python3 - <<PY
import csv
import json
from pathlib import Path

bg = ${bg}
status = ${status}
run_dir = Path(${latest_run@Q})
stats_path = run_dir / "stats.jsonl"

out = {
    "batch_gpu": bg,
    "status": "ok" if status == 0 else "failed",
    "last_tick": "",
    "last_kimg": "",
    "last_sec_per_kimg": "",
    "avg_sec_per_kimg": "",
    "peak_gpu_mem_gb": "",
    "run_dir": str(run_dir),
}

rows = []
if stats_path.exists():
    for line in stats_path.read_text(errors="ignore").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        tick = obj.get("Progress/tick", {}).get("mean")
        if tick is None or float(tick) < 1:
            continue
        rows.append(obj)

if rows:
    sec_vals = [float(r["Timing/sec_per_kimg"]["mean"]) for r in rows]
    last = rows[-1]
    out["last_tick"] = int(last["Progress/tick"]["mean"])
    out["last_kimg"] = float(last["Progress/kimg"]["mean"])
    out["last_sec_per_kimg"] = float(last["Timing/sec_per_kimg"]["mean"])
    out["avg_sec_per_kimg"] = sum(sec_vals) / len(sec_vals)
    out["peak_gpu_mem_gb"] = float(last["Resources/peak_gpu_mem_gb"]["mean"])

with open(${SUMMARY_CSV@Q}, "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        out["batch_gpu"],
        out["status"],
        out["last_tick"],
        out["last_kimg"],
        out["last_sec_per_kimg"],
        out["avg_sec_per_kimg"],
        out["peak_gpu_mem_gb"],
        out["run_dir"],
    ])

print("[calib]", out)
PY
done

echo "[calib] summary written to ${SUMMARY_CSV}"
