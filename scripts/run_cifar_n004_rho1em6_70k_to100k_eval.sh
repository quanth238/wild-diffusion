#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

VENV_DIR="${VENV_DIR:-/home/bachlc/.venvs/wild-diffusion-h100}"
SOURCE_RUN_DIR="${SOURCE_RUN_DIR:-/mnt/data/bachlc/GM-CDRO/training-runs/paper-cifar10-cdro-fp16-n004-rho1em6-edm200mimg_20260422T150118Z/rho1em06/00000-cifar10-32x32-train20pct-seed0-uncond-ddpmpp-cdroedm-gpus1-batch1024-fp16-paper-cifar10-uncond-ddpmpp-cdro-20pct-n004-rho0p0-i1-aw1p00-cw0p00-pel1-bg1024-resume040000-wcu460371}"
SOURCE_RESUME_KIMG="${SOURCE_RESUME_KIMG:-70000}"
TARGET_KIMG_LIST="${TARGET_KIMG_LIST:-80000,90000,100000}"
CAMPAIGN_TAG="${CAMPAIGN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
TRAIN_OUTROOT="${TRAIN_OUTROOT:-${ROOT_DIR}/training-runs/paper-cifar10-cdro-fp16-n004-rho1em6-continue100k_${CAMPAIGN_TAG}}"
EVAL_SWEEP_ROOT="${EVAL_SWEEP_ROOT:-${ROOT_DIR}/training-runs/fid-sweeps/cdro_n004_rho1em6_continue100k_${CAMPAIGN_TAG}}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/training-runs/fid-sweeps/logs}"
RUN_LOG="${RUN_LOG:-${LOG_DIR}/cdro_n004_rho1em6_continue100k_${CAMPAIGN_TAG}.log}"
PYTORCH_FID_REF="${PYTORCH_FID_REF:-${ROOT_DIR}/training-runs/fid-sweeps/cifar10_baseline_vs_wdro_coarse_20260414/pytorch_fid_cifar10_train_ref_stats.npz}"
TRAIN_LR="${TRAIN_LR:-1e-5}"
TRAIN_TICK_KIMG="${TRAIN_TICK_KIMG:-1000}"
TRAIN_SNAP_TICKS="${TRAIN_SNAP_TICKS:-10}"
TRAIN_DUMP_TICKS="${TRAIN_DUMP_TICKS:-10}"

mkdir -p "${LOG_DIR}" "${TRAIN_OUTROOT}" "${EVAL_SWEEP_ROOT}"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "[ERROR] Missing venv: ${VENV_DIR}"
  exit 1
fi
if [[ ! -d "${SOURCE_RUN_DIR}" ]]; then
  echo "[ERROR] Missing source run dir: ${SOURCE_RUN_DIR}"
  exit 1
fi
if [[ ! -f "${SOURCE_RUN_DIR}/network-snapshot-$(printf '%06d' "${SOURCE_RESUME_KIMG}").pkl" ]]; then
  echo "[ERROR] Missing source snapshot for ${SOURCE_RESUME_KIMG} kimg"
  exit 1
fi
if [[ ! -f "${SOURCE_RUN_DIR}/training-state-$(printf '%06d' "${SOURCE_RESUME_KIMG}").pt" ]]; then
  echo "[ERROR] Missing source training state for ${SOURCE_RESUME_KIMG} kimg"
  exit 1
fi
if [[ ! -f "${PYTORCH_FID_REF}" ]]; then
  echo "[ERROR] Missing pytorch-fid reference stats: ${PYTORCH_FID_REF}"
  exit 1
fi

for cmd_name in python bash find tee sort jq; do
  if ! command -v "${cmd_name}" >/dev/null 2>&1; then
    echo "[ERROR] Missing command: ${cmd_name}"
    exit 1
  fi
done

if [[ -r /proc/self/fd ]]; then
  exec > >(tee -a "${RUN_LOG}") 2>&1
else
  exec >>"${RUN_LOG}" 2>&1
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ ! -r /proc/self/maps || ! -r /proc/cpuinfo || ! -r /proc/sys/vm/mmap_min_addr ]]; then
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/scripts/ensure_procfs_compat.sh"
fi

target_wcu_for_kimg() {
  python - "$1" <<'PY'
import math
import sys

from scripts.cifar_cdro_budget_utils import (
    DEFAULT_CALIBRATION_JSON,
    DEFAULT_WDRO_SUMMARY_JSON,
    build_budget_plan,
    calibration_from_path,
    cdro_robust_step_compute_be,
    cdro_robust_step_weighted_compute_units,
    load_warmup_summary,
)

target_kimg = int(sys.argv[1])
batch_size = 1024
warmup = load_warmup_summary(DEFAULT_WDRO_SUMMARY_JSON)
if target_kimg <= int(round(float(warmup["warmup_kimg"]))):
    raise SystemExit(f"target kimg must exceed warmup kimg={warmup['warmup_kimg']}, got {target_kimg}")

calibration = calibration_from_path(DEFAULT_CALIBRATION_JSON)
step_wcu = cdro_robust_step_weighted_compute_units(
    calibration=calibration,
    n_steps_path=4,
    attack_num_steps=1,
    outer_attack_weight=1.0,
    outer_clean_weight=0.0,
    total_budget_rho=1e-6,
)
step_compute_be = cdro_robust_step_compute_be(
    n_steps_path=4,
    attack_num_steps=1,
    outer_attack_weight=1.0,
    outer_clean_weight=0.0,
    total_budget_rho=1e-6,
)
warmup_nimg = int(round(float(warmup["warmup_kimg"]) * 1000.0))
target_nimg = int(target_kimg) * 1000
robust_steps = int(math.ceil(max(target_nimg - warmup_nimg, 0) / float(batch_size)))
target_wcu = float(warmup["warmup_weighted_compute_units"]) + float(robust_steps) * float(step_wcu)
plan = build_budget_plan(
    target_wcu=target_wcu,
    warmup_kimg=warmup["warmup_kimg"],
    warmup_wcu=warmup["warmup_weighted_compute_units"],
    warmup_compute_be=warmup["warmup_compute_be"],
    batch_size=batch_size,
    robust_step_wcu=step_wcu,
    robust_step_compute_be=step_compute_be,
)
if int(plan["total_kimg_int"]) != target_kimg:
    raise SystemExit(f"target WCU calculation produced total_kimg={plan['total_kimg_int']} not {target_kimg}: {plan}")
print(f"{target_wcu:.12f}")
PY
}

latest_run_dir() {
  find -L "${TRAIN_OUTROOT}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1
}

latest_training_state() {
  local run_dir="$1"
  find -L "${run_dir}" -mindepth 1 -maxdepth 1 -type f -name 'training-state-*.pt' | sort | tail -n 1
}

resume_existing_run() {
  local run_dir="$1"
  local target_kimg="$2"
  local resume_state

  resume_state="$(latest_training_state "${run_dir}")"
  if [[ -z "${resume_state}" || ! -f "${resume_state}" ]]; then
    echo "[ERROR] Could not find a resume state in ${run_dir}"
    exit 1
  fi

  echo "[INFO] Resuming rho=1e-6 N=4 from ${resume_state##*/} to ${target_kimg} kimg"
  RESUME="${resume_state}" \
  DURATION_MIMG="$(python -c "print(${target_kimg} / 1000.0)")" \
  BATCH=1024 \
  BATCH_GPU=1024 \
  LR="${TRAIN_LR}" \
  WORKERS=16 \
  ARCH=ddpmpp \
  PRECOND=cdroedm \
  COND=0 \
  FP16=1 \
  AUGMENT=0.12 \
  DEBUG_EVAL=0 \
  DEBUG_ADV_VISUAL=0 \
  CIFAR_TRAIN_PERCENT=20 \
  CIFAR_TRAIN_SEED=0 \
  TICK_KIMG="${TRAIN_TICK_KIMG}" \
  SNAP_TICKS="${TRAIN_SNAP_TICKS}" \
  DUMP_TICKS="${TRAIN_DUMP_TICKS}" \
  SEED=0 \
  ENV_MODE=venv \
  INSTALL_DEPS=0 \
  VENV_DIR="${VENV_DIR}" \
  EXTRA_TRAIN_ARGS="--cdro-n-steps-path=4 --cdro-step-size=0.02 --cdro-total-budget-rho=1e-6 --cdro-time-horizon=1.0 --cdro-sigma-min=0.002 --cdro-sigma-max=80.0 --cdro-edm-ladder-mode=stochastic_stratified_quantile --cdro-per-example-sigma-ladders=True --attack-num-steps=1 --outer-attack-weight=1.0 --outer-clean-weight=0.0" \
    bash "${ROOT_DIR}/scripts/setup_and_train_cifar10.sh"
}

ensure_run_to_target() {
  local target_kimg="$1"
  local run_dir
  local target_state
  run_dir="$(latest_run_dir)"

  if [[ -z "${run_dir}" || ! -d "${run_dir}" ]]; then
    local target_wcu
    target_wcu="$(target_wcu_for_kimg "${target_kimg}")"
    echo "[INFO] Launching local rho=1e-6 N=4 continuation from ${SOURCE_RESUME_KIMG} to ${target_kimg} kimg"
    echo "[INFO] Target WCU for ${target_kimg} kimg: ${target_wcu}"
    python "${ROOT_DIR}/scripts/launch_cifar20_cdro_budgeted.py" \
      --outdir-root "${TRAIN_OUTROOT}" \
      --baseline-run-dir "${SOURCE_RUN_DIR}" \
      --baseline-resume-kimg "${SOURCE_RESUME_KIMG}" \
      --target-wcu "${target_wcu}" \
      --batch-size 1024 \
      --batch-gpu 1024 \
      --lr "${TRAIN_LR}" \
      --cdro-n-steps-path 4 \
      --cdro-total-budget-rho 1e-6 \
      --outer-attack-weight 1.0 \
      --outer-clean-weight 0.0 \
      --tick-kimg "${TRAIN_TICK_KIMG}" \
      --snap-ticks "${TRAIN_SNAP_TICKS}" \
      --dump-ticks "${TRAIN_DUMP_TICKS}" \
      --env-mode venv \
      --venv-dir "${VENV_DIR}" \
      --install-deps 0 \
      --launch
    run_dir="$(latest_run_dir)"
  fi

  if [[ -z "${run_dir}" || ! -d "${run_dir}" ]]; then
    echo "[ERROR] Could not resolve continuation run dir"
    exit 1
  fi

  target_state="$(printf "%s/training-state-%06d.pt" "${run_dir}" "${target_kimg}")"
  if [[ -f "${target_state}" ]]; then
    echo "[SKIP] Found ${target_state}; skipping training to ${target_kimg}"
  else
    resume_existing_run "${run_dir}" "${target_kimg}"
  fi
}

completed_kimg_csv() {
  local upto_target="$1"
  python - "${TARGET_KIMG_LIST}" "${upto_target}" <<'PY'
import sys

values = [int(item.strip()) for item in sys.argv[1].split(",") if item.strip()]
upto = int(sys.argv[2])
print(",".join(str(value) for value in values if value <= upto))
PY
}

build_eval_manifest() {
  local kimg_csv="$1"
  local run_dir="$2"
  python "${ROOT_DIR}/scripts/build_cifar_cdro_fid_manifest.py" \
    --cdro-run-dir "${run_dir}" \
    --outdir "${EVAL_SWEEP_ROOT}" \
    --manifest-name "cifar10_cdro_budget_manifest_eval80_90_100.csv" \
    --summary-name "cifar10_cdro_budget_manifest_eval80_90_100_summary.json" \
    --kimg "${kimg_csv}" \
    --batch-size 1024 \
    --seed 0 \
    --train-percent-label "20%" \
    --ref-path "${PYTORCH_FID_REF}"
}

run_pending_evals() {
  local manifest_csv="$1"
  python "${ROOT_DIR}/scripts/run_cifar_fid_manifest.py" \
    --manifest-csv "${manifest_csv}" \
    --env-mode venv \
    --venv-dir "${VENV_DIR}" \
    --install-deps 0 \
    --prepare-dataset 0 \
    --fid-backend pytorch_fid \
    --ref-mode path \
    --ref-path "${PYTORCH_FID_REF}" \
    --only-pending \
    --num-images 50000 \
    --gen-batch 128 \
    --fid-batch 64 \
    --gen-steps 18 \
    --nproc-per-node 1

  python "${ROOT_DIR}/scripts/run_cifar_loss_probe_manifest.py" \
    --manifest-csv "${manifest_csv}" \
    --only-pending \
    --device cuda \
    --batch-size 512 \
    --max-images 10000 \
    --seed 0
}

write_summary_csv() {
  local manifest_csv="$1"
  python - "${manifest_csv}" "${EVAL_SWEEP_ROOT}/rho1em6_eval80_90_100_results.csv" <<'PY'
import csv
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
out = Path(sys.argv[2])
if not manifest.is_file():
    raise SystemExit(f"missing manifest: {manifest}")
with manifest.open("r", encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))
preferred = [
    "step",
    "images_shown_m",
    "fid",
    "loss_probe_clean",
    "loss_final",
    "total_train_pflops",
    "train_wall_clock_sec",
    "weighted_compute_units",
    "network_pkl",
]
fieldnames = preferred + [key for key in rows[0].keys() if key not in preferred] if rows else preferred
with out.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
print(f"[OK] Wrote summary CSV: {out}")
PY
}

echo "[INFO] Starting rho=1e-6 N=4 staged continuation on $(hostname)"
echo "[INFO] Campaign tag: ${CAMPAIGN_TAG}"
echo "[INFO] Source run dir: ${SOURCE_RUN_DIR}"
echo "[INFO] Source resume kimg: ${SOURCE_RESUME_KIMG}"
echo "[INFO] Targets: ${TARGET_KIMG_LIST}"
echo "[INFO] Train outroot: ${TRAIN_OUTROOT}"
echo "[INFO] Eval sweep root: ${EVAL_SWEEP_ROOT}"
echo "[INFO] Run log: ${RUN_LOG}"
echo "[INFO] TRAIN_TICK_KIMG: ${TRAIN_TICK_KIMG}"
echo "[INFO] TRAIN_SNAP_TICKS: ${TRAIN_SNAP_TICKS}"
echo "[INFO] TRAIN_DUMP_TICKS: ${TRAIN_DUMP_TICKS}"
date -u
nvidia-smi || true

IFS=',' read -r -a targets <<< "${TARGET_KIMG_LIST}"
for raw_target in "${targets[@]}"; do
  target_kimg="$(echo "${raw_target}" | tr -d '[:space:]')"
  if [[ -z "${target_kimg}" ]]; then
    continue
  fi
  ensure_run_to_target "${target_kimg}"
  run_dir="$(latest_run_dir)"
  kimg_csv="$(completed_kimg_csv "${target_kimg}")"
  manifest_csv="${EVAL_SWEEP_ROOT}/cifar10_cdro_budget_manifest_eval80_90_100.csv"
  echo "[INFO] Building/evaluating manifest for completed kimg: ${kimg_csv}"
  build_eval_manifest "${kimg_csv}" "${run_dir}"
  run_pending_evals "${manifest_csv}"
  write_summary_csv "${manifest_csv}"
done

echo "[OK] rho=1e-6 N=4 staged continuation complete."
date -u
