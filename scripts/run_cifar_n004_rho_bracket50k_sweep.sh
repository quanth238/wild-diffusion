#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

VENV_DIR="${VENV_DIR:-/home/bachlc/.venvs/wild-diffusion-h100}"
TARGET_KIMG="${TARGET_KIMG:-50000}"
RHO_VALUES="${RHO_VALUES:-1e-12,1e-10,1e-8}"
CAMPAIGN_TAG="${CAMPAIGN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
TRAIN_OUTROOT_BASE="${TRAIN_OUTROOT_BASE:-/mnt/data/bachlc/GM-CDRO/training-runs/paper-cifar10-cdro-fp16-n004-rho-bracket50k_${CAMPAIGN_TAG}}"
EVAL_SWEEP_ROOT_BASE="${EVAL_SWEEP_ROOT_BASE:-/mnt/data/bachlc/GM-CDRO/training-runs/fid-sweeps/cdro_n004_rho_bracket50k_${CAMPAIGN_TAG}}"
LOG_DIR="${LOG_DIR:-/mnt/data/bachlc/GM-CDRO/training-runs/fid-sweeps/logs}"
RUN_LOG="${RUN_LOG:-${LOG_DIR}/cdro_n004_rho_bracket50k_${CAMPAIGN_TAG}.log}"
PYTORCH_FID_REF="${PYTORCH_FID_REF:-${ROOT_DIR}/training-runs/fid-sweeps/cifar10_baseline_vs_wdro_coarse_20260414/pytorch_fid_cifar10_train_ref_stats.npz}"
TRAIN_LR="${TRAIN_LR:-1e-5}"
TRAIN_TICK_KIMG="${TRAIN_TICK_KIMG:-1000}"
TRAIN_SNAP_TICKS="${TRAIN_SNAP_TICKS:-10}"
TRAIN_DUMP_TICKS="${TRAIN_DUMP_TICKS:-10}"

mkdir -p "${LOG_DIR}" "${TRAIN_OUTROOT_BASE}" "${EVAL_SWEEP_ROOT_BASE}"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "[ERROR] Missing venv: ${VENV_DIR}"
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

rho_tag() {
  python - "$1" <<'PY'
import sys

value = float(sys.argv[1])
text = format(value, ".6g")
print("rho" + text.replace("-", "m").replace("+", "").replace(".", "p"))
PY
}

latest_run_dir() {
  local outroot="$1"
  find -L "${outroot}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1
}

latest_training_state() {
  local run_dir="$1"
  find -L "${run_dir}" -mindepth 1 -maxdepth 1 -type f -name 'training-state-*.pt' | sort | tail -n 1
}

target_wcu_for_kimg() {
  python - "${TARGET_KIMG}" <<'PY'
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
    raise SystemExit(f"TARGET_KIMG must exceed warmup kimg={warmup['warmup_kimg']}, got {target_kimg}")

calibration = calibration_from_path(DEFAULT_CALIBRATION_JSON)
step_wcu = cdro_robust_step_weighted_compute_units(
    calibration=calibration,
    n_steps_path=4,
    attack_num_steps=1,
    outer_attack_weight=1.0,
    outer_clean_weight=0.0,
    total_budget_rho=1e-12,
)
step_compute_be = cdro_robust_step_compute_be(
    n_steps_path=4,
    attack_num_steps=1,
    outer_attack_weight=1.0,
    outer_clean_weight=0.0,
    total_budget_rho=1e-12,
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
    raise SystemExit(f"Internal target WCU calculation produced total_kimg={plan['total_kimg_int']} not {target_kimg}: {plan}")
print(f"{target_wcu:.12f}")
PY
}

resume_existing_run() {
  local run_dir="$1"
  local rho="$2"
  local resume_state

  resume_state="$(latest_training_state "${run_dir}")"
  if [[ -z "${resume_state}" || ! -f "${resume_state}" ]]; then
    echo "[ERROR] Could not find a resume state in ${run_dir}"
    exit 1
  fi

  echo "[INFO] Resuming N=4 rho=${rho} from ${resume_state##*/} to ${TARGET_KIMG} kimg"
  RESUME="${resume_state}" \
  DURATION_MIMG="$(python -c "print(${TARGET_KIMG} / 1000.0)")" \
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
  EXTRA_TRAIN_ARGS="--cdro-n-steps-path=4 --cdro-step-size=0.02 --cdro-total-budget-rho=${rho} --cdro-time-horizon=1.0 --cdro-sigma-min=0.002 --cdro-sigma-max=80.0 --cdro-edm-ladder-mode=stochastic_stratified_quantile --cdro-per-example-sigma-ladders=True --attack-num-steps=1 --outer-attack-weight=1.0 --outer-clean-weight=0.0" \
    bash "${ROOT_DIR}/scripts/setup_and_train_cifar10.sh"
}

launch_and_eval_rho() {
  local rho="$1"
  local target_wcu="$2"
  local rho_dir_tag
  local train_outroot
  local eval_outdir
  local run_dir
  local target_state
  local manifest_csv

  rho_dir_tag="$(rho_tag "${rho}")"
  train_outroot="${TRAIN_OUTROOT_BASE}/${rho_dir_tag}"
  eval_outdir="${EVAL_SWEEP_ROOT_BASE}/${rho_dir_tag}"
  manifest_csv="${eval_outdir}/cifar10_cdro_budget_manifest.csv"

  mkdir -p "${train_outroot}" "${eval_outdir}"

  run_dir="$(latest_run_dir "${train_outroot}")"
  if [[ -z "${run_dir}" || ! -d "${run_dir}" ]]; then
    echo "[INFO] Launching N=4 rho=${rho} to ${TARGET_KIMG} kimg (${rho_dir_tag})"
    python "${ROOT_DIR}/scripts/launch_cifar20_cdro_budgeted.py" \
      --outdir-root "${train_outroot}" \
      --target-wcu "${target_wcu}" \
      --batch-size 1024 \
      --batch-gpu 1024 \
      --lr "${TRAIN_LR}" \
      --cdro-n-steps-path 4 \
      --cdro-total-budget-rho "${rho}" \
      --outer-attack-weight 1.0 \
      --outer-clean-weight 0.0 \
      --tick-kimg "${TRAIN_TICK_KIMG}" \
      --snap-ticks "${TRAIN_SNAP_TICKS}" \
      --dump-ticks "${TRAIN_DUMP_TICKS}" \
      --env-mode venv \
      --venv-dir "${VENV_DIR}" \
      --install-deps 0 \
      --launch
    run_dir="$(latest_run_dir "${train_outroot}")"
  fi
  if [[ -z "${run_dir}" || ! -d "${run_dir}" ]]; then
    echo "[ERROR] Could not resolve run dir for rho=${rho}"
    exit 1
  fi

  target_state="$(printf "%s/training-state-%06d.pt" "${run_dir}" "${TARGET_KIMG}")"
  if [[ ! -f "${target_state}" ]]; then
    resume_existing_run "${run_dir}" "${rho}"
  else
    echo "[SKIP] Found ${target_state}; skipping training for rho=${rho}"
  fi

  echo "[INFO] Completed training for rho=${rho}; run_dir=${run_dir}"
  echo "[INFO] Building manifest at kimg=${TARGET_KIMG}"
  python "${ROOT_DIR}/scripts/build_cifar_cdro_fid_manifest.py" \
    --cdro-run-dir "${run_dir}" \
    --outdir "${eval_outdir}" \
    --kimg "${TARGET_KIMG}" \
    --batch-size 1024 \
    --seed 0 \
    --train-percent-label "20%" \
    --ref-path "${PYTORCH_FID_REF}"

  echo "[INFO] Running pytorch-fid eval for rho=${rho}"
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

  echo "[INFO] Running clean EDM probe for rho=${rho}"
  python "${ROOT_DIR}/scripts/run_cifar_loss_probe_manifest.py" \
    --manifest-csv "${manifest_csv}" \
    --only-pending \
    --device cuda \
    --batch-size 512 \
    --max-images 10000 \
    --seed 0

  echo "[INFO] Final manifest row for rho=${rho}:"
  tail -n 1 "${manifest_csv}"
}

write_summary_csv() {
  python - "${EVAL_SWEEP_ROOT_BASE}" "${RHO_VALUES}" <<'PY'
import csv
import sys
from pathlib import Path

root = Path(sys.argv[1])
rho_values = [item.strip() for item in sys.argv[2].split(",") if item.strip()]
rows = []
for rho in rho_values:
    tag = "rho" + format(float(rho), ".6g").replace("-", "m").replace("+", "").replace(".", "p")
    manifest = root / tag / "cifar10_cdro_budget_manifest.csv"
    if not manifest.is_file():
        continue
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    if not manifest_rows:
        continue
    row = dict(manifest_rows[-1])
    row["rho"] = rho
    row["rho_tag"] = tag
    rows.append(row)
if rows:
    preferred = ["rho", "rho_tag", "step", "fid", "clean_loss", "images_shown_m", "train_wall_clock_sec", "weighted_compute_units", "network_pkl", "row_origin"]
    fieldnames = preferred + sorted({key for row in rows for key in row if key not in preferred})
    out = root / "rho_bracket50k_results.csv"
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OK] Wrote summary CSV: {out}")
else:
    print("[WARN] No completed manifest rows found for summary CSV.")
PY
}

target_wcu="$(target_wcu_for_kimg)"

echo "[INFO] Starting N=4 rho bracket sweep on $(hostname)"
echo "[INFO] Campaign tag: ${CAMPAIGN_TAG}"
echo "[INFO] Target kimg: ${TARGET_KIMG}"
echo "[INFO] Target WCU for ${TARGET_KIMG} kimg: ${target_wcu}"
echo "[INFO] RHO_VALUES: ${RHO_VALUES}"
echo "[INFO] Train outroot base: ${TRAIN_OUTROOT_BASE}"
echo "[INFO] Eval sweep root base: ${EVAL_SWEEP_ROOT_BASE}"
echo "[INFO] Run log: ${RUN_LOG}"
echo "[INFO] TRAIN_TICK_KIMG: ${TRAIN_TICK_KIMG}"
echo "[INFO] TRAIN_SNAP_TICKS: ${TRAIN_SNAP_TICKS}"
echo "[INFO] TRAIN_DUMP_TICKS: ${TRAIN_DUMP_TICKS}"
date -u
nvidia-smi || true

IFS=',' read -r -a rho_list <<< "${RHO_VALUES}"
for raw_rho in "${rho_list[@]}"; do
  rho="$(echo "${raw_rho}" | tr -d '[:space:]')"
  if [[ -z "${rho}" ]]; then
    continue
  fi
  launch_and_eval_rho "${rho}" "${target_wcu}"
  write_summary_csv
done

write_summary_csv
echo "[OK] N=4 rho bracket 50K sweep complete."
date -u
