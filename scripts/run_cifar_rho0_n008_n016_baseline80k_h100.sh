#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

VENV_DIR="${VENV_DIR:-/home/bachlc/.venvs/wild-diffusion-h100}"
TARGET_WCU="${TARGET_WCU:-184148.22639230292}"
CAMPAIGN_TAG="${CAMPAIGN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
TRAIN_OUTROOT="${TRAIN_OUTROOT:-${ROOT_DIR}/training-runs/paper-cifar10-cdro-fp16-rho0-baseline80k_${CAMPAIGN_TAG}}"
EVAL_SWEEP_ROOT="${EVAL_SWEEP_ROOT:-${ROOT_DIR}/training-runs/fid-sweeps/cdro_rho0_n008_n016_baseline80k_${CAMPAIGN_TAG}}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/training-runs/fid-sweeps/logs}"
PYTORCH_FID_REF="${PYTORCH_FID_REF:-${ROOT_DIR}/training-runs/fid-sweeps/cifar10_baseline_vs_wdro_coarse_20260414/pytorch_fid_cifar10_train_ref_stats.npz}"
RUN_LOG="${RUN_LOG:-${LOG_DIR}/cdro_rho0_n008_n016_baseline80k_${CAMPAIGN_TAG}.log}"

mkdir -p "${LOG_DIR}" "${TRAIN_OUTROOT}" "${EVAL_SWEEP_ROOT}"

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

exec > >(tee -a "${RUN_LOG}") 2>&1

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

find_run_dir() {
  local n_steps="$1"
  find "${TRAIN_OUTROOT}" -mindepth 1 -maxdepth 1 -type d \
    -name "*-n$(printf '%03d' "${n_steps}")-rho0p0-*" | sort | tail -n 1
}

launch_and_eval() {
  local n_steps="$1"
  local run_dir
  local total_kimg
  local eval_outdir="${EVAL_SWEEP_ROOT}/n$(printf '%03d' "${n_steps}")"
  local manifest_csv="${eval_outdir}/cifar10_cdro_budget_manifest.csv"

  mkdir -p "${eval_outdir}"

  echo "[INFO] Launching rho=0, N=${n_steps} to target WCU ${TARGET_WCU}"
  python "${ROOT_DIR}/scripts/launch_cifar20_cdro_budgeted.py" \
    --outdir-root "${TRAIN_OUTROOT}" \
    --target-wcu "${TARGET_WCU}" \
    --batch-size 1024 \
    --batch-gpu 1024 \
    --cdro-n-steps-path "${n_steps}" \
    --cdro-total-budget-rho 0.0 \
    --outer-attack-weight 1.0 \
    --outer-clean-weight 0.0 \
    --tick-kimg 128 \
    --snap-ticks 1 \
    --dump-ticks 8 \
    --env-mode venv \
    --venv-dir "${VENV_DIR}" \
    --install-deps 0 \
    --launch

  run_dir="$(find_run_dir "${n_steps}")"
  if [[ -z "${run_dir}" || ! -d "${run_dir}" ]]; then
    echo "[ERROR] Could not resolve run dir for N=${n_steps}"
    exit 1
  fi

  total_kimg="$(jq -r '.budget_plan.total_kimg_int' "${run_dir}/cdro_budget_plan.json")"
  if [[ -z "${total_kimg}" || "${total_kimg}" == "null" ]]; then
    echo "[ERROR] Could not resolve total_kimg_int from ${run_dir}/cdro_budget_plan.json"
    exit 1
  fi

  echo "[INFO] Completed training for N=${n_steps}; run_dir=${run_dir}"
  echo "[INFO] Building manifest at kimg=${total_kimg}"
  python "${ROOT_DIR}/scripts/build_cifar_cdro_fid_manifest.py" \
    --cdro-run-dir "${run_dir}" \
    --outdir "${eval_outdir}" \
    --kimg "${total_kimg}" \
    --batch-size 1024 \
    --seed 0 \
    --train-percent-label "20%" \
    --ref-path "${PYTORCH_FID_REF}"

  echo "[INFO] Running pytorch-fid eval for N=${n_steps}"
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

  echo "[INFO] Running clean EDM probe for N=${n_steps}"
  python "${ROOT_DIR}/scripts/run_cifar_loss_probe_manifest.py" \
    --manifest-csv "${manifest_csv}" \
    --only-pending \
    --device cuda \
    --batch-size 512 \
    --max-images 10000 \
    --seed 0

  echo "[INFO] Final manifest row for N=${n_steps}:"
  tail -n 1 "${manifest_csv}"
}

echo "[INFO] Starting rho=0 N=8/N=16 baseline-80k workflow on $(hostname)"
echo "[INFO] Campaign tag: ${CAMPAIGN_TAG}"
echo "[INFO] Train outroot: ${TRAIN_OUTROOT}"
echo "[INFO] Eval sweep root: ${EVAL_SWEEP_ROOT}"
echo "[INFO] Run log: ${RUN_LOG}"
date -u
nvidia-smi || true

launch_and_eval 8
launch_and_eval 16

echo "[OK] rho=0 N=8/N=16 baseline-80k workflow complete."
