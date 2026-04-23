#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

TARGET_JOB_ID="${TARGET_JOB_ID:-10640}"
CURRENT_RUN_LOG="${CURRENT_RUN_LOG:-/mnt/data/bachlc/GM-CDRO/training-runs/fid-sweeps/logs/cdro_rho0_n004_baseline80k_20260422T094845Z.log}"
CURRENT_DONE_PATTERN="${CURRENT_DONE_PATTERN:-[OK] rho=0 N=4 baseline-80k workflow complete.}"
POLL_SEC="${POLL_SEC:-60}"
RHO_VALUES="${RHO_VALUES:-0.001,0.01,0.05}"
CAMPAIGN_TAG="${CAMPAIGN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
VENV_DIR="${VENV_DIR:-/home/bachlc/.venvs/wild-diffusion-h100}"
TARGET_WCU="${TARGET_WCU:-184148.22639230292}"
TRAIN_OUTROOT_BASE="${TRAIN_OUTROOT_BASE:-/mnt/data/bachlc/GM-CDRO/training-runs/paper-cifar10-cdro-fp16-n004-positive-rho_${CAMPAIGN_TAG}}"
EVAL_SWEEP_ROOT_BASE="${EVAL_SWEEP_ROOT_BASE:-/mnt/data/bachlc/GM-CDRO/training-runs/fid-sweeps/cdro_n004_positive_rho_${CAMPAIGN_TAG}}"
LOG_DIR="${LOG_DIR:-/mnt/data/bachlc/GM-CDRO/training-runs/fid-sweeps/logs}"
QUEUE_LOG="${QUEUE_LOG:-${LOG_DIR}/cdro_n004_positive_rho_queue_${CAMPAIGN_TAG}.log}"
RUN_LOG="${RUN_LOG:-${LOG_DIR}/cdro_n004_positive_rho_${CAMPAIGN_TAG}.log}"

mkdir -p "${LOG_DIR}" "${TRAIN_OUTROOT_BASE}" "${EVAL_SWEEP_ROOT_BASE}"

if [[ -r /proc/self/fd ]]; then
  exec > >(tee -a "${QUEUE_LOG}") 2>&1
else
  exec >>"${QUEUE_LOG}" 2>&1
fi

echo "[INFO] Waiting for current N=4 rho=0 workflow to finish before launching positive-rho runs."
echo "[INFO] Target Slurm job: ${TARGET_JOB_ID}"
echo "[INFO] Current run log: ${CURRENT_RUN_LOG}"
echo "[INFO] Done pattern: ${CURRENT_DONE_PATTERN}"
echo "[INFO] Queued RHO_VALUES: ${RHO_VALUES}"
echo "[INFO] Positive-rho campaign tag: ${CAMPAIGN_TAG}"
date -u

while true; do
  if [[ -f "${CURRENT_RUN_LOG}" ]] && grep -Fq "${CURRENT_DONE_PATTERN}" "${CURRENT_RUN_LOG}"; then
    echo "[INFO] Detected current workflow completion."
    break
  fi
  if ! squeue -h -j "${TARGET_JOB_ID}" | grep -q .; then
    echo "[ERROR] Slurm job ${TARGET_JOB_ID} is no longer visible before current workflow completed."
    exit 1
  fi
  sleep "${POLL_SEC}"
done

echo "[INFO] Launching N=4 positive-rho sweep inside job ${TARGET_JOB_ID}"
date -u

srun --jobid="${TARGET_JOB_ID}" --overlap --quiet \
  env \
    CAMPAIGN_TAG="${CAMPAIGN_TAG}" \
    VENV_DIR="${VENV_DIR}" \
    TARGET_WCU="${TARGET_WCU}" \
    TRAIN_OUTROOT_BASE="${TRAIN_OUTROOT_BASE}" \
    EVAL_SWEEP_ROOT_BASE="${EVAL_SWEEP_ROOT_BASE}" \
    LOG_DIR="${LOG_DIR}" \
    RUN_LOG="${RUN_LOG}" \
    RHO_VALUES="${RHO_VALUES}" \
    bash "${ROOT_DIR}/scripts/run_cifar_n004_positive_rho_sweep.sh"

echo "[OK] Queued N=4 positive-rho sweep finished."
date -u
