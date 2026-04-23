#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

TARGET_JOB_ID="${TARGET_JOB_ID:-10640}"
CURRENT_RUN_LOG="${CURRENT_RUN_LOG:-/mnt/data/bachlc/GM-CDRO/training-runs/fid-sweeps/logs/cdro_rho0_n004_baseline80k_20260422T094845Z.log}"
CURRENT_DONE_PATTERN="${CURRENT_DONE_PATTERN:-[OK] rho=0 N=4 baseline-80k workflow complete.}"
POLL_SEC="${POLL_SEC:-60}"
CAMPAIGN_TAG="${CAMPAIGN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
VENV_DIR="${VENV_DIR:-/home/bachlc/.venvs/wild-diffusion-h100}"
TARGET_WCU="${TARGET_WCU:-184148.22639230292}"
TRAIN_LR="${TRAIN_LR:-1e-6}"
TRAIN_OUTROOT="${TRAIN_OUTROOT:-/mnt/data/bachlc/GM-CDRO/training-runs/paper-cifar10-cdro-fp16-rho0-lr1e6-baseline80k_${CAMPAIGN_TAG}}"
EVAL_SWEEP_ROOT="${EVAL_SWEEP_ROOT:-/mnt/data/bachlc/GM-CDRO/training-runs/fid-sweeps/cdro_rho0_lr1e6_n004_baseline80k_${CAMPAIGN_TAG}}"
LOG_DIR="${LOG_DIR:-/mnt/data/bachlc/GM-CDRO/training-runs/fid-sweeps/logs}"
QUEUE_LOG="${QUEUE_LOG:-${LOG_DIR}/cdro_rho0_lr1e6_n004_queue_${CAMPAIGN_TAG}.log}"
RUN_LOG="${RUN_LOG:-${LOG_DIR}/cdro_rho0_lr1e6_n004_baseline80k_${CAMPAIGN_TAG}.log}"

mkdir -p "${LOG_DIR}" "${TRAIN_OUTROOT}" "${EVAL_SWEEP_ROOT}"

if [[ -r /proc/self/fd ]]; then
  exec > >(tee -a "${QUEUE_LOG}") 2>&1
else
  exec >>"${QUEUE_LOG}" 2>&1
fi

echo "[INFO] Waiting for current N=4 rho=0 workflow to finish before launching lr=${TRAIN_LR} follow-up."
echo "[INFO] Target Slurm job: ${TARGET_JOB_ID}"
echo "[INFO] Current run log: ${CURRENT_RUN_LOG}"
echo "[INFO] Done pattern: ${CURRENT_DONE_PATTERN}"
echo "[INFO] Follow-up campaign tag: ${CAMPAIGN_TAG}"
echo "[INFO] Follow-up TRAIN_OUTROOT: ${TRAIN_OUTROOT}"
echo "[INFO] Follow-up EVAL_SWEEP_ROOT: ${EVAL_SWEEP_ROOT}"
echo "[INFO] Follow-up RUN_LOG: ${RUN_LOG}"
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

echo "[INFO] Launching rho=0, N=4, lr=${TRAIN_LR} workflow inside job ${TARGET_JOB_ID}"
date -u

srun --jobid="${TARGET_JOB_ID}" --overlap --quiet \
  env \
    CAMPAIGN_TAG="${CAMPAIGN_TAG}" \
    VENV_DIR="${VENV_DIR}" \
    TARGET_WCU="${TARGET_WCU}" \
    TRAIN_LR="${TRAIN_LR}" \
    TRAIN_OUTROOT="${TRAIN_OUTROOT}" \
    EVAL_SWEEP_ROOT="${EVAL_SWEEP_ROOT}" \
    LOG_DIR="${LOG_DIR}" \
    RUN_LOG="${RUN_LOG}" \
    RUN_N_STEPS=4 \
    bash "${ROOT_DIR}/scripts/run_cifar_rho0_lr1e6_n004_baseline80k_h100.sh"

echo "[OK] Queued rho=0, N=4, lr=${TRAIN_LR} follow-up finished."
date -u
