#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

TARGET_JOB_ID="${TARGET_JOB_ID:?TARGET_JOB_ID is required}"
CURRENT_RUN_LOG="${CURRENT_RUN_LOG:?CURRENT_RUN_LOG is required}"
CURRENT_DONE_PATTERN="${CURRENT_DONE_PATTERN:-[OK] N=4 positive-rho sweep complete.}"
POLL_SEC="${POLL_SEC:-60}"
RUN_DIR="${RUN_DIR:?RUN_DIR is required}"
EVAL_SWEEP_ROOT="${EVAL_SWEEP_ROOT:?EVAL_SWEEP_ROOT is required}"
PLOT_TAG="${PLOT_TAG:-cdro}"
VENV_DIR="${VENV_DIR:-/home/bachlc/.venvs/wild-diffusion-h100}"
LOG_DIR="${LOG_DIR:-/mnt/data/bachlc/GM-CDRO/training-runs/fid-sweeps/logs}"
QUEUE_LOG="${QUEUE_LOG:-${LOG_DIR}/${PLOT_TAG}_compare_curves_queue.log}"
RUN_LOG="${RUN_LOG:-${LOG_DIR}/${PLOT_TAG}_compare_curves.log}"
LOSS_POINTS_MAX="${LOSS_POINTS_MAX:-50}"
FID_POINTS_MAX="${FID_POINTS_MAX:-8}"

mkdir -p "${LOG_DIR}" "${EVAL_SWEEP_ROOT}"

if [[ -r /proc/self/fd ]]; then
  exec > >(tee -a "${QUEUE_LOG}") 2>&1
else
  exec >>"${QUEUE_LOG}" 2>&1
fi

echo "[INFO] Waiting for current workflow to finish before launching compare-curve job."
echo "[INFO] Target Slurm job: ${TARGET_JOB_ID}"
echo "[INFO] Current run log: ${CURRENT_RUN_LOG}"
echo "[INFO] Done pattern: ${CURRENT_DONE_PATTERN}"
echo "[INFO] RUN_DIR: ${RUN_DIR}"
echo "[INFO] EVAL_SWEEP_ROOT: ${EVAL_SWEEP_ROOT}"
echo "[INFO] PLOT_TAG: ${PLOT_TAG}"
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

echo "[INFO] Launching compare-curve generation inside job ${TARGET_JOB_ID}"
date -u

srun --jobid="${TARGET_JOB_ID}" --overlap --quiet \
  env \
    RUN_DIR="${RUN_DIR}" \
    EVAL_SWEEP_ROOT="${EVAL_SWEEP_ROOT}" \
    PLOT_TAG="${PLOT_TAG}" \
    VENV_DIR="${VENV_DIR}" \
    LOG_DIR="${LOG_DIR}" \
    RUN_LOG="${RUN_LOG}" \
    LOSS_POINTS_MAX="${LOSS_POINTS_MAX}" \
    FID_POINTS_MAX="${FID_POINTS_MAX}" \
    bash "${ROOT_DIR}/scripts/run_cifar_cdro_compare_curves.sh"

echo "[OK] Queued compare-curve generation finished for ${PLOT_TAG}."
date -u
