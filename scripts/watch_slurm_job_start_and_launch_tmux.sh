#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

JOB_ID="${JOB_ID:?JOB_ID is required}"
TMUX_TARGET="${TMUX_TARGET:-bachlc:h100}"
LAUNCH_SCRIPT="${LAUNCH_SCRIPT:-${ROOT_DIR}/scripts/run_cifar_rho0_n001_confirmation_h100.sh}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/training-runs/fid-sweeps/logs}"
POLL_INTERVAL_SEC="${POLL_INTERVAL_SEC:-60}"
POST_START_DELAY_SEC="${POST_START_DELAY_SEC:-15}"
WATCH_LABEL="${WATCH_LABEL:-cdro_rho0_n001_confirm}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
WATCH_LOG="${WATCH_LOG:-${LOG_DIR}/${WATCH_LABEL}_job${JOB_ID}_watch_${STAMP}.log}"
LAUNCH_LOG="${LAUNCH_LOG:-${LOG_DIR}/${WATCH_LABEL}_job${JOB_ID}_launch_${STAMP}.log}"
LAUNCH_STAMP="${LAUNCH_STAMP:-${LOG_DIR}/${WATCH_LABEL}_job${JOB_ID}.launched}"

mkdir -p "${LOG_DIR}"

if [[ ! -f "${LAUNCH_SCRIPT}" ]]; then
  echo "[ERROR] Launch script not found: ${LAUNCH_SCRIPT}"
  exit 1
fi

if ! tmux list-panes -t "${TMUX_TARGET}" >/dev/null 2>&1; then
  echo "[ERROR] tmux target not found: ${TMUX_TARGET}"
  exit 1
fi

log() {
  local message="$1"
  printf '[%s] %s\n' "$(date -u '+%Y-%m-%d %H:%M:%S UTC')" "${message}" | tee -a "${WATCH_LOG}"
}

pane_already_contains_launch() {
  tmux capture-pane -pt "${TMUX_TARGET}" -S -200 2>/dev/null | grep -Fq "$(basename "${LAUNCH_SCRIPT}")"
}

launch_into_tmux() {
  if pane_already_contains_launch; then
    log "Launch script already appears in ${TMUX_TARGET}; not sending duplicate command."
    touch "${LAUNCH_STAMP}"
    return 0
  fi

  local launch_cmd
  launch_cmd="cd ${ROOT_DIR} && bash ${LAUNCH_SCRIPT} 2>&1 | tee -a ${LAUNCH_LOG}"
  tmux send-keys -t "${TMUX_TARGET}" "${launch_cmd}" C-m
  touch "${LAUNCH_STAMP}"
  log "Sent launch command to ${TMUX_TARGET}: ${launch_cmd}"
}

log "Watching Slurm job ${JOB_ID} for RUNNING state; target tmux pane is ${TMUX_TARGET}"
if [[ -f "${LAUNCH_STAMP}" ]]; then
  log "Existing launch stamp found at ${LAUNCH_STAMP}; exiting without re-arming."
  exit 0
fi

while true; do
  state="$(squeue -h -j "${JOB_ID}" -o '%T' 2>/dev/null | head -n 1 || true)"
  if [[ -z "${state}" ]]; then
    log "Job ${JOB_ID} is no longer visible in squeue and was not launched; exiting."
    exit 1
  fi

  log "Observed Slurm state for job ${JOB_ID}: ${state}"
  if [[ "${state}" == "RUNNING" ]]; then
    log "Job ${JOB_ID} is running; waiting ${POST_START_DELAY_SEC}s before sending launch command."
    sleep "${POST_START_DELAY_SEC}"
    launch_into_tmux
    exit 0
  fi

  sleep "${POLL_INTERVAL_SEC}"
done
