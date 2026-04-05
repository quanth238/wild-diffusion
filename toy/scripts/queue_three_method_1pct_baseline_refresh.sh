#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"

PY="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "${PY}" ]]; then
  echo "[error] missing venv python: ${PY}" >&2
  exit 1
fi

WAIT_PID="${1:-}"
OUTDIR="${REPO_ROOT}/toy_outputs/simpsons_mnist_rgb_three_method_1pct_multiseed_raw"
PREFIX="simpsons_mnist_rgb_three_method_1pct_multiseed_raw"
QUEUE_LOG="${REPO_ROOT}/toy_outputs/queue_three_method_1pct_baseline_refresh_20260405.log"
REFRESH_LOG="${REPO_ROOT}/toy_outputs/three_method_1pct_multiseed_raw_baseline_refresh_20260405.log"

WDRO_RAW_CSV="${OUTDIR}/${PREFIX}_wdro_raw_seed_rows.csv"
CDRO_RAW_CSV="${OUTDIR}/${PREFIX}_cdro_raw_seed_rows.csv"
COMBINED_RAW_CSV="${OUTDIR}/${PREFIX}_all_methods_raw_seed_rows.csv"
MANIFEST_JSON="${OUTDIR}/${PREFIX}_manifest.json"
BASELINE_DIR="${OUTDIR}/baseline"
BASELINE_RAW_CSV="${OUTDIR}/${PREFIX}_baseline_raw_seed_rows.csv"
BASELINE_ALL_EVAL_RAW_CSV="${OUTDIR}/${PREFIX}_baseline_all_eval_raw_seed_rows.csv"

if [[ -z "${FID_DETECTOR_PATH:-}" ]]; then
  DEFAULT_FID_DETECTOR="/root/.cache/dnnlib/downloads/18d9c1159d16cd4cc6adf7db0f2dd2a9_https___api.ngc.nvidia.com_v2_models_nvidia_research_stylegan3_versions_1_files_metrics_inception-2015-12-05.pkl"
  if [[ -f "${DEFAULT_FID_DETECTOR}" ]]; then
    export FID_DETECTOR_PATH="${DEFAULT_FID_DETECTOR}"
  fi
fi

mkdir -p "$(dirname "${QUEUE_LOG}")"
exec > >(tee -a "${QUEUE_LOG}") 2>&1

wait_for_pid() {
  local pid="$1"
  if [[ -z "${pid}" ]]; then
    return 0
  fi
  echo "[queue] waiting for pid=${pid} at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  while kill -0 "${pid}" 2>/dev/null; do
    echo "[queue] pid=${pid} still running at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    sleep 60
  done
  echo "[queue] pid=${pid} finished at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}

require_finished_main_run() {
  local missing=0
  for path in "${WDRO_RAW_CSV}" "${CDRO_RAW_CSV}" "${COMBINED_RAW_CSV}" "${MANIFEST_JSON}"; do
    if [[ ! -f "${path}" ]]; then
      echo "[queue] missing required finished-run artifact: ${path}"
      missing=1
    fi
  done
  if [[ "${missing}" -ne 0 ]]; then
    echo "[queue] current three-method sweep does not appear to have finished cleanly; aborting queued refresh"
    exit 1
  fi
}

backup_baseline_artifacts() {
  local stamp
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  local backup_dir="${OUTDIR}/baseline_pre_aux_refresh_${stamp}"
  mkdir -p "${backup_dir}"
  echo "[queue] backing up baseline artifacts to ${backup_dir}"
  if [[ -d "${BASELINE_DIR}" ]]; then
    mv "${BASELINE_DIR}" "${backup_dir}/baseline"
  fi
  if [[ -f "${BASELINE_RAW_CSV}" ]]; then
    mv "${BASELINE_RAW_CSV}" "${backup_dir}/$(basename "${BASELINE_RAW_CSV}")"
  fi
  if [[ -f "${BASELINE_ALL_EVAL_RAW_CSV}" ]]; then
    mv "${BASELINE_ALL_EVAL_RAW_CSV}" "${backup_dir}/$(basename "${BASELINE_ALL_EVAL_RAW_CSV}")"
  fi
}

run_refresh_collector() {
  echo "[queue] starting baseline refresh collector at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  "${PY}" "${REPO_ROOT}/toy/scripts/collect_three_method_seed_data.py" \
    --outdir "${OUTDIR}" \
    --prefix "${PREFIX}" \
    --skip-existing \
    > "${REFRESH_LOG}" 2>&1
  echo "[queue] finished baseline refresh collector at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "[queue] refresh log: ${REFRESH_LOG}"
}

wait_for_pid "${WAIT_PID}"
require_finished_main_run
backup_baseline_artifacts
run_refresh_collector

