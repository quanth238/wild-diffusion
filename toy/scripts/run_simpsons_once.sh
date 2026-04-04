#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/toy_data/simpsons_images}"
TRAIN_ROOT="${TRAIN_ROOT:-${DATA_ROOT}/imagefolder/train}"
VAL_ROOT="${VAL_ROOT:-${DATA_ROOT}/imagefolder/val}"
FID_REF_PATH="${FID_REF_PATH:-${DATA_ROOT}/fid_refs/simpsons_val_seed0.npz}"
if [[ -z "${FID_DETECTOR_PATH:-}" ]]; then
  DEFAULT_FID_DETECTOR="/root/.cache/dnnlib/downloads/18d9c1159d16cd4cc6adf7db0f2dd2a9_https___api.ngc.nvidia.com_v2_models_nvidia_research_stylegan3_versions_1_files_metrics_inception-2015-12-05.pkl"
  if [[ -f "${DEFAULT_FID_DETECTOR}" ]]; then
    export FID_DETECTOR_PATH="${DEFAULT_FID_DETECTOR}"
  fi
fi

if [[ ! -d "${TRAIN_ROOT}" ]]; then
  echo "[error] missing Simpsons train root: ${TRAIN_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${VAL_ROOT}" ]]; then
  echo "[error] missing Simpsons val root: ${VAL_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${FID_REF_PATH}" ]]; then
  echo "[error] missing Simpsons FID ref: ${FID_REF_PATH}" >&2
  exit 1
fi

RUN_MODE="${RUN_MODE:-baseline}"   # baseline|robust
EXP_NAME="${EXP_NAME:-simpsons_smoke}"
OUTDIR="${OUTDIR:-toy_outputs/simpsons_runs}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-0}"
IMAGE_SIZE="${IMAGE_SIZE:-32}"
IMAGE_CHANNELS="${IMAGE_CHANNELS:-3}"
IMAGE_TRAIN_SIZE="${IMAGE_TRAIN_SIZE:-32}"
IMAGE_VAL_SIZE="${IMAGE_VAL_SIZE:-15}"
STEPS="${STEPS:-100}"
BATCH_SIZE="${BATCH_SIZE:-16}"
HIDDEN_DIM="${HIDDEN_DIM:-64}"
EVAL_SAMPLES="${EVAL_SAMPLES:-128}"
DEBUG_EVAL_BATCH="${DEBUG_EVAL_BATCH:-32}"
LOG_EVERY="${LOG_EVERY:-25}"

EXTRA_ARGS="${EXTRA_ARGS:-}"
EXTRA_ARGS="--compute-fid --fid-ref-path ${FID_REF_PATH} ${EXTRA_ARGS}"

export PYTHON_BIN RUN_MODE EXP_NAME OUTDIR DEVICE SEED
export DATASET_PATH="${TRAIN_ROOT}"
export DATASET_VAL_PATH="${VAL_ROOT}"
export IMAGE_SIZE IMAGE_CHANNELS IMAGE_TRAIN_SIZE IMAGE_VAL_SIZE
export STEPS BATCH_SIZE HIDDEN_DIM EVAL_SAMPLES DEBUG_EVAL_BATCH LOG_EVERY
export EXTRA_ARGS

bash "${ROOT_DIR}/toy/scripts/run_image_once.sh"
