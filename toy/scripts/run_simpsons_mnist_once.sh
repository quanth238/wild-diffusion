#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/toy_data/simpsons_mnist_rgb}"
TRAIN_ROOT="${TRAIN_ROOT:-${DATA_ROOT}/imagefolder/train}"
TEST_ROOT="${TEST_ROOT:-${DATA_ROOT}/imagefolder/test}"
FID_REF_PATH="${FID_REF_PATH:-${DATA_ROOT}/fid_refs/simpsons_mnist_rgb_test_28x28.npz}"

if [[ -z "${FID_DETECTOR_PATH:-}" ]]; then
  DEFAULT_FID_DETECTOR="/root/.cache/dnnlib/downloads/18d9c1159d16cd4cc6adf7db0f2dd2a9_https___api.ngc.nvidia.com_v2_models_nvidia_research_stylegan3_versions_1_files_metrics_inception-2015-12-05.pkl"
  if [[ -f "${DEFAULT_FID_DETECTOR}" ]]; then
    export FID_DETECTOR_PATH="${DEFAULT_FID_DETECTOR}"
  fi
fi

if [[ ! -d "${TRAIN_ROOT}" ]]; then
  echo "[error] missing Simpsons-MNIST RGB train root: ${TRAIN_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${TEST_ROOT}" ]]; then
  echo "[error] missing Simpsons-MNIST RGB test root: ${TEST_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${FID_REF_PATH}" ]]; then
  echo "[error] missing Simpsons-MNIST RGB FID ref: ${FID_REF_PATH}" >&2
  exit 1
fi

RUN_MODE="${RUN_MODE:-robust}"   # baseline|robust
METHOD_VERSION="${METHOD_VERSION:-cdro}"
EXP_NAME="${EXP_NAME:-simpsons_mnist_rgb_cdro_smoke}"
OUTDIR="${OUTDIR:-toy_outputs/simpsons_mnist_rgb_runs}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-0}"
IMAGE_SIZE="${IMAGE_SIZE:-28}"
IMAGE_CHANNELS="${IMAGE_CHANNELS:-3}"
IMAGE_TRAIN_SIZE="${IMAGE_TRAIN_SIZE:-8000}"
IMAGE_VAL_SIZE="${IMAGE_VAL_SIZE:-2000}"
STEPS="${STEPS:-100}"
BATCH_SIZE="${BATCH_SIZE:-32}"
HIDDEN_DIM="${HIDDEN_DIM:-64}"
EVAL_SAMPLES="${EVAL_SAMPLES:-256}"
DEBUG_EVAL_BATCH="${DEBUG_EVAL_BATCH:-64}"
LOG_EVERY="${LOG_EVERY:-25}"
N_STEPS_PATH="${N_STEPS_PATH:-64}"
OUTER_ATTACK_WEIGHT="${OUTER_ATTACK_WEIGHT:-0.3}"
OUTER_CLEAN_WEIGHT="${OUTER_CLEAN_WEIGHT:-1.0}"
USE_EMA_EVAL="${USE_EMA_EVAL:-0}"
EMA_DECAY="${EMA_DECAY:-0.999}"
CDRO_ANTITHETIC_ROLLOUTS="${CDRO_ANTITHETIC_ROLLOUTS:-0}"

EXTRA_ARGS="${EXTRA_ARGS:-}"
EXTRA_ARGS="--compute-fid --fid-ref-path ${FID_REF_PATH} --cdro-step-size 0.02 --cdro-total-budget-rho 4.0 --cdro-time-horizon 1.0 --cdro-warmup-fraction 0.05 ${EXTRA_ARGS}"
if [[ "${USE_EMA_EVAL}" == "1" ]]; then
  EXTRA_ARGS="--use-ema-eval --ema-decay ${EMA_DECAY} ${EXTRA_ARGS}"
fi
if [[ "${CDRO_ANTITHETIC_ROLLOUTS}" == "1" ]]; then
  EXTRA_ARGS="--cdro-antithetic-rollouts ${EXTRA_ARGS}"
fi

export PYTHON_BIN RUN_MODE METHOD_VERSION EXP_NAME OUTDIR DEVICE SEED
export DATASET_PATH="${TRAIN_ROOT}"
export DATASET_VAL_PATH="${TEST_ROOT}"
export IMAGE_SIZE IMAGE_CHANNELS IMAGE_TRAIN_SIZE IMAGE_VAL_SIZE
export STEPS BATCH_SIZE HIDDEN_DIM EVAL_SAMPLES DEBUG_EVAL_BATCH LOG_EVERY
export N_STEPS_PATH OUTER_ATTACK_WEIGHT OUTER_CLEAN_WEIGHT
export EXTRA_ARGS

bash "${ROOT_DIR}/toy/scripts/run_image_once.sh"
