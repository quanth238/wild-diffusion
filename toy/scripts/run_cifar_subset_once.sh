#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/toy_data/cifar10_subset_airplane_automobile}"
SUMMARY_PATH="${SUMMARY_PATH:-${DATA_ROOT}/summary.json}"
TRAIN_ROOT="${TRAIN_ROOT:-}"
TEST_ROOT="${TEST_ROOT:-}"
FID_REF_PATH="${FID_REF_PATH:-}"
IMAGE_TRAIN_SIZE="${IMAGE_TRAIN_SIZE:-}"
IMAGE_VAL_SIZE="${IMAGE_VAL_SIZE:-}"

if [[ -f "${SUMMARY_PATH}" ]]; then
  eval "$(
    SUMMARY_PATH="${SUMMARY_PATH}" "${PYTHON_BIN}" - <<'PY'
import json
import os
import shlex

path = os.environ["SUMMARY_PATH"]
with open(path, "r", encoding="utf-8") as fh:
    payload = json.load(fh)
print(f"SUMMARY_TRAIN_ROOT={shlex.quote(str(payload['train_root']))}")
print(f"SUMMARY_TEST_ROOT={shlex.quote(str(payload['test_root']))}")
print(f"SUMMARY_FID_REF_PATH={shlex.quote(str(payload.get('fid_ref_path') or ''))}")
dataset = payload.get("dataset", {})
print(f"SUMMARY_TRAIN_TOTAL={int(dataset.get('train_total', 0))}")
print(f"SUMMARY_TEST_TOTAL={int(dataset.get('test_total', 0))}")
PY
  )"
fi

TRAIN_ROOT="${TRAIN_ROOT:-${SUMMARY_TRAIN_ROOT:-${DATA_ROOT}/imagefolder/train}}"
TEST_ROOT="${TEST_ROOT:-${SUMMARY_TEST_ROOT:-${DATA_ROOT}/imagefolder/test}}"
FID_REF_PATH="${FID_REF_PATH:-${SUMMARY_FID_REF_PATH:-${DATA_ROOT}/fid_refs/cifar10_airplane_automobile_test_32x32.npz}}"
IMAGE_TRAIN_SIZE="${IMAGE_TRAIN_SIZE:-${SUMMARY_TRAIN_TOTAL:-2000}}"
IMAGE_VAL_SIZE="${IMAGE_VAL_SIZE:-${SUMMARY_TEST_TOTAL:-2000}}"

if [[ ! -d "${TRAIN_ROOT}" ]]; then
  echo "[error] missing CIFAR subset train root: ${TRAIN_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${TEST_ROOT}" ]]; then
  echo "[error] missing CIFAR subset test root: ${TEST_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${FID_REF_PATH}" ]]; then
  echo "[error] missing CIFAR subset FID ref: ${FID_REF_PATH}" >&2
  echo "[hint] run toy/scripts/setup_cifar_subset_imagefolder.py first (without --skip-fid-ref)." >&2
  exit 1
fi

if [[ -z "${FID_DETECTOR_PATH:-}" ]]; then
  DEFAULT_FID_DETECTOR="/root/.cache/dnnlib/downloads/18d9c1159d16cd4cc6adf7db0f2dd2a9_https___api.ngc.nvidia.com_v2_models_nvidia_research_stylegan3_versions_1_files_metrics_inception-2015-12-05.pkl"
  if [[ -f "${DEFAULT_FID_DETECTOR}" ]]; then
    export FID_DETECTOR_PATH="${DEFAULT_FID_DETECTOR}"
  fi
fi

RUN_MODE="${RUN_MODE:-robust}"   # baseline|robust
METHOD_VERSION="${METHOD_VERSION:-cdro}"
EXP_NAME="${EXP_NAME:-cifar2_image_conv_cdro_smoke}"
OUTDIR="${OUTDIR:-toy_outputs/cifar2_image_conv_runs}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-0}"
IMAGE_SIZE="${IMAGE_SIZE:-32}"
IMAGE_CHANNELS="${IMAGE_CHANNELS:-3}"
STEPS="${STEPS:-100}"
BATCH_SIZE="${BATCH_SIZE:-256}"
HIDDEN_DIM="${HIDDEN_DIM:-64}"
EVAL_SAMPLES="${EVAL_SAMPLES:-256}"
DEBUG_EVAL_BATCH="${DEBUG_EVAL_BATCH:-64}"
LOG_EVERY="${LOG_EVERY:-25}"
N_STEPS_PATH="${N_STEPS_PATH:-32}"
OUTER_ATTACK_WEIGHT="${OUTER_ATTACK_WEIGHT:-0.3}"
OUTER_CLEAN_WEIGHT="${OUTER_CLEAN_WEIGHT:-0.0}"
USE_EMA_EVAL="${USE_EMA_EVAL:-1}"
EMA_MODE="${EMA_MODE:-official}"
EMA_DECAY="${EMA_DECAY:-0.999}"
EMA_HALFLIFE_KIMG="${EMA_HALFLIFE_KIMG:-500}"
EMA_RAMPUP_RATIO="${EMA_RAMPUP_RATIO:-0.05}"
DISABLE_EMA_RAMPUP="${DISABLE_EMA_RAMPUP:-0}"
CDRO_STEP_SIZE="${CDRO_STEP_SIZE:-0.02}"
CDRO_TOTAL_BUDGET_RHO="${CDRO_TOTAL_BUDGET_RHO:-0.001}"
CDRO_TIME_HORIZON="${CDRO_TIME_HORIZON:-1.0}"
CDRO_WARMUP_FRACTION="${CDRO_WARMUP_FRACTION:-0.2}"
CDRO_EDM_LADDER_MODE="${CDRO_EDM_LADDER_MODE:-stochastic_stratified_quantile}"

EXTRA_ARGS="${EXTRA_ARGS:-}"
EXTRA_ARGS="--compute-fid --fid-ref-path ${FID_REF_PATH} --cdro-step-size ${CDRO_STEP_SIZE} --cdro-total-budget-rho ${CDRO_TOTAL_BUDGET_RHO} --cdro-time-horizon ${CDRO_TIME_HORIZON} --cdro-edm-ladder-mode ${CDRO_EDM_LADDER_MODE} --cdro-warmup-fraction ${CDRO_WARMUP_FRACTION} ${EXTRA_ARGS}"
if [[ "${USE_EMA_EVAL}" == "1" ]]; then
  EXTRA_ARGS="--use-ema-eval --ema-mode ${EMA_MODE} ${EXTRA_ARGS}"
  if [[ "${EMA_MODE}" == "fixed" ]]; then
    EXTRA_ARGS="--ema-decay ${EMA_DECAY} ${EXTRA_ARGS}"
  else
    EXTRA_ARGS="--ema-halflife-kimg ${EMA_HALFLIFE_KIMG} ${EXTRA_ARGS}"
    if [[ "${DISABLE_EMA_RAMPUP}" == "1" ]]; then
      EXTRA_ARGS="--disable-ema-rampup ${EXTRA_ARGS}"
    else
      EXTRA_ARGS="--ema-rampup-ratio ${EMA_RAMPUP_RATIO} ${EXTRA_ARGS}"
    fi
  fi
fi

export PYTHON_BIN RUN_MODE METHOD_VERSION EXP_NAME OUTDIR DEVICE SEED
export DATASET_PATH="${TRAIN_ROOT}"
export DATASET_VAL_PATH="${TEST_ROOT}"
export IMAGE_SIZE IMAGE_CHANNELS IMAGE_TRAIN_SIZE IMAGE_VAL_SIZE
export STEPS BATCH_SIZE HIDDEN_DIM EVAL_SAMPLES DEBUG_EVAL_BATCH LOG_EVERY
export N_STEPS_PATH OUTER_ATTACK_WEIGHT OUTER_CLEAN_WEIGHT
export EXTRA_ARGS

bash "${ROOT_DIR}/toy/scripts/run_image_once.sh"
