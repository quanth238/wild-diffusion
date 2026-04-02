#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_MODE="${RUN_MODE:-robust}"   # robust|baseline
EXP_NAME="${EXP_NAME:-image_once}"
OUTDIR="${OUTDIR:-toy_outputs}"
DEVICE="${DEVICE:-auto}"         # auto|cpu|cuda
SEED="${SEED:-0}"
BASELINE_CKPT_PATH="${BASELINE_CKPT_PATH:-}"

DATASET_PATH="${DATASET_PATH:-}"
DATASET_VAL_PATH="${DATASET_VAL_PATH:-}"
IMAGE_SIZE="${IMAGE_SIZE:-32}"
IMAGE_CHANNELS="${IMAGE_CHANNELS:-3}"
IMAGE_TRAIN_SIZE="${IMAGE_TRAIN_SIZE:-2000}"
IMAGE_VAL_SIZE="${IMAGE_VAL_SIZE:-2000}"
IMAGE_SPLIT_SEED="${IMAGE_SPLIT_SEED:-0}"

STEPS="${STEPS:-3000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LOG_EVERY="${LOG_EVERY:-200}"
EVAL_SAMPLES="${EVAL_SAMPLES:-256}"
DEBUG_EVAL_BATCH="${DEBUG_EVAL_BATCH:-64}"
DEBUG_TERMINAL_STEP="${DEBUG_TERMINAL_STEP:-20}"
N_STEPS_PATH="${N_STEPS_PATH:-24}"
HIDDEN_DIM="${HIDDEN_DIM:-64}"

KAPPA="${KAPPA:-0.3}"
OUTER_CLEAN_WEIGHT="${OUTER_CLEAN_WEIGHT:-1.0}"
OUTER_ATTACK_WEIGHT="${OUTER_ATTACK_WEIGHT:-0.5}"
WARMUP_CLEAN_STEPS="${WARMUP_CLEAN_STEPS:-900}"
WARMUP_RAMP_STEPS="${WARMUP_RAMP_STEPS:-600}"

RUN_CHECKS="${RUN_CHECKS:-1}"                    # 1|0
BASELINE_GATE_HARD_FAIL="${BASELINE_GATE_HARD_FAIL:-1}"  # 1|0
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [[ "${RUN_MODE}" != "robust" && "${RUN_MODE}" != "baseline" ]]; then
  echo "[error] RUN_MODE must be 'robust' or 'baseline', got: ${RUN_MODE}"
  exit 1
fi
if [[ -z "${DATASET_PATH}" ]]; then
  echo "[error] DATASET_PATH is required for run_image_once.sh"
  exit 1
fi

mkdir -p "${OUTDIR}"

args=(
  "--exp-name" "${EXP_NAME}"
  "--outdir" "${OUTDIR}"
  "--device" "${DEVICE}"
  "--seed" "${SEED}"
  "--steps" "${STEPS}"
  "--batch-size" "${BATCH_SIZE}"
  "--log-every" "${LOG_EVERY}"
  "--eval-samples" "${EVAL_SAMPLES}"
  "--debug-eval-batch" "${DEBUG_EVAL_BATCH}"
  "--debug-terminal-step" "${DEBUG_TERMINAL_STEP}"
  "--n-steps-path" "${N_STEPS_PATH}"
  "--hidden-dim" "${HIDDEN_DIM}"
  "--dataset-kind" "image_folder"
  "--model-kind" "image_conv"
  "--diagnostics-kind" "image_basic"
  "--dataset-path" "${DATASET_PATH}"
  "--image-size" "${IMAGE_SIZE}"
  "--image-channels" "${IMAGE_CHANNELS}"
  "--image-train-size" "${IMAGE_TRAIN_SIZE}"
  "--image-val-size" "${IMAGE_VAL_SIZE}"
  "--image-split-seed" "${IMAGE_SPLIT_SEED}"
)

if [[ -n "${DATASET_VAL_PATH}" ]]; then
  args+=("--dataset-val-path" "${DATASET_VAL_PATH}")
fi

if [[ "${RUN_MODE}" == "baseline" ]]; then
  args+=("--baseline-only")
else
  args+=(
    "--control-radius-kappa" "${KAPPA}"
    "--outer-clean-weight" "${OUTER_CLEAN_WEIGHT}"
    "--outer-attack-weight" "${OUTER_ATTACK_WEIGHT}"
    "--warmup-clean-steps" "${WARMUP_CLEAN_STEPS}"
    "--warmup-ramp-steps" "${WARMUP_RAMP_STEPS}"
  )
fi

if [[ "${RUN_CHECKS}" == "1" ]]; then
  args+=("--run-checks")
else
  args+=("--skip-checks")
fi

if [[ "${BASELINE_GATE_HARD_FAIL}" == "1" ]]; then
  args+=("--baseline-gate-error-on-fail")
fi

if [[ -n "${EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  extra_arr=( ${EXTRA_ARGS} )
  args+=("${extra_arr[@]}")
fi

if [[ -n "${BASELINE_CKPT_PATH}" ]]; then
  args+=("--baseline-ckpt-path" "${BASELINE_CKPT_PATH}")
fi

echo "[run_image_once] mode=${RUN_MODE} exp=${EXP_NAME} outdir=${OUTDIR} device=${DEVICE} seed=${SEED}"
echo "[run_image_once] dataset=${DATASET_PATH}"
echo "[run_image_once] command: ${PYTHON_BIN} toy/run_toy.py ${args[*]}"
"${PYTHON_BIN}" toy/run_toy.py "${args[@]}"

echo "[run_image_once] done: ${OUTDIR}/${EXP_NAME}"
echo "[run_image_once] metrics: ${OUTDIR}/${EXP_NAME}/metrics.json"
