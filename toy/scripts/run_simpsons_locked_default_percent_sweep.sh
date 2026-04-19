#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
DEVICE="${DEVICE:-cuda}"
SEEDS="${SEEDS:-0}"
PERCENTS="${PERCENTS:-5}"

DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/toy_data/simpsons_mnist_rgb}"
DATASET_PATH="${DATASET_PATH:-${DATA_ROOT}/imagefolder/train}"
DATASET_VAL_PATH="${DATASET_VAL_PATH:-${DATA_ROOT}/imagefolder/test}"
FID_REF_PATH="${FID_REF_PATH:-${DATA_ROOT}/fid_refs/simpsons_mnist_rgb_test_28x28.npz}"
CALIBRATION_PATH="${CALIBRATION_PATH:-${ROOT_DIR}/toy_outputs/compute_calibration/simpsons_mnist_rgb_image_conv_edm_b256_h64_cuda.json}"

OUTDIR="${OUTDIR:-${ROOT_DIR}/toy_outputs/simpsons5_cdro_400k_warmup20_rho0p001_n032_cw0_aw100_perex_default}"
PREFIX_ROOT="${PREFIX_ROOT:-simpsons5_cdro_400k_warmup20_rho0p001_n032_i1_aw100_cw000_perex_default}"

IMAGE_SIZE="${IMAGE_SIZE:-28}"
IMAGE_CHANNELS="${IMAGE_CHANNELS:-3}"
IMAGE_VAL_SIZE="${IMAGE_VAL_SIZE:-2000}"
TRAIN_POPULATION_SIZE="${TRAIN_POPULATION_SIZE:-8000}"
IMAGE_SPLIT_SEED="${IMAGE_SPLIT_SEED:-0}"

BATCH_SIZE="${BATCH_SIZE:-256}"
HIDDEN_DIM="${HIDDEN_DIM:-64}"
EVAL_SAMPLES="${EVAL_SAMPLES:-2000}"
FID_SAMPLES="${FID_SAMPLES:-2000}"
FID_GEN_BATCH="${FID_GEN_BATCH:-64}"
DEBUG_EVAL_BATCH="${DEBUG_EVAL_BATCH:-64}"
DEBUG_TERMINAL_STEP="${DEBUG_TERMINAL_STEP:-20}"
LOG_EVERY="${LOG_EVERY:-200}"
N_STEPS_PATH="${N_STEPS_PATH:-64}"
USE_EMA_EVAL="${USE_EMA_EVAL:-1}"
EMA_MODE="${EMA_MODE:-official}"
EMA_DECAY="${EMA_DECAY:-0.999}"
EMA_HALFLIFE_KIMG="${EMA_HALFLIFE_KIMG:-500}"
EMA_RAMPUP_RATIO="${EMA_RAMPUP_RATIO:-0.05}"
DISABLE_EMA_RAMPUP="${DISABLE_EMA_RAMPUP:-0}"

GRID_TEMPLATE="${GRID_TEMPLATE:-denser}"
FID_EVAL_TEMPLATE="${FID_EVAL_TEMPLATE:-balanced}"
BASELINE_FID_MODE="${BASELINE_FID_MODE:-posthoc_from_checkpoints}"
ROBUST_FID_MODE="${ROBUST_FID_MODE:-posthoc_from_checkpoints}"
SHARED_WEIGHTED_CAP="${SHARED_WEIGHTED_CAP:-400000}"
BASELINE_MAX_STEPS="${BASELINE_MAX_STEPS:-160000}"
WDRO_MAX_TOTAL_STEPS="${WDRO_MAX_TOTAL_STEPS:-160000}"

INNER_STEPS="${INNER_STEPS:-1}"
OUTER_ATTACK_WEIGHT="${OUTER_ATTACK_WEIGHT:-1.0}"
OUTER_CLEAN_WEIGHT="${OUTER_CLEAN_WEIGHT:-0.0}"
WDRO_WARMUP_FRACTION="${WDRO_WARMUP_FRACTION:-0.2}"
CDRO_STEP_SIZE="${CDRO_STEP_SIZE:-0.02}"
CDRO_TOTAL_BUDGET_RHO="${CDRO_TOTAL_BUDGET_RHO:-0.001}"
CDRO_TIME_HORIZON="${CDRO_TIME_HORIZON:-1.0}"
CDRO_EDM_LADDER_MODE="${CDRO_EDM_LADDER_MODE:-stochastic_stratified_quantile}"
CDRO_WARMUP_FRACTION="${CDRO_WARMUP_FRACTION:-0.2}"
CDRO_N_STEPS_PATH="${CDRO_N_STEPS_PATH:-32}"

mkdir -p "${OUTDIR}"

if [[ "${PYTHON_BIN}" == */* ]]; then
  if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "[error] missing python binary: ${PYTHON_BIN}" >&2
    exit 1
  fi
elif ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[error] missing python binary on PATH: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -d "${DATASET_PATH}" ]]; then
  echo "[error] missing dataset path: ${DATASET_PATH}" >&2
  exit 1
fi
if [[ ! -d "${DATASET_VAL_PATH}" ]]; then
  echo "[error] missing dataset val path: ${DATASET_VAL_PATH}" >&2
  exit 1
fi
if [[ ! -f "${FID_REF_PATH}" ]]; then
  echo "[error] missing FID ref: ${FID_REF_PATH}" >&2
  exit 1
fi
if [[ ! -f "${CALIBRATION_PATH}" ]]; then
  echo "[error] missing weighted compute calibration: ${CALIBRATION_PATH}" >&2
  exit 1
fi

for pct in ${PERCENTS}; do
  if ! [[ "${pct}" =~ ^[0-9]+$ ]]; then
    echo "[error] percent must be an integer, got: ${pct}" >&2
    exit 1
  fi
  if (( pct <= 0 || pct > 100 )); then
    echo "[error] percent must be in 1..100, got: ${pct}" >&2
    exit 1
  fi

  train_size=$(( TRAIN_POPULATION_SIZE * pct / 100 ))
  if (( train_size <= 0 )); then
    echo "[error] resolved train size is zero for pct=${pct}" >&2
    exit 1
  fi

  pct_tag="$(printf "%03d" "${pct}")"
  run_outdir="${OUTDIR}/${pct_tag}pct"
  run_prefix="${PREFIX_ROOT}_${pct_tag}pct"
  extra_args=()
  if [[ "${USE_EMA_EVAL}" == "1" ]]; then
    extra_args+=(--use-ema-eval --ema-mode "${EMA_MODE}")
    if [[ "${EMA_MODE}" == "fixed" ]]; then
      extra_args+=(--ema-decay "${EMA_DECAY}")
    else
      extra_args+=(--ema-halflife-kimg "${EMA_HALFLIFE_KIMG}")
      if [[ "${DISABLE_EMA_RAMPUP}" == "1" ]]; then
        extra_args+=(--disable-ema-rampup)
      else
        extra_args+=(--ema-rampup-ratio "${EMA_RAMPUP_RATIO}")
      fi
    fi
  fi
  echo "[locked-default-sweep] pct=${pct}% train_size=${train_size} outdir=${run_outdir}"

  "${PYTHON_BIN}" "${ROOT_DIR}/toy/scripts/collect_three_method_seed_data.py" \
    --outdir "${run_outdir}" \
    --prefix "${run_prefix}" \
    --python-bin "${PYTHON_BIN}" \
    --seeds "${SEEDS}" \
    --device "${DEVICE}" \
    --dataset-path "${DATASET_PATH}" \
    --dataset-val-path "${DATASET_VAL_PATH}" \
    --fid-ref-path "${FID_REF_PATH}" \
    --weighted-compute-calibration-path "${CALIBRATION_PATH}" \
    --image-size "${IMAGE_SIZE}" \
    --image-channels "${IMAGE_CHANNELS}" \
    --image-train-size "${train_size}" \
    --image-val-size "${IMAGE_VAL_SIZE}" \
    --train-percent-label "${pct}%" \
    --image-split-seed "${IMAGE_SPLIT_SEED}" \
    --batch-size "${BATCH_SIZE}" \
    --hidden-dim "${HIDDEN_DIM}" \
    --eval-samples "${EVAL_SAMPLES}" \
    --fid-samples "${FID_SAMPLES}" \
    --fid-gen-batch "${FID_GEN_BATCH}" \
    --debug-eval-batch "${DEBUG_EVAL_BATCH}" \
    --debug-terminal-step "${DEBUG_TERMINAL_STEP}" \
    --log-every "${LOG_EVERY}" \
    --n-steps-path "${N_STEPS_PATH}" \
    --grid-template "${GRID_TEMPLATE}" \
    --fid-eval-template "${FID_EVAL_TEMPLATE}" \
    --baseline-fid-mode "${BASELINE_FID_MODE}" \
    --robust-fid-mode "${ROBUST_FID_MODE}" \
    --shared-weighted-cap "${SHARED_WEIGHTED_CAP}" \
    --baseline-max-steps "${BASELINE_MAX_STEPS}" \
    --wdro-max-total-steps "${WDRO_MAX_TOTAL_STEPS}" \
    --inner-steps "${INNER_STEPS}" \
    --outer-attack-weight "${OUTER_ATTACK_WEIGHT}" \
    --outer-clean-weight "${OUTER_CLEAN_WEIGHT}" \
    --wdro-warmup-fraction "${WDRO_WARMUP_FRACTION}" \
    --cdro-step-size "${CDRO_STEP_SIZE}" \
    --cdro-total-budget-rho "${CDRO_TOTAL_BUDGET_RHO}" \
    --cdro-time-horizon "${CDRO_TIME_HORIZON}" \
    --cdro-edm-ladder-mode "${CDRO_EDM_LADDER_MODE}" \
    --cdro-warmup-fraction "${CDRO_WARMUP_FRACTION}" \
    --cdro-n-steps-path "${CDRO_N_STEPS_PATH}" \
    "${extra_args[@]}"
done

echo "[locked-default-sweep] done: ${OUTDIR}"
