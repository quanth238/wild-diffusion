#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"

OUTDIR="${OUTDIR:-cifar_small_outputs}"
PREFIX="${PREFIX:-cifar_small_once}"
SEEDS="${SEEDS:-0}"
RUN_ORDER="${RUN_ORDER:-baseline_first}" # baseline_first|robust_first

ENV_MODE="${ENV_MODE:-auto}"
INSTALL_DEPS="${INSTALL_DEPS:-auto}"
EXPECTED_CONDA_ENV="${EXPECTED_CONDA_ENV:-quanth}"
STRICT_CONDA_ENV="${STRICT_CONDA_ENV:-1}"

SERVER_STORAGE_ROOT="${SERVER_STORAGE_ROOT:-/mnt/data/quanth}"
DATA_ROOT="${DATA_ROOT:-${SERVER_STORAGE_ROOT}/datasets}"
CIFAR_DIR="${CIFAR_DIR:-${DATA_ROOT}/cifar10-32x32}"
FID_DETECTOR_PATH="${FID_DETECTOR_PATH:-}"

DURATION_MIMG="${DURATION_MIMG:-8}"
BATCH="${BATCH:-1024}"
BATCH_GPU="${BATCH_GPU:-512}"
FP16="${FP16:-1}"
LR="${LR:-1e-4}"
WORKERS="${WORKERS:-16}"

CIFAR_TRAIN_PERCENT="${CIFAR_TRAIN_PERCENT:-20}"
CIFAR_TRAIN_SEED="${CIFAR_TRAIN_SEED:-0}"

WDRO_WARMUP_RATIO="${WDRO_WARMUP_RATIO:-0.2}"
WDRO_M_EPOCHS="${WDRO_M_EPOCHS:-80}"
WDRO_K="${WDRO_K:-2}"
WDRO_STEP_SIZE="${WDRO_STEP_SIZE:-0.01}"
WDRO_GAMMA="${WDRO_GAMMA:-1.0}"
BASELINE_P_ADV="${BASELINE_P_ADV:-0.0}"
ROBUST_P_ADV="${ROBUST_P_ADV:-0.3}"

DEBUG_EVAL="${DEBUG_EVAL:-1}"
DEBUG_EVAL_INIT="${DEBUG_EVAL_INIT:-1}"
DEBUG_EVAL_NUM="${DEBUG_EVAL_NUM:-128}"
DEBUG_EVAL_STEPS="${DEBUG_EVAL_STEPS:-18}"
DEBUG_EVAL_BATCH="${DEBUG_EVAL_BATCH:-64}"
DEBUG_EVAL_VISUAL="${DEBUG_EVAL_VISUAL:-32}"
DEBUG_ADV_VISUAL="${DEBUG_ADV_VISUAL:-16}"

NUM_IMAGES="${NUM_IMAGES:-5000}"
GEN_BATCH="${GEN_BATCH:-128}"
FID_BATCH="${FID_BATCH:-64}"
GEN_STEPS="${GEN_STEPS:-18}"
REF_MODE="${REF_MODE:-path}"
REF_PATH="${REF_PATH:-${DATA_ROOT}/fid-refs/cifar10-32x32.npz}"
EVAL_TAG_PREFIX="${EVAL_TAG_PREFIX:-final_eval}"

ROBUST_WIN_RATIO_MIN="${ROBUST_WIN_RATIO_MIN:-0.5}"
MEAN_FID_GAP_MAX="${MEAN_FID_GAP_MAX:-0.0}"
MIN_SEED_PASS_RATIO="${MIN_SEED_PASS_RATIO:-0.5}"

IFS=',' read -r -a seed_list <<< "${SEEDS}"

mkdir -p "${OUTDIR}"

resolve_single_run_dir() {
  local run_root="$1"
  find "${run_root}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1
}

run_train() {
  local mode="$1"
  local seed="$2"
  local p_adv="$3"
  local train_outdir="$4"

  echo "[cifar-small] mode=${mode} seed=${seed} train_outdir=${train_outdir}"
  env \
    ENV_MODE="${ENV_MODE}" \
    INSTALL_DEPS="${INSTALL_DEPS}" \
    EXPECTED_CONDA_ENV="${EXPECTED_CONDA_ENV}" \
    STRICT_CONDA_ENV="${STRICT_CONDA_ENV}" \
    DATASET_ONLY=0 \
    CIFAR_ALLOW_DOWNLOAD=0 \
    DATA_ROOT="${DATA_ROOT}" \
    CIFAR_DIR="${CIFAR_DIR}" \
    OUTDIR="${train_outdir}" \
    FID_DETECTOR_PATH="${FID_DETECTOR_PATH}" \
    SEED="${seed}" \
    DURATION_MIMG="${DURATION_MIMG}" \
    BATCH="${BATCH}" \
    BATCH_GPU="${BATCH_GPU}" \
    FP16="${FP16}" \
    LR="${LR}" \
    WORKERS="${WORKERS}" \
    CIFAR_TRAIN_PERCENT="${CIFAR_TRAIN_PERCENT}" \
    CIFAR_TRAIN_SEED="${CIFAR_TRAIN_SEED}" \
    WDRO_WARMUP_RATIO="${WDRO_WARMUP_RATIO}" \
    WDRO_M_EPOCHS="${WDRO_M_EPOCHS}" \
    WDRO_K="${WDRO_K}" \
    WDRO_STEP_SIZE="${WDRO_STEP_SIZE}" \
    WDRO_GAMMA="${WDRO_GAMMA}" \
    WDRO_P_ADV="${p_adv}" \
    DEBUG_EVAL="${DEBUG_EVAL}" \
    DEBUG_EVAL_INIT="${DEBUG_EVAL_INIT}" \
    DEBUG_EVAL_NUM="${DEBUG_EVAL_NUM}" \
    DEBUG_EVAL_STEPS="${DEBUG_EVAL_STEPS}" \
    DEBUG_EVAL_BATCH="${DEBUG_EVAL_BATCH}" \
    DEBUG_EVAL_VISUAL="${DEBUG_EVAL_VISUAL}" \
    DEBUG_ADV_VISUAL="${DEBUG_ADV_VISUAL}" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    bash scripts/setup_and_train_cifar10.sh
}

run_eval() {
  local mode="$1"
  local seed="$2"
  local run_dir="$3"
  local eval_tag="${EVAL_TAG_PREFIX}_${mode}_s${seed}"

  echo "[cifar-small] mode=${mode} seed=${seed} eval_run_dir=${run_dir}"
  env \
    ENV_MODE="${ENV_MODE}" \
    INSTALL_DEPS="${INSTALL_DEPS}" \
    EXPECTED_CONDA_ENV="${EXPECTED_CONDA_ENV}" \
    STRICT_CONDA_ENV="${STRICT_CONDA_ENV}" \
    DATA_ROOT="${DATA_ROOT}" \
    CIFAR_DIR="${CIFAR_DIR}" \
    FID_DETECTOR_PATH="${FID_DETECTOR_PATH}" \
    RUN_DIR="${run_dir}" \
    EVAL_TAG="${eval_tag}" \
    NUM_IMAGES="${NUM_IMAGES}" \
    GEN_BATCH="${GEN_BATCH}" \
    FID_BATCH="${FID_BATCH}" \
    GEN_STEPS="${GEN_STEPS}" \
    REF_MODE="${REF_MODE}" \
    REF_PATH="${REF_PATH}" \
    bash scripts/setup_and_eval_cifar10.sh
}

run_pair() {
  local mode="$1"
  local seed="$2"
  local p_adv="$3"
  local run_root="$4"

  run_train "${mode}" "${seed}" "${p_adv}" "${run_root}"
  local run_dir
  run_dir="$(resolve_single_run_dir "${run_root}")"
  if [[ -z "${run_dir}" || ! -d "${run_dir}" ]]; then
    echo "[ERROR] Could not resolve ${mode} run dir for seed=${seed}" >&2
    exit 1
  fi
  run_eval "${mode}" "${seed}" "${run_dir}"
}

for raw_seed in "${seed_list[@]}"; do
  seed="$(echo "${raw_seed}" | xargs)"
  if [[ -z "${seed}" ]]; then
    continue
  fi

  baseline_root="${OUTDIR}/${PREFIX}_baseline_s${seed}"
  robust_root="${OUTDIR}/${PREFIX}_robust_s${seed}"
  mkdir -p "${baseline_root}" "${robust_root}"

  case "${RUN_ORDER}" in
    robust_first)
      run_pair robust "${seed}" "${ROBUST_P_ADV}" "${robust_root}"
      run_pair baseline "${seed}" "${BASELINE_P_ADV}" "${baseline_root}"
      ;;
    baseline_first)
      run_pair baseline "${seed}" "${BASELINE_P_ADV}" "${baseline_root}"
      run_pair robust "${seed}" "${ROBUST_P_ADV}" "${robust_root}"
      ;;
    *)
      echo "[ERROR] RUN_ORDER must be one of: baseline_first, robust_first" >&2
      exit 1
      ;;
  esac
done

echo "[cifar-small] aggregating baseline vs robust summary"
"${PYTHON_BIN}" scripts/evaluate_cifar_small.py \
  --outdir "${OUTDIR}" \
  --prefix "${PREFIX}" \
  --seeds "${SEEDS}" \
  --robust-win-ratio-min "${ROBUST_WIN_RATIO_MIN}" \
  --mean-fid-gap-max "${MEAN_FID_GAP_MAX}" \
  --min-seed-pass-ratio "${MIN_SEED_PASS_RATIO}"

echo "[cifar-small] done"
