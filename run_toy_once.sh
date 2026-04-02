#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_MODE="${RUN_MODE:-robust}"   # robust|baseline
EXP_NAME="${EXP_NAME:-toy_once}"
OUTDIR="${OUTDIR:-toy_outputs}"
DEVICE="${DEVICE:-auto}"         # auto|cpu|cuda
SEED="${SEED:-0}"
BASELINE_CKPT_PATH="${BASELINE_CKPT_PATH:-}"

STEPS="${STEPS:-3000}"
BATCH_SIZE="${BATCH_SIZE:-512}"
LOG_EVERY="${LOG_EVERY:-500}"

KAPPA="${KAPPA:-0.15}"
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

mkdir -p "${OUTDIR}"

args=(
  "--exp-name" "${EXP_NAME}"
  "--outdir" "${OUTDIR}"
  "--device" "${DEVICE}"
  "--seed" "${SEED}"
  "--steps" "${STEPS}"
  "--batch-size" "${BATCH_SIZE}"
  "--log-every" "${LOG_EVERY}"
)

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

echo "[run_toy_once] mode=${RUN_MODE} exp=${EXP_NAME} outdir=${OUTDIR} device=${DEVICE} seed=${SEED}"
echo "[run_toy_once] command: ${PYTHON_BIN} toy/run_toy.py ${args[*]}"
"${PYTHON_BIN}" toy/run_toy.py "${args[@]}"

echo "[run_toy_once] done: ${OUTDIR}/${EXP_NAME}"
echo "[run_toy_once] metrics: ${OUTDIR}/${EXP_NAME}/metrics.json"
