#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTDIR="${OUTDIR:-toy_outputs_stageA}"
PREFIX="${PREFIX:-stageA_once}"
SEEDS="${SEEDS:-0,1,2}"
DEVICE="${DEVICE:-cpu}"

STEPS="${STEPS:-3000}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
LOG_EVERY="${LOG_EVERY:-500}"

KAPPA="${KAPPA:-0.3}"
OUTER_CLEAN_WEIGHT="${OUTER_CLEAN_WEIGHT:-1.0}"
OUTER_ATTACK_WEIGHT="${OUTER_ATTACK_WEIGHT:-0.5}"
WARMUP_CLEAN_STEPS="${WARMUP_CLEAN_STEPS:-900}"
WARMUP_RAMP_STEPS="${WARMUP_RAMP_STEPS:-600}"

RUN_CHECKS="${RUN_CHECKS:-1}"       # 1 or 0
BASELINE_GATE_HARD_FAIL="${BASELINE_GATE_HARD_FAIL:-1}"  # 1 or 0

# Evaluation thresholds (override if needed).
ATTACK_WIN_RATIO_MIN="${ATTACK_WIN_RATIO_MIN:-0.6}"
ATTACK_MEAN_GAP_MAX="${ATTACK_MEAN_GAP_MAX:-0.0}"
CLEAN_MEAN_GAP_MAX="${CLEAN_MEAN_GAP_MAX:-0.03}"
CLEAN_TERMINAL_GAP_MAX="${CLEAN_TERMINAL_GAP_MAX:-0.15}"
SAMPLE_AVG_DELTA_MAX="${SAMPLE_AVG_DELTA_MAX:-0.03}"
SAMPLE_P90_DELTA_MAX="${SAMPLE_P90_DELTA_MAX:-0.05}"
TAIL_SATURATION_MAX="${TAIL_SATURATION_MAX:-0.995}"
MIN_SEED_PASS_RATIO="${MIN_SEED_PASS_RATIO:-0.67}"

if [[ "${RUN_CHECKS}" == "1" ]]; then
  CHECK_FLAG="--run-checks"
else
  CHECK_FLAG="--skip-checks"
fi

if [[ "${BASELINE_GATE_HARD_FAIL}" == "1" ]]; then
  GATE_FLAG="--baseline-gate-error-on-fail"
else
  GATE_FLAG=""
fi

kappa_tag="${KAPPA//./p}"
IFS=',' read -r -a seed_list <<< "${SEEDS}"

mkdir -p "${OUTDIR}"

for raw_seed in "${seed_list[@]}"; do
  seed="$(echo "${raw_seed}" | xargs)"
  if [[ -z "${seed}" ]]; then
    continue
  fi

  baseline_exp="${PREFIX}_baseline_s${seed}"
  robust_exp="${PREFIX}_robust_k${kappa_tag}_s${seed}"

  echo "[stageA] seed=${seed} baseline_exp=${baseline_exp}"
  "${PYTHON_BIN}" toy/run_toy.py \
    --exp-name "${baseline_exp}" \
    --outdir "${OUTDIR}" \
    --seed "${seed}" \
    --device "${DEVICE}" \
    --baseline-only \
    --steps "${STEPS}" \
    --batch-size "${BATCH_SIZE}" \
    --log-every "${LOG_EVERY}" \
    ${GATE_FLAG} \
    ${CHECK_FLAG}

  echo "[stageA] seed=${seed} robust_exp=${robust_exp}"
  "${PYTHON_BIN}" toy/run_toy.py \
    --exp-name "${robust_exp}" \
    --outdir "${OUTDIR}" \
    --seed "${seed}" \
    --device "${DEVICE}" \
    --steps "${STEPS}" \
    --batch-size "${BATCH_SIZE}" \
    --log-every "${LOG_EVERY}" \
    --control-radius-kappa "${KAPPA}" \
    --outer-clean-weight "${OUTER_CLEAN_WEIGHT}" \
    --outer-attack-weight "${OUTER_ATTACK_WEIGHT}" \
    --warmup-clean-steps "${WARMUP_CLEAN_STEPS}" \
    --warmup-ramp-steps "${WARMUP_RAMP_STEPS}" \
    ${GATE_FLAG} \
    ${CHECK_FLAG}
done

echo "[stageA] evaluating baseline vs robust across seeds=${SEEDS}"
"${PYTHON_BIN}" toy/scripts/evaluate_stageA.py \
  --outdir "${OUTDIR}" \
  --prefix "${PREFIX}" \
  --seeds "${SEEDS}" \
  --kappa "${KAPPA}" \
  --attack-win-ratio-min "${ATTACK_WIN_RATIO_MIN}" \
  --attack-mean-gap-max "${ATTACK_MEAN_GAP_MAX}" \
  --clean-mean-gap-max "${CLEAN_MEAN_GAP_MAX}" \
  --clean-terminal-gap-max "${CLEAN_TERMINAL_GAP_MAX}" \
  --sample-avg-delta-max "${SAMPLE_AVG_DELTA_MAX}" \
  --sample-p90-delta-max "${SAMPLE_P90_DELTA_MAX}" \
  --tail-saturation-max "${TAIL_SATURATION_MAX}" \
  --min-seed-pass-ratio "${MIN_SEED_PASS_RATIO}"

summary_txt="${OUTDIR}/${PREFIX}_summary_k${kappa_tag}.txt"
summary_json="${OUTDIR}/${PREFIX}_summary_k${kappa_tag}.json"
echo "[stageA] done"
echo "[stageA] summary_txt=${summary_txt}"
echo "[stageA] summary_json=${summary_json}"
