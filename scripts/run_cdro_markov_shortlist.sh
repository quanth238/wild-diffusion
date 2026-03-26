#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"

OUTDIR="${OUTDIR:-${ROOT_DIR}/toy-runs/cdro_markov_manual_shortlist_eight100_v1}"
LOG_DIR="${LOG_DIR:-${OUTDIR}/logs}"
DATASET="${DATASET:-eight_gaussians}"
NUM_SAMPLES="${NUM_SAMPLES:-2000}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-256}"
EVAL_EVERY="${EVAL_EVERY:-5}"
NUM_EVAL_SAMPLES="${NUM_EVAL_SAMPLES:-2048}"
METRIC_SAMPLES="${METRIC_SAMPLES:-1024}"
MAX_JOBS="${MAX_JOBS:-1}"
THREADS_PER_JOB="${THREADS_PER_JOB:-1}"

mkdir -p "${OUTDIR}"
mkdir -p "${LOG_DIR}"

BASE_ARGS=(
  -m toy_2d.train_cdro_markov
  --dataset "${DATASET}"
  --num-samples "${NUM_SAMPLES}"
  --seed "${SEED}"
  --outdir "${OUTDIR}"
  --device "${DEVICE}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --eval-every "${EVAL_EVERY}"
  --num-eval-samples "${NUM_EVAL_SAMPLES}"
  --metric-samples "${METRIC_SAMPLES}"
  --torch-num-threads "${THREADS_PER_JOB}"
  --fast-tuning
)

wait_for_slot() {
  while (( $(jobs -pr | wc -l) >= MAX_JOBS )); do
    wait -n
  done
}

run_cfg() {
  local name="$1"
  shift
  local log_path="${LOG_DIR}/${name}.log"
  wait_for_slot
  echo "[INFO] Launching ${name} -> ${log_path}"
  (
    export OMP_NUM_THREADS="${THREADS_PER_JOB}"
    export MKL_NUM_THREADS="${THREADS_PER_JOB}"
    export OPENBLAS_NUM_THREADS="${THREADS_PER_JOB}"
    export NUMEXPR_NUM_THREADS="${THREADS_PER_JOB}"
    "${PYTHON_BIN}" "${BASE_ARGS[@]}" --outdir "${OUTDIR}/${name}" "$@"
  ) >"${log_path}" 2>&1 &
}

run_cfg "fine16_inv" \
  --score-lr 0.001 \
  --control-lr 0.0002 \
  --lambda-lr 0.02 \
  --lambda-init 0.1 \
  --lambda-min 0.02 \
  --control-radius 0.05 \
  --warmup-epochs 8 \
  --adversary-steps 1 \
  --score-steps 8 \
  --terminal-momentum 0.95 \
  --num-steps 16 \
  --total-time 1.0 \
  --beta-min 0.2 \
  --beta-max 8.0 \
  --score-weight-schedule inv_sigma_sq \
  --score-hidden-dim 128 \
  --score-depth 4 \
  --control-arch mlp \
  --control-hidden-dim 64 \
  --control-depth 3 \
  --control-scale 0.5

run_cfg "fine16_big_inv_more_score" \
  --score-lr 0.001 \
  --control-lr 0.0002 \
  --lambda-lr 0.02 \
  --lambda-init 0.1 \
  --lambda-min 0.02 \
  --control-radius 0.05 \
  --warmup-epochs 8 \
  --adversary-steps 1 \
  --score-steps 12 \
  --terminal-momentum 0.95 \
  --num-steps 16 \
  --total-time 1.0 \
  --beta-min 0.2 \
  --beta-max 8.0 \
  --score-weight-schedule inv_sigma_sq \
  --score-hidden-dim 192 \
  --score-depth 4 \
  --control-arch mlp \
  --control-hidden-dim 96 \
  --control-depth 3 \
  --control-scale 0.5

run_cfg "fine16_big_inv_more_score_freeze150" \
  --score-lr 0.001 \
  --control-lr 0.0002 \
  --lambda-lr 0.02 \
  --lambda-init 0.1 \
  --lambda-min 0.02 \
  --control-radius 0.05 \
  --warmup-epochs 8 \
  --adversary-steps 1 \
  --adversary-stop-epoch 150 \
  --score-steps 12 \
  --terminal-momentum 0.95 \
  --num-steps 16 \
  --total-time 1.0 \
  --beta-min 0.2 \
  --beta-max 8.0 \
  --score-weight-schedule inv_sigma_sq \
  --score-hidden-dim 192 \
  --score-depth 4 \
  --control-arch mlp \
  --control-hidden-dim 96 \
  --control-depth 3 \
  --control-scale 0.5

run_cfg "fine16_big_inv_more_score_proxy_frontload" \
  --score-lr 0.001 \
  --control-lr 0.0002 \
  --lambda-lr 0.02 \
  --lambda-init 0.1 \
  --lambda-min 0.02 \
  --control-radius 0.05 \
  --budget-mode match_wdro_proxy \
  --reference-wdro-k 5 \
  --reference-wdro-step-size 0.05 \
  --reference-wdro-gamma 1.0 \
  --budget-estimate-batch-size 256 \
  --budget-scale 1.0 \
  --budget-schedule frontload \
  --budget-frontload-power 2.0 \
  --budget-frontload-floor 0.2 \
  --warmup-epochs 8 \
  --adversary-steps 1 \
  --score-steps 12 \
  --terminal-momentum 0.95 \
  --num-steps 16 \
  --total-time 1.0 \
  --beta-min 0.2 \
  --beta-max 8.0 \
  --score-weight-schedule inv_sigma_sq \
  --score-hidden-dim 192 \
  --score-depth 4 \
  --control-arch mlp \
  --control-hidden-dim 96 \
  --control-depth 3 \
  --control-scale 0.5

run_cfg "fine16_big_inv_more_score_proxy_frontload_x50_freeze150" \
  --score-lr 0.001 \
  --control-lr 0.0002 \
  --lambda-lr 0.02 \
  --lambda-init 0.1 \
  --lambda-min 0.02 \
  --control-radius 0.05 \
  --budget-mode match_wdro_proxy \
  --reference-wdro-k 5 \
  --reference-wdro-step-size 0.05 \
  --reference-wdro-gamma 1.0 \
  --budget-estimate-batch-size 256 \
  --budget-scale 50.0 \
  --budget-schedule frontload \
  --budget-frontload-power 2.0 \
  --budget-frontload-floor 0.2 \
  --warmup-epochs 8 \
  --adversary-steps 1 \
  --adversary-stop-epoch 150 \
  --score-steps 12 \
  --terminal-momentum 0.95 \
  --num-steps 16 \
  --total-time 1.0 \
  --beta-min 0.2 \
  --beta-max 8.0 \
  --score-weight-schedule inv_sigma_sq \
  --score-hidden-dim 192 \
  --score-depth 4 \
  --control-arch mlp \
  --control-hidden-dim 96 \
  --control-depth 3 \
  --control-scale 0.5

wait
echo "[INFO] All shortlist jobs finished."
