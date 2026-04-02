#!/bin/bash
#SBATCH --job-name=mnist_fid_curve_4m
#SBATCH --output=toy_outputs/slurm_fid_curve_4m_%j.out
#SBATCH --error=toy_outputs/slurm_fid_curve_4m_%j.err
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G

set -euo pipefail

echo "[*] SLURM Job started on $(hostname)"
echo "[*] Current directory: $(pwd)"
echo "[*] SLURM_JOB_ID=${SLURM_JOB_ID:-N/A} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-N/A}"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "[ERROR] nvidia-smi not found"
  exit 1
fi

source /home/quanth/working_space/scripts/conda.sh
cd /home/quanth/working_space/Wild-Diffusion

export CUDA_VISIBLE_DEVICES=0
export FID_DETECTOR_PATH=/mnt/data/quanth/models/inception-2015-12-05.pkl

python3 - <<'PY'
import torch
print('[torch] cuda_available:', torch.cuda.is_available())
print('[torch] device_count:', torch.cuda.device_count())
if not torch.cuda.is_available() or torch.cuda.device_count() <= 0:
    raise SystemExit('[ERROR] CUDA is required for this workflow, but no GPU is visible.')
print('[torch] current_device:', torch.cuda.current_device())
print('[torch] device_name:', torch.cuda.get_device_name(torch.cuda.current_device()))
PY

SEED="${SEED:-0}"
STEPS_LIST="${STEPS_LIST:-1000,2000,4000,8000,12000,16000,20000}"
METHODS="${METHODS:-baseline,wild,v1.1,v1.2}"
BATCH_SIZE="${BATCH_SIZE:-32}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
EVAL_SAMPLES="${EVAL_SAMPLES:-2000}"
FID_SAMPLES="${FID_SAMPLES:-2000}"
MNIST_TRAIN_PERCENT="${MNIST_TRAIN_PERCENT:-20}"
MNIST_VAL_PERCENT="${MNIST_VAL_PERCENT:-100}"
OUTDIR="${OUTDIR:-/mnt/data/quanth/wild_diffusion_outputs/mnist_fid_curve_4methods}"
PREFIX="${PREFIX:-mnist_${MNIST_TRAIN_PERCENT}pct_s${SEED}_fidcurve}"
BASELINE_CKPT_SOURCE="${BASELINE_CKPT_SOURCE:-/mnt/data/quanth/models/wild_diffusion_baseline_ckpt/baseline_edm_s${SEED}_st20000_bs${BATCH_SIZE}_hd${HIDDEN_DIM}_tp${MNIST_TRAIN_PERCENT}.pt}"

SIGMA_MIN="${SIGMA_MIN:-0.01}"
SIGMA_MAX="${SIGMA_MAX:-80}"
AUTO_LOG_NORMAL="${AUTO_LOG_NORMAL:-1}"
USE_EMA_EVAL="${USE_EMA_EVAL:-1}"
EMA_DECAY="${EMA_DECAY:-0.999}"
WARMUP_CLEAN_STEPS="${WARMUP_CLEAN_STEPS:-900}"
WARMUP_RAMP_STEPS="${WARMUP_RAMP_STEPS:-600}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

# WILD
WILD_UPDATE_INTERVAL="${WILD_UPDATE_INTERVAL:-20}"
WILD_CACHE_BATCHES="${WILD_CACHE_BATCHES:-4}"
WILD_INNER_STEPS="${WILD_INNER_STEPS:-3}"
WILD_STEP_SIZE="${WILD_STEP_SIZE:-0.05}"
WILD_GAMMA="${WILD_GAMMA:-2.0}"
WILD_FIXED_NOISE_INNER="${WILD_FIXED_NOISE_INNER:-1}"
WILD_CLAMP_SAMPLES="${WILD_CLAMP_SAMPLES:-0}"
WILD_SAMPLE_MIN="${WILD_SAMPLE_MIN:--1.0}"
WILD_SAMPLE_MAX="${WILD_SAMPLE_MAX:-1.0}"
WILD_DELTA_RATIO_DENOM="${WILD_DELTA_RATIO_DENOM:-1.0}"

# v1.1
V11_STEP_SIZE="${V11_STEP_SIZE:-0.0005}"
V11_GAMMA="${V11_GAMMA:-2.0}"
V11_RHO="${V11_RHO:-0.02}"
V11_PROJ_MODE="${V11_PROJ_MODE:-global_remaining}"

# v1.2
V12_ADV_STEPS="${V12_ADV_STEPS:-1}"
V12_STEP_SIZE="${V12_STEP_SIZE:-0.02}"
V12_LAMBDA_INIT="${V12_LAMBDA_INIT:-0.1}"
V12_LAMBDA_LR="${V12_LAMBDA_LR:-0.001}"
V12_RHO_TARGET="${V12_RHO_TARGET:-0.0001}"
V12_ROBUST_MIX="${V12_ROBUST_MIX:-0.3}"
V12_START_STEP="${V12_START_STEP:-0}"
V12_RAMP_STEPS="${V12_RAMP_STEPS:-0}"
V12_MAX_DELTA="${V12_MAX_DELTA:-0.05}"
V12_SIGMA_FLOOR="${V12_SIGMA_FLOOR:-0.0}"
V12_SIGMA_CUT="${V12_SIGMA_CUT:-0.5}"
V12_GATE_POWER="${V12_GATE_POWER:-2.0}"
V12_DELTA_SPACE="${V12_DELTA_SPACE:-image}"

echo "=== FID curve config ==="
echo "SEED=${SEED}"
echo "STEPS_LIST=${STEPS_LIST}"
echo "METHODS=${METHODS}"
echo "BATCH_SIZE=${BATCH_SIZE} HIDDEN_DIM=${HIDDEN_DIM}"
echo "OUTDIR=${OUTDIR} PREFIX=${PREFIX}"
echo "BASELINE_CKPT_SOURCE=${BASELINE_CKPT_SOURCE}"
echo "MNIST_TRAIN_PERCENT=${MNIST_TRAIN_PERCENT} MNIST_VAL_PERCENT=${MNIST_VAL_PERCENT}"
echo "SIGMA_MIN=${SIGMA_MIN} SIGMA_MAX=${SIGMA_MAX}"
echo "WARMUP_CLEAN_STEPS=${WARMUP_CLEAN_STEPS} WARMUP_RAMP_STEPS=${WARMUP_RAMP_STEPS}"

CMD=(
  python3 toy/scripts/compare_fid_curve_methods.py
  --outdir "${OUTDIR}"
  --prefix "${PREFIX}"
  --seed "${SEED}"
  --steps-list "${STEPS_LIST}"
  --methods "${METHODS}"
  --baseline-ckpt-source "${BASELINE_CKPT_SOURCE}"
  --dataset-kind mnist
  --image-channels 1
  --image-size 32
  --mnist-train-percent "${MNIST_TRAIN_PERCENT}"
  --mnist-val-percent "${MNIST_VAL_PERCENT}"
  --batch-size "${BATCH_SIZE}"
  --hidden-dim "${HIDDEN_DIM}"
  --training-objective edm
  --n-steps-path 24
  --sigma-min "${SIGMA_MIN}"
  --sigma-max "${SIGMA_MAX}"
  --eval-samples "${EVAL_SAMPLES}"
  --fid-samples "${FID_SAMPLES}"
  --warmup-clean-steps "${WARMUP_CLEAN_STEPS}"
  --warmup-ramp-steps "${WARMUP_RAMP_STEPS}"
  --device cuda
  --require-cuda
  --wild-update-interval "${WILD_UPDATE_INTERVAL}"
  --wild-cache-batches "${WILD_CACHE_BATCHES}"
  --wild-inner-steps "${WILD_INNER_STEPS}"
  --wild-step-size "${WILD_STEP_SIZE}"
  --wild-gamma "${WILD_GAMMA}"
  --wild-sample-min "${WILD_SAMPLE_MIN}"
  --wild-sample-max "${WILD_SAMPLE_MAX}"
  --wild-delta-ratio-denom "${WILD_DELTA_RATIO_DENOM}"
  --v11-step-size "${V11_STEP_SIZE}"
  --v11-transport-gamma "${V11_GAMMA}"
  --v11-total-budget-rho "${V11_RHO}"
  --v11-projection-mode "${V11_PROJ_MODE}"
  --v12-adv-steps "${V12_ADV_STEPS}"
  --v12-step-size "${V12_STEP_SIZE}"
  --v12-lambda-init "${V12_LAMBDA_INIT}"
  --v12-lambda-lr "${V12_LAMBDA_LR}"
  --v12-rho-target "${V12_RHO_TARGET}"
  --v12-robust-mix "${V12_ROBUST_MIX}"
  --v12-start-step "${V12_START_STEP}"
  --v12-ramp-steps "${V12_RAMP_STEPS}"
  --v12-max-delta "${V12_MAX_DELTA}"
  --v12-sigma-floor "${V12_SIGMA_FLOOR}"
  --v12-sigma-cut "${V12_SIGMA_CUT}"
  --v12-gate-power "${V12_GATE_POWER}"
  --v12-delta-space "${V12_DELTA_SPACE}"
)

if [[ "${AUTO_LOG_NORMAL}" == "1" ]]; then
  CMD+=(--auto-log-normal-params)
fi
if [[ "${USE_EMA_EVAL}" == "1" ]]; then
  CMD+=(--use-ema-eval --ema-decay "${EMA_DECAY}")
fi
if [[ "${SKIP_EXISTING}" == "1" ]]; then
  CMD+=(--skip-existing)
fi
if [[ "${WILD_FIXED_NOISE_INNER}" == "1" ]]; then
  CMD+=(--wild-fixed-noise-inner)
fi
if [[ "${WILD_CLAMP_SAMPLES}" == "1" ]]; then
  CMD+=(--wild-clamp-samples)
fi

echo "[cmd] ${CMD[*]}"
"${CMD[@]}"

echo "[*] Job finished."

