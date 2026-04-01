#!/bin/bash
#SBATCH --job-name=mnist_b_v2_v21
#SBATCH --output=toy_outputs/slurm_b_v2_v21_%j.out
#SBATCH --error=toy_outputs/slurm_b_v2_v21_%j.err
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
# Locked tuned config for fair baseline/v2/v2.1 comparison.
STEPS="20000"
BATCH_SIZE="128"
HIDDEN_DIM="256"
EVAL_SAMPLES="2000"
FID_SAMPLES="2000"
MNIST_TRAIN_PERCENT="100"
MNIST_VAL_PERCENT="100"
OUTDIR="${OUTDIR:-/mnt/data/quanth/wild_diffusion_outputs/mnist_baseline_v2_v21}"
PREFIX="${PREFIX:-mnist_fullpct_s${SEED}_t${STEPS}}"
V21_RHO="${V21_RHO:-0.8}"
PICK_SEED="${PICK_SEED:-7}"
SIGMA_MIN="0.01"
SIGMA_MAX="80"
AUTO_LOG_NORMAL="1"
USE_EMA_EVAL="1"
EMA_DECAY="0.999"

export SEED
export STEPS
export MNIST_TRAIN_PERCENT
export MNIST_VAL_PERCENT
export OUTDIR
export PREFIX

mkdir -p "${OUTDIR}"
BASELINE_CKPT_DIR="${BASELINE_CKPT_DIR:-/mnt/data/quanth/models/wild_diffusion_baseline_ckpt}"
mkdir -p "${BASELINE_CKPT_DIR}"
BASELINE_CKPT_PATH="${BASELINE_CKPT_DIR}/baseline_edm_s${SEED}_st${STEPS}_bs${BATCH_SIZE}_hd${HIDDEN_DIM}_tp${MNIST_TRAIN_PERCENT}.pt"

echo "=== Locked tuned config ==="
echo "SEED=${SEED}"
echo "STEPS=${STEPS} BATCH_SIZE=${BATCH_SIZE} HIDDEN_DIM=${HIDDEN_DIM}"
echo "SIGMA_MIN=${SIGMA_MIN} SIGMA_MAX=${SIGMA_MAX} AUTO_LOG_NORMAL=${AUTO_LOG_NORMAL}"
echo "USE_EMA_EVAL=${USE_EMA_EVAL} EMA_DECAY=${EMA_DECAY}"
echo "MNIST_TRAIN_PERCENT=${MNIST_TRAIN_PERCENT} MNIST_VAL_PERCENT=${MNIST_VAL_PERCENT}"
echo "OUTDIR=${OUTDIR} PREFIX=${PREFIX}"
echo "BASELINE_CKPT_DIR=${BASELINE_CKPT_DIR}"
echo "BASELINE_CKPT_PATH=${BASELINE_CKPT_PATH}"

echo "=== Ensure MNIST FID reference ==="
python3 toy/export_mnist_fid_ref.py

COMMON_ARGS=(
  --dataset-kind mnist
  --model-kind image_conv
  --diagnostics-kind image_basic
  --image-channels 1
  --image-size 32
  --mnist-train-percent "${MNIST_TRAIN_PERCENT}"
  --mnist-val-percent "${MNIST_VAL_PERCENT}"
  --device cuda
  --training-objective edm
  --steps "${STEPS}"
  --batch-size "${BATCH_SIZE}"
  --hidden-dim "${HIDDEN_DIM}"
  --n-steps-path 24
  --sigma-min "${SIGMA_MIN}"
  --sigma-max "${SIGMA_MAX}"
  --eval-samples "${EVAL_SAMPLES}"
  --compute-fid
  --fid-samples "${FID_SAMPLES}"
  --skip-checks
  --disable-baseline-gate
  --seed "${SEED}"
  --outdir "${OUTDIR}"
)

if [[ "${AUTO_LOG_NORMAL}" == "1" ]]; then
  COMMON_ARGS+=(--auto-log-normal-params)
fi

if [[ "${USE_EMA_EVAL}" == "1" ]]; then
  COMMON_ARGS+=(--use-ema-eval --ema-decay "${EMA_DECAY}")
fi

BASELINE_EXP="${PREFIX}_baseline_s${SEED}"
V2_EXP="${PREFIX}_v2_s${SEED}"
V21_EXP="${PREFIX}_2_1_s${SEED}"

echo "=== 1/4 Baseline EDM (baseline-only) ==="
python3 toy/run_toy.py "${COMMON_ARGS[@]}" \
  --exp-name "${BASELINE_EXP}" \
  --method-version v2 \
  --baseline-ckpt-path "${BASELINE_CKPT_PATH}" \
  --baseline-only

echo "=== 2/4 v2 robust ==="
python3 toy/run_toy.py "${COMMON_ARGS[@]}" \
  --exp-name "${V2_EXP}" \
  --method-version v2 \
  --baseline-ckpt-path "${BASELINE_CKPT_PATH}"

echo "=== 3/4 v2.1 robust ==="
python3 toy/run_toy.py "${COMMON_ARGS[@]}" \
  --exp-name "${V21_EXP}" \
  --method-version 2.1 \
  --baseline-ckpt-path "${BASELINE_CKPT_PATH}" \
  --v21-rho "${V21_RHO}"

echo "=== 4/4 Build panel (Real | Baseline EDM | v2 | v2.1) ==="
python3 toy/scripts/make_v2_v21_panel.py \
  --outdir "${OUTDIR}" \
  --prefix "${PREFIX}" \
  --seed "${SEED}" \
  --baseline-dir "${OUTDIR}/${BASELINE_EXP}/fid_baseline" \
  --v2-dir "${OUTDIR}/${V2_EXP}/fid_robust" \
  --v21-dir "${OUTDIR}/${V21_EXP}/fid_robust" \
  --rows 8 \
  --cols 8 \
  --tile-size 32 \
  --tile-pad 2 \
  --pick-seed "${PICK_SEED}" \
  --output "${OUTDIR}/${PREFIX}_panel_real_baseline_v2_v21_s${SEED}.png"

python3 - <<'PY'
import json
import os
from pathlib import Path

outdir = Path(os.environ['OUTDIR'])
prefix = os.environ['PREFIX']
seed = os.environ['SEED']

baseline_exp = f"{prefix}_baseline_s{seed}"
v2_exp = f"{prefix}_v2_s{seed}"
v21_exp = f"{prefix}_2_1_s{seed}"

def load_fid(exp: str):
    p = outdir / exp / 'metrics.json'
    m = json.loads(p.read_text())
    q = m['metrics']['sample_quality_debug']
    return float(q['baseline_fid']), float(q['robust_fid'])

b_base, b_rob = load_fid(baseline_exp)
v2_base, v2_rob = load_fid(v2_exp)
v21_base, v21_rob = load_fid(v21_exp)

summary = {
    'seed': int(seed),
    'steps': int(os.environ['STEPS']),
    'mnist_train_percent': float(os.environ['MNIST_TRAIN_PERCENT']),
    'mnist_val_percent': float(os.environ['MNIST_VAL_PERCENT']),
    'baseline_only': {'exp': baseline_exp, 'baseline_fid': b_base, 'robust_fid': b_rob},
    'v2': {'exp': v2_exp, 'baseline_fid': v2_base, 'robust_fid': v2_rob},
    'v2_1': {'exp': v21_exp, 'baseline_fid': v21_base, 'robust_fid': v21_rob},
    'panel': str(outdir / f"{prefix}_panel_real_baseline_v2_v21_s{seed}.png"),
}
(outdir / f"{prefix}_summary_baseline_v2_v21_s{seed}.json").write_text(json.dumps(summary, indent=2))

print('[done] summary:', outdir / f"{prefix}_summary_baseline_v2_v21_s{seed}.json")
print('[done] panel:', outdir / f"{prefix}_panel_real_baseline_v2_v21_s{seed}.png")
PY

echo "[*] Finished baseline-v2-v2.1 panel workflow"
