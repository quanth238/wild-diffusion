#!/bin/bash
#SBATCH --job-name=wd_compare
#SBATCH --output=toy_outputs/slurm_%j.out
#SBATCH --error=toy_outputs/slurm_%j.err
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

echo "[*] SLURM Job started on $(hostname)"
echo "[*] Current directory: $(pwd)"
echo "[*] SLURM_JOB_ID=${SLURM_JOB_ID:-N/A} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-N/A}"

echo "=== GPU Sanity Check (NVIDIA) ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "[!] nvidia-smi not found. Exiting."
  exit 1
fi

# Source user's provided conda script
source /home/quanth/working_space/scripts/conda.sh
# Conda script cd's into /home/quanth, so we must return to Wild-Diffusion
cd /home/quanth/working_space/Wild-Diffusion

python3 - <<'PY'
import torch
print("[torch] cuda_available:", torch.cuda.is_available())
print("[torch] device_count:", torch.cuda.device_count())
if not torch.cuda.is_available() or torch.cuda.device_count() <= 0:
    raise SystemExit("[ERROR] CUDA is required for this workflow, but no GPU is visible.")
print("[torch] current_device:", torch.cuda.current_device())
print("[torch] device_name:", torch.cuda.get_device_name(torch.cuda.current_device()))
PY

# Create outputs dir if not exists
mkdir -p toy_outputs

export CUDA_VISIBLE_DEVICES=0
export FID_DETECTOR_PATH=/mnt/data/quanth/models/inception-2015-12-05.pkl
export BASELINE_CKPT_DIR=toy_outputs/_baseline_cache_shared
mkdir -p "${BASELINE_CKPT_DIR}"

echo "=== 0/4: Generating MNIST Clean Reference Statistics ==="
python3 toy/export_mnist_fid_ref.py

COMMON_ARGS=(
  --dataset-kind=mnist
  --image-channels=1
  --device=cuda
  --steps=2000
  --batch-size=128
  --mnist-train-percent=20
  --mnist-val-percent=100
  --eval-samples=2000
  --compute-fid
  --fid-samples=2000
  --skip-checks
)

echo "=== 1/4: Running WILD baseline with EDM (CUDA) ==="
python3 toy/run_toy.py "${COMMON_ARGS[@]}" \
  --exp-name=wild_edm \
  --method-version=wild \
  --training-objective=edm \
  --baseline-ckpt-path="${BASELINE_CKPT_DIR}/mnist_edm_seed0.pt"

echo "=== 2/4: Running WILD baseline with Score Matching (CUDA) ==="
python3 toy/run_toy.py "${COMMON_ARGS[@]}" \
  --exp-name=wild_score \
  --method-version=wild \
  --training-objective=score \
  --baseline-ckpt-path="${BASELINE_CKPT_DIR}/mnist_score_seed0.pt"

echo "=== 3/4: Running Proposed v2 with EDM (CUDA) ==="
python3 toy/run_toy.py "${COMMON_ARGS[@]}" \
  --exp-name=v2_edm \
  --method-version=v2 \
  --training-objective=edm \
  --baseline-ckpt-path="${BASELINE_CKPT_DIR}/mnist_edm_seed0.pt"

echo "=== 4/4: Running Proposed v2 with Score Matching (CUDA) ==="
python3 toy/run_toy.py "${COMMON_ARGS[@]}" \
  --exp-name=v2_score \
  --method-version=v2 \
  --training-objective=score \
  --baseline-ckpt-path="${BASELINE_CKPT_DIR}/mnist_score_seed0.pt"

echo "=== Generating Comparison Plots ==="
python3 toy/compare_robustness.py

echo "[*] SLURM Job finished."
