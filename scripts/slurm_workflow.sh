#!/bin/bash
#SBATCH --job-name=wd_compare
#SBATCH --output=toy_outputs/slurm_%j.out
#SBATCH --error=toy_outputs/slurm_%j.err
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

echo "[*] SLURM Job started on $(hostname)"
echo "[*] Current directory: $(pwd)"

# Source user's provided conda script
source /home/quanth/working_space/scripts/conda.sh
# Conda script cd's into /home/quanth, so we must return to Wild-Diffusion
cd /home/quanth/working_space/Wild-Diffusion

# Create outputs dir if not exists
mkdir -p toy_outputs

export CUDA_VISIBLE_DEVICES=0
export FID_DETECTOR_PATH=/mnt/data/quanth/models/inception-2015-12-05.pkl

echo "=== 0/4: Generating MNIST Clean Reference Statistics ==="
python3 toy/export_mnist_fid_ref.py

# Re-run the bash script (It uses MPS, but SLURM cluster usually uses CUDA. Let's override device to CUDA in script)
echo "=== 1/4: Running Proposed v2 with EDM (CUDA) ==="
python3 toy/run_toy.py --exp-name=v2_edm --dataset-kind=mnist --image-channels=1 --device=cuda --steps=2000 --batch-size=128 --image-train-size=2000 --image-val-size=500 --eval-samples=2000 --compute-fid --method-version=v2 --training-objective=edm --skip-checks

echo "=== 2/4: Running Proposed v2 with Score Matching (CUDA) ==="
python3 toy/run_toy.py --exp-name=v2_score --dataset-kind=mnist --image-channels=1 --device=cuda --steps=2000 --batch-size=128 --image-train-size=2000 --image-val-size=500 --eval-samples=2000 --compute-fid --method-version=v2 --training-objective=score --skip-checks

echo "=== 3/4: Running WILD-Diffusion with EDM (CUDA) ==="
python3 toy/run_toy.py --exp-name=wild_edm --dataset-kind=mnist --image-channels=1 --device=cuda --steps=2000 --batch-size=128 --image-train-size=2000 --image-val-size=500 --eval-samples=2000 --compute-fid --method-version=wild --training-objective=edm --skip-checks

echo "=== 4/4: Running WILD-Diffusion with Score Matching (CUDA) ==="
python3 toy/run_toy.py --exp-name=wild_score --dataset-kind=mnist --image-channels=1 --device=cuda --steps=2000 --batch-size=128 --image-train-size=2000 --image-val-size=500 --eval-samples=2000 --compute-fid --method-version=wild --training-objective=score --skip-checks

echo "=== Generating Comparison Plots ==="
python3 toy/compare_robustness.py

echo "[*] SLURM Job finished."
