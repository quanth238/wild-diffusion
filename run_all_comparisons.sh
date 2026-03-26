#!/bin/bash
# Script to run all 4 settings and generate the comparison plot

echo "=== 1/4: Running Proposed v2 with EDM ==="
python3 toy/run_toy.py --exp-name=v2_edm --dataset-kind=mnist --image-channels=1 --device=mps --steps=2000 --batch-size=128 --image-train-size=2000 --image-val-size=500 --eval-samples=2000 --compute-fid --method-version=v2 --training-objective=edm --skip-checks

echo "=== 2/4: Running Proposed v2 with Score Matching ==="
python3 toy/run_toy.py --exp-name=v2_score --dataset-kind=mnist --image-channels=1 --device=mps --steps=2000 --batch-size=128 --image-train-size=2000 --image-val-size=500 --eval-samples=2000 --compute-fid --method-version=v2 --training-objective=score --skip-checks

echo "=== 3/4: Running WILD-Diffusion with EDM ==="
python3 toy/run_toy.py --exp-name=wild_edm --dataset-kind=mnist --image-channels=1 --device=mps --steps=2000 --batch-size=128 --image-train-size=2000 --image-val-size=500 --eval-samples=2000 --compute-fid --method-version=wild --training-objective=edm --skip-checks

echo "=== 4/4: Running WILD-Diffusion with Score Matching ==="
python3 toy/run_toy.py --exp-name=wild_score --dataset-kind=mnist --image-channels=1 --device=mps --steps=2000 --batch-size=128 --image-train-size=2000 --image-val-size=500 --eval-samples=2000 --compute-fid --method-version=wild --training-objective=score --skip-checks

echo "=== Generating Comparison Plots ==="
python3 toy/compare_robustness.py
