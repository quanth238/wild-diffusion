#!/bin/bash
# Run four MNIST experiments:
# - Baseline family: wild_edm, wild_score
# - Proposed family: v2_edm, v2_score
# Then generate comparison plots/report.

set -euo pipefail
cd "$(dirname "$0")"

COMMON_ARGS=(
  --dataset-kind=mnist
  --image-channels=1
  --device=mps
  --steps=2000
  --batch-size=128
  --image-train-size=2000
  --image-val-size=500
  --eval-samples=2000
  --compute-fid
  --fid-samples=2000
  --skip-checks
)

echo "=== 1/4: Running WILD baseline with EDM ==="
python3 toy/run_toy.py "${COMMON_ARGS[@]}" --exp-name=wild_edm --method-version=wild --training-objective=edm

echo "=== 2/4: Running WILD baseline with Score Matching ==="
python3 toy/run_toy.py "${COMMON_ARGS[@]}" --exp-name=wild_score --method-version=wild --training-objective=score

echo "=== 3/4: Running Proposed v2 with EDM ==="
python3 toy/run_toy.py "${COMMON_ARGS[@]}" --exp-name=v2_edm --method-version=v2 --training-objective=edm

echo "=== 4/4: Running Proposed v2 with Score Matching ==="
python3 toy/run_toy.py "${COMMON_ARGS[@]}" --exp-name=v2_score --method-version=v2 --training-objective=score

echo "=== Generating Comparison Plots ==="
python3 toy/compare_robustness.py
