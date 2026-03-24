#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/toy-runs/smoke-$(date +%Y%m%d-%H%M%S)}"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "[ERROR] Virtualenv not found at ${VENV_DIR}. Run bash scripts/setup_toy_2d_env.sh first."
  exit 1
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

run() {
  echo
  echo "[INFO] $*"
  "$@"
}

run python - <<'PY'
import matplotlib
import numpy
import scipy
import torch

print(f"[INFO] torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
print(f"[INFO] numpy={numpy.__version__}")
print(f"[INFO] matplotlib={matplotlib.__version__}")
print(f"[INFO] scipy={scipy.__version__}")
PY

run python -m toy_2d.train_wild \
  --method baseline \
  --dataset two_moons \
  --device cpu \
  --epochs 2 \
  --eval-every 1 \
  --num-samples 256 \
  --num-eval-samples 256 \
  --metric-samples 256 \
  --batch-size 64 \
  --outdir "${OUT_ROOT}/baseline"

run python -m toy_2d.train_wild \
  --method wdro \
  --dataset two_moons \
  --device cpu \
  --epochs 2 \
  --eval-every 1 \
  --num-samples 256 \
  --num-eval-samples 256 \
  --metric-samples 256 \
  --batch-size 64 \
  --wdro-warmup-epochs 0 \
  --wdro-refresh-every 1 \
  --wdro-k 1 \
  --wdro-debug-points 64 \
  --wdro-p-adv 0.5 \
  --outdir "${OUT_ROOT}/wdro"

run python -m toy_2d.train_wild \
  --method cdro \
  --dataset two_moons \
  --device cpu \
  --epochs 2 \
  --eval-every 1 \
  --num-samples 256 \
  --num-eval-samples 256 \
  --metric-samples 256 \
  --batch-size 64 \
  --cdro-warmup-epochs 0 \
  --cdro-inner-steps 1 \
  --cdro-path-steps 4 \
  --cdro-debug-points 64 \
  --outdir "${OUT_ROOT}/cdro"

run python -m toy_2d.compare_methods \
  --outdir "${OUT_ROOT}/compare_methods" \
  --datasets two_moons \
  --fractions 0.2 \
  --methods baseline wdro cdro \
  --full-samples 256 \
  --seeds 0 \
  --workers 1 \
  --epochs 2 \
  --batch-size 64 \
  --eval-every 1 \
  --num-eval-samples 256 \
  --metric-samples 256 \
  --sampler-steps 8 \
  --wdro-k 1 \
  --wdro-warmup-epochs 0 \
  --wdro-refresh-every 1 \
  --cdro-warmup-epochs 0 \
  --cdro-inner-steps 1 \
  --cdro-path-steps 4

echo
echo "[INFO] toy_2d smoke test completed successfully."
echo "[INFO] Output directory: ${OUT_ROOT}"
