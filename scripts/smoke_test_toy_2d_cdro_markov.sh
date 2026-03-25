#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/toy-runs/cdro-markov-smoke-$(date +%Y%m%d-%H%M%S)}"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "[ERROR] Virtualenv not found at ${VENV_DIR}. Run bash scripts/setup_toy_2d_env.sh first."
  exit 1
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

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

run python -m toy_2d.train_cdro_markov \
  --outdir "${OUT_ROOT}/train" \
  --dataset two_moons \
  --device cpu \
  --epochs 3 \
  --eval-every 1 \
  --num-samples 256 \
  --num-eval-samples 256 \
  --metric-samples 256 \
  --batch-size 64 \
  --num-steps 6 \
  --warmup-epochs 1 \
  --adversary-steps 1 \
  --score-steps 2 \
  --score-hidden-dim 64 \
  --control-hidden-dim 32 \
  --embedding-dim 16 \
  --control-scale 0.5 \
  --beta-min 0.2 \
  --beta-max 4.0 \
  --control-radius 0.05 \
  --lambda-min 0.02 \
  --lambda-lr 0.02 \
  --num-snapshot-steps 4

echo
echo "[INFO] toy_2d Markov CDRO smoke test completed successfully."
echo "[INFO] Output directory: ${OUT_ROOT}"
