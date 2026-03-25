#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/toy-runs/cdro-sde-smoke-$(date +%Y%m%d-%H%M%S)}"

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

FORWARD_DIR="${OUT_ROOT}/forward"
REVERSE_DIR="${OUT_ROOT}/reverse"
SAMPLE_DIR="${OUT_ROOT}/sample"

run python -m toy_2d.train_cdro_sde \
  --outdir "${FORWARD_DIR}" \
  --dataset two_moons \
  --device cpu \
  --epochs 2 \
  --batch-size 64 \
  --num-samples 256 \
  --num-steps 6 \
  --adv-steps 1 \
  --eval-every 1 \
  --debug-points 64 \
  --num-plot-paths 16 \
  --replay-samples 256

run python -m toy_2d.train_cdro_reverse \
  --forward-run-dir "${FORWARD_DIR}" \
  --outdir "${REVERSE_DIR}" \
  --device cpu \
  --epochs 2 \
  --batch-size 64 \
  --eval-every 1 \
  --num-eval-samples 256 \
  --metric-samples 256 \
  --num-plot-paths 16

run python -m toy_2d.sample_cdro_reverse \
  --reverse-run-dir "${REVERSE_DIR}" \
  --outdir "${SAMPLE_DIR}" \
  --device cpu \
  --num-samples 256 \
  --metric-samples 256 \
  --num-plot-paths 16

echo
echo "[INFO] toy_2d CDRO SDE smoke test completed successfully."
echo "[INFO] Output directory: ${OUT_ROOT}"
