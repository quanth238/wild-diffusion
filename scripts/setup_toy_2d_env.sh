#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"
INSTALL_TORCH="${INSTALL_TORCH:-1}"
TORCH_INSTALL_SPEC="${TORCH_INSTALL_SPEC:-torch}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[ERROR] ${PYTHON_BIN} not found. Set PYTHON_BIN to a working Python 3 executable."
  exit 1
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "[INFO] Creating virtualenv at ${VENV_DIR}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

echo "[INFO] Upgrading pip/setuptools/wheel"
python -m pip install --upgrade pip setuptools wheel

echo "[INFO] Installing toy_2d Python dependencies"
python -m pip install -r "${ROOT_DIR}/requirements-toy2d.txt"

if [[ "${INSTALL_TORCH}" == "1" ]]; then
  read -r -a torch_specs <<< "${TORCH_INSTALL_SPEC}"
  echo "[INFO] Installing PyTorch: ${TORCH_INSTALL_SPEC}"
  if [[ -n "${TORCH_INDEX_URL}" ]]; then
    python -m pip install --index-url "${TORCH_INDEX_URL}" "${torch_specs[@]}"
  else
    python -m pip install "${torch_specs[@]}"
  fi
else
  echo "[INFO] INSTALL_TORCH=0, skipping PyTorch installation"
fi

python - <<'PY'
import importlib

required = ["torch", "numpy", "matplotlib", "PIL", "scipy"]
missing = []
for name in required:
    try:
        importlib.import_module(name)
    except Exception as err:
        missing.append(f"{name}: {err}")

if missing:
    raise SystemExit("[ERROR] Missing modules after setup:\n" + "\n".join(missing))

import matplotlib
import numpy
import scipy
import torch

print(f"[INFO] Python: {__import__('sys').version.split()[0]}")
print(f"[INFO] torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
print(f"[INFO] numpy={numpy.__version__}")
print(f"[INFO] matplotlib={matplotlib.__version__}")
print(f"[INFO] scipy={scipy.__version__}")
PY

echo "[INFO] Environment ready."
echo "[INFO] Activate with: source ${VENV_DIR}/bin/activate"
echo "[INFO] Smoke test with: bash scripts/smoke_test_toy_2d.sh"
