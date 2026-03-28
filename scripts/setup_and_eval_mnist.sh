#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
SERVER_STORAGE_ROOT="${SERVER_STORAGE_ROOT:-/mnt/data/quanth}"
if [[ -d "${SERVER_STORAGE_ROOT}" && -w "${SERVER_STORAGE_ROOT}" ]]; then
  DEFAULT_VENV_DIR="${SERVER_STORAGE_ROOT}/venvs/wild-diffusion"
  DEFAULT_DATA_ROOT="${SERVER_STORAGE_ROOT}/datasets"
else
  DEFAULT_VENV_DIR="${ROOT_DIR}/.venv"
  DEFAULT_DATA_ROOT="${ROOT_DIR}/datasets"
fi

VENV_DIR="${VENV_DIR:-${DEFAULT_VENV_DIR}}"
DATA_ROOT="${DATA_ROOT:-${DEFAULT_DATA_ROOT}}"
MNIST_DIR="${MNIST_DIR:-${DATA_ROOT}/mnist-32x32-train}"
REF_DATA="${REF_DATA:-${MNIST_DIR}}"
REF_PATH="${REF_PATH:-${DATA_ROOT}/fid-refs/mnist-32x32-train.npz}"

ENV_MODE="${ENV_MODE:-auto}"              # auto|conda|venv
INSTALL_DEPS="${INSTALL_DEPS:-auto}"      # auto|0|1
EXPECTED_CONDA_ENV="${EXPECTED_CONDA_ENV:-quanth}"
STRICT_CONDA_ENV="${STRICT_CONDA_ENV:-1}"
INSTALL_ONLY="${INSTALL_ONLY:-0}"

PREPARE_MNIST="${PREPARE_MNIST:-auto}"    # auto|0|1
MNIST_RGB="${MNIST_RGB:-0}"
MNIST_FORCE_PREPARE="${MNIST_FORCE_PREPARE:-0}"

RUN_DIR="${RUN_DIR:-}"
NETWORK_PKL="${NETWORK_PKL:-}"
RESUME_STATE="${RESUME_STATE:-}"
EVAL_ROOT="${EVAL_ROOT:-}"
EVAL_TAG="${EVAL_TAG:-eval-$(date +%Y%m%d-%H%M%S)}"

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
NUM_IMAGES="${NUM_IMAGES:-10000}"
SEED_START="${SEED_START:-0}"
GEN_BATCH="${GEN_BATCH:-128}"
FID_BATCH="${FID_BATCH:-64}"
GEN_STEPS="${GEN_STEPS:-35}"
CLASS_IDX="${CLASS_IDX:-}"
REF_MODE="${REF_MODE:-auto}"  # auto|compute|path

mkdir -p "${DATA_ROOT}"
export DATA_ROOT MNIST_DIR

if [[ "${ENV_MODE}" == "auto" ]]; then
  if [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
    ENV_MODE="conda"
  else
    ENV_MODE="venv"
  fi
fi

if [[ "${ENV_MODE}" == "conda" ]]; then
  if [[ -z "${CONDA_DEFAULT_ENV:-}" ]]; then
    echo "[ERROR] ENV_MODE=conda but no active conda environment was found."
    echo "Activate your conda env first (expected: ${EXPECTED_CONDA_ENV}), or set ENV_MODE=venv."
    exit 1
  fi
  if [[ -n "${EXPECTED_CONDA_ENV}" && "${CONDA_DEFAULT_ENV}" != "${EXPECTED_CONDA_ENV}" ]]; then
    if [[ "${STRICT_CONDA_ENV}" == "1" ]]; then
      echo "[ERROR] Active conda env is '${CONDA_DEFAULT_ENV}', expected '${EXPECTED_CONDA_ENV}'."
      echo "Set EXPECTED_CONDA_ENV or STRICT_CONDA_ENV=0 if this is intentional."
      exit 1
    else
      echo "[WARN] Active conda env is '${CONDA_DEFAULT_ENV}', expected '${EXPECTED_CONDA_ENV}'."
    fi
  fi
  echo "[INFO] ENV_MODE=conda (active env: ${CONDA_DEFAULT_ENV})"
elif [[ "${ENV_MODE}" == "venv" ]]; then
  if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "[ERROR] ${PYTHON_BIN} not found. Set PYTHON_BIN to a valid Python 3 executable."
    exit 1
  fi
  mkdir -p "$(dirname "${VENV_DIR}")"
  if [[ ! -d "${VENV_DIR}" ]]; then
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  fi
  # shellcheck disable=SC1090
  source "${VENV_DIR}/bin/activate"
  echo "[INFO] ENV_MODE=venv"
else
  echo "[ERROR] ENV_MODE must be one of: auto, conda, venv"
  exit 1
fi

if ! command -v python >/dev/null 2>&1; then
  echo "[ERROR] python is not available in the current environment."
  exit 1
fi

if [[ "${INSTALL_DEPS}" == "auto" ]]; then
  if python - <<'PY'
modules = ["torch", "torchvision", "click", "numpy", "psutil", "scipy", "tqdm", "requests"]
missing = []
for name in modules:
    try:
        __import__(name)
    except Exception:
        missing.append(name)
try:
    from PIL import Image  # noqa: F401
except Exception:
    missing.append("PIL")
raise SystemExit(1 if missing else 0)
PY
  then
    INSTALL_DEPS="0"
  else
    INSTALL_DEPS="1"
  fi
fi

if [[ "${INSTALL_DEPS}" == "1" ]]; then
  echo "[INFO] INSTALL_DEPS=1 (installing/refreshing Python packages)"
  python -m pip install --upgrade pip setuptools wheel
  python -m pip install --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}" torch torchvision
  python -m pip install click numpy pillow psutil scipy tqdm requests
else
  echo "[INFO] INSTALL_DEPS=0 (reusing packages from current environment)"
fi

if [[ "${INSTALL_ONLY}" == "1" ]]; then
  echo "[INFO] INSTALL_ONLY=1, dependency setup completed. Exiting before evaluation."
  exit 0
fi

if [[ "${PREPARE_MNIST}" == "auto" ]]; then
  if [[ -d "${MNIST_DIR}" && -f "${MNIST_DIR}/dataset.json" ]]; then
    PREPARE_MNIST="0"
  else
    PREPARE_MNIST="1"
  fi
fi

if [[ "${PREPARE_MNIST}" == "1" ]]; then
  echo "[INFO] Preparing MNIST dataset at ${MNIST_DIR}"
  prep_cmd=(
    python scripts/prepare_mnist_dataset.py
    "--data-root=${DATA_ROOT}"
    "--raw-root=${DATA_ROOT}/raw"
    "--splits=train"
    "--resolution=32"
  )
  if [[ "${MNIST_RGB}" == "1" ]]; then
    prep_cmd+=("--rgb")
  fi
  if [[ "${MNIST_FORCE_PREPARE}" == "1" ]]; then
    prep_cmd+=("--force")
  fi
  "${prep_cmd[@]}"
fi

python - <<PY
import os
import pathlib
import re

run_dir = os.environ.get("RUN_DIR", "").strip()
network_pkl = os.environ.get("NETWORK_PKL", "").strip()
resume_state = os.environ.get("RESUME_STATE", "").strip()

if network_pkl and not pathlib.Path(network_pkl).is_file():
    raise SystemExit(f"[ERROR] NETWORK_PKL not found: {network_pkl}")

if not network_pkl and resume_state:
    p = pathlib.Path(resume_state)
    if not p.is_file():
        raise SystemExit(f"[ERROR] RESUME_STATE not found: {resume_state}")
    m = re.fullmatch(r"training-state-(\\d+)\\.pt", p.name)
    if not m:
        raise SystemExit("[ERROR] RESUME_STATE must match training-state-*.pt")
    candidate = p.parent / f"network-snapshot-{m.group(1)}.pkl"
    if not candidate.is_file():
        raise SystemExit(f"[ERROR] Matching snapshot not found for RESUME_STATE: {candidate}")
    print(f"[INFO] Resolved NETWORK_PKL from RESUME_STATE: {candidate}")
    network_pkl = str(candidate)

if not network_pkl:
    if not run_dir:
        raise SystemExit("[ERROR] Please set one of: NETWORK_PKL, RESUME_STATE, or RUN_DIR")
    run = pathlib.Path(run_dir)
    if not run.is_dir():
        raise SystemExit(f"[ERROR] RUN_DIR not found: {run_dir}")
    snaps = sorted(run.glob("network-snapshot-*.pkl"))
    if not snaps:
        raise SystemExit(f"[ERROR] No network-snapshot-*.pkl found in RUN_DIR: {run_dir}")
    network_pkl = str(snaps[-1])
    print(f"[INFO] Resolved latest NETWORK_PKL from RUN_DIR: {network_pkl}")

if not run_dir:
    run_dir = str(pathlib.Path(network_pkl).resolve().parent)

print(f'RUN_DIR=\"{run_dir}\"')
print(f'NETWORK_PKL=\"{network_pkl}\"')
PY

eval "$(python - <<PY
import os
import pathlib
import re

run_dir = os.environ.get("RUN_DIR", "").strip()
network_pkl = os.environ.get("NETWORK_PKL", "").strip()
resume_state = os.environ.get("RESUME_STATE", "").strip()

if not network_pkl and resume_state:
    p = pathlib.Path(resume_state)
    m = re.fullmatch(r"training-state-(\\d+)\\.pt", p.name)
    network_pkl = str(p.parent / f"network-snapshot-{m.group(1)}.pkl")
if not network_pkl and run_dir:
    snaps = sorted(pathlib.Path(run_dir).glob("network-snapshot-*.pkl"))
    network_pkl = str(snaps[-1])
if not run_dir:
    run_dir = str(pathlib.Path(network_pkl).resolve().parent)
print(f'RUN_DIR=\"{run_dir}\"')
print(f'NETWORK_PKL=\"{network_pkl}\"')
PY
)"
export RUN_DIR NETWORK_PKL

if [[ -z "${EVAL_ROOT}" ]]; then
  EVAL_ROOT="${RUN_DIR}/eval"
fi
EVAL_DIR="${EVAL_ROOT}/${EVAL_TAG}"
SAMPLES_DIR="${EVAL_DIR}/samples"
mkdir -p "${EVAL_DIR}" "${SAMPLES_DIR}" "$(dirname "${REF_PATH}")"

if [[ "${REF_MODE}" == "auto" ]]; then
  if [[ -f "${REF_PATH}" ]]; then
    REF_MODE="path"
  else
    REF_MODE="compute"
  fi
fi

if [[ "${REF_MODE}" == "compute" ]]; then
  echo "[INFO] Computing MNIST reference stats at ${REF_PATH}"
  python fid.py ref \
    --data "${REF_DATA}" \
    --dest "${REF_PATH}" \
    --batch "${FID_BATCH}"
fi

if [[ ! -f "${REF_PATH}" ]]; then
  echo "[ERROR] Reference stats file not found: ${REF_PATH}"
  exit 1
fi

SEED_END=$((SEED_START + NUM_IMAGES - 1))
gen_cmd=(
  torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" generate.py
  "--network=${NETWORK_PKL}"
  "--outdir=${SAMPLES_DIR}"
  "--seeds=${SEED_START}-${SEED_END}"
  "--subdirs"
  "--batch=${GEN_BATCH}"
  "--steps=${GEN_STEPS}"
)
if [[ -n "${CLASS_IDX}" ]]; then
  gen_cmd+=("--class=${CLASS_IDX}")
fi
"${gen_cmd[@]}"

FID_VALUE="$(
  python fid.py calc \
    --images "${SAMPLES_DIR}" \
    --ref "${REF_PATH}" \
    --num "${NUM_IMAGES}" \
    --batch "${FID_BATCH}" | tail -n 1
)"

python - <<PY
import json
from pathlib import Path

eval_dir = Path("${EVAL_DIR}")
payload = {
    "dataset": "mnist-32x32",
    "run_dir": "${RUN_DIR}",
    "network_pkl": "${NETWORK_PKL}",
    "ref_path": "${REF_PATH}",
    "num_images": int("${NUM_IMAGES}"),
    "seed_start": int("${SEED_START}"),
    "seed_end": int("${SEED_END}"),
    "gen_steps": int("${GEN_STEPS}"),
    "gen_batch": int("${GEN_BATCH}"),
    "fid_batch": int("${FID_BATCH}"),
    "fid": float("${FID_VALUE}"),
}
(eval_dir / "evaluation_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
(eval_dir / "evaluation_result.txt").write_text(
    "\\n".join([
        "MNIST Evaluation Result",
        f"run_dir: ${RUN_DIR}",
        f"network_pkl: ${NETWORK_PKL}",
        f"ref_path: ${REF_PATH}",
        f"num_images: ${NUM_IMAGES}",
        f"fid: ${FID_VALUE}",
    ]) + "\\n",
    encoding="utf-8",
)
print(json.dumps(payload, indent=2))
PY

echo "[INFO] Evaluation finished. Artifacts written to ${EVAL_DIR}"
