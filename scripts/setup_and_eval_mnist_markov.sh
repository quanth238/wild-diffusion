#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"
DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/datasets}"
MNIST_DIR="${MNIST_DIR:-${DATA_ROOT}/mnist-32x32-train}"
REF_DATA="${REF_DATA:-${MNIST_DIR}}"
REF_PATH="${REF_PATH:-${DATA_ROOT}/fid-refs/mnist-32x32-train.npz}"
RUN_DIR="${RUN_DIR:-}"
NETWORK_PKL="${NETWORK_PKL:-}"
EVAL_ROOT="${EVAL_ROOT:-}"
EVAL_TAG="${EVAL_TAG:-eval-$(date +%Y%m%d-%H%M%S)}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
NUM_IMAGES="${NUM_IMAGES:-10000}"
SEED_START="${SEED_START:-0}"
GEN_BATCH="${GEN_BATCH:-64}"
FID_BATCH="${FID_BATCH:-64}"
CLASS_IDX="${CLASS_IDX:-}"
REVERSE_CONTROL_SCALE="${REVERSE_CONTROL_SCALE:-}"
REVERSE_NOISE_SCALE="${REVERSE_NOISE_SCALE:-}"

if [[ ! -d "${VENV_DIR}" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

if [[ -z "${NETWORK_PKL}" ]]; then
  if [[ -z "${RUN_DIR}" ]]; then
    echo "[ERROR] Please set NETWORK_PKL or RUN_DIR" >&2
    exit 1
  fi
  NETWORK_PKL="$(find "${RUN_DIR}" -maxdepth 1 -name 'network-snapshot-*.pkl' | sort | tail -n 1)"
fi
if [[ -z "${RUN_DIR}" ]]; then
  RUN_DIR="$(dirname "${NETWORK_PKL}")"
fi
if [[ ! -f "${NETWORK_PKL}" ]]; then
  echo "[ERROR] NETWORK_PKL not found: ${NETWORK_PKL}" >&2
  exit 1
fi

if [[ -z "${EVAL_ROOT}" ]]; then
  EVAL_ROOT="${RUN_DIR}/eval_markov"
fi
EVAL_DIR="${EVAL_ROOT}/${EVAL_TAG}"
SAMPLES_DIR="${EVAL_DIR}/samples"
mkdir -p "${EVAL_DIR}" "${SAMPLES_DIR}" "$(dirname "${REF_PATH}")"

if [[ ! -f "${REF_PATH}" ]]; then
  python fid.py ref --data "${REF_DATA}" --dest "${REF_PATH}" --batch "${FID_BATCH}"
fi

SEED_END=$((SEED_START + NUM_IMAGES - 1))
gen_cmd=(
  torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" generate_markov.py
  "--network=${NETWORK_PKL}"
  "--outdir=${SAMPLES_DIR}"
  "--seeds=${SEED_START}-${SEED_END}"
  "--batch=${GEN_BATCH}"
)
if [[ -n "${CLASS_IDX}" ]]; then
  gen_cmd+=("--class=${CLASS_IDX}")
fi
if [[ -n "${REVERSE_CONTROL_SCALE}" ]]; then
  gen_cmd+=("--reverse-control-scale=${REVERSE_CONTROL_SCALE}")
fi
if [[ -n "${REVERSE_NOISE_SCALE}" ]]; then
  gen_cmd+=("--noise-scale=${REVERSE_NOISE_SCALE}")
fi

"${gen_cmd[@]}"
if [[ "${NPROC_PER_NODE}" == "1" ]]; then
  fid_value="$(python fid.py calc --images "${SAMPLES_DIR}" --ref "${REF_PATH}" --num "${NUM_IMAGES}" --seed 0 --batch "${FID_BATCH}" | tail -n 1)"
else
  fid_value="$(torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" fid.py calc --images "${SAMPLES_DIR}" --ref "${REF_PATH}" --num "${NUM_IMAGES}" --seed 0 --batch "${FID_BATCH}" | tail -n 1)"
fi

export RUN_DIR NETWORK_PKL REF_PATH NUM_IMAGES SEED_START SEED_END GEN_BATCH FID_BATCH fid_value EVAL_DIR
python3 - <<'PY'
import json
import os
from pathlib import Path

payload = {
    "dataset": "mnist-32x32",
    "run_dir": os.environ["RUN_DIR"],
    "network_pkl": os.environ["NETWORK_PKL"],
    "ref_path": os.environ["REF_PATH"],
    "num_images": int(os.environ["NUM_IMAGES"]),
    "seed_start": int(os.environ["SEED_START"]),
    "seed_end": int(os.environ["SEED_END"]),
    "gen_batch": int(os.environ["GEN_BATCH"]),
    "fid_batch": int(os.environ["FID_BATCH"]),
    "fid": float(os.environ["fid_value"]),
}
out = Path(os.environ["EVAL_DIR"]) / "evaluation_result.json"
out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(out)
PY
