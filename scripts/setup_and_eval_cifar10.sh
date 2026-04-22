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
  DEFAULT_VENV_DIR="${HOME}/.venvs/wild-diffusion-h100"
  DEFAULT_DATA_ROOT="${ROOT_DIR}/datasets"
fi

VENV_DIR="${VENV_DIR:-${DEFAULT_VENV_DIR}}"
DATA_ROOT="${DATA_ROOT:-${DEFAULT_DATA_ROOT}}"
CIFAR_DIR="${CIFAR_DIR:-${DATA_ROOT}/cifar10-32x32}"
if [[ -z "${DNNLIB_CACHE_DIR:-}" && -d "${SERVER_STORAGE_ROOT}" && -w "${SERVER_STORAGE_ROOT}" ]]; then
  DNNLIB_CACHE_DIR="${SERVER_STORAGE_ROOT}/cache/dnnlib"
fi

ENV_MODE="${ENV_MODE:-auto}"              # auto|conda|venv
INSTALL_DEPS="${INSTALL_DEPS:-auto}"      # auto|0|1
EXPECTED_CONDA_ENV="${EXPECTED_CONDA_ENV:-quanth}"
STRICT_CONDA_ENV="${STRICT_CONDA_ENV:-1}" # 1|0
INSTALL_ONLY="${INSTALL_ONLY:-0}"         # 1: stop after env/dependency setup

CIFAR_ALLOW_DOWNLOAD="${CIFAR_ALLOW_DOWNLOAD:-0}"   # 1|0
CIFAR_DOWNLOAD_RETRIES="${CIFAR_DOWNLOAD_RETRIES:-3}"
CIFAR_DOWNLOAD_TIMEOUT="${CIFAR_DOWNLOAD_TIMEOUT:-30}" # seconds
PREPARE_DATASET="${PREPARE_DATASET:-auto}"          # auto|0|1

RUN_DIR="${RUN_DIR:-}"
NETWORK_PKL="${NETWORK_PKL:-}"
RESUME_STATE="${RESUME_STATE:-}"
EVAL_ROOT="${EVAL_ROOT:-}"
EVAL_TAG="${EVAL_TAG:-eval-$(date +%Y%m%d-%H%M%S)}"

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
NUM_IMAGES="${NUM_IMAGES:-50000}"
SEED_START="${SEED_START:-0}"
GEN_BATCH="${GEN_BATCH:-128}"
FID_BATCH="${FID_BATCH:-64}"
GEN_STEPS="${GEN_STEPS:-18}"
CLASS_IDX="${CLASS_IDX:-}" # optional
FID_BACKEND="${FID_BACKEND:-edm}" # edm|pytorch_fid
REUSE_GENERATED_SAMPLES="${REUSE_GENERATED_SAMPLES:-1}" # 1|0

REF_MODE="${REF_MODE:-auto}" # auto|compute|url|path
REF_URL="${REF_URL:-https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz}"
REF_PATH="${REF_PATH:-${DATA_ROOT}/fid-refs/cifar10-32x32.npz}"

EXTRA_GEN_ARGS="${EXTRA_GEN_ARGS:-}"
EXTRA_FID_ARGS="${EXTRA_FID_ARGS:-}"

mkdir -p "${DATA_ROOT}"
export DATA_ROOT CIFAR_DIR CIFAR_ALLOW_DOWNLOAD CIFAR_DOWNLOAD_RETRIES CIFAR_DOWNLOAD_TIMEOUT
if [[ -n "${DNNLIB_CACHE_DIR:-}" ]]; then
  mkdir -p "${DNNLIB_CACHE_DIR}/downloads"
  export DNNLIB_CACHE_DIR
fi

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

if [[ ! -r /proc/self/maps || ! -r /proc/cpuinfo || ! -r /proc/sys/vm/mmap_min_addr ]]; then
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/ensure_procfs_compat.sh"
fi

if [[ "${INSTALL_DEPS}" == "auto" ]]; then
  if [[ "${ENV_MODE}" == "conda" ]]; then
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

python - <<PY
import os
modules = ["torch", "torchvision", "click", "numpy", "psutil", "scipy", "tqdm", "requests"]
if os.environ.get("FID_BACKEND", "edm") == "pytorch_fid":
    modules.append("pytorch_fid")
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
if missing:
    raise SystemExit(
        "[ERROR] Missing Python modules in current environment: "
        + ", ".join(sorted(set(missing)))
        + ". Set INSTALL_DEPS=1 to install them automatically."
    )
PY

if [[ "${INSTALL_ONLY}" == "1" ]]; then
  echo "[INFO] INSTALL_ONLY=1, dependency setup completed. Exiting before evaluation."
  exit 0
fi

export RUN_DIR NETWORK_PKL RESUME_STATE
python - <<PY
import os
import pathlib
import re
import sys

run_dir = os.environ.get("RUN_DIR", "").strip()
network_pkl = os.environ.get("NETWORK_PKL", "").strip()
resume_state = os.environ.get("RESUME_STATE", "").strip()

if network_pkl and not pathlib.Path(network_pkl).is_file():
    raise SystemExit(f"[ERROR] NETWORK_PKL not found: {network_pkl}")

if not network_pkl and resume_state:
    p = pathlib.Path(resume_state)
    if not p.is_file():
        raise SystemExit(f"[ERROR] RESUME_STATE not found: {resume_state}")
    m = re.fullmatch(r"training-state-(\d+)\.pt", p.name)
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

print(f"RUN_DIR={run_dir}")
print(f"NETWORK_PKL={network_pkl}")
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
    m = re.fullmatch(r"training-state-(\d+)\.pt", p.name)
    network_pkl = str(p.parent / f"network-snapshot-{m.group(1)}.pkl")
if not network_pkl and run_dir:
    snaps = sorted(pathlib.Path(run_dir).glob("network-snapshot-*.pkl"))
    network_pkl = str(snaps[-1])
if not run_dir:
    run_dir = str(pathlib.Path(network_pkl).resolve().parent)
print(f'RUN_DIR="{run_dir}"')
print(f'NETWORK_PKL="{network_pkl}"')
PY
)"
export RUN_DIR NETWORK_PKL

if [[ -z "${EVAL_ROOT}" ]]; then
  EVAL_ROOT="${RUN_DIR}/eval"
fi
EVAL_DIR="${EVAL_ROOT}/${EVAL_TAG}"
SAMPLES_DIR="${EVAL_DIR}/samples"
mkdir -p "${EVAL_DIR}" "${SAMPLES_DIR}"

if [[ "${PREPARE_DATASET}" == "auto" ]]; then
  PREPARE_DATASET="$(
    python - <<'PY'
from pathlib import Path
import os
cifar = Path(os.environ["CIFAR_DIR"])
png_count = sum(1 for _ in cifar.rglob("*.png")) if cifar.exists() else 0
labels_ok = (cifar / "dataset.json").exists()
print("0" if (png_count >= 50000 and labels_ok) else "1")
PY
  )"
fi

if [[ "${PREPARE_DATASET}" == "1" ]]; then
  echo "[INFO] Preparing CIFAR-10 at ${CIFAR_DIR}"
  python - <<'PY'
import json
import os
import socket
import time
from pathlib import Path

from torchvision.datasets import CIFAR10

root = Path(os.environ["DATA_ROOT"])
out_dir = Path(os.environ["CIFAR_DIR"])
allow_download = os.environ.get("CIFAR_ALLOW_DOWNLOAD", "0") == "1"
download_retries = int(os.environ.get("CIFAR_DOWNLOAD_RETRIES", "3"))
download_timeout = int(os.environ.get("CIFAR_DOWNLOAD_TIMEOUT", "30"))
out_dir.mkdir(parents=True, exist_ok=True)

png_count = sum(1 for _ in out_dir.rglob("*.png"))
labels_path = out_dir / "dataset.json"
if png_count >= 50000 and labels_path.exists():
    print(f"[INFO] Reusing existing CIFAR-10 directory: {out_dir}")
    raise SystemExit(0)

socket.setdefaulttimeout(download_timeout)
ds = None
for attempt in range(1, download_retries + 1):
    try:
        print(f"[INFO] Loading CIFAR-10 from: {root}")
        ds = CIFAR10(root=str(root), train=True, download=False)
        break
    except Exception:
        if not allow_download:
            raise SystemExit(
                "[ERROR] CIFAR-10 not found locally and CIFAR_ALLOW_DOWNLOAD=0. "
                "Prepare dataset first or rerun with CIFAR_ALLOW_DOWNLOAD=1."
            )
        try:
            print(f"[INFO] Downloading CIFAR-10 to: {root} (attempt {attempt}/{download_retries})")
            ds = CIFAR10(root=str(root), train=True, download=True)
            break
        except Exception as err:
            if attempt == download_retries:
                raise SystemExit(
                    "[ERROR] Failed to download CIFAR-10 after retries.\n"
                    f"Last error: {err}"
                )
            print(f"[WARN] CIFAR download failed: {err}. Retrying...")
            time.sleep(min(5 * attempt, 15))

if ds is None:
    raise SystemExit("[ERROR] Internal error: CIFAR dataset object was not created.")

classes = ds.classes
labels = []
for idx, (img, label) in enumerate(ds):
    cls = classes[label]
    cls_dir = out_dir / cls
    cls_dir.mkdir(parents=True, exist_ok=True)
    rel_path = Path(cls) / f"{idx:05d}.png"
    abs_path = out_dir / rel_path
    img.save(abs_path)
    labels.append([str(rel_path).replace(os.sep, "/"), int(label)])

with open(labels_path, "w", encoding="utf-8") as f:
    json.dump({"labels": labels}, f)

print(f"[INFO] Wrote {len(labels)} images and labels to: {out_dir}")
PY
else
  echo "[INFO] PREPARE_DATASET=0 (skip CIFAR-10 preparation)"
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[ERROR] NVIDIA GPU not detected (nvidia-smi missing)."
  exit 1
fi

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("[ERROR] torch.cuda.is_available() is False. Install a CUDA-enabled PyTorch build.")
print(f"[INFO] CUDA is available with {torch.cuda.device_count()} GPU(s).")
PY

REF_INPUT=""
if [[ "${REF_MODE}" == "auto" ]]; then
  if [[ -f "${REF_PATH}" ]]; then
    REF_INPUT="${REF_PATH}"
  elif [[ -f "${CIFAR_DIR}/dataset.json" ]]; then
    REF_MODE="compute"
  elif [[ "${FID_BACKEND}" == "edm" ]]; then
    REF_MODE="url"
  else
    REF_MODE="compute"
  fi
fi

if [[ "${REF_MODE}" == "compute" ]]; then
  mkdir -p "$(dirname "${REF_PATH}")"
  if [[ "${FID_BACKEND}" == "pytorch_fid" && -f "${REF_PATH}" ]]; then
    echo "[INFO] Reusing existing pytorch-fid reference stats: ${REF_PATH}"
  elif [[ "${FID_BACKEND}" == "pytorch_fid" ]]; then
    ref_cmd=(
      python scripts/pytorch_fid_recursive.py save-stats
      "--images=${CIFAR_DIR}"
      "--dest=${REF_PATH}"
      "--device=cuda"
      "--batch-size=${FID_BATCH}"
    )
    echo "[INFO] Building local pytorch-fid reference stats: ${REF_PATH}"
    echo "[CMD] ${ref_cmd[*]}"
    "${ref_cmd[@]}" 2>&1 | tee "${EVAL_DIR}/ref.log"
  else
    ref_cmd=(
      torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" fid.py ref
      "--data=${CIFAR_DIR}"
      "--dest=${REF_PATH}"
      "--batch=${FID_BATCH}"
    )
    echo "[INFO] Building local reference stats: ${REF_PATH}"
    echo "[CMD] ${ref_cmd[*]}"
    "${ref_cmd[@]}" 2>&1 | tee "${EVAL_DIR}/ref.log"
  fi
  REF_INPUT="${REF_PATH}"
elif [[ "${REF_MODE}" == "path" ]]; then
  if [[ ! -f "${REF_PATH}" ]]; then
    echo "[ERROR] REF_MODE=path but REF_PATH not found: ${REF_PATH}"
    exit 1
  fi
  REF_INPUT="${REF_PATH}"
elif [[ "${REF_MODE}" == "url" ]]; then
  if [[ "${FID_BACKEND}" != "edm" ]]; then
    echo "[ERROR] REF_MODE=url is only supported with FID_BACKEND=edm"
    exit 1
  fi
  REF_INPUT="${REF_URL}"
else
  echo "[ERROR] REF_MODE must be one of: auto, compute, url, path"
  exit 1
fi

if [[ -z "${REF_INPUT}" ]]; then
  echo "[ERROR] Failed to resolve FID reference input."
  exit 1
fi

SEED_END=$((SEED_START + NUM_IMAGES - 1))
SEED_RANGE="${SEED_START}-${SEED_END}"

gen_cmd=(
  torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" generate.py
  "--network=${NETWORK_PKL}"
  "--outdir=${SAMPLES_DIR}"
  "--seeds=${SEED_RANGE}"
  "--batch=${GEN_BATCH}"
  "--steps=${GEN_STEPS}"
  "--subdirs"
)
if [[ -n "${CLASS_IDX}" ]]; then
  gen_cmd+=("--class=${CLASS_IDX}")
fi
if [[ -n "${EXTRA_GEN_ARGS}" ]]; then
  # shellcheck disable=SC2206
  extra_gen=( ${EXTRA_GEN_ARGS} )
  gen_cmd+=("${extra_gen[@]}")
fi

PNG_COUNT_EXISTING="$(find "${SAMPLES_DIR}" -type f -name '*.png' | wc -l | awk '{print $1}')"
if [[ "${REUSE_GENERATED_SAMPLES}" == "1" && "${PNG_COUNT_EXISTING}" -ge "${NUM_IMAGES}" ]]; then
  echo "[INFO] Reusing existing generated images in ${SAMPLES_DIR} (${PNG_COUNT_EXISTING} pngs)"
else
  echo "[INFO] Generating images for evaluation..."
  echo "[CMD] ${gen_cmd[*]}"
  "${gen_cmd[@]}" 2>&1 | tee "${EVAL_DIR}/generate.log"
fi

if [[ "${FID_BACKEND}" == "pytorch_fid" ]]; then
  fid_cmd=(
    python scripts/pytorch_fid_recursive.py calc
    "--images=${SAMPLES_DIR}"
    "--ref=${REF_INPUT}"
    "--device=cuda"
    "--batch-size=${FID_BATCH}"
  )
else
  fid_cmd=(
    torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" fid.py calc
    "--images=${SAMPLES_DIR}"
    "--ref=${REF_INPUT}"
    "--num=${NUM_IMAGES}"
    "--seed=${SEED_START}"
    "--batch=${FID_BATCH}"
  )
fi
if [[ -n "${EXTRA_FID_ARGS}" ]]; then
  # shellcheck disable=SC2206
  extra_fid=( ${EXTRA_FID_ARGS} )
  fid_cmd+=("${extra_fid[@]}")
fi

echo "[INFO] Calculating FID..."
echo "[CMD] ${fid_cmd[*]}"
"${fid_cmd[@]}" 2>&1 | tee "${EVAL_DIR}/fid.log"

FID_VALUE="$(python - "${EVAL_DIR}/fid.log" <<'PY'
import re
import sys

path = sys.argv[1]
value = None
with open(path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        s = line.strip()
        m = re.search(r"FID:\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)", s)
        if m:
            value = m.group(1)
        if re.fullmatch(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", s):
            value = s
print("" if value is None else value)
PY
)"

PNG_COUNT="$(find "${SAMPLES_DIR}" -type f -name '*.png' | wc -l | awk '{print $1}')"
FINISH_TIME="$(date '+%Y-%m-%d %H:%M:%S')"

export RUN_DIR NETWORK_PKL EVAL_DIR SAMPLES_DIR REF_INPUT FID_VALUE PNG_COUNT FINISH_TIME
export NUM_IMAGES SEED_START SEED_END GEN_BATCH FID_BATCH GEN_STEPS NPROC_PER_NODE
export FID_BACKEND
python - <<'PY'
import json
import os
from datetime import datetime

result = {
    "finished_at": os.environ["FINISH_TIME"],
    "run_dir": os.environ["RUN_DIR"],
    "network_pkl": os.environ["NETWORK_PKL"],
    "eval_dir": os.environ["EVAL_DIR"],
    "samples_dir": os.environ["SAMPLES_DIR"],
    "reference": os.environ["REF_INPUT"],
    "num_images_requested": int(os.environ["NUM_IMAGES"]),
    "num_images_generated": int(os.environ["PNG_COUNT"]),
    "seed_start": int(os.environ["SEED_START"]),
    "seed_end": int(os.environ["SEED_END"]),
    "gen_batch": int(os.environ["GEN_BATCH"]),
    "fid_batch": int(os.environ["FID_BATCH"]),
    "gen_steps": int(os.environ["GEN_STEPS"]),
    "nproc_per_node": int(os.environ["NPROC_PER_NODE"]),
    "fid": None if os.environ["FID_VALUE"] == "" else float(os.environ["FID_VALUE"]),
    "fid_backend": os.environ["FID_BACKEND"],
}

json_path = os.path.join(os.environ["EVAL_DIR"], "evaluation_result.json")
txt_path = os.path.join(os.environ["EVAL_DIR"], "evaluation_result.txt")

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)

lines = [
    "CIFAR-10 Evaluation Result",
    "==========================",
    f"finished_at: {result['finished_at']}",
    f"run_dir: {result['run_dir']}",
    f"network_pkl: {result['network_pkl']}",
    f"samples_dir: {result['samples_dir']}",
    f"reference: {result['reference']}",
    f"num_images_requested: {result['num_images_requested']}",
    f"num_images_generated: {result['num_images_generated']}",
    f"seed_range: {result['seed_start']}..{result['seed_end']}",
    f"gen_batch: {result['gen_batch']}",
    f"fid_batch: {result['fid_batch']}",
    f"gen_steps: {result['gen_steps']}",
    f"nproc_per_node: {result['nproc_per_node']}",
    f"fid: {result['fid']}",
]
with open(txt_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

print(f"[OK] Wrote evaluation result JSON: {json_path}")
print(f"[OK] Wrote evaluation result TXT:  {txt_path}")
PY

if [[ -z "${FID_VALUE}" ]]; then
  echo "[WARN] Could not parse FID value from log. Check: ${EVAL_DIR}/fid.log"
else
  echo "[OK] Final FID: ${FID_VALUE}"
fi
echo "[OK] Evaluation artifacts directory: ${EVAL_DIR}"
