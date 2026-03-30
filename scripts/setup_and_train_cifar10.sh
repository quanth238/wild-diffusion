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
  DEFAULT_OUTDIR="${SERVER_STORAGE_ROOT}/experiments/wild-diffusion"
else
  DEFAULT_VENV_DIR="${ROOT_DIR}/.venv"
  DEFAULT_DATA_ROOT="${ROOT_DIR}/datasets"
  DEFAULT_OUTDIR="${ROOT_DIR}/training-runs"
fi

VENV_DIR="${VENV_DIR:-${DEFAULT_VENV_DIR}}"
DATA_ROOT="${DATA_ROOT:-${DEFAULT_DATA_ROOT}}"
CIFAR_DIR="${CIFAR_DIR:-${DATA_ROOT}/cifar10-32x32}"
OUTDIR="${OUTDIR:-${DEFAULT_OUTDIR}}"

# Paper-aligned defaults for CIFAR-10 limited-data (20%) training.
# You can still override any value via env vars.
DURATION_MIMG="${DURATION_MIMG:-200}"
BATCH="${BATCH:-1024}"
BATCH_GPU="${BATCH_GPU:-1024}"
LR="${LR:-1e-5}"
WORKERS="${WORKERS:-16}"
AUGMENT="${AUGMENT:-0.12}"
ARCH="${ARCH:-ddpmpp}"
PRECOND="${PRECOND:-wdroedm}"
TRAINER="${TRAINER:-}"
CBASE="${CBASE:-}"
CRES="${CRES:-}"
DROPOUT="${DROPOUT:-}"
FP16="${FP16:-1}"
EMA="${EMA:-0.5}"
TICK="${TICK:-}"
SNAP="${SNAP:-}"
DUMP="${DUMP:-}"
SEED="${SEED:-}"
DESC="${DESC:-}"
RESUME="${RESUME:-}"
WDRO_WARMUP_RATIO="${WDRO_WARMUP_RATIO:-0.2}"
WDRO_M_EPOCHS="${WDRO_M_EPOCHS:-20}"
WDRO_K="${WDRO_K:-5}"
WDRO_STEP_SIZE="${WDRO_STEP_SIZE:-0.01}"
WDRO_GAMMA="${WDRO_GAMMA:-1.0}"
WDRO_P_ADV="${WDRO_P_ADV:-1.0}"
CDRO_MIX="${CDRO_MIX:-0.3}"
CDRO_ADV_STEPS="${CDRO_ADV_STEPS:-2}"
CDRO_STEP_SIZE="${CDRO_STEP_SIZE:-0.02}"
CDRO_MAX_DELTA="${CDRO_MAX_DELTA:-0.05}"
CDRO_RHO="${CDRO_RHO:-1e-4}"
CDRO_LAMBDA_INIT="${CDRO_LAMBDA_INIT:-0.1}"
CDRO_LAMBDA_LR="${CDRO_LAMBDA_LR:-1e-3}"
CDRO_START_KIMG="${CDRO_START_KIMG:-0.0}"
CDRO_RAMP_KIMG="${CDRO_RAMP_KIMG:-0.0}"
CDRO_SIGMA_FLOOR="${CDRO_SIGMA_FLOOR:-0.0}"
CDRO_SIGMA_CUT="${CDRO_SIGMA_CUT:-0.5}"
CDRO_GATE_POWER="${CDRO_GATE_POWER:-2.0}"
CDRO_DELTA_SPACE="${CDRO_DELTA_SPACE:-image}"
DEBUG_EVAL="${DEBUG_EVAL:-1}"
DEBUG_EVAL_INIT="${DEBUG_EVAL_INIT:-1}"
DEBUG_EVAL_NUM="${DEBUG_EVAL_NUM:-128}"
DEBUG_EVAL_STEPS="${DEBUG_EVAL_STEPS:-18}"
DEBUG_EVAL_BATCH="${DEBUG_EVAL_BATCH:-64}"
DEBUG_EVAL_VISUAL="${DEBUG_EVAL_VISUAL:-32}"
DEBUG_EVAL_REF="${DEBUG_EVAL_REF:-}"
DEBUG_ADV_VISUAL="${DEBUG_ADV_VISUAL:-16}"
# Limited-data setting: use only 20% of CIFAR-10 by default (10,000 images).
CIFAR_TRAIN_PERCENT="${CIFAR_TRAIN_PERCENT:-20}"  # 1..100
CIFAR_TRAIN_SEED="${CIFAR_TRAIN_SEED:-0}"
TRAIN_CIFAR_DIR="${TRAIN_CIFAR_DIR:-}"
DRY_RUN="${DRY_RUN:-0}"
ENV_MODE="${ENV_MODE:-auto}"      # auto|conda|venv
INSTALL_DEPS="${INSTALL_DEPS:-auto}"  # auto|0|1
EXPECTED_CONDA_ENV="${EXPECTED_CONDA_ENV:-quanth}"
STRICT_CONDA_ENV="${STRICT_CONDA_ENV:-1}"  # 1: require expected env name, 0: warn only
INSTALL_ONLY="${INSTALL_ONLY:-0}"  # 1: stop after env/dependency setup
DATASET_ONLY="${DATASET_ONLY:-0}"  # 1: prepare CIFAR data then exit (no GPU required)
CIFAR_ALLOW_DOWNLOAD="${CIFAR_ALLOW_DOWNLOAD:-1}"  # 1|0
CIFAR_DOWNLOAD_RETRIES="${CIFAR_DOWNLOAD_RETRIES:-3}"
CIFAR_DOWNLOAD_TIMEOUT="${CIFAR_DOWNLOAD_TIMEOUT:-30}"  # seconds

if [[ -z "${TRAINER}" ]]; then
  if [[ "${PRECOND}" == "cdroedm" ]]; then
    TRAINER="baseline"
  else
    TRAINER="wdro"
  fi
fi

if [[ -n "${RESUME}" ]]; then
  if [[ ! -f "${RESUME}" ]]; then
    echo "[ERROR] RESUME file not found: ${RESUME}"
    exit 1
  fi
  if [[ "${RESUME##*/}" != training-state-*.pt ]]; then
    echo "[ERROR] RESUME must point to training-state-*.pt"
    exit 1
  fi
  OUTDIR="$(dirname "${RESUME}")"
  echo "[INFO] Resume mode enabled. OUTDIR is forced to checkpoint directory:"
  echo "       ${OUTDIR}"
fi

python3 - <<PY
try:
    pct = int("${CIFAR_TRAIN_PERCENT}")
except Exception:
    raise SystemExit("[ERROR] CIFAR_TRAIN_PERCENT must be an integer in [1, 100].")
if pct < 1 or pct > 100:
    raise SystemExit("[ERROR] CIFAR_TRAIN_PERCENT must be in [1, 100].")
try:
    int("${CIFAR_TRAIN_SEED}")
except Exception:
    raise SystemExit("[ERROR] CIFAR_TRAIN_SEED must be an integer.")
if "${TRAINER}" not in {"baseline", "wdro"}:
    raise SystemExit("[ERROR] TRAINER must be one of: baseline, wdro")
if "${PRECOND}" not in {"wdroedm", "advedm", "cdroedm"}:
    raise SystemExit("[ERROR] PRECOND must be one of: wdroedm, advedm, cdroedm")
if "${PRECOND}" == "cdroedm" and "${TRAINER}" != "baseline":
    raise SystemExit("[ERROR] PRECOND=cdroedm currently requires TRAINER=baseline")
PY

if [[ -z "${TRAIN_CIFAR_DIR}" ]]; then
  if [[ "${CIFAR_TRAIN_PERCENT}" == "100" ]]; then
    TRAIN_CIFAR_DIR="${CIFAR_DIR}"
  else
    TRAIN_CIFAR_DIR="${DATA_ROOT}/cifar10-32x32-train${CIFAR_TRAIN_PERCENT}pct-seed${CIFAR_TRAIN_SEED}"
  fi
fi

mkdir -p "${DATA_ROOT}" "${OUTDIR}"
export DATA_ROOT CIFAR_DIR TRAIN_CIFAR_DIR CIFAR_TRAIN_PERCENT CIFAR_TRAIN_SEED
export CIFAR_ALLOW_DOWNLOAD CIFAR_DOWNLOAD_RETRIES CIFAR_DOWNLOAD_TIMEOUT

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
    echo "Activate your conda env first (expected: quanth), or set ENV_MODE=venv."
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
  if [ ! -d "${VENV_DIR}" ]; then
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
  if [[ "${ENV_MODE}" == "conda" ]]; then
    INSTALL_DEPS="0"
  else
    INSTALL_DEPS="1"
  fi
fi

python - <<PY
try:
    aug = float("${AUGMENT}")
except Exception:
    raise SystemExit("[ERROR] AUGMENT must be a float.")
if "${TRAINER}" == "wdro":
    if aug <= 0:
        raise SystemExit("[ERROR] AUGMENT must be > 0 when TRAINER=wdro (augment pipe is required).")
else:
    if aug < 0:
        raise SystemExit("[ERROR] AUGMENT must be >= 0 when TRAINER=baseline.")
PY

if [[ "${INSTALL_DEPS}" == "1" ]]; then
  echo "[INFO] INSTALL_DEPS=1 (installing/refreshing Python packages)"
  python -m pip install --upgrade pip setuptools wheel
  python -m pip install --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}" torch torchvision
  python -m pip install click numpy pillow psutil scipy tqdm requests
else
  echo "[INFO] INSTALL_DEPS=0 (reusing packages from current environment)"
fi

python - <<'PY'
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

if missing:
    raise SystemExit(
        "[ERROR] Missing Python modules in current environment: "
        + ", ".join(sorted(set(missing)))
        + ". Set INSTALL_DEPS=1 to install them automatically."
    )
PY

echo "[INFO] VENV_DIR=${VENV_DIR}"
echo "[INFO] DATA_ROOT=${DATA_ROOT}"
echo "[INFO] CIFAR_DIR=${CIFAR_DIR}"
echo "[INFO] TRAIN_CIFAR_DIR=${TRAIN_CIFAR_DIR}"
echo "[INFO] CIFAR_TRAIN_PERCENT=${CIFAR_TRAIN_PERCENT}"
echo "[INFO] CIFAR_TRAIN_SEED=${CIFAR_TRAIN_SEED}"
echo "[INFO] OUTDIR=${OUTDIR}"
echo "[INFO] TRAINER=${TRAINER}"
echo "[INFO] PRECOND=${PRECOND}"
echo "[INFO] DEBUG_EVAL=${DEBUG_EVAL} (init=${DEBUG_EVAL_INIT}, num=${DEBUG_EVAL_NUM}, steps=${DEBUG_EVAL_STEPS}, batch=${DEBUG_EVAL_BATCH}, visual=${DEBUG_EVAL_VISUAL}, adv_visual=${DEBUG_ADV_VISUAL})"
if [[ "${PRECOND}" == "cdroedm" ]]; then
  echo "[INFO] CDRO mix/steps/step=${CDRO_MIX}/${CDRO_ADV_STEPS}/${CDRO_STEP_SIZE}"
  echo "[INFO] CDRO max_delta/rho=${CDRO_MAX_DELTA}/${CDRO_RHO}"
  echo "[INFO] CDRO lambda init/lr=${CDRO_LAMBDA_INIT}/${CDRO_LAMBDA_LR}"
  echo "[INFO] CDRO sigma cut/power=${CDRO_SIGMA_CUT}/${CDRO_GATE_POWER}"
  echo "[INFO] CDRO delta_space=${CDRO_DELTA_SPACE}"
fi

if [[ "${INSTALL_ONLY}" == "1" ]]; then
  echo "[INFO] INSTALL_ONLY=1, dependency setup completed. Exiting before dataset prep/training."
  exit 0
fi

python - <<'PY'
import json
import os
import socket
import time
from pathlib import Path

from torchvision.datasets import CIFAR10

root = Path(os.environ["DATA_ROOT"])
out_dir = Path(os.environ["CIFAR_DIR"])
allow_download = os.environ.get("CIFAR_ALLOW_DOWNLOAD", "1") == "1"
download_retries = int(os.environ.get("CIFAR_DOWNLOAD_RETRIES", "3"))
download_timeout = int(os.environ.get("CIFAR_DOWNLOAD_TIMEOUT", "30"))
out_dir.mkdir(parents=True, exist_ok=True)

png_count = sum(1 for _ in out_dir.rglob("*.png"))
labels_path = out_dir / "dataset.json"

if png_count >= 50000 and labels_path.exists():
    print(f"[INFO] Reusing existing CIFAR-10 directory: {out_dir}")
else:
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
                    "Prepare dataset first, then rerun training."
                )
            try:
                print(f"[INFO] Downloading CIFAR-10 to: {root} (attempt {attempt}/{download_retries})")
                ds = CIFAR10(root=str(root), train=True, download=True)
                break
            except Exception as err:
                if attempt == download_retries:
                    raise SystemExit(
                        "[ERROR] Failed to download CIFAR-10 after retries. "
                        "This usually means worker node has no internet access. "
                        "Prepare dataset once on a node with internet, then run training with "
                        "CIFAR_ALLOW_DOWNLOAD=0.\n"
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

python - <<'PY'
import json
import os
import random
import shutil
from pathlib import Path

src_dir = Path(os.environ["CIFAR_DIR"])
dst_dir = Path(os.environ["TRAIN_CIFAR_DIR"])
pct = int(os.environ["CIFAR_TRAIN_PERCENT"])
seed = int(os.environ["CIFAR_TRAIN_SEED"])

src_labels_path = src_dir / "dataset.json"
if not src_labels_path.exists():
    raise SystemExit(f"[ERROR] Missing source dataset.json: {src_labels_path}")

with open(src_labels_path, "r", encoding="utf-8") as f:
    src_labels = json.load(f).get("labels", [])

if not src_labels:
    raise SystemExit(f"[ERROR] Source dataset has no labels: {src_labels_path}")

if pct >= 100:
    print(f"[INFO] CIFAR_TRAIN_PERCENT=100 -> using full dataset: {src_dir}")
    print(f"[INFO] TRAIN_CIFAR_DIR={src_dir}")
    raise SystemExit(0)

target_count = max(1, int(len(src_labels) * pct / 100))
dst_labels_path = dst_dir / "dataset.json"
dst_png_count = sum(1 for _ in dst_dir.rglob("*.png")) if dst_dir.exists() else 0

if dst_labels_path.exists() and dst_png_count == target_count:
    print(f"[INFO] Reusing limited dataset: {dst_dir} ({dst_png_count} images)")
    print(f"[INFO] TRAIN_CIFAR_DIR={dst_dir}")
    raise SystemExit(0)

if dst_dir.exists():
    shutil.rmtree(dst_dir)
dst_dir.mkdir(parents=True, exist_ok=True)

rng = random.Random(seed)
indices = list(range(len(src_labels)))
rng.shuffle(indices)
keep_set = set(indices[:target_count])

subset_labels = []
for idx, (rel_path, label) in enumerate(src_labels):
    if idx not in keep_set:
        continue
    src_img = src_dir / rel_path
    dst_img = dst_dir / rel_path
    dst_img.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_img, dst_img)
    subset_labels.append([rel_path, int(label)])

with open(dst_labels_path, "w", encoding="utf-8") as f:
    json.dump({"labels": subset_labels}, f)

print(
    f"[INFO] Built limited CIFAR-10 subset: {dst_dir} "
    f"({len(subset_labels)} / {len(src_labels)} images, pct={pct}, seed={seed})"
)
print(f"[INFO] TRAIN_CIFAR_DIR={dst_dir}")
PY

if [[ "${DATASET_ONLY}" == "1" ]]; then
  echo "[INFO] DATASET_ONLY=1, CIFAR dataset preparation completed. Exiting before training."
  exit 0
fi

if [[ "${DRY_RUN}" != "1" ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[ERROR] NVIDIA GPU not detected (nvidia-smi missing)."
    echo "Training requires CUDA/NCCL. Use DRY_RUN=1 or DATASET_ONLY=1 for non-GPU nodes."
    exit 1
  fi

  # Final sanity check before launch.
  python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("[ERROR] torch.cuda.is_available() is False. Install a CUDA-enabled PyTorch build.")
print(f"[INFO] CUDA is available with {torch.cuda.device_count()} GPU(s).")
PY
fi

train_cmd=(
  torchrun --standalone --nproc_per_node=1 train.py
  "--outdir=${OUTDIR}"
  "--data=${TRAIN_CIFAR_DIR}"
  "--cond=1"
  "--arch=${ARCH}"
  "--precond=${PRECOND}"
  "--trainer=${TRAINER}"
  "--duration=${DURATION_MIMG}"
  "--batch=${BATCH}"
  "--lr=${LR}"
  "--workers=${WORKERS}"
  "--augment=${AUGMENT}"
  "--fp16=${FP16}"
  "--ema=${EMA}"
  "--wdro-warmup-ratio=${WDRO_WARMUP_RATIO}"
  "--wdro-m-epochs=${WDRO_M_EPOCHS}"
  "--wdro-k=${WDRO_K}"
  "--wdro-step-size=${WDRO_STEP_SIZE}"
  "--wdro-gamma=${WDRO_GAMMA}"
  "--wdro-p-adv=${WDRO_P_ADV}"
  "--debug-eval=${DEBUG_EVAL}"
  "--debug-eval-init=${DEBUG_EVAL_INIT}"
  "--debug-eval-num=${DEBUG_EVAL_NUM}"
  "--debug-eval-steps=${DEBUG_EVAL_STEPS}"
  "--debug-eval-batch=${DEBUG_EVAL_BATCH}"
  "--debug-eval-visual=${DEBUG_EVAL_VISUAL}"
  "--debug-adv-visual=${DEBUG_ADV_VISUAL}"
)

if [[ "${PRECOND}" == "cdroedm" ]]; then
  train_cmd+=(
    "--cdro-mix=${CDRO_MIX}"
    "--cdro-adv-steps=${CDRO_ADV_STEPS}"
    "--cdro-step-size=${CDRO_STEP_SIZE}"
    "--cdro-max-delta=${CDRO_MAX_DELTA}"
    "--cdro-rho=${CDRO_RHO}"
    "--cdro-lambda-init=${CDRO_LAMBDA_INIT}"
    "--cdro-lambda-lr=${CDRO_LAMBDA_LR}"
    "--cdro-start-kimg=${CDRO_START_KIMG}"
    "--cdro-ramp-kimg=${CDRO_RAMP_KIMG}"
    "--cdro-sigma-floor=${CDRO_SIGMA_FLOOR}"
    "--cdro-sigma-cut=${CDRO_SIGMA_CUT}"
    "--cdro-gate-power=${CDRO_GATE_POWER}"
    "--cdro-delta-space=${CDRO_DELTA_SPACE}"
  )
fi

if [[ -n "${DEBUG_EVAL_REF}" ]]; then
  train_cmd+=("--debug-eval-ref=${DEBUG_EVAL_REF}")
fi

if [[ -n "${RESUME}" ]]; then
  train_cmd+=("--nosubdir")
  train_cmd+=("--resume=${RESUME}")
fi

if [[ -n "${BATCH_GPU}" ]]; then
  train_cmd+=("--batch-gpu=${BATCH_GPU}")
fi

if [[ -n "${CBASE}" ]]; then
  train_cmd+=("--cbase=${CBASE}")
fi

if [[ -n "${CRES}" ]]; then
  train_cmd+=("--cres=${CRES}")
fi

if [[ -n "${DROPOUT}" ]]; then
  train_cmd+=("--dropout=${DROPOUT}")
fi

if [[ -n "${TICK}" ]]; then
  train_cmd+=("--tick=${TICK}")
fi

if [[ -n "${SNAP}" ]]; then
  train_cmd+=("--snap=${SNAP}")
fi

if [[ -n "${DUMP}" ]]; then
  train_cmd+=("--dump=${DUMP}")
fi

if [[ -n "${SEED}" ]]; then
  train_cmd+=("--seed=${SEED}")
fi

if [[ -n "${DESC}" ]]; then
  train_cmd+=("--desc=${DESC}")
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  train_cmd+=("--dry-run")
fi

exec "${train_cmd[@]}"
