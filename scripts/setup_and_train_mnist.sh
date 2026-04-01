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
MNIST_DIR="${MNIST_DIR:-${DATA_ROOT}/mnist-32x32-train}"
OUTDIR="${OUTDIR:-${DEFAULT_OUTDIR}}"

TRAINER="${TRAINER:-baseline}"   # baseline|wdro
PRECOND="${PRECOND:-wdroedm}"    # wdroedm|advedm|cdroedm|cdromarkovedm|cdromarkovfull
COND="${COND:-1}"
ARCH="${ARCH:-ddpmpp}"
CBASE="${CBASE:-}"
CRES="${CRES:-}"
DROPOUT="${DROPOUT:-}"
DURATION_MIMG="${DURATION_MIMG:-10}"
BATCH="${BATCH:-512}"
BATCH_GPU="${BATCH_GPU:-256}"
LR="${LR:-1e-4}"
WORKERS="${WORKERS:-8}"
AUGMENT="${AUGMENT:-0.0}"
FP16="${FP16:-1}"
EMA="${EMA:-0.5}"
TICK="${TICK:-}"
SNAP="${SNAP:-}"
DUMP="${DUMP:-}"
SEED="${SEED:-}"
DESC="${DESC:-}"
MNIST_TRAIN_PERCENT="${MNIST_TRAIN_PERCENT:-100}"
MNIST_TRAIN_SEED="${MNIST_TRAIN_SEED:-0}"
TRAIN_MNIST_DIR="${TRAIN_MNIST_DIR:-}"
RESUME="${RESUME:-}"
TRANSFER="${TRANSFER:-}"
DRY_RUN="${DRY_RUN:-0}"

WDRO_WARMUP_RATIO="${WDRO_WARMUP_RATIO:-0.2}"
WDRO_M_EPOCHS="${WDRO_M_EPOCHS:-10}"
WDRO_K="${WDRO_K:-3}"
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
CDRO_CONTROL_CBASE="${CDRO_CONTROL_CBASE:-64}"
CDRO_CONTROL_DROPOUT="${CDRO_CONTROL_DROPOUT:-0.0}"
MARKOV_NUM_STEPS="${MARKOV_NUM_STEPS:-8}"
MARKOV_TOTAL_TIME="${MARKOV_TOTAL_TIME:-1.0}"
MARKOV_BETA_MIN="${MARKOV_BETA_MIN:-0.1}"
MARKOV_BETA_MAX="${MARKOV_BETA_MAX:-12.0}"
MARKOV_SDE_FAMILY="${MARKOV_SDE_FAMILY:-vp_cosine}"
MARKOV_WEIGHT_SCHEDULE="${MARKOV_WEIGHT_SCHEDULE:-uniform}"
MARKOV_CONTROL_LR="${MARKOV_CONTROL_LR:-2e-4}"
MARKOV_LAMBDA_MIN="${MARKOV_LAMBDA_MIN:-0.0}"
MARKOV_REVERSE_CONTROL_SCALE="${MARKOV_REVERSE_CONTROL_SCALE:-1.0}"
MARKOV_REVERSE_NOISE_SCALE="${MARKOV_REVERSE_NOISE_SCALE:-1.0}"
MARKOV_TERMINAL_MOMENTUM="${MARKOV_TERMINAL_MOMENTUM:-0.95}"
DEBUG_EVAL="${DEBUG_EVAL:-0}"
DEBUG_EVAL_INIT="${DEBUG_EVAL_INIT:-1}"
DEBUG_EVAL_NUM="${DEBUG_EVAL_NUM:-128}"
DEBUG_EVAL_STEPS="${DEBUG_EVAL_STEPS:-18}"
DEBUG_EVAL_BATCH="${DEBUG_EVAL_BATCH:-64}"
DEBUG_EVAL_VISUAL="${DEBUG_EVAL_VISUAL:-32}"
DEBUG_EVAL_REF="${DEBUG_EVAL_REF:-}"
DEBUG_ADV_VISUAL="${DEBUG_ADV_VISUAL:-16}"

ENV_MODE="${ENV_MODE:-auto}"      # auto|conda|venv
INSTALL_DEPS="${INSTALL_DEPS:-auto}"  # auto|0|1
EXPECTED_CONDA_ENV="${EXPECTED_CONDA_ENV:-quanth}"
STRICT_CONDA_ENV="${STRICT_CONDA_ENV:-1}"
INSTALL_ONLY="${INSTALL_ONLY:-0}"
DATASET_ONLY="${DATASET_ONLY:-0}"
PREPARE_MNIST="${PREPARE_MNIST:-auto}"  # auto|0|1
MNIST_RGB="${MNIST_RGB:-0}"
MNIST_FORCE_PREPARE="${MNIST_FORCE_PREPARE:-0}"

if [[ -n "${RESUME}" && -n "${TRANSFER}" ]]; then
  echo "[ERROR] RESUME and TRANSFER cannot both be set."
  exit 1
fi

if [[ -n "${TRANSFER}" ]]; then
  if [[ ! -f "${TRANSFER}" ]]; then
    echo "[ERROR] TRANSFER file not found: ${TRANSFER}"
    exit 1
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
    pct = int("${MNIST_TRAIN_PERCENT}")
except Exception:
    raise SystemExit("[ERROR] MNIST_TRAIN_PERCENT must be an integer in [1, 100].")
if pct < 1 or pct > 100:
    raise SystemExit("[ERROR] MNIST_TRAIN_PERCENT must be in [1, 100].")
if "${TRAINER}" not in {"baseline", "wdro"}:
    raise SystemExit("[ERROR] TRAINER must be one of: baseline, wdro")
if "${PRECOND}" not in {"wdroedm", "advedm", "cdroedm", "cdromarkovedm", "cdromarkovfull"}:
    raise SystemExit("[ERROR] PRECOND must be one of: wdroedm, advedm, cdroedm, cdromarkovedm, cdromarkovfull")
if "${PRECOND}" in {"cdroedm", "cdromarkovedm", "cdromarkovfull"} and "${TRAINER}" != "baseline":
    raise SystemExit("[ERROR] PRECOND=${PRECOND} currently requires TRAINER=baseline")
PY

if [[ -z "${TRAIN_MNIST_DIR}" ]]; then
  if [[ "${MNIST_TRAIN_PERCENT}" == "100" ]]; then
    TRAIN_MNIST_DIR="${MNIST_DIR}"
  else
    TRAIN_MNIST_DIR="${DATA_ROOT}/mnist-32x32-train${MNIST_TRAIN_PERCENT}pct-seed${MNIST_TRAIN_SEED}"
  fi
fi

mkdir -p "${DATA_ROOT}" "${OUTDIR}"
export DATA_ROOT MNIST_DIR TRAIN_MNIST_DIR MNIST_TRAIN_PERCENT MNIST_TRAIN_SEED

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

echo "[INFO] DATA_ROOT=${DATA_ROOT}"
echo "[INFO] MNIST_DIR=${MNIST_DIR}"
echo "[INFO] TRAIN_MNIST_DIR=${TRAIN_MNIST_DIR}"
echo "[INFO] TRAINER=${TRAINER}"
echo "[INFO] PRECOND=${PRECOND}"
echo "[INFO] MNIST_TRAIN_PERCENT=${MNIST_TRAIN_PERCENT}"
echo "[INFO] OUTDIR=${OUTDIR}"
if [[ -n "${TRANSFER}" ]]; then
  echo "[INFO] TRANSFER=${TRANSFER}"
fi
if [[ "${PRECOND}" == "cdroedm" || "${PRECOND}" == "cdromarkovedm" || "${PRECOND}" == "cdromarkovfull" ]]; then
  echo "[INFO] CDRO mix/steps/step=${CDRO_MIX}/${CDRO_ADV_STEPS}/${CDRO_STEP_SIZE}"
  echo "[INFO] CDRO max_delta/rho=${CDRO_MAX_DELTA}/${CDRO_RHO}"
  echo "[INFO] CDRO lambda init/lr=${CDRO_LAMBDA_INIT}/${CDRO_LAMBDA_LR}"
  echo "[INFO] CDRO start/ramp kimg=${CDRO_START_KIMG}/${CDRO_RAMP_KIMG}"
  echo "[INFO] CDRO sigma floor/cut=${CDRO_SIGMA_FLOOR}/${CDRO_SIGMA_CUT}"
  echo "[INFO] CDRO gate power=${CDRO_GATE_POWER}"
  echo "[INFO] CDRO delta_space=${CDRO_DELTA_SPACE}"
  if [[ "${PRECOND}" == "cdromarkovedm" || "${PRECOND}" == "cdromarkovfull" ]]; then
    echo "[INFO] CDRO control cbase/dropout=${CDRO_CONTROL_CBASE}/${CDRO_CONTROL_DROPOUT}"
  fi
  if [[ "${PRECOND}" == "cdromarkovfull" ]]; then
    echo "[INFO] Markov steps/time=${MARKOV_NUM_STEPS}/${MARKOV_TOTAL_TIME}"
    echo "[INFO] Markov beta min/max=${MARKOV_BETA_MIN}/${MARKOV_BETA_MAX}"
    echo "[INFO] Markov family/weights=${MARKOV_SDE_FAMILY}/${MARKOV_WEIGHT_SCHEDULE}"
    echo "[INFO] Markov control lr/lambda min=${MARKOV_CONTROL_LR}/${MARKOV_LAMBDA_MIN}"
    echo "[INFO] Markov reverse control/noise=${MARKOV_REVERSE_CONTROL_SCALE}/${MARKOV_REVERSE_NOISE_SCALE}"
    echo "[INFO] Markov terminal momentum=${MARKOV_TERMINAL_MOMENTUM}"
  fi
fi

if [[ "${INSTALL_ONLY}" == "1" ]]; then
  echo "[INFO] INSTALL_ONLY=1, dependency setup completed. Exiting before dataset prep/training."
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

if [[ ! -d "${MNIST_DIR}" || ! -f "${MNIST_DIR}/dataset.json" ]]; then
  echo "[ERROR] MNIST dataset directory is missing or incomplete: ${MNIST_DIR}"
  exit 1
fi

if [[ "${MNIST_TRAIN_PERCENT}" != "100" ]]; then
  python scripts/build_imagefolder_subset.py \
    --src "${MNIST_DIR}" \
    --dest "${TRAIN_MNIST_DIR}" \
    --percent "${MNIST_TRAIN_PERCENT}" \
    --seed "${MNIST_TRAIN_SEED}"
fi

if [[ "${DATASET_ONLY}" == "1" ]]; then
  echo "[INFO] DATASET_ONLY=1, MNIST dataset preparation completed. Exiting before training."
  exit 0
fi

if [[ "${DRY_RUN}" != "1" ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[ERROR] NVIDIA GPU not detected (nvidia-smi missing)."
    echo "Training requires CUDA/NCCL. Use DRY_RUN=1 or DATASET_ONLY=1 for non-GPU nodes."
    exit 1
  fi
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
  "--data=${TRAIN_MNIST_DIR}"
  "--cond=${COND}"
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
  "--cdro-control-cbase=${CDRO_CONTROL_CBASE}"
  "--cdro-control-dropout=${CDRO_CONTROL_DROPOUT}"
  "--markov-num-steps=${MARKOV_NUM_STEPS}"
  "--markov-total-time=${MARKOV_TOTAL_TIME}"
  "--markov-beta-min=${MARKOV_BETA_MIN}"
  "--markov-beta-max=${MARKOV_BETA_MAX}"
  "--markov-sde-family=${MARKOV_SDE_FAMILY}"
  "--markov-weight-schedule=${MARKOV_WEIGHT_SCHEDULE}"
  "--markov-control-lr=${MARKOV_CONTROL_LR}"
  "--markov-lambda-min=${MARKOV_LAMBDA_MIN}"
  "--markov-reverse-control-scale=${MARKOV_REVERSE_CONTROL_SCALE}"
  "--markov-reverse-noise-scale=${MARKOV_REVERSE_NOISE_SCALE}"
  "--markov-terminal-momentum=${MARKOV_TERMINAL_MOMENTUM}"
  "--debug-eval=${DEBUG_EVAL}"
  "--debug-eval-init=${DEBUG_EVAL_INIT}"
  "--debug-eval-num=${DEBUG_EVAL_NUM}"
  "--debug-eval-steps=${DEBUG_EVAL_STEPS}"
  "--debug-eval-batch=${DEBUG_EVAL_BATCH}"
  "--debug-eval-visual=${DEBUG_EVAL_VISUAL}"
  "--debug-adv-visual=${DEBUG_ADV_VISUAL}"
)

if [[ -n "${DEBUG_EVAL_REF}" ]]; then
  train_cmd+=("--debug-eval-ref=${DEBUG_EVAL_REF}")
fi

if [[ -n "${TRANSFER}" ]]; then
  train_cmd+=("--transfer=${TRANSFER}")
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
