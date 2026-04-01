#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

# Locked paper-facing CIFAR-10 CDRO screen.
# This is the first practical extension of the MNIST-winning plug-in CDRO recipe.
# It is intentionally narrow: same benchmark for baseline / WDRO / CDRO, with only
# the CDRO knobs set to the current default research preset.

RUN_TAG="${RUN_TAG:-cifar10-cdro-locked-screen-v1}"

# Benchmark scope.
METHODS="${METHODS:-baseline wdro cdro}"
CIFAR_TRAIN_PERCENT="${CIFAR_TRAIN_PERCENT:-20}"
CIFAR_TRAIN_SEED="${CIFAR_TRAIN_SEED:-0}"
DURATION_MIMG="${DURATION_MIMG:-8}"

# Stable training defaults for a single-GPU screen.
BATCH="${BATCH:-1024}"
BATCH_GPU="${BATCH_GPU:-256}"
FP16="${FP16:-1}"
WORKERS="${WORKERS:-16}"
LR="${LR:-1e-5}"
ARCH="${ARCH:-ddpmpp}"
COND="${COND:-1}"
AUGMENT="${AUGMENT:-0.12}"

# End-of-run evaluation budget for screening, not final paper numbers.
NUM_IMAGES="${NUM_IMAGES:-5000}"
GEN_BATCH="${GEN_BATCH:-128}"
FID_BATCH="${FID_BATCH:-64}"
GEN_STEPS="${GEN_STEPS:-35}"
REF_MODE="${REF_MODE:-compute}"

# Locked plug-in CDRO defaults.
CDRO_MIX="${CDRO_MIX:-0.05}"
CDRO_ADV_STEPS="${CDRO_ADV_STEPS:-1}"
CDRO_STEP_SIZE="${CDRO_STEP_SIZE:-0.02}"
CDRO_MAX_DELTA="${CDRO_MAX_DELTA:-0.03}"
CDRO_RHO="${CDRO_RHO:-2e-5}"
CDRO_LAMBDA_INIT="${CDRO_LAMBDA_INIT:-0.001}"
CDRO_LAMBDA_LR="${CDRO_LAMBDA_LR:-5e-4}"
CDRO_SIGMA_FLOOR="${CDRO_SIGMA_FLOOR:-0.12}"
CDRO_SIGMA_CUT="${CDRO_SIGMA_CUT:-0.70}"
CDRO_GATE_POWER="${CDRO_GATE_POWER:-1.0}"
CDRO_DELTA_SPACE="${CDRO_DELTA_SPACE:-noise}"

# Warm-up as a CIFAR starting point, not a theorem. With 8 MIMG, 20% start and 20% ramp.
CDRO_START_KIMG="${CDRO_START_KIMG:-1600}"
CDRO_RAMP_KIMG="${CDRO_RAMP_KIMG:-1600}"

# Keep WDRO on the current repo-paper defaults unless explicitly overridden.
WDRO_WARMUP_RATIO="${WDRO_WARMUP_RATIO:-0.2}"
WDRO_M_EPOCHS="${WDRO_M_EPOCHS:-20}"
WDRO_K="${WDRO_K:-5}"
WDRO_STEP_SIZE="${WDRO_STEP_SIZE:-0.01}"
WDRO_GAMMA="${WDRO_GAMMA:-1.0}"
WDRO_P_ADV="${WDRO_P_ADV:-1.0}"

echo "[INFO] Running locked CIFAR-10 CDRO screen"
echo "[INFO] RUN_TAG=${RUN_TAG}"
echo "[INFO] METHODS=${METHODS}"
echo "[INFO] CIFAR_TRAIN_PERCENT=${CIFAR_TRAIN_PERCENT}"
echo "[INFO] DURATION_MIMG=${DURATION_MIMG}"
echo "[INFO] CDRO preset: mix=${CDRO_MIX}, adv_steps=${CDRO_ADV_STEPS}, max_delta=${CDRO_MAX_DELTA}, rho=${CDRO_RHO}"
echo "[INFO] CDRO sigma band: floor=${CDRO_SIGMA_FLOOR}, cut=${CDRO_SIGMA_CUT}, gate_power=${CDRO_GATE_POWER}, delta_space=${CDRO_DELTA_SPACE}"
echo "[INFO] CDRO warm-up: start_kimg=${CDRO_START_KIMG}, ramp_kimg=${CDRO_RAMP_KIMG}"

RUN_TAG="${RUN_TAG}" \
METHODS="${METHODS}" \
CIFAR_TRAIN_PERCENT="${CIFAR_TRAIN_PERCENT}" \
CIFAR_TRAIN_SEED="${CIFAR_TRAIN_SEED}" \
DURATION_MIMG="${DURATION_MIMG}" \
BATCH="${BATCH}" \
BATCH_GPU="${BATCH_GPU}" \
FP16="${FP16}" \
WORKERS="${WORKERS}" \
LR="${LR}" \
ARCH="${ARCH}" \
COND="${COND}" \
AUGMENT="${AUGMENT}" \
NUM_IMAGES="${NUM_IMAGES}" \
GEN_BATCH="${GEN_BATCH}" \
FID_BATCH="${FID_BATCH}" \
GEN_STEPS="${GEN_STEPS}" \
REF_MODE="${REF_MODE}" \
CDRO_MIX="${CDRO_MIX}" \
CDRO_ADV_STEPS="${CDRO_ADV_STEPS}" \
CDRO_STEP_SIZE="${CDRO_STEP_SIZE}" \
CDRO_MAX_DELTA="${CDRO_MAX_DELTA}" \
CDRO_RHO="${CDRO_RHO}" \
CDRO_LAMBDA_INIT="${CDRO_LAMBDA_INIT}" \
CDRO_LAMBDA_LR="${CDRO_LAMBDA_LR}" \
CDRO_START_KIMG="${CDRO_START_KIMG}" \
CDRO_RAMP_KIMG="${CDRO_RAMP_KIMG}" \
CDRO_SIGMA_FLOOR="${CDRO_SIGMA_FLOOR}" \
CDRO_SIGMA_CUT="${CDRO_SIGMA_CUT}" \
CDRO_GATE_POWER="${CDRO_GATE_POWER}" \
CDRO_DELTA_SPACE="${CDRO_DELTA_SPACE}" \
WDRO_WARMUP_RATIO="${WDRO_WARMUP_RATIO}" \
WDRO_M_EPOCHS="${WDRO_M_EPOCHS}" \
WDRO_K="${WDRO_K}" \
WDRO_STEP_SIZE="${WDRO_STEP_SIZE}" \
WDRO_GAMMA="${WDRO_GAMMA}" \
WDRO_P_ADV="${WDRO_P_ADV}" \
bash scripts/run_cifar_compare.sh
