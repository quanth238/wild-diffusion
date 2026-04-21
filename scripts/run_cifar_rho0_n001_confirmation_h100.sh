#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

VENV_DIR="${VENV_DIR:-/home/bachlc/.venvs/wild-diffusion-h100}"
RUN_DIR="${RUN_DIR:-${ROOT_DIR}/training-runs/paper-cifar10-cdro-fp16-rho0-n001-probe-mig_20260421T044122Z/00000-cifar10-32x32-train20pct-seed0-uncond-ddpmpp-cdroedm-gpus1-batch1024-fp16-paper-cifar10-uncond-ddpmpp-cdro-20pct-n001-rho0p0-i1-aw1p00-cw0p00-pel1-bg512-resume040000-wcu138113}"
EVAL_OUTDIR="${EVAL_OUTDIR:-${ROOT_DIR}/training-runs/fid-sweeps/cdro_rho0_n001_k080000_k100000_confirm_h100}"
REF_PATH="${REF_PATH:-${ROOT_DIR}/datasets/fid-refs/cifar10-32x32.npz}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/training-runs/fid-sweeps/logs}"

mkdir -p "${LOG_DIR}" "${EVAL_OUTDIR}"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "[ERROR] Missing venv: ${VENV_DIR}"
  exit 1
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RESUME_060000="${RUN_DIR}/training-state-060000.pt"
RESUME_080000="${RUN_DIR}/training-state-080000.pt"
RESUME_100000="${RUN_DIR}/training-state-100000.pt"
SNAPSHOT_080000="${RUN_DIR}/network-snapshot-080000.pkl"
SNAPSHOT_100000="${RUN_DIR}/network-snapshot-100000.pkl"
MANIFEST_CSV="${EVAL_OUTDIR}/cifar10_cdro_budget_manifest.csv"
PATCHED_N032_SUMMARY="${ROOT_DIR}/training-runs/fid-sweeps/cdro_rho0_n032_wcu300k_fid50k_20260420T065211Z/cifar10_cdro_budget_manifest_summary.json"

for required_path in "${RESUME_060000}" "${REF_PATH}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "[ERROR] Required path is missing: ${required_path}"
    exit 1
  fi
done

run_resume_phase() {
  local resume_state="$1"
  local target_kimg="$2"
  local target_state
  target_state="$(printf "%s/training-state-%06d.pt" "${RUN_DIR}" "${target_kimg}")"
  if [[ -f "${target_state}" ]]; then
    echo "[SKIP] Found ${target_state}; skipping resume-to-${target_kimg}kimg phase."
    return 0
  fi
  if [[ ! -f "${resume_state}" ]]; then
    echo "[ERROR] Resume state not found: ${resume_state}"
    exit 1
  fi

  echo "[INFO] Resuming ${RUN_DIR} from ${resume_state##*/} to ${target_kimg} kimg"
  RESUME="${resume_state}" \
  DURATION_MIMG="$(python -c "print(${target_kimg} / 1000.0)")" \
  BATCH=1024 \
  BATCH_GPU=512 \
  LR=1e-5 \
  WORKERS=16 \
  ARCH=ddpmpp \
  PRECOND=cdroedm \
  COND=0 \
  FP16=1 \
  AUGMENT=0.12 \
  DEBUG_EVAL=0 \
  DEBUG_ADV_VISUAL=0 \
  CIFAR_TRAIN_PERCENT=20 \
  CIFAR_TRAIN_SEED=0 \
  TICK_KIMG=128 \
  SNAP_TICKS=1 \
  DUMP_TICKS=1 \
  SEED=0 \
  ENV_MODE=venv \
  INSTALL_DEPS=0 \
  VENV_DIR="${VENV_DIR}" \
  EXTRA_TRAIN_ARGS="--cdro-n-steps-path=1 --cdro-step-size=0.02 --cdro-total-budget-rho=0.0 --cdro-time-horizon=1.0 --cdro-sigma-min=0.002 --cdro-sigma-max=80.0 --cdro-edm-ladder-mode=stochastic_stratified_quantile --cdro-per-example-sigma-ladders=True --attack-num-steps=1 --outer-attack-weight=1.0 --outer-clean-weight=0.0" \
    bash "${ROOT_DIR}/scripts/setup_and_train_cifar10.sh"
}

build_manifest() {
  local kimg_list="$1"
  echo "[INFO] Building manifest for kimg=${kimg_list} in ${EVAL_OUTDIR}"
  python "${ROOT_DIR}/scripts/build_cifar_cdro_fid_manifest.py" \
    --cdro-run-dir "${RUN_DIR}" \
    --outdir "${EVAL_OUTDIR}" \
    --kimg "${kimg_list}" \
    --batch-size 1024 \
    --seed 0 \
    --train-percent-label "20%" \
    --ref-path "${REF_PATH}"
}

run_eval_pass() {
  local expected_snapshot="$1"
  if [[ ! -f "${expected_snapshot}" ]]; then
    echo "[ERROR] Expected snapshot is missing: ${expected_snapshot}"
    exit 1
  fi
  if [[ ! -f "${MANIFEST_CSV}" ]]; then
    echo "[ERROR] Manifest is missing: ${MANIFEST_CSV}"
    exit 1
  fi

  echo "[INFO] Running FID evals for manifest ${MANIFEST_CSV}"
  python "${ROOT_DIR}/scripts/run_cifar_fid_manifest.py" \
    --manifest-csv "${MANIFEST_CSV}" \
    --env-mode venv \
    --venv-dir "${VENV_DIR}" \
    --install-deps 0 \
    --prepare-dataset 0 \
    --fid-backend edm \
    --ref-mode path \
    --ref-path "${REF_PATH}" \
    --only-pending \
    --num-images 50000 \
    --gen-batch 128 \
    --fid-batch 64 \
    --gen-steps 18 \
    --nproc-per-node 1

  echo "[INFO] Running clean EDM probes for manifest ${MANIFEST_CSV}"
  python "${ROOT_DIR}/scripts/run_cifar_loss_probe_manifest.py" \
    --manifest-csv "${MANIFEST_CSV}" \
    --only-pending \
    --device cuda \
    --batch-size 512 \
    --max-images 10000 \
    --seed 0
}

echo "[INFO] Starting rho=0 N=1 confirmation workflow on $(hostname)"
date -u
nvidia-smi || true
echo "[INFO] Existing patched rho=0 N=32 reference: ${PATCHED_N032_SUMMARY}"

run_resume_phase "${RESUME_060000}" 80000
build_manifest "80000"
run_eval_pass "${SNAPSHOT_080000}"

run_resume_phase "${RESUME_080000}" 100000
build_manifest "80000,100000"
run_eval_pass "${SNAPSHOT_100000}"

echo "[OK] Confirmation workflow complete."
echo "[OK] Run dir: ${RUN_DIR}"
echo "[OK] Manifest: ${MANIFEST_CSV}"
