#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

VENV_DIR="${VENV_DIR:-/home/bachlc/.venvs/wild-diffusion-h100}"
TARGET_WCU="${TARGET_WCU:-184148.22639230292}"
CAMPAIGN_TAG="${CAMPAIGN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
TRAIN_OUTROOT="${TRAIN_OUTROOT:-${ROOT_DIR}/training-runs/paper-cifar10-cdro-fp16-rho0-baseline80k_${CAMPAIGN_TAG}}"
EVAL_SWEEP_ROOT="${EVAL_SWEEP_ROOT:-${ROOT_DIR}/training-runs/fid-sweeps/cdro_rho0_n008_n016_baseline80k_${CAMPAIGN_TAG}}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/training-runs/fid-sweeps/logs}"
PYTORCH_FID_REF="${PYTORCH_FID_REF:-${ROOT_DIR}/training-runs/fid-sweeps/cifar10_baseline_vs_wdro_coarse_20260414/pytorch_fid_cifar10_train_ref_stats.npz}"
RUN_LOG="${RUN_LOG:-${LOG_DIR}/cdro_rho0_n008_n016_baseline80k_${CAMPAIGN_TAG}.log}"
TRAIN_LR="${TRAIN_LR:-1e-5}"
TICK_KIMG="${TICK_KIMG:-128}"
DUMP_TICKS_FRESH="${DUMP_TICKS_FRESH:-1}"
DUMP_TICKS_RESUME="${DUMP_TICKS_RESUME:-1}"
RUN_N_STEPS="${RUN_N_STEPS:-8,16}"

mkdir -p "${LOG_DIR}" "${TRAIN_OUTROOT}" "${EVAL_SWEEP_ROOT}"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "[ERROR] Missing venv: ${VENV_DIR}"
  exit 1
fi
if [[ ! -f "${PYTORCH_FID_REF}" ]]; then
  echo "[ERROR] Missing pytorch-fid reference stats: ${PYTORCH_FID_REF}"
  exit 1
fi

for cmd_name in python bash find tee sort jq; do
  if ! command -v "${cmd_name}" >/dev/null 2>&1; then
    echo "[ERROR] Missing command: ${cmd_name}"
    exit 1
  fi
done

if [[ -r /proc/self/fd ]]; then
  exec > >(tee -a "${RUN_LOG}") 2>&1
else
  exec >>"${RUN_LOG}" 2>&1
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ ! -r /proc/self/maps || ! -r /proc/cpuinfo || ! -r /proc/sys/vm/mmap_min_addr ]]; then
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/scripts/ensure_procfs_compat.sh"
fi

find_run_dir() {
  local n_steps="$1"
  find -L "${TRAIN_OUTROOT}" -mindepth 1 -maxdepth 1 -type d \
    -name "*-n$(printf '%03d' "${n_steps}")-rho0p0-*" | sort | tail -n 1
}

latest_training_state() {
  local run_dir="$1"
  find -L "${run_dir}" -mindepth 1 -maxdepth 1 -type f -name 'training-state-*.pt' | sort | tail -n 1
}

resume_existing_run() {
  local run_dir="$1"
  local target_kimg="$2"
  local n_steps="$3"
  local resume_state

  resume_state="$(latest_training_state "${run_dir}")"
  if [[ -z "${resume_state}" || ! -f "${resume_state}" ]]; then
    echo "[ERROR] Could not find a resume state in ${run_dir}"
    exit 1
  fi

  echo "[INFO] Resuming existing rho=0, N=${n_steps} run from ${resume_state##*/} to ${target_kimg} kimg"
  RESUME="${resume_state}" \
  DURATION_MIMG="$(python -c "print(${target_kimg} / 1000.0)")" \
  BATCH=1024 \
  BATCH_GPU=1024 \
  LR="${TRAIN_LR}" \
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
  TICK_KIMG="${TICK_KIMG}" \
  SNAP_TICKS=1 \
  DUMP_TICKS="${DUMP_TICKS_RESUME}" \
  SEED=0 \
  ENV_MODE=venv \
  INSTALL_DEPS=0 \
  VENV_DIR="${VENV_DIR}" \
  EXTRA_TRAIN_ARGS="--cdro-n-steps-path=${n_steps} --cdro-step-size=0.02 --cdro-total-budget-rho=0.0 --cdro-time-horizon=1.0 --cdro-sigma-min=0.002 --cdro-sigma-max=80.0 --cdro-edm-ladder-mode=stochastic_stratified_quantile --cdro-per-example-sigma-ladders=True --attack-num-steps=1 --outer-attack-weight=1.0 --outer-clean-weight=0.0" \
    bash "${ROOT_DIR}/scripts/setup_and_train_cifar10.sh"
}

launch_and_eval() {
  local n_steps="$1"
  local run_dir
  local total_kimg
  local target_state
  local eval_outdir="${EVAL_SWEEP_ROOT}/n$(printf '%03d' "${n_steps}")"
  local manifest_csv="${eval_outdir}/cifar10_cdro_budget_manifest.csv"

  mkdir -p "${eval_outdir}"

  run_dir="$(find_run_dir "${n_steps}")"
  if [[ -z "${run_dir}" || ! -d "${run_dir}" ]]; then
    echo "[INFO] Launching fresh rho=0, N=${n_steps} to target WCU ${TARGET_WCU}"
    python "${ROOT_DIR}/scripts/launch_cifar20_cdro_budgeted.py" \
      --outdir-root "${TRAIN_OUTROOT}" \
      --target-wcu "${TARGET_WCU}" \
      --batch-size 1024 \
      --batch-gpu 1024 \
      --lr "${TRAIN_LR}" \
      --cdro-n-steps-path "${n_steps}" \
      --cdro-total-budget-rho 0.0 \
      --outer-attack-weight 1.0 \
      --outer-clean-weight 0.0 \
      --tick-kimg "${TICK_KIMG}" \
      --snap-ticks 1 \
      --dump-ticks "${DUMP_TICKS_FRESH}" \
      --env-mode venv \
      --venv-dir "${VENV_DIR}" \
      --install-deps 0 \
      --launch
    run_dir="$(find_run_dir "${n_steps}")"
  fi
  if [[ -z "${run_dir}" || ! -d "${run_dir}" ]]; then
    echo "[ERROR] Could not resolve run dir for N=${n_steps}"
    exit 1
  fi

  total_kimg="$(jq -r '.budget_plan.total_kimg_int' "${run_dir}/cdro_budget_plan.json")"
  if [[ -z "${total_kimg}" || "${total_kimg}" == "null" ]]; then
    echo "[ERROR] Could not resolve total_kimg_int from ${run_dir}/cdro_budget_plan.json"
    exit 1
  fi
  target_state="$(printf "%s/training-state-%06d.pt" "${run_dir}" "${total_kimg}")"

  if [[ ! -f "${target_state}" ]]; then
    resume_existing_run "${run_dir}" "${total_kimg}" "${n_steps}"
  else
    echo "[SKIP] Found ${target_state}; skipping training for N=${n_steps}"
  fi

  echo "[INFO] Completed training for N=${n_steps}; run_dir=${run_dir}"
  echo "[INFO] Building manifest at kimg=${total_kimg}"
  python "${ROOT_DIR}/scripts/build_cifar_cdro_fid_manifest.py" \
    --cdro-run-dir "${run_dir}" \
    --outdir "${eval_outdir}" \
    --kimg "${total_kimg}" \
    --batch-size 1024 \
    --seed 0 \
    --train-percent-label "20%" \
    --ref-path "${PYTORCH_FID_REF}"

  echo "[INFO] Running pytorch-fid eval for N=${n_steps}"
  python "${ROOT_DIR}/scripts/run_cifar_fid_manifest.py" \
    --manifest-csv "${manifest_csv}" \
    --env-mode venv \
    --venv-dir "${VENV_DIR}" \
    --install-deps 0 \
    --prepare-dataset 0 \
    --fid-backend pytorch_fid \
    --ref-mode path \
    --ref-path "${PYTORCH_FID_REF}" \
    --only-pending \
    --num-images 50000 \
    --gen-batch 128 \
    --fid-batch 64 \
    --gen-steps 18 \
    --nproc-per-node 1

  echo "[INFO] Running clean EDM probe for N=${n_steps}"
  python "${ROOT_DIR}/scripts/run_cifar_loss_probe_manifest.py" \
    --manifest-csv "${manifest_csv}" \
    --only-pending \
    --device cuda \
    --batch-size 512 \
    --max-images 10000 \
    --seed 0

  echo "[INFO] Final manifest row for N=${n_steps}:"
  tail -n 1 "${manifest_csv}"
}

echo "[INFO] Starting rho=0 N=${RUN_N_STEPS} baseline-80k workflow on $(hostname)"
echo "[INFO] Campaign tag: ${CAMPAIGN_TAG}"
echo "[INFO] Train outroot: ${TRAIN_OUTROOT}"
echo "[INFO] Eval sweep root: ${EVAL_SWEEP_ROOT}"
echo "[INFO] Run log: ${RUN_LOG}"
echo "[INFO] TRAIN_LR: ${TRAIN_LR}"
echo "[INFO] RUN_N_STEPS: ${RUN_N_STEPS}"
date -u
nvidia-smi || true

IFS=',' read -r -a run_n_steps_list <<< "${RUN_N_STEPS}"
for raw_n_steps in "${run_n_steps_list[@]}"; do
  n_steps="$(echo "${raw_n_steps}" | tr -d '[:space:]')"
  if [[ -z "${n_steps}" ]]; then
    continue
  fi
  case "${n_steps}" in
    4|8|16)
      launch_and_eval "${n_steps}"
      ;;
    *)
      echo "[ERROR] Unsupported RUN_N_STEPS entry: ${n_steps} (expected 4, 8, and/or 16)"
      exit 1
      ;;
  esac
done

echo "[OK] rho=0 N=${RUN_N_STEPS} baseline-80k workflow complete."
