#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${ROOT_DIR}/.." && pwd)"

SLURM_WRAPPER="${SLURM_WRAPPER:-${WORKSPACE_ROOT}/scripts/slurm_workflow.sh}"
PROJECT_DIR="${PROJECT_DIR:-${ROOT_DIR}}"
SERVER_STORAGE_ROOT="${SERVER_STORAGE_ROOT:-/mnt/data/quanth}"
EXPERIMENTS_DIR="${EXPERIMENTS_DIR:-${SERVER_STORAGE_ROOT}/experiments}"
SLURM_LOG_DIR="${SLURM_LOG_DIR:-${SERVER_STORAGE_ROOT}/slurm_logs}"
CONDA_INIT="${CONDA_INIT:-/mnt/data/quanth/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-quanth}"

JOB_PREFIX="${JOB_PREFIX:-wd-cifar20-theory}"
TAG="${TAG:-$(date +%Y%m%d-%H%M%S)}"
OUTROOT="${OUTROOT:-${SERVER_STORAGE_ROOT}/experiments/wild-diffusion/cifar20_theory_${TAG}}"
SEEDS="${SEEDS:-0,1}"

DATA_ROOT="${DATA_ROOT:-${SERVER_STORAGE_ROOT}/datasets}"
CIFAR_DIR="${CIFAR_DIR:-${DATA_ROOT}/cifar10-32x32}"
CIFAR_TRAIN_PERCENT="${CIFAR_TRAIN_PERCENT:-20}"
CIFAR_TRAIN_SEED="${CIFAR_TRAIN_SEED:-0}"

DURATION_MIMG="${DURATION_MIMG:-8}"
BATCH="${BATCH:-1024}"
BATCH_GPU="${BATCH_GPU:-512}"
FP16="${FP16:-1}"
LR="${LR:-1e-4}"
WORKERS="${WORKERS:-16}"

WDRO_WARMUP_RATIO="${WDRO_WARMUP_RATIO:-0.2}"
WDRO_M_EPOCHS="${WDRO_M_EPOCHS:-80}"
WDRO_K="${WDRO_K:-2}"
WDRO_STEP_SIZE="${WDRO_STEP_SIZE:-0.01}"
WDRO_GAMMA="${WDRO_GAMMA:-1.0}"
BASELINE_P_ADV="${BASELINE_P_ADV:-0.0}"
ROBUST_P_ADV="${ROBUST_P_ADV:-0.3}"

DEBUG_EVAL="${DEBUG_EVAL:-1}"
DEBUG_EVAL_INIT="${DEBUG_EVAL_INIT:-1}"
DEBUG_EVAL_NUM="${DEBUG_EVAL_NUM:-128}"
DEBUG_EVAL_STEPS="${DEBUG_EVAL_STEPS:-18}"
DEBUG_EVAL_BATCH="${DEBUG_EVAL_BATCH:-64}"
DEBUG_EVAL_VISUAL="${DEBUG_EVAL_VISUAL:-32}"
DEBUG_ADV_VISUAL="${DEBUG_ADV_VISUAL:-16}"

NUM_IMAGES="${NUM_IMAGES:-5000}"
GEN_BATCH="${GEN_BATCH:-128}"
FID_BATCH="${FID_BATCH:-64}"
GEN_STEPS="${GEN_STEPS:-18}"
REF_MODE="${REF_MODE:-path}"
REF_PATH="${REF_PATH:-${DATA_ROOT}/fid-refs/cifar10-32x32.npz}"
FID_DETECTOR_PATH="${FID_DETECTOR_PATH:-}"

DEFAULT_PARTITION="${DEFAULT_PARTITION:-main}"
DEFAULT_ACCOUNT="${DEFAULT_ACCOUNT:-normal}"
DEFAULT_QOS="${DEFAULT_QOS:-normal}"
DEFAULT_GPUS="${DEFAULT_GPUS:-1}"
DEFAULT_CPUS="${DEFAULT_CPUS:-16}"
DEFAULT_MEM="${DEFAULT_MEM:-64G}"
TRAIN_TIME="${TRAIN_TIME:-08:00:00}"
EVAL_TIME="${EVAL_TIME:-04:00:00}"

mkdir -p "${OUTROOT}" "${EXPERIMENTS_DIR}" "${SLURM_LOG_DIR}"

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "[ERROR] missing command: $1" >&2
    exit 1
  }
}

require_path() {
  local path="$1"
  [[ -e "${path}" ]] || {
    echo "[ERROR] missing path: ${path}" >&2
    exit 1
  }
}

extract_job_id() {
  awk '/Submitted batch job/ {print $4}' | tail -n 1
}

submit_train_job() {
  local mode="$1"
  local seed="$2"
  local p_adv="$3"
  local job_name="${JOB_PREFIX}-${mode}-s${seed}"
  local train_outdir="${OUTROOT}/${mode}_s${seed}"
  mkdir -p "${train_outdir}"

  local submit_out
  submit_out="$(
    PROJECT_DIR="${PROJECT_DIR}" \
    SERVER_STORAGE_ROOT="${SERVER_STORAGE_ROOT}" \
    EXPERIMENTS_DIR="${EXPERIMENTS_DIR}" \
    SLURM_LOG_DIR="${SLURM_LOG_DIR}" \
    DEFAULT_PARTITION="${DEFAULT_PARTITION}" \
    DEFAULT_ACCOUNT="${DEFAULT_ACCOUNT}" \
    DEFAULT_QOS="${DEFAULT_QOS}" \
    DEFAULT_GPUS="${DEFAULT_GPUS}" \
    DEFAULT_CPUS="${DEFAULT_CPUS}" \
    DEFAULT_MEM="${DEFAULT_MEM}" \
    DEFAULT_TIME="${TRAIN_TIME}" \
    CONDA_INIT="${CONDA_INIT}" \
    CONDA_ENV="${CONDA_ENV}" \
    "${SLURM_WRAPPER}" submit "${job_name}" -- \
    env ENV_MODE=conda EXPECTED_CONDA_ENV="${CONDA_ENV}" STRICT_CONDA_ENV=1 \
        INSTALL_DEPS=0 DATASET_ONLY=0 CIFAR_ALLOW_DOWNLOAD=0 \
        DATA_ROOT="${DATA_ROOT}" CIFAR_DIR="${CIFAR_DIR}" OUTDIR="${train_outdir}" \
        FID_DETECTOR_PATH="${FID_DETECTOR_PATH}" \
        SEED="${seed}" DURATION_MIMG="${DURATION_MIMG}" BATCH="${BATCH}" BATCH_GPU="${BATCH_GPU}" \
        FP16="${FP16}" LR="${LR}" WORKERS="${WORKERS}" \
        CIFAR_TRAIN_PERCENT="${CIFAR_TRAIN_PERCENT}" CIFAR_TRAIN_SEED="${CIFAR_TRAIN_SEED}" \
        WDRO_WARMUP_RATIO="${WDRO_WARMUP_RATIO}" WDRO_M_EPOCHS="${WDRO_M_EPOCHS}" \
        WDRO_K="${WDRO_K}" WDRO_STEP_SIZE="${WDRO_STEP_SIZE}" WDRO_GAMMA="${WDRO_GAMMA}" \
        WDRO_P_ADV="${p_adv}" DEBUG_EVAL="${DEBUG_EVAL}" DEBUG_EVAL_INIT="${DEBUG_EVAL_INIT}" \
        DEBUG_EVAL_NUM="${DEBUG_EVAL_NUM}" DEBUG_EVAL_STEPS="${DEBUG_EVAL_STEPS}" \
        DEBUG_EVAL_BATCH="${DEBUG_EVAL_BATCH}" DEBUG_EVAL_VISUAL="${DEBUG_EVAL_VISUAL}" \
        DEBUG_ADV_VISUAL="${DEBUG_ADV_VISUAL}" \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        bash scripts/setup_and_train_cifar10.sh
  )"

  local job_id
  job_id="$(printf '%s\n' "${submit_out}" | extract_job_id)"
  if [[ -z "${job_id}" ]]; then
    echo "[ERROR] failed to parse training job id for ${job_name}" >&2
    printf '%s\n' "${submit_out}" >&2
    exit 1
  fi

  printf '%s\n' "${submit_out}"
  printf '%s\t%s\t%s\t%s\n' "${mode}" "${seed}" "${job_id}" "${train_outdir}"
}

submit_eval_job() {
  local mode="$1"
  local seed="$2"
  local dep_job_id="$3"
  local train_outdir="$4"

  local job_name="${JOB_PREFIX}-${mode}-s${seed}-eval"
  local ts
  ts="$(date '+%Y%m%d-%H%M%S')"
  local exp_dir="${EXPERIMENTS_DIR}/${job_name}-${ts}"
  mkdir -p "${exp_dir}" "${SLURM_LOG_DIR}"

  local run_script="${exp_dir}/run.sh"
  cat > "${run_script}" <<EOF
#!/usr/bin/env bash
set -euo pipefail

EXP_DIR="${exp_dir}"
EXIT_FILE="\${EXP_DIR}/run.exitcode"
finish() {
  rc=\$?
  echo "\${rc}" > "\${EXIT_FILE}"
  echo "End time: \$(date '+%F %T')"
  echo "Exit code: \${rc}"
}
trap finish EXIT

if [[ -f "${CONDA_INIT}" ]]; then
  source "${CONDA_INIT}"
  conda activate "${CONDA_ENV}"
else
  echo "[ERROR] CONDA_INIT not found: ${CONDA_INIT}" >&2
  exit 1
fi

cd "${PROJECT_DIR}"

RUN_DIR="\$(find "${train_outdir}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
if [[ -z "\${RUN_DIR}" || ! -d "\${RUN_DIR}" ]]; then
  echo "[ERROR] Could not resolve training run dir inside ${train_outdir}" >&2
  exit 1
fi

echo "Start time: \$(date '+%F %T')"
echo "Host: \$(hostname)"
echo "Project dir: \$(pwd)"
echo "Resolved RUN_DIR: \${RUN_DIR}"

env ENV_MODE=conda EXPECTED_CONDA_ENV="${CONDA_ENV}" STRICT_CONDA_ENV=1 \
    INSTALL_DEPS=0 DATA_ROOT="${DATA_ROOT}" CIFAR_DIR="${CIFAR_DIR}" \
    FID_DETECTOR_PATH="${FID_DETECTOR_PATH}" \
    RUN_DIR="\${RUN_DIR}" NUM_IMAGES="${NUM_IMAGES}" GEN_BATCH="${GEN_BATCH}" \
    FID_BATCH="${FID_BATCH}" GEN_STEPS="${GEN_STEPS}" \
    REF_MODE="${REF_MODE}" REF_PATH="${REF_PATH}" \
    bash scripts/setup_and_eval_cifar10.sh
EOF
  chmod +x "${run_script}"

  local sbatch_out
  sbatch_out="$(
    sbatch \
      --job-name="${job_name}" \
      --dependency="afterok:${dep_job_id}" \
      --chdir="${PROJECT_DIR}" \
      --output="${SLURM_LOG_DIR}/slurm-%j.out" \
      --error="${SLURM_LOG_DIR}/slurm-%j.err" \
      --open-mode=append \
      --time="${EVAL_TIME}" \
      --cpus-per-task="${DEFAULT_CPUS}" \
      --mem="${DEFAULT_MEM}" \
      --partition="${DEFAULT_PARTITION}" \
      --account="${DEFAULT_ACCOUNT}" \
      --qos="${DEFAULT_QOS}" \
      --gres="gpu:${DEFAULT_GPUS}" \
      "${run_script}"
  )"

  local eval_job_id
  eval_job_id="$(printf '%s\n' "${sbatch_out}" | extract_job_id)"
  if [[ -z "${eval_job_id}" ]]; then
    echo "[ERROR] failed to parse eval job id for ${job_name}" >&2
    printf '%s\n' "${sbatch_out}" >&2
    exit 1
  fi

  cat > "${exp_dir}/job.info" <<EOF
job_id=${eval_job_id}
job_name=${job_name}
dependency=afterok:${dep_job_id}
exp_dir=${exp_dir}
stdout_log=${SLURM_LOG_DIR}/slurm-${eval_job_id}.out
stderr_log=${SLURM_LOG_DIR}/slurm-${eval_job_id}.err
exit_code_file=${exp_dir}/run.exitcode
project_dir=${PROJECT_DIR}
train_outdir=${train_outdir}
submitted_at=$(date '+%F %T')
command=${run_script}
EOF

  printf '%s\n' "${sbatch_out}"
  printf '%s\t%s\t%s\t%s\n' "${mode}" "${seed}" "${eval_job_id}" "${exp_dir}"
}

main() {
  require_cmd sbatch
  require_path "${SLURM_WRAPPER}"
  require_path "${PROJECT_DIR}"
  require_path "${CIFAR_DIR}"
  require_path "${REF_PATH}"

  local -a seed_list
  IFS=',' read -r -a seed_list <<< "${SEEDS}"

  echo "[INFO] PROJECT_DIR=${PROJECT_DIR}"
  echo "[INFO] OUTROOT=${OUTROOT}"
  echo "[INFO] SEEDS=${SEEDS}"
  echo "[INFO] Train profile: duration=${DURATION_MIMG} mimg, batch=${BATCH}, batch_gpu=${BATCH_GPU}, lr=${LR}"
  echo "[INFO] Robust profile: warmup_ratio=${WDRO_WARMUP_RATIO}, m_epochs=${WDRO_M_EPOCHS}, k=${WDRO_K}, p_adv=${ROBUST_P_ADV}"
  echo "[INFO] Eval profile: num_images=${NUM_IMAGES}, gen_steps=${GEN_STEPS}, ref=${REF_PATH}"
  echo

  for seed in "${seed_list[@]}"; do
    seed="$(echo "${seed}" | xargs)"
    [[ -n "${seed}" ]] || continue

    local train_base_info
    train_base_info="$(submit_train_job baseline "${seed}" "${BASELINE_P_ADV}" | tail -n 1)"
    IFS=$'\t' read -r _ _ train_base_job_id train_base_outdir <<< "${train_base_info}"

    local eval_base_info
    eval_base_info="$(submit_eval_job baseline "${seed}" "${train_base_job_id}" "${train_base_outdir}" | tail -n 1)"

    local train_robust_info
    train_robust_info="$(submit_train_job robust "${seed}" "${ROBUST_P_ADV}" | tail -n 1)"
    IFS=$'\t' read -r _ _ train_robust_job_id train_robust_outdir <<< "${train_robust_info}"

    local eval_robust_info
    eval_robust_info="$(submit_eval_job robust "${seed}" "${train_robust_job_id}" "${train_robust_outdir}" | tail -n 1)"

    echo "[SUMMARY] seed=${seed}"
    echo "  baseline_train_job=${train_base_job_id} outdir=${train_base_outdir}"
    echo "  baseline_eval_job=$(printf '%s' "${eval_base_info}" | cut -f3)"
    echo "  robust_train_job=${train_robust_job_id} outdir=${train_robust_outdir}"
    echo "  robust_eval_job=$(printf '%s' "${eval_robust_info}" | cut -f3)"
    echo
  done
}

main "$@"
