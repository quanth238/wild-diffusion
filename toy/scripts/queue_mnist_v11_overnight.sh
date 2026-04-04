#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"

PY="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "${PY}" ]]; then
  echo "[error] missing venv python: ${PY}" >&2
  exit 1
fi

WAIT_PID="${1:-}"
LOGDIR="${REPO_ROOT}/toy_outputs/logs"
MNIST_TRAIN_PERCENT="${MNIST_TRAIN_PERCENT:-5}"
SEED="${SEED:-0}"
BASELINE_STEP="${BASELINE_STEP:-500}"

format_pct_label() {
  local pct="$1"
  "${PY}" - "${pct}" <<'PY'
import sys

pct = float(sys.argv[1])
text = f"{pct:g}".replace(".", "p")
print(f"{text}pct")
PY
}

PCT_LABEL="$(format_pct_label "${MNIST_TRAIN_PERCENT}")"
printf -v BASELINE_STEP_PAD "%05d" "${BASELINE_STEP}"
OUTDIR="${OUTDIR:-${REPO_ROOT}/toy_outputs/mnist_${PCT_LABEL}_v11_overnight}"
BASELINE_CKPT_DEFAULT="${REPO_ROOT}/toy_outputs/mnist_convergence_ckpt_${PCT_LABEL}_long/mnist_baseline_curve_ckpt_${PCT_LABEL}_long_${PCT_LABEL}_s${SEED}/checkpoints/baseline_step${BASELINE_STEP_PAD}.pt"
BASELINE_CKPT="${BASELINE_CKPT:-${BASELINE_CKPT_DEFAULT}}"

mkdir -p "${LOGDIR}" "${OUTDIR}"

if [[ ! -f "${BASELINE_CKPT}" ]]; then
  echo "[error] baseline checkpoint not found: ${BASELINE_CKPT}" >&2
  exit 1
fi

if [[ -z "${FID_DETECTOR_PATH:-}" ]]; then
  DEFAULT_FID_DETECTOR="/root/.cache/dnnlib/downloads/18d9c1159d16cd4cc6adf7db0f2dd2a9_https___api.ngc.nvidia.com_v2_models_nvidia_research_stylegan3_versions_1_files_metrics_inception-2015-12-05.pkl"
  if [[ -f "${DEFAULT_FID_DETECTOR}" ]]; then
    export FID_DETECTOR_PATH="${DEFAULT_FID_DETECTOR}"
  fi
fi

wait_for_pid() {
  local pid="$1"
  if [[ -z "${pid}" ]]; then
    return 0
  fi
  echo "[queue] waiting for pid=${pid} to finish before starting overnight v1.1 sweep"
  while kill -0 "${pid}" 2>/dev/null; do
    echo "[queue] pid=${pid} still running at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    sleep 60
  done
  echo "[queue] pid=${pid} is done; starting queued runs"
}

compute_v11_steps() {
  local inner_steps="$1"
  local path_steps="$2"
  "${PY}" - "${inner_steps}" "${path_steps}" <<'PY'
import math
import sys

inner_steps = int(sys.argv[1])
path_steps = int(sys.argv[2])
targets = [1300, 2600, 3900, 5200]
units_per_step = path_steps * (inner_steps + 3)
steps = [max(1, int(round(t / units_per_step))) for t in targets]
dedup = []
for s in steps:
    if not dedup or dedup[-1] != s:
        dedup.append(s)
print(",".join(str(s) for s in dedup))
PY
}

run_v11_case() {
  local prefix="$1"
  local inner_steps="$2"
  local step_size="$3"
  local rho="$4"
  local projection_mode="$5"
  local gamma="$6"
  local path_steps="$7"
  local steps_list
  local log_path

  steps_list="$(compute_v11_steps "${inner_steps}" "${path_steps}")"
  log_path="${LOGDIR}/${prefix}.log"

  echo "[queue] starting ${prefix}"
  echo "[queue] inner_steps=${inner_steps} path_steps=${path_steps} step_size=${step_size} rho=${rho} projection=${projection_mode} gamma=${gamma} steps=${steps_list}"

  "${PY}" toy/scripts/compare_fid_curve_methods.py \
    --methods 1.1 \
    --steps-list "${steps_list}" \
    --seed "${SEED}" \
    --dataset-kind mnist \
    --image-channels 1 \
    --image-size 32 \
    --mnist-train-percent "${MNIST_TRAIN_PERCENT}" \
    --mnist-val-percent 100 \
    --batch-size 256 \
    --hidden-dim 256 \
    --training-objective edm \
    --n-steps-path "${path_steps}" \
    --sigma-min 0.01 \
    --sigma-max 80 \
    --auto-log-normal-params \
    --use-ema-eval \
    --warmup-clean-steps 0 \
    --warmup-ramp-steps 0 \
    --device cuda \
    --require-cuda \
    --skip-existing \
    --outdir "${OUTDIR}" \
    --prefix "${prefix}" \
    --baseline-ckpt-source "${BASELINE_CKPT}" \
    --v11-inner-steps "${inner_steps}" \
    --v11-step-size "${step_size}" \
    --v11-total-budget-rho "${rho}" \
    --v11-projection-mode "${projection_mode}" \
    --v11-transport-gamma "${gamma}" \
    2>&1 | tee "${log_path}"

  echo "[queue] finished ${prefix}"
}

wait_for_pid "${WAIT_PID}"

run_v11_case "mnist_${PCT_LABEL}_v11_default_i1_n24_rho002_step5e4_global" 1 0.0005 0.02 global_remaining 2.0 24
run_v11_case "mnist_${PCT_LABEL}_v11_stepclip_i1_n24_rho002_step5e4" 1 0.0005 0.02 step_clip 2.0 24
run_v11_case "mnist_${PCT_LABEL}_v11_inner2_i2_n24_rho002_step5e4_global" 2 0.0005 0.02 global_remaining 2.0 24
run_v11_case "mnist_${PCT_LABEL}_v11_rho005_i1_n24_step5e4_global" 1 0.0005 0.05 global_remaining 2.0 24
run_v11_case "mnist_${PCT_LABEL}_v11_stepsz1e3_i1_n24_rho002_global" 1 0.0010 0.02 global_remaining 2.0 24
run_v11_case "mnist_${PCT_LABEL}_v11_default_i1_n12_rho002_step5e4_global" 1 0.0005 0.02 global_remaining 2.0 12
run_v11_case "mnist_${PCT_LABEL}_v11_default_i1_n32_rho002_step5e4_global" 1 0.0005 0.02 global_remaining 2.0 32

echo "[queue] overnight v1.1 sweep completed at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
