#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

CAMPAIGN_TAG="${CAMPAIGN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
TRAIN_OUTROOT="${TRAIN_OUTROOT:-${ROOT_DIR}/training-runs/paper-cifar10-cdro-fp16-rho0-baseline80k_${CAMPAIGN_TAG}}"
EVAL_SWEEP_ROOT="${EVAL_SWEEP_ROOT:-${ROOT_DIR}/training-runs/fid-sweeps/cdro_rho0_n004_baseline80k_${CAMPAIGN_TAG}}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/training-runs/fid-sweeps/logs}"
RUN_LOG="${RUN_LOG:-${LOG_DIR}/cdro_rho0_n004_baseline80k_${CAMPAIGN_TAG}.log}"

export CAMPAIGN_TAG TRAIN_OUTROOT EVAL_SWEEP_ROOT LOG_DIR RUN_LOG
export RUN_N_STEPS="${RUN_N_STEPS:-4}"

exec bash "${ROOT_DIR}/scripts/run_cifar_rho0_n008_n016_baseline80k_h100.sh"
