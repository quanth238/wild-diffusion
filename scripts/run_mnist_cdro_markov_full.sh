#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

RUN_TAG="${RUN_TAG:-mnist-cdro-markov-full-a-seed0}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/training-runs/${RUN_TAG}}"
SUMMARY_JSON="${SUMMARY_JSON:-${RUN_ROOT}/summary.json}"
SUMMARY_MD="${SUMMARY_MD:-${RUN_ROOT}/summary.md}"

BASELINE_NETWORK_PKL="${BASELINE_NETWORK_PKL:-${ROOT_DIR}/training-runs/mnist-compare-100pct-v3/runs/baseline/00000-mnist-32x32-train-cond-ddpmpp-wdroedm-baseline-gpus1-batch512-fp16/network-snapshot-003000.pkl}"
USE_TRANSFER="${USE_TRANSFER:-1}"
BASELINE_EVAL_JSON="${BASELINE_EVAL_JSON:-${ROOT_DIR}/training-runs/mnist-compare-100pct-v3/eval/baseline-3mimg/eval-20260330-024016/evaluation_result.json}"
PLUGIN_CDRO_EVAL_JSON="${PLUGIN_CDRO_EVAL_JSON:-${ROOT_DIR}/training-runs/mnist-cdro-finetune-a-100pct-v1x/eval-400kimg-10k/eval-20260330-121753/evaluation_result.json}"

DURATION_MIMG="${DURATION_MIMG:-0.05}"
BATCH="${BATCH:-32}"
BATCH_GPU="${BATCH_GPU:-32}"
WORKERS="${WORKERS:-4}"
NUM_IMAGES="${NUM_IMAGES:-5000}"
GEN_BATCH="${GEN_BATCH:-32}"
FID_BATCH="${FID_BATCH:-32}"
SEED="${SEED:-0}"

mkdir -p "${RUN_ROOT}"

train_cmd=(
  env
  "OUTDIR=${RUN_ROOT}/runs/cdro_markov_full"
  "TRAINER=baseline"
  "PRECOND=cdromarkovfull"
  "DURATION_MIMG=${DURATION_MIMG}"
  "BATCH=${BATCH}"
  "BATCH_GPU=${BATCH_GPU}"
  "WORKERS=${WORKERS}"
  "AUGMENT=0"
  "TICK=10"
  "SNAP=10"
  "DUMP=10"
  "SEED=${SEED}"
  "CDRO_MIX=${CDRO_MIX:-0.05}"
  "CDRO_RHO=${CDRO_RHO:-2e-5}"
  "CDRO_LAMBDA_INIT=${CDRO_LAMBDA_INIT:-0.001}"
  "CDRO_LAMBDA_LR=${CDRO_LAMBDA_LR:-5e-4}"
  "CDRO_START_KIMG=${CDRO_START_KIMG:-25}"
  "CDRO_RAMP_KIMG=${CDRO_RAMP_KIMG:-50}"
  "CDRO_CONTROL_CBASE=${CDRO_CONTROL_CBASE:-64}"
  "MARKOV_NUM_STEPS=${MARKOV_NUM_STEPS:-8}"
  "MARKOV_TOTAL_TIME=${MARKOV_TOTAL_TIME:-1.0}"
  "MARKOV_BETA_MIN=${MARKOV_BETA_MIN:-0.1}"
  "MARKOV_BETA_MAX=${MARKOV_BETA_MAX:-12.0}"
  "MARKOV_SDE_FAMILY=${MARKOV_SDE_FAMILY:-vp_cosine}"
  "MARKOV_WEIGHT_SCHEDULE=${MARKOV_WEIGHT_SCHEDULE:-uniform}"
  "MARKOV_CONTROL_LR=${MARKOV_CONTROL_LR:-2e-4}"
  "MARKOV_REVERSE_CONTROL_SCALE=${MARKOV_REVERSE_CONTROL_SCALE:-1.0}"
  "MARKOV_REVERSE_NOISE_SCALE=${MARKOV_REVERSE_NOISE_SCALE:-1.0}"
)
if [[ "${USE_TRANSFER}" == "1" ]]; then
  train_cmd+=("TRANSFER=${BASELINE_NETWORK_PKL}")
fi
train_cmd+=(bash scripts/setup_and_train_mnist.sh)
"${train_cmd[@]}"

RUN_DIR="$(find "${RUN_ROOT}/runs/cdro_markov_full" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
RUN_DIR="${RUN_DIR}" \
EVAL_ROOT="${RUN_ROOT}/eval/cdro_markov_full" \
NUM_IMAGES="${NUM_IMAGES}" \
GEN_BATCH="${GEN_BATCH}" \
FID_BATCH="${FID_BATCH}" \
bash scripts/setup_and_eval_mnist_markov.sh

export RUN_ROOT SUMMARY_JSON SUMMARY_MD BASELINE_EVAL_JSON PLUGIN_CDRO_EVAL_JSON
python3 - <<'PY'
import json
import os
from pathlib import Path

def read_eval(path_str):
    path = Path(path_str)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))

run_root = Path(os.environ["RUN_ROOT"])
evals = sorted((run_root / "eval" / "cdro_markov_full").glob("*/evaluation_result.json"))
if not evals:
    raise SystemExit("Missing Markov full evaluation result")
markov = read_eval(str(evals[-1]))
baseline = read_eval(os.environ["BASELINE_EVAL_JSON"])
plugin = read_eval(os.environ["PLUGIN_CDRO_EVAL_JSON"])
payload = {"baseline": baseline, "plugin_cdro": plugin, "markov_full": markov}
Path(os.environ["SUMMARY_JSON"]).write_text(json.dumps(payload, indent=2), encoding="utf-8")
lines = [
    "# MNIST Full Markov CDRO Summary",
    "",
    f"- baseline EDM FID: `{baseline['fid']:.4f}`" if baseline else "- baseline EDM FID: unavailable",
    f"- plug-in CDRO FID: `{plugin['fid']:.4f}`" if plugin else "- plug-in CDRO FID: unavailable",
    f"- full Markov CDRO FID: `{markov['fid']:.4f}`",
    "",
    "## Artifacts",
    "",
    f"- eval: `{evals[-1]}`",
]
Path(os.environ["SUMMARY_MD"]).write_text("\n".join(lines) + "\n", encoding="utf-8")
print(os.environ["SUMMARY_MD"])
PY
