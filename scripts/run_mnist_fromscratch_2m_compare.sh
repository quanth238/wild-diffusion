#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

RUN_TAG="${RUN_TAG:-mnist-fromscratch-2m-compare-v1}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/training-runs/${RUN_TAG}}"
SUMMARY_JSON="${SUMMARY_JSON:-${RUN_ROOT}/summary.json}"
SUMMARY_MD="${SUMMARY_MD:-${RUN_ROOT}/summary.md}"

SEED="${SEED:-0}"
ARCH="${ARCH:-ddpmpp}"
MNIST_TRAIN_PERCENT="${MNIST_TRAIN_PERCENT:-100}"
MNIST_TRAIN_SEED="${MNIST_TRAIN_SEED:-0}"

TARGET_MIMG="${TARGET_MIMG:-2.0}"
TICK="${TICK:-25}"
SNAP="${SNAP:-25}"
DUMP="${DUMP:-25}"
BATCH="${BATCH:-96}"
BATCH_GPU="${BATCH_GPU:-96}"
WORKERS="${WORKERS:-4}"
AUGMENT="${AUGMENT:-0}"
NUM_IMAGES="${NUM_IMAGES:-10000}"
GEN_BATCH="${GEN_BATCH:-96}"
FID_BATCH="${FID_BATCH:-64}"
GEN_STEPS="${GEN_STEPS:-35}"

BASELINE_RESUME="${BASELINE_RESUME:-${ROOT_DIR}/training-runs/mnist-cdro-fromscratch-screen-v1-core-seed0/runs/baseline/00000-mnist-32x32-train-cond-ddpmpp-wdroedm-baseline-gpus1-batch96-fp16/training-state-001000.pt}"
WDRO_RESUME="${WDRO_RESUME:-${ROOT_DIR}/training-runs/mnist-cdro-fromscratch-screen-v1-core-seed0/runs/wdro/00000-mnist-32x32-train-cond-ddpmpp-wdroedm-wdro-gpus1-batch96-fp16/training-state-001000.pt}"
CDRO_RESUME="${CDRO_RESUME:-${ROOT_DIR}/training-runs/mnist-cdro-fromscratch-followup-v1-rho2e5-seed0/runs/cdro/00000-mnist-32x32-train-cond-ddpmpp-cdroedm-baseline-gpus1-batch96-fp16/training-state-001000.pt}"

latest_json() {
  local dir="$1"
  if [[ ! -d "${dir}" ]]; then
    return 1
  fi
  find "${dir}" -type f -name 'evaluation_result.json' | sort | tail -n 1
}

resume_and_eval() {
  local label="$1"
  local trainer="$2"
  local precond="$3"
  local resume="$4"
  local outdir="$5"
  local eval_root="$6"

  if [[ ! -f "${resume}" ]]; then
    echo "[ERROR] Missing resume state for ${label}: ${resume}" >&2
    exit 1
  fi

  RESUME="${resume}" \
  TRAINER="${trainer}" \
  PRECOND="${precond}" \
  DURATION_MIMG="${TARGET_MIMG}" \
  BATCH="${BATCH}" \
  BATCH_GPU="${BATCH_GPU}" \
  WORKERS="${WORKERS}" \
  AUGMENT="${AUGMENT}" \
  TICK="${TICK}" \
  SNAP="${SNAP}" \
  DUMP="${DUMP}" \
  ARCH="${ARCH}" \
  SEED="${SEED}" \
  bash scripts/setup_and_train_mnist.sh

  local run_dir
  run_dir="$(dirname "${resume}")"
  local network_pkl="${run_dir}/network-snapshot-002000.pkl"
  if [[ ! -f "${network_pkl}" ]]; then
    echo "[ERROR] Missing expected 2.0 MIMG snapshot for ${label}: ${network_pkl}" >&2
    exit 1
  fi

  RUN_DIR="${run_dir}" \
  NETWORK_PKL="${network_pkl}" \
  EVAL_ROOT="${eval_root}" \
  NUM_IMAGES="${NUM_IMAGES}" \
  GEN_BATCH="${GEN_BATCH}" \
  FID_BATCH="${FID_BATCH}" \
  GEN_STEPS="${GEN_STEPS}" \
  bash scripts/setup_and_eval_mnist.sh
}

mkdir -p "${RUN_ROOT}"

baseline_eval_root="${RUN_ROOT}/eval/baseline"
wdro_eval_root="${RUN_ROOT}/eval/wdro"
cdro_eval_root="${RUN_ROOT}/eval/cdro"

if [[ -z "$(latest_json "${baseline_eval_root}" 2>/dev/null || true)" ]]; then
  echo "[INFO] Resuming baseline to ${TARGET_MIMG} MIMG"
  resume_and_eval "baseline" "baseline" "wdroedm" "${BASELINE_RESUME}" "${RUN_ROOT}/runs/baseline" "${baseline_eval_root}"
fi
if [[ -z "$(latest_json "${wdro_eval_root}" 2>/dev/null || true)" ]]; then
  echo "[INFO] Resuming WDRO to ${TARGET_MIMG} MIMG"
  resume_and_eval "wdro" "wdro" "wdroedm" "${WDRO_RESUME}" "${RUN_ROOT}/runs/wdro" "${wdro_eval_root}"
fi
if [[ -z "$(latest_json "${cdro_eval_root}" 2>/dev/null || true)" ]]; then
  echo "[INFO] Resuming CDRO to ${TARGET_MIMG} MIMG"
  resume_and_eval "cdro" "baseline" "cdroedm" "${CDRO_RESUME}" "${RUN_ROOT}/runs/cdro" "${cdro_eval_root}"
fi

export RUN_ROOT SUMMARY_JSON SUMMARY_MD
python3 - <<'PY'
import json
import os
from pathlib import Path

run_root = Path(os.environ["RUN_ROOT"])
summary_json = Path(os.environ["SUMMARY_JSON"])
summary_md = Path(os.environ["SUMMARY_MD"])
rows = []
for method in ("baseline", "wdro", "cdro"):
    eval_root = run_root / "eval" / method
    results = sorted(eval_root.glob("*/evaluation_result.json"))
    if not results:
        continue
    result_path = results[-1]
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    rows.append({
        "method": method,
        "fid": float(payload["fid"]),
        "run_dir": payload["run_dir"],
        "network_pkl": payload["network_pkl"],
        "eval_result": str(result_path),
    })
rows.sort(key=lambda item: item["fid"])
summary_json.write_text(json.dumps({"run_root": str(run_root), "results": rows}, indent=2), encoding="utf-8")
lines = [
    "# MNIST From-Scratch 2.0 MIMG Compare",
    "",
    f"Run root: `{run_root}`",
    "",
    "| Rank | Method | FID |",
    "| --- | --- | ---: |",
]
for idx, row in enumerate(rows, start=1):
    lines.append(f"| {idx} | `{row['method']}` | {row['fid']:.4f} |")
lines.extend(["", "## Artifacts", ""])
for row in rows:
    lines.append(f"- `{row['method']}` eval: `{row['eval_result']}`")
summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(summary_md)
PY

echo "[INFO] From-scratch 2.0 MIMG summary: ${SUMMARY_MD}"
