#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

RUN_TAG_PREFIX="${RUN_TAG_PREFIX:-mnist-cdro-markov-lite}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/training-runs}"
CANDIDATES_RAW="${CANDIDATES:-a b}"
CANDIDATES="${CANDIDATES_RAW//,/ }"
SUMMARY_JSON="${SUMMARY_JSON:-${RUN_ROOT}/${RUN_TAG_PREFIX}-summary.json}"
SUMMARY_MD="${SUMMARY_MD:-${RUN_ROOT}/${RUN_TAG_PREFIX}-summary.md}"

BASELINE_NETWORK_PKL="${BASELINE_NETWORK_PKL:-${ROOT_DIR}/training-runs/mnist-compare-100pct-v3/runs/baseline/00000-mnist-32x32-train-cond-ddpmpp-wdroedm-baseline-gpus1-batch512-fp16/network-snapshot-003000.pkl}"
BASELINE_EVAL_JSON="${BASELINE_EVAL_JSON:-${ROOT_DIR}/training-runs/mnist-compare-100pct-v3/eval/baseline-3mimg/eval-20260330-024016/evaluation_result.json}"
PLUGIN_CDRO_EVAL_JSON="${PLUGIN_CDRO_EVAL_JSON:-${ROOT_DIR}/training-runs/mnist-cdro-finetune-a-100pct-v1x/eval-400kimg-10k/eval-20260330-121753/evaluation_result.json}"

DURATION_MIMG="${DURATION_MIMG:-0.2}"
TICK="${TICK:-25}"
SNAP="${SNAP:-25}"
DUMP="${DUMP:-25}"
BATCH="${BATCH:-64}"
BATCH_GPU="${BATCH_GPU:-64}"
WORKERS="${WORKERS:-4}"
AUGMENT="${AUGMENT:-0}"
NUM_IMAGES="${NUM_IMAGES:-10000}"
GEN_BATCH="${GEN_BATCH:-64}"
FID_BATCH="${FID_BATCH:-64}"
GEN_STEPS="${GEN_STEPS:-35}"
MNIST_TRAIN_PERCENT="${MNIST_TRAIN_PERCENT:-100}"
MNIST_TRAIN_SEED="${MNIST_TRAIN_SEED:-0}"
ARCH="${ARCH:-ddpmpp}"
SEED="${SEED:-0}"

candidate_params() {
  local name="$1"
  case "${name}" in
    a)
      cat <<'EOF'
CDRO_MIX=0.05
CDRO_MAX_DELTA=0.03
CDRO_RHO=2e-5
CDRO_LAMBDA_INIT=0.001
CDRO_LAMBDA_LR=5e-4
CDRO_START_KIMG=25
CDRO_RAMP_KIMG=50
CDRO_SIGMA_FLOOR=0.12
CDRO_SIGMA_CUT=0.70
CDRO_GATE_POWER=1.0
CDRO_DELTA_SPACE=noise
CDRO_CONTROL_CBASE=64
CDRO_CONTROL_DROPOUT=0.0
EOF
      ;;
    b)
      cat <<'EOF'
CDRO_MIX=0.03
CDRO_MAX_DELTA=0.025
CDRO_RHO=2e-5
CDRO_LAMBDA_INIT=0.001
CDRO_LAMBDA_LR=5e-4
CDRO_START_KIMG=50
CDRO_RAMP_KIMG=50
CDRO_SIGMA_FLOOR=0.12
CDRO_SIGMA_CUT=0.70
CDRO_GATE_POWER=1.0
CDRO_DELTA_SPACE=noise
CDRO_CONTROL_CBASE=48
CDRO_CONTROL_DROPOUT=0.0
EOF
      ;;
    *)
      echo "[ERROR] Unknown markov-lite candidate preset: ${name}" >&2
      exit 1
      ;;
  esac
}

mkdir -p "${RUN_ROOT}"

if [[ ! -f "${BASELINE_NETWORK_PKL}" ]]; then
  echo "[ERROR] BASELINE_NETWORK_PKL not found: ${BASELINE_NETWORK_PKL}" >&2
  exit 1
fi

for candidate in ${CANDIDATES}; do
  run_tag="${RUN_TAG_PREFIX}-${candidate}-seed${SEED}"
  run_root="${ROOT_DIR}/training-runs/${run_tag}"
  if compgen -G "${run_root}/eval/cdro_markov/*/evaluation_result.json" > /dev/null; then
    echo "[INFO] Candidate ${candidate} already complete"
    continue
  fi

  while IFS='=' read -r key value; do
    export "${key}"="${value}"
  done < <(candidate_params "${candidate}")

  echo "[INFO] Running non-plug-in CDRO candidate ${candidate} -> ${run_tag}"
  RUN_TAG="${run_tag}" \
  RUN_ROOT="${run_root}" \
  METHODS="cdro_markov" \
  TRANSFER="${BASELINE_NETWORK_PKL}" \
  DURATION_MIMG="${DURATION_MIMG}" \
  TICK="${TICK}" \
  SNAP="${SNAP}" \
  DUMP="${DUMP}" \
  BATCH="${BATCH}" \
  BATCH_GPU="${BATCH_GPU}" \
  WORKERS="${WORKERS}" \
  AUGMENT="${AUGMENT}" \
  NUM_IMAGES="${NUM_IMAGES}" \
  GEN_BATCH="${GEN_BATCH}" \
  FID_BATCH="${FID_BATCH}" \
  GEN_STEPS="${GEN_STEPS}" \
  MNIST_TRAIN_PERCENT="${MNIST_TRAIN_PERCENT}" \
  MNIST_TRAIN_SEED="${MNIST_TRAIN_SEED}" \
  ARCH="${ARCH}" \
  SEED="${SEED}" \
  CDRO_MIX="${CDRO_MIX}" \
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
  CDRO_CONTROL_CBASE="${CDRO_CONTROL_CBASE}" \
  CDRO_CONTROL_DROPOUT="${CDRO_CONTROL_DROPOUT}" \
  bash scripts/run_mnist_compare.sh
done

export RUN_ROOT RUN_TAG_PREFIX SUMMARY_JSON SUMMARY_MD CANDIDATES SEED BASELINE_EVAL_JSON PLUGIN_CDRO_EVAL_JSON
python3 - <<'PY'
import json
import os
from pathlib import Path

run_root = Path(os.environ["RUN_ROOT"])
run_tag_prefix = os.environ["RUN_TAG_PREFIX"]
summary_json = Path(os.environ["SUMMARY_JSON"])
summary_md = Path(os.environ["SUMMARY_MD"])
candidates = os.environ["CANDIDATES"].split()
seed = int(os.environ["SEED"])

def read_eval(path_str):
    path = Path(path_str)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "fid": float(payload["fid"]),
        "run_dir": payload["run_dir"],
        "network_pkl": payload["network_pkl"],
        "eval_result": str(path),
    }

rows = []
for candidate in candidates:
    base = run_root / f"{run_tag_prefix}-{candidate}-seed{seed}" / "eval" / "cdro_markov"
    evals = sorted(base.glob("*/evaluation_result.json"))
    if not evals:
        continue
    payload = json.loads(evals[-1].read_text(encoding="utf-8"))
    rows.append({
        "candidate": candidate,
        "fid": float(payload["fid"]),
        "run_dir": payload["run_dir"],
        "network_pkl": payload["network_pkl"],
        "eval_result": str(evals[-1]),
    })

rows.sort(key=lambda item: item["fid"])
payload = {
    "seed": seed,
    "baseline_reference": read_eval(os.environ["BASELINE_EVAL_JSON"]),
    "plugin_cdro_reference": read_eval(os.environ["PLUGIN_CDRO_EVAL_JSON"]),
    "results": rows,
}
summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

lines = [
    "# MNIST Non-Plug-In CDRO Summary",
    "",
    f"Seed: `{seed}`",
    "",
]

baseline_ref = payload["baseline_reference"]
if baseline_ref is not None:
    lines.extend([
        "## References",
        "",
        f"- baseline EDM FID: `{baseline_ref['fid']:.4f}`",
    ])
plugin_ref = payload["plugin_cdro_reference"]
if plugin_ref is not None:
    lines.append(f"- plug-in CDRO FID: `{plugin_ref['fid']:.4f}`")

lines.extend(["", "## Candidates", "", "| Rank | Candidate | FID |", "| --- | --- | ---: |"])
for idx, row in enumerate(rows, start=1):
    lines.append(f"| {idx} | `{row['candidate']}` | {row['fid']:.4f} |")

lines.extend(["", "## Artifacts", ""])
for row in rows:
    lines.append(f"- `{row['candidate']}` eval: `{row['eval_result']}`")

summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(summary_md)
PY

echo "[INFO] Non-plug-in CDRO summary: ${SUMMARY_MD}"
