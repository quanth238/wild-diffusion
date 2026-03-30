#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

RUN_TAG_PREFIX="${RUN_TAG_PREFIX:-mnist-cdro-finetune}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/training-runs}"
CANDIDATES_RAW="${CANDIDATES:-a b c}"
CANDIDATES="${CANDIDATES_RAW//,/ }"
SUMMARY_JSON="${SUMMARY_JSON:-${RUN_ROOT}/${RUN_TAG_PREFIX}-summary.json}"
SUMMARY_MD="${SUMMARY_MD:-${RUN_ROOT}/${RUN_TAG_PREFIX}-summary.md}"

BASELINE_NETWORK_PKL="${BASELINE_NETWORK_PKL:-${ROOT_DIR}/training-runs/mnist-compare-100pct-v3/runs/baseline/00000-mnist-32x32-train-cond-ddpmpp-wdroedm-baseline-gpus1-batch512-fp16/network-snapshot-003000.pkl}"
BASELINE_EVAL_JSON="${BASELINE_EVAL_JSON:-${ROOT_DIR}/training-runs/mnist-compare-100pct-v3/eval/baseline-3mimg/eval-20260330-024016/evaluation_result.json}"

DURATION_MIMG="${DURATION_MIMG:-0.1}"
TICK="${TICK:-25}"
SNAP="${SNAP:-25}"
BATCH="${BATCH:-96}"
BATCH_GPU="${BATCH_GPU:-96}"
WORKERS="${WORKERS:-4}"
AUGMENT="${AUGMENT:-0}"
NUM_IMAGES="${NUM_IMAGES:-5000}"
GEN_BATCH="${GEN_BATCH:-128}"
FID_BATCH="${FID_BATCH:-64}"
GEN_STEPS="${GEN_STEPS:-35}"
MNIST_TRAIN_PERCENT="${MNIST_TRAIN_PERCENT:-100}"
MNIST_TRAIN_SEED="${MNIST_TRAIN_SEED:-0}"
ARCH="${ARCH:-ddpmpp}"

candidate_params() {
  local name="$1"
  case "${name}" in
    a)
      cat <<'EOF'
CDRO_MIX=0.05
CDRO_ADV_STEPS=1
CDRO_STEP_SIZE=0.02
CDRO_MAX_DELTA=0.03
CDRO_RHO=5e-5
CDRO_LAMBDA_INIT=0.001
CDRO_LAMBDA_LR=5e-4
CDRO_SIGMA_FLOOR=0.12
CDRO_SIGMA_CUT=0.70
CDRO_GATE_POWER=1.0
CDRO_DELTA_SPACE=noise
EOF
      ;;
    b)
      cat <<'EOF'
CDRO_MIX=0.10
CDRO_ADV_STEPS=2
CDRO_STEP_SIZE=0.02
CDRO_MAX_DELTA=0.03
CDRO_RHO=5e-5
CDRO_LAMBDA_INIT=0.001
CDRO_LAMBDA_LR=5e-4
CDRO_SIGMA_FLOOR=0.12
CDRO_SIGMA_CUT=0.70
CDRO_GATE_POWER=1.0
CDRO_DELTA_SPACE=noise
EOF
      ;;
    c)
      cat <<'EOF'
CDRO_MIX=0.05
CDRO_ADV_STEPS=2
CDRO_STEP_SIZE=0.03
CDRO_MAX_DELTA=0.04
CDRO_RHO=1e-4
CDRO_LAMBDA_INIT=0.001
CDRO_LAMBDA_LR=5e-4
CDRO_SIGMA_FLOOR=0.15
CDRO_SIGMA_CUT=0.80
CDRO_GATE_POWER=1.0
CDRO_DELTA_SPACE=noise
EOF
      ;;
    *)
      echo "[ERROR] Unknown candidate preset: ${name}" >&2
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
  run_tag="${RUN_TAG_PREFIX}-${candidate}-100pct-v1"
  if [[ "${MNIST_TRAIN_PERCENT}" != "100" ]]; then
    run_tag="${RUN_TAG_PREFIX}-${candidate}-${MNIST_TRAIN_PERCENT}pct-v1"
  fi
  echo "[INFO] Running CDRO fine-tune candidate ${candidate} -> ${run_tag}"

  while IFS='=' read -r key value; do
    export "${key}"="${value}"
  done < <(candidate_params "${candidate}")

  RUN_TAG="${run_tag}" \
  METHODS="cdro" \
  TRANSFER="${BASELINE_NETWORK_PKL}" \
  DURATION_MIMG="${DURATION_MIMG}" \
  TICK="${TICK}" \
  SNAP="${SNAP}" \
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
  bash scripts/run_mnist_compare.sh
done

python3 - <<'PY'
import json
import os
from pathlib import Path

run_root = Path(os.environ["RUN_ROOT"])
run_tag_prefix = os.environ["RUN_TAG_PREFIX"]
candidates = os.environ["CANDIDATES"].split()
mnist_train_percent = os.environ["MNIST_TRAIN_PERCENT"]
summary_json = Path(os.environ["SUMMARY_JSON"])
summary_md = Path(os.environ["SUMMARY_MD"])
baseline_network_pkl = os.environ["BASELINE_NETWORK_PKL"]
baseline_eval_json = os.environ.get("BASELINE_EVAL_JSON", "")

rows = []
for candidate in candidates:
    run_tag = f"{run_tag_prefix}-{candidate}-100pct-v1"
    if mnist_train_percent != "100":
        run_tag = f"{run_tag_prefix}-{candidate}-{mnist_train_percent}pct-v1"
    result_path = run_root / run_tag / "eval" / "cdro"
    eval_files = sorted(result_path.glob("*/evaluation_result.json"))
    if not eval_files:
        continue
    payload = json.loads(eval_files[-1].read_text(encoding="utf-8"))
    rows.append(
        {
            "candidate": candidate,
            "fid": float(payload["fid"]),
            "run_dir": payload["run_dir"],
            "eval_result": str(eval_files[-1]),
            "network_pkl": payload["network_pkl"],
        }
    )

baseline_row = None
if baseline_eval_json and Path(baseline_eval_json).is_file():
    payload = json.loads(Path(baseline_eval_json).read_text(encoding="utf-8"))
    baseline_row = {
        "label": "baseline_reference",
        "fid": float(payload["fid"]),
        "run_dir": payload["run_dir"],
        "eval_result": baseline_eval_json,
        "network_pkl": payload["network_pkl"],
    }
else:
    baseline_row = {
        "label": "baseline_reference",
        "fid": None,
        "run_dir": None,
        "eval_result": None,
        "network_pkl": baseline_network_pkl,
    }

rows.sort(key=lambda item: item["fid"])
payload = {
    "baseline_reference": baseline_row,
    "results": rows,
}
summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

lines = [
    "# MNIST CDRO Fine-Tune Summary",
    "",
    f"Run root: `{run_root}`",
    "",
    f"Baseline init checkpoint: `{baseline_network_pkl}`",
]
if baseline_row["fid"] is not None:
    lines.extend(
        [
            "",
            "## Baseline Reference",
            "",
            f"- FID: `{baseline_row['fid']:.4f}`",
            f"- run: `{baseline_row['run_dir']}`",
            f"- eval: `{baseline_row['eval_result']}`",
        ]
    )

lines.extend(["", "## Candidates", "", "| Rank | Candidate | FID |", "| --- | --- | ---: |"])
for idx, row in enumerate(rows, start=1):
    lines.append(f"| {idx} | `{row['candidate']}` | {row['fid']:.4f} |")

lines.extend(["", "## Artifacts", ""])
for row in rows:
    lines.append(f"- `{row['candidate']}` run: `{row['run_dir']}`")
    lines.append(f"- `{row['candidate']}` eval: `{row['eval_result']}`")

summary_md.write_text("\\n".join(lines) + "\\n", encoding="utf-8")
print(summary_md)
PY

echo "[INFO] CDRO fine-tune sweep complete."
echo "[INFO] Summary markdown: ${SUMMARY_MD}"
echo "[INFO] Summary json: ${SUMMARY_JSON}"
