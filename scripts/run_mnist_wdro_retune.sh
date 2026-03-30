#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

RUN_TAG_PREFIX="${RUN_TAG_PREFIX:-mnist-wdro-retune}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/training-runs}"
CANDIDATES_RAW="${CANDIDATES:-a b c}"
CANDIDATES="${CANDIDATES_RAW//,/ }"
BASELINE_EVAL_JSON="${BASELINE_EVAL_JSON:-}"
SUMMARY_JSON="${SUMMARY_JSON:-${RUN_ROOT}/${RUN_TAG_PREFIX}-summary.json}"
SUMMARY_MD="${SUMMARY_MD:-${RUN_ROOT}/${RUN_TAG_PREFIX}-summary.md}"

DURATION_MIMG="${DURATION_MIMG:-2.0}"
TICK="${TICK:-50}"
SNAP="${SNAP:-50}"
BATCH="${BATCH:-512}"
BATCH_GPU="${BATCH_GPU:-512}"
WORKERS="${WORKERS:-4}"
AUGMENT="${AUGMENT:-0}"
NUM_IMAGES="${NUM_IMAGES:-10000}"
GEN_BATCH="${GEN_BATCH:-256}"
FID_BATCH="${FID_BATCH:-128}"
GEN_STEPS="${GEN_STEPS:-35}"
MNIST_TRAIN_PERCENT="${MNIST_TRAIN_PERCENT:-100}"
MNIST_TRAIN_SEED="${MNIST_TRAIN_SEED:-0}"
PRECOND="${PRECOND:-wdroedm}"
ARCH="${ARCH:-ddpmpp}"

candidate_params() {
  local name="$1"
  case "${name}" in
    a)
      cat <<'EOF'
WDRO_WARMUP_RATIO=0.5
WDRO_M_EPOCHS=10
WDRO_K=1
WDRO_STEP_SIZE=0.002
WDRO_GAMMA=5.0
WDRO_P_ADV=0.3
EOF
      ;;
    b)
      cat <<'EOF'
WDRO_WARMUP_RATIO=0.75
WDRO_M_EPOCHS=20
WDRO_K=1
WDRO_STEP_SIZE=0.001
WDRO_GAMMA=10.0
WDRO_P_ADV=0.2
EOF
      ;;
    c)
      cat <<'EOF'
WDRO_WARMUP_RATIO=0.6
WDRO_M_EPOCHS=20
WDRO_K=1
WDRO_STEP_SIZE=0.0015
WDRO_GAMMA=8.0
WDRO_P_ADV=0.15
EOF
      ;;
    d)
      cat <<'EOF'
WDRO_WARMUP_RATIO=0.8
WDRO_M_EPOCHS=20
WDRO_K=1
WDRO_STEP_SIZE=0.00075
WDRO_GAMMA=10.0
WDRO_P_ADV=0.2
EOF
      ;;
    e)
      cat <<'EOF'
WDRO_WARMUP_RATIO=0.75
WDRO_M_EPOCHS=20
WDRO_K=1
WDRO_STEP_SIZE=0.001
WDRO_GAMMA=12.0
WDRO_P_ADV=0.15
EOF
      ;;
    f)
      cat <<'EOF'
WDRO_WARMUP_RATIO=0.7
WDRO_M_EPOCHS=15
WDRO_K=1
WDRO_STEP_SIZE=0.0008
WDRO_GAMMA=9.0
WDRO_P_ADV=0.2
EOF
      ;;
    g)
      cat <<'EOF'
WDRO_WARMUP_RATIO=0.85
WDRO_M_EPOCHS=20
WDRO_K=1
WDRO_STEP_SIZE=0.0006
WDRO_GAMMA=10.0
WDRO_P_ADV=0.2
EOF
      ;;
    h)
      cat <<'EOF'
WDRO_WARMUP_RATIO=0.8
WDRO_M_EPOCHS=20
WDRO_K=1
WDRO_STEP_SIZE=0.00075
WDRO_GAMMA=12.0
WDRO_P_ADV=0.15
EOF
      ;;
    i)
      cat <<'EOF'
WDRO_WARMUP_RATIO=0.9
WDRO_M_EPOCHS=20
WDRO_K=1
WDRO_STEP_SIZE=0.0005
WDRO_GAMMA=8.0
WDRO_P_ADV=0.15
EOF
      ;;
    j)
      cat <<'EOF'
WDRO_WARMUP_RATIO=0.85
WDRO_M_EPOCHS=20
WDRO_K=1
WDRO_STEP_SIZE=0.0005
WDRO_GAMMA=10.0
WDRO_P_ADV=0.2
EOF
      ;;
    k)
      cat <<'EOF'
WDRO_WARMUP_RATIO=0.85
WDRO_M_EPOCHS=20
WDRO_K=1
WDRO_STEP_SIZE=0.0006
WDRO_GAMMA=8.0
WDRO_P_ADV=0.2
EOF
      ;;
    l)
      cat <<'EOF'
WDRO_WARMUP_RATIO=0.9
WDRO_M_EPOCHS=20
WDRO_K=1
WDRO_STEP_SIZE=0.0006
WDRO_GAMMA=10.0
WDRO_P_ADV=0.2
EOF
      ;;
    *)
      echo "[ERROR] Unknown candidate preset: ${name}" >&2
      exit 1
      ;;
  esac
}

latest_eval_json() {
  local eval_root="$1"
  find "${eval_root}" -mindepth 2 -maxdepth 2 -name 'evaluation_result.json' | sort | tail -n 1
}

mkdir -p "${RUN_ROOT}"

for candidate in ${CANDIDATES}; do
  run_tag="${RUN_TAG_PREFIX}-${candidate}-100pct-v1"
  if [[ "${MNIST_TRAIN_PERCENT}" != "100" ]]; then
    run_tag="${RUN_TAG_PREFIX}-${candidate}-${MNIST_TRAIN_PERCENT}pct-v1"
  fi
  echo "[INFO] Running WDRO candidate ${candidate} -> ${run_tag}"

  # Export preset knobs for this invocation.
  while IFS='=' read -r key value; do
    export "${key}"="${value}"
  done < <(candidate_params "${candidate}")

  RUN_TAG="${run_tag}" \
  METHODS="wdro" \
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
  PRECOND="${PRECOND}" \
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
baseline_eval_json = os.environ.get("BASELINE_EVAL_JSON", "")
summary_json = Path(os.environ["SUMMARY_JSON"])
summary_md = Path(os.environ["SUMMARY_MD"])

rows = []
for candidate in candidates:
    run_tag = f"{run_tag_prefix}-{candidate}-100pct-v1"
    if mnist_train_percent != "100":
        run_tag = f"{run_tag_prefix}-{candidate}-{mnist_train_percent}pct-v1"
    result_path = run_root / run_tag / "eval" / "wdro"
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
if baseline_eval_json:
    payload = json.loads(Path(baseline_eval_json).read_text(encoding="utf-8"))
    baseline_row = {
        "label": "baseline_reference",
        "fid": float(payload["fid"]),
        "run_dir": payload["run_dir"],
        "eval_result": baseline_eval_json,
        "network_pkl": payload["network_pkl"],
    }

rows.sort(key=lambda item: item["fid"])
summary_json.write_text(
    json.dumps({"baseline_reference": baseline_row, "results": rows}, indent=2),
    encoding="utf-8",
)

lines = [
    "# MNIST WDRO Retune Summary",
    "",
]
if baseline_row is not None:
    lines.append(f"Baseline reference FID: `{baseline_row['fid']:.4f}`")
    lines.append("")
lines.extend(
    [
        "| Rank | Candidate | FID |",
        "| --- | --- | ---: |",
    ]
)
for idx, row in enumerate(rows, start=1):
    lines.append(f"| {idx} | `{row['candidate']}` | {row['fid']:.4f} |")

lines.extend(["", "## Artifacts", ""])
if baseline_row is not None:
    lines.append(f"- `baseline_reference` run: `{baseline_row['run_dir']}`")
    lines.append(f"- `baseline_reference` eval: `{baseline_row['eval_result']}`")
for row in rows:
    lines.append(f"- `candidate {row['candidate']}` run: `{row['run_dir']}`")
    lines.append(f"- `candidate {row['candidate']}` eval: `{row['eval_result']}`")

summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(summary_md)
PY

echo "[INFO] WDRO retune summary written to ${SUMMARY_MD}"
