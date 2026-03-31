#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

RUN_TAG_PREFIX="${RUN_TAG_PREFIX:-mnist-cdro-fromscratch-screen}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/training-runs}"
SUMMARY_JSON="${SUMMARY_JSON:-${RUN_ROOT}/${RUN_TAG_PREFIX}-summary.json}"
SUMMARY_MD="${SUMMARY_MD:-${RUN_ROOT}/${RUN_TAG_PREFIX}-summary.md}"

SEED="${SEED:-0}"
MNIST_TRAIN_PERCENT="${MNIST_TRAIN_PERCENT:-100}"
MNIST_TRAIN_SEED="${MNIST_TRAIN_SEED:-0}"
ARCH="${ARCH:-ddpmpp}"

DURATION_MIMG="${DURATION_MIMG:-1.0}"
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

WDRO_WARMUP_RATIO="${WDRO_WARMUP_RATIO:-0.2}"
WDRO_M_EPOCHS="${WDRO_M_EPOCHS:-10}"
WDRO_K="${WDRO_K:-3}"
WDRO_STEP_SIZE="${WDRO_STEP_SIZE:-0.01}"
WDRO_GAMMA="${WDRO_GAMMA:-1.0}"
WDRO_P_ADV="${WDRO_P_ADV:-1.0}"

CDRO_MIX="${CDRO_MIX:-0.05}"
CDRO_ADV_STEPS="${CDRO_ADV_STEPS:-1}"
CDRO_STEP_SIZE="${CDRO_STEP_SIZE:-0.02}"
CDRO_MAX_DELTA="${CDRO_MAX_DELTA:-0.03}"
CDRO_RHO="${CDRO_RHO:-5e-5}"
CDRO_LAMBDA_INIT="${CDRO_LAMBDA_INIT:-0.001}"
CDRO_LAMBDA_LR="${CDRO_LAMBDA_LR:-5e-4}"
CDRO_SIGMA_FLOOR="${CDRO_SIGMA_FLOOR:-0.12}"
CDRO_SIGMA_CUT="${CDRO_SIGMA_CUT:-0.70}"
CDRO_GATE_POWER="${CDRO_GATE_POWER:-1.0}"
CDRO_DELTA_SPACE="${CDRO_DELTA_SPACE:-noise}"

CDRO_CANDIDATES_RAW="${CDRO_CANDIDATES:-w0 w20 w40}"
CDRO_CANDIDATES="${CDRO_CANDIDATES_RAW//,/ }"

mkdir -p "${RUN_ROOT}"

candidate_schedule() {
  local name="$1"
  case "${name}" in
    w0)
      cat <<'EOF'
CDRO_START_KIMG=0
CDRO_RAMP_KIMG=0
EOF
      ;;
    w20)
      cat <<'EOF'
CDRO_START_KIMG=200
CDRO_RAMP_KIMG=200
EOF
      ;;
    w40)
      cat <<'EOF'
CDRO_START_KIMG=400
CDRO_RAMP_KIMG=200
EOF
      ;;
    *)
      echo "[ERROR] Unknown CDRO candidate: ${name}" >&2
      exit 1
      ;;
  esac
}

latest_json() {
  local dir="$1"
  if [[ ! -d "${dir}" ]]; then
    return 1
  fi
  find "${dir}" -type f -name 'evaluation_result.json' | sort | tail -n 1
}

core_tag="${RUN_TAG_PREFIX}-core-seed${SEED}"
core_root="${ROOT_DIR}/training-runs/${core_tag}"

if [[ -z "$(latest_json "${core_root}/eval/baseline" 2>/dev/null || true)" || -z "$(latest_json "${core_root}/eval/wdro" 2>/dev/null || true)" ]]; then
  echo "[INFO] Running baseline and WDRO from scratch screening core"
  RUN_TAG="${core_tag}" \
  METHODS="baseline wdro" \
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
  WDRO_WARMUP_RATIO="${WDRO_WARMUP_RATIO}" \
  WDRO_M_EPOCHS="${WDRO_M_EPOCHS}" \
  WDRO_K="${WDRO_K}" \
  WDRO_STEP_SIZE="${WDRO_STEP_SIZE}" \
  WDRO_GAMMA="${WDRO_GAMMA}" \
  WDRO_P_ADV="${WDRO_P_ADV}" \
  bash scripts/run_mnist_compare.sh
fi

for candidate in ${CDRO_CANDIDATES}; do
  while IFS='=' read -r key value; do
    export "${key}"="${value}"
  done < <(candidate_schedule "${candidate}")

  cdro_tag="${RUN_TAG_PREFIX}-${candidate}-seed${SEED}"
  cdro_root="${ROOT_DIR}/training-runs/${cdro_tag}"
  if [[ -n "$(latest_json "${cdro_root}/eval/cdro" 2>/dev/null || true)" ]]; then
    echo "[INFO] CDRO candidate ${candidate} already available"
    continue
  fi

  echo "[INFO] Running CDRO from scratch screening candidate ${candidate}"
  RUN_TAG="${cdro_tag}" \
  METHODS="cdro" \
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
  CDRO_ADV_STEPS="${CDRO_ADV_STEPS}" \
  CDRO_STEP_SIZE="${CDRO_STEP_SIZE}" \
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
  bash scripts/run_mnist_compare.sh
done

export RUN_ROOT RUN_TAG_PREFIX SUMMARY_JSON SUMMARY_MD SEED CDRO_CANDIDATES
python3 - <<'PY'
import json
import os
from pathlib import Path

run_root = Path(os.environ["RUN_ROOT"])
run_tag_prefix = os.environ["RUN_TAG_PREFIX"]
summary_json = Path(os.environ["SUMMARY_JSON"])
summary_md = Path(os.environ["SUMMARY_MD"])
seed = int(os.environ["SEED"])
cdro_candidates = os.environ["CDRO_CANDIDATES"].split()

def read_result(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "fid": float(payload["fid"]),
        "run_dir": payload["run_dir"],
        "network_pkl": payload["network_pkl"],
        "eval_result": str(path),
    }

rows = {}
core_root = run_root / f"{run_tag_prefix}-core-seed{seed}" / "eval"
baseline_jsons = sorted((core_root / "baseline").glob("*/evaluation_result.json"))
wdro_jsons = sorted((core_root / "wdro").glob("*/evaluation_result.json"))
if baseline_jsons:
    rows["baseline"] = read_result(baseline_jsons[-1])
if wdro_jsons:
    rows["wdro"] = read_result(wdro_jsons[-1])
for candidate in cdro_candidates:
    eval_root = run_root / f"{run_tag_prefix}-{candidate}-seed{seed}" / "eval" / "cdro"
    cdro_jsons = sorted(eval_root.glob("*/evaluation_result.json"))
    if cdro_jsons:
        rows[f"cdro_{candidate}"] = read_result(cdro_jsons[-1])

summary_json.write_text(json.dumps({"seed": seed, "results": rows}, indent=2), encoding="utf-8")

ordered = sorted(rows.items(), key=lambda item: item[1]["fid"])
lines = [
    "# MNIST CDRO From-Scratch Screen",
    "",
    f"Seed: `{seed}`",
    "",
    "| Rank | Model | FID |",
    "| --- | --- | ---: |",
]
for idx, (name, row) in enumerate(ordered, start=1):
    lines.append(f"| {idx} | `{name}` | {row['fid']:.4f} |")
lines.extend(["", "## Artifacts", ""])
for name, row in ordered:
    lines.append(f"- `{name}` run: `{row['run_dir']}`")
    lines.append(f"- `{name}` eval: `{row['eval_result']}`")
summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(summary_md)
PY

echo "[INFO] From-scratch screening summary:"
echo "       ${SUMMARY_MD}"
