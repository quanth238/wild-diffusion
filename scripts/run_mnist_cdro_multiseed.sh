#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

COLLECTION_TAG="${COLLECTION_TAG:-mnist-cdro-multiseed-v1}"
COLLECTION_ROOT="${COLLECTION_ROOT:-${ROOT_DIR}/training-runs/${COLLECTION_TAG}}"
SUMMARY_JSON="${SUMMARY_JSON:-${COLLECTION_ROOT}/summary.json}"
SUMMARY_MD="${SUMMARY_MD:-${COLLECTION_ROOT}/summary.md}"

SEEDS_RAW="${SEEDS:-1 2}"
SEEDS="${SEEDS_RAW//,/ }"
MNIST_TRAIN_PERCENT="${MNIST_TRAIN_PERCENT:-100}"
MNIST_TRAIN_SEED="${MNIST_TRAIN_SEED:-0}"
ARCH="${ARCH:-ddpmpp}"

BASELINE_DURATION_MIMG="${BASELINE_DURATION_MIMG:-3.0}"
BASELINE_MATCH_DURATION_MIMG="${BASELINE_MATCH_DURATION_MIMG:-3.4}"
CDRO_FT_DURATION_MIMG="${CDRO_FT_DURATION_MIMG:-0.4}"

BASELINE_BATCH="${BASELINE_BATCH:-512}"
BASELINE_BATCH_GPU="${BASELINE_BATCH_GPU:-512}"
CDRO_BATCH="${CDRO_BATCH:-96}"
CDRO_BATCH_GPU="${CDRO_BATCH_GPU:-96}"
WORKERS="${WORKERS:-4}"
AUGMENT="${AUGMENT:-0}"
GEN_STEPS="${GEN_STEPS:-35}"
NUM_IMAGES="${NUM_IMAGES:-10000}"
GEN_BATCH="${GEN_BATCH:-128}"
FID_BATCH="${FID_BATCH:-64}"
TICK="${TICK:-25}"
SNAP="${SNAP:-25}"
DUMP="${DUMP:-25}"

CDRO_MIX="${CDRO_MIX:-0.05}"
CDRO_ADV_STEPS="${CDRO_ADV_STEPS:-1}"
CDRO_STEP_SIZE="${CDRO_STEP_SIZE:-0.02}"
CDRO_MAX_DELTA="${CDRO_MAX_DELTA:-0.03}"
CDRO_RHO="${CDRO_RHO:-5e-5}"
CDRO_LAMBDA_INIT="${CDRO_LAMBDA_INIT:-0.001}"
CDRO_LAMBDA_LR="${CDRO_LAMBDA_LR:-5e-4}"
CDRO_START_KIMG="${CDRO_START_KIMG:-0.0}"
CDRO_RAMP_KIMG="${CDRO_RAMP_KIMG:-0.0}"
CDRO_SIGMA_FLOOR="${CDRO_SIGMA_FLOOR:-0.12}"
CDRO_SIGMA_CUT="${CDRO_SIGMA_CUT:-0.70}"
CDRO_GATE_POWER="${CDRO_GATE_POWER:-1.0}"
CDRO_DELTA_SPACE="${CDRO_DELTA_SPACE:-noise}"

mkdir -p "${COLLECTION_ROOT}"

latest_json() {
  local dir="$1"
  if [[ ! -d "${dir}" ]]; then
    return 1
  fi
  find "${dir}" -type f -name 'evaluation_result.json' | sort | tail -n 1
}

latest_run_dir() {
  local dir="$1"
  if [[ ! -d "${dir}" ]]; then
    return 1
  fi
  find "${dir}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1
}

for seed in ${SEEDS}; do
  baseline_tag="${COLLECTION_TAG}-seed${seed}-baseline3m"
  baseline_root="${ROOT_DIR}/training-runs/${baseline_tag}"
  baseline_eval_json="$(latest_json "${baseline_root}/eval/baseline" 2>/dev/null || true)"

  if [[ -z "${baseline_eval_json}" ]]; then
    echo "[INFO] seed=${seed}: training baseline 3.0 MIMG"
    RUN_TAG="${baseline_tag}" \
    METHODS="baseline" \
    DURATION_MIMG="${BASELINE_DURATION_MIMG}" \
    BATCH="${BASELINE_BATCH}" \
    BATCH_GPU="${BASELINE_BATCH_GPU}" \
    WORKERS="${WORKERS}" \
    AUGMENT="${AUGMENT}" \
    TICK="${TICK}" \
    SNAP="${SNAP}" \
    DUMP="${DUMP}" \
    NUM_IMAGES="${NUM_IMAGES}" \
    GEN_BATCH="${GEN_BATCH}" \
    FID_BATCH="${FID_BATCH}" \
    GEN_STEPS="${GEN_STEPS}" \
    MNIST_TRAIN_PERCENT="${MNIST_TRAIN_PERCENT}" \
    MNIST_TRAIN_SEED="${MNIST_TRAIN_SEED}" \
    ARCH="${ARCH}" \
    SEED="${seed}" \
    bash scripts/run_mnist_compare.sh
    baseline_eval_json="$(latest_json "${baseline_root}/eval/baseline")"
  else
    echo "[INFO] seed=${seed}: baseline 3.0 MIMG already available"
  fi

  baseline_run_dir="$(latest_run_dir "${baseline_root}/runs/baseline")"
  baseline_snap_3m="${baseline_run_dir}/network-snapshot-003000.pkl"
  baseline_state_3m="${baseline_run_dir}/training-state-003000.pt"
  if [[ ! -f "${baseline_snap_3m}" || ! -f "${baseline_state_3m}" ]]; then
    echo "[ERROR] seed=${seed}: missing baseline 3.0 MIMG snapshot/state under ${baseline_run_dir}" >&2
    exit 1
  fi

  baseline34_eval_root="${COLLECTION_ROOT}/seed${seed}/baseline34m/eval/baseline"
  baseline34_eval_json="$(latest_json "${baseline34_eval_root}" 2>/dev/null || true)"
  if [[ -z "${baseline34_eval_json}" ]]; then
    echo "[INFO] seed=${seed}: resuming baseline to 3.4 MIMG"
    RESUME="${baseline_state_3m}" \
    TRAINER="baseline" \
    PRECOND="wdroedm" \
    DURATION_MIMG="${BASELINE_MATCH_DURATION_MIMG}" \
    BATCH="${BASELINE_BATCH}" \
    BATCH_GPU="${BASELINE_BATCH_GPU}" \
    WORKERS="${WORKERS}" \
    AUGMENT="${AUGMENT}" \
    TICK="${TICK}" \
    SNAP="${SNAP}" \
    DUMP="${DUMP}" \
    ARCH="${ARCH}" \
    SEED="${seed}" \
    bash scripts/setup_and_train_mnist.sh

    baseline_snap_34m="${baseline_run_dir}/network-snapshot-003400.pkl"
    if [[ ! -f "${baseline_snap_34m}" ]]; then
      echo "[ERROR] seed=${seed}: missing baseline 3.4 MIMG snapshot after resume" >&2
      exit 1
    fi

    RUN_DIR="${baseline_run_dir}" \
    NETWORK_PKL="${baseline_snap_34m}" \
    EVAL_ROOT="${baseline34_eval_root}" \
    NUM_IMAGES="${NUM_IMAGES}" \
    GEN_BATCH="${GEN_BATCH}" \
    FID_BATCH="${FID_BATCH}" \
    GEN_STEPS="${GEN_STEPS}" \
    bash scripts/setup_and_eval_mnist.sh
    baseline34_eval_json="$(latest_json "${baseline34_eval_root}")"
  else
    echo "[INFO] seed=${seed}: baseline 3.4 MIMG eval already available"
  fi

  cdro_tag="${COLLECTION_TAG}-seed${seed}-cdroft04"
  cdro_root="${ROOT_DIR}/training-runs/${cdro_tag}"
  cdro_eval_json="$(latest_json "${cdro_root}/eval/cdro" 2>/dev/null || true)"
  if [[ -z "${cdro_eval_json}" ]]; then
    echo "[INFO] seed=${seed}: training baseline-initialized CDRO fine-tune"
    RUN_TAG="${cdro_tag}" \
    METHODS="cdro" \
    TRANSFER="${baseline_snap_3m}" \
    DURATION_MIMG="${CDRO_FT_DURATION_MIMG}" \
    BATCH="${CDRO_BATCH}" \
    BATCH_GPU="${CDRO_BATCH_GPU}" \
    WORKERS="${WORKERS}" \
    AUGMENT="${AUGMENT}" \
    TICK="${TICK}" \
    SNAP="${SNAP}" \
    DUMP="${DUMP}" \
    NUM_IMAGES="${NUM_IMAGES}" \
    GEN_BATCH="${GEN_BATCH}" \
    FID_BATCH="${FID_BATCH}" \
    GEN_STEPS="${GEN_STEPS}" \
    MNIST_TRAIN_PERCENT="${MNIST_TRAIN_PERCENT}" \
    MNIST_TRAIN_SEED="${MNIST_TRAIN_SEED}" \
    ARCH="${ARCH}" \
    SEED="${seed}" \
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
    cdro_eval_json="$(latest_json "${cdro_root}/eval/cdro")"
  else
    echo "[INFO] seed=${seed}: CDRO fine-tune already available"
  fi
done

export COLLECTION_ROOT SUMMARY_JSON SUMMARY_MD SEEDS
python3 - <<'PY'
import json
import math
import os
from pathlib import Path

collection_root = Path(os.environ["COLLECTION_ROOT"])
summary_json = Path(os.environ["SUMMARY_JSON"])
summary_md = Path(os.environ["SUMMARY_MD"])
seeds = [int(x) for x in os.environ["SEEDS"].split()]
collection_tag = collection_root.name
root_dir = collection_root.parents[1]
train_root = root_dir / "training-runs"

def latest_json(path: Path):
    matches = sorted(path.glob("**/evaluation_result.json"))
    return matches[-1] if matches else None

rows = []
for seed in seeds:
    baseline_root = train_root / f"{collection_tag}-seed{seed}-baseline3m"
    cdro_root = train_root / f"{collection_tag}-seed{seed}-cdroft04"
    baseline_3 = latest_json(baseline_root / "eval" / "baseline")
    baseline_34 = latest_json(collection_root / f"seed{seed}" / "baseline34m" / "eval" / "baseline")
    cdro = latest_json(cdro_root / "eval" / "cdro")
    if not (baseline_3 and baseline_34 and cdro):
        continue
    payloads = {
        "baseline_3.0mimg": json.loads(baseline_3.read_text(encoding="utf-8")),
        "baseline_3.4mimg": json.loads(baseline_34.read_text(encoding="utf-8")),
        "cdro_ft_0.4mimg": json.loads(cdro.read_text(encoding="utf-8")),
    }
    rows.append(
        {
            "seed": seed,
            "baseline_3.0mimg": {
                "fid": float(payloads["baseline_3.0mimg"]["fid"]),
                "eval_result": str(baseline_3),
                "run_dir": payloads["baseline_3.0mimg"]["run_dir"],
                "network_pkl": payloads["baseline_3.0mimg"]["network_pkl"],
            },
            "baseline_3.4mimg": {
                "fid": float(payloads["baseline_3.4mimg"]["fid"]),
                "eval_result": str(baseline_34),
                "run_dir": payloads["baseline_3.4mimg"]["run_dir"],
                "network_pkl": payloads["baseline_3.4mimg"]["network_pkl"],
            },
            "cdro_ft_0.4mimg": {
                "fid": float(payloads["cdro_ft_0.4mimg"]["fid"]),
                "eval_result": str(cdro),
                "run_dir": payloads["cdro_ft_0.4mimg"]["run_dir"],
                "network_pkl": payloads["cdro_ft_0.4mimg"]["network_pkl"],
            },
        }
    )

def summarize(key: str):
    vals = [row[key]["fid"] for row in rows]
    if not vals:
        return None
    mean = sum(vals) / len(vals)
    if len(vals) == 1:
        std = 0.0
    else:
        std = math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1))
    return {"mean": mean, "std": std, "n": len(vals)}

summary = {
    "collection_root": str(collection_root),
    "seeds": seeds,
    "rows": rows,
    "aggregates": {
        "baseline_3.0mimg": summarize("baseline_3.0mimg"),
        "baseline_3.4mimg": summarize("baseline_3.4mimg"),
        "cdro_ft_0.4mimg": summarize("cdro_ft_0.4mimg"),
    },
}
summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

lines = [
    "# MNIST CDRO Multi-Seed Summary",
    "",
    f"Collection root: `{collection_root}`",
    "",
    "## Aggregate FID",
    "",
    "| Model | Mean | Std | N |",
    "| --- | ---: | ---: | ---: |",
]
for key in ("baseline_3.0mimg", "baseline_3.4mimg", "cdro_ft_0.4mimg"):
    agg = summary["aggregates"][key]
    if agg is None:
        continue
    lines.append(f"| `{key}` | {agg['mean']:.4f} | {agg['std']:.4f} | {agg['n']} |")

lines.extend(["", "## Per Seed", "", "| Seed | baseline_3.0mimg | baseline_3.4mimg | cdro_ft_0.4mimg |", "| ---: | ---: | ---: | ---: |"])
for row in rows:
    lines.append(
        f"| {row['seed']} | {row['baseline_3.0mimg']['fid']:.4f} | "
        f"{row['baseline_3.4mimg']['fid']:.4f} | {row['cdro_ft_0.4mimg']['fid']:.4f} |"
    )

lines.extend(["", "## Artifacts", ""])
for row in rows:
    lines.append(f"- seed `{row['seed']}` baseline 3.0 eval: `{row['baseline_3.0mimg']['eval_result']}`")
    lines.append(f"- seed `{row['seed']}` baseline 3.4 eval: `{row['baseline_3.4mimg']['eval_result']}`")
    lines.append(f"- seed `{row['seed']}` CDRO eval: `{row['cdro_ft_0.4mimg']['eval_result']}`")

summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(summary_md)
PY

echo "[INFO] Multi-seed CDRO summary:"
echo "       ${SUMMARY_MD}"
