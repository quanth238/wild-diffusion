#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

RUN_TAG_PREFIX="${RUN_TAG_PREFIX:-mnist-cdro-fromscratch-followup}"
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

CDRO_ADV_STEPS="${CDRO_ADV_STEPS:-1}"
CDRO_STEP_SIZE="${CDRO_STEP_SIZE:-0.02}"
CDRO_MAX_DELTA="${CDRO_MAX_DELTA:-0.03}"
CDRO_LAMBDA_INIT="${CDRO_LAMBDA_INIT:-0.001}"
CDRO_LAMBDA_LR="${CDRO_LAMBDA_LR:-5e-4}"
CDRO_SIGMA_FLOOR="${CDRO_SIGMA_FLOOR:-0.12}"
CDRO_SIGMA_CUT="${CDRO_SIGMA_CUT:-0.70}"
CDRO_GATE_POWER="${CDRO_GATE_POWER:-1.0}"
CDRO_DELTA_SPACE="${CDRO_DELTA_SPACE:-noise}"
CDRO_START_KIMG="${CDRO_START_KIMG:-200}"
CDRO_RAMP_KIMG="${CDRO_RAMP_KIMG:-200}"

CANDIDATES_RAW="${CANDIDATES:-mix03 mix07 rho2e5 rho1e4}"
CANDIDATES="${CANDIDATES_RAW//,/ }"

mkdir -p "${RUN_ROOT}"

candidate_params() {
  local name="$1"
  case "${name}" in
    mix03)
      cat <<'EOF'
CDRO_MIX=0.03
CDRO_RHO=5e-5
EOF
      ;;
    mix07)
      cat <<'EOF'
CDRO_MIX=0.07
CDRO_RHO=5e-5
EOF
      ;;
    rho2e5)
      cat <<'EOF'
CDRO_MIX=0.05
CDRO_RHO=2e-5
EOF
      ;;
    rho1e4)
      cat <<'EOF'
CDRO_MIX=0.05
CDRO_RHO=1e-4
EOF
      ;;
    *)
      echo "[ERROR] Unknown follow-up candidate: ${name}" >&2
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

export RUN_ROOT RUN_TAG_PREFIX SUMMARY_JSON SUMMARY_MD SEED CANDIDATES

for candidate in ${CANDIDATES}; do
  while IFS='=' read -r key value; do
    export "${key}"="${value}"
  done < <(candidate_params "${candidate}")

  run_tag="${RUN_TAG_PREFIX}-${candidate}-seed${SEED}"
  run_root="${ROOT_DIR}/training-runs/${run_tag}"
  if [[ -n "$(latest_json "${run_root}/eval/cdro" 2>/dev/null || true)" ]]; then
    echo "[INFO] Candidate ${candidate} already complete"
    continue
  fi

  echo "[INFO] Running from-scratch CDRO follow-up candidate ${candidate}"
  RUN_TAG="${run_tag}" \
  RUN_ROOT="${ROOT_DIR}/training-runs/${run_tag}" \
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

python3 - <<'PY'
import json
import os
from pathlib import Path

run_root = Path(os.environ["RUN_ROOT"])
run_tag_prefix = os.environ["RUN_TAG_PREFIX"]
summary_json = Path(os.environ["SUMMARY_JSON"])
summary_md = Path(os.environ["SUMMARY_MD"])
seed = int(os.environ["SEED"])
candidates = os.environ["CANDIDATES"].split()

def latest_json(path: Path):
    matches = sorted(path.glob("*/evaluation_result.json"))
    return matches[-1] if matches else None

def read_last_stats(stats_path: Path):
    if not stats_path.is_file():
        return {}
    last = None
    with stats_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                last = json.loads(line)
    if last is None:
        return {}
    def extract(key):
        entry = last.get(key)
        if not entry:
            return None
        return float(entry["mean"])
    return {
        "lambda_dual": extract("CDRO/lambda_dual"),
        "transport_cost": extract("CDRO/mean_transport_cost"),
        "activation_scale": extract("CDRO/activation_scale"),
        "gate_mean": extract("CDRO/gate_mean"),
        "applied_delta_rms": extract("CDRO/applied_delta_rms"),
    }

rows = []
for candidate in candidates:
    base = run_root / f"{run_tag_prefix}-{candidate}-seed{seed}"
    eval_root = base / "eval" / "cdro"
    result_path = latest_json(eval_root)
    if result_path is None:
        continue
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    stats_files = sorted((base / "runs" / "cdro").glob("*/stats.jsonl"))
    stats = read_last_stats(stats_files[-1]) if stats_files else {}
    rows.append({
        "candidate": candidate,
        "fid": float(payload["fid"]),
        "run_dir": payload["run_dir"],
        "eval_result": str(result_path),
        "network_pkl": payload["network_pkl"],
        "stats": stats,
    })

rows.sort(key=lambda item: item["fid"])
summary_json.write_text(json.dumps({"seed": seed, "results": rows}, indent=2), encoding="utf-8")

lines = [
    "# MNIST CDRO From-Scratch Follow-Up",
    "",
    f"Seed: `{seed}`",
    "",
    "| Rank | Candidate | FID | Activation | Gate | Transport | Delta RMS |",
    "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
]
for idx, row in enumerate(rows, start=1):
    s = row["stats"]
    def fmt(x):
        return "-" if x is None else f"{x:.6g}"
    lines.append(
        f"| {idx} | `{row['candidate']}` | {row['fid']:.4f} | {fmt(s.get('activation_scale'))} | "
        f"{fmt(s.get('gate_mean'))} | {fmt(s.get('transport_cost'))} | {fmt(s.get('applied_delta_rms'))} |"
    )
lines.extend(["", "## Artifacts", ""])
for row in rows:
    lines.append(f"- `{row['candidate']}` eval: `{row['eval_result']}`")
summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(summary_md)
PY

echo "[INFO] From-scratch follow-up summary:"
echo "       ${SUMMARY_MD}"
