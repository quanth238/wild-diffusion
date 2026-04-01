#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

INITIAL_RUN_TAG_PREFIX="${INITIAL_RUN_TAG_PREFIX:-mnist-cdro-markov-lite}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/training-runs}"
INITIAL_SUMMARY_JSON="${INITIAL_SUMMARY_JSON:-${RUN_ROOT}/${INITIAL_RUN_TAG_PREFIX}-summary.json}"
FOLLOWUP_TAG="${FOLLOWUP_TAG:-${INITIAL_RUN_TAG_PREFIX}-best-extend}"
FOLLOWUP_ROOT="${FOLLOWUP_ROOT:-${ROOT_DIR}/training-runs/${FOLLOWUP_TAG}}"
FOLLOWUP_SUMMARY_JSON="${FOLLOWUP_SUMMARY_JSON:-${RUN_ROOT}/${FOLLOWUP_TAG}-summary.json}"
FOLLOWUP_SUMMARY_MD="${FOLLOWUP_SUMMARY_MD:-${RUN_ROOT}/${FOLLOWUP_TAG}-summary.md}"

SEED="${SEED:-0}"
TARGET_MIMG="${TARGET_MIMG:-0.4}"
BATCH="${BATCH:-64}"
BATCH_GPU="${BATCH_GPU:-64}"
WORKERS="${WORKERS:-4}"
NUM_IMAGES="${NUM_IMAGES:-10000}"
GEN_BATCH="${GEN_BATCH:-64}"
FID_BATCH="${FID_BATCH:-64}"
GEN_STEPS="${GEN_STEPS:-35}"

BASELINE_EVAL_JSON="${BASELINE_EVAL_JSON:-${ROOT_DIR}/training-runs/mnist-compare-100pct-v3/eval/baseline-3mimg/eval-20260330-024016/evaluation_result.json}"
PLUGIN_CDRO_EVAL_JSON="${PLUGIN_CDRO_EVAL_JSON:-${ROOT_DIR}/training-runs/mnist-cdro-finetune-a-100pct-v1x/eval-400kimg-10k/eval-20260330-121753/evaluation_result.json}"

if [[ ! -f "${INITIAL_SUMMARY_JSON}" ]]; then
  echo "[ERROR] Initial summary not found: ${INITIAL_SUMMARY_JSON}" >&2
  exit 1
fi

best_eval=$(
  python3 - <<'PY' "${INITIAL_SUMMARY_JSON}"
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
rows = payload.get("results", [])
if not rows:
    raise SystemExit(1)
print(rows[0]["eval_result"])
PY
)

best_candidate=$(
  python3 - <<'PY' "${INITIAL_SUMMARY_JSON}"
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
rows = payload.get("results", [])
if not rows:
    raise SystemExit(1)
print(rows[0]["candidate"])
PY
)

best_run_dir=$(
  python3 - <<'PY' "${best_eval}"
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload["run_dir"])
PY
)

resume_state="${best_run_dir}/training-state-000200.pt"
network_pkl="${best_run_dir}/network-snapshot-000400.pkl"
eval_root="${FOLLOWUP_ROOT}/eval/cdro_markov"

if [[ ! -f "${resume_state}" ]]; then
  echo "[ERROR] Missing resume state for best candidate ${best_candidate}: ${resume_state}" >&2
  exit 1
fi

mkdir -p "${FOLLOWUP_ROOT}"

if ! compgen -G "${eval_root}/*/evaluation_result.json" > /dev/null; then
  echo "[INFO] Resuming best non-plug-in candidate ${best_candidate} to ${TARGET_MIMG} MIMG"
  OUTDIR="${best_run_dir}" \
  TRAINER=baseline \
  PRECOND=cdromarkovedm \
  RESUME="${resume_state}" \
  DURATION_MIMG="${TARGET_MIMG}" \
  BATCH="${BATCH}" \
  BATCH_GPU="${BATCH_GPU}" \
  WORKERS="${WORKERS}" \
  AUGMENT=0 \
  NUM_IMAGES="${NUM_IMAGES}" \
  GEN_BATCH="${GEN_BATCH}" \
  FID_BATCH="${FID_BATCH}" \
  GEN_STEPS="${GEN_STEPS}" \
  bash scripts/setup_and_train_mnist.sh

  if [[ ! -f "${network_pkl}" ]]; then
    echo "[ERROR] Missing resumed snapshot: ${network_pkl}" >&2
    exit 1
  fi

  RUN_DIR="${best_run_dir}" \
  NETWORK_PKL="${network_pkl}" \
  EVAL_ROOT="${eval_root}" \
  NUM_IMAGES="${NUM_IMAGES}" \
  GEN_BATCH="${GEN_BATCH}" \
  FID_BATCH="${FID_BATCH}" \
  GEN_STEPS="${GEN_STEPS}" \
  bash scripts/setup_and_eval_mnist.sh
fi

export best_candidate best_run_dir INITIAL_SUMMARY_JSON FOLLOWUP_SUMMARY_JSON FOLLOWUP_SUMMARY_MD BASELINE_EVAL_JSON PLUGIN_CDRO_EVAL_JSON eval_root
python3 - <<'PY'
import json
import os
from pathlib import Path

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

eval_root = Path(os.environ["eval_root"])
evals = sorted(eval_root.glob("*/evaluation_result.json"))
if not evals:
    raise SystemExit("No follow-up evaluation result found.")
follow = read_eval(str(evals[-1]))
payload = {
    "best_candidate": os.environ["best_candidate"],
    "best_run_dir": os.environ["best_run_dir"],
    "initial_summary_json": os.environ["INITIAL_SUMMARY_JSON"],
    "baseline_reference": read_eval(os.environ["BASELINE_EVAL_JSON"]),
    "plugin_cdro_reference": read_eval(os.environ["PLUGIN_CDRO_EVAL_JSON"]),
    "followup": follow,
}
summary_json = Path(os.environ["FOLLOWUP_SUMMARY_JSON"])
summary_md = Path(os.environ["FOLLOWUP_SUMMARY_MD"])
summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

lines = [
    "# MNIST Non-Plug-In CDRO Follow-Up",
    "",
    f"Best initial candidate: `{payload['best_candidate']}`",
    "",
]
if payload["baseline_reference"] is not None:
    lines.append(f"- baseline EDM FID: `{payload['baseline_reference']['fid']:.4f}`")
if payload["plugin_cdro_reference"] is not None:
    lines.append(f"- plug-in CDRO FID: `{payload['plugin_cdro_reference']['fid']:.4f}`")
lines.append(f"- non-plug-in CDRO FID @ resumed target: `{follow['fid']:.4f}`")
lines.append("")
lines.append("## Artifacts")
lines.append("")
lines.append(f"- initial summary: `{payload['initial_summary_json']}`")
lines.append(f"- follow-up eval: `{follow['eval_result']}`")
summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(summary_md)
PY

echo "[INFO] Non-plug-in CDRO follow-up summary: ${FOLLOWUP_SUMMARY_MD}"
