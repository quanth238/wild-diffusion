#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

WAIT_PREFIX="${WAIT_PREFIX:?WAIT_PREFIX is required}"
NEXT_PREFIX="${NEXT_PREFIX:?NEXT_PREFIX is required}"
BASELINE_EVAL_JSON="${BASELINE_EVAL_JSON:?BASELINE_EVAL_JSON is required}"
WAIT_SUMMARY_JSON="${WAIT_SUMMARY_JSON:-${ROOT_DIR}/training-runs/${WAIT_PREFIX}-summary.json}"
WAIT_SUMMARY_MD="${WAIT_SUMMARY_MD:-${ROOT_DIR}/training-runs/${WAIT_PREFIX}-summary.md}"
NEXT_CANDIDATES="${NEXT_CANDIDATES:-g h i}"
POLL_SECONDS="${POLL_SECONDS:-120}"
STOP_MARGIN="${STOP_MARGIN:-0.0}"

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

mkdir -p "${ROOT_DIR}/training-runs"

echo "[INFO] Waiting for summary: ${WAIT_SUMMARY_JSON}"
while [[ ! -f "${WAIT_SUMMARY_JSON}" ]]; do
  sleep "${POLL_SECONDS}"
done

readarray -t FIDS < <(python3 - <<'PY'
import json, os
baseline = json.loads(open(os.environ["BASELINE_EVAL_JSON"], "r", encoding="utf-8").read())["fid"]
summary = json.loads(open(os.environ["WAIT_SUMMARY_JSON"], "r", encoding="utf-8").read())
best = min((row["fid"] for row in summary.get("results", [])), default=float("inf"))
print(baseline)
print(best)
PY
)

BASELINE_FID="${FIDS[0]}"
BEST_FID="${FIDS[1]}"

echo "[INFO] Baseline FID: ${BASELINE_FID}"
echo "[INFO] Best completed WDRO FID from ${WAIT_PREFIX}: ${BEST_FID}"

SHOULD_CONTINUE="$(python3 - <<'PY'
import os
baseline = float(os.environ["BASELINE_FID"])
best = float(os.environ["BEST_FID"])
margin = float(os.environ["STOP_MARGIN"])
print("1" if best > baseline + margin else "0")
PY
)"

if [[ "${SHOULD_CONTINUE}" != "1" ]]; then
  cat > "${ROOT_DIR}/training-runs/${NEXT_PREFIX}-not-needed.md" <<EOF
# WDRO Follow-up Not Needed

- waited for: \`${WAIT_SUMMARY_MD}\`
- baseline fid: \`${BASELINE_FID}\`
- best wdro fid in waited batch: \`${BEST_FID}\`

The follow-up batch was skipped because the completed batch already matched or beat the baseline within the configured margin.
EOF
  echo "[INFO] Follow-up skipped. ${WAIT_PREFIX} already met the stop criterion."
  exit 0
fi

echo "[INFO] Launching follow-up WDRO batch: ${NEXT_PREFIX} with candidates ${NEXT_CANDIDATES}"
RUN_TAG_PREFIX="${NEXT_PREFIX}" \
BASELINE_EVAL_JSON="${BASELINE_EVAL_JSON}" \
CANDIDATES="${NEXT_CANDIDATES}" \
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
bash scripts/run_mnist_wdro_retune.sh
