#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

RUN_TAG="${RUN_TAG:-cifar10-compare-$(date +%Y%m%d-%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/training-runs/${RUN_TAG}}"
RUNS_DIR="${RUN_ROOT}/runs"
EVAL_DIR="${RUN_ROOT}/eval"
SUMMARY_JSON="${RUN_ROOT}/summary.json"
SUMMARY_MD="${RUN_ROOT}/summary.md"

METHODS_RAW="${METHODS:-baseline wdro}"
METHODS="${METHODS_RAW//,/ }"

TRAIN_ONLY="${TRAIN_ONLY:-0}"
EVAL_ONLY="${EVAL_ONLY:-0}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "${RUNS_DIR}" "${EVAL_DIR}"
export RUN_ROOT SUMMARY_JSON SUMMARY_MD METHODS

latest_run_dir() {
  local base_dir="$1"
  if [[ ! -d "${base_dir}" ]]; then
    return 1
  fi
  find "${base_dir}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1
}

for method in ${METHODS}; do
  if [[ "${method}" != "baseline" && "${method}" != "wdro" && "${method}" != "cdro" ]]; then
    echo "[ERROR] Unsupported method in METHODS: ${method}"
    echo "        Supported methods: baseline wdro cdro"
    exit 1
  fi

  trainer="${method}"
  precond="wdroedm"
  if [[ "${method}" == "cdro" ]]; then
    trainer="baseline"
    precond="cdroedm"
  fi

  method_outdir="${RUNS_DIR}/${method}"
  mkdir -p "${method_outdir}"

  if [[ "${EVAL_ONLY}" != "1" ]]; then
    echo "[INFO] Starting CIFAR-10 training for method=${method}"
    OUTDIR="${method_outdir}" \
    TRAINER="${trainer}" \
    PRECOND="${precond}" \
    DRY_RUN="${DRY_RUN}" \
    bash scripts/setup_and_train_cifar10.sh
  fi

  if [[ "${DRY_RUN}" == "1" || "${TRAIN_ONLY}" == "1" ]]; then
    continue
  fi

  run_dir="$(latest_run_dir "${method_outdir}")"
  if [[ -z "${run_dir}" ]]; then
    echo "[ERROR] Could not resolve latest run directory for method=${method} under ${method_outdir}"
    exit 1
  fi

  echo "[INFO] Evaluating CIFAR-10 run for method=${method}: ${run_dir}"
  RUN_DIR="${run_dir}" \
  EVAL_ROOT="${EVAL_DIR}/${method}" \
  bash scripts/setup_and_eval_cifar10.sh
done

if [[ "${DRY_RUN}" == "1" || "${TRAIN_ONLY}" == "1" ]]; then
  echo "[INFO] CIFAR-10 compare runner finished without summary generation (DRY_RUN=${DRY_RUN}, TRAIN_ONLY=${TRAIN_ONLY})."
  exit 0
fi

python3 - <<'PY'
import json
import os
from pathlib import Path

run_root = Path(os.environ["RUN_ROOT"])
summary_json = Path(os.environ["SUMMARY_JSON"])
summary_md = Path(os.environ["SUMMARY_MD"])
rows = []

for method in os.environ["METHODS"].split():
    eval_root = run_root / "eval" / method
    results = sorted(eval_root.glob("*/evaluation_result.json"))
    if not results:
        continue
    result_path = results[-1]
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    fid_value = payload.get("fid")
    rows.append(
        {
            "method": method,
            "fid": None if fid_value is None else float(fid_value),
            "run_dir": payload["run_dir"],
            "network_pkl": payload["network_pkl"],
            "eval_result": str(result_path),
        }
    )

rows.sort(key=lambda item: float("inf") if item["fid"] is None else item["fid"])
summary_json.write_text(json.dumps({"run_root": str(run_root), "results": rows}, indent=2), encoding="utf-8")

lines = [
    "# CIFAR-10 Comparison Summary",
    "",
    f"Run root: `{run_root}`",
    "",
    "| Rank | Method | FID |",
    "| --- | --- | ---: |",
]
for idx, row in enumerate(rows, start=1):
    fid_text = "n/a" if row["fid"] is None else f"{row['fid']:.4f}"
    lines.append(f"| {idx} | `{row['method']}` | {fid_text} |")

lines.extend(["", "## Artifacts", ""])
for row in rows:
    lines.append(f"- `{row['method']}` run: `{row['run_dir']}`")
    lines.append(f"- `{row['method']}` eval: `{row['eval_result']}`")

summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(summary_md)
PY

echo "[INFO] CIFAR-10 comparison complete."
echo "[INFO] Summary markdown: ${SUMMARY_MD}"
echo "[INFO] Summary json: ${SUMMARY_JSON}"
