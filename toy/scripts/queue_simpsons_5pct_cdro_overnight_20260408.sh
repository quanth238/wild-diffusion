#!/usr/bin/env bash
set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
DEVICE="${DEVICE:-cuda}"
DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/toy_data/simpsons_mnist_rgb}"
DATASET_PATH="${DATASET_PATH:-${DATA_ROOT}/imagefolder/train}"
DATASET_VAL_PATH="${DATASET_VAL_PATH:-${DATA_ROOT}/imagefolder/test}"
FID_REF_PATH="${FID_REF_PATH:-${DATA_ROOT}/fid_refs/simpsons_mnist_rgb_test_28x28.npz}"
CALIBRATION_PATH="${CALIBRATION_PATH:-${ROOT_DIR}/toy_outputs/compute_calibration/simpsons_mnist_rgb_image_conv_edm_b256_h64_cuda_4090_20260408.json}"

REUSE_BASELINE_RUNS_CSV="${REUSE_BASELINE_RUNS_CSV:-${ROOT_DIR}/toy_outputs/simpsons_mnist_rgb_three_method_current_default_5pct_4090_20260408/005pct/baseline/simpsons_mnist_rgb_three_method_current_default_5pct_4090_20260408_005pct_baseline_runs.csv}"
REUSE_BASELINE_AGG_CSV="${REUSE_BASELINE_AGG_CSV:-${ROOT_DIR}/toy_outputs/simpsons_mnist_rgb_three_method_current_default_5pct_4090_20260408/005pct/baseline/simpsons_mnist_rgb_three_method_current_default_5pct_4090_20260408_005pct_baseline_aggregate.csv}"
REUSE_WDRO_RAW_CSV="${REUSE_WDRO_RAW_CSV:-${ROOT_DIR}/toy_outputs/simpsons_mnist_rgb_three_method_current_default_5pct_4090_20260408/005pct/simpsons_mnist_rgb_three_method_current_default_5pct_4090_20260408_005pct_wdro_raw_seed_rows.csv}"

OUTDIR_ROOT="${OUTDIR_ROOT:-${ROOT_DIR}/toy_outputs/simpsons_mnist_rgb_5pct_cdro_overnight_mix_20260408}"
PREFIX_ROOT="${PREFIX_ROOT:-simpsons_mnist_rgb_5pct_cdro_overnight_mix_20260408}"
IMAGE_TRAIN_SIZE="${IMAGE_TRAIN_SIZE:-400}"
IMAGE_VAL_SIZE="${IMAGE_VAL_SIZE:-2000}"
IMAGE_SPLIT_SEED="${IMAGE_SPLIT_SEED:-0}"
TRAIN_PERCENT_LABEL="${TRAIN_PERCENT_LABEL:-5%}"

BATCH_SIZE="${BATCH_SIZE:-256}"
HIDDEN_DIM="${HIDDEN_DIM:-64}"
EVAL_SAMPLES="${EVAL_SAMPLES:-2000}"
FID_SAMPLES="${FID_SAMPLES:-2000}"
FID_GEN_BATCH="${FID_GEN_BATCH:-64}"
POSTHOC_FID_BATCH_SIZE="${POSTHOC_FID_BATCH_SIZE:-64}"
DEBUG_EVAL_BATCH="${DEBUG_EVAL_BATCH:-64}"
DEBUG_TERMINAL_STEP="${DEBUG_TERMINAL_STEP:-20}"
LOG_EVERY="${LOG_EVERY:-200}"

N_STEPS_PATH_BASELINE="${N_STEPS_PATH_BASELINE:-64}"
GRID_TEMPLATE="${GRID_TEMPLATE:-denser}"
FID_EVAL_TEMPLATE="${FID_EVAL_TEMPLATE:-balanced}"
BASELINE_FID_MODE="${BASELINE_FID_MODE:-posthoc_from_checkpoints}"
ROBUST_FID_MODE="${ROBUST_FID_MODE:-posthoc_from_checkpoints}"
SHARED_WEIGHTED_CAP="${SHARED_WEIGHTED_CAP:-200000}"
INNER_STEPS="${INNER_STEPS:-1}"
CDRO_STEP_SIZE="${CDRO_STEP_SIZE:-0.02}"
CDRO_TIME_HORIZON="${CDRO_TIME_HORIZON:-1.0}"
CDRO_WARMUP_FRACTION="${CDRO_WARMUP_FRACTION:-0.05}"
WDRO_WARMUP_FRACTION="${WDRO_WARMUP_FRACTION:-0.05}"
N32_FIXED_WARMUP_STEPS="${N32_FIXED_WARMUP_STEPS:-3358}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[error] missing python binary: ${PYTHON_BIN}" >&2
  exit 1
fi

mkdir -p "${OUTDIR_ROOT}"

QUEUE_LOG="${OUTDIR_ROOT}/queue.log"
STATUS_CSV="${OUTDIR_ROOT}/queue_status.csv"
CASES_CSV="${OUTDIR_ROOT}/planned_cases.csv"

exec > >(tee -a "${QUEUE_LOG}") 2>&1

echo "case_id,aw,cw,rho,n_steps_path,ramp_steps,attack_start,warmup_clean_steps,mode" > "${CASES_CSV}"
cat >> "${CASES_CSV}" <<'EOF'
static_n28_aw030_cw100_rho4p0,0.30,1.00,4.0,28,0,0.0,0,static
static_n36_aw030_cw100_rho4p0,0.30,1.00,4.0,36,0,0.0,0,static
static_n40_aw030_cw100_rho4p0,0.30,1.00,4.0,40,0,0.0,0,static
static_n32_aw030_cw075_rho4p0,0.30,0.75,4.0,32,0,0.0,0,static
static_n32_aw030_cw125_rho4p0,0.30,1.25,4.0,32,0,0.0,0,static
static_n32_aw030_cw150_rho4p0,0.30,1.50,4.0,32,0,0.0,0,static
static_n32_aw025_cw125_rho4p0,0.25,1.25,4.0,32,0,0.0,0,static
static_n32_aw035_cw125_rho4p0,0.35,1.25,4.0,32,0,0.0,0,static
static_n32_aw030_cw100_rho3p0,0.30,1.00,3.0,32,0,0.0,0,static
static_n32_aw030_cw100_rho5p0,0.30,1.00,5.0,32,0,0.0,0,static
ramp64_n32_aw030_cw100_rho4p0,0.30,1.00,4.0,32,64,0.0,3358,ramp
ramp128_n32_aw030_cw100_rho4p0,0.30,1.00,4.0,32,128,0.0,3358,ramp
ramp256_n32_aw030_cw100_rho4p0,0.30,1.00,4.0,32,256,0.0,3358,ramp
ramp512_n32_aw030_cw100_rho4p0,0.30,1.00,4.0,32,512,0.0,3358,ramp
ramp128_n32_aw030_cw125_rho4p0,0.30,1.25,4.0,32,128,0.0,3358,ramp
ramp256_n32_aw030_cw125_rho4p0,0.30,1.25,4.0,32,256,0.0,3358,ramp
ramp512_n32_aw030_cw125_rho4p0,0.30,1.25,4.0,32,512,0.0,3358,ramp
ramp128_n32_aw030_cw150_rho4p0,0.30,1.50,4.0,32,128,0.0,3358,ramp
ramp256_n32_aw030_cw150_rho4p0,0.30,1.50,4.0,32,256,0.0,3358,ramp
ramp256_n32_aw035_cw125_rho4p0,0.35,1.25,4.0,32,256,0.0,3358,ramp
EOF

echo "case_id,status,duration_sec,best_baseline_fid,best_wdro_fid,best_cdro_fid" > "${STATUS_CSV}"

run_case() {
  local case_id="$1"
  local aw="$2"
  local cw="$3"
  local rho="$4"
  local n_steps_path="$5"
  local ramp_steps="$6"
  local attack_start="$7"
  local warmup_clean_steps="$8"
  local mode="$9"

  local case_outdir="${OUTDIR_ROOT}/${case_id}"
  local case_prefix="${PREFIX_ROOT}_${case_id}"
  local logs_dir="${case_outdir}/logs"
  local collector_log="${logs_dir}/${case_prefix}_collector.log"
  local plot_log="${logs_dir}/${case_prefix}_plot.log"
  local combined_csv="${case_outdir}/${case_prefix}_all_methods_raw_seed_rows.csv"
  local plots_dir="${case_outdir}/plots"

  mkdir -p "${logs_dir}"

  local -a cmd=(
    "${PYTHON_BIN}" toy/scripts/collect_three_method_seed_data.py
    --outdir "${case_outdir}"
    --prefix "${case_prefix}"
    --python-bin "${PYTHON_BIN}"
    --seeds 0
    --device "${DEVICE}"
    --dataset-path "${DATASET_PATH}"
    --dataset-val-path "${DATASET_VAL_PATH}"
    --fid-ref-path "${FID_REF_PATH}"
    --weighted-compute-calibration-path "${CALIBRATION_PATH}"
    --image-size 28
    --image-channels 3
    --image-train-size "${IMAGE_TRAIN_SIZE}"
    --image-val-size "${IMAGE_VAL_SIZE}"
    --train-percent-label "${TRAIN_PERCENT_LABEL}"
    --image-split-seed "${IMAGE_SPLIT_SEED}"
    --batch-size "${BATCH_SIZE}"
    --hidden-dim "${HIDDEN_DIM}"
    --eval-samples "${EVAL_SAMPLES}"
    --fid-samples "${FID_SAMPLES}"
    --fid-gen-batch "${FID_GEN_BATCH}"
    --posthoc-fid-batch-size "${POSTHOC_FID_BATCH_SIZE}"
    --debug-eval-batch "${DEBUG_EVAL_BATCH}"
    --debug-terminal-step "${DEBUG_TERMINAL_STEP}"
    --log-every "${LOG_EVERY}"
    --n-steps-path "${N_STEPS_PATH_BASELINE}"
    --grid-template "${GRID_TEMPLATE}"
    --fid-eval-template "${FID_EVAL_TEMPLATE}"
    --baseline-fid-mode "${BASELINE_FID_MODE}"
    --robust-fid-mode "${ROBUST_FID_MODE}"
    --shared-weighted-cap "${SHARED_WEIGHTED_CAP}"
    --inner-steps "${INNER_STEPS}"
    --outer-attack-weight "${aw}"
    --outer-clean-weight "${cw}"
    --wdro-warmup-fraction "${WDRO_WARMUP_FRACTION}"
    --cdro-step-size "${CDRO_STEP_SIZE}"
    --cdro-total-budget-rho "${rho}"
    --cdro-time-horizon "${CDRO_TIME_HORIZON}"
    --cdro-warmup-fraction "${CDRO_WARMUP_FRACTION}"
    --cdro-n-steps-path "${n_steps_path}"
    --reuse-baseline-runs-csv "${REUSE_BASELINE_RUNS_CSV}"
    --reuse-baseline-aggregate-csv "${REUSE_BASELINE_AGG_CSV}"
    --reuse-wdro-raw-csv "${REUSE_WDRO_RAW_CSV}"
    --skip-existing
  )

  if [[ "${mode}" == "ramp" ]]; then
    cmd+=(
      --warmup-clean-steps "${warmup_clean_steps}"
      --warmup-ramp-steps "${ramp_steps}"
      --warmup-attack-weight-start "${attack_start}"
      --warmup-phi-lr-scale-start 0.0
    )
  fi

  echo "[overnight] starting ${case_id} aw=${aw} cw=${cw} rho=${rho} n=${n_steps_path} mode=${mode} ramp=${ramp_steps}"
  local start_ts
  start_ts="$(date +%s)"

  local status="ok"
  if ! "${cmd[@]}" > "${collector_log}" 2>&1; then
    status="collector_failed"
  fi

  if [[ "${status}" == "ok" ]]; then
    if ! "${PYTHON_BIN}" toy/scripts/plot_three_method_fid_curves.py \
      --combined-csv "${combined_csv}" \
      --outdir "${plots_dir}" \
      --prefix "${case_prefix}" \
      --train-percent-label "${TRAIN_PERCENT_LABEL}" > "${plot_log}" 2>&1; then
      status="plot_failed"
    fi
  fi

  local end_ts
  end_ts="$(date +%s)"
  local duration_sec=$(( end_ts - start_ts ))

  local best_baseline_fid=""
  local best_wdro_fid=""
  local best_cdro_fid=""

  if [[ "${status}" == "ok" ]]; then
    read -r best_baseline_fid best_wdro_fid best_cdro_fid < <(
      "${PYTHON_BIN}" - <<'PY' "${combined_csv}"
import csv
import math
import sys

path = sys.argv[1]
best = {"baseline_edm": None, "wdro": None, "cdro": None}
with open(path, newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        method = str(row.get("method", ""))
        if method not in best:
            continue
        try:
            fid = float(row.get("fid", ""))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(fid):
            continue
        if best[method] is None or fid < best[method]:
            best[method] = fid
print(
    "" if best["baseline_edm"] is None else f"{best['baseline_edm']:.6f}",
    "" if best["wdro"] is None else f"{best['wdro']:.6f}",
    "" if best["cdro"] is None else f"{best['cdro']:.6f}",
)
PY
    )
  fi

  echo "${case_id},${status},${duration_sec},${best_baseline_fid},${best_wdro_fid},${best_cdro_fid}" >> "${STATUS_CSV}"
  echo "[overnight] finished ${case_id} status=${status} duration=${duration_sec}s cdro_best=${best_cdro_fid}"
}

while IFS=, read -r case_id aw cw rho n_steps_path ramp_steps attack_start warmup_clean_steps mode; do
  if [[ "${case_id}" == "case_id" ]]; then
    continue
  fi
  run_case "${case_id}" "${aw}" "${cw}" "${rho}" "${n_steps_path}" "${ramp_steps}" "${attack_start}" "${warmup_clean_steps}" "${mode}"
done < "${CASES_CSV}"

"${PYTHON_BIN}" - <<'PY' "${OUTDIR_ROOT}" "${STATUS_CSV}"
import csv
import os
import sys

outdir = sys.argv[1]
status_csv = sys.argv[2]
rows = []
with open(status_csv, newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        rows.append(row)
rows.sort(key=lambda row: float(row["best_cdro_fid"]) if row.get("best_cdro_fid") not in ("", None) else float("inf"))
summary_path = os.path.join(outdir, "queue_status_sorted_by_cdro.csv")
with open(summary_path, "w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["case_id"])
    writer.writeheader()
    writer.writerows(rows)
print(f"[overnight] wrote {summary_path}")
PY

echo "[overnight] done: ${OUTDIR_ROOT}"
