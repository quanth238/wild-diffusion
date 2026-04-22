#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
SUMMARY_PATH="${SUMMARY_PATH:-${ROOT_DIR}/toy_data/cifar10_subset_airplane_automobile/summary.json}"
DEVICE="${DEVICE:-cuda}"
SEEDS="${SEEDS:-0}"
PERCENTS="${PERCENTS:-100}"
DRY_RUN="${DRY_RUN:-0}"
AUTO_CALIBRATE="${AUTO_CALIBRATE:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

if [[ ! -f "${SUMMARY_PATH}" ]]; then
  echo "[error] missing CIFAR subset summary: ${SUMMARY_PATH}" >&2
  echo "[hint] run toy/scripts/setup_cifar_subset_imagefolder.py first." >&2
  exit 1
fi

eval "$(
  SUMMARY_PATH="${SUMMARY_PATH}" "${PYTHON_BIN}" - <<'PY'
import json
import os
import shlex

with open(os.environ["SUMMARY_PATH"], "r", encoding="utf-8") as fh:
    payload = json.load(fh)
dataset = payload.get("dataset", {})
subset = payload.get("subset", {})
class_names = [str(name) for name in subset.get("class_names", [])]
class_tag = "_".join(class_names) if class_names else "cifar2"
class_label = " vs ".join(class_names) if class_names else "2-class subset"
print(f"SUMMARY_TRAIN_ROOT={shlex.quote(str(payload['train_root']))}")
print(f"SUMMARY_TEST_ROOT={shlex.quote(str(payload['test_root']))}")
print(f"SUMMARY_FID_REF_PATH={shlex.quote(str(payload.get('fid_ref_path') or ''))}")
print(f"SUMMARY_IMAGE_SIZE={int(dataset.get('image_size', 32))}")
print(f"SUMMARY_IMAGE_CHANNELS={int(dataset.get('image_channels', 3))}")
print(f"SUMMARY_TRAIN_TOTAL={int(dataset.get('train_total', 0))}")
print(f"SUMMARY_TEST_TOTAL={int(dataset.get('test_total', 0))}")
print(f"SUMMARY_CLASS_TAG={shlex.quote(class_tag)}")
print(f"SUMMARY_CLASS_LABEL={shlex.quote(class_label)}")
PY
)"

DATASET_PATH="${DATASET_PATH:-${SUMMARY_TRAIN_ROOT}}"
DATASET_VAL_PATH="${DATASET_VAL_PATH:-${SUMMARY_TEST_ROOT}}"
FID_REF_PATH="${FID_REF_PATH:-${SUMMARY_FID_REF_PATH}}"
IMAGE_SIZE="${IMAGE_SIZE:-${SUMMARY_IMAGE_SIZE}}"
IMAGE_CHANNELS="${IMAGE_CHANNELS:-${SUMMARY_IMAGE_CHANNELS}}"
TRAIN_POPULATION_SIZE="${TRAIN_POPULATION_SIZE:-${SUMMARY_TRAIN_TOTAL}}"
IMAGE_VAL_SIZE="${IMAGE_VAL_SIZE:-${SUMMARY_TEST_TOTAL}}"
DATASET_LABEL="${DATASET_LABEL:-CIFAR-10 subset (${SUMMARY_CLASS_LABEL})}"
DATASET_TAG="${DATASET_TAG:-${SUMMARY_CLASS_TAG}}"

OUTDIR="${OUTDIR:-${ROOT_DIR}/toy_outputs/cifar2_${DATASET_TAG}_family_compare_default_v1}"
PREFIX_ROOT="${PREFIX_ROOT:-cifar2_${DATASET_TAG}_family_compare_default_v1}"

CALIBRATION_DIR="${CALIBRATION_DIR:-${ROOT_DIR}/toy_outputs/compute_calibration}"
EDM_CALIBRATION_PATH="${EDM_CALIBRATION_PATH:-${CALIBRATION_DIR}/cifar2_image_conv_edm_b256_h64_cuda.json}"
RF_CALIBRATION_PATH="${RF_CALIBRATION_PATH:-${CALIBRATION_DIR}/cifar2_image_conv_rf_b256_h64_cuda.json}"

IMAGE_SPLIT_SEED="${IMAGE_SPLIT_SEED:-0}"
BATCH_SIZE="${BATCH_SIZE:-256}"
HIDDEN_DIM="${HIDDEN_DIM:-64}"
IMAGE_BACKBONE="${IMAGE_BACKBONE:-conv}"
EVAL_SAMPLES="${EVAL_SAMPLES:-${IMAGE_VAL_SIZE}}"
FID_SAMPLES="${FID_SAMPLES:-${IMAGE_VAL_SIZE}}"
FID_GEN_BATCH="${FID_GEN_BATCH:-64}"
DEBUG_EVAL_BATCH="${DEBUG_EVAL_BATCH:-64}"
DEBUG_TERMINAL_STEP="${DEBUG_TERMINAL_STEP:-20}"
LOG_EVERY="${LOG_EVERY:-200}"
GRID_TEMPLATE="${GRID_TEMPLATE:-denser}"
FID_EVAL_TEMPLATE="${FID_EVAL_TEMPLATE:-balanced}"
BASELINE_FID_MODE="${BASELINE_FID_MODE:-posthoc_from_checkpoints}"
ROBUST_FID_MODE="${ROBUST_FID_MODE:-posthoc_from_checkpoints}"
SHARED_WEIGHTED_CAP="${SHARED_WEIGHTED_CAP:-400000}"
BASELINE_MAX_STEPS="${BASELINE_MAX_STEPS:-160000}"
WDRO_MAX_TOTAL_STEPS="${WDRO_MAX_TOTAL_STEPS:-160000}"
INNER_STEPS="${INNER_STEPS:-1}"
OUTER_ATTACK_WEIGHT="${OUTER_ATTACK_WEIGHT:-1.0}"
OUTER_CLEAN_WEIGHT="${OUTER_CLEAN_WEIGHT:-0.0}"
WDRO_WARMUP_FRACTION="${WDRO_WARMUP_FRACTION:-0.2}"
CDRO_STEP_SIZE="${CDRO_STEP_SIZE:-0.02}"
CDRO_TIME_HORIZON="${CDRO_TIME_HORIZON:-1.0}"
CDRO_EDM_LADDER_MODE="${CDRO_EDM_LADDER_MODE:-stochastic_stratified_quantile}"
CDRO_WARMUP_FRACTION="${CDRO_WARMUP_FRACTION:-0.2}"

EDM_N_STEPS_PATH="${EDM_N_STEPS_PATH:-64}"
EDM_CDRO_N_STEPS_PATH="${EDM_CDRO_N_STEPS_PATH:-32}"
EDM_CDRO_TOTAL_BUDGET_RHO="${EDM_CDRO_TOTAL_BUDGET_RHO:-0.001}"

RF_N_STEPS_PATH="${RF_N_STEPS_PATH:-40}"
RF_CDRO_N_STEPS_PATH="${RF_CDRO_N_STEPS_PATH:-40}"
RF_CDRO_TOTAL_BUDGET_RHO="${RF_CDRO_TOTAL_BUDGET_RHO:-0.001}"
RF_TEACHER_N_STEPS_PATH="${RF_TEACHER_N_STEPS_PATH:-40}"
RF_EVAL_N_STEPS_PATH="${RF_EVAL_N_STEPS_PATH:-9}"
RF_EDM_TEACHER_SAMPLER="${RF_EDM_TEACHER_SAMPLER:-edm_heun}"
RF_CDRO_PAIR_SOURCE="${RF_CDRO_PAIR_SOURCE:-reflow}"

USE_EMA_EVAL="${USE_EMA_EVAL:-1}"
EMA_MODE="${EMA_MODE:-official}"
EMA_DECAY="${EMA_DECAY:-0.999}"
EMA_HALFLIFE_KIMG="${EMA_HALFLIFE_KIMG:-500}"
EMA_RAMPUP_RATIO="${EMA_RAMPUP_RATIO:-0.05}"
DISABLE_EMA_RAMPUP="${DISABLE_EMA_RAMPUP:-0}"

if [[ ! -d "${DATASET_PATH}" ]]; then
  echo "[error] missing train root: ${DATASET_PATH}" >&2
  exit 1
fi
if [[ ! -d "${DATASET_VAL_PATH}" ]]; then
  echo "[error] missing val root: ${DATASET_VAL_PATH}" >&2
  exit 1
fi
if [[ ! -f "${FID_REF_PATH}" ]]; then
  echo "[error] missing FID ref: ${FID_REF_PATH}" >&2
  exit 1
fi
if (( TRAIN_POPULATION_SIZE <= 0 )); then
  echo "[error] invalid train population size: ${TRAIN_POPULATION_SIZE}" >&2
  exit 1
fi
if [[ "${AUTO_CALIBRATE}" == "1" ]]; then
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[dry-run] PYTHON_BIN=%q SUMMARY_PATH=%q %q\n' \
      "${PYTHON_BIN}" "${SUMMARY_PATH}" "${ROOT_DIR}/toy/scripts/calibrate_cifar_subset_weighted_compute.sh"
  else
    PYTHON_BIN="${PYTHON_BIN}" SUMMARY_PATH="${SUMMARY_PATH}" \
      "${ROOT_DIR}/toy/scripts/calibrate_cifar_subset_weighted_compute.sh"
  fi
fi
if [[ ! -f "${EDM_CALIBRATION_PATH}" ]]; then
  echo "[error] missing EDM weighted-compute calibration: ${EDM_CALIBRATION_PATH}" >&2
  echo "[hint] run toy/scripts/calibrate_cifar_subset_weighted_compute.sh first or set AUTO_CALIBRATE=1." >&2
  exit 1
fi
if [[ ! -f "${RF_CALIBRATION_PATH}" ]]; then
  echo "[error] missing RF weighted-compute calibration: ${RF_CALIBRATION_PATH}" >&2
  echo "[hint] run toy/scripts/calibrate_cifar_subset_weighted_compute.sh first or set AUTO_CALIBRATE=1." >&2
  exit 1
fi

ema_args=()
if [[ "${USE_EMA_EVAL}" == "1" ]]; then
  ema_args+=(--use-ema-eval --ema-mode "${EMA_MODE}")
  if [[ "${EMA_MODE}" == "fixed" ]]; then
    ema_args+=(--ema-decay "${EMA_DECAY}")
  else
    ema_args+=(--ema-halflife-kimg "${EMA_HALFLIFE_KIMG}")
    if [[ "${DISABLE_EMA_RAMPUP}" == "1" ]]; then
      ema_args+=(--disable-ema-rampup)
    else
      ema_args+=(--ema-rampup-ratio "${EMA_RAMPUP_RATIO}")
    fi
  fi
fi

skip_existing_args=()
if [[ "${SKIP_EXISTING}" == "1" ]]; then
  skip_existing_args+=(--skip-existing)
fi

run_cmd() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[dry-run]'; printf ' %q' "$@"; printf '\n'
    return 0
  fi
  "$@"
}

infer_rf_warmup_ckpt() {
  local combined_csv="$1"
  "${PYTHON_BIN}" - "${combined_csv}" <<'PY'
import csv
import os
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
for row in rows:
    if str(row.get("method", "")).strip() != "baseline":
        continue
    step = str(row.get("step", "")).strip()
    fixed_warmup_steps = str(row.get("fixed_warmup_steps", "")).strip()
    checkpoint_path = str(row.get("checkpoint_path", "")).strip()
    if not step or not fixed_warmup_steps or step != fixed_warmup_steps or not checkpoint_path:
        continue
    if os.path.isfile(checkpoint_path):
        print(checkpoint_path)
        raise SystemExit(0)
raise SystemExit("[error] could not infer shared EDM warmup checkpoint from baseline rows")
PY
}

mkdir -p "${OUTDIR}"

for pct in ${PERCENTS}; do
  if ! [[ "${pct}" =~ ^[0-9]+$ ]]; then
    echo "[error] percent must be an integer, got: ${pct}" >&2
    exit 1
  fi
  if (( pct <= 0 || pct > 100 )); then
    echo "[error] percent must be in 1..100, got: ${pct}" >&2
    exit 1
  fi
  train_size=$(( TRAIN_POPULATION_SIZE * pct / 100 ))
  if (( train_size <= 0 )); then
    echo "[error] resolved train size is zero for pct=${pct}" >&2
    exit 1
  fi

  pct_tag="$(printf "%03d" "${pct}")pct"
  edm_outdir="${OUTDIR}/${pct_tag}/edm"
  rf_outdir="${OUTDIR}/${pct_tag}/rf"
  compare_outdir="${OUTDIR}/${pct_tag}/family_compare"
  edm_prefix="${PREFIX_ROOT}_${pct_tag}_edm"
  rf_prefix="${PREFIX_ROOT}_${pct_tag}_rf"
  compare_prefix="${PREFIX_ROOT}_${pct_tag}"
  edm_combined_csv="${edm_outdir}/${edm_prefix}_all_methods_raw_seed_rows.csv"
  rf_combined_csv="${rf_outdir}/${rf_prefix}_all_methods_raw_seed_rows.csv"

  echo "[cifar-family-compare] pct=${pct}% train_size=${train_size}"
  run_cmd "${PYTHON_BIN}" "${ROOT_DIR}/toy/scripts/collect_three_method_seed_data.py" \
    --outdir "${edm_outdir}" \
    --prefix "${edm_prefix}" \
    --python-bin "${PYTHON_BIN}" \
    --seeds "${SEEDS}" \
    --device "${DEVICE}" \
    --dataset-path "${DATASET_PATH}" \
    --dataset-val-path "${DATASET_VAL_PATH}" \
    --fid-ref-path "${FID_REF_PATH}" \
    --weighted-compute-calibration-path "${EDM_CALIBRATION_PATH}" \
    --image-size "${IMAGE_SIZE}" \
    --image-channels "${IMAGE_CHANNELS}" \
    --image-train-size "${train_size}" \
    --image-val-size "${IMAGE_VAL_SIZE}" \
    --train-percent-label "${pct}%" \
    --image-split-seed "${IMAGE_SPLIT_SEED}" \
    --batch-size "${BATCH_SIZE}" \
    --hidden-dim "${HIDDEN_DIM}" \
    --image-backbone "${IMAGE_BACKBONE}" \
    --training-objective edm \
    --eval-samples "${EVAL_SAMPLES}" \
    --fid-samples "${FID_SAMPLES}" \
    --fid-gen-batch "${FID_GEN_BATCH}" \
    --debug-eval-batch "${DEBUG_EVAL_BATCH}" \
    --debug-terminal-step "${DEBUG_TERMINAL_STEP}" \
    --log-every "${LOG_EVERY}" \
    --n-steps-path "${EDM_N_STEPS_PATH}" \
    --grid-template "${GRID_TEMPLATE}" \
    --fid-eval-template "${FID_EVAL_TEMPLATE}" \
    --baseline-fid-mode "${BASELINE_FID_MODE}" \
    --robust-fid-mode "${ROBUST_FID_MODE}" \
    --shared-weighted-cap "${SHARED_WEIGHTED_CAP}" \
    --baseline-max-steps "${BASELINE_MAX_STEPS}" \
    --wdro-max-total-steps "${WDRO_MAX_TOTAL_STEPS}" \
    --inner-steps "${INNER_STEPS}" \
    --outer-attack-weight "${OUTER_ATTACK_WEIGHT}" \
    --outer-clean-weight "${OUTER_CLEAN_WEIGHT}" \
    --wdro-warmup-fraction "${WDRO_WARMUP_FRACTION}" \
    --cdro-step-size "${CDRO_STEP_SIZE}" \
    --cdro-total-budget-rho "${EDM_CDRO_TOTAL_BUDGET_RHO}" \
    --cdro-time-horizon "${CDRO_TIME_HORIZON}" \
    --cdro-edm-ladder-mode "${CDRO_EDM_LADDER_MODE}" \
    --cdro-warmup-fraction "${CDRO_WARMUP_FRACTION}" \
    --cdro-n-steps-path "${EDM_CDRO_N_STEPS_PATH}" \
    "${skip_existing_args[@]}" \
    "${ema_args[@]}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    rf_warmup_ckpt="<dry_run_edm_warmup_checkpoint>"
  else
    if [[ ! -f "${edm_combined_csv}" ]]; then
      echo "[error] missing EDM combined CSV after collector run: ${edm_combined_csv}" >&2
      exit 1
    fi
    rf_warmup_ckpt="$(infer_rf_warmup_ckpt "${edm_combined_csv}")"
  fi

  run_cmd "${PYTHON_BIN}" "${ROOT_DIR}/toy/scripts/collect_three_method_seed_data.py" \
    --outdir "${rf_outdir}" \
    --prefix "${rf_prefix}" \
    --python-bin "${PYTHON_BIN}" \
    --seeds "${SEEDS}" \
    --device "${DEVICE}" \
    --dataset-path "${DATASET_PATH}" \
    --dataset-val-path "${DATASET_VAL_PATH}" \
    --fid-ref-path "${FID_REF_PATH}" \
    --weighted-compute-calibration-path "${RF_CALIBRATION_PATH}" \
    --image-size "${IMAGE_SIZE}" \
    --image-channels "${IMAGE_CHANNELS}" \
    --image-train-size "${train_size}" \
    --image-val-size "${IMAGE_VAL_SIZE}" \
    --train-percent-label "${pct}%" \
    --image-split-seed "${IMAGE_SPLIT_SEED}" \
    --batch-size "${BATCH_SIZE}" \
    --hidden-dim "${HIDDEN_DIM}" \
    --image-backbone "${IMAGE_BACKBONE}" \
    --training-objective rf \
    --eval-samples "${EVAL_SAMPLES}" \
    --fid-samples "${FID_SAMPLES}" \
    --fid-gen-batch "${FID_GEN_BATCH}" \
    --debug-eval-batch "${DEBUG_EVAL_BATCH}" \
    --debug-terminal-step "${DEBUG_TERMINAL_STEP}" \
    --log-every "${LOG_EVERY}" \
    --n-steps-path "${RF_N_STEPS_PATH}" \
    --grid-template "${GRID_TEMPLATE}" \
    --fid-eval-template "${FID_EVAL_TEMPLATE}" \
    --baseline-fid-mode "${BASELINE_FID_MODE}" \
    --robust-fid-mode "${ROBUST_FID_MODE}" \
    --shared-weighted-cap "${SHARED_WEIGHTED_CAP}" \
    --baseline-max-steps "${BASELINE_MAX_STEPS}" \
    --wdro-max-total-steps "${WDRO_MAX_TOTAL_STEPS}" \
    --inner-steps "${INNER_STEPS}" \
    --outer-attack-weight "${OUTER_ATTACK_WEIGHT}" \
    --outer-clean-weight "${OUTER_CLEAN_WEIGHT}" \
    --wdro-warmup-fraction "${WDRO_WARMUP_FRACTION}" \
    --cdro-step-size "${CDRO_STEP_SIZE}" \
    --cdro-total-budget-rho "${RF_CDRO_TOTAL_BUDGET_RHO}" \
    --cdro-time-horizon "${CDRO_TIME_HORIZON}" \
    --cdro-edm-ladder-mode "${CDRO_EDM_LADDER_MODE}" \
    --cdro-warmup-fraction "${CDRO_WARMUP_FRACTION}" \
    --cdro-n-steps-path "${RF_CDRO_N_STEPS_PATH}" \
    --reuse-baseline-raw-csv "${edm_combined_csv}" \
    --rf-edm-init-ckpt-path "${rf_warmup_ckpt}" \
    --rf-cdro-pair-source "${RF_CDRO_PAIR_SOURCE}" \
    --rf-edm-teacher-sampler "${RF_EDM_TEACHER_SAMPLER}" \
    --rf-teacher-n-steps-path "${RF_TEACHER_N_STEPS_PATH}" \
    --rf-eval-n-steps-path "${RF_EVAL_N_STEPS_PATH}" \
    "${skip_existing_args[@]}" \
    "${ema_args[@]}"

  run_cmd "${PYTHON_BIN}" "${ROOT_DIR}/toy/scripts/plot_three_method_family_compare_from_runs.py" \
    --combined-csv "${edm_combined_csv}" \
    --combined-csv "${rf_combined_csv}" \
    --outdir "${compare_outdir}" \
    --prefix "${compare_prefix}" \
    --dataset-label "${DATASET_LABEL}" \
    --train-percent-label "${pct}%"
done

echo "[cifar-family-compare] done: ${OUTDIR}"
