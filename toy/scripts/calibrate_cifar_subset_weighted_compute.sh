#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
SUMMARY_PATH="${SUMMARY_PATH:-${ROOT_DIR}/toy_data/cifar10_subset_airplane_automobile/summary.json}"
DEVICE="${DEVICE:-cuda}"
AMP_DTYPE="${AMP_DTYPE:-auto}"
SEED="${SEED:-0}"
TRAINING_OBJECTIVES="${TRAINING_OBJECTIVES:-edm rf}"
IMAGE_BACKBONE="${IMAGE_BACKBONE:-conv}"
BATCH_SIZE="${BATCH_SIZE:-256}"
HIDDEN_DIM="${HIDDEN_DIM:-64}"
CALIBRATION_DIR="${CALIBRATION_DIR:-${ROOT_DIR}/toy_outputs/compute_calibration}"
CALIBRATION_TAG="${CALIBRATION_TAG:-cifar2_image_conv}"
WARMUP_ITERS="${WARMUP_ITERS:-10}"
MEASURE_ITERS="${MEASURE_ITERS:-40}"
OVERWRITE="${OVERWRITE:-0}"

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
class_names = subset.get("class_names", [])
print(f"SUMMARY_IMAGE_SIZE={int(dataset.get('image_size', 32))}")
print(f"SUMMARY_IMAGE_CHANNELS={int(dataset.get('image_channels', 3))}")
print(f"SUMMARY_CLASS_TAG={shlex.quote('_'.join(str(name) for name in class_names) or 'cifar2')}")
PY
)"

mkdir -p "${CALIBRATION_DIR}"

for objective in ${TRAINING_OBJECTIVES}; do
  if [[ "${objective}" != "edm" && "${objective}" != "rf" && "${objective}" != "score" ]]; then
    echo "[error] unsupported training objective in TRAINING_OBJECTIVES: ${objective}" >&2
    exit 1
  fi
  out_json="${CALIBRATION_DIR}/${CALIBRATION_TAG}_${objective}_b${BATCH_SIZE}_h${HIDDEN_DIM}_cuda.json"
  if [[ -f "${out_json}" && "${OVERWRITE}" != "1" ]]; then
    echo "[cifar-calibration] reusing existing ${objective} calibration: ${out_json}"
    continue
  fi
  echo "[cifar-calibration] objective=${objective} out=${out_json}"
  "${PYTHON_BIN}" "${ROOT_DIR}/toy/scripts/calibrate_weighted_compute.py" \
    --out-json "${out_json}" \
    --device "${DEVICE}" \
    --amp-dtype "${AMP_DTYPE}" \
    --seed "${SEED}" \
    --training-objective "${objective}" \
    --image-size "${SUMMARY_IMAGE_SIZE}" \
    --image-channels "${SUMMARY_IMAGE_CHANNELS}" \
    --batch-size "${BATCH_SIZE}" \
    --hidden-dim "${HIDDEN_DIM}" \
    --image-backbone "${IMAGE_BACKBONE}" \
    --warmup-iters "${WARMUP_ITERS}" \
    --measure-iters "${MEASURE_ITERS}"
done

echo "[cifar-calibration] done: ${CALIBRATION_DIR}"
