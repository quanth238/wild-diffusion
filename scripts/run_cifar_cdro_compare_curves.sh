#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

RUN_DIR="${RUN_DIR:?RUN_DIR is required}"
EVAL_SWEEP_ROOT="${EVAL_SWEEP_ROOT:?EVAL_SWEEP_ROOT is required}"
VENV_DIR="${VENV_DIR:-/home/bachlc/.venvs/wild-diffusion-h100}"
BASE_COMPARE_CSV="${BASE_COMPARE_CSV:-/home/bachlc/GM-CDRO/training-runs/fid-sweeps/cifar10_baseline_vs_wdro_coarse_20260414/cifar10_baseline_vs_wdro_coarse_20260414_three_method_compare.csv}"
PYTORCH_FID_REF="${PYTORCH_FID_REF:-${ROOT_DIR}/training-runs/fid-sweeps/cifar10_baseline_vs_wdro_coarse_20260414/pytorch_fid_cifar10_train_ref_stats.npz}"
FLOP_CALIBRATION_JSON="${FLOP_CALIBRATION_JSON:-}"
TRAIN_PERCENT_LABEL="${TRAIN_PERCENT_LABEL:-20%}"
DATASET_LABEL="${DATASET_LABEL:-CIFAR-10}"
PLOT_TAG="${PLOT_TAG:-cdro}"
LOSS_POINTS_MAX="${LOSS_POINTS_MAX:-50}"
FID_POINTS_MAX="${FID_POINTS_MAX:-8}"
LOSS_PROBE_MAX_IMAGES="${LOSS_PROBE_MAX_IMAGES:-10000}"
LOSS_PROBE_BATCH_SIZE="${LOSS_PROBE_BATCH_SIZE:-512}"
FID_NUM_IMAGES="${FID_NUM_IMAGES:-50000}"
FID_GEN_BATCH="${FID_GEN_BATCH:-128}"
FID_BATCH="${FID_BATCH:-64}"
FID_GEN_STEPS="${FID_GEN_STEPS:-18}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
LOG_DIR="${LOG_DIR:-/mnt/data/bachlc/GM-CDRO/training-runs/fid-sweeps/logs}"
RUN_LOG="${RUN_LOG:-${LOG_DIR}/${PLOT_TAG}_compare_curves.log}"

mkdir -p "${LOG_DIR}" "${EVAL_SWEEP_ROOT}"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "[ERROR] Missing venv: ${VENV_DIR}"
  exit 1
fi
if [[ ! -f "${BASE_COMPARE_CSV}" ]]; then
  echo "[ERROR] Missing base compare CSV: ${BASE_COMPARE_CSV}"
  exit 1
fi
if [[ ! -f "${PYTORCH_FID_REF}" ]]; then
  echo "[ERROR] Missing pytorch-fid reference stats: ${PYTORCH_FID_REF}"
  exit 1
fi
if [[ -n "${FLOP_CALIBRATION_JSON}" && ! -f "${FLOP_CALIBRATION_JSON}" ]]; then
  echo "[ERROR] Missing FLOP calibration JSON: ${FLOP_CALIBRATION_JSON}"
  exit 1
fi

if [[ -r /proc/self/fd ]]; then
  exec > >(tee -a "${RUN_LOG}") 2>&1
else
  exec >>"${RUN_LOG}" 2>&1
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ ! -r /proc/self/maps || ! -r /proc/cpuinfo || ! -r /proc/sys/vm/mmap_min_addr ]]; then
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/scripts/ensure_procfs_compat.sh"
fi

select_snapshot_kimg() {
  local run_dir="$1"
  local max_points="$2"
  python - "${run_dir}" "${max_points}" <<'PY'
import sys
from pathlib import Path

run_dir = Path(sys.argv[1]).resolve()
max_points = int(sys.argv[2])
snapshots = sorted(
    {
        int(path.stem.replace("network-snapshot-", ""))
        for path in run_dir.glob("network-snapshot-*.pkl")
        if path.stem.replace("network-snapshot-", "").isdigit()
    }
)
if not snapshots:
    raise SystemExit("No snapshots found")
if len(snapshots) <= max_points:
    chosen = snapshots
else:
    chosen = []
    for idx in range(max_points):
        pos = round(idx * (len(snapshots) - 1) / (max_points - 1))
        chosen.append(snapshots[pos])
    deduped = []
    for value in chosen:
        if not deduped or value != deduped[-1]:
            deduped.append(value)
    if deduped[-1] != snapshots[-1]:
        deduped.append(snapshots[-1])
    chosen = deduped
print(",".join(str(value) for value in chosen))
PY
}

LOSS_OUTDIR="${EVAL_SWEEP_ROOT}/posthoc_loss_curve"
FID_OUTDIR="${EVAL_SWEEP_ROOT}/posthoc_fid_curve"
LOSS_MANIFEST="${LOSS_OUTDIR}/cifar10_cdro_budget_manifest_dense_losscurve.csv"
LOSS_SUMMARY="${LOSS_OUTDIR}/cifar10_cdro_budget_manifest_dense_losscurve_summary.json"
LOSS_COMPARE="${LOSS_OUTDIR}/cifar10_baseline_vs_wdro_${PLOT_TAG}_dense_loss_compare.csv"
LOSS_COMPARE_SUMMARY="${LOSS_OUTDIR}/cifar10_baseline_vs_wdro_${PLOT_TAG}_dense_loss_compare_summary.json"
LOSS_PLOT_CSV="${LOSS_OUTDIR}/cifar10_baseline_vs_wdro_${PLOT_TAG}_dense_loss_comparison_zoomed.csv"
LOSS_PLOT_PNG="${LOSS_OUTDIR}/cifar10_baseline_vs_wdro_${PLOT_TAG}_dense_loss_comparison_zoomed.png"
FID_MANIFEST="${FID_OUTDIR}/cifar10_cdro_budget_manifest_coarse_fidcurve.csv"
FID_SUMMARY="${FID_OUTDIR}/cifar10_cdro_budget_manifest_coarse_fidcurve_summary.json"
FID_COMPARE="${FID_OUTDIR}/cifar10_baseline_vs_wdro_${PLOT_TAG}_coarse_fid_compare.csv"
FID_COMPARE_SUMMARY="${FID_OUTDIR}/cifar10_baseline_vs_wdro_${PLOT_TAG}_coarse_fid_compare_summary.json"
FID_PLOT_CSV="${FID_OUTDIR}/cifar10_baseline_vs_wdro_${PLOT_TAG}_coarse_fid_comparison_zoomed.csv"
FID_PLOT_PNG="${FID_OUTDIR}/cifar10_baseline_vs_wdro_${PLOT_TAG}_coarse_fid_comparison_zoomed.png"

mkdir -p "${LOSS_OUTDIR}" "${FID_OUTDIR}"

LOSS_KIMG_LIST="$(select_snapshot_kimg "${RUN_DIR}" "${LOSS_POINTS_MAX}")"
FID_KIMG_LIST="$(select_snapshot_kimg "${RUN_DIR}" "${FID_POINTS_MAX}")"
build_manifest_flop_args=()
if [[ -n "${FLOP_CALIBRATION_JSON}" ]]; then
  build_manifest_flop_args=(--flop-calibration-json "${FLOP_CALIBRATION_JSON}")
fi

echo "[INFO] Starting posthoc compare-curve generation on $(hostname)"
echo "[INFO] RUN_DIR: ${RUN_DIR}"
echo "[INFO] EVAL_SWEEP_ROOT: ${EVAL_SWEEP_ROOT}"
echo "[INFO] PLOT_TAG: ${PLOT_TAG}"
echo "[INFO] FLOP_CALIBRATION_JSON: ${FLOP_CALIBRATION_JSON:-none}"
echo "[INFO] LOSS_KIMG_LIST: ${LOSS_KIMG_LIST}"
echo "[INFO] FID_KIMG_LIST: ${FID_KIMG_LIST}"
date -u
nvidia-smi || true

echo "[INFO] Building dense loss manifest"
python "${ROOT_DIR}/scripts/build_cifar_cdro_fid_manifest.py" \
  --cdro-run-dir "${RUN_DIR}" \
  --outdir "${LOSS_OUTDIR}" \
  --manifest-name "$(basename "${LOSS_MANIFEST}")" \
  --summary-name "$(basename "${LOSS_SUMMARY}")" \
  --kimg "${LOSS_KIMG_LIST}" \
  --batch-size 1024 \
  --seed 0 \
  --train-percent-label "${TRAIN_PERCENT_LABEL}" \
  --ref-path "${PYTORCH_FID_REF}" \
  "${build_manifest_flop_args[@]}"

echo "[INFO] Running dense loss probes"
python "${ROOT_DIR}/scripts/run_cifar_loss_probe_manifest.py" \
  --manifest-csv "${LOSS_MANIFEST}" \
  --only-pending \
  --device cuda \
  --batch-size "${LOSS_PROBE_BATCH_SIZE}" \
  --max-images "${LOSS_PROBE_MAX_IMAGES}" \
  --seed 0

echo "[INFO] Merging dense loss compare CSV"
python "${ROOT_DIR}/scripts/merge_cifar_compare_csv.py" \
  --base-compare-csv "${BASE_COMPARE_CSV}" \
  --cdro-manifest-csv "${LOSS_MANIFEST}" \
  --out-csv "${LOSS_COMPARE}" \
  --out-summary-json "${LOSS_COMPARE_SUMMARY}" \
  --cdro-row-mode all

echo "[INFO] Plotting dense loss comparison"
python "${ROOT_DIR}/scripts/plot_cifar_two_method_loss_zoom.py" \
  --manifest-csv "${LOSS_COMPARE}" \
  --out-png "${LOSS_PLOT_PNG}" \
  --out-csv "${LOSS_PLOT_CSV}" \
  --dataset-label "${DATASET_LABEL}" \
  --train-percent-label "${TRAIN_PERCENT_LABEL}"

echo "[INFO] Building coarse FID manifest"
python "${ROOT_DIR}/scripts/build_cifar_cdro_fid_manifest.py" \
  --cdro-run-dir "${RUN_DIR}" \
  --outdir "${FID_OUTDIR}" \
  --manifest-name "$(basename "${FID_MANIFEST}")" \
  --summary-name "$(basename "${FID_SUMMARY}")" \
  --kimg "${FID_KIMG_LIST}" \
  --batch-size 1024 \
  --seed 0 \
  --train-percent-label "${TRAIN_PERCENT_LABEL}" \
  --ref-path "${PYTORCH_FID_REF}" \
  "${build_manifest_flop_args[@]}"

echo "[INFO] Running coarse FID evals"
python "${ROOT_DIR}/scripts/run_cifar_fid_manifest.py" \
  --manifest-csv "${FID_MANIFEST}" \
  --env-mode venv \
  --venv-dir "${VENV_DIR}" \
  --install-deps 0 \
  --prepare-dataset 0 \
  --fid-backend pytorch_fid \
  --ref-mode path \
  --ref-path "${PYTORCH_FID_REF}" \
  --only-pending \
  --num-images "${FID_NUM_IMAGES}" \
  --gen-batch "${FID_GEN_BATCH}" \
  --fid-batch "${FID_BATCH}" \
  --gen-steps "${FID_GEN_STEPS}" \
  --nproc-per-node "${NPROC_PER_NODE}"

echo "[INFO] Merging coarse FID compare CSV"
python "${ROOT_DIR}/scripts/merge_cifar_compare_csv.py" \
  --base-compare-csv "${BASE_COMPARE_CSV}" \
  --cdro-manifest-csv "${FID_MANIFEST}" \
  --out-csv "${FID_COMPARE}" \
  --out-summary-json "${FID_COMPARE_SUMMARY}" \
  --cdro-row-mode fid_evaluated

echo "[INFO] Plotting coarse FID comparison"
python "${ROOT_DIR}/scripts/plot_cifar_two_method_fid_zoom.py" \
  --manifest-csv "${FID_COMPARE}" \
  --out-png "${FID_PLOT_PNG}" \
  --out-csv "${FID_PLOT_CSV}" \
  --dataset-label "${DATASET_LABEL}" \
  --train-percent-label "${TRAIN_PERCENT_LABEL}"

echo "[OK] Posthoc compare-curve generation complete for ${PLOT_TAG}."
date -u
