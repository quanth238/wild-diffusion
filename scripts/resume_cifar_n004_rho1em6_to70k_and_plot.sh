#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

TARGET_KIMG="${TARGET_KIMG:-70000}"
TARGET_MIMG="$(python3 - "${TARGET_KIMG}" <<'PY'
import sys
print(f"{int(sys.argv[1]) / 1000.0:.6f}")
PY
)"

VENV_DIR="${VENV_DIR:-/home/bachlc/.venvs/wild-diffusion-h100}"
RUN_DIR="${RUN_DIR:-/mnt/data/bachlc/GM-CDRO/training-runs/paper-cifar10-cdro-fp16-n004-rho1em6-edm200mimg_20260422T150118Z/rho1em06/00000-cifar10-32x32-train20pct-seed0-uncond-ddpmpp-cdroedm-gpus1-batch1024-fp16-paper-cifar10-uncond-ddpmpp-cdro-20pct-n004-rho0p0-i1-aw1p00-cw0p00-pel1-bg1024-resume040000-wcu460371}"
SWEEP_ROOT="${SWEEP_ROOT:-/mnt/data/bachlc/GM-CDRO/training-runs/fid-sweeps/cdro_n004_rho1em6_edm200mimg_20260422T150118Z/rho1em06}"
LOG_DIR="${LOG_DIR:-/mnt/data/bachlc/GM-CDRO/training-runs/fid-sweeps/logs}"
RUN_LOG="${RUN_LOG:-${LOG_DIR}/cdro_n004_rho1em6_resume${TARGET_KIMG}k_and_plot_$(date -u +%Y%m%dT%H%M%SZ).log}"
TRAIN_TICK_KIMG="${TRAIN_TICK_KIMG:-128}"
TRAIN_SNAP_TICKS="${TRAIN_SNAP_TICKS:-1}"
TRAIN_DUMP_TICKS="${TRAIN_DUMP_TICKS:-1}"
FLOP_CALIBRATION_JSON="${FLOP_CALIBRATION_JSON:-}"
EXTRA_FID_KIMG="${EXTRA_FID_KIMG:-}"
EXTRA_LOSS_KIMG="${EXTRA_LOSS_KIMG:-}"

FID_OUTDIR="${FID_OUTDIR:-${SWEEP_ROOT}/posthoc_fid_curve}"
LOSS_OUTDIR="${LOSS_OUTDIR:-${SWEEP_ROOT}/posthoc_loss_curve}"
PYTORCH_FID_REF="${PYTORCH_FID_REF:-${ROOT_DIR}/training-runs/fid-sweeps/cifar10_baseline_vs_wdro_coarse_20260414/pytorch_fid_cifar10_train_ref_stats.npz}"
BASE_COMPARE_CSV="${BASE_COMPARE_CSV:-${ROOT_DIR}/training-runs/fid-sweeps/cifar10_baseline_vs_wdro_coarse_20260414/cifar10_baseline_vs_wdro_coarse_20260414_three_method_compare.csv}"

FID_MANIFEST="${FID_OUTDIR}/cifar10_cdro_budget_manifest_coarse_fidcurve.csv"
FID_COMPARE="${FID_OUTDIR}/cifar10_baseline_vs_wdro_cdro_n004_rho1em6_coarse_fid_compare.csv"
FID_COMPARE_SUMMARY="${FID_OUTDIR}/cifar10_baseline_vs_wdro_cdro_n004_rho1em6_coarse_fid_compare_summary.json"
FID_PLOT_CSV="${FID_OUTDIR}/cifar10_baseline_vs_wdro_cdro_n004_rho1em6_coarse_fid_comparison_zoomed.csv"
FID_PLOT_PNG="${FID_OUTDIR}/cifar10_baseline_vs_wdro_cdro_n004_rho1em6_coarse_fid_comparison_zoomed.png"

LOSS_MANIFEST="${LOSS_OUTDIR}/cifar10_cdro_budget_manifest_dense_losscurve.csv"
LOSS_COMPARE="${LOSS_OUTDIR}/cifar10_baseline_vs_wdro_cdro_n004_rho1em6_dense_loss_compare.csv"
LOSS_COMPARE_SUMMARY="${LOSS_OUTDIR}/cifar10_baseline_vs_wdro_cdro_n004_rho1em6_dense_loss_compare_summary.json"
LOSS_PLOT_CSV="${LOSS_OUTDIR}/cifar10_baseline_vs_wdro_cdro_n004_rho1em6_dense_loss_comparison_zoomed.csv"
LOSS_PLOT_PNG="${LOSS_OUTDIR}/cifar10_baseline_vs_wdro_cdro_n004_rho1em6_dense_loss_comparison_zoomed.png"

mkdir -p "${LOG_DIR}" "${FID_OUTDIR}" "${LOSS_OUTDIR}"

if [[ -r /proc/self/fd ]]; then
  exec > >(tee -a "${RUN_LOG}") 2>&1
else
  exec >>"${RUN_LOG}" 2>&1
fi

echo "[INFO] Started at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[INFO] Host: $(hostname)"
echo "[INFO] Target kimg: ${TARGET_KIMG}"
echo "[INFO] Run dir: ${RUN_DIR}"
echo "[INFO] Run log: ${RUN_LOG}"
echo "[INFO] Train tick kimg: ${TRAIN_TICK_KIMG}"
echo "[INFO] Snapshot ticks: ${TRAIN_SNAP_TICKS}"
echo "[INFO] State dump ticks: ${TRAIN_DUMP_TICKS}"
echo "[INFO] FLOP calibration JSON: ${FLOP_CALIBRATION_JSON:-none}"

for path in "${RUN_DIR}" "${VENV_DIR}" "${PYTORCH_FID_REF}" "${BASE_COMPARE_CSV}"; do
  if [[ ! -e "${path}" ]]; then
    echo "[ERROR] Required path is missing: ${path}"
    exit 1
  fi
done
if [[ -n "${FLOP_CALIBRATION_JSON}" && ! -f "${FLOP_CALIBRATION_JSON}" ]]; then
  echo "[ERROR] FLOP calibration JSON is missing: ${FLOP_CALIBRATION_JSON}"
  exit 1
fi

for cmd_name in python3 python bash find sort tail tee; do
  if ! command -v "${cmd_name}" >/dev/null 2>&1; then
    echo "[ERROR] Missing command: ${cmd_name}"
    exit 1
  fi
done

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ ! -r /proc/self/maps || ! -r /proc/cpuinfo || ! -r /proc/sys/vm/mmap_min_addr ]]; then
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/scripts/ensure_procfs_compat.sh"
fi

python - <<'PY'
import socket
import torch

print(f"[INFO] CUDA probe host={socket.gethostname()} available={torch.cuda.is_available()} count={torch.cuda.device_count()}", flush=True)
if not torch.cuda.is_available():
    raise SystemExit("[ERROR] CUDA is not available in this Slurm step.")
print(f"[INFO] CUDA device 0: {torch.cuda.get_device_name(0)}", flush=True)
PY

latest_training_state() {
  find -L "${RUN_DIR}" -mindepth 1 -maxdepth 1 -type f -name 'training-state-*.pt' | sort | tail -n 1
}

latest_snapshot_kimg() {
  python - "${RUN_DIR}" <<'PY'
import re
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
values = []
for path in run_dir.glob("network-snapshot-*.pkl"):
    match = re.fullmatch(r"network-snapshot-(\d+)\.pkl", path.name)
    if match:
        values.append(int(match.group(1)))
print(max(values) if values else 0)
PY
}

append_unique_kimg_csv() {
  python - "$@" <<'PY'
import csv
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
target = int(sys.argv[2])
fallback = [int(x) for x in sys.argv[3].split(",") if x]
values = []
if manifest.is_file():
    with manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("robust_method", row.get("method", "")) != "cdro":
                continue
            value = row.get("snapshot_kimg") or row.get("step")
            if value:
                values.append(int(float(value)))
else:
    values.extend(fallback)
values.append(target)
print(",".join(str(x) for x in sorted(set(values))))
PY
}

restore_loss_probe_paths() {
  python - "${LOSS_MANIFEST}" <<'PY'
import csv
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
rows = list(csv.DictReader(manifest.open(newline="", encoding="utf-8")))
fieldnames = list(rows[0].keys()) if rows else []
extra_fields = [
    "loss_probe_result_json",
    "loss_probe_result_txt",
    "loss_probe_eval_dir",
]
for key in extra_fields:
    if key not in fieldnames:
        fieldnames.append(key)
for row in rows:
    eval_dir = Path(row.get("eval_dir") or Path(row["eval_root"]) / row["eval_tag"]).resolve()
    row["loss_probe_eval_dir"] = str(eval_dir)
    row["loss_probe_result_json"] = str(eval_dir / "loss_probe_result.json")
    row["loss_probe_result_txt"] = str(eval_dir / "loss_probe_result.txt")
with manifest.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
PY
}

current_kimg="$(latest_snapshot_kimg)"
echo "[INFO] Latest snapshot before resume: ${current_kimg}"

if (( current_kimg < TARGET_KIMG )); then
  resume_state="$(latest_training_state)"
  if [[ -z "${resume_state}" || ! -f "${resume_state}" ]]; then
    echo "[ERROR] Could not find a training-state checkpoint in ${RUN_DIR}"
    exit 1
  fi
  echo "[INFO] Resuming from ${resume_state##*/} to ${TARGET_KIMG} kimg"
  RESUME="${resume_state}" \
  DURATION_MIMG="${TARGET_MIMG}" \
  BATCH=1024 \
  BATCH_GPU=1024 \
  LR=1e-5 \
  WORKERS=16 \
  ARCH=ddpmpp \
  PRECOND=cdroedm \
  COND=0 \
  FP16=1 \
  AUGMENT=0.12 \
  DEBUG_EVAL=0 \
  DEBUG_ADV_VISUAL=0 \
  CIFAR_TRAIN_PERCENT=20 \
  CIFAR_TRAIN_SEED=0 \
  TICK_KIMG="${TRAIN_TICK_KIMG}" \
  SNAP_TICKS="${TRAIN_SNAP_TICKS}" \
  DUMP_TICKS="${TRAIN_DUMP_TICKS}" \
  SEED=0 \
  ENV_MODE=venv \
  INSTALL_DEPS=0 \
  VENV_DIR="${VENV_DIR}" \
  EXTRA_TRAIN_ARGS="--cdro-n-steps-path=4 --cdro-step-size=0.02 --cdro-total-budget-rho=1e-06 --cdro-time-horizon=1.0 --cdro-sigma-min=0.002 --cdro-sigma-max=80.0 --cdro-edm-ladder-mode=stochastic_stratified_quantile --cdro-per-example-sigma-ladders=True --attack-num-steps=1 --outer-attack-weight=1.0 --outer-clean-weight=0.0" \
    bash "${ROOT_DIR}/scripts/setup_and_train_cifar10.sh"
else
  echo "[SKIP] Training already has snapshot >= ${TARGET_KIMG} kimg"
fi

build_manifest_flop_args=()
if [[ -n "${FLOP_CALIBRATION_JSON}" ]]; then
  build_manifest_flop_args=(--flop-calibration-json "${FLOP_CALIBRATION_JSON}")
fi

final_kimg="$(latest_snapshot_kimg)"
echo "[INFO] Latest snapshot after resume: ${final_kimg}"
if (( final_kimg < TARGET_KIMG )); then
  echo "[ERROR] Final snapshot ${final_kimg} is below target ${TARGET_KIMG}"
  exit 1
fi

backup_tag="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ -f "${FID_MANIFEST}" ]]; then
  cp -p "${FID_MANIFEST}" "${FID_MANIFEST}.bak_${backup_tag}"
fi
if [[ -f "${LOSS_MANIFEST}" ]]; then
  cp -p "${LOSS_MANIFEST}" "${LOSS_MANIFEST}.bak_${backup_tag}"
fi

fid_seed_kimg="40000,41921,43969,46017,48065,50113,52161,61967"
loss_seed_kimg="40000,40129,40513,40769,41025,41281,41665,41921,42177,42561,42817,43073,43329,43713,43969,44225,44609,44865,45121,45377,45761,46017,46273,46657,46913,47169,47425,47809,48065,48321,48705,48961,49217,49473,49857,50113,50369,50753,51009,51265,51521,51905,52161,52417,52801,52930,54978,57026,60098,61967"
if [[ -n "${EXTRA_FID_KIMG}" ]]; then
  fid_seed_kimg="${fid_seed_kimg},${EXTRA_FID_KIMG}"
fi
if [[ -n "${EXTRA_LOSS_KIMG}" ]]; then
  loss_seed_kimg="${loss_seed_kimg},${EXTRA_LOSS_KIMG}"
fi
fid_kimg_list="$(append_unique_kimg_csv "${FID_MANIFEST}" "${final_kimg}" "${fid_seed_kimg}")"
loss_kimg_list="$(append_unique_kimg_csv "${LOSS_MANIFEST}" "${final_kimg}" "${loss_seed_kimg}")"

echo "[INFO] Rebuilding loss manifest for kimg: ${loss_kimg_list}"
python "${ROOT_DIR}/scripts/build_cifar_cdro_fid_manifest.py" \
  --cdro-run-dir "${RUN_DIR}" \
  --outdir "${LOSS_OUTDIR}" \
  --manifest-name "$(basename "${LOSS_MANIFEST}")" \
  --summary-name "cifar10_cdro_budget_manifest_dense_losscurve_summary.json" \
  --kimg "${loss_kimg_list}" \
  --batch-size 1024 \
  --seed 0 \
  --train-percent-label "20%" \
  --ref-path "${PYTORCH_FID_REF}" \
  "${build_manifest_flop_args[@]}"

restore_loss_probe_paths

echo "[INFO] Running posthoc clean loss probes; existing rows should be reused"
python "${ROOT_DIR}/scripts/run_cifar_loss_probe_manifest.py" \
  --manifest-csv "${LOSS_MANIFEST}" \
  --only-pending \
  --device cuda \
  --batch-size 512 \
  --max-images 10000 \
  --seed 0

echo "[INFO] Plotting updated loss curve"
python "${ROOT_DIR}/scripts/merge_cifar_compare_csv.py" \
  --base-compare-csv "${BASE_COMPARE_CSV}" \
  --cdro-manifest-csv "${LOSS_MANIFEST}" \
  --out-csv "${LOSS_COMPARE}" \
  --out-summary-json "${LOSS_COMPARE_SUMMARY}" \
  --cdro-row-mode all

python "${ROOT_DIR}/scripts/plot_cifar_two_method_loss_zoom.py" \
  --manifest-csv "${LOSS_COMPARE}" \
  --out-png "${LOSS_PLOT_PNG}" \
  --out-csv "${LOSS_PLOT_CSV}" \
  --dataset-label CIFAR-10 \
  --train-percent-label "20%"

echo "[INFO] Rebuilding FID manifest for kimg: ${fid_kimg_list}"
python "${ROOT_DIR}/scripts/build_cifar_cdro_fid_manifest.py" \
  --cdro-run-dir "${RUN_DIR}" \
  --outdir "${FID_OUTDIR}" \
  --manifest-name "$(basename "${FID_MANIFEST}")" \
  --summary-name "cifar10_cdro_budget_manifest_coarse_fidcurve_summary.json" \
  --kimg "${fid_kimg_list}" \
  --batch-size 1024 \
  --seed 0 \
  --train-percent-label "20%" \
  --ref-path "${PYTORCH_FID_REF}" \
  "${build_manifest_flop_args[@]}"

echo "[INFO] Running posthoc FID; existing rows should be reused"
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
  --num-images 50000 \
  --gen-batch 128 \
  --fid-batch 64 \
  --gen-steps 18 \
  --nproc-per-node 1

echo "[INFO] Plotting updated FID curve"
python "${ROOT_DIR}/scripts/merge_cifar_compare_csv.py" \
  --base-compare-csv "${BASE_COMPARE_CSV}" \
  --cdro-manifest-csv "${FID_MANIFEST}" \
  --out-csv "${FID_COMPARE}" \
  --out-summary-json "${FID_COMPARE_SUMMARY}" \
  --cdro-row-mode fid_evaluated

python "${ROOT_DIR}/scripts/plot_cifar_two_method_fid_zoom.py" \
  --manifest-csv "${FID_COMPARE}" \
  --out-png "${FID_PLOT_PNG}" \
  --out-csv "${FID_PLOT_CSV}" \
  --dataset-label CIFAR-10 \
  --train-percent-label "20%"

python - "${FID_MANIFEST}" "${LOSS_MANIFEST}" <<'PY'
import csv
import sys
from pathlib import Path

fid_rows = list(csv.DictReader(Path(sys.argv[1]).open(newline="", encoding="utf-8")))
loss_rows = list(csv.DictReader(Path(sys.argv[2]).open(newline="", encoding="utf-8")))
cdro_fid = [r for r in fid_rows if r.get("robust_method") == "cdro" and r.get("fid")]
cdro_loss = [r for r in loss_rows if r.get("robust_method") == "cdro" and (r.get("loss_probe_clean") or r.get("loss_comparable_final"))]
if cdro_fid:
    last = max(cdro_fid, key=lambda r: int(float(r["snapshot_kimg"])))
    print(f"[RESULT] Final FID row: kimg={last['snapshot_kimg']} fid={float(last['fid']):.6g}")
if cdro_loss:
    last = max(cdro_loss, key=lambda r: int(float(r["snapshot_kimg"])))
    loss = last.get("loss_probe_clean") or last.get("loss_comparable_final")
    print(f"[RESULT] Final loss row: kimg={last['snapshot_kimg']} clean_loss={float(loss):.9g}")
PY

echo "[INFO] Refreshing lightweight comparison graph workspace"
python "${ROOT_DIR}/scripts/sync_comparison_graphs.py" \
  --source-root "${FID_OUTDIR}" \
  --source-root "${LOSS_OUTDIR}" \
  --source-root "$(dirname "${BASE_COMPARE_CSV}")"

echo "[OK] Completed at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[OK] Loss plot: ${LOSS_PLOT_PNG}"
echo "[OK] FID plot: ${FID_PLOT_PNG}"
