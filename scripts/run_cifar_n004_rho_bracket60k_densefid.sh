#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

VENV_DIR="${VENV_DIR:-/home/bachlc/.venvs/wild-diffusion-h100}"
RHO_VALUES="${RHO_VALUES:-1e-12,1e-10,1e-8}"
SOURCE_KIMG="${SOURCE_KIMG:-50000}"
TARGET_KIMG="${TARGET_KIMG:-60000}"
MIN_EVAL_KIMG="${MIN_EVAL_KIMG:-52000}"
CAMPAIGN_TAG="${CAMPAIGN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"

TRAIN_OUTROOT_BASE="${TRAIN_OUTROOT_BASE:-${ROOT_DIR}/training-runs/paper-cifar10-cdro-fp16-n004-rho-bracket50k_20260425T144042Z}"
OLD_EVAL_ROOT_BASE="${OLD_EVAL_ROOT_BASE:-${ROOT_DIR}/training-runs/fid-sweeps/cdro_n004_rho_bracket50k_20260425T144042Z}"
EVAL_SWEEP_ROOT_BASE="${EVAL_SWEEP_ROOT_BASE:-${ROOT_DIR}/training-runs/fid-sweeps/cdro_n004_rho_bracket60k_densefid_${CAMPAIGN_TAG}}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/training-runs/fid-sweeps/logs}"
RUN_LOG="${RUN_LOG:-${LOG_DIR}/cdro_n004_rho_bracket60k_densefid_${CAMPAIGN_TAG}.log}"
PYTORCH_FID_REF="${PYTORCH_FID_REF:-${ROOT_DIR}/training-runs/fid-sweeps/cifar10_baseline_vs_wdro_coarse_20260414/pytorch_fid_cifar10_train_ref_stats.npz}"

TRAIN_LR="${TRAIN_LR:-1e-5}"
TRAIN_TICK_KIMG="${TRAIN_TICK_KIMG:-2000}"
TRAIN_SNAP_TICKS="${TRAIN_SNAP_TICKS:-1}"
TRAIN_DUMP_TICKS="${TRAIN_DUMP_TICKS:-1}"

FID_NUM_IMAGES="${FID_NUM_IMAGES:-50000}"
FID_GEN_BATCH="${FID_GEN_BATCH:-128}"
FID_BATCH="${FID_BATCH:-64}"
FID_GEN_STEPS="${FID_GEN_STEPS:-18}"
LOSS_PROBE_MAX_IMAGES="${LOSS_PROBE_MAX_IMAGES:-10000}"
LOSS_PROBE_BATCH_SIZE="${LOSS_PROBE_BATCH_SIZE:-512}"

mkdir -p "${LOG_DIR}" "${EVAL_SWEEP_ROOT_BASE}"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "[ERROR] Missing venv: ${VENV_DIR}"
  exit 1
fi
if [[ ! -d "${TRAIN_OUTROOT_BASE}" ]]; then
  echo "[ERROR] Missing train outroot base: ${TRAIN_OUTROOT_BASE}"
  exit 1
fi
if [[ ! -d "${OLD_EVAL_ROOT_BASE}" ]]; then
  echo "[ERROR] Missing old eval root base: ${OLD_EVAL_ROOT_BASE}"
  exit 1
fi
if [[ ! -f "${PYTORCH_FID_REF}" ]]; then
  echo "[ERROR] Missing pytorch-fid reference stats: ${PYTORCH_FID_REF}"
  exit 1
fi

for cmd_name in python bash find tee sort; do
  if ! command -v "${cmd_name}" >/dev/null 2>&1; then
    echo "[ERROR] Missing command: ${cmd_name}"
    exit 1
  fi
done

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

rho_tag() {
  python - "$1" <<'PY'
import sys

value = float(sys.argv[1])
text = format(value, ".6g")
print("rho" + text.replace("-", "m").replace("+", "").replace(".", "p"))
PY
}

latest_run_dir() {
  local train_outroot="$1"
  find -L "${train_outroot}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1
}

latest_training_state() {
  local run_dir="$1"
  find -L "${run_dir}" -mindepth 1 -maxdepth 1 -type f -name 'training-state-*.pt' | sort | tail -n 1
}

discover_eval_kimg_csv() {
  local run_dir="$1"
  python - "${run_dir}" "${MIN_EVAL_KIMG}" "${TARGET_KIMG}" <<'PY'
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
min_kimg = int(sys.argv[2])
target_kimg = int(sys.argv[3])
values = []
for path in run_dir.glob("network-snapshot-*.pkl"):
    suffix = path.stem.replace("network-snapshot-", "")
    if not suffix.isdigit():
        continue
    value = int(suffix)
    if min_kimg <= value <= target_kimg:
        values.append(value)
values = sorted(set(values))
if not values:
    raise SystemExit(f"No snapshots in [{min_kimg}, {target_kimg}] under {run_dir}")
print(",".join(str(value) for value in values))
PY
}

resume_existing_run() {
  local run_dir="$1"
  local rho="$2"
  local resume_state

  resume_state="$(latest_training_state "${run_dir}")"
  if [[ -z "${resume_state}" || ! -f "${resume_state}" ]]; then
    echo "[ERROR] Could not find a resume state in ${run_dir}"
    exit 1
  fi

  echo "[INFO] Resuming N=4 rho=${rho} from ${resume_state##*/} to ${TARGET_KIMG} kimg"
  RESUME="${resume_state}" \
  DURATION_MIMG="$(python -c "print(${TARGET_KIMG} / 1000.0)")" \
  BATCH=1024 \
  BATCH_GPU=1024 \
  LR="${TRAIN_LR}" \
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
  EXTRA_TRAIN_ARGS="--cdro-n-steps-path=4 --cdro-step-size=0.02 --cdro-total-budget-rho=${rho} --cdro-time-horizon=1.0 --cdro-sigma-min=0.002 --cdro-sigma-max=80.0 --cdro-edm-ladder-mode=stochastic_stratified_quantile --cdro-per-example-sigma-ladders=True --attack-num-steps=1 --outer-attack-weight=1.0 --outer-clean-weight=0.0" \
    bash "${ROOT_DIR}/scripts/setup_and_train_cifar10.sh"
}

ensure_rho_to_target() {
  local rho="$1"
  local rho_dir_tag="$2"
  local train_outroot="${TRAIN_OUTROOT_BASE}/${rho_dir_tag}"
  local run_dir
  local target_state

  if [[ ! -d "${train_outroot}" ]]; then
    echo "[ERROR] Missing train outroot for rho=${rho}: ${train_outroot}"
    exit 1
  fi

  run_dir="$(latest_run_dir "${train_outroot}")"
  if [[ -z "${run_dir}" || ! -d "${run_dir}" ]]; then
    echo "[ERROR] Could not resolve run dir for rho=${rho} under ${train_outroot}"
    exit 1
  fi

  if [[ ! -f "${run_dir}/training-state-$(printf '%06d' "${SOURCE_KIMG}").pt" ]]; then
    echo "[ERROR] Missing source ${SOURCE_KIMG} state for rho=${rho}: ${run_dir}"
    exit 1
  fi

  target_state="$(printf "%s/training-state-%06d.pt" "${run_dir}" "${TARGET_KIMG}")"
  if [[ -f "${target_state}" ]]; then
    echo "[SKIP] Found ${target_state}; skipping training for rho=${rho}"
  else
    resume_existing_run "${run_dir}" "${rho}"
  fi

  DENSE_RUN_DIR="${run_dir}"
}

build_and_run_evals() {
  local rho="$1"
  local rho_dir_tag="$2"
  local run_dir="$3"
  local eval_outdir="${EVAL_SWEEP_ROOT_BASE}/${rho_dir_tag}"
  local manifest_csv="${eval_outdir}/cifar10_cdro_budget_manifest_densefid_50k60k.csv"
  local summary_json="${eval_outdir}/cifar10_cdro_budget_manifest_densefid_50k60k_summary.json"
  local kimg_csv

  mkdir -p "${eval_outdir}"
  kimg_csv="$(discover_eval_kimg_csv "${run_dir}")"
  echo "[INFO] Building dense FID manifest for rho=${rho}; kimg=${kimg_csv}"
  python "${ROOT_DIR}/scripts/build_cifar_cdro_fid_manifest.py" \
    --cdro-run-dir "${run_dir}" \
    --outdir "${eval_outdir}" \
    --manifest-name "$(basename "${manifest_csv}")" \
    --summary-name "$(basename "${summary_json}")" \
    --kimg "${kimg_csv}" \
    --batch-size 1024 \
    --seed 0 \
    --train-percent-label "20%" \
    --ref-path "${PYTORCH_FID_REF}"

  echo "[INFO] Running pytorch-fid evals for rho=${rho}"
  python "${ROOT_DIR}/scripts/run_cifar_fid_manifest.py" \
    --manifest-csv "${manifest_csv}" \
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
    --nproc-per-node 1

  echo "[INFO] Running clean EDM probes for rho=${rho}"
  python "${ROOT_DIR}/scripts/run_cifar_loss_probe_manifest.py" \
    --manifest-csv "${manifest_csv}" \
    --only-pending \
    --device cuda \
    --batch-size "${LOSS_PROBE_BATCH_SIZE}" \
    --max-images "${LOSS_PROBE_MAX_IMAGES}" \
    --seed 0
}

write_combined_outputs() {
  python - "${OLD_EVAL_ROOT_BASE}" "${EVAL_SWEEP_ROOT_BASE}" "${RHO_VALUES}" <<'PY'
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

old_root = Path(sys.argv[1])
new_root = Path(sys.argv[2])
rho_values = [item.strip() for item in sys.argv[3].split(",") if item.strip()]

def rho_tag(rho):
    return "rho" + format(float(rho), ".6g").replace("-", "m").replace("+", "").replace(".", "p")

def read_rows(path):
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))

rows = []
for rho in rho_values:
    tag = rho_tag(rho)
    old_manifest = old_root / tag / "cifar10_cdro_budget_manifest.csv"
    new_manifest = new_root / tag / "cifar10_cdro_budget_manifest_densefid_50k60k.csv"
    for row in read_rows(old_manifest):
        if int(float(row.get("step", 0))) == 50000:
            out = dict(row)
            out["rho"] = rho
            out["rho_tag"] = tag
            out["dense_source_manifest"] = str(old_manifest)
            rows.append(out)
    for row in read_rows(new_manifest):
        out = dict(row)
        out["rho"] = rho
        out["rho_tag"] = tag
        out["dense_source_manifest"] = str(new_manifest)
        rows.append(out)

if not rows:
    raise SystemExit("No rows found to summarize.")

preferred = [
    "rho",
    "rho_tag",
    "step",
    "images_shown_m",
    "fid",
    "loss_probe_clean",
    "loss_final",
    "total_train_pflops",
    "train_wall_clock_sec",
    "weighted_compute_units",
    "network_pkl",
    "dense_source_manifest",
]
fieldnames = preferred + sorted({key for row in rows for key in row if key not in preferred})
summary_csv = new_root / "rho_bracket50k_to60k_densefid_results.csv"
with summary_csv.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
print(f"[OK] Wrote summary CSV: {summary_csv}")

def f(row, key):
    try:
        value = float(row.get(key, ""))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value

plot_rows = [row for row in rows if f(row, "fid") is not None and f(row, "step") is not None]
by_rho = defaultdict(list)
for row in plot_rows:
    by_rho[row["rho"]].append(row)
for rho in by_rho:
    by_rho[rho].sort(key=lambda item: f(item, "step"))

colors = {"1e-12": "#66a61e", "1e-10": "#e7298a", "1e-8": "#7570b3"}
markers = {"1e-12": "X", "1e-10": "P", "1e-8": "D"}

fig, ax = plt.subplots(figsize=(9.5, 5.6), dpi=160)
for rho in rho_values:
    points = by_rho.get(rho, [])
    if not points:
        continue
    ax.plot(
        [f(row, "step") / 1000.0 for row in points],
        [f(row, "fid") for row in points],
        marker=markers.get(rho, "o"),
        linewidth=2.2,
        markersize=6,
        color=colors.get(rho),
        label=f"rho={rho}",
    )
ax.set_title("N=4 CDRO lower-rho dense FID trajectory")
ax.set_xlabel("Clean MIMG")
ax.set_ylabel("FID, lower is better")
ax.grid(True, color="#d9d9d9", linewidth=0.8, alpha=0.8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(frameon=False)
fig.tight_layout()
fid_png = new_root / "rho_bracket50k_to60k_densefid_fid_trajectory.png"
fig.savefig(fid_png)
plt.close(fig)
print(f"[OK] Wrote FID plot: {fid_png}")

loss_rows = [row for row in rows if (f(row, "loss_probe_clean") is not None or f(row, "loss_comparable_final") is not None) and f(row, "step") is not None]
by_rho = defaultdict(list)
for row in loss_rows:
    by_rho[row["rho"]].append(row)
for rho in by_rho:
    by_rho[rho].sort(key=lambda item: f(item, "step"))

fig, ax = plt.subplots(figsize=(9.5, 5.6), dpi=160)
for rho in rho_values:
    points = by_rho.get(rho, [])
    if not points:
        continue
    ys = []
    xs = []
    for row in points:
        loss = f(row, "loss_probe_clean")
        if loss is None:
            loss = f(row, "loss_comparable_final")
        if loss is None:
            continue
        xs.append(f(row, "step") / 1000.0)
        ys.append(loss)
    ax.plot(
        xs,
        ys,
        marker=markers.get(rho, "o"),
        linewidth=2.2,
        markersize=6,
        color=colors.get(rho),
        label=f"rho={rho}",
    )
ax.set_title("N=4 CDRO lower-rho clean loss probe trajectory")
ax.set_xlabel("Clean MIMG")
ax.set_ylabel("Clean EDM loss probe, lower is better")
ax.grid(True, color="#d9d9d9", linewidth=0.8, alpha=0.8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(frameon=False)
fig.tight_layout()
loss_png = new_root / "rho_bracket50k_to60k_densefid_clean_loss_trajectory.png"
fig.savefig(loss_png)
plt.close(fig)
print(f"[OK] Wrote loss plot: {loss_png}")
PY
}

echo "[INFO] Starting N=4 lower-rho dense-FID continuation on $(hostname)"
echo "[INFO] Campaign tag: ${CAMPAIGN_TAG}"
echo "[INFO] Source kimg: ${SOURCE_KIMG}"
echo "[INFO] Target kimg: ${TARGET_KIMG}"
echo "[INFO] RHO_VALUES: ${RHO_VALUES}"
echo "[INFO] Train outroot base: ${TRAIN_OUTROOT_BASE}"
echo "[INFO] Old eval root base: ${OLD_EVAL_ROOT_BASE}"
echo "[INFO] New eval root base: ${EVAL_SWEEP_ROOT_BASE}"
echo "[INFO] Run log: ${RUN_LOG}"
echo "[INFO] TRAIN_TICK_KIMG: ${TRAIN_TICK_KIMG}"
echo "[INFO] TRAIN_SNAP_TICKS: ${TRAIN_SNAP_TICKS}"
echo "[INFO] TRAIN_DUMP_TICKS: ${TRAIN_DUMP_TICKS}"
date -u
nvidia-smi || true

IFS=',' read -r -a rho_list <<< "${RHO_VALUES}"
for raw_rho in "${rho_list[@]}"; do
  rho="$(echo "${raw_rho}" | tr -d '[:space:]')"
  if [[ -z "${rho}" ]]; then
    continue
  fi
  tag="$(rho_tag "${rho}")"
  DENSE_RUN_DIR=""
  ensure_rho_to_target "${rho}" "${tag}"
  run_dir="${DENSE_RUN_DIR}"
  if [[ -z "${run_dir}" ]]; then
    echo "[ERROR] Failed to resolve run dir for rho=${rho}"
    exit 1
  fi
  echo "[INFO] Completed training check for rho=${rho}; run_dir=${run_dir}"
  build_and_run_evals "${rho}" "${tag}" "${run_dir}"
  write_combined_outputs
done

write_combined_outputs
echo "[OK] N=4 lower-rho dense-FID continuation complete."
date -u
