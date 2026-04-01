#!/bin/bash
#SBATCH --job-name=mnist_b_v11_v2_v21
#SBATCH --output=toy_outputs/slurm_b_v11_v2_v21_%j.out
#SBATCH --error=toy_outputs/slurm_b_v11_v2_v21_%j.err
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G

set -euo pipefail

echo "[*] SLURM Job started on $(hostname)"
echo "[*] Current directory: $(pwd)"
echo "[*] SLURM_JOB_ID=${SLURM_JOB_ID:-N/A} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-N/A}"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "[ERROR] nvidia-smi not found"
  exit 1
fi

source /home/quanth/working_space/scripts/conda.sh
cd /home/quanth/working_space/Wild-Diffusion

export CUDA_VISIBLE_DEVICES=0
export FID_DETECTOR_PATH=/mnt/data/quanth/models/inception-2015-12-05.pkl

python3 - <<'PY'
import torch
print('[torch] cuda_available:', torch.cuda.is_available())
print('[torch] device_count:', torch.cuda.device_count())
if not torch.cuda.is_available() or torch.cuda.device_count() <= 0:
    raise SystemExit('[ERROR] CUDA is required for this workflow, but no GPU is visible.')
print('[torch] current_device:', torch.cuda.current_device())
print('[torch] device_name:', torch.cuda.get_device_name(torch.cuda.current_device()))
PY

SEED="${SEED:-0}"
STEPS="${STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
EVAL_SAMPLES="${EVAL_SAMPLES:-2000}"
FID_SAMPLES="${FID_SAMPLES:-2000}"
MNIST_TRAIN_PERCENT="${MNIST_TRAIN_PERCENT:-20}"
MNIST_VAL_PERCENT="${MNIST_VAL_PERCENT:-100}"
OUTDIR="${OUTDIR:-/mnt/data/quanth/wild_diffusion_outputs/mnist_baseline_v11_v2_v21}"
PREFIX="${PREFIX:-mnist_${MNIST_TRAIN_PERCENT}pct_s${SEED}_t${STEPS}}"
V21_RHO="${V21_RHO:-0.8}"
V11_STEP_SIZE="${V11_STEP_SIZE:-0.0005}"
V11_GAMMA="${V11_GAMMA:-2.0}"
V11_RHO="${V11_RHO:-0.02}"
V11_PROJ_MODE="${V11_PROJ_MODE:-global_remaining}"
PICK_SEED="${PICK_SEED:-7}"
SIGMA_MIN="${SIGMA_MIN:-0.01}"
SIGMA_MAX="${SIGMA_MAX:-80}"
AUTO_LOG_NORMAL="${AUTO_LOG_NORMAL:-1}"
USE_EMA_EVAL="${USE_EMA_EVAL:-1}"
EMA_DECAY="${EMA_DECAY:-0.999}"
WARMUP_CLEAN_STEPS="${WARMUP_CLEAN_STEPS:-900}"
WARMUP_RAMP_STEPS="${WARMUP_RAMP_STEPS:-600}"

export SEED
export STEPS
export MNIST_TRAIN_PERCENT
export MNIST_VAL_PERCENT
export OUTDIR
export PREFIX
export PICK_SEED

mkdir -p "${OUTDIR}"
BASELINE_CKPT_DIR="${BASELINE_CKPT_DIR:-/mnt/data/quanth/models/wild_diffusion_baseline_ckpt}"
mkdir -p "${BASELINE_CKPT_DIR}"
BASELINE_CKPT_PATH="${BASELINE_CKPT_DIR}/baseline_edm_s${SEED}_st${STEPS}_bs${BATCH_SIZE}_hd${HIDDEN_DIM}_tp${MNIST_TRAIN_PERCENT}.pt"
STRICT_FAIRNESS_CHECK="${STRICT_FAIRNESS_CHECK:-1}"

echo "=== Locked tuned config ==="
echo "SEED=${SEED}"
echo "STEPS=${STEPS} BATCH_SIZE=${BATCH_SIZE} HIDDEN_DIM=${HIDDEN_DIM}"
echo "SIGMA_MIN=${SIGMA_MIN} SIGMA_MAX=${SIGMA_MAX} AUTO_LOG_NORMAL=${AUTO_LOG_NORMAL}"
echo "USE_EMA_EVAL=${USE_EMA_EVAL} EMA_DECAY=${EMA_DECAY}"
echo "WARMUP_CLEAN_STEPS=${WARMUP_CLEAN_STEPS} WARMUP_RAMP_STEPS=${WARMUP_RAMP_STEPS}"
echo "MNIST_TRAIN_PERCENT=${MNIST_TRAIN_PERCENT} MNIST_VAL_PERCENT=${MNIST_VAL_PERCENT}"
echo "V11: step_size=${V11_STEP_SIZE} gamma=${V11_GAMMA} rho=${V11_RHO} mode=${V11_PROJ_MODE}"
echo "V21_RHO=${V21_RHO}"
echo "OUTDIR=${OUTDIR} PREFIX=${PREFIX}"
echo "BASELINE_CKPT_DIR=${BASELINE_CKPT_DIR}"
echo "BASELINE_CKPT_PATH=${BASELINE_CKPT_PATH}"
echo "STRICT_FAIRNESS_CHECK=${STRICT_FAIRNESS_CHECK}"

echo "=== Ensure MNIST FID reference ==="
python3 toy/export_mnist_fid_ref.py

COMMON_ARGS=(
  --dataset-kind mnist
  --model-kind image_conv
  --diagnostics-kind image_basic
  --image-channels 1
  --image-size 32
  --mnist-train-percent "${MNIST_TRAIN_PERCENT}"
  --mnist-val-percent "${MNIST_VAL_PERCENT}"
  --device cuda
  --training-objective edm
  --steps "${STEPS}"
  --batch-size "${BATCH_SIZE}"
  --hidden-dim "${HIDDEN_DIM}"
  --n-steps-path 24
  --sigma-min "${SIGMA_MIN}"
  --sigma-max "${SIGMA_MAX}"
  --eval-samples "${EVAL_SAMPLES}"
  --compute-fid
  --fid-samples "${FID_SAMPLES}"
  --skip-checks
  --disable-baseline-gate
  --seed "${SEED}"
  --outdir "${OUTDIR}"
  --warmup-clean-steps "${WARMUP_CLEAN_STEPS}"
  --warmup-ramp-steps "${WARMUP_RAMP_STEPS}"
  --baseline-ckpt-path "${BASELINE_CKPT_PATH}"
)

if [[ "${AUTO_LOG_NORMAL}" == "1" ]]; then
  COMMON_ARGS+=(--auto-log-normal-params)
fi

if [[ "${USE_EMA_EVAL}" == "1" ]]; then
  COMMON_ARGS+=(--use-ema-eval --ema-decay "${EMA_DECAY}")
fi

BASELINE_EXP="${PREFIX}_baseline_s${SEED}"
V11_EXP="${PREFIX}_1_1_s${SEED}"
V2_EXP="${PREFIX}_v2_s${SEED}"
V21_EXP="${PREFIX}_2_1_s${SEED}"

echo "=== 1/5 Baseline EDM (baseline-only) ==="
python3 toy/run_toy.py "${COMMON_ARGS[@]}" \
  --exp-name "${BASELINE_EXP}" \
  --method-version v2 \
  --baseline-only

echo "=== 2/5 v1.1 robust ==="
python3 toy/run_toy.py "${COMMON_ARGS[@]}" \
  --exp-name "${V11_EXP}" \
  --method-version 1.1 \
  --v11-step-size "${V11_STEP_SIZE}" \
  --v11-transport-gamma "${V11_GAMMA}" \
  --v11-total-budget-rho "${V11_RHO}" \
  --v11-projection-mode "${V11_PROJ_MODE}"

echo "=== 3/5 v2 robust ==="
python3 toy/run_toy.py "${COMMON_ARGS[@]}" \
  --exp-name "${V2_EXP}" \
  --method-version v2

echo "=== 4/5 v2.1 robust ==="
python3 toy/run_toy.py "${COMMON_ARGS[@]}" \
  --exp-name "${V21_EXP}" \
  --method-version 2.1 \
  --v21-rho "${V21_RHO}"

echo "=== 5/5 Aggregate metrics + build annotated panel ==="
python3 - <<'PY'
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path

outdir = Path(os.environ['OUTDIR'])
prefix = os.environ['PREFIX']
seed = os.environ['SEED']

baseline_exp = f"{prefix}_baseline_s{seed}"
v11_exp = f"{prefix}_1_1_s{seed}"
v2_exp = f"{prefix}_v2_s{seed}"
v21_exp = f"{prefix}_2_1_s{seed}"

def _safe_float(v):
    if v is None:
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")

def _fmt_fid(v):
    return "nan" if not math.isfinite(v) else f"{v:.2f}"

def _fmt_runtime(v):
    if not math.isfinite(v):
        return "n/a"
    if v < 60:
        return f"{v:.1f}s"
    if v < 3600:
        return f"{v/60.0:.1f}m"
    return f"{v/3600.0:.2f}h"

def load_row(exp: str):
    p = outdir / exp / 'metrics.json'
    m = json.loads(p.read_text())
    q = m['metrics'].get('sample_quality_debug', {})
    flow = m['metrics'].get('flow_debug', {})
    runtime = flow.get('runtime', {})
    cdbg = m['metrics'].get('constraint_debug', {})
    rec = m['metrics'].get('recovery_debug', {})
    fid_base_t = _safe_float(runtime.get('fid_baseline', 0.0))
    fid_rob_t = _safe_float(runtime.get('fid_robust', 0.0))
    if not math.isfinite(fid_base_t):
        fid_base_t = 0.0
    if not math.isfinite(fid_rob_t):
        fid_rob_t = 0.0
    return {
        'exp': exp,
        'baseline_fid': float(q.get('baseline_fid', 'nan')),
        'robust_fid': float(q.get('robust_fid', 'nan')),
        'attack_training_executed': bool(m['metrics'].get('baseline_gate', {}).get('attack_training_executed', False)),
        'baseline_ckpt_loaded': bool(flow.get('baseline_ckpt_loaded', False)),
        'baseline_ckpt_saved': bool(flow.get('baseline_ckpt_saved', False)),
        'baseline_ckpt_signature_hash': flow.get('baseline_ckpt_signature_hash'),
        'attack_win_ratio_overall': float(cdbg.get('attack_gap_windows_heldout', {}).get('overall', {}).get('win_ratio', 'nan')),
        'attack_gap_overall': float(cdbg.get('attack_gap_windows_heldout', {}).get('overall', {}).get('mean_attack_gap', 'nan')),
        'recovery_mse_ref_terminal': float(rec.get('baseline_x0_mse_from_ref_terminal', 'nan')),
        'recovery_mse_attack_terminal': float(rec.get('baseline_x0_mse_from_attack_terminal', 'nan')),
        'runtime_total_sec': _safe_float(runtime.get('total', flow.get('runtime_total_sec'))),
        'runtime_total_without_fid_sec': _safe_float(runtime.get('total_without_fid')),
        'runtime_baseline_phase_sec': _safe_float(runtime.get('baseline_phase')),
        'runtime_baseline_train_sec': _safe_float(runtime.get('baseline_train')),
        'runtime_robust_phase_sec': _safe_float(runtime.get('robust_phase')),
        'runtime_post_eval_sec': _safe_float(runtime.get('post_train_eval')),
        'runtime_fid_total_sec': float(fid_base_t + fid_rob_t),
    }

rows = {
    'baseline_only': load_row(baseline_exp),
    'v1_1': load_row(v11_exp),
    'v2': load_row(v2_exp),
    'v2_1': load_row(v21_exp),
}

baseline_fids = [
    rows['baseline_only']['baseline_fid'],
    rows['v1_1']['baseline_fid'],
    rows['v2']['baseline_fid'],
    rows['v2_1']['baseline_fid'],
]
finite_baseline_fids = [v for v in baseline_fids if math.isfinite(v)]
baseline_fid_spread_abs = (
    float(max(finite_baseline_fids) - min(finite_baseline_fids))
    if finite_baseline_fids
    else None
)

ckpt_hashes = {
    rows['baseline_only']['baseline_ckpt_signature_hash'],
    rows['v1_1']['baseline_ckpt_signature_hash'],
    rows['v2']['baseline_ckpt_signature_hash'],
    rows['v2_1']['baseline_ckpt_signature_hash'],
}

def _ranking(metric_key: str):
    items = []
    for method_name, row in rows.items():
        val = _safe_float(row.get(metric_key))
        if math.isfinite(val):
            items.append((method_name, float(val)))
    return [ {'method': k, 'seconds': v} for k, v in sorted(items, key=lambda kv: kv[1]) ]

runtime_table_path = outdir / f"{prefix}_runtime_table_baseline_v11_v2_v21_s{seed}.csv"
with runtime_table_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(
        [
            "method",
            "exp",
            "baseline_fid",
            "robust_fid",
            "runtime_total_sec",
            "runtime_total_without_fid_sec",
            "runtime_baseline_train_sec",
            "runtime_robust_phase_sec",
            "runtime_post_eval_sec",
            "runtime_fid_total_sec",
        ]
    )
    for method_name in ("baseline_only", "v1_1", "v2", "v2_1"):
        r = rows[method_name]
        writer.writerow(
            [
                method_name,
                r["exp"],
                r["baseline_fid"],
                r["robust_fid"],
                r["runtime_total_sec"],
                r["runtime_total_without_fid_sec"],
                r["runtime_baseline_train_sec"],
                r["runtime_robust_phase_sec"],
                r["runtime_post_eval_sec"],
                r["runtime_fid_total_sec"],
            ]
        )

def _panel_note(row):
    return (
        f"FID={_fmt_fid(_safe_float(row['robust_fid']))} | "
        f"total={_fmt_runtime(_safe_float(row['runtime_total_sec']))} | "
        f"robust={_fmt_runtime(_safe_float(row['runtime_robust_phase_sec']))}"
    )

panel_path = outdir / f"{prefix}_panel_real_baseline_v11_v2_v21_s{seed}.png"
panel_cmd = [
    sys.executable,
    "toy/scripts/make_v2_v21_panel.py",
    "--outdir",
    str(outdir),
    "--prefix",
    str(prefix),
    "--seed",
    str(seed),
    "--baseline-dir",
    str(outdir / baseline_exp / "fid_baseline"),
    "--v11-dir",
    str(outdir / v11_exp / "fid_robust"),
    "--v2-dir",
    str(outdir / v2_exp / "fid_robust"),
    "--v21-dir",
    str(outdir / v21_exp / "fid_robust"),
    "--rows",
    "8",
    "--cols",
    "8",
    "--tile-size",
    "32",
    "--tile-pad",
    "2",
    "--pick-seed",
    str(os.environ.get("PICK_SEED", "7")),
    "--real-note",
    f"N={8*8}",
    "--baseline-note",
    _panel_note(rows["baseline_only"]),
    "--v11-note",
    _panel_note(rows["v1_1"]),
    "--v2-note",
    _panel_note(rows["v2"]),
    "--v21-note",
    _panel_note(rows["v2_1"]),
    "--output",
    str(panel_path),
]
subprocess.run(panel_cmd, check=True)

summary = {
    'seed': int(seed),
    'steps': int(os.environ['STEPS']),
    'mnist_train_percent': float(os.environ['MNIST_TRAIN_PERCENT']),
    'mnist_val_percent': float(os.environ['MNIST_VAL_PERCENT']),
    'rows': rows,
    'runtime_ranking': {
        'by_total_sec': _ranking('runtime_total_sec'),
        'by_robust_phase_sec': _ranking('runtime_robust_phase_sec'),
        'by_total_without_fid_sec': _ranking('runtime_total_without_fid_sec'),
    },
    'fairness_checks': {
        'baseline_fid_spread_abs': baseline_fid_spread_abs,
        'baseline_ckpt_hash_unique_count': int(len(ckpt_hashes)),
        'baseline_ckpt_hashes': sorted(h for h in ckpt_hashes if h is not None),
    },
    'panel': str(panel_path),
    'runtime_table_csv': str(runtime_table_path),
}

summary_path = outdir / f"{prefix}_summary_baseline_v11_v2_v21_s{seed}.json"
summary_path.write_text(json.dumps(summary, indent=2))

print('[done] summary:', summary_path)
print('[done] panel:', panel_path)
print('[done] runtime_table:', runtime_table_path)
print('[fair] baseline_fid_spread_abs=', summary['fairness_checks']['baseline_fid_spread_abs'])
print('[fair] baseline_ckpt_hash_unique_count=', summary['fairness_checks']['baseline_ckpt_hash_unique_count'])
print('[runtime] fastest_by_total=', summary['runtime_ranking']['by_total_sec'][0] if summary['runtime_ranking']['by_total_sec'] else None)
print('[runtime] fastest_by_robust_phase=', summary['runtime_ranking']['by_robust_phase_sec'][0] if summary['runtime_ranking']['by_robust_phase_sec'] else None)

strict = os.environ.get('STRICT_FAIRNESS_CHECK', '1').strip() not in ('0', 'false', 'False')
if strict:
    robust_loaded = [
        bool(rows['v1_1']['baseline_ckpt_loaded']),
        bool(rows['v2']['baseline_ckpt_loaded']),
        bool(rows['v2_1']['baseline_ckpt_loaded']),
    ]
    if summary['fairness_checks']['baseline_ckpt_hash_unique_count'] != 1:
        raise SystemExit(
            f"[fairness-fail] expected exactly 1 baseline ckpt hash, "
            f"got {summary['fairness_checks']['baseline_ckpt_hash_unique_count']}"
        )
    if not all(robust_loaded):
        raise SystemExit(f"[fairness-fail] expected all robust runs to load baseline ckpt, got {robust_loaded}")
    print('[fair] strict checks passed')
PY

echo "[*] Finished baseline-v1.1-v2-v2.1 panel workflow"
