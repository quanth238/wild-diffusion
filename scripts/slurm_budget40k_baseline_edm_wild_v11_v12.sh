#!/bin/bash
#SBATCH --job-name=mnist_b40k_4m
#SBATCH --output=toy_outputs/slurm_b40k_4m_%j.out
#SBATCH --error=toy_outputs/slurm_b40k_4m_%j.err
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

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

# Environment bootstrap (supports both VinUni SLURM path and standalone SSH path).
if [[ -f /home/quanth/working_space/scripts/conda.sh ]]; then
  source /home/quanth/working_space/scripts/conda.sh
fi

if [[ -d /home/quanth/working_space/Wild-Diffusion ]]; then
  cd /home/quanth/working_space/Wild-Diffusion
elif [[ -d /root/wild-diffusion ]]; then
  cd /root/wild-diffusion
else
  echo "[ERROR] Could not find repository directory."
  exit 1
fi

if [[ -f .venv/bin/activate ]]; then
  # Prefer project-local virtualenv when available.
  source .venv/bin/activate
fi

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
TOTAL_STEPS="${TOTAL_STEPS:-40000}"
WARMUP_STEPS="${WARMUP_STEPS:-8000}"
ROBUST_STEPS="${ROBUST_STEPS:-32000}"
WARMUP4K_STEPS="${WARMUP4K_STEPS:-4000}"
ENSURE_WARMUP4K="${ENSURE_WARMUP4K:-1}"

BATCH_SIZE="${BATCH_SIZE:-32}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
EVAL_SAMPLES="${EVAL_SAMPLES:-2000}"
FID_SAMPLES="${FID_SAMPLES:-2000}"
MNIST_TRAIN_PERCENT="${MNIST_TRAIN_PERCENT:-20}"
MNIST_VAL_PERCENT="${MNIST_VAL_PERCENT:-100}"
OUTDIR="${OUTDIR:-/mnt/data/quanth/wild_diffusion_outputs/mnist_budget40k_baseline_edm_wild_v11_v12}"
PREFIX="${PREFIX:-mnist_${MNIST_TRAIN_PERCENT}pct_s${SEED}_tot${TOTAL_STEPS}_w${WARMUP_STEPS}_r${ROBUST_STEPS}}"
PICK_SEED="${PICK_SEED:-7}"

SIGMA_MIN="${SIGMA_MIN:-0.01}"
SIGMA_MAX="${SIGMA_MAX:-80}"
AUTO_LOG_NORMAL="${AUTO_LOG_NORMAL:-1}"
USE_EMA_EVAL="${USE_EMA_EVAL:-1}"
EMA_DECAY="${EMA_DECAY:-0.999}"

# In robust continuation phase, warmup is already done by the shared 8k checkpoint.
ROBUST_WARMUP_CLEAN_STEPS="${ROBUST_WARMUP_CLEAN_STEPS:-0}"
ROBUST_WARMUP_RAMP_STEPS="${ROBUST_WARMUP_RAMP_STEPS:-0}"

# v1.1
V11_STEP_SIZE="${V11_STEP_SIZE:-0.0005}"
V11_GAMMA="${V11_GAMMA:-2.0}"
V11_RHO="${V11_RHO:-0.02}"
V11_PROJ_MODE="${V11_PROJ_MODE:-global_remaining}"

# WILD
WILD_UPDATE_INTERVAL="${WILD_UPDATE_INTERVAL:-20}"
WILD_CACHE_BATCHES="${WILD_CACHE_BATCHES:-4}"
WILD_INNER_STEPS="${WILD_INNER_STEPS:-3}"
WILD_STEP_SIZE="${WILD_STEP_SIZE:-0.05}"
WILD_GAMMA="${WILD_GAMMA:-2.0}"
WILD_FIXED_NOISE_INNER="${WILD_FIXED_NOISE_INNER:-1}"
WILD_CLAMP_SAMPLES="${WILD_CLAMP_SAMPLES:-0}"
WILD_SAMPLE_MIN="${WILD_SAMPLE_MIN:--1.0}"
WILD_SAMPLE_MAX="${WILD_SAMPLE_MAX:-1.0}"
WILD_DELTA_RATIO_DENOM="${WILD_DELTA_RATIO_DENOM:-1.0}"

# v1.2 (cdroedm-aligned)
V12_ADV_STEPS="${V12_ADV_STEPS:-1}"
V12_STEP_SIZE="${V12_STEP_SIZE:-0.02}"
V12_LAMBDA_INIT="${V12_LAMBDA_INIT:-0.1}"
V12_LAMBDA_LR="${V12_LAMBDA_LR:-0.001}"
V12_RHO_TARGET="${V12_RHO_TARGET:-0.0001}"
V12_ROBUST_MIX="${V12_ROBUST_MIX:-0.3}"
V12_START_STEP="${V12_START_STEP:-0}"
V12_RAMP_STEPS="${V12_RAMP_STEPS:-0}"
V12_MAX_DELTA="${V12_MAX_DELTA:-0.05}"
V12_SIGMA_FLOOR="${V12_SIGMA_FLOOR:-0.0}"
V12_SIGMA_CUT="${V12_SIGMA_CUT:-0.5}"
V12_GATE_POWER="${V12_GATE_POWER:-2.0}"
V12_DELTA_SPACE="${V12_DELTA_SPACE:-image}"

STRICT_FAIRNESS_CHECK="${STRICT_FAIRNESS_CHECK:-1}"
BASELINE_FID_SPREAD_TOL="${BASELINE_FID_SPREAD_TOL:-1e-6}"
SKIP_EDM_CLEAN="${SKIP_EDM_CLEAN:-0}"
SKIP_WILD="${SKIP_WILD:-0}"
SKIP_V11="${SKIP_V11:-0}"
SKIP_V12="${SKIP_V12:-0}"
SKIP_AGG="${SKIP_AGG:-0}"
FORCE_REBUILD_WARMUP8K="${FORCE_REBUILD_WARMUP8K:-0}"
FORCE_REBUILD_WARMUP4K="${FORCE_REBUILD_WARMUP4K:-0}"

mkdir -p "${OUTDIR}"
BASELINE_CKPT_DIR="${BASELINE_CKPT_DIR:-/mnt/data/quanth/models/wild_diffusion_baseline_ckpt}"
mkdir -p "${BASELINE_CKPT_DIR}"

WARMUP8K_CKPT="${BASELINE_CKPT_DIR}/baseline_edm_s${SEED}_st${WARMUP_STEPS}_bs${BATCH_SIZE}_hd${HIDDEN_DIM}_tp${MNIST_TRAIN_PERCENT}.pt"
WARMUP4K_CKPT="${BASELINE_CKPT_DIR}/baseline_edm_s${SEED}_st${WARMUP4K_STEPS}_bs${BATCH_SIZE}_hd${HIDDEN_DIM}_tp${MNIST_TRAIN_PERCENT}.pt"
ALIAS32K_CKPT="${BASELINE_CKPT_DIR}/baseline_edm_s${SEED}_st${WARMUP_STEPS}_bs${BATCH_SIZE}_hd${HIDDEN_DIM}_tp${MNIST_TRAIN_PERCENT}_alias_st${ROBUST_STEPS}.pt"

export SEED
export TOTAL_STEPS
export WARMUP_STEPS
export ROBUST_STEPS
export MNIST_TRAIN_PERCENT
export MNIST_VAL_PERCENT
export OUTDIR
export PREFIX
export PICK_SEED
export STRICT_FAIRNESS_CHECK
export BASELINE_FID_SPREAD_TOL
export WARMUP8K_CKPT
export WARMUP4K_CKPT
export ALIAS32K_CKPT

python3 - <<'PY'
import os
T = int(os.environ['TOTAL_STEPS'])
W = int(os.environ['WARMUP_STEPS'])
R = int(os.environ['ROBUST_STEPS'])
if W + R != T:
    raise SystemExit(f"[ERROR] Require WARMUP_STEPS + ROBUST_STEPS == TOTAL_STEPS, got {W}+{R}!={T}")
print(f"[check] total budget OK: warmup={W}, robust={R}, total={T}")
PY

echo "=== Budget-locked fairness config (40k protocol) ==="
echo "SEED=${SEED}"
echo "TOTAL_STEPS=${TOTAL_STEPS} WARMUP_STEPS=${WARMUP_STEPS} ROBUST_STEPS=${ROBUST_STEPS}"
echo "BATCH_SIZE=${BATCH_SIZE} HIDDEN_DIM=${HIDDEN_DIM}"
echo "MNIST_TRAIN_PERCENT=${MNIST_TRAIN_PERCENT} MNIST_VAL_PERCENT=${MNIST_VAL_PERCENT}"
echo "SIGMA_MIN=${SIGMA_MIN} SIGMA_MAX=${SIGMA_MAX} AUTO_LOG_NORMAL=${AUTO_LOG_NORMAL}"
echo "USE_EMA_EVAL=${USE_EMA_EVAL} EMA_DECAY=${EMA_DECAY}"
echo "ROBUST_WARMUP_CLEAN_STEPS=${ROBUST_WARMUP_CLEAN_STEPS} ROBUST_WARMUP_RAMP_STEPS=${ROBUST_WARMUP_RAMP_STEPS}"
echo "OUTDIR=${OUTDIR} PREFIX=${PREFIX}"
echo "WARMUP8K_CKPT=${WARMUP8K_CKPT}"
echo "WARMUP4K_CKPT=${WARMUP4K_CKPT} (ensure=${ENSURE_WARMUP4K})"
echo "ALIAS32K_CKPT=${ALIAS32K_CKPT}"

echo "=== Ensure MNIST FID reference ==="
python3 toy/export_mnist_fid_ref.py

WARMUP_COMMON_ARGS=(
  --dataset-kind mnist
  --model-kind image_conv
  --diagnostics-kind image_basic
  --image-channels 1
  --image-size 32
  --mnist-train-percent "${MNIST_TRAIN_PERCENT}"
  --mnist-val-percent "${MNIST_VAL_PERCENT}"
  --device cuda
  --training-objective edm
  --batch-size "${BATCH_SIZE}"
  --hidden-dim "${HIDDEN_DIM}"
  --n-steps-path 24
  --sigma-min "${SIGMA_MIN}"
  --sigma-max "${SIGMA_MAX}"
  --eval-samples "${EVAL_SAMPLES}"
  --skip-checks
  --disable-baseline-gate
  --seed "${SEED}"
  --outdir "${OUTDIR}"
)
if [[ "${AUTO_LOG_NORMAL}" == "1" ]]; then
  WARMUP_COMMON_ARGS+=(--auto-log-normal-params)
fi
if [[ "${USE_EMA_EVAL}" == "1" ]]; then
  WARMUP_COMMON_ARGS+=(--use-ema-eval --ema-decay "${EMA_DECAY}")
fi

echo "=== 1/8 Build/verify shared warmup-${WARMUP_STEPS} checkpoint ==="
WARMUP8K_EXP="${PREFIX}_warmup${WARMUP_STEPS}_baseline_s${SEED}"
WARMUP8K_FLAGS=()
if [[ "${FORCE_REBUILD_WARMUP8K}" == "1" ]]; then
  WARMUP8K_FLAGS+=(--baseline-ckpt-force-retrain)
fi
python3 toy/run_toy.py "${WARMUP_COMMON_ARGS[@]}" \
  --steps "${WARMUP_STEPS}" \
  --exp-name "${WARMUP8K_EXP}" \
  --method-version v2 \
  --baseline-only \
  --baseline-ckpt-path "${WARMUP8K_CKPT}" \
  "${WARMUP8K_FLAGS[@]}"

if [[ "${ENSURE_WARMUP4K}" == "1" ]]; then
  echo "=== 2/8 Build/verify auxiliary warmup-${WARMUP4K_STEPS} checkpoint ==="
  WARMUP4K_EXP="${PREFIX}_warmup${WARMUP4K_STEPS}_baseline_s${SEED}"
  WARMUP4K_FLAGS=()
  if [[ "${FORCE_REBUILD_WARMUP4K}" == "1" ]]; then
    WARMUP4K_FLAGS+=(--baseline-ckpt-force-retrain)
  fi
  python3 toy/run_toy.py "${WARMUP_COMMON_ARGS[@]}" \
    --steps "${WARMUP4K_STEPS}" \
    --exp-name "${WARMUP4K_EXP}" \
    --method-version v2 \
    --baseline-only \
    --baseline-ckpt-path "${WARMUP4K_CKPT}" \
    "${WARMUP4K_FLAGS[@]}"
else
  echo "=== 2/8 Auxiliary warmup-${WARMUP4K_STEPS} checkpoint [SKIPPED] ==="
fi

echo "=== 3/8 Create alias checkpoint for baseline_steps=${ROBUST_STEPS} from warmup-${WARMUP_STEPS} weights ==="
python3 - <<'PY'
import os
from datetime import datetime, timezone
import torch

src = os.environ['WARMUP8K_CKPT']
dst = os.environ['ALIAS32K_CKPT']
step = int(os.environ['ROBUST_STEPS'])
os.makedirs(os.path.dirname(dst), exist_ok=True)
if not os.path.isfile(src):
    raise SystemExit(f"[ERROR] warmup checkpoint not found: {src}")
payload = torch.load(src, map_location='cpu')
if not isinstance(payload, dict):
    raise SystemExit(f"[ERROR] invalid checkpoint payload: {src}")
if 'baseline_state_dict' not in payload:
    raise SystemExit(f"[ERROR] missing baseline_state_dict: {src}")

sig = payload.get('baseline_signature', {})
if not isinstance(sig, dict):
    sig = {}
sig = dict(sig)
sig['baseline_steps'] = int(step)

payload = dict(payload)
payload['baseline_signature'] = sig
payload['alias_from_ckpt'] = str(src)
payload['alias_for_baseline_steps'] = int(step)
payload['alias_saved_at'] = datetime.now(timezone.utc).isoformat()

tmp = dst + f".tmp.{os.getpid()}"
torch.save(payload, tmp)
os.replace(tmp, dst)
print(f"[done] alias checkpoint: {dst}")
PY

ROBUST_COMMON_ARGS=(
  --dataset-kind mnist
  --model-kind image_conv
  --diagnostics-kind image_basic
  --image-channels 1
  --image-size 32
  --mnist-train-percent "${MNIST_TRAIN_PERCENT}"
  --mnist-val-percent "${MNIST_VAL_PERCENT}"
  --device cuda
  --training-objective edm
  --steps "${ROBUST_STEPS}"
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
  --warmup-clean-steps "${ROBUST_WARMUP_CLEAN_STEPS}"
  --warmup-ramp-steps "${ROBUST_WARMUP_RAMP_STEPS}"
  --baseline-ckpt-path "${ALIAS32K_CKPT}"
)
if [[ "${AUTO_LOG_NORMAL}" == "1" ]]; then
  ROBUST_COMMON_ARGS+=(--auto-log-normal-params)
fi
if [[ "${USE_EMA_EVAL}" == "1" ]]; then
  ROBUST_COMMON_ARGS+=(--use-ema-eval --ema-decay "${EMA_DECAY}")
fi

WILD_ARGS=(
  --wild-update-interval "${WILD_UPDATE_INTERVAL}"
  --wild-cache-batches "${WILD_CACHE_BATCHES}"
  --wild-inner-steps "${WILD_INNER_STEPS}"
  --wild-step-size "${WILD_STEP_SIZE}"
  --wild-gamma "${WILD_GAMMA}"
  --wild-sample-min "${WILD_SAMPLE_MIN}"
  --wild-sample-max "${WILD_SAMPLE_MAX}"
  --wild-delta-ratio-denom "${WILD_DELTA_RATIO_DENOM}"
)
if [[ "${WILD_FIXED_NOISE_INNER}" == "1" ]]; then
  WILD_ARGS+=(--wild-fixed-noise-inner)
else
  WILD_ARGS+=(--disable-wild-fixed-noise-inner)
fi
if [[ "${WILD_CLAMP_SAMPLES}" == "1" ]]; then
  WILD_ARGS+=(--wild-clamp-samples)
fi

V12_ARGS=(
  --inner-steps "${V12_ADV_STEPS}"
  --v12-step-size "${V12_STEP_SIZE}"
  --v12-lambda-init "${V12_LAMBDA_INIT}"
  --v12-lambda-lr "${V12_LAMBDA_LR}"
  --v12-rho-target "${V12_RHO_TARGET}"
  --v12-robust-mix "${V12_ROBUST_MIX}"
  --v12-start-step "${V12_START_STEP}"
  --v12-ramp-steps "${V12_RAMP_STEPS}"
  --v12-max-delta "${V12_MAX_DELTA}"
  --v12-sigma-floor "${V12_SIGMA_FLOOR}"
  --v12-sigma-cut "${V12_SIGMA_CUT}"
  --v12-gate-power "${V12_GATE_POWER}"
  --v12-delta-space "${V12_DELTA_SPACE}"
)

EDM_EXP="${PREFIX}_edm_clean_s${SEED}"
WILD_EXP="${PREFIX}_wild_s${SEED}"
V11_EXP="${PREFIX}_1_1_s${SEED}"
V12_EXP="${PREFIX}_1_2_s${SEED}"

if [[ "${SKIP_EDM_CLEAN}" == "1" ]]; then
  echo "=== 4/8 EDM clean continuation [SKIPPED] ==="
else
  echo "=== 4/8 EDM clean continuation: warmup-${WARMUP_STEPS} + ${ROBUST_STEPS} ==="
  python3 toy/run_toy.py "${ROBUST_COMMON_ARGS[@]}" \
    --exp-name "${EDM_EXP}" \
    --method-version v2 \
    --inner-steps 0 \
    --outer-attack-weight 0.0 \
    --outer-clean-weight 1.0
fi

if [[ "${SKIP_WILD}" == "1" ]]; then
  echo "=== 5/8 WILD robust [SKIPPED] ==="
else
  echo "=== 5/8 WILD robust ==="
  python3 toy/run_toy.py "${ROBUST_COMMON_ARGS[@]}" \
    --exp-name "${WILD_EXP}" \
    --method-version wild \
    "${WILD_ARGS[@]}"
fi

if [[ "${SKIP_V11}" == "1" ]]; then
  echo "=== 6/8 v1.1 robust [SKIPPED] ==="
else
  echo "=== 6/8 v1.1 robust ==="
  python3 toy/run_toy.py "${ROBUST_COMMON_ARGS[@]}" \
    --exp-name "${V11_EXP}" \
    --method-version 1.1 \
    --v11-step-size "${V11_STEP_SIZE}" \
    --v11-transport-gamma "${V11_GAMMA}" \
    --v11-total-budget-rho "${V11_RHO}" \
    --v11-projection-mode "${V11_PROJ_MODE}"
fi

if [[ "${SKIP_V12}" == "1" ]]; then
  echo "=== 7/8 v1.2 robust [SKIPPED] ==="
else
  echo "=== 7/8 v1.2 robust ==="
  python3 toy/run_toy.py "${ROBUST_COMMON_ARGS[@]}" \
    --exp-name "${V12_EXP}" \
    --method-version 1.2 \
    "${V12_ARGS[@]}"
fi

if [[ "${SKIP_AGG}" == "1" ]]; then
  echo "=== 8/8 Aggregate metrics + panel [SKIPPED] ==="
else
  echo "=== 8/8 Aggregate metrics + panel ==="
  python3 - <<'PY'
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

outdir = Path(os.environ['OUTDIR'])
prefix = os.environ['PREFIX']
seed = os.environ['SEED']
strict = os.environ.get('STRICT_FAIRNESS_CHECK', '1').strip() not in ('0', 'false', 'False')
fid_spread_tol = float(os.environ.get('BASELINE_FID_SPREAD_TOL', '1e-6'))

edm_exp = f"{prefix}_edm_clean_s{seed}"
wild_exp = f"{prefix}_wild_s{seed}"
v11_exp = f"{prefix}_1_1_s{seed}"
v12_exp = f"{prefix}_1_2_s{seed}"

LOCKED_KEYS = [
    'dataset_kind',
    'model_kind',
    'diagnostics_kind',
    'image_size',
    'image_channels',
    'limited_data_enabled',
    'mnist_use_percent_split',
    'mnist_train_percent',
    'mnist_val_percent',
    'steps',
    'batch_size',
    'hidden_dim',
    'training_objective',
    'n_steps_path',
    'sigma_min',
    'sigma_max',
    'use_log_normal_sigma_sampling',
    'p_mean',
    'p_std',
    'seed',
    'image_split_seed',
    'warmup_clean_steps',
    'warmup_ramp_steps',
]

def _safe_float(v):
    if v is None:
        return float('nan')
    try:
        return float(v)
    except (TypeError, ValueError):
        return float('nan')

def _fmt_fid(v):
    return 'nan' if not math.isfinite(v) else f"{v:.2f}"

def _fmt_runtime(v):
    if not math.isfinite(v):
        return 'n/a'
    if v < 60:
        return f"{v:.1f}s"
    if v < 3600:
        return f"{v/60.0:.1f}m"
    return f"{v/3600.0:.2f}h"

def _cfg_subset(cfg):
    return {k: cfg.get(k) for k in LOCKED_KEYS}

def _json_hash(obj):
    blob = json.dumps(obj, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha1(blob).hexdigest()

def load_row(exp: str):
    p = outdir / exp / 'metrics.json'
    payload = json.loads(p.read_text())
    metrics = payload.get('metrics', {})
    cfg = payload.get('config', {})
    flow = metrics.get('flow_debug', {})
    runtime = flow.get('runtime', {})
    q = metrics.get('sample_quality_debug', {})
    cdbg = metrics.get('constraint_debug', {})
    rec = metrics.get('recovery_debug', {})
    gate = metrics.get('baseline_gate', {})
    obj = metrics.get('objective_debug', {})

    fid_base_t = _safe_float(runtime.get('fid_baseline', 0.0))
    fid_rob_t = _safe_float(runtime.get('fid_robust', 0.0))
    if not math.isfinite(fid_base_t):
        fid_base_t = 0.0
    if not math.isfinite(fid_rob_t):
        fid_rob_t = 0.0

    return {
        'exp': exp,
        'method_version': flow.get('method_version'),
        'baseline_fid': _safe_float(q.get('baseline_fid')),
        'robust_fid': _safe_float(q.get('robust_fid')),
        'fid_delta': _safe_float(q.get('robust_fid')) - _safe_float(q.get('baseline_fid')),
        'attack_training_executed': bool(gate.get('attack_training_executed', False)),
        'baseline_ckpt_loaded': bool(flow.get('baseline_ckpt_loaded', False)),
        'baseline_ckpt_saved': bool(flow.get('baseline_ckpt_saved', False)),
        'baseline_ckpt_signature_hash': flow.get('baseline_ckpt_signature_hash'),
        'baseline_ckpt_path': flow.get('baseline_ckpt_path'),
        'attack_win_ratio_overall': _safe_float(cdbg.get('attack_gap_windows_heldout', {}).get('overall', {}).get('win_ratio')),
        'attack_gap_overall': _safe_float(cdbg.get('attack_gap_windows_heldout', {}).get('overall', {}).get('mean_attack_gap')),
        'recovery_mse_ref_terminal': _safe_float(rec.get('baseline_x0_mse_from_ref_terminal')),
        'recovery_mse_attack_terminal': _safe_float(rec.get('baseline_x0_mse_from_attack_terminal')),
        'runtime_total_sec': _safe_float(runtime.get('total', flow.get('runtime_total_sec'))),
        'runtime_total_without_fid_sec': _safe_float(runtime.get('total_without_fid')),
        'runtime_baseline_phase_sec': _safe_float(runtime.get('baseline_phase')),
        'runtime_baseline_train_sec': _safe_float(runtime.get('baseline_train')),
        'runtime_robust_phase_sec': _safe_float(runtime.get('robust_phase')),
        'runtime_post_eval_sec': _safe_float(runtime.get('post_train_eval')),
        'runtime_fid_total_sec': float(fid_base_t + fid_rob_t),
        'runtime_steps_per_sec_robust': _safe_float(runtime.get('robust_steps_per_sec')),
        'dataset_policy': metrics.get('dataset_debug', {}).get('train_selection_policy'),
        'train_subset_size_resolved': metrics.get('dataset_debug', {}).get('train_subset_size_resolved'),
        'val_subset_size_resolved': metrics.get('dataset_debug', {}).get('val_subset_size_resolved'),
        'cfg_locked_subset': _cfg_subset(cfg),
        'eval_seed_gate': flow.get('eval_seed_gate'),
        'eval_seed_metrics': flow.get('eval_seed_metrics'),
        'eval_shared_terminal_noise': flow.get('eval_shared_terminal_noise'),
        'eval_shared_reverse_noise': flow.get('eval_shared_reverse_noise'),
        'sched_attack_weight_mean_last': _safe_float(obj.get('robust_sched_attack_weight', {}).get('mean_last')),
        'sched_clean_weight_mean_last': _safe_float(obj.get('robust_sched_clean_weight', {}).get('mean_last')),
    }

rows = {
    'edm_clean': load_row(edm_exp),
    'wild': load_row(wild_exp),
    'v1_1': load_row(v11_exp),
    'v1_2': load_row(v12_exp),
}

baseline_fids = [rows[k]['baseline_fid'] for k in ('edm_clean', 'wild', 'v1_1', 'v1_2')]
finite_baseline_fids = [v for v in baseline_fids if math.isfinite(v)]
baseline_fid_spread_abs = (
    float(max(finite_baseline_fids) - min(finite_baseline_fids))
    if finite_baseline_fids
    else None
)

ckpt_hashes = {rows[k]['baseline_ckpt_signature_hash'] for k in rows.keys()}
policy_hashes = {_json_hash(rows[k]['dataset_policy']) for k in rows.keys()}
cfg_hashes = {_json_hash(rows[k]['cfg_locked_subset']) for k in rows.keys()}
train_sizes = [rows[k]['train_subset_size_resolved'] for k in rows.keys()]
val_sizes = [rows[k]['val_subset_size_resolved'] for k in rows.keys()]
eval_proto = {
    _json_hash(
        {
            'gate': rows[k]['eval_seed_gate'],
            'metrics': rows[k]['eval_seed_metrics'],
            'shared_terminal': rows[k]['eval_shared_terminal_noise'],
            'shared_reverse': rows[k]['eval_shared_reverse_noise'],
        }
    )
    for k in rows.keys()
}

def _ranking(metric_key: str):
    items = []
    for method_name, row in rows.items():
        val = _safe_float(row.get(metric_key))
        if math.isfinite(val):
            items.append((method_name, float(val)))
    return [{'method': k, 'seconds': v} for k, v in sorted(items, key=lambda kv: kv[1])]

runtime_table_path = outdir / f"{prefix}_runtime_table_edm_wild_v11_v12_s{seed}.csv"
with runtime_table_path.open('w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow([
        'method', 'exp', 'baseline_fid', 'robust_fid', 'fid_delta',
        'attack_training_executed', 'sched_attack_weight_mean_last',
        'runtime_total_sec', 'runtime_total_without_fid_sec', 'runtime_baseline_train_sec',
        'runtime_robust_phase_sec', 'runtime_post_eval_sec', 'runtime_fid_total_sec',
        'runtime_steps_per_sec_robust',
    ])
    for method_name in ('edm_clean', 'wild', 'v1_1', 'v1_2'):
        r = rows[method_name]
        writer.writerow([
            method_name,
            r['exp'],
            r['baseline_fid'],
            r['robust_fid'],
            r['fid_delta'],
            int(r['attack_training_executed']),
            r['sched_attack_weight_mean_last'],
            r['runtime_total_sec'],
            r['runtime_total_without_fid_sec'],
            r['runtime_baseline_train_sec'],
            r['runtime_robust_phase_sec'],
            r['runtime_post_eval_sec'],
            r['runtime_fid_total_sec'],
            r['runtime_steps_per_sec_robust'],
        ])

def _panel_note(row):
    return (
        f"FID={_fmt_fid(_safe_float(row['robust_fid']))} | "
        f"total={_fmt_runtime(_safe_float(row['runtime_total_sec']))} | "
        f"robust={_fmt_runtime(_safe_float(row['runtime_robust_phase_sec']))}"
    )

panel_path = outdir / f"{prefix}_panel_real_edm_wild_v11_v12_s{seed}.png"
panel_cmd = [
    sys.executable,
    'toy/scripts/make_v2_v21_panel.py',
    '--outdir', str(outdir),
    '--prefix', str(prefix),
    '--seed', str(seed),
    '--baseline-dir', str(outdir / edm_exp / 'fid_robust'),
    '--v11-dir', str(outdir / v11_exp / 'fid_robust'),
    '--v2-dir', str(outdir / wild_exp / 'fid_robust'),
    '--v21-dir', str(outdir / v12_exp / 'fid_robust'),
    '--rows', '8',
    '--cols', '8',
    '--tile-size', '32',
    '--tile-pad', '2',
    '--pick-seed', str(os.environ.get('PICK_SEED', '7')),
    '--real-title', 'Real',
    '--baseline-title', 'EDM Clean (8k+32k)',
    '--v11-title', 'v1.1 Robust (8k+32k)',
    '--v2-title', 'WILD Robust (8k+32k)',
    '--v21-title', 'v1.2 Robust (8k+32k)',
    '--real-note', f"N={8*8}",
    '--baseline-note', _panel_note(rows['edm_clean']),
    '--v11-note', _panel_note(rows['v1_1']),
    '--v2-note', _panel_note(rows['wild']),
    '--v21-note', _panel_note(rows['v1_2']),
    '--output', str(panel_path),
]
subprocess.run(panel_cmd, check=True)

fairness_checks = {
    'budget_protocol': {
        'total_steps': int(os.environ['TOTAL_STEPS']),
        'warmup_steps': int(os.environ['WARMUP_STEPS']),
        'robust_steps': int(os.environ['ROBUST_STEPS']),
    },
    'baseline_fid_spread_abs': baseline_fid_spread_abs,
    'baseline_fid_spread_tol': fid_spread_tol,
    'baseline_ckpt_hash_unique_count': int(len(ckpt_hashes)),
    'baseline_ckpt_hashes': sorted(h for h in ckpt_hashes if h is not None),
    'all_runs_loaded_baseline_ckpt': {k: bool(rows[k]['baseline_ckpt_loaded']) for k in rows.keys()},
    'attack_training_executed': {k: bool(rows[k]['attack_training_executed']) for k in rows.keys()},
    'sched_attack_weight_mean_last': {k: rows[k]['sched_attack_weight_mean_last'] for k in rows.keys()},
    'dataset_policy_hash_unique_count': int(len(policy_hashes)),
    'locked_config_hash_unique_count': int(len(cfg_hashes)),
    'eval_protocol_hash_unique_count': int(len(eval_proto)),
    'train_subset_size_spread_abs': float(max(train_sizes) - min(train_sizes)),
    'val_subset_size_spread_abs': float(max(val_sizes) - min(val_sizes)),
}

summary = {
    'seed': int(seed),
    'total_steps': int(os.environ['TOTAL_STEPS']),
    'warmup_steps': int(os.environ['WARMUP_STEPS']),
    'robust_steps': int(os.environ['ROBUST_STEPS']),
    'mnist_train_percent': float(os.environ['MNIST_TRAIN_PERCENT']),
    'mnist_val_percent': float(os.environ['MNIST_VAL_PERCENT']),
    'rows': rows,
    'runtime_ranking': {
        'by_total_sec': _ranking('runtime_total_sec'),
        'by_robust_phase_sec': _ranking('runtime_robust_phase_sec'),
        'by_total_without_fid_sec': _ranking('runtime_total_without_fid_sec'),
    },
    'fairness_checks': fairness_checks,
    'artifacts': {
        'panel': str(panel_path),
        'runtime_table_csv': str(runtime_table_path),
    },
    'locked_config_keys': LOCKED_KEYS,
}

summary_path = outdir / f"{prefix}_summary_edm_wild_v11_v12_s{seed}.json"
summary_path.write_text(json.dumps(summary, indent=2))

print('[done] summary:', summary_path)
print('[done] panel:', panel_path)
print('[done] runtime_table:', runtime_table_path)
print('[fair] baseline_fid_spread_abs=', fairness_checks['baseline_fid_spread_abs'])
print('[fair] baseline_ckpt_hash_unique_count=', fairness_checks['baseline_ckpt_hash_unique_count'])
print('[fair] dataset_policy_hash_unique_count=', fairness_checks['dataset_policy_hash_unique_count'])
print('[fair] locked_config_hash_unique_count=', fairness_checks['locked_config_hash_unique_count'])
print('[runtime] fastest_by_total=', summary['runtime_ranking']['by_total_sec'][0] if summary['runtime_ranking']['by_total_sec'] else None)
print('[runtime] fastest_by_robust_phase=', summary['runtime_ranking']['by_robust_phase_sec'][0] if summary['runtime_ranking']['by_robust_phase_sec'] else None)

if strict:
    if fairness_checks['baseline_ckpt_hash_unique_count'] != 1:
        raise SystemExit(
            f"[fairness-fail] expected exactly 1 baseline ckpt hash, got {fairness_checks['baseline_ckpt_hash_unique_count']}"
        )
    if not all(fairness_checks['all_runs_loaded_baseline_ckpt'].values()):
        raise SystemExit(
            f"[fairness-fail] some runs did not load shared baseline checkpoint: {fairness_checks['all_runs_loaded_baseline_ckpt']}"
        )
    if fairness_checks['dataset_policy_hash_unique_count'] != 1:
        raise SystemExit(
            f"[fairness-fail] dataset policy mismatch across methods, hash_count={fairness_checks['dataset_policy_hash_unique_count']}"
        )
    if fairness_checks['locked_config_hash_unique_count'] != 1:
        raise SystemExit(
            f"[fairness-fail] locked config mismatch across methods, hash_count={fairness_checks['locked_config_hash_unique_count']}"
        )
    if fairness_checks['eval_protocol_hash_unique_count'] != 1:
        raise SystemExit(
            f"[fairness-fail] eval protocol mismatch across methods, hash_count={fairness_checks['eval_protocol_hash_unique_count']}"
        )
    if fairness_checks['train_subset_size_spread_abs'] != 0.0 or fairness_checks['val_subset_size_spread_abs'] != 0.0:
        raise SystemExit(
            '[fairness-fail] train/val subset size mismatch across methods: '
            f"train_spread={fairness_checks['train_subset_size_spread_abs']} "
            f"val_spread={fairness_checks['val_subset_size_spread_abs']}"
        )
    spread = fairness_checks['baseline_fid_spread_abs']
    if spread is None or spread > fid_spread_tol:
        raise SystemExit(
            '[fairness-fail] baseline_fid mismatch across methods: '
            f"spread={spread}, tol={fid_spread_tol}"
        )
    for m in ('wild', 'v1_1', 'v1_2'):
        if not fairness_checks['attack_training_executed'].get(m, False):
            raise SystemExit(f"[fairness-fail] {m} did not execute robust training")
    print('[fair] strict checks passed')
PY
fi

echo "[*] Finished 40k fairness workflow (EDM clean + WILD + v1.1 + v1.2)"
