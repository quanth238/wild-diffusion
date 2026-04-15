#!/bin/bash
set -euo pipefail

# User-locked protocol:
# - Baseline EDM: full 40k (single reported baseline row).
# - Warmup 8k is auxiliary only (for robust initialization), NOT reported as baseline.
# - WILD: start from warmup-8k checkpoint, then run +32k.
# - v1.1: start from warmup-8k checkpoint, then run +865 (= round(32000/37)).
# - v1.2: start from warmup-8k checkpoint, then run +32k.

# Bootstrap env (supports VinUni path and standalone SSH).
if [[ -f /home/quanth/working_space/scripts/conda.sh ]]; then
  source /home/quanth/working_space/scripts/conda.sh
fi
if [[ -d /home/quanth/working_space/Wild-Diffusion ]]; then
  cd /home/quanth/working_space/Wild-Diffusion
elif [[ -d /root/wild-diffusion ]]; then
  cd /root/wild-diffusion
else
  echo "[ERROR] repo dir not found"
  exit 1
fi
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "[ERROR] nvidia-smi not found"
  exit 1
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "[info] Using scheduler/device override CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
else
  echo "[info] CUDA_VISIBLE_DEVICES is unset; relying on the runtime default device mapping."
fi
export FID_DETECTOR_PATH="${FID_DETECTOR_PATH:-/mnt/data/quanth/models/inception-2015-12-05.pkl}"

python3 - <<'PY'
import torch
print('[torch] cuda_available:', torch.cuda.is_available())
print('[torch] device_count:', torch.cuda.device_count())
if not torch.cuda.is_available() or torch.cuda.device_count() <= 0:
    raise SystemExit('[ERROR] CUDA is required but not visible.')
print('[torch] current_device:', torch.cuda.current_device())
print('[torch] device_name:', torch.cuda.get_device_name(torch.cuda.current_device()))
PY

SEED="${SEED:-0}"
TOTAL_STEPS="${TOTAL_STEPS:-40000}"
WARMUP_STEPS="${WARMUP_STEPS:-8000}"
ROBUST_STEPS="${ROBUST_STEPS:-32000}"
V11_EQ_STEPS="${V11_EQ_STEPS:-865}"

BATCH_SIZE="${BATCH_SIZE:-32}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
EVAL_SAMPLES="${EVAL_SAMPLES:-2000}"
FID_SAMPLES="${FID_SAMPLES:-2000}"
MNIST_TRAIN_PERCENT="${MNIST_TRAIN_PERCENT:-20}"
MNIST_VAL_PERCENT="${MNIST_VAL_PERCENT:-100}"
SIGMA_MIN="${SIGMA_MIN:-0.01}"
SIGMA_MAX="${SIGMA_MAX:-80}"
AUTO_LOG_NORMAL="${AUTO_LOG_NORMAL:-1}"
USE_EMA_EVAL="${USE_EMA_EVAL:-1}"
EMA_DECAY="${EMA_DECAY:-0.999}"

OUTDIR="${OUTDIR:-/mnt/data/quanth/wild_diffusion_outputs/mnist_budget40k_user_protocol}"
PREFIX="${PREFIX:-mnist_20pct_s${SEED}_baseline40k_wild32k_v11eq${V11_EQ_STEPS}_v12_32k}"
PICK_SEED="${PICK_SEED:-7}"

# Keep robust continuation without extra warmup inside robust trainer.
ROBUST_WARMUP_CLEAN_STEPS="${ROBUST_WARMUP_CLEAN_STEPS:-0}"
ROBUST_WARMUP_RAMP_STEPS="${ROBUST_WARMUP_RAMP_STEPS:-0}"

# v1.1 hyperparams
V11_STEP_SIZE="${V11_STEP_SIZE:-0.0005}"
V11_GAMMA="${V11_GAMMA:-2.0}"
V11_RHO="${V11_RHO:-0.02}"
V11_PROJ_MODE="${V11_PROJ_MODE:-global_remaining}"

# WILD hyperparams
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

# v1.2 hyperparams
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

STRICT_CHECKS="${STRICT_CHECKS:-1}"
FORCE_REBUILD_BASELINE40K="${FORCE_REBUILD_BASELINE40K:-0}"
FORCE_REBUILD_WARMUP8K="${FORCE_REBUILD_WARMUP8K:-0}"
SKIP_BASELINE40K="${SKIP_BASELINE40K:-0}"
SKIP_WARMUP8K="${SKIP_WARMUP8K:-0}"
SKIP_WILD="${SKIP_WILD:-0}"
SKIP_V11="${SKIP_V11:-0}"
SKIP_V12="${SKIP_V12:-0}"
SKIP_AGG="${SKIP_AGG:-0}"

mkdir -p "$OUTDIR"
BASELINE_CKPT_DIR="${BASELINE_CKPT_DIR:-/mnt/data/quanth/models/wild_diffusion_baseline_ckpt}"
mkdir -p "$BASELINE_CKPT_DIR"

BASELINE40K_CKPT="$BASELINE_CKPT_DIR/baseline_edm_s${SEED}_st${TOTAL_STEPS}_bs${BATCH_SIZE}_hd${HIDDEN_DIM}_tp${MNIST_TRAIN_PERCENT}.pt"
WARMUP8K_CKPT="$BASELINE_CKPT_DIR/warmup_edm_s${SEED}_st${WARMUP_STEPS}_bs${BATCH_SIZE}_hd${HIDDEN_DIM}_tp${MNIST_TRAIN_PERCENT}.pt"
LEGACY_WARMUP8K_CKPT="$BASELINE_CKPT_DIR/baseline_edm_s${SEED}_st${WARMUP_STEPS}_bs${BATCH_SIZE}_hd${HIDDEN_DIM}_tp${MNIST_TRAIN_PERCENT}.pt"
ALIAS32K_CKPT="$BASELINE_CKPT_DIR/warmup_edm_s${SEED}_st${WARMUP_STEPS}_bs${BATCH_SIZE}_hd${HIDDEN_DIM}_tp${MNIST_TRAIN_PERCENT}_alias_st${ROBUST_STEPS}.pt"
ALIAS_V11_CKPT="$BASELINE_CKPT_DIR/warmup_edm_s${SEED}_st${WARMUP_STEPS}_bs${BATCH_SIZE}_hd${HIDDEN_DIM}_tp${MNIST_TRAIN_PERCENT}_alias_st${V11_EQ_STEPS}.pt"

if [[ ! -f "$WARMUP8K_CKPT" && -f "$LEGACY_WARMUP8K_CKPT" ]]; then
  echo "[info] warmup checkpoint fallback: $LEGACY_WARMUP8K_CKPT"
  WARMUP8K_CKPT="$LEGACY_WARMUP8K_CKPT"
fi

export SEED TOTAL_STEPS WARMUP_STEPS ROBUST_STEPS V11_EQ_STEPS
export MNIST_TRAIN_PERCENT MNIST_VAL_PERCENT OUTDIR PREFIX PICK_SEED STRICT_CHECKS
export BASELINE40K_CKPT WARMUP8K_CKPT ALIAS32K_CKPT ALIAS_V11_CKPT

echo "=== User protocol config ==="
echo "Baseline EDM full steps: $TOTAL_STEPS"
echo "Warmup checkpoint for robust methods: $WARMUP_STEPS"
echo "WILD robust steps: $ROBUST_STEPS"
echo "v1.1 eq-compute robust steps: $V11_EQ_STEPS"
echo "v1.2 robust steps: $ROBUST_STEPS"
echo "OUTDIR=$OUTDIR"
echo "PREFIX=$PREFIX"
echo "SKIP_BASELINE40K=$SKIP_BASELINE40K SKIP_WARMUP8K=$SKIP_WARMUP8K SKIP_WILD=$SKIP_WILD SKIP_V11=$SKIP_V11 SKIP_V12=$SKIP_V12 SKIP_AGG=$SKIP_AGG"

echo "=== Ensure MNIST FID reference ==="
python3 toy/export_mnist_fid_ref.py

COMMON_FID_ARGS=(
  --dataset-kind mnist
  --model-kind image_conv
  --diagnostics-kind image_basic
  --image-channels 1
  --image-size 32
  --mnist-train-percent "$MNIST_TRAIN_PERCENT"
  --mnist-val-percent "$MNIST_VAL_PERCENT"
  --device cuda
  --training-objective edm
  --batch-size "$BATCH_SIZE"
  --hidden-dim "$HIDDEN_DIM"
  --n-steps-path 24
  --sigma-min "$SIGMA_MIN"
  --sigma-max "$SIGMA_MAX"
  --eval-samples "$EVAL_SAMPLES"
  --skip-checks
  --disable-baseline-gate
  --seed "$SEED"
  --outdir "$OUTDIR"
)
if [[ "$AUTO_LOG_NORMAL" == "1" ]]; then
  COMMON_FID_ARGS+=(--auto-log-normal-params)
fi
if [[ "$USE_EMA_EVAL" == "1" ]]; then
  COMMON_FID_ARGS+=(--use-ema-eval --ema-decay "$EMA_DECAY")
fi

COMMON_FID_ARGS+=(--compute-fid --fid-samples "$FID_SAMPLES")

WARMUP_ONLY_ARGS=(
  --dataset-kind mnist
  --model-kind image_conv
  --diagnostics-kind image_basic
  --image-channels 1
  --image-size 32
  --mnist-train-percent "$MNIST_TRAIN_PERCENT"
  --mnist-val-percent "$MNIST_VAL_PERCENT"
  --device cuda
  --training-objective edm
  --batch-size "$BATCH_SIZE"
  --hidden-dim "$HIDDEN_DIM"
  --n-steps-path 24
  --sigma-min "$SIGMA_MIN"
  --sigma-max "$SIGMA_MAX"
  --eval-samples "$EVAL_SAMPLES"
  --skip-checks
  --disable-baseline-gate
  --seed "$SEED"
  --outdir "$OUTDIR"
)
if [[ "$AUTO_LOG_NORMAL" == "1" ]]; then
  WARMUP_ONLY_ARGS+=(--auto-log-normal-params)
fi
if [[ "$USE_EMA_EVAL" == "1" ]]; then
  WARMUP_ONLY_ARGS+=(--use-ema-eval --ema-decay "$EMA_DECAY")
fi

echo "=== 1/7 Baseline EDM full 40k ==="
BASELINE_EXP="${PREFIX}_baseline40k_s${SEED}"
BASELINE40K_FLAGS=()
if [[ "$FORCE_REBUILD_BASELINE40K" == "1" ]]; then
  BASELINE40K_FLAGS+=(--baseline-ckpt-force-retrain)
fi
if [[ "$SKIP_BASELINE40K" == "1" ]]; then
  echo "[skip] baseline40k stage skipped"
else
  python3 toy/run_toy.py "${COMMON_FID_ARGS[@]}" \
    --steps "$TOTAL_STEPS" \
    --exp-name "$BASELINE_EXP" \
    --method-version v2 \
    --baseline-only \
    --baseline-ckpt-path "$BASELINE40K_CKPT" \
    "${BASELINE40K_FLAGS[@]}"
fi

echo "=== 2/7 Build/verify shared warmup-8k checkpoint (for robust methods only) ==="
WARMUP8K_EXP="${PREFIX}_warmup8k_s${SEED}"
WARMUP8K_FLAGS=()
if [[ "$FORCE_REBUILD_WARMUP8K" == "1" ]]; then
  WARMUP8K_FLAGS+=(--baseline-ckpt-force-retrain)
fi
if [[ "$SKIP_WARMUP8K" == "1" ]]; then
  echo "[skip] warmup8k stage skipped; expecting checkpoint at: $WARMUP8K_CKPT"
  if [[ ! -f "$WARMUP8K_CKPT" ]]; then
    echo "[ERROR] warmup checkpoint not found while SKIP_WARMUP8K=1: $WARMUP8K_CKPT"
    exit 1
  fi
else
  python3 toy/run_toy.py "${WARMUP_ONLY_ARGS[@]}" \
    --steps "$WARMUP_STEPS" \
    --exp-name "$WARMUP8K_EXP" \
    --method-version v2 \
    --baseline-only \
    --baseline-ckpt-path "$WARMUP8K_CKPT" \
    "${WARMUP8K_FLAGS[@]}"
fi

echo "=== 3/7 Create alias checkpoints from warmup-8k weights ==="
python3 - <<'PY'
import os
from datetime import datetime, timezone
import torch

src = os.environ['WARMUP8K_CKPT']
aliases = [
    (os.environ['ALIAS32K_CKPT'], int(os.environ['ROBUST_STEPS'])),
    (os.environ['ALIAS_V11_CKPT'], int(os.environ['V11_EQ_STEPS'])),
]
if not os.path.isfile(src):
    raise SystemExit(f"[ERROR] warmup checkpoint missing: {src}")
payload = torch.load(src, map_location='cpu')
if not isinstance(payload, dict) or 'baseline_state_dict' not in payload:
    raise SystemExit(f"[ERROR] invalid warmup checkpoint: {src}")

for dst, steps in aliases:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    sig = payload.get('baseline_signature', {})
    if not isinstance(sig, dict):
        sig = {}
    sig = dict(sig)
    sig['baseline_steps'] = int(steps)

    out = dict(payload)
    out['baseline_signature'] = sig
    out['alias_from_ckpt'] = str(src)
    out['alias_for_baseline_steps'] = int(steps)
    out['alias_saved_at'] = datetime.now(timezone.utc).isoformat()

    tmp = dst + f".tmp.{os.getpid()}"
    torch.save(out, tmp)
    os.replace(tmp, dst)
    print(f"[done] alias: {dst} (baseline_steps={steps})")
PY

ROBUST_COMMON_ARGS=(
  --dataset-kind mnist
  --model-kind image_conv
  --diagnostics-kind image_basic
  --image-channels 1
  --image-size 32
  --mnist-train-percent "$MNIST_TRAIN_PERCENT"
  --mnist-val-percent "$MNIST_VAL_PERCENT"
  --device cuda
  --training-objective edm
  --batch-size "$BATCH_SIZE"
  --hidden-dim "$HIDDEN_DIM"
  --n-steps-path 24
  --sigma-min "$SIGMA_MIN"
  --sigma-max "$SIGMA_MAX"
  --eval-samples "$EVAL_SAMPLES"
  --compute-fid
  --fid-samples "$FID_SAMPLES"
  --skip-checks
  --disable-baseline-gate
  --seed "$SEED"
  --outdir "$OUTDIR"
  --warmup-clean-steps "$ROBUST_WARMUP_CLEAN_STEPS"
  --warmup-ramp-steps "$ROBUST_WARMUP_RAMP_STEPS"
)
if [[ "$AUTO_LOG_NORMAL" == "1" ]]; then
  ROBUST_COMMON_ARGS+=(--auto-log-normal-params)
fi
if [[ "$USE_EMA_EVAL" == "1" ]]; then
  ROBUST_COMMON_ARGS+=(--use-ema-eval --ema-decay "$EMA_DECAY")
fi

WILD_ARGS=(
  --wild-update-interval "$WILD_UPDATE_INTERVAL"
  --wild-cache-batches "$WILD_CACHE_BATCHES"
  --wild-inner-steps "$WILD_INNER_STEPS"
  --wild-step-size "$WILD_STEP_SIZE"
  --wild-gamma "$WILD_GAMMA"
  --wild-sample-min "$WILD_SAMPLE_MIN"
  --wild-sample-max "$WILD_SAMPLE_MAX"
  --wild-delta-ratio-denom "$WILD_DELTA_RATIO_DENOM"
)
if [[ "$WILD_FIXED_NOISE_INNER" == "1" ]]; then
  WILD_ARGS+=(--wild-fixed-noise-inner)
else
  WILD_ARGS+=(--disable-wild-fixed-noise-inner)
fi
if [[ "$WILD_CLAMP_SAMPLES" == "1" ]]; then
  WILD_ARGS+=(--wild-clamp-samples)
fi

V12_ARGS=(
  --inner-steps "$V12_ADV_STEPS"
  --v12-step-size "$V12_STEP_SIZE"
  --v12-lambda-init "$V12_LAMBDA_INIT"
  --v12-lambda-lr "$V12_LAMBDA_LR"
  --v12-rho-target "$V12_RHO_TARGET"
  --v12-robust-mix "$V12_ROBUST_MIX"
  --v12-start-step "$V12_START_STEP"
  --v12-ramp-steps "$V12_RAMP_STEPS"
  --v12-max-delta "$V12_MAX_DELTA"
  --v12-sigma-floor "$V12_SIGMA_FLOOR"
  --v12-sigma-cut "$V12_SIGMA_CUT"
  --v12-gate-power "$V12_GATE_POWER"
  --v12-delta-space "$V12_DELTA_SPACE"
)

echo "=== 4/7 WILD: warmup 8k -> +32k ==="
WILD_EXP="${PREFIX}_wild_s${SEED}"
if [[ "$SKIP_WILD" == "1" ]]; then
  echo "[skip] wild stage skipped"
else
  python3 toy/run_toy.py "${ROBUST_COMMON_ARGS[@]}" \
    --steps "$ROBUST_STEPS" \
    --exp-name "$WILD_EXP" \
    --baseline-ckpt-path "$ALIAS32K_CKPT" \
    --method-version wild \
    "${WILD_ARGS[@]}"
fi

echo "=== 5/7 v1.1: warmup 8k -> +${V11_EQ_STEPS} (eq-compute) ==="
V11_EXP="${PREFIX}_1_1_eq_s${SEED}"
if [[ "$SKIP_V11" == "1" ]]; then
  echo "[skip] v1.1 stage skipped"
else
  python3 toy/run_toy.py "${ROBUST_COMMON_ARGS[@]}" \
    --steps "$V11_EQ_STEPS" \
    --exp-name "$V11_EXP" \
    --baseline-ckpt-path "$ALIAS_V11_CKPT" \
    --method-version 1.1 \
    --v11-step-size "$V11_STEP_SIZE" \
    --v11-transport-gamma "$V11_GAMMA" \
    --v11-total-budget-rho "$V11_RHO" \
    --v11-projection-mode "$V11_PROJ_MODE"
fi

echo "=== 6/7 v1.2: warmup 8k -> +32k ==="
V12_EXP="${PREFIX}_1_2_s${SEED}"
if [[ "$SKIP_V12" == "1" ]]; then
  echo "[skip] v1.2 stage skipped"
else
  python3 toy/run_toy.py "${ROBUST_COMMON_ARGS[@]}" \
    --steps "$ROBUST_STEPS" \
    --exp-name "$V12_EXP" \
    --baseline-ckpt-path "$ALIAS32K_CKPT" \
    --method-version 1.2 \
    "${V12_ARGS[@]}"
fi

echo "=== 7/7 Aggregate summary + panel ==="
if [[ "$SKIP_AGG" == "1" ]]; then
  echo "[skip] aggregate stage skipped"
else
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
strict = os.environ.get('STRICT_CHECKS', '1').strip() not in ('0','false','False')

exp = {
    'baseline40k': f"{prefix}_baseline40k_s{seed}",
    'wild': f"{prefix}_wild_s{seed}",
    'v1_1_eq': f"{prefix}_1_1_eq_s{seed}",
    'v1_2': f"{prefix}_1_2_s{seed}",
}

def sf(v):
    try:
        return float(v)
    except Exception:
        return float('nan')

def load_row(name):
    p = outdir / exp[name] / 'metrics.json'
    if not p.is_file():
        raise SystemExit(f"[ERROR] missing metrics: {p}")
    payload = json.loads(p.read_text())
    m = payload.get('metrics', {})
    flow = m.get('flow_debug', {})
    rt = flow.get('runtime', {})
    q = m.get('sample_quality_debug', {})
    gate = m.get('baseline_gate', {})
    obj = m.get('objective_debug', {})
    return {
        'exp': exp[name],
        'method_version': flow.get('method_version'),
        'baseline_fid': sf(q.get('baseline_fid')),
        'robust_fid': sf(q.get('robust_fid')),
        'fid_delta': sf(q.get('robust_fid')) - sf(q.get('baseline_fid')),
        'attack_training_executed': bool(gate.get('attack_training_executed', False)),
        'baseline_ckpt_loaded': bool(flow.get('baseline_ckpt_loaded', False)),
        'baseline_ckpt_signature_hash': flow.get('baseline_ckpt_signature_hash'),
        'baseline_ckpt_path': flow.get('baseline_ckpt_path'),
        'runtime_total_sec': sf(rt.get('total', flow.get('runtime_total_sec'))),
        'runtime_robust_phase_sec': sf(rt.get('robust_phase')),
        'runtime_total_without_fid_sec': sf(rt.get('total_without_fid')),
        'runtime_baseline_train_sec': sf(rt.get('baseline_train')),
        'sched_attack_weight_mean_last': sf(obj.get('robust_sched_attack_weight', {}).get('mean_last')),
    }

rows = {k: load_row(k) for k in exp.keys()}

# Fairness checks among robust trio only (wild/v1.1eq/v1.2 share warmup-8k init weights).
robust_keys = ['wild', 'v1_1_eq', 'v1_2']
robust_baseline_fids = [rows[k]['baseline_fid'] for k in robust_keys if math.isfinite(rows[k]['baseline_fid'])]
robust_baseline_fid_spread = float(max(robust_baseline_fids) - min(robust_baseline_fids)) if robust_baseline_fids else None

csv_path = outdir / f"{prefix}_runtime_table_baseline40k_wild_v11eq_v12_s{seed}.csv"
with csv_path.open('w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow([
        'method', 'exp', 'method_version', 'baseline_fid', 'robust_fid', 'fid_delta',
        'attack_training_executed', 'baseline_ckpt_loaded', 'runtime_total_sec',
        'runtime_robust_phase_sec', 'runtime_total_without_fid_sec', 'runtime_baseline_train_sec',
        'sched_attack_weight_mean_last', 'baseline_ckpt_path'
    ])
    for k in ['baseline40k', 'wild', 'v1_1_eq', 'v1_2']:
        r = rows[k]
        w.writerow([
            k, r['exp'], r['method_version'], r['baseline_fid'], r['robust_fid'], r['fid_delta'],
            int(r['attack_training_executed']), int(r['baseline_ckpt_loaded']), r['runtime_total_sec'],
            r['runtime_robust_phase_sec'], r['runtime_total_without_fid_sec'], r['runtime_baseline_train_sec'],
            r['sched_attack_weight_mean_last'], r['baseline_ckpt_path']
        ])

# Panel: Real | Baseline40k | WILD | v1.1-eq | v1.2
panel_path = outdir / f"{prefix}_panel_real_baseline40k_wild_v11eq_v12_s{seed}.png"
def note(r):
    bf = r['baseline_fid']
    rf = r['robust_fid']
    return f"bfid={bf:.2f} rfid={rf:.2f}" if math.isfinite(bf) and math.isfinite(rf) else 'fid=n/a'

cmd = [
    sys.executable, 'toy/scripts/make_v2_v21_panel.py',
    '--outdir', str(outdir),
    '--prefix', str(prefix),
    '--seed', str(seed),
    '--baseline-dir', str(outdir / exp['baseline40k'] / 'fid_robust'),
    '--v11-dir', str(outdir / exp['v1_1_eq'] / 'fid_robust'),
    '--v2-dir', str(outdir / exp['wild'] / 'fid_robust'),
    '--v21-dir', str(outdir / exp['v1_2'] / 'fid_robust'),
    '--rows', '8', '--cols', '8', '--tile-size', '32', '--tile-pad', '2',
    '--pick-seed', str(os.environ.get('PICK_SEED', '7')),
    '--real-title', 'Real',
    '--baseline-title', 'Baseline EDM 40k',
    '--v11-title', 'v1.1 (8k + 865)',
    '--v2-title', 'WILD (8k + 32k)',
    '--v21-title', 'v1.2 (8k + 32k)',
    '--real-note', f"N={8*8}",
    '--baseline-note', note(rows['baseline40k']),
    '--v11-note', note(rows['v1_1_eq']),
    '--v2-note', note(rows['wild']),
    '--v21-note', note(rows['v1_2']),
    '--output', str(panel_path),
]
subprocess.run(cmd, check=True)

summary = {
    'protocol': {
        'baseline_full_steps': int(os.environ['TOTAL_STEPS']),
        'warmup_steps_for_robust_methods': int(os.environ['WARMUP_STEPS']),
        'wild_additional_steps': int(os.environ['ROBUST_STEPS']),
        'v1_1_additional_steps': int(os.environ['V11_EQ_STEPS']),
        'v1_2_additional_steps': int(os.environ['ROBUST_STEPS']),
    },
    'rows': rows,
    'fairness_checks': {
        'robust_methods_baseline_fid_spread_abs': robust_baseline_fid_spread,
        'robust_methods_baseline_ckpt_loaded': {k: rows[k]['baseline_ckpt_loaded'] for k in robust_keys},
        'attack_training_executed': {k: rows[k]['attack_training_executed'] for k in rows.keys()},
    },
    'artifacts': {
        'panel': str(panel_path),
        'runtime_csv': str(csv_path),
    },
}
summary_path = outdir / f"{prefix}_summary_baseline40k_wild_v11eq_v12_s{seed}.json"
summary_path.write_text(json.dumps(summary, indent=2))
print('[done] summary:', summary_path)
print('[done] panel:', panel_path)
print('[done] runtime_csv:', csv_path)

if strict:
    for k in robust_keys:
        if not rows[k]['baseline_ckpt_loaded']:
            raise SystemExit(f"[fairness-fail] {k} did not load warmup checkpoint")
    if not rows['wild']['attack_training_executed'] or not rows['v1_1_eq']['attack_training_executed'] or not rows['v1_2']['attack_training_executed']:
        raise SystemExit('[fairness-fail] robust training not executed for one of wild/v1.1/v1.2')
    print('[fair] strict checks passed')
PY
fi

echo "[*] User protocol run completed."
