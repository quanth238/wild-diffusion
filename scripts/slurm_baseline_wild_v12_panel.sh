#!/bin/bash
#SBATCH --job-name=mnist_b_wild_v12
#SBATCH --output=toy_outputs/slurm_b_wild_v12_%j.out
#SBATCH --error=toy_outputs/slurm_b_wild_v12_%j.err
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

# Let Slurm keep the correct GPU mapping for both full-GPU and MIG allocations.
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
OUTDIR="${OUTDIR:-/mnt/data/quanth/wild_diffusion_outputs/mnist_baseline_wild_v12}"
PREFIX="${PREFIX:-mnist_${MNIST_TRAIN_PERCENT}pct_s${SEED}_t${STEPS}}"
PICK_SEED="${PICK_SEED:-7}"
SIGMA_MIN="${SIGMA_MIN:-0.01}"
SIGMA_MAX="${SIGMA_MAX:-80}"
AUTO_LOG_NORMAL="${AUTO_LOG_NORMAL:-1}"
USE_EMA_EVAL="${USE_EMA_EVAL:-1}"
EMA_DECAY="${EMA_DECAY:-0.999}"
WARMUP_CLEAN_STEPS="${WARMUP_CLEAN_STEPS:-900}"
WARMUP_RAMP_STEPS="${WARMUP_RAMP_STEPS:-600}"

# WILD (method-version=wild)
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

# v1.2 (CDRO-EDM-aligned single-sigma robust training)
V12_STEP_SIZE="${V12_STEP_SIZE:-0.02}"
V12_ADV_STEPS="${V12_ADV_STEPS:-1}"
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
SKIP_BASELINE="${SKIP_BASELINE:-0}"
SKIP_WILD="${SKIP_WILD:-0}"
SKIP_V12="${SKIP_V12:-0}"
SKIP_AGG="${SKIP_AGG:-0}"

export SEED
export STEPS
export MNIST_TRAIN_PERCENT
export MNIST_VAL_PERCENT
export OUTDIR
export PREFIX
export PICK_SEED
export STRICT_FAIRNESS_CHECK
export BASELINE_FID_SPREAD_TOL

mkdir -p "${OUTDIR}"
BASELINE_CKPT_DIR="${BASELINE_CKPT_DIR:-/mnt/data/quanth/models/wild_diffusion_baseline_ckpt}"
mkdir -p "${BASELINE_CKPT_DIR}"
BASELINE_CKPT_PATH="${BASELINE_CKPT_DIR}/baseline_edm_s${SEED}_st${STEPS}_bs${BATCH_SIZE}_hd${HIDDEN_DIM}_tp${MNIST_TRAIN_PERCENT}.pt"

echo "=== Locked fairness config ==="
echo "SEED=${SEED}"
echo "STEPS=${STEPS} BATCH_SIZE=${BATCH_SIZE} HIDDEN_DIM=${HIDDEN_DIM}"
echo "MNIST_TRAIN_PERCENT=${MNIST_TRAIN_PERCENT} MNIST_VAL_PERCENT=${MNIST_VAL_PERCENT}"
echo "SIGMA_MIN=${SIGMA_MIN} SIGMA_MAX=${SIGMA_MAX} AUTO_LOG_NORMAL=${AUTO_LOG_NORMAL}"
echo "USE_EMA_EVAL=${USE_EMA_EVAL} EMA_DECAY=${EMA_DECAY}"
echo "WARMUP_CLEAN_STEPS=${WARMUP_CLEAN_STEPS} WARMUP_RAMP_STEPS=${WARMUP_RAMP_STEPS}"
echo "OUTDIR=${OUTDIR} PREFIX=${PREFIX}"
echo "BASELINE_CKPT_PATH=${BASELINE_CKPT_PATH}"
echo "WILD: interval=${WILD_UPDATE_INTERVAL} cache_batches=${WILD_CACHE_BATCHES} inner_steps=${WILD_INNER_STEPS} step_size=${WILD_STEP_SIZE} gamma=${WILD_GAMMA}"
echo "v1.2: adv_steps=${V12_ADV_STEPS} step_size=${V12_STEP_SIZE} lambda_init=${V12_LAMBDA_INIT} lambda_lr=${V12_LAMBDA_LR} rho_target=${V12_RHO_TARGET} robust_mix=${V12_ROBUST_MIX}"
echo "STRICT_FAIRNESS_CHECK=${STRICT_FAIRNESS_CHECK} BASELINE_FID_SPREAD_TOL=${BASELINE_FID_SPREAD_TOL}"
echo "SKIP_BASELINE=${SKIP_BASELINE} SKIP_WILD=${SKIP_WILD} SKIP_V12=${SKIP_V12} SKIP_AGG=${SKIP_AGG}"

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

BASELINE_EXP="${PREFIX}_baseline_s${SEED}"
WILD_EXP="${PREFIX}_wild_s${SEED}"
V12_EXP="${PREFIX}_1_2_s${SEED}"

if [[ "${SKIP_BASELINE}" == "1" ]]; then
  echo "=== 1/4 Baseline EDM (baseline-only) [SKIPPED] ==="
else
  echo "=== 1/4 Baseline EDM (baseline-only) ==="
  python3 toy/run_toy.py "${COMMON_ARGS[@]}" \
    --exp-name "${BASELINE_EXP}" \
    --method-version v2 \
    --baseline-only
fi

if [[ "${SKIP_WILD}" == "1" ]]; then
  echo "=== 2/4 WILD robust [SKIPPED] ==="
else
  echo "=== 2/4 WILD robust ==="
  python3 toy/run_toy.py "${COMMON_ARGS[@]}" \
    --exp-name "${WILD_EXP}" \
    --method-version wild \
    "${WILD_ARGS[@]}"
fi

if [[ "${SKIP_V12}" == "1" ]]; then
  echo "=== 3/4 v1.2 robust [SKIPPED] ==="
else
  echo "=== 3/4 v1.2 robust ==="
  python3 toy/run_toy.py "${COMMON_ARGS[@]}" \
    --exp-name "${V12_EXP}" \
    --method-version 1.2 \
    "${V12_ARGS[@]}"
fi

if [[ "${SKIP_AGG}" == "1" ]]; then
  echo "=== 4/4 Aggregate metrics + build annotated panel [SKIPPED] ==="
else
  echo "=== 4/4 Aggregate metrics + build annotated panel ==="
  python3 - <<'PY'
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

outdir = Path(os.environ["OUTDIR"])
prefix = os.environ["PREFIX"]
seed = os.environ["SEED"]
strict = os.environ.get("STRICT_FAIRNESS_CHECK", "1").strip() not in ("0", "false", "False")
fid_spread_tol = float(os.environ.get("BASELINE_FID_SPREAD_TOL", "1e-9"))

baseline_exp = f"{prefix}_baseline_s{seed}"
wild_exp = f"{prefix}_wild_s{seed}"
v12_exp = f"{prefix}_1_2_s{seed}"

LOCKED_KEYS = [
    "dataset_kind",
    "model_kind",
    "diagnostics_kind",
    "image_size",
    "image_channels",
    "limited_data_enabled",
    "mnist_use_percent_split",
    "mnist_train_percent",
    "mnist_val_percent",
    "steps",
    "batch_size",
    "hidden_dim",
    "training_objective",
    "n_steps_path",
    "sigma_min",
    "sigma_max",
    "use_log_normal_sigma_sampling",
    "p_mean",
    "p_std",
    "seed",
    "image_split_seed",
]

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

def _cfg_subset(cfg):
    return {k: cfg.get(k) for k in LOCKED_KEYS}

def _json_hash(obj):
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()

def load_row(exp: str):
    p = outdir / exp / "metrics.json"
    payload = json.loads(p.read_text())
    metrics = payload.get("metrics", {})
    cfg = payload.get("config", {})
    flow = metrics.get("flow_debug", {})
    runtime = flow.get("runtime", {})
    q = metrics.get("sample_quality_debug", {})
    cdbg = metrics.get("constraint_debug", {})
    rec = metrics.get("recovery_debug", {})
    ddbg = metrics.get("dataset_debug", {})
    fid_base_t = _safe_float(runtime.get("fid_baseline", 0.0))
    fid_rob_t = _safe_float(runtime.get("fid_robust", 0.0))
    if not math.isfinite(fid_base_t):
        fid_base_t = 0.0
    if not math.isfinite(fid_rob_t):
        fid_rob_t = 0.0
    return {
        "exp": exp,
        "method_version": flow.get("method_version"),
        "baseline_fid": _safe_float(q.get("baseline_fid")),
        "robust_fid": _safe_float(q.get("robust_fid")),
        "attack_training_executed": bool(metrics.get("baseline_gate", {}).get("attack_training_executed", False)),
        "baseline_ckpt_loaded": bool(flow.get("baseline_ckpt_loaded", False)),
        "baseline_ckpt_saved": bool(flow.get("baseline_ckpt_saved", False)),
        "baseline_ckpt_signature_hash": flow.get("baseline_ckpt_signature_hash"),
        "attack_win_ratio_overall": _safe_float(cdbg.get("attack_gap_windows_heldout", {}).get("overall", {}).get("win_ratio")),
        "attack_gap_overall": _safe_float(cdbg.get("attack_gap_windows_heldout", {}).get("overall", {}).get("mean_attack_gap")),
        "recovery_mse_ref_terminal": _safe_float(rec.get("baseline_x0_mse_from_ref_terminal")),
        "recovery_mse_attack_terminal": _safe_float(rec.get("baseline_x0_mse_from_attack_terminal")),
        "runtime_total_sec": _safe_float(runtime.get("total", flow.get("runtime_total_sec"))),
        "runtime_total_without_fid_sec": _safe_float(runtime.get("total_without_fid")),
        "runtime_baseline_phase_sec": _safe_float(runtime.get("baseline_phase")),
        "runtime_baseline_train_sec": _safe_float(runtime.get("baseline_train")),
        "runtime_robust_phase_sec": _safe_float(runtime.get("robust_phase")),
        "runtime_post_eval_sec": _safe_float(runtime.get("post_train_eval")),
        "runtime_fid_total_sec": float(fid_base_t + fid_rob_t),
        "runtime_steps_per_sec_robust": _safe_float(runtime.get("robust_steps_per_sec")),
        "dataset_policy": ddbg.get("train_selection_policy"),
        "train_subset_size_resolved": ddbg.get("train_subset_size_resolved"),
        "val_subset_size_resolved": ddbg.get("val_subset_size_resolved"),
        "cfg_locked_subset": _cfg_subset(cfg),
        "eval_seed_gate": flow.get("eval_seed_gate"),
        "eval_seed_metrics": flow.get("eval_seed_metrics"),
        "eval_shared_terminal_noise": flow.get("eval_shared_terminal_noise"),
        "eval_shared_reverse_noise": flow.get("eval_shared_reverse_noise"),
    }

rows = {
    "baseline_only": load_row(baseline_exp),
    "wild": load_row(wild_exp),
    "v1_2": load_row(v12_exp),
}

baseline_fids = [rows["baseline_only"]["baseline_fid"], rows["wild"]["baseline_fid"], rows["v1_2"]["baseline_fid"]]
finite_baseline_fids = [v for v in baseline_fids if math.isfinite(v)]
baseline_fid_spread_abs = (
    float(max(finite_baseline_fids) - min(finite_baseline_fids))
    if finite_baseline_fids
    else None
)

ckpt_hashes = {rows["baseline_only"]["baseline_ckpt_signature_hash"], rows["wild"]["baseline_ckpt_signature_hash"], rows["v1_2"]["baseline_ckpt_signature_hash"]}
policy_hashes = {_json_hash(rows[k]["dataset_policy"]) for k in rows.keys()}
cfg_hashes = {_json_hash(rows[k]["cfg_locked_subset"]) for k in rows.keys()}
train_sizes = [rows[k]["train_subset_size_resolved"] for k in rows.keys()]
val_sizes = [rows[k]["val_subset_size_resolved"] for k in rows.keys()]
eval_proto = {
    _json_hash(
        {
            "gate": rows[k]["eval_seed_gate"],
            "metrics": rows[k]["eval_seed_metrics"],
            "shared_terminal": rows[k]["eval_shared_terminal_noise"],
            "shared_reverse": rows[k]["eval_shared_reverse_noise"],
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
    return [{"method": k, "seconds": v} for k, v in sorted(items, key=lambda kv: kv[1])]

runtime_table_path = outdir / f"{prefix}_runtime_table_baseline_wild_v12_s{seed}.csv"
with runtime_table_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(
        [
            "method",
            "exp",
            "baseline_fid",
            "robust_fid",
            "attack_training_executed",
            "runtime_total_sec",
            "runtime_total_without_fid_sec",
            "runtime_baseline_train_sec",
            "runtime_robust_phase_sec",
            "runtime_post_eval_sec",
            "runtime_fid_total_sec",
            "runtime_steps_per_sec_robust",
        ]
    )
    for method_name in ("baseline_only", "wild", "v1_2"):
        r = rows[method_name]
        writer.writerow(
            [
                method_name,
                r["exp"],
                r["baseline_fid"],
                r["robust_fid"],
                int(r["attack_training_executed"]),
                r["runtime_total_sec"],
                r["runtime_total_without_fid_sec"],
                r["runtime_baseline_train_sec"],
                r["runtime_robust_phase_sec"],
                r["runtime_post_eval_sec"],
                r["runtime_fid_total_sec"],
                r["runtime_steps_per_sec_robust"],
            ]
        )

def _panel_note(row):
    return (
        f"FID={_fmt_fid(_safe_float(row['robust_fid']))} | "
        f"total={_fmt_runtime(_safe_float(row['runtime_total_sec']))} | "
        f"robust={_fmt_runtime(_safe_float(row['runtime_robust_phase_sec']))}"
    )

panel_path = outdir / f"{prefix}_panel_real_baseline_wild_v12_s{seed}.png"
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
    "--v2-dir",
    str(outdir / wild_exp / "fid_robust"),
    "--v21-dir",
    str(outdir / v12_exp / "fid_robust"),
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
    "--real-title",
    "Real",
    "--baseline-title",
    "Baseline EDM",
    "--v2-title",
    "WILD Robust",
    "--v21-title",
    "v1.2 Robust",
    "--real-note",
    f"N={8*8}",
    "--baseline-note",
    _panel_note(rows["baseline_only"]),
    "--v2-note",
    _panel_note(rows["wild"]),
    "--v21-note",
    _panel_note(rows["v1_2"]),
    "--output",
    str(panel_path),
]
subprocess.run(panel_cmd, check=True)

fairness_checks = {
    "baseline_fid_spread_abs": baseline_fid_spread_abs,
    "baseline_fid_spread_tol": fid_spread_tol,
    "baseline_ckpt_hash_unique_count": int(len(ckpt_hashes)),
    "baseline_ckpt_hashes": sorted(h for h in ckpt_hashes if h is not None),
    "robust_runs_loaded_baseline_ckpt": {
        "wild": bool(rows["wild"]["baseline_ckpt_loaded"]),
        "v1_2": bool(rows["v1_2"]["baseline_ckpt_loaded"]),
    },
    "attack_training_executed": {
        "wild": bool(rows["wild"]["attack_training_executed"]),
        "v1_2": bool(rows["v1_2"]["attack_training_executed"]),
    },
    "dataset_policy_hash_unique_count": int(len(policy_hashes)),
    "locked_config_hash_unique_count": int(len(cfg_hashes)),
    "eval_protocol_hash_unique_count": int(len(eval_proto)),
    "train_subset_size_spread_abs": float(max(train_sizes) - min(train_sizes)),
    "val_subset_size_spread_abs": float(max(val_sizes) - min(val_sizes)),
}

summary = {
    "seed": int(seed),
    "steps": int(os.environ["STEPS"]),
    "mnist_train_percent": float(os.environ["MNIST_TRAIN_PERCENT"]),
    "mnist_val_percent": float(os.environ["MNIST_VAL_PERCENT"]),
    "rows": rows,
    "runtime_ranking": {
        "by_total_sec": _ranking("runtime_total_sec"),
        "by_robust_phase_sec": _ranking("runtime_robust_phase_sec"),
        "by_total_without_fid_sec": _ranking("runtime_total_without_fid_sec"),
    },
    "fairness_checks": fairness_checks,
    "panel": str(panel_path),
    "runtime_table_csv": str(runtime_table_path),
    "locked_config_keys": LOCKED_KEYS,
}

summary_path = outdir / f"{prefix}_summary_baseline_wild_v12_s{seed}.json"
summary_path.write_text(json.dumps(summary, indent=2))

print("[done] summary:", summary_path)
print("[done] panel:", panel_path)
print("[done] runtime_table:", runtime_table_path)
print("[fair] baseline_fid_spread_abs=", fairness_checks["baseline_fid_spread_abs"])
print("[fair] baseline_ckpt_hash_unique_count=", fairness_checks["baseline_ckpt_hash_unique_count"])
print("[fair] dataset_policy_hash_unique_count=", fairness_checks["dataset_policy_hash_unique_count"])
print("[fair] locked_config_hash_unique_count=", fairness_checks["locked_config_hash_unique_count"])
print("[runtime] fastest_by_total=", summary["runtime_ranking"]["by_total_sec"][0] if summary["runtime_ranking"]["by_total_sec"] else None)
print("[runtime] fastest_by_robust_phase=", summary["runtime_ranking"]["by_robust_phase_sec"][0] if summary["runtime_ranking"]["by_robust_phase_sec"] else None)

if strict:
    if fairness_checks["baseline_ckpt_hash_unique_count"] != 1:
        raise SystemExit(
            f"[fairness-fail] expected exactly 1 baseline ckpt hash, got {fairness_checks['baseline_ckpt_hash_unique_count']}"
        )
    if not all(fairness_checks["robust_runs_loaded_baseline_ckpt"].values()):
        raise SystemExit(
            f"[fairness-fail] robust runs did not load shared baseline checkpoint: {fairness_checks['robust_runs_loaded_baseline_ckpt']}"
        )
    if fairness_checks["dataset_policy_hash_unique_count"] != 1:
        raise SystemExit(
            f"[fairness-fail] dataset policy mismatch across methods, hash_count={fairness_checks['dataset_policy_hash_unique_count']}"
        )
    if fairness_checks["locked_config_hash_unique_count"] != 1:
        raise SystemExit(
            f"[fairness-fail] locked config mismatch across methods, hash_count={fairness_checks['locked_config_hash_unique_count']}"
        )
    if fairness_checks["eval_protocol_hash_unique_count"] != 1:
        raise SystemExit(
            f"[fairness-fail] eval protocol mismatch across methods, hash_count={fairness_checks['eval_protocol_hash_unique_count']}"
        )
    if fairness_checks["train_subset_size_spread_abs"] != 0.0 or fairness_checks["val_subset_size_spread_abs"] != 0.0:
        raise SystemExit(
            "[fairness-fail] train/val subset size mismatch across methods: "
            f"train_spread={fairness_checks['train_subset_size_spread_abs']} "
            f"val_spread={fairness_checks['val_subset_size_spread_abs']}"
        )
    spread = fairness_checks["baseline_fid_spread_abs"]
    if spread is None or spread > fid_spread_tol:
        raise SystemExit(
            "[fairness-fail] baseline_fid mismatch across methods: "
            f"spread={spread}, tol={fid_spread_tol}"
        )
    if not all(fairness_checks["attack_training_executed"].values()):
        raise SystemExit(
            f"[fairness-fail] attack training was skipped: {fairness_checks['attack_training_executed']}"
        )
    print("[fair] strict checks passed")
PY
fi

echo "[*] Finished baseline-WILD-v1.2 panel workflow"
