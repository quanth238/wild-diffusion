#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/wild-diffusion}"
VENV_PY="${VENV_PY:-$ROOT/.venv/bin/python}"

BASELINE_EVAL="${BASELINE_EVAL:-$ROOT/training-runs/mnist-compare-100pct-v3/eval/baseline-3mimg/eval-20260330-024016/evaluation_result.json}"
WDRO_EVAL="${WDRO_EVAL:-$ROOT/training-runs/mnist-wdro-retune3-g-100pct-v1/eval-3mimg/eval-20260330-015225/evaluation_result.json}"
BASELINE_SAMPLES="${BASELINE_SAMPLES:-$ROOT/training-runs/mnist-compare-100pct-v3/eval/baseline-3mimg/eval-20260330-024016/samples}"
WDRO_SAMPLES="${WDRO_SAMPLES:-$ROOT/training-runs/mnist-wdro-retune3-g-100pct-v1/eval-3mimg/eval-20260330-015225/samples}"
DATASET_DIR="${DATASET_DIR:-$ROOT/datasets/mnist-32x32-train}"

RUN_DIR="${RUN_DIR:-$ROOT/training-runs/mnist-cdro-finetune-a-100pct-v1x/runs/cdro/00000-mnist-32x32-train-cond-ddpmpp-cdroedm-baseline-gpus1-batch96-fp16}"
EVAL_ROOT="${EVAL_ROOT:-$ROOT/training-runs/mnist-cdro-finetune-a-100pct-v1x/eval-300kimg-10k}"
SUMMARY_MD="${SUMMARY_MD:-$ROOT/training-runs/mnist-cdro-finetune-a-100pct-v1x/followup_summary.md}"
OUT_PNG="${OUT_PNG:-$ROOT/training-runs/mnist-cdro-finetune-a-100pct-v1x/visual_samples_balanced_300k_10k.png}"

TARGET_MIMG="${TARGET_MIMG:-0.4}"
CONTINUE_TARGETS="${CONTINUE_TARGETS:-400 500}"
RESUME_STATE="${RESUME_STATE:-$RUN_DIR/training-state-000300.pt}"

TRAINER="${TRAINER:-baseline}"
PRECOND="${PRECOND:-cdroedm}"
BATCH="${BATCH:-96}"
BATCH_GPU="${BATCH_GPU:-96}"
WORKERS="${WORKERS:-4}"
AUGMENT="${AUGMENT:-0}"
GEN_BATCH="${GEN_BATCH:-128}"
FID_BATCH="${FID_BATCH:-64}"
GEN_STEPS="${GEN_STEPS:-35}"
CDRO_MIX="${CDRO_MIX:-0.05}"
CDRO_ADV_STEPS="${CDRO_ADV_STEPS:-1}"
CDRO_STEP_SIZE="${CDRO_STEP_SIZE:-0.02}"
CDRO_MAX_DELTA="${CDRO_MAX_DELTA:-0.03}"
CDRO_RHO="${CDRO_RHO:-5e-5}"
CDRO_LAMBDA_INIT="${CDRO_LAMBDA_INIT:-0.001}"
CDRO_LAMBDA_LR="${CDRO_LAMBDA_LR:-5e-4}"
CDRO_START_KIMG="${CDRO_START_KIMG:-0}"
CDRO_RAMP_KIMG="${CDRO_RAMP_KIMG:-0}"
CDRO_SIGMA_FLOOR="${CDRO_SIGMA_FLOOR:-0.12}"
CDRO_SIGMA_CUT="${CDRO_SIGMA_CUT:-0.70}"
CDRO_GATE_POWER="${CDRO_GATE_POWER:-1.0}"
CDRO_DELTA_SPACE="${CDRO_DELTA_SPACE:-noise}"

latest_eval_json() {
  find "$EVAL_ROOT" -maxdepth 2 -name evaluation_result.json | sort | tail -n 1
}

parse_fid() {
  "$VENV_PY" - "$1" <<'PY'
import json, sys
path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    payload = json.load(f)
print(payload.get("fid50k_full", payload.get("fid")))
PY
}

write_summary() {
  local cdro_eval="$1"
  local cdro_samples="$2"
  local cdro_fid="$3"
  local baseline_fid="$4"
  local wdro_fid="$5"
  cat > "$SUMMARY_MD" <<EOF
# MNIST CDRO Follow-Up

- Baseline EDM 3.0 MIMG FID: \`$baseline_fid\`
- WDRO-EDM 3.0 MIMG FID: \`$wdro_fid\`
- CDRO fine-tune FID: \`$cdro_fid\`
- CDRO eval: [$cdro_eval]($cdro_eval)
- Comparison sheet: [$OUT_PNG]($OUT_PNG)
EOF

  "$VENV_PY" "$ROOT/scripts/export_mnist_comparison_sheet.py" \
    --dataset-dir "$DATASET_DIR" \
    --baseline-samples "$BASELINE_SAMPLES" \
    --wdro-samples "$WDRO_SAMPLES" \
    --cdro-samples "$cdro_samples" \
    --baseline-eval "$BASELINE_EVAL" \
    --wdro-eval "$WDRO_EVAL" \
    --cdro-eval "$cdro_eval" \
    --output "$OUT_PNG"
}

baseline_fid="$(parse_fid "$BASELINE_EVAL")"
wdro_fid="$(parse_fid "$WDRO_EVAL")"

while true; do
  eval_json="$(latest_eval_json || true)"
  if [[ -n "${eval_json:-}" && -f "$eval_json" ]]; then
    break
  fi
  sleep 15
done

cdro_fid="$(parse_fid "$eval_json")"
cdro_samples="$(dirname "$eval_json")/samples"
write_summary "$eval_json" "$cdro_samples" "$cdro_fid" "$baseline_fid" "$wdro_fid"

"$VENV_PY" - "$cdro_fid" "$baseline_fid" <<'PY'
import sys
cdro = float(sys.argv[1])
baseline = float(sys.argv[2])
sys.exit(0 if cdro <= baseline else 1)
PY

echo "CDRO beat baseline: $cdro_fid <= $baseline_fid"

for target_kimg in $CONTINUE_TARGETS; do
  target_mimg="$("$VENV_PY" - "$target_kimg" <<'PY'
import sys
print(int(sys.argv[1]) / 1000.0)
PY
)"
  echo "CDRO missed baseline at current stage: $cdro_fid > $baseline_fid"
  echo "Continuing from $RESUME_STATE to ${target_kimg} kimg"

  ENV_MODE=venv \
  TRAINER="$TRAINER" \
  PRECOND="$PRECOND" \
  MNIST_DIR="$DATASET_DIR" \
  RESUME="$RESUME_STATE" \
  DURATION_MIMG="$target_mimg" \
  BATCH="$BATCH" \
  BATCH_GPU="$BATCH_GPU" \
  WORKERS="$WORKERS" \
  AUGMENT="$AUGMENT" \
  CDRO_MIX="$CDRO_MIX" \
  CDRO_ADV_STEPS="$CDRO_ADV_STEPS" \
  CDRO_STEP_SIZE="$CDRO_STEP_SIZE" \
  CDRO_MAX_DELTA="$CDRO_MAX_DELTA" \
  CDRO_RHO="$CDRO_RHO" \
  CDRO_LAMBDA_INIT="$CDRO_LAMBDA_INIT" \
  CDRO_LAMBDA_LR="$CDRO_LAMBDA_LR" \
  CDRO_START_KIMG="$CDRO_START_KIMG" \
  CDRO_RAMP_KIMG="$CDRO_RAMP_KIMG" \
  CDRO_SIGMA_FLOOR="$CDRO_SIGMA_FLOOR" \
  CDRO_SIGMA_CUT="$CDRO_SIGMA_CUT" \
  CDRO_GATE_POWER="$CDRO_GATE_POWER" \
  CDRO_DELTA_SPACE="$CDRO_DELTA_SPACE" \
  bash "$ROOT/scripts/setup_and_train_mnist.sh"

  RESUME_STATE="$RUN_DIR/training-state-$(printf '%06d' "$target_kimg").pt"
  network_pkl="$RUN_DIR/network-snapshot-$(printf '%06d' "$target_kimg").pkl"
  target_eval_root="$(dirname "$EVAL_ROOT")/eval-${target_kimg}kimg-10k"

  ENV_MODE=venv \
  RUN_DIR="$RUN_DIR" \
  NETWORK_PKL="$network_pkl" \
  EVAL_ROOT="$target_eval_root" \
  MNIST_DIR="$DATASET_DIR" \
  NUM_IMAGES=10000 \
  GEN_BATCH="$GEN_BATCH" \
  FID_BATCH="$FID_BATCH" \
  GEN_STEPS="$GEN_STEPS" \
  bash "$ROOT/scripts/setup_and_eval_mnist.sh"

  eval_json="$(find "$target_eval_root" -maxdepth 2 -name evaluation_result.json | sort | tail -n 1)"
  cdro_fid="$(parse_fid "$eval_json")"
  cdro_samples="$(dirname "$eval_json")/samples"
  write_summary "$eval_json" "$cdro_samples" "$cdro_fid" "$baseline_fid" "$wdro_fid"

  if "$VENV_PY" - "$cdro_fid" "$baseline_fid" <<'PY'
import sys
cdro = float(sys.argv[1])
baseline = float(sys.argv[2])
sys.exit(0 if cdro <= baseline else 1)
PY
  then
    echo "CDRO beat baseline after continuation: $cdro_fid <= $baseline_fid"
    exit 0
  fi
done

echo "CDRO did not beat baseline after continuations. Best checked FID: $cdro_fid vs baseline $baseline_fid"
exit 1
