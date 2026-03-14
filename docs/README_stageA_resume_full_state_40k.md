# Stage A (Resume Full-State from 40k): Minimal Runbook

This is the exact, minimal protocol for supervisor-style screening:

- start from shared post-warmup 40k full-state,
- continue `+20 MIMG` (to total `60 MIMG`),
- evaluate with official FID-50k pipeline.

## Fixed anchor files (confirmed)

```text
/mnt/data/quanth/models/warm_up_wild_diffusion_with_20_data/network-snapshot-040000.pkl
/mnt/data/quanth/models/warm_up_wild_diffusion_with_20_data/training-state-040000.pt
```

You need both files for `--resume` full-state.

## 1) Prepare isolated branch directory

```bash
ANCHOR_DIR=/mnt/data/quanth/models/warm_up_wild_diffusion_with_20_data
EXP_NAME=stageA-baseline-plus20-seed0
BRANCH_DIR=/mnt/data/quanth/experiments/wild-diffusion/${EXP_NAME}

mkdir -p "$BRANCH_DIR"
cp "$ANCHOR_DIR/training-state-040000.pt" "$BRANCH_DIR/"
cp "$ANCHOR_DIR/network-snapshot-040000.pkl" "$BRANCH_DIR/"
```

## 2) Train continuation (+20 MIMG from 40k)

`40k + 20k => target total 60k`, so set `DURATION_MIMG=60`.

```bash
cd /home/quanth/working_space

EXP_NAME=stageA-baseline-plus20-seed0
BRANCH_DIR=/mnt/data/quanth/experiments/wild-diffusion/${EXP_NAME}
WARMUP_STATE=${BRANCH_DIR}/training-state-040000.pt

PROJECT_DIR=/home/quanth/working_space/Wild-Diffusion \
./scripts/slurm_workflow.sh submit "${EXP_NAME}" -- \
env ENV_MODE=conda EXPECTED_CONDA_ENV=quanth STRICT_CONDA_ENV=1 \
    INSTALL_DEPS=0 DATASET_ONLY=0 CIFAR_ALLOW_DOWNLOAD=0 \
    DATA_ROOT=/mnt/data/quanth/datasets \
    CIFAR_DIR=/mnt/data/quanth/datasets/cifar10-32x32 \
    RESUME="$WARMUP_STATE" \
    DURATION_MIMG=60 BATCH=1024 BATCH_GPU=1024 FP16=1 LR=1e-5 WORKERS=16 \
    CIFAR_TRAIN_PERCENT=20 CIFAR_TRAIN_SEED=0 SEED=0 \
    WDRO_WARMUP_RATIO=0.2 WDRO_M_EPOCHS=20 WDRO_K=5 WDRO_STEP_SIZE=0.01 WDRO_GAMMA=1.0 WDRO_P_ADV=1.0 \
    DEBUG_EVAL=0 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    bash scripts/setup_and_train_cifar10.sh
```

## 3) Verify this is true full-state resume

```bash
JOB=<TRAIN_JOB_ID>
LOG=/mnt/data/quanth/slurm_logs/slurm-${JOB}.out
RUN_DIR=$(grep -oP 'Output directory:\s+\K.*' "$LOG" | tail -n1)
export RUN_DIR
echo "$RUN_DIR"

grep -E "Loading network weights from|Loading training state from" "$RUN_DIR/log.txt" | tail -n 10

python - <<'PY'
import json, os
cfg = json.load(open(os.path.join(os.environ["RUN_DIR"], "training_options.json")))
print("resume_pkl       =", cfg.get("resume_pkl"))
print("resume_state_dump=", cfg.get("resume_state_dump"))
print("resume_kimg      =", cfg.get("resume_kimg"))
PY
```

Expected:

- `resume_state_dump` is not `None`
- log contains `Loading training state from "...training-state-040000.pt"...`

## 4) Evaluate (official 50k images)

```bash
cd /home/quanth/working_space

EXP_NAME=stageA-baseline-plus20-seed0
RUN_DIR=/mnt/data/quanth/experiments/wild-diffusion/${EXP_NAME}

PROJECT_DIR=/home/quanth/working_space/Wild-Diffusion \
./scripts/slurm_workflow.sh submit wd-stageA-plus20-eval -- \
env ENV_MODE=conda EXPECTED_CONDA_ENV=quanth STRICT_CONDA_ENV=1 INSTALL_DEPS=0 \
    RUN_DIR="$RUN_DIR" \
    CIFAR_DIR=/mnt/data/quanth/datasets/cifar10-32x32 \
    REF_MODE=compute NUM_IMAGES=50000 GEN_BATCH=128 FID_BATCH=64 GEN_STEPS=35 NPROC_PER_NODE=1 \
    FID_DETECTOR_PATH=/mnt/data/quanth/models/inception-2015-12-05.pkl \
    bash scripts/setup_and_eval_cifar10.sh
```

Read result:

```bash
EVAL_DIR=$(ls -1dt "$RUN_DIR"/eval/eval-* | head -n 1)
cat "$EVAL_DIR"/evaluation_result.txt
cat "$EVAL_DIR"/evaluation_result.json
```

## 5) Reuse for A/B

For each candidate algorithm, set a new `EXP_NAME`, create new `BRANCH_DIR`, copy the same two anchor files, keep seed/hyperparameters fixed, run the same `+20 MIMG` and eval pipeline.
