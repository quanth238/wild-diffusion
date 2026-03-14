# Transfer Train 80k + Eval (from Base `.pkl`)

This guide runs a new experiment by initializing from:

- `/mnt/data/quanth/models/wild-diffusion-base.pkl`

and trains to **80 MIMG** (`kimg=80000`), then evaluates with 50k generated images.

## 0) Assumptions

1. Code is on server at:
   - `/home/quanth/working_space/Wild-Diffusion`
2. Conda env is `quanth`.
3. CIFAR data exists:
   - Full CIFAR: `/mnt/data/quanth/datasets/cifar10-32x32`
   - 20% subset will be created automatically at:
     `/mnt/data/quanth/datasets/cifar10-32x32-train20pct-seed0`
4. Local Inception detector exists:
   - `/mnt/data/quanth/models/inception-2015-12-05.pkl`

## 1) Submit training (80 MIMG, transfer from base pkl)

```bash
cd /home/quanth/working_space

PROJECT_DIR=/home/quanth/working_space/Wild-Diffusion \
./scripts/slurm_workflow.sh submit wd-transfer80k-from-base -- \
env ENV_MODE=conda EXPECTED_CONDA_ENV=quanth STRICT_CONDA_ENV=1 \
    INSTALL_DEPS=0 DATASET_ONLY=0 CIFAR_ALLOW_DOWNLOAD=0 \
    DATA_ROOT=/mnt/data/quanth/datasets \
    CIFAR_DIR=/mnt/data/quanth/datasets/cifar10-32x32 \
    OUTDIR=/mnt/data/quanth/experiments/wild-diffusion \
    TRANSFER_PKL=/mnt/data/quanth/models/wild-diffusion-base.pkl \
    DURATION_MIMG=80 BATCH=1024 BATCH_GPU=1024 FP16=1 LR=1e-5 WORKERS=16 \
    CIFAR_TRAIN_PERCENT=20 CIFAR_TRAIN_SEED=0 \
    WDRO_WARMUP_RATIO=0.2 WDRO_M_EPOCHS=20 WDRO_K=5 WDRO_STEP_SIZE=0.01 WDRO_GAMMA=1.0 WDRO_P_ADV=1.0 \
    DEBUG_EVAL=0 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    bash scripts/setup_and_train_cifar10.sh
```

Notes:
- `TRANSFER_PKL` means start from pretrained/base network weights, not resume optimizer state.
- `DEBUG_EVAL=0` is recommended for faster/more stable throughput during method screening.

## 2) Locate the produced RUN_DIR

After training finishes, get latest run:

```bash
RUN_DIR=$(ls -1dt /mnt/data/quanth/experiments/wild-diffusion/*train20pct-seed0-cond-ddpmpp-wdroedm-gpus1-batch1024-fp16 | head -n 1)
echo "$RUN_DIR"
```

Check final snapshot exists:

```bash
ls -lh "$RUN_DIR"/network-snapshot-*.pkl | tail -n 5
```

For `DURATION_MIMG=80`, expected final snapshot is around:
- `network-snapshot-080000.pkl` (exact naming follows internal tick/save schedule).

## 3) Submit evaluation (50k samples + FID)

```bash
cd /home/quanth/working_space

RUN_DIR=$(ls -1dt /mnt/data/quanth/experiments/wild-diffusion/*train20pct-seed0-cond-ddpmpp-wdroedm-gpus1-batch1024-fp16 | head -n 1)

PROJECT_DIR=/home/quanth/working_space/Wild-Diffusion \
./scripts/slurm_workflow.sh submit wd-transfer80k-eval -- \
env ENV_MODE=conda EXPECTED_CONDA_ENV=quanth STRICT_CONDA_ENV=1 INSTALL_DEPS=0 \
    RUN_DIR="$RUN_DIR" \
    CIFAR_DIR=/mnt/data/quanth/datasets/cifar10-32x32 \
    REF_MODE=compute NUM_IMAGES=50000 GEN_BATCH=128 FID_BATCH=64 GEN_STEPS=35 NPROC_PER_NODE=1 \
    FID_DETECTOR_PATH=/mnt/data/quanth/models/inception-2015-12-05.pkl \
    bash scripts/setup_and_eval_cifar10.sh
```

## 4) Read final result

```bash
EVAL_DIR=$(ls -1dt "$RUN_DIR"/eval/eval-* | head -n 1)
cat "$EVAL_DIR"/evaluation_result.txt
```

Machine-readable:

```bash
cat "$EVAL_DIR"/evaluation_result.json
```

## 5) Why this setup is reasonable for fast method iteration

1. Starts from a fixed base model (`TRANSFER_PKL`) for reproducible branching.
2. Uses 20% data and paper-like WDRO settings but shorter duration (`80` instead of `200`).
3. Disables in-training quick eval to reduce overhead.
4. Keeps final evaluation protocol fixed (`50k` samples + standard CIFAR ref stats).
