# One-Command MNIST Setup + Train + Eval

This repo now has MNIST-specific wrappers for the image-level diffusion pipeline:

- `scripts/setup_and_train_mnist.sh`
- `scripts/setup_and_eval_mnist.sh`
- `scripts/run_mnist_compare.sh`

They are designed for quick iteration on:

- baseline EDM-style training
- WDRO-EDM training

using the locally prepared `mnist-32x32-*` ImageFolderDataset format.

## Train baseline EDM-style MNIST

```bash
TRAINER=baseline PRECOND=wdroedm \
MNIST_DIR=datasets/mnist-32x32-train \
OUTDIR=training-runs/mnist \
DURATION_MIMG=10 BATCH=512 BATCH_GPU=256 LR=1e-4 AUGMENT=0 \
bash scripts/setup_and_train_mnist.sh
```

## Train WDRO-EDM MNIST

```bash
TRAINER=wdro PRECOND=wdroedm \
MNIST_DIR=datasets/mnist-32x32-train \
OUTDIR=training-runs/mnist \
DURATION_MIMG=10 BATCH=512 BATCH_GPU=256 LR=1e-4 AUGMENT=0 \
WDRO_WARMUP_RATIO=0.2 WDRO_M_EPOCHS=10 WDRO_K=3 WDRO_STEP_SIZE=0.01 WDRO_GAMMA=1.0 WDRO_P_ADV=1.0 \
bash scripts/setup_and_train_mnist.sh
```

## Build a limited-data subset

```bash
MNIST_TRAIN_PERCENT=20 MNIST_TRAIN_SEED=0 DATASET_ONLY=1 \
bash scripts/setup_and_train_mnist.sh
```

This creates a deterministic subset under:

- `datasets/mnist-32x32-train20pct-seed0`

## Evaluate a finished MNIST run

```bash
RUN_DIR=training-runs/mnist/00000-mnist-32x32-train-cond-ddpmpp-wdroedm-baseline-gpus1-batch512-fp16 \
MNIST_DIR=datasets/mnist-32x32-train \
NUM_IMAGES=10000 GEN_BATCH=128 FID_BATCH=64 GEN_STEPS=35 \
bash scripts/setup_and_eval_mnist.sh
```

The eval script:

- resolves the latest `network-snapshot-*.pkl` from `RUN_DIR`
- computes local MNIST FID reference stats if needed
- generates samples
- computes FID
- writes `evaluation_result.json` and `evaluation_result.txt`

## Run baseline vs WDRO comparison end to end

```bash
RUN_TAG=mnist-pilot DURATION_MIMG=10 \
bash scripts/run_mnist_compare.sh
```

This trains `baseline` and `wdro`, evaluates both, and writes:

- `training-runs/<run-tag>/summary.md`
- `training-runs/<run-tag>/summary.json`

## Notes

- `TRAINER=baseline` uses the standard EDM training loop.
- `TRAINER=wdro` uses the WDRO training loop already used by WILD-Diffusion.
- `PRECOND=wdroedm` is the standard EDM-style preconditioning/loss choice in this repo.
- MNIST is grayscale, so `AUGMENT=0` is a reasonable default for quick iteration.
