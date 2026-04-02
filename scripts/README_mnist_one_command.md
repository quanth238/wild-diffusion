# One-Command MNIST Setup + Train + Eval

This repo now has MNIST-specific wrappers for the image-level diffusion pipeline:

- `scripts/setup_and_train_mnist.sh`
- `scripts/setup_and_eval_mnist.sh`
- `scripts/run_mnist_compare.sh`

They are designed for quick iteration on:

- baseline EDM-style training
- WDRO-EDM training
- archived plug-in CDRO-EDM ablations

using the locally prepared `mnist-32x32-*` ImageFolderDataset format.

## Status of plug-in `cdroedm`

As of 2026-04-02, `PRECOND=cdroedm` is not recommended for new MNIST sweeps.

- In matched 0.1 MIMG smoke tests, baseline, `cdro_off`, and `cdro_on` produced nearly identical loss and FID while `cdroedm` cost about 2.6x more `sec/kimg`.
- A bounded repair made `adv_step_size` RMS-correct and added `CDRO/radius_utilization`, but the patched `cdro_on` run still showed no useful end-metric gain.
- Keep `cdroedm` only for archival ablations, regression checks, or perturbation diagnostics.

See `docs/cdro_locked_method.md` for the current decision and historical preset.

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

## Train archived plug-in CDRO-EDM MNIST

```bash
TRAINER=baseline PRECOND=cdroedm \
MNIST_DIR=datasets/mnist-32x32-train \
OUTDIR=training-runs/mnist \
DURATION_MIMG=10 BATCH=512 BATCH_GPU=256 LR=1e-4 AUGMENT=0 \
CDRO_MIX=0.3 CDRO_ADV_STEPS=2 CDRO_STEP_SIZE=0.02 CDRO_MAX_DELTA=0.05 \
CDRO_RHO=1e-4 CDRO_LAMBDA_INIT=0.1 CDRO_LAMBDA_LR=1e-3 \
CDRO_SIGMA_CUT=0.5 CDRO_GATE_POWER=2.0 \
bash scripts/setup_and_train_mnist.sh
```

This path is kept for diagnostic or historical comparison only.

## Low-memory smoke run

```bash
TRAINER=baseline PRECOND=cdroedm \
MNIST_DIR=datasets/mnist-32x32-train \
OUTDIR=training-runs/mnist \
DURATION_MIMG=1 BATCH=32 BATCH_GPU=16 CBASE=64 DROPOUT=0.0 FP16=1 AUGMENT=0 \
TICK=1 SNAP=1 DUMP=1 DEBUG_EVAL=0 \
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

## Run baseline vs WDRO vs archived CDRO comparison end to end

```bash
RUN_TAG=mnist-pilot-cdro METHODS="baseline wdro cdro" DURATION_MIMG=10 \
bash scripts/run_mnist_compare.sh
```

This trains the selected methods, evaluates them, and writes:

- `training-runs/<run-tag>/summary.md`
- `training-runs/<run-tag>/summary.json`

## Notes

- `TRAINER=baseline` uses the standard EDM training loop.
- `TRAINER=wdro` uses the WDRO training loop already used by WILD-Diffusion.
- `PRECOND=wdroedm` is the standard EDM-style preconditioning/loss choice in this repo.
- `PRECOND=cdroedm` uses the standard EDM training loop with the plug-in sigma-gated CDRO robust loss, but it is currently archived rather than recommended.
- `stats.jsonl` now reports `CDRO/radius_utilization`, which should be checked before treating a `cdroedm` run as a meaningful robustness test.
- `CBASE`, `CRES`, and `DROPOUT` can be used to shrink the network for smoke tests.
- MNIST is grayscale, so `AUGMENT=0` is a reasonable default for quick iteration.
