An official code repository of paper WILD-DIFFUSION: A WDRO INSPIRED TRAINING METHOD FOR DIFFUSION MODELS UNDER LIMITED DATA ([Paper Link](https://openreview.net/forum?id=OrCVuQAYzF)).

# WILD-Diffusion
Paper: [WILD-DIFFUSION: A WDRO Inspired Training Method for Diffusion Models Under Limited Data](https://openreview.net/forum?id=OrCVuQAYzF)

## 1) Train from scratch (CIFAR-10 20%, cond. implemented as in WDRO paper)

Run on server:

```bash
cd /home/quanth/working_space

PROJECT_DIR=/home/quanth/working_space/Wild-Diffusion \
./scripts/slurm_workflow.sh submit wd-cifar10-paper-20pct -- \
env ENV_MODE=conda EXPECTED_CONDA_ENV=quanth STRICT_CONDA_ENV=1 \
    INSTALL_DEPS=0 DATASET_ONLY=0 CIFAR_ALLOW_DOWNLOAD=0 \
    DATA_ROOT=/mnt/data/quanth/datasets \
    CIFAR_DIR=/mnt/data/quanth/datasets/cifar10-32x32 \
    OUTDIR=/mnt/data/quanth/experiments/wild-diffusion \
    DURATION_MIMG=200 BATCH=1024 BATCH_GPU=1024 FP16=1 LR=1e-5 WORKERS=16 \
    CIFAR_TRAIN_PERCENT=20 CIFAR_TRAIN_SEED=0 \
    WDRO_WARMUP_RATIO=0.2 WDRO_M_EPOCHS=20 WDRO_K=5 WDRO_STEP_SIZE=0.01 WDRO_GAMMA=1.0 WDRO_P_ADV=1.0 \
    DEBUG_EVAL=1 DEBUG_EVAL_INIT=1 DEBUG_EVAL_NUM=128 DEBUG_EVAL_STEPS=18 DEBUG_EVAL_BATCH=64 DEBUG_EVAL_VISUAL=32 DEBUG_ADV_VISUAL=16 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    bash scripts/setup_and_train_cifar10.sh
```

Expected output run dir:

```text
/mnt/data/quanth/experiments/wild-diffusion/00012-cifar10-32x32-train20pct-seed0-cond-ddpmpp-wdroedm-gpus1-batch1024-fp16
```

## 2) Evaluate (Generate 50k + FID)

```bash
cd /home/quanth/working_space

PROJECT_DIR=/home/quanth/working_space/Wild-Diffusion \
./scripts/slurm_workflow.sh submit wd-cifar10-eval-00012 -- \
env ENV_MODE=conda EXPECTED_CONDA_ENV=quanth STRICT_CONDA_ENV=1 INSTALL_DEPS=0 \
    RUN_DIR=/mnt/data/quanth/experiments/wild-diffusion/00012-cifar10-32x32-train20pct-seed0-cond-ddpmpp-wdroedm-gpus1-batch1024-fp16 \
    CIFAR_DIR=/mnt/data/quanth/datasets/cifar10-32x32 \
    REF_MODE=compute NUM_IMAGES=50000 GEN_BATCH=128 FID_BATCH=64 GEN_STEPS=35 NPROC_PER_NODE=1 \
    FID_DETECTOR_PATH=/mnt/data/quanth/models/inception-2015-12-05.pkl \
    bash scripts/setup_and_eval_cifar10.sh
```

## 3) Result (verified)
- Training finished at `kimg 200000.5` with `Exit code: 0`.
- Snapshot used for final eval: `network-snapshot-200000.pkl`.
- Final FID: `6.17323`.
- Eval artifact dir:
  `/mnt/data/quanth/experiments/wild-diffusion/00012-cifar10-32x32-train20pct-seed0-cond-ddpmpp-wdroedm-gpus1-batch1024-fp16/eval/eval-20260314-093057`

## 4) Changes vs Original Wild-Diffusion (ordered by importance)

This branch is not a pure upstream copy. The main changes are:

1. WDRO transport penalty in adversarial inner loop was changed to the paper form:
   `0.5 * ||x_adv - x||_2^2` (instead of unsquared mean L2 norm).
2. A critical input-scale fix was added in training:
   raw dataset batches (`uint8`) are normalized by `/127.5 - 1`, while WDRO-combined batches already in `[-1,1]` are kept as-is.
3. WDRO schedule/hyperparameters became configurable from CLI:
   `wdro_warmup_ratio`, `wdro_m_epochs`, `wdro_k`, `wdro_step_size`, `wdro_gamma`, `wdro_p_adv`.
4. Resume logic now preserves WDRO phase:
   `next_wdro_kimg` is saved/restored in `training-state-*.pt`; resume avoids expensive WDRO catch-up loops.
5. Quick interval evaluation was added:
   quick FID + generated previews at init and each WDRO refresh (`quick_eval/` outputs).
6. Adversarial debugging outputs were added per WDRO **interval**:
   `orig_grid.png`, `adv_grid.png`, `delta_grid.png`, `triplet_grid_3row.png`, plus per-image exports.
7. CIFAR-10 limited-data pipeline was added in train script:
   automatic 20% subset build (`10000` images) with deterministic seed control.
8. End-to-end evaluation script was added:
   generate 50k samples, compute FID, and export machine-readable results (`evaluation_result.json/txt`).
9. FID can use a local detector checkpoint via `FID_DETECTOR_PATH` (cluster/offline-safe).
10. Logger/telemetry were hardened:
    training no longer crashes when stdout quota is exceeded or process memory query is transiently unavailable.
11. WDRO helper functions were refactored into `training/wdro_utils.py` to reduce complexity in `training_wdro_loop.py` (no intended algorithmic change).

## 5) Notes for paper **comparison**
1. This run uses CIFAR-10 20% training data (`10000` images) with class-conditional training (`--cond=1`).
2. Final FID is computed with standard protocol against full CIFAR-10 train reference stats (`50000` real images).

## 6) Fast MNIST Iteration

For quicker image-level iteration on baseline EDM vs WDRO-EDM, use the MNIST wrappers:

- [`scripts/setup_and_train_mnist.sh`](scripts/setup_and_train_mnist.sh)
- [`scripts/setup_and_eval_mnist.sh`](scripts/setup_and_eval_mnist.sh)
- [`scripts/run_mnist_compare.sh`](scripts/run_mnist_compare.sh)
- [`scripts/README_mnist_one_command.md`](scripts/README_mnist_one_command.md)

Status note as of 2026-04-02:

- `PRECOND=cdroedm` is currently kept as an archival plug-in ablation path, not a recommended mainline method.
- Matched 0.1 MIMG MNIST smoke tests, before and after fixing an RMS-vs-L2 step-scaling bug in `EDMLossCDRO`, still showed about 2.6x compute cost with no material loss or FID gain.
- For new CDRO work, do not start from `cdroedm` by default. Use it only as a historical control and read [`docs/cdro_locked_method.md`](docs/cdro_locked_method.md) first.
