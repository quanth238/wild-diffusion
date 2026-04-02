# One-Command CIFAR-10 Setup + Train

From the repo root, run:

```bash
bash scripts/setup_and_train_cifar10.sh
```

This single command will:
1. Reuse active conda env if available (or create/use virtualenv as fallback).
2. Install (or validate) required Python packages.
3. Download CIFAR-10 and convert it into `${DATA_ROOT}/cifar10-32x32` (with `dataset.json`).
4. Launch training via `torchrun`.

The wrapper now supports three image-training modes through the same entrypoint:

- `TRAINER=wdro PRECOND=wdroedm` for the WILD-Diffusion WDRO path
- `TRAINER=baseline PRECOND=wdroedm` for plain EDM-style baseline training
- `TRAINER=baseline PRECOND=cdroedm` for the archived plug-in CDRO-EDM ablation path

There is also a compare runner:

- `scripts/run_cifar_compare.sh`

Status note as of 2026-04-02:

- `PRECOND=cdroedm` is currently kept for archival comparisons and diagnostics, not as a recommended starting point for new sweeps.
- Matched MNIST smoke tests, before and after an RMS-step repair, still showed about 2.6x compute cost with no useful loss or FID gain.
- Read `docs/cdro_locked_method.md` before launching new `cdroedm` runs.

## Environment mode (important for your server)

- Default is `ENV_MODE=auto`.
- If a conda env is already active (e.g. `quanth` from `slurm_workflow.sh`), script uses it.
- In conda mode, default is `INSTALL_DEPS=0` (do not pip install into shared env).
- By default it expects env name `quanth` (`EXPECTED_CONDA_ENV=quanth`).
- To force behavior:
  - `ENV_MODE=conda` (require active conda env)
  - `ENV_MODE=venv` (force virtualenv flow)

## Offline/cluster note (important)

- If worker nodes cannot access internet, CIFAR download will timeout.
- Prepare dataset once on a node with internet using `DATASET_ONLY=1`.
- Then submit training jobs with `CIFAR_ALLOW_DOWNLOAD=0` to avoid hanging on download.

## Default storage behavior

- If `/mnt/data/quanth` exists and is writable, defaults are:
  - `VENV_DIR=/mnt/data/quanth/venvs/wild-diffusion`
  - `DATA_ROOT=/mnt/data/quanth/datasets`
  - `OUTDIR=/mnt/data/quanth/experiments/wild-diffusion`
- Otherwise it falls back to repo-local paths (`.venv`, `datasets`, `training-runs`).

## Optional overrides (still one command)

Current default profile is paper-aligned for CIFAR-10 limited-data 20%:
- `DURATION_MIMG=200`
- `BATCH=1024`
- `BATCH_GPU=1024`
- `FP16=1`
- `LR=1e-5` (20% setting)
- `WDRO_WARMUP_RATIO=0.2`
- `WDRO_M_EPOCHS=20`
- `WDRO_K=5`
- `WDRO_STEP_SIZE=0.01`
- `WDRO_GAMMA=1.0`
- `WDRO_P_ADV=1.0`
- `CIFAR_TRAIN_PERCENT=20`
- `DEBUG_EVAL=1` (quick eval at init + each WDRO interval)
- `DEBUG_EVAL_NUM=128`
- `DEBUG_EVAL_STEPS=18`
- `DEBUG_EVAL_VISUAL=32`
- `DEBUG_ADV_VISUAL=16`
- `TRAINER=wdro`
- `PRECOND=wdroedm`

Override example:

```bash
DURATION_MIMG=8 BATCH=1024 BATCH_GPU=512 LR=1e-4 CIFAR_TRAIN_PERCENT=100 bash scripts/setup_and_train_cifar10.sh
```

Other optional overrides:
- `ARCH=ddpmpp|ncsnpp|adm`
- `TRAINER=baseline|wdro`
- `PRECOND=wdroedm|advedm|cdroedm`
- `BATCH=<global_batch>`
- `BATCH_GPU=<micro_batch_per_gpu>`
- `CBASE=<int>`
- `CRES=<comma-separated channel mults>`
- `DROPOUT=<float>`
- `FP16=1|0`
- `EMA=<float>`
- `WDRO_WARMUP_RATIO=<float>`
- `WDRO_M_EPOCHS=<int>`
- `WDRO_K=<int>`
- `WDRO_STEP_SIZE=<float>`
- `WDRO_GAMMA=<float>`
- `WDRO_P_ADV=<0..1>`
- `DEBUG_EVAL=1|0`
- `DEBUG_EVAL_INIT=1|0`
- `DEBUG_EVAL_NUM=<int>`
- `DEBUG_EVAL_STEPS=<int>`
- `DEBUG_EVAL_BATCH=<int>`
- `DEBUG_EVAL_VISUAL=<int>`
- `DEBUG_EVAL_REF=/path/to/ref_stats.npz` (optional precomputed FID reference stats)
- `DEBUG_ADV_VISUAL=<int>`
- `CDRO_MIX=<0..1>`
- `CDRO_ADV_STEPS=<int>`
- `CDRO_STEP_SIZE=<float>`
- `CDRO_MAX_DELTA=<float>`
- `CDRO_RHO=<float>`
- `CDRO_LAMBDA_INIT=<float>`
- `CDRO_LAMBDA_LR=<float>`
- `CDRO_SIGMA_CUT=<float>`
- `CDRO_GATE_POWER=<float>`
- `OUTDIR=/path/to/output`
- `TICK=<int>`
- `SNAP=<int>`
- `DUMP=<int>`
- `SEED=<int>`
- `DESC=<string>`
- `DRY_RUN=1|0`
- `VENV_DIR=/path/to/venv`
- `DATA_ROOT=/path/to/datasets`
- `CIFAR_DIR=/path/to/cifar10-32x32`
- `CIFAR_TRAIN_PERCENT=<1..100>` (default `20`)
- `CIFAR_TRAIN_SEED=<int>` (default `0`)
- `TRAIN_CIFAR_DIR=/path/to/subset-dir` (optional explicit subset path)
- `SERVER_STORAGE_ROOT=/mnt/data/quanth`
- `ENV_MODE=auto|conda|venv`
- `INSTALL_DEPS=0|1`
- `INSTALL_ONLY=1|0` (install/check env only, skip dataset+train)
- `DATASET_ONLY=1|0` (prepare CIFAR only, skip training)
- `CIFAR_ALLOW_DOWNLOAD=1|0`
- `CIFAR_DOWNLOAD_RETRIES=<int>`
- `CIFAR_DOWNLOAD_TIMEOUT=<seconds>`
- `EXPECTED_CONDA_ENV=quanth`
- `STRICT_CONDA_ENV=1|0`
- `TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121`

Memory note:
- On 1 GPU, `BATCH=1024` in fp32 can OOM.
- Use smaller `BATCH` (e.g. 128/64), or set `BATCH_GPU` for gradient accumulation, and enable `FP16=1`.

## Example: baseline EDM on CIFAR-10 20%

```bash
TRAINER=baseline PRECOND=wdroedm \
DURATION_MIMG=8 BATCH=512 BATCH_GPU=128 FP16=1 LR=1e-4 WORKERS=16 \
CIFAR_TRAIN_PERCENT=20 CIFAR_TRAIN_SEED=0 \
bash scripts/setup_and_train_cifar10.sh
```

## Example: archived plug-in CDRO-EDM on CIFAR-10 20%

```bash
TRAINER=baseline PRECOND=cdroedm \
DURATION_MIMG=8 BATCH=512 BATCH_GPU=128 FP16=1 LR=1e-4 WORKERS=16 \
CIFAR_TRAIN_PERCENT=20 CIFAR_TRAIN_SEED=0 \
CDRO_MIX=0.3 CDRO_ADV_STEPS=2 CDRO_STEP_SIZE=0.02 CDRO_MAX_DELTA=0.05 \
CDRO_RHO=1e-4 CDRO_LAMBDA_INIT=0.1 CDRO_LAMBDA_LR=1e-3 \
CDRO_SIGMA_CUT=0.5 CDRO_GATE_POWER=2.0 \
bash scripts/setup_and_train_cifar10.sh
```

Use this path only for archival or diagnostic comparison.

## Example: low-memory smoke run

```bash
TRAINER=baseline PRECOND=cdroedm \
DURATION_MIMG=1 BATCH=32 BATCH_GPU=16 CBASE=64 DROPOUT=0.0 FP16=1 \
TICK=1 SNAP=1 DUMP=1 DEBUG_EVAL=0 \
CIFAR_TRAIN_PERCENT=20 CIFAR_TRAIN_SEED=0 \
bash scripts/setup_and_train_cifar10.sh
```

## Run baseline vs WDRO vs archived CDRO comparison end to end

```bash
RUN_TAG=cifar10-pilot METHODS="baseline wdro cdro" DURATION_MIMG=8 \
bash scripts/run_cifar_compare.sh
```

Notes:
- If `TRAINER` is left unset, the script defaults to `wdro` for `wdroedm` and `advedm`, and to `baseline` for `cdroedm`.
- `TRAINER=wdro` still requires `AUGMENT>0`.
- `PRECOND=cdroedm` currently supports only `TRAINER=baseline` and is kept for archival/ablation use rather than recommended mainline training.
- `stats.jsonl` now reports `CDRO/radius_utilization`; check it before concluding that a `cdroedm` run used a meaningful fraction of its perturbation budget.
- `DRY_RUN=1` skips the GPU check and forwards `--dry-run` to `train.py`.

Quick debug eval note:
- With `DEBUG_EVAL=1`, training writes interval diagnostics under `run_dir/quick_eval/`:
  - `kimg-XXXXXX/preview_grid.png` + `preview_images/*.png`
  - `quick_eval_metrics.jsonl` containing per-interval quick FID
  - `adv_debug/kimg-XXXXXX/{orig_grid,adv_grid,delta_grid,triplet_grid_3row}.png`
  - `delta_grid.png` is a black/white magnitude map of `|adv-orig|` for easier visual inspection
- If you need maximum training throughput, set `DEBUG_EVAL=0`.

Paper-faithful WDRO settings (from 9149_WILD_Diffusion_A_WDRO_Ins.pdf):
- `WDRO_WARMUP_RATIO=0.2`
- `WDRO_M_EPOCHS=20`
- `WDRO_K=5`
- `WDRO_STEP_SIZE=0.01`
- `WDRO_GAMMA=1.0`
- `WDRO_P_ADV=1.0`
- `CIFAR_TRAIN_PERCENT=20` (limited-data setting in Table 1)
- `LR=1e-5` for CIFAR-10 20% (Table 4), `LR=1e-4` for CIFAR-10 100%

## Recommended Profiles

Use these two presets on your server with `scripts/slurm_workflow.sh`.

### `paper_exact_20pct` (closest to paper limited-data setting)

Purpose:
- Reproduce paper-like setting as closely as possible on CIFAR-10.

Expected runtime on 1x H100:
- Typically much longer than 24h (often around 1-2 days).

Command:

```bash
cd /home/quanth/working_space

PROJECT_DIR=/home/quanth/working_space/Wild-Diffusion \
./scripts/slurm_workflow.sh submit wd-cifar10-paper-exact -- \
env ENV_MODE=conda EXPECTED_CONDA_ENV=quanth STRICT_CONDA_ENV=1 \
    INSTALL_DEPS=0 DATASET_ONLY=0 CIFAR_ALLOW_DOWNLOAD=0 \
    DATA_ROOT=/mnt/data/quanth/datasets \
    CIFAR_DIR=/mnt/data/quanth/datasets/cifar10-32x32 \
    OUTDIR=/mnt/data/quanth/experiments/wild-diffusion \
    DURATION_MIMG=200 BATCH=1024 BATCH_GPU=1024 FP16=1 LR=1e-5 WORKERS=16 \
    CIFAR_TRAIN_PERCENT=20 CIFAR_TRAIN_SEED=0 \
    WDRO_WARMUP_RATIO=0.2 WDRO_M_EPOCHS=20 WDRO_K=5 WDRO_STEP_SIZE=0.01 WDRO_GAMMA=1.0 WDRO_P_ADV=1.0 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    bash scripts/setup_and_train_cifar10.sh
```

### `fast_debug_20pct` (quick experiment + evaluation)

Purpose:
- Finish in about 1-2 hours and still produce meaningful checkpoints/metrics for sanity test and pipeline evaluation.

Expected runtime on 1x H100:
- About 1-2 hours (depends on queue and WDRO update cost).

Command:

```bash
cd /home/quanth/working_space

PROJECT_DIR=/home/quanth/working_space/Wild-Diffusion \
./scripts/slurm_workflow.sh submit wd-cifar10-fast-debug -- \
env ENV_MODE=conda EXPECTED_CONDA_ENV=quanth STRICT_CONDA_ENV=1 \
    INSTALL_DEPS=0 DATASET_ONLY=0 CIFAR_ALLOW_DOWNLOAD=0 \
    DATA_ROOT=/mnt/data/quanth/datasets \
    CIFAR_DIR=/mnt/data/quanth/datasets/cifar10-32x32 \
    OUTDIR=/mnt/data/quanth/experiments/wild-diffusion \
    DURATION_MIMG=8 BATCH=1024 BATCH_GPU=512 FP16=1 LR=1e-4 WORKERS=16 \
    CIFAR_TRAIN_PERCENT=20 CIFAR_TRAIN_SEED=0 \
    WDRO_WARMUP_RATIO=0.2 WDRO_M_EPOCHS=80 WDRO_K=2 WDRO_STEP_SIZE=0.01 WDRO_GAMMA=1.0 WDRO_P_ADV=0.3 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    bash scripts/setup_and_train_cifar10.sh
```

## What Was Changed (to support profiles)

- Added WDRO CLI params in `train.py`:
  - `--wdro-warmup-ratio`, `--wdro-m-epochs`, `--wdro-k`, `--wdro-step-size`, `--wdro-gamma`, `--wdro-p-adv`
- Wired WDRO params into `training/training_wdro_loop.py`:
  - Warmup start and interval are now configurable.
  - Inner attack uses configurable `K/step/gamma`.
  - Adversarial sampling probability is configurable (`wdro_p_adv`).
- Updated WDRO transport penalty to paper form:
  - `c(x,x0) = 0.5 * ||x-x0||^2`
- Updated `scripts/setup_and_train_cifar10.sh`:
  - Supports `BATCH_GPU`, `FP16`, and all WDRO profile env vars.
  - Forwards these values to `train.py`.
- Updated this README with copy-paste profile commands.

## Requirement

This repo requires NVIDIA CUDA (the training code forces NCCL/CUDA in distributed init).

## Few-shot Transfer

For Section 4.3-style few-shot transfer (pretrained backbone + fine-tune/adapt, FID 5k protocol), see:

- `docs/FEWSHOT_TRANSFER_REPRO_GUIDE.md`

## Evaluation Pipeline

For one-command CIFAR-10 evaluation (generate + FID + exported JSON/TXT logs), see:

- `scripts/README_cifar10_eval_one_command.md`
