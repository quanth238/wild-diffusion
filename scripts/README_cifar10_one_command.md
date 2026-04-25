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

Override example:

```bash
DURATION_MIMG=8 BATCH=1024 BATCH_GPU=512 LR=1e-4 CIFAR_TRAIN_PERCENT=100 bash scripts/setup_and_train_cifar10.sh
```

Other optional overrides:
- `ARCH=ddpmpp|ncsnpp|adm`
- `PRECOND=wdroedm|advedm`
- `BATCH=<global_batch>`
- `BATCH_GPU=<micro_batch_per_gpu>`
- `FP16=1|0`
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
- `OUTDIR=/path/to/output`
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

No maintained few-shot transfer doc is kept in the active docs set.

Treat older transfer writeups as stale unless they are revalidated and reintroduced.

## Evaluation Pipeline

For one-command CIFAR-10 evaluation (generate + FID + exported JSON/TXT logs), see:

- `scripts/README_cifar10_eval_one_command.md`

## Compute Calibration

The repo now has separate calibration scripts for the CIFAR image stack:

- Weighted compute ratios: `python scripts/calibrate_cifar_weighted_compute.py --out-json training-runs/compute_calibration/cifar10_32x32_ddpmpp_wdroedm_fp16_b1024_h100_weighted.json`
- FLOP calibration for `TFLOPs` / `PFLOPs` manifest columns: `python scripts/calibrate_cifar_flops.py --out-json training-runs/compute_calibration/cifar10_32x32_ddpmpp_wdroedm_fp16_b1024_h100_flops.json`
- Optional Nsight Compute kernel diagnostic: `python scripts/profile_cifar_ncu_flops.py --out-json training-runs/compute_diagnostics/cifar10_32x32_ddpmpp_wdroedm_fp16_b1024_h100_ncu.json`

The FLOP script uses `torch.profiler` operator FLOP accounting on the same three primitive denoiser operations used by the WCU pipeline: `forward_only`, `forward_plus_inputgrad`, and `forward_plus_parambackward`.

Use the FLOP calibration output for paper `TFLOPs` / `PFLOPs`. The Nsight Compute output is a secondary hardware/kernel diagnostic based on selected derived instruction-equivalent metrics inside NVTX-marked ranges; pass it to manifest builders with `--hardware-flop-diagnostic-json` when you want provenance columns in the CSV, not when choosing the main compute axis.
