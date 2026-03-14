# One-Command CIFAR-10 Evaluation (Generate + FID + Export)

From repo root, run:

```bash
bash scripts/setup_and_eval_cifar10.sh
```

This single command will:
1. Reuse active conda env (or create/use venv fallback).
2. Install/validate Python dependencies.
3. Resolve the target network snapshot (`NETWORK_PKL`, `RESUME_STATE`, or latest in `RUN_DIR`).
4. Generate images with `generate.py`.
5. Compute FID with `fid.py`.
6. Export result files:
   - `evaluation_result.json`
   - `evaluation_result.txt`
   - `generate.log`, `fid.log` (and `ref.log` if building local ref stats)

## Quick start for your completed run

Use the latest snapshot in your run directory:

```bash
RUN_DIR=/mnt/data/quanth/experiments/wild-diffusion/00006-cifar10-32x32-cond-ddpmpp-wdroedm-gpus1-batch1024-fp16 \
ENV_MODE=conda EXPECTED_CONDA_ENV=quanth STRICT_CONDA_ENV=1 INSTALL_DEPS=0 \
NUM_IMAGES=50000 GEN_BATCH=128 FID_BATCH=64 GEN_STEPS=35 \
REF_MODE=compute CIFAR_DIR=/mnt/data/quanth/datasets/cifar10-32x32 \
bash scripts/setup_and_eval_cifar10.sh
```

## Resolve snapshot from training-state

If you only have `training-state-*.pt`:

```bash
RESUME_STATE=/mnt/data/quanth/experiments/wild-diffusion/00006-cifar10-32x32-cond-ddpmpp-wdroedm-gpus1-batch1024-fp16/training-state-200000.pt \
ENV_MODE=conda EXPECTED_CONDA_ENV=quanth STRICT_CONDA_ENV=1 INSTALL_DEPS=0 \
NUM_IMAGES=50000 REF_MODE=compute CIFAR_DIR=/mnt/data/quanth/datasets/cifar10-32x32 \
bash scripts/setup_and_eval_cifar10.sh
```

The script auto-maps:
- `training-state-XXXXXX.pt` -> `network-snapshot-XXXXXX.pkl`

## Output location

Default output folder:
- `${RUN_DIR}/eval/eval-YYYYmmdd-HHMMSS`

Override with:
- `EVAL_ROOT=/path/to/evals`
- `EVAL_TAG=my-eval-tag`

## Important options

- `RUN_DIR`: training run directory containing `network-snapshot-*.pkl`
- `NETWORK_PKL`: explicit snapshot path (highest priority)
- `RESUME_STATE`: `training-state-*.pt` to auto-resolve matching snapshot
- `NUM_IMAGES`: number of generated images for FID (default `50000`)
- `SEED_START`: start seed (default `0`, seeds become `SEED_START..SEED_START+NUM_IMAGES-1`)
- `GEN_BATCH`: generation batch size (default `128`)
- `FID_BATCH`: FID batch size (default `64`)
- `GEN_STEPS`: sampler steps for `generate.py` (default `35`)
- `CLASS_IDX`: optional fixed class index (default empty = random classes)
- `NPROC_PER_NODE`: number of GPUs used by `torchrun` (default `1`)
- `REF_MODE=auto|compute|url|path`:
  - `compute`: build local reference from `CIFAR_DIR` (recommended on restricted clusters)
  - `url`: use NVIDIA CIFAR-10 ref URL
  - `path`: use existing `REF_PATH`
  - `auto`: prefer local `REF_PATH`, else compute if CIFAR exists, else URL
- `REF_PATH`: local `.npz` path (default `${DATA_ROOT}/fid-refs/cifar10-32x32.npz`)
- `CIFAR_DIR`: CIFAR-10 imagefolder path with `dataset.json`

Environment/setup options (same style as training script):
- `ENV_MODE=auto|conda|venv`
- `INSTALL_DEPS=0|1|auto`
- `EXPECTED_CONDA_ENV`, `STRICT_CONDA_ENV`
- `INSTALL_ONLY=1` (stop after env/deps)
- `PREPARE_DATASET=auto|0|1`
- `CIFAR_ALLOW_DOWNLOAD=0|1`

## Notes

- For stable comparison with paper-style protocol, keep:
  - `NUM_IMAGES=50000`
  - default random-class generation (leave `CLASS_IDX` empty)
- If worker nodes have no internet:
  - use `REF_MODE=compute`
  - set `CIFAR_ALLOW_DOWNLOAD=0` and ensure `CIFAR_DIR` already exists.

## Offline Inception Detector (recommended on your cluster)

If the NGC detector URL is blocked/unreliable, upload the detector once and force local loading.

1. Upload from local machine to server storage:

```bash
scp "$HOME/Downloads/inception-2015-12-05.pkl" \
  vinuni:/mnt/data/quanth/models/inception-2015-12-05.pkl
```

2. Submit eval with local detector path:

```bash
cd /home/quanth/working_space

PROJECT_DIR=/home/quanth/working_space/Wild-Diffusion \
./scripts/slurm_workflow.sh submit wd-cifar10-eval-final -- \
env ENV_MODE=conda EXPECTED_CONDA_ENV=quanth STRICT_CONDA_ENV=1 INSTALL_DEPS=0 \
    RUN_DIR=/mnt/data/quanth/experiments/wild-diffusion/00006-cifar10-32x32-cond-ddpmpp-wdroedm-gpus1-batch1024-fp16 \
    CIFAR_DIR=/mnt/data/quanth/datasets/cifar10-32x32 \
    REF_MODE=compute NUM_IMAGES=50000 GEN_BATCH=128 FID_BATCH=64 GEN_STEPS=35 NPROC_PER_NODE=1 \
    FID_DETECTOR_PATH=/mnt/data/quanth/models/inception-2015-12-05.pkl \
    bash scripts/setup_and_eval_cifar10.sh
```

When `FID_DETECTOR_PATH` is set, `fid.py` loads that local file directly and does not call the remote detector URL.
