# CIFAR-10 Paper-First Protocol

This note records the first paper-faithful CIFAR-10 target for the image-scale lane.

## Goal

First reproduce the unconditional `EDM-DDPM++` baseline row from Table 1 of the ICLR 2026 WILD-Diffusion paper before running `wdro` and `cdro`.

Initial target:

- Dataset: CIFAR-10 `(32 x 32)`
- Data fraction: `20%`
- Method row: `EDM-DDPM++ (Karras et al., 2022)`
- Target FID: `13.91`

## Source Precedence

Use settings in this order:

1. WILD-Diffusion ICLR 2026 paper and appendix
2. Official EDM repo only when the WILD paper leaves a field unspecified
3. Local repo code only for execution mechanics, not for protocol authority

## Resolved Settings For The First Run

These settings are fixed for the first baseline reproduction.

- Dataset split: random `20%` subset of CIFAR-10 training data
- Local dataset path: `datasets/cifar10-32x32-train20pct-seed0`
- Resolution: `32 x 32`
- Conditioning: unconditional (`--cond=0`)
- Backbone: `DDPM++` (`--arch=ddpmpp`)
- Training duration: `200 Mimg`
- Minibatch size: `1024`
- Learning rate: `1e-5`
- FID sample count: `50,000`
- FID reference: the full CIFAR-10 training set reference statistics

For later `wdro` runs, the WILD paper specifies the following defaults:

- Warmup ratio: `20%` of total training epochs
- WDRO refresh interval `m`: `20` epochs
- WDRO inner steps `K`: `5`
- WDRO step size: `0.01`
- WDRO gamma: `1`

## EDM-Inherited Defaults

The WILD paper does not restate every low-level CIFAR training and sampling default, so inherit the missing pieces from the official EDM setup.

Use these inherited defaults for the first baseline run:

- Sampler: deterministic EDM sampler
- Sampling steps: `18`
- FID aggregation: run FID `3` times and report the minimum
- Dropout: `0.13`
- Augment probability: `0.12`
- EMA half-life: `0.5 Mimg`
- X-flips: disabled
- Mixed precision: disabled unless we intentionally choose a different hardware-specific reproduction profile later

## Local Code Mapping

In this repo, the baseline should run through the standard `EDMLossWdro` path with WDRO disabled.

- Use `--precond=wdroedm`
- Disable WDRO by setting `--wdro-warmup-ratio=1.0`

This is necessary because `training.loss.EDMLossWdro` is the plain EDM loss, while the WDRO behavior is introduced by `training/training_wdro_loop.py`.

## Important Local Mismatches To Override

Do not trust local defaults for the paper run unless they match this document.

- `train.py` currently defaults to `--wdro-warmup-ratio=0.4`
- `train.py` currently defaults to `--wdro-m-epochs=100`
- `train.py` currently defaults to `--wdro-k=2`
- `train.py` currently defaults to `--wdro-step-size=1e-3`
- `train.py` currently defaults to `--batch=512`
- `scripts/setup_and_eval_cifar10.sh` currently defaults to `GEN_STEPS=35`

## Exact First Training Command

This is the first paper-faithful baseline command to run locally.

```bash
torchrun --standalone --nproc_per_node=1 train.py \
  --outdir=training-runs/paper-cifar10-baseline \
  --data=datasets/cifar10-32x32-train20pct-seed0 \
  --cond=0 \
  --arch=ddpmpp \
  --precond=wdroedm \
  --duration=200 \
  --batch=1024 \
  --lr=1e-5 \
  --ema=0.5 \
  --dropout=0.13 \
  --augment=0.12 \
  --xflip=0 \
  --fp16=0 \
  --wdro-warmup-ratio=1.0 \
  --desc=paper-cifar10-uncond-ddpmpp-baseline-20pct
```

## Exact First Endpoint Evaluation Protocol

For paper-compatible endpoint reporting:

1. Generate `50,000` images with `18` deterministic EDM steps.
2. Compute FID against `datasets/fid-refs/cifar10-32x32.npz`.
3. Repeat three times with disjoint seed ranges.
4. Report the minimum of the three FIDs.

Suggested seed ranges:

- `0-49999`
- `50000-99999`
- `100000-149999`

Example for one repeat:

```bash
torchrun --standalone --nproc_per_node=1 generate.py \
  --network=PATH_TO_NETWORK_PKL \
  --outdir=eval/paper-cifar10-baseline-20pct/fid-seeds-0-49999 \
  --seeds=0-49999 \
  --subdirs \
  --batch=128 \
  --steps=18

torchrun --standalone --nproc_per_node=1 fid.py calc \
  --images=eval/paper-cifar10-baseline-20pct/fid-seeds-0-49999 \
  --ref=datasets/fid-refs/cifar10-32x32.npz \
  --num=50000 \
  --seed=0 \
  --batch=64
```

## Notes

- The paper states that low-resolution experiments use `DDPM++` as the default backbone.
- The paper states that FID is computed between `50k` generated samples and the full set of training images.
- The paper appendix explicitly sets the WDRO warmup stage to `20%` of total training epochs.
- The paper appendix gives CIFAR-10 hyperparameters for `20% / 100%` data, but does not explicitly spell out the `50%` learning rate. Do not lock the `50%` run until that gap is resolved.
- The repo currently has the full CIFAR-10 dataset plus `20%` and `50%` subsets on disk:
  `datasets/cifar10-32x32-train20pct-seed0` and `datasets/cifar10-32x32-train50pct-seed0`.
