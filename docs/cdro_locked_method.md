# Locked CDRO Method

This file records the practical plug-in CDRO method that is locked for paper-facing runs.

## Scope

- Mainline method: `PRECOND=cdroedm`
- Interpretation: path-space inspired, random-time local robust surrogate on noisy states
- Not the same as the full image-level Markov CDRO branch

## Default practical recipe

- weak robust mixing
- noise-space perturbation
- sigma-banded activation
- warm-up / ramp for from-scratch runs
- baseline-initialized fine-tune for continuation runs

## Current CIFAR-10 starting preset

- `CDRO_MIX=0.05`
- `CDRO_ADV_STEPS=1`
- `CDRO_STEP_SIZE=0.02`
- `CDRO_MAX_DELTA=0.03`
- `CDRO_RHO=2e-5`
- `CDRO_LAMBDA_INIT=0.001`
- `CDRO_LAMBDA_LR=5e-4`
- `CDRO_SIGMA_FLOOR=0.12`
- `CDRO_SIGMA_CUT=0.70`
- `CDRO_GATE_POWER=1.0`
- `CDRO_DELTA_SPACE=noise`

## Warm-up

For from-scratch CIFAR-10 screens:

- `CDRO_START_KIMG=1600`
- `CDRO_RAMP_KIMG=1600`

This is an optimization schedule, not part of the theorem.

## Scientific framing

- Exact full path-space Markov CDRO remains exploratory at image scale.
- The locked method is the scalable surrogate used for paper-facing experiments.
- From-scratch and continuation protocols should be reported separately.
