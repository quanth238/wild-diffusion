# Archived Plug-in CDRO Method

This file records the status of the practical plug-in CDRO method behind `PRECOND=cdroedm`.

It remains in the repo for archival comparisons, regression checks, and perturbation debugging, but it is no longer the recommended paper-facing or mainline CDRO path.

## Scope

- Archived method: `PRECOND=cdroedm`
- Interpretation: path-space inspired, random-time local robust surrogate on noisy states
- Not the same as the full image-level Markov CDRO branch

## Decision as of 2026-04-02

- Do not start new sweeps from `PRECOND=cdroedm` by default.
- Keep it only for ablations, controls, telemetry checks, and debugging.
- Prefer more theory-faithful CDRO directions for new research work.

## Why this was de-prioritized

- MNIST 0.1 MIMG smoke test before repair, baseline: `1.5242 sec/kimg`, loss `1.367120`, FID `343.402`.
- MNIST 0.1 MIMG smoke test before repair, `cdro_on`: `4.0003 sec/kimg`, loss `1.369090`, FID `343.386`.
- A bounded repair fixed an RMS-vs-L2 step-scaling bug in `EDMLossCDRO`, so the perturbation no longer collapsed for purely geometric reasons.
- MNIST 0.1 MIMG smoke test after repair, patched `cdro_on`: `4.0039 sec/kimg`, loss `1.369449`, FID `343.437`.
- The repair increased perturbation magnitude, but average `CDRO/radius_utilization` on the patched `cdro_on` smoke run was still only about `1.68%`, and the end metrics still did not improve.

## Last bounded repair that was kept

- `adv_step_size` in `EDMLossCDRO` is now interpreted in per-sample RMS units instead of raw L2 units.
- `stats.jsonl` now logs `CDRO/radius_utilization` and `CDROMarkov/radius_utilization`.
- `scripts/analyze_mnist_cdro_perturbations.py` now reconstructs the loss from `training_options.json` first so local loss-code edits are reflected when analyzing older snapshots.

## Historical practical recipe

- weak robust mixing
- noise-space perturbation
- sigma-banded activation
- warm-up / ramp for from-scratch runs
- baseline-initialized fine-tune for continuation runs

## Historical CIFAR-10 starting preset

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

For from-scratch CIFAR-10 screens, the historical schedule was:

- `CDRO_START_KIMG=1600`
- `CDRO_RAMP_KIMG=1600`

This is an optimization schedule, not part of the theorem.

## What is still useful

- historical ablations against baseline or WDRO
- regression checks for `CDRO/*` telemetry and perturbation plumbing
- confirming whether a future CDRO variant is genuinely stronger than this weak plug-in surrogate

## Scientific framing

- Exact full path-space Markov CDRO remains exploratory at image scale.
- This plug-in method is a weak scalable surrogate that is now treated as archival.
- From-scratch and continuation protocols should be reported separately.
