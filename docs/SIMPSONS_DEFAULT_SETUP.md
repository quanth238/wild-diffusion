# Simpsons Default Setup

This is the locked active setup for Simpsons-MNIST RGB work in this repo.

## Locked Simpsons Default

- Label: `warm=5%, aw=0.30, cw=1.00, rho=4.00, N=64`
- Source of truth:
  - `toy/scripts/run_simpsons_mnist_once.sh`
  - `toy/scripts/collect_three_method_seed_data.py`
  - `toy/scripts/run_simpsons_locked_default_percent_sweep.sh`
- Shared knob defaults live in `toy/config.py`, but the active Simpsons method choice is locked by the Simpsons-specific wrappers and collectors.
- This is the locked Simpsons configuration used by the active percent sweep launcher.
- The generic toy CLI still carries older method defaults for historical workflows.
- Simpsons wrappers and collectors should pass `method_version=cdro` explicitly.
- Historical note:
  - Older Simpsons artifacts in `toy_outputs/` used `rho=0.01`.
  - Those artifacts are still useful as historical references, but they are not the current locked Simpsons defaults.

## Historical Reference Artifact

- Case id: `default_warm05_aw0p30_cw1p00_rho0p01_n64`
- Historical family summary:
  - `toy_outputs/simpsons_mnist_rgb_three_method_1pct_ablation_default_n64_rho_stress_20260407/summary/simpsons_mnist_rgb_three_method_1pct_ablation_default_n64_rho_stress_20260407_summary.json`
  - `toy_outputs/simpsons_mnist_rgb_three_method_1pct_ablation_default_n64_rho_stress_20260407/summary/simpsons_mnist_rgb_three_method_1pct_ablation_default_n64_rho_stress_20260407_case_method_summary.csv`
- Historical manifest:
  - `toy_outputs/simpsons_mnist_rgb_three_method_1pct_ablation_default_n64_rho_stress_20260407/default_warm05_aw0p30_cw1p00_rho0p01_n64/simpsons_mnist_rgb_three_method_1pct_ablation_default_n64_rho_stress_20260407_default_warm05_aw0p30_cw1p00_rho0p01_n64_manifest.json`

## Dataset

- Dataset family: Simpsons-MNIST RGB
- Dataset summary: `toy_data/simpsons_mnist_rgb/summary.json`
- Source repo: `https://github.com/alvarobartt/simpsons-mnist`
- Source commit: `f70f4e60d21a5fac0e6bd99f7879f5751ccef673`
- Image shape: `3 x 28 x 28`
- Full population:
  - train: `8000`
  - test: `2000`
- Current percent-sweep convention:
  - train pool is derived from the full train population of `8000`
  - val pool: `2000`
  - split seed: `0`
- Historical 1% artifact subset:
  - train pool: `80`
  - val pool: `2000`
  - train fraction resolved: `0.01`
  - split seed: `0`
- Paths:
  - train root: `toy_data/simpsons_mnist_rgb/imagefolder/train`
  - test root: `toy_data/simpsons_mnist_rgb/imagefolder/test`
  - FID ref: `toy_data/simpsons_mnist_rgb/fid_refs/simpsons_mnist_rgb_test_28x28.npz`

## Model And Training

- `method_version=cdro`
- Canonical path for this choice:
  - `toy/scripts/run_simpsons_mnist_once.sh`
  - `toy/scripts/collect_three_method_seed_data.py`
  - `toy/scripts/run_simpsons_locked_default_percent_sweep.sh`
- `dataset_kind=image_folder`
- `model_kind=image_conv`
- `diagnostics_kind=image_basic`
- `training_objective=edm`
- `hidden_dim=64`
- `batch_size=256`
- `steps` are compared on a shared weighted-compute grid, not by one fixed step count
- `inner_steps=1`
- `outer_attack_weight=0.3`
- `outer_clean_weight=1.0`
- `baseline_gate_enabled=false`
- `compute_fid=true`
- `fid_samples=2000`

## CDRO Knobs

- `cdro_step_size=0.02`
- `cdro_total_budget_rho=4.0`
- `cdro_time_horizon=1.0`
- `cdro_warmup_fraction=0.05`
- `n_steps_path=64`
- effective CDRO path steps in accounting: `64`

## Comparison Protocol

- Protocol name: `three_method_single_trajectory_weighted_grid`
- Seeds in the locked artifact: `0`
- Shared grid template: `wdro_dense_41_relative`
- Default FID-eval template: `weighted_balanced_20_relative`
- Default baseline FID mode: `posthoc_from_checkpoints`
- Default WDRO/CDRO FID mode: `posthoc_from_checkpoints`
- Primary metric: `weighted_compute_units`
- Secondary metric: `train_wall_clock_sec`
- Default wall-clock accounting: normalize batch-equivalent training compute to the current optimized baseline speed with `--wall-clock-mode current_sec_per_kimg`
- Current normalized wall-clock calibration: `0.629646 sec/kimg`
- Shared weighted cap: `200000`
- Dense checkpoints are still saved on the shared comparison grid, plus exact warmup-support checkpoints.
- The launcher now computes FID only on the coarser balanced knot subset by default, with extra transition sentinels for WDRO/CDRO at the exact warmup-support checkpoint plus the first three robust checkpoints.
- The launcher now defaults all three methods to two-phase collection: save dense checkpoints first, then fill only the selected FIDs later from checkpoints.
- Omit the new template flag to keep the old evaluate-every-comparison-knot behavior, set `--baseline-fid-mode in_run` to restore baseline in-run FID, and set `--robust-fid-mode in_run` to restore the old WDRO/CDRO in-run FID path.
- Use `--wall-clock-mode observed` if you explicitly want the old run-time-measured wall-clock instead of the normalized current-speed estimate.
- Weighted-compute calibration:
  - `toy_outputs/compute_calibration/simpsons_mnist_rgb_image_conv_edm_b256_h64_cuda.json`

## Historical Artifact Pointers

- Historical family summary:
  - `toy_outputs/simpsons_mnist_rgb_three_method_1pct_ablation_default_n64_rho_stress_20260407/summary/`
- Targeted follow-up family summary:
  - `toy_outputs/simpsons_mnist_rgb_three_method_1pct_ablation_targeted_followup_ckptreuse_20260406/summary/`
- Example historical metrics file:
  - `toy_outputs/simpsons_mnist_rgb_three_method_1pct_ablation_default_n64_rho_stress_20260407/default_warm05_aw0p30_cw1p00_rho0p01_n64/cdro/s0/simpsons_mnist_rgb_three_method_1pct_ablation_default_n64_rho_stress_20260407_default_warm05_aw0p30_cw1p00_rho0p01_n64_cdro_s0_st3142/metrics.json`

## Canonical Commands

Prepare the dataset:

```bash
python toy/scripts/setup_simpsons_mnist_rgb.py
```

Run one smoke experiment:

```bash
toy/scripts/run_simpsons_mnist_once.sh
```

The smoke wrapper now defaults to `RUN_MODE=robust` and `METHOD_VERSION=cdro`. Override only if you intentionally want baseline-only or a historical method:

```bash
METHOD_VERSION=cdro toy/scripts/run_simpsons_mnist_once.sh
```

The smoke wrapper is intentionally smaller than the full protocol. It keeps the locked CDRO method knobs, but uses smoke-scale training length and eval size.

Run the locked three-method default family:

```bash
python toy/scripts/collect_three_method_seed_data.py \
  --outdir toy_outputs/simpsons_mnist_rgb_three_method_locked_default \
  --prefix simpsons_mnist_rgb_three_method_locked_default \
  --seeds 0 \
  --device cuda \
  --grid-template denser \
  --dataset-path toy_data/simpsons_mnist_rgb/imagefolder/train \
  --dataset-val-path toy_data/simpsons_mnist_rgb/imagefolder/test \
  --fid-ref-path toy_data/simpsons_mnist_rgb/fid_refs/simpsons_mnist_rgb_test_28x28.npz \
  --weighted-compute-calibration-path toy_outputs/compute_calibration/simpsons_mnist_rgb_image_conv_edm_b256_h64_cuda.json \
  --image-size 28 \
  --image-channels 3 \
  --image-train-size 80 \
  --image-val-size 2000 \
  --train-percent-label 1% \
  --image-split-seed 0 \
  --batch-size 256 \
  --hidden-dim 64 \
  --eval-samples 2000 \
  --fid-samples 2000 \
  --wall-clock-mode current_sec_per_kimg \
  --wall-clock-sec-per-kimg 0.629646 \
  --debug-eval-batch 64 \
  --debug-terminal-step 20 \
  --log-every 200 \
  --n-steps-path 64 \
  --fid-eval-template balanced \
  --baseline-fid-mode posthoc_from_checkpoints \
  --robust-fid-mode posthoc_from_checkpoints \
  --outer-attack-weight 0.3 \
  --outer-clean-weight 1.0 \
  --wdro-warmup-fraction 0.05 \
  --cdro-step-size 0.02 \
  --cdro-total-budget-rho 4.0 \
  --cdro-time-horizon 1.0 \
  --cdro-warmup-fraction 0.05 \
  --cdro-n-steps-path 64 \
  --shared-weighted-cap 200000
```

## Notes

- The locked Simpsons default is `cdro`, but not every generic toy wrapper uses that unless it passes `--method-version` explicitly.
- Historical Simpsons artifacts in `toy_outputs/` may disagree with the current locked Simpsons defaults.
- Root `run_image_once.sh` and `run_toy_once.sh` are compatibility shims only. Prefer the `toy/scripts/` paths.
- If a narrative doc disagrees with a run artifact, trust the artifact for that run.
