# Simpsons Default Setup

This is the locked active setup for Simpsons-MNIST RGB work in this repo.

## Locked Default Case

- Case id: `default_warm05_aw0p30_cw1p00_rho0p01_n64`
- Label: `warm=5%, aw=0.30, cw=1.00, rho=0.01, N=64`
- Family summary:
  - `toy_outputs/simpsons_mnist_rgb_three_method_1pct_ablation_default_n64_rho_stress_20260407/summary/simpsons_mnist_rgb_three_method_1pct_ablation_default_n64_rho_stress_20260407_summary.json`
  - `toy_outputs/simpsons_mnist_rgb_three_method_1pct_ablation_default_n64_rho_stress_20260407/summary/simpsons_mnist_rgb_three_method_1pct_ablation_default_n64_rho_stress_20260407_case_method_summary.csv`
- Canonical manifest:
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
- Active resolved subset in the locked default run:
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
- `cdro_total_budget_rho=0.01`
- `cdro_time_horizon=1.0`
- `cdro_warmup_fraction=0.05`
- `n_steps_path=64`
- effective CDRO path steps in accounting: `64`

## Comparison Protocol

- Protocol name: `three_method_single_trajectory_weighted_grid`
- Seeds in the locked artifact: `0`
- Shared grid template: `wdro_dense_41_relative`
- Primary metric: `weighted_compute_units`
- Secondary metric: `train_wall_clock_sec`
- Shared weighted cap: `200000`
- Weighted-compute calibration:
  - `toy_outputs/compute_calibration/simpsons_mnist_rgb_image_conv_edm_b256_h64_cuda.json`

## Current Default Artifact Pointers

- Default family summary:
  - `toy_outputs/simpsons_mnist_rgb_three_method_1pct_ablation_default_n64_rho_stress_20260407/summary/`
- Targeted follow-up family summary:
  - `toy_outputs/simpsons_mnist_rgb_three_method_1pct_ablation_targeted_followup_ckptreuse_20260406/summary/`
- Example default-case metrics file:
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
  --debug-eval-batch 64 \
  --debug-terminal-step 20 \
  --log-every 200 \
  --n-steps-path 64 \
  --outer-attack-weight 0.3 \
  --outer-clean-weight 1.0 \
  --wdro-warmup-fraction 0.05 \
  --cdro-step-size 0.02 \
  --cdro-total-budget-rho 0.01 \
  --cdro-time-horizon 1.0 \
  --cdro-warmup-fraction 0.05 \
  --cdro-n-steps-path 64 \
  --shared-weighted-cap 200000
```

## Notes

- The locked default is a Simpsons-only working context.
- If a narrative doc disagrees with a manifest or `metrics.json`, trust the artifact file.
- The next validation step after this locked default is multi-seed reruns on the same setup.
