# Simpsons Default Setup

This is the active Simpsons-MNIST RGB default for CDRO-EDM work in this repo.

## Locked Default

- Label: `simpsons5_cdro_400k_warmup20_rho32_n032_i1_aw100_cw000`
- Method: `method_version=cdro`
- Training objective: `training_objective=edm`
- Dataset: Simpsons-MNIST RGB, 5% train subset
- Seed: `0`
- Batch size: `256`
- Hidden dim: `64`
- EMA eval: official EMA enabled
- Warmup fraction: `0.2`
- Attack steps: `inner_steps=1`
- Outer weights: `outer_attack_weight=1.0`, `outer_clean_weight=0.0`
- CDRO path steps: `cdro_n_steps_path=32`
- CDRO ladder: `cdro_edm_ladder_mode=stochastic_stratified_quantile`, sampled per example
- CDRO step size: `cdro_step_size=0.02`
- CDRO budget: `cdro_total_budget_rho=32.0`

`rho=32` is the current default/provisional anchor. Keep rho easy to override, because `rho=64` is the next comparison target:

```bash
--cdro-total-budget-rho 64
```

Source-of-truth code paths:

- `toy/config.py`
- `toy/scripts/run_simpsons_mnist_once.sh`
- `toy/scripts/collect_three_method_seed_data.py`
- `toy/scripts/run_simpsons_locked_default_percent_sweep.sh`

The generic toy CLI no longer carries old v1/v1.1/v1.2 CDRO-style knobs or `cdro_antithetic_rollouts`. Old collector reuse flags remain for checkpoint and CSV reuse.

## Dataset

- Dataset family: Simpsons-MNIST RGB
- Dataset summary: `toy_data/simpsons_mnist_rgb/summary.json`
- Source repo: `https://github.com/alvarobartt/simpsons-mnist`
- Source commit: `f70f4e60d21a5fac0e6bd99f7879f5751ccef673`
- Image shape: `3 x 28 x 28`
- Full population: train `8000`, test `2000`
- Active train pool: `400` images, which is 5% of the train population
- Active val pool: `2000`
- Split seed: `0`
- Train root: `toy_data/simpsons_mnist_rgb/imagefolder/train`
- Test root: `toy_data/simpsons_mnist_rgb/imagefolder/test`
- FID ref: `toy_data/simpsons_mnist_rgb/fid_refs/simpsons_mnist_rgb_test_28x28.npz`

Historical 1% artifacts used `image_train_size=80`, `train_percent_label=1%`, warmup 5%, clean outer weight 1.0, lower rho values, and usually `N=64`. They remain useful for archaeology, not for current defaults.

## Model And Eval

- `dataset_kind=image_folder`
- `model_kind=image_conv`
- `diagnostics_kind=image_basic`
- `training_objective=edm`
- `hidden_dim=64`
- `batch_size=256`
- `use_ema_eval=true`
- `ema_mode=official`
- `ema_halflife_kimg=500`
- `ema_rampup_ratio=0.05`
- `compute_fid=true` in the Simpsons wrappers and collectors
- `fid_samples=2000`
- `fid_gen_batch=64`

For direct `toy/run_toy.py` runs, `toy/config.py` now defaults to this CDRO image-folder setup. Override `--method-version`, `--dataset-kind`, `--dataset-path`, `--n-steps-path`, or any active CDRO knob explicitly when recreating older runs.

## CDRO Knobs

- `inner_steps=1`
- `attack_num_steps` is optional and remains a legacy-compatible override; unset means use `inner_steps`
- `outer_attack_weight=1.0`
- `outer_clean_weight=0.0`
- `cdro_step_size=0.02`
- `cdro_total_budget_rho=32.0`
- `cdro_time_horizon=1.0`
- `cdro_warmup_fraction=0.2`
- `cdro_edm_ladder_mode=stochastic_stratified_quantile`
- `cdro_per_example_sigma_ladders=true`
- effective CDRO path steps: `32`

## Comparison Protocol

- Protocol name: `three_method_single_trajectory_weighted_grid`
- Seeds: `0`
- Shared grid template: `denser`
- Default FID-eval template: `balanced`
- Baseline FID mode: `posthoc_from_checkpoints`
- WDRO/CDRO FID mode: `posthoc_from_checkpoints`
- Primary metric: `weighted_compute_units`
- Secondary metric: `train_wall_clock_sec`
- Wall-clock mode: `observed`
- Shared weighted cap: `400000`
- Baseline max steps: `160000`
- WDRO max total steps: `160000`
- Weighted-compute calibration: `toy_outputs/compute_calibration/simpsons_mnist_rgb_image_conv_edm_b256_h64_cuda.json`

The collector saves dense checkpoints on the shared grid, then fills selected FIDs from checkpoints. Use `--baseline-fid-mode in_run` or `--robust-fid-mode in_run` only when intentionally restoring the older in-run FID behavior.

## Active Artifact Pointers

- Current rho=32 anchor: `toy_outputs/simpsons5_cdro_400k_warmup20_rho32_n032_cw0_20260414/`
- Rho=64 comparison run area: `toy_outputs/simpsons5_cdro_400k_warmup20_rho64_n032_cw0_20260414/`
- Earlier nrho sweep reference: `toy_outputs/simpsons5_cdro_400k_warmup20_nrho_20260413/`

Historical artifact roots:

- `toy_outputs/simpsons_mnist_rgb_three_method_1pct_ablation_default_n64_rho_stress_20260407/summary/`
- `toy_outputs/simpsons_mnist_rgb_three_method_1pct_ablation_targeted_followup_ckptreuse_20260406/summary/`

## Canonical Commands

Prepare the dataset:

```bash
python toy/scripts/setup_simpsons_mnist_rgb.py
```

Run one smoke experiment:

```bash
toy/scripts/run_simpsons_mnist_once.sh
```

The smoke wrapper uses the active CDRO method knobs, but keeps smoke-scale `STEPS` and `EVAL_SAMPLES`.

Run the locked 5% three-method family with explicit knobs:

```bash
python toy/scripts/collect_three_method_seed_data.py \
  --outdir toy_outputs/simpsons5_cdro_400k_warmup20_rho32_n032_cw0_default \
  --prefix simpsons5_cdro_400k_warmup20_rho32_n032_i1_aw100_cw000_default \
  --seeds 0 \
  --device cuda \
  --amp-dtype auto \
  --use-ema-eval \
  --ema-mode official \
  --ema-halflife-kimg 500 \
  --ema-rampup-ratio 0.05 \
  --dataset-path toy_data/simpsons_mnist_rgb/imagefolder/train \
  --dataset-val-path toy_data/simpsons_mnist_rgb/imagefolder/test \
  --fid-ref-path toy_data/simpsons_mnist_rgb/fid_refs/simpsons_mnist_rgb_test_28x28.npz \
  --weighted-compute-calibration-path toy_outputs/compute_calibration/simpsons_mnist_rgb_image_conv_edm_b256_h64_cuda.json \
  --image-size 28 \
  --image-channels 3 \
  --image-train-size 400 \
  --image-val-size 2000 \
  --train-percent-label 5% \
  --image-split-seed 0 \
  --batch-size 256 \
  --hidden-dim 64 \
  --training-objective edm \
  --eval-samples 2000 \
  --fid-samples 2000 \
  --fid-gen-batch 64 \
  --debug-eval-batch 64 \
  --debug-terminal-step 20 \
  --log-every 200 \
  --n-steps-path 64 \
  --shared-weighted-cap 400000 \
  --grid-template denser \
  --fid-eval-template balanced \
  --baseline-fid-mode posthoc_from_checkpoints \
  --robust-fid-mode posthoc_from_checkpoints \
  --baseline-max-steps 160000 \
  --wdro-max-total-steps 160000 \
  --robust-warmup-mode shared_exact \
  --wdro-warmup-fraction 0.2 \
  --inner-steps 1 \
  --outer-attack-weight 1.0 \
  --outer-clean-weight 0.0 \
  --cdro-step-size 0.02 \
  --cdro-total-budget-rho 32.0 \
  --cdro-time-horizon 1.0 \
  --cdro-edm-ladder-mode stochastic_stratified_quantile \
  --cdro-warmup-fraction 0.2 \
  --cdro-n-steps-path 32
```

Use the launcher shortcut for the same default lane:

```bash
toy/scripts/run_simpsons_locked_default_percent_sweep.sh
```

Set `PERCENTS="5 10 20 50 100"` only when intentionally running a percent sweep beyond the active 5% lane.

## Notes

- Do not infer current defaults from old output paths alone; trust each artifact manifest for that run.
- Root-level wrappers are compatibility shims. Prefer the `toy/scripts/` paths.
- If a narrative doc disagrees with a run artifact, trust the artifact for that run.
