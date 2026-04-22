# Toy Script Inventory

This folder is the canonical script surface for the toy-side workflow.

## Active Simpsons-MNIST RGB Path

These are the scripts that match the current working decision in this repo:

- `setup_simpsons_mnist_rgb.py`: stage the RGB dataset under `toy_data/simpsons_mnist_rgb/` and build the FID ref.
- `run_simpsons_mnist_once.sh`: single Simpsons smoke run. Defaults to `RUN_MODE=robust` and `METHOD_VERSION=cdro`, with smoke-scale training/eval sizes but the locked CDRO method knobs.
- `collect_three_method_seed_data.py`: canonical family-aware weighted-grid collector for EDM/RF baselines and robust methods.
- `plot_three_method_family_compare_from_runs.py`: discover one or more combined CSV buckets, keep colliding series source-distinct when needed, and render the family-aware compare plots.
- `run_simpsons_locked_default_percent_sweep.sh`: locked percent sweep for the current Simpsons defaults.
- `plot_three_method_fid_curves.py`: plotting for the collected three-method runs.
- `reevaluate_three_method_fids_from_checkpoints.py`: re-run FID from saved checkpoints when needed; it records `edm_clean_probe` for EDM checkpoints, `rf_clean_probe` for RF checkpoints, and `objective_clean_probe` as the family-native alias.
- `materialize_toy_pngs.py`: assemble checked-in PNG summaries from collected outputs.

Current CDRO-EDM default lane: Simpsons-MNIST RGB 5%, seed 0, batch size 256, hidden dim 64, official EMA, warmup fraction 0.2 with `robust_warmup_mode=shared_exact`, `inner_steps=1`, `outer_attack_weight=1.0`, `outer_clean_weight=0.0`, `cdro_n_steps_path=32`, stochastic stratified per-example quantile ladders, `cdro_step_size=0.02`, and `cdro_total_budget_rho=0.001`. Override rho explicitly for follow-up runs.

Current comparison naming convention:

- `Baseline EDM`
- `Wild-Diffusion`
- `CDRO-EDM`
- `RF`
- `CDRO-RF`
- `Wild-Diffusion-RF`

Direct `Wild-Diffusion-RF` runs are supported through `toy/run_toy.py`, resuming RF robust training from the shared EDM checkpoint and ignoring `wdro_warmup_fraction`. The shared-grid collector still skips WDRO when `training_objective=rf`.

RF-family support is now explicitly reflow-only: legacy `rf_baseline_mode=plain` and `rf_cdro_pair_source=data_noise` are no longer part of the supported public surface.

For the EDM family, the canonical shared-grid flow is still `shared_exact`: one shared baseline warmup checkpoint across methods, then nearby same-method continuation checkpoints for later loss/FID reevaluation.

## Shared Wrappers

These are generic helpers, not the source of truth for the Simpsons method choice:

- `run_image_once.sh`: generic image-folder wrapper around `toy/run_toy.py`. Pass `METHOD_VERSION` explicitly when method choice matters.
- `run_toy_once.sh`: generic toy-wrapper helper. Pass `METHOD_VERSION` explicitly when method choice matters.
- `setup_cifar_subset_imagefolder.py`: stage a class-filtered CIFAR-10 train/test image_folder dataset plus matching FID ref for low-budget toy image runs.
- `calibrate_cifar_subset_weighted_compute.sh`: build matching EDM/RF weighted-compute calibrations for the CIFAR subset defaults.
- `run_cifar_subset_once.sh`: smoke wrapper for the toy `image_conv` CIFAR subset path. Defaults to a 2-class CIFAR setup and keeps the same lightweight architecture family used by Simpsons.
- `run_cifar_subset_family_compare_pipeline.sh`: run the locked EDM family, branch the RF family from the shared EDM warmup checkpoint, and then emit combined EDM-vs-RF compare plots for the CIFAR subset.

## Historical Or Experimental Scripts

These remain in-tree for reproducibility, but they are not the current default lane:

- `run_simpsons_once.sh`, `setup_simpsons_imagefolder.py`: older Simpsons image-folder setup predating `simpsons_mnist_rgb`.
- `queue_three_method_ablation_matrix.py`, `queue_three_method_1pct_baseline_refresh.sh`: targeted April 2026 ablation orchestration.
- `sweep_cdro_compute_curve.py`, `sweep_wdro_compute_curve.py`: method-specific compute sweeps, not the locked default family.
- `compare_mnist_objectives.py`, `compare_mnist_wild_objectives.py`, `sweep_mnist_convergence.py`, `sweep_mnist_convergence_checkpointed.py`: older MNIST comparison workflows.
- `compare_toy_wild_objectives.py`, `run_stageA_once_and_eval.sh`, `evaluate_stageA.py`: 2D toy or stage-A workflows.
- `compare_v2_vs_v21.py`, `make_v2_v21_panel.py`: archived v2/v2.1 comparison utilities.

## Working Rule

- Prefer `toy/scripts/` over same-named files at repo root.
- Prefer the Simpsons scripts above when the task is about the current active setup.
- Treat everything else here as historical unless a run artifact or issue explicitly points to it.
