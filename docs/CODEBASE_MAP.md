# Codebase Map

This is the shortest reliable map of the repo.

## Start Here

If you are doing toy/MNIST/pathwise-method work, start in:

- [toy/run_toy.py](/root/wild-diffusion/toy/run_toy.py): toy entrypoint
- [toy/app/cli.py](/root/wild-diffusion/toy/app/cli.py): toy CLI to config
- [toy/app/experiment.py](/root/wild-diffusion/toy/app/experiment.py): main toy orchestration
- [toy/versions/registry.py](/root/wild-diffusion/toy/versions/registry.py): resolves `--method-version`

If you are doing image-scale EDM / WILD / WDRO work, start in:

- [train.py](/root/wild-diffusion/train.py): image training entrypoint
- [training/loss.py](/root/wild-diffusion/training/loss.py): loss definitions
- [training/training_loop.py](/root/wild-diffusion/training/training_loop.py): baseline image training loop
- [training/training_wdro_loop.py](/root/wild-diffusion/training/training_wdro_loop.py): WDRO/WILD-style image training loop
- [training/networks.py](/root/wild-diffusion/training/networks.py): EDM network definitions
- [generate.py](/root/wild-diffusion/generate.py): sample from saved image-model checkpoints

## What Lives Where

### `toy/`

Canonical toy research stack.

- [toy/config.py](/root/wild-diffusion/toy/config.py): full toy config dataclass
- [toy/app/cli.py](/root/wild-diffusion/toy/app/cli.py): CLI parser and validation
- [toy/app/experiment.py](/root/wild-diffusion/toy/app/experiment.py): full run pipeline
- [toy/data_backends/provider.py](/root/wild-diffusion/toy/data_backends/provider.py): `toy_gmm`, `mnist`, `image_folder`
- [toy/model_backends/provider.py](/root/wild-diffusion/toy/model_backends/provider.py): model/control builders
- [toy/diagnostics_backends/provider.py](/root/wild-diffusion/toy/diagnostics_backends/provider.py): plots and diagnostics
- [toy/shared/](/root/wild-diffusion/toy/shared): version-agnostic utilities
- [toy/versions/](/root/wild-diffusion/toy/versions): method-specific implementations
- [toy/scripts/](/root/wild-diffusion/toy/scripts): sweeps, comparison scripts, one-command runners

### `toy/versions/`

Method implementations for the toy stack.

- [toy/versions/v1_1/](/root/wild-diffusion/toy/versions/v1_1): current pathwise robust candidate
- [toy/versions/wild/](/root/wild-diffusion/toy/versions/wild): WILD baseline
- [toy/versions/v1_2/](/root/wild-diffusion/toy/versions/v1_2): single-sigma CDRO-style ablation
- [toy/versions/v2/](/root/wild-diffusion/toy/versions/v2): constrained baseline family
- [toy/versions/v2_1/](/root/wild-diffusion/toy/versions/v2_1): v2.1 variant
- [toy/versions/v1/](/root/wild-diffusion/toy/versions/v1): earlier version

Within a method folder, the pattern is:

- `method.py`: method metadata and public API
- `trainer.py`: robust training loop
- `diffusion.py`: rollout / attack logic

### `training/`

Image-scale training stack used by `train.py`.

- [training/loss.py](/root/wild-diffusion/training/loss.py): EDM, adversarial, and WDRO losses
- [training/training_loop.py](/root/wild-diffusion/training/training_loop.py): standard training
- [training/training_wdro_loop.py](/root/wild-diffusion/training/training_wdro_loop.py): WDRO loop
- [training/networks.py](/root/wild-diffusion/training/networks.py): EDM preconditioned nets
- [training/dataset.py](/root/wild-diffusion/training/dataset.py): image dataset loader
- [training/wdro_utils.py](/root/wild-diffusion/training/wdro_utils.py): WDRO helper code
- [training/augment.py](/root/wild-diffusion/training/augment.py): augmentation pipe

### `scripts/`

Mostly shell workflows around the toy stack and some image runs.

Good first places:

- [scripts/slurm_fid_curve_baseline_wild_v11_v12.sh](/root/wild-diffusion/scripts/slurm_fid_curve_baseline_wild_v11_v12.sh): method-vs-step FID sweep
- [scripts/run_budget40k_user_protocol.sh](/root/wild-diffusion/scripts/run_budget40k_user_protocol.sh): heavier comparison protocol
- [scripts/slurm_baseline_v11_v2_v21_panel.sh](/root/wild-diffusion/scripts/slurm_baseline_v11_v2_v21_panel.sh): panel generation workflow

### `docs/`

Project notes, protocol docs, and experiment writeups.

These are useful for context, but not the implementation entrypoint.

### `datasets/`

Input datasets and prebuilt FID references.

### `toy_outputs/` and `training-runs/`

Generated outputs, logs, run artifacts, and checkpoints.

These are outputs, not source code.

## Practical Navigation

If you want to change toy method behavior:

1. Start at [toy/app/experiment.py](/root/wild-diffusion/toy/app/experiment.py).
2. Check [toy/versions/registry.py](/root/wild-diffusion/toy/versions/registry.py).
3. Edit the selected method folder under [toy/versions/](/root/wild-diffusion/toy/versions).

If you want to change toy dataset handling or MNIST subset policy:

1. Start at [toy/data_backends/provider.py](/root/wild-diffusion/toy/data_backends/provider.py).
2. Check FID reference behavior in [toy/export_mnist_fid_ref.py](/root/wild-diffusion/toy/export_mnist_fid_ref.py).
3. Check FID loading in [toy/app/experiment.py](/root/wild-diffusion/toy/app/experiment.py#L1070).

If you want to change toy logging or saved metrics:

1. Start at [toy/app/experiment.py](/root/wild-diffusion/toy/app/experiment.py).
2. Search for `metrics = {` and `flow_debug`, `objective_debug`, `sample_quality_debug`.

If you want to change MNIST sweep protocols:

- [toy/scripts/sweep_mnist_convergence.py](/root/wild-diffusion/toy/scripts/sweep_mnist_convergence.py): from-scratch baseline convergence
- [toy/scripts/compare_fid_curve_methods.py](/root/wild-diffusion/toy/scripts/compare_fid_curve_methods.py): fixed-baseline method comparison

If you want to change image-scale loss behavior:

1. Start at [training/loss.py](/root/wild-diffusion/training/loss.py).
2. Then check [train.py](/root/wild-diffusion/train.py) for how the loss class is selected.
3. Then check the relevant training loop in [training/training_loop.py](/root/wild-diffusion/training/training_loop.py) or [training/training_wdro_loop.py](/root/wild-diffusion/training/training_wdro_loop.py).

## Canonical Call Paths

### Toy run

`toy/run_toy.py`

-> [toy/app/cli.py](/root/wild-diffusion/toy/app/cli.py)

-> [toy/app/experiment.py](/root/wild-diffusion/toy/app/experiment.py)

-> [toy/data_backends/provider.py](/root/wild-diffusion/toy/data_backends/provider.py)

-> [toy/model_backends/provider.py](/root/wild-diffusion/toy/model_backends/provider.py)

-> [toy/versions/registry.py](/root/wild-diffusion/toy/versions/registry.py)

-> chosen method under [toy/versions/](/root/wild-diffusion/toy/versions)

### Image training run

`train.py`

-> [training/dataset.py](/root/wild-diffusion/training/dataset.py)

-> [training/networks.py](/root/wild-diffusion/training/networks.py)

-> [training/loss.py](/root/wild-diffusion/training/loss.py)

-> [training/training_loop.py](/root/wild-diffusion/training/training_loop.py) or [training/training_wdro_loop.py](/root/wild-diffusion/training/training_wdro_loop.py)

## Files You Can Usually Ignore

- `__pycache__/`
- old run outputs under `toy_outputs/`, `toy-runs/`, `training-runs/`
- dataset payloads under `datasets/`

Also: the root-level toy-named files

- [cli.py](/root/wild-diffusion/cli.py)
- [experiment.py](/root/wild-diffusion/experiment.py)
- [method.py](/root/wild-diffusion/method.py)
- [registry.py](/root/wild-diffusion/registry.py)

do not appear to be the main path used by the current scripts, which consistently invoke [toy/run_toy.py](/root/wild-diffusion/toy/run_toy.py). Treat those root-level copies as legacy or duplicate until proven otherwise.

## Suggested Working Rule

When you touch the repo, ask which lane you are in first:

- toy method research
- MNIST / toy evaluation scripts
- image-scale EDM training
- outputs / docs / analysis

Then stay inside that lane. Most of the confusion here is from several parallel stacks living in one repo.
