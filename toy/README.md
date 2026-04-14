# Toy Stack

This package is the canonical toy-side implementation in this repo.

The current active workflow is Simpsons-MNIST RGB with `method_version=cdro`.

Recent capability note:

- The toy stack now also supports RF-family experiments alongside EDM-family ones.
- Public comparison naming should read: `Baseline EDM`, `Wild-Diffusion`, `CDRO-EDM`, `RF`, `CDRO-RF`.
- In plots/presentation, the historically named `wdro` method should display as `Wild-Diffusion`.
- `Wild-Diffusion-RF` is still intentionally deferred.

Important caveat:

- The generic `toy/run_toy.py` CLI still supports many historical methods.
- The active Simpsons choice is enforced by the Simpsons-specific wrappers and collectors, not by treating every generic default as canonical.

## Active Path

Use these first when the task is about the current working setup:

- `toy/scripts/setup_simpsons_mnist_rgb.py`: prepare the RGB dataset and FID ref under `toy_data/simpsons_mnist_rgb/`.
- `toy/scripts/run_simpsons_mnist_once.sh`: single smoke run. Defaults to `RUN_MODE=robust` and `METHOD_VERSION=cdro`, with smoke-scale training/eval sizes but the locked CDRO method knobs.
- `toy/scripts/collect_three_method_seed_data.py`: canonical family-aware collector for EDM/RF baselines and robust methods on a shared comparison schema.
- `toy/scripts/run_simpsons_locked_default_percent_sweep.sh`: locked percent sweep launcher for the active Simpsons defaults.
- `docs/SIMPSONS_DEFAULT_SETUP.md`: locked configuration, artifact roots, and canonical command examples.
- `docs/RF_CDRO_RF_STATUS.md`: current RF / CDRO-RF implementation status and naming conventions.

## Core Files

- `run_toy.py`: thin entrypoint.
- `app/cli.py`: CLI parsing and validation.
- `app/experiment.py`: orchestration, training, eval, metrics, and artifact writing.
- `config.py`: full config dataclass, including CDRO/WDRO/WILD knobs.
- `shared/trainer_common.py`: baseline training path, including the strong public RF baseline.
- `shared/sigma.py`: family-aware EDM/RF grid helpers, including RF solver and stratified training grids.
- `versions/registry.py`: resolves `--method-version`.
- `compute_accounting.py`: weighted-compute accounting and warmup calibration.
- `shared/`: version-agnostic diffusion, objective, runtime, and trainer utilities.
- `versions/`: method-specific implementations.

## Methods Retained In Tree

- `cdro`: current active Simpsons method.
- `wdro` and `wild`: retained comparison baselines for shared-grid evaluation.
- `training_objective=rf`: rectified-flow family support under the same toy stack.
- `clean`, `v1`, `v1.1`, `v1.2`, `v2`, `v2.1`: historical or ablation variants kept for reproducibility.

Do not infer the active repo direction from the presence of those historical methods alone.

## Where To Edit

If you need to change the active Simpsons CDRO behavior:

- attack/training logic: `toy/versions/cdro/diffusion.py`, `toy/versions/cdro/trainer.py`
- method metadata and eval helpers: `toy/versions/cdro/method.py`
- shared knobs and CLI defaults: `toy/config.py`, `toy/app/cli.py`
- shared-grid orchestration: `toy/scripts/collect_three_method_seed_data.py`
- locked Simpsons launcher defaults: `toy/scripts/run_simpsons_locked_default_percent_sweep.sh`

If you need to change dataset handling:

- `toy/data_backends/provider.py`
- `toy/scripts/setup_simpsons_mnist_rgb.py`

If you need to change plots or saved diagnostics:

- `toy/app/experiment.py`
- `toy/diagnostics_backends/provider.py`
- `toy/scripts/plot_three_method_fid_curves.py`

## Quick Commands

Prepare the active dataset:

```bash
python toy/scripts/setup_simpsons_mnist_rgb.py
```

Run one active smoke experiment:

```bash
toy/scripts/run_simpsons_mnist_once.sh
```

Run the locked percent sweep:

```bash
toy/scripts/run_simpsons_locked_default_percent_sweep.sh
```

Run the generic image wrapper with an explicit method choice:

```bash
METHOD_VERSION=cdro toy/scripts/run_image_once.sh
```

## Working Rules

- Treat `toy/` as canonical over same-named root-level toy files.
- Treat `toy/scripts/` as canonical over same-named root-level helper scripts.
- Treat manifests and `metrics.json` files under `toy_outputs/` as the source of truth for any finished run.
- Treat older MNIST, v1.x, v2.x, and stage-A scripts as historical unless the task explicitly points to them.
