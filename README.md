# Wild Diffusion

Start with [docs/README.md](/root/wild-diffusion/docs/README.md).

Current active lane in this repo is Simpsons-MNIST RGB under the [toy/](/root/wild-diffusion/toy) stack, using the `cdro` method for the locked Simpsons workflow.

Canonical entrypoints:

- Toy experiments: `python toy/run_toy.py ...`
- Simpsons-MNIST RGB smoke run: `toy/scripts/run_simpsons_mnist_once.sh`
- Simpsons-MNIST RGB locked sweep: `toy/scripts/run_simpsons_locked_default_percent_sweep.sh`
- Image-scale EDM/WDRO training: `python train.py ...`
- Image generation from saved EDM checkpoints: `python generate.py ...`

Weights & Biases:

- Toy runs: add `--wandb --wandb-project <project>` to `python toy/run_toy.py ...`
- Image runs: add `--wandb --wandb-project <project>` to `python train.py ...`
- Optional flags on both paths: `--wandb-entity`, `--wandb-name`, `--wandb-group`, `--wandb-tags`, `--wandb-mode offline`

Useful maps:

- [toy/README.md](/root/wild-diffusion/toy/README.md): current toy-side status
- [toy/scripts/README.md](/root/wild-diffusion/toy/scripts/README.md): active vs historical script inventory
- [docs/SIMPSONS_DEFAULT_SETUP.md](/root/wild-diffusion/docs/SIMPSONS_DEFAULT_SETUP.md): locked Simpsons defaults and artifact roots

Compatibility note:

- Root `run_toy_once.sh` and `run_image_once.sh` are compatibility shims. Use the `toy/scripts/` copies as the canonical paths.
- Root `cli.py`, `experiment.py`, `method.py`, and `registry.py` are legacy copies, not the main path used by the current scripts.

## Lightweight Comparison Graphs

Use `comparison-graphs/` to browse paper/result plots without expanding heavy training outputs.

- Refresh the lightweight graph workspace after new runs:

  ```bash
  python scripts/sync_comparison_graphs.py
  ```

- CIFAR comparison merge/plot scripts mirror their PNG/CSV/JSON outputs there automatically.
- Canonical run artifacts remain under `training-runs/`; `comparison-graphs/by-source/` contains symlinks or small copies only.
- In VS Code, open `/home/bachlc/GM-CDRO/comparison-graphs` when reviewing plots. Avoid opening or expanding `training-runs/` unless you need raw run artifacts.

