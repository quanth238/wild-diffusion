# Wild Diffusion

Start with [docs/README.md](/root/wild-diffusion/docs/README.md).

Canonical entrypoints:

- Toy experiments: `python toy/run_toy.py ...`
- Image-scale EDM/WDRO training: `python train.py ...`
- Image generation from saved EDM checkpoints: `python generate.py ...`

The repo contains some root-level toy-named files (`cli.py`, `experiment.py`, `method.py`, `registry.py`) that are not the main path used by the current scripts. Treat the `toy/` package as the canonical toy implementation unless you have a specific reason not to.
