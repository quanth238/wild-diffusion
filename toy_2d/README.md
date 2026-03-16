# Toy 2D Diffusion Baselines

This directory provides a CPU-friendly baseline for continuous-data experiments before moving to the full image pipeline.

What it includes:
- synthetic 2D datasets: `eight_gaussians`, `two_moons`, `spiral`
- a tiny EDM-style denoiser MLP
- `baseline`, WILD-style `wdro`, and a first `causal_wdro` prototype
- `causal_wdro` now attacks a discretized sigma path with sequential no-lookahead controls
- causal transport cost is reported both per step and as accumulated path cost
- plots, checkpoints, and simple sample-quality metrics

Recommended first run:

```bash
python -m toy_2d.train_wild --dataset eight_gaussians --epochs 40 --eval-every 10 --outdir toy-runs/wild_eight_gaussians
```

Useful CPU smoke test:

```bash
python -m toy_2d.train_wild --dataset two_moons --epochs 10 --num-samples 1024 --batch-size 128 --wdro-warmup-epochs 2 --wdro-refresh-every 2 --outdir toy-runs/smoke
```

Three-method comparison:

```bash
python -m toy_2d.compare_methods --method-configs toy-runs/tuned_method_configs_v1.json --outdir toy-runs/method_table_path_v1
```

Full experiment command sequences are in `toy_2d/EXPERIMENT_COMMANDS.md`.

Outputs:
- `config.json`: run configuration
- `metrics.jsonl`: per-eval metrics
- `summary.json`: final summary
- `checkpoint_best.pt` / `checkpoint_last.pt`: saved EMA models
- `plots/`: generated-vs-real scatters, adversarial refresh plots, and training curves
- `samples_latest.npz`: latest real/generated sample arrays
