# Toy 2D Diffusion Baselines

This directory provides a CPU-friendly baseline for continuous-data experiments before moving to the full image pipeline.

Quick setup:

```bash
bash scripts/setup_toy_2d_env.sh
bash scripts/smoke_test_toy_2d.sh
bash scripts/smoke_test_toy_2d_cdro_sde.sh
bash scripts/smoke_test_toy_2d_cdro_markov.sh
```

Notes:
- The setup script creates a local `.venv` and installs the `toy_2d` dependencies.
- The setup script defaults to the CPU PyTorch wheel so the toy workflow comes up quickly.
- You can override the wheel source for a CUDA-specific build:
  `TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 bash scripts/setup_toy_2d_env.sh`

What it includes:
- synthetic 2D datasets: `eight_gaussians`, `two_moons`, `spiral`
- a tiny EDM-style denoiser MLP
- `baseline`, WILD-style `wdro`, and a first `cdro` prototype
- `cdro` now attacks a discretized sigma path with sequential no-lookahead controls
- `cdro` transport cost is reported both per step and as accumulated path cost
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

Including the new Markov score-based CDRO method:

```bash
python -m toy_2d.compare_methods --method-configs toy-runs/tuned_method_configs_v1.json --methods baseline wdro cdro cdro_markov --outdir toy-runs/method_table_with_markov
```

Score-matching-only baselines on the same VP chain:

```bash
python -m toy_2d.train_cdro_markov --method baseline_score --outdir toy-runs/baseline_score
python -m toy_2d.train_cdro_markov --method wdro_score --outdir toy-runs/wdro_score
python -m toy_2d.train_cdro_markov --method cdro_markov --outdir toy-runs/cdro_markov
```

Full experiment command sequences are in `toy_2d/EXPERIMENT_COMMANDS.md`.

Outputs:
- `config.json`: run configuration
- `metrics.jsonl`: per-eval metrics
- `summary.json`: final summary
- `checkpoint_best.pt` / `checkpoint_last.pt`: saved EMA models
- `plots/`: generated-vs-real scatters, adversarial refresh plots, and training curves
- `samples_latest.npz`: latest real/generated sample arrays

Experimental SDE-based CDRO pipeline:
- `python -m toy_2d.train_cdro_sde`: trains the forward VP-SDE causal-DRO predictor/adversary/dual loop and exports a robust trajectory replay buffer
- `python -m toy_2d.train_cdro_reverse`: trains the reverse teacher-forced sampler on saved robust trajectories
- `python -m toy_2d.sample_cdro_reverse`: samples from the learned reverse model and writes plots/metrics

Experimental Markov score-based CDRO pipeline:
- `python -m toy_2d.train_cdro_markov`: trains a state-Markov robust forward chain with exact one-step Gaussian score targets, alternating score descent, control ascent, and dual updates
- `--method baseline_score`: pure Markov score matching with no adversary
- `--method wdro_score`: score matching with WDRO-style adversarial refreshes on clean points
- `--method cdro_markov`: score matching with Markov drift control and dual-budget updates
- keeps the frozen forward control inside reverse-time sampling, rather than fitting a separate reverse model
- `python -m toy_2d.eval_cdro_markov_checkpoint`: re-evaluates a saved Markov checkpoint with the current sampler/ablation logic and writes fresh metrics without retraining
- `python -m toy_2d.sweep_cdro_markov_reeval`: sweeps a comparison directory, re-evaluates every `cdro_markov` seed run, and writes one aggregate re-eval report
- writes checkpoints, sample plots, reverse-process plots, and metric logs

Minimal Markov example:

```bash
python -m toy_2d.train_cdro_markov --outdir toy-runs/cdro_markov
```

Re-evaluate a saved checkpoint:

```bash
python -m toy_2d.eval_cdro_markov_checkpoint --run-dir toy-runs/cdro_markov --checkpoint-mode best --outdir toy-runs/cdro_markov_reeval
```

Sweep a comparison directory:

```bash
python -m toy_2d.sweep_cdro_markov_reeval --comparison-dir toy-runs/score_family_best_200ep_v7
```

Minimal end-to-end example:

```bash
python -m toy_2d.train_cdro_sde --outdir toy-runs/cdro_sde_forward
python -m toy_2d.train_cdro_reverse --forward-run-dir toy-runs/cdro_sde_forward --outdir toy-runs/cdro_sde_reverse
python -m toy_2d.sample_cdro_reverse --reverse-run-dir toy-runs/cdro_sde_reverse --outdir toy-runs/cdro_sde_samples
```
