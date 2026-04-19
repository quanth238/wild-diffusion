# Active Docs

This docs folder is intentionally limited to the active Simpsons-MNIST RGB workflow.

Older CIFAR, transfer, toy-stage diagnostic, and WDRO paper-alignment writeups were removed on 2026-04-08 because they are stale and no longer represent the current working context.

One fresh exception now exists for the next image-scale step:

- `docs/CIFAR10_PAPER_FIRST_PROTOCOL.md`: paper-first CIFAR-10 baseline protocol for reproducing the first unconditional DDPM++ row from the WILD-Diffusion ICLR paper

The active Simpsons decision is:

- Locked Simpsons method: `method_version=cdro`
- Locked Simpsons regime: CDRO-EDM, Simpsons-MNIST RGB 5%, seed 0, official EMA, warmup 0.2, rho 0.001, CDRO path steps 32
- Canonical workflow root: `toy/`
- Canonical script folder: `toy/scripts/`

Start here:

- `docs/SIMPSONS_DEFAULT_SETUP.md`: locked default Simpsons configuration and canonical artifact paths.
- `docs/CODEBASE_MAP.md`: code navigation map.
- `docs/RF_CDRO_RF_STATUS.md`: current toy-side RF / CDRO-RF status note.
- `toy/README.md`: toy-side current status.
- `toy/scripts/README.md`: active vs historical script inventory.

Active artifact roots:

- `toy_data/simpsons_mnist_rgb/summary.json`
- `toy_outputs/simpsons50_cdro_400k_warmup20_rho0p001_n032_cw0_aw100_perex_default/050pct/`
- `toy_outputs/simpsons100_cdro_400k_warmup20_rho0p001_n032_cw0_aw100_perex_sharedckpt/100pct/`
- `toy_outputs/simpsons5_cdro_400k_warmup20_rho64_n032_cw0_20260414/`

Historical artifact roots:

- `toy_outputs/simpsons_mnist_rgb_three_method_1pct_ablation_default_n64_rho_stress_20260407/summary/`
- `toy_outputs/simpsons_mnist_rgb_three_method_1pct_ablation_targeted_followup_ckptreuse_20260406/summary/`

Working rule:

- Treat docs as a thin index only.
- Treat manifests and `metrics.json` files under `toy_outputs/` as the source of truth for experiment settings and results.
- Treat `toy/scripts/` as canonical when a script also exists at repo root.
