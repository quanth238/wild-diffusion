# Toy: Trajectory-Perturbed Robust Diffusion (Versioned: v1/v2/wild)

This folder is a Torch-first toy implementation for the draft method in `pdfs/method.md`.
Current runnable targets are:

- `--method-version v2`: robust min-max training with trajectory control under a **hard per-step constraint**.
- `--method-version v1`: soft regularization by **energy penalty only** (closeness disabled),
  with optional dual-lambda update for
  \(\rho\lambda + \sup_u[\mathcal L_{\mathrm{attack}}-\lambda \mathcal C_{\mathrm{energy}}]\).
- `--method-version wild`: sample-level WDRO surrogate update (WILD style):
  interval refresh of adversarial samples from
  `argmax_x' [loss(x') - gamma * 0.5||x'-x||^2]`.


## Structure

- `run_toy.py`: thin entrypoint.
- `app/cli.py`: CLI parser -> `ToyConfig`.
- `app/experiment.py`: orchestration pipeline (train, gate, eval, save), resolves method version.
- `data_backends/provider.py`: dataset template layer (`dataset_kind` backend).
- `model_backends/provider.py`: denoiser/control factory layer (`model_kind` backend).
- `diagnostics_backends/provider.py`: diagnostics/plotting family layer (`diagnostics_kind` backend).
- `config.py`: dataclass config.
- `shared/`: version-agnostic components.
  - `shared/sigma.py`: sigma ladder + sigma-step sampling.
  - `shared/objective.py`: EDM weighted denoise loss.
  - `shared/reverse.py`: reverse posterior + reverse trajectory samplers.
  - `shared/train_utils.py`, `shared/trainer_common.py`: shared trainer utilities + baseline trainer.
- `versions/`: version-specific robust method implementations.
  - `versions/v2/`: hard-constrained rollout + robust trainer (implemented).
  - `versions/v1/`: energy-penalty rollout + robust trainer (implemented, closeness disabled).
  - `versions/wild/`: WDRO surrogate sample-level trainer (implemented).
- `data.py`: 2D toy data generation (8-mode Gaussian ring).
- `models.py`: denoiser/control MLPs.
  - Denoiser now uses EDM preconditioning (`c_skip`, `c_out`, `c_in`, `c_noise`) to match EDM/WILD behavior.
- `diffusion.py`, `objective.py`, `trainer.py`: backward-compatible re-export shims.
- `checks.py`: rollout sanity checks + finite-difference gradient checks.
- `metrics.py`: denoise/recovery debug metrics aligned with the draft objective.
- `plotting.py`: focused debug plots for forward/backward + objective losses.

## Template Extension

- Current defaults:
  - `dataset_kind=toy_gmm`
  - `model_kind=auto` -> resolves to `toy_mlp`
  - `diagnostics_kind=auto` -> resolves to `toy_gmm`
- Added image-capable backend set:
  - `dataset_kind=image_folder`
  - `model_kind=image_conv`
  - `diagnostics_kind=image_basic`
- To add a new experiment family without breaking the current protocol:
  - add a new builder in `data_backends/provider.py` returning the `DatasetBundle` contract;
  - add a new branch in `model_backends/provider.py` for denoiser/control construction;
  - add a new branch in `diagnostics_backends/provider.py` for plots and optional backend-specific diagnostics.
- `trainer.py`, `objective.py`, `checks.py`, and `reverse_paths_from_terminal()` are now shape-agnostic over trailing dimensions.
- `sample_reverse_paths()` remains generic when a backend provides `sample_terminal_batch_fn`; without that callback it falls back to the legacy 2D Gaussian terminal sampler.

## Objective Implemented (v2)

Inner maximization:

- maximize `train_loss` under hard constraint:
  - `delta_k = Proj_{||.|| <= kappa * delta_sigma_k}(delta_raw_k)`.

Outer minimization:

- minimize weighted clean + attacked denoising loss
  (`outer_loss = attack_weight * L_attack + clean_weight * L_clean`).
- current defaults use `outer_clean_weight=1.0`, `outer_attack_weight=0.5`,
  with `warmup_clean_steps=900` and `warmup_ramp_steps=600`.

No `R_close` term is used.

## WILD Objective (method-version=`wild`)

Inner step (sample update):
- keep model fixed, update a batch of synthetic training samples by gradient ascent on
  `loss(x') - wild_gamma * 0.5||x' - x||^2` for `wild_inner_steps`.
- refresh this adversarial cache every `wild_update_interval` steps.

Outer step (parameter update):
- train denoiser on weighted clean/adv objective using `outer_clean_weight` and `outer_attack_weight`.
- works for both `--training-objective edm` and `--training-objective score`.

Main knobs:
- `--wild-update-interval`
- `--wild-cache-batches`
- `--wild-inner-steps`
- `--wild-step-size`
- `--wild-gamma`
- `--wild-clamp-samples --wild-sample-min --wild-sample-max` (optional)

### Training Objective Switch (`--training-objective`)

You can keep the same rollout/control framework and switch denoiser training loss:

- `--training-objective edm` (default):
  - weighted x0-regression (`EDM` style).
- `--training-objective score`:
  - weighted score matching on noisy states.
  - configure score weighting with `--score-matching-weight-power` (default `2.0`).

For fair convergence comparison, `metrics.json` also stores:
- `objective_debug.baseline_loss` (selected primary objective).
- `objective_debug.baseline_proxy_weighted_denoise_loss` (common x0-regression proxy metric).

## Flow Modes (important)

- `baseline_only=true`:
  - only baseline training; robust attack branch is skipped.
- `baseline_only=false` and `baseline_gate_enabled=true` (default):
  - run baseline first; robust branch only runs if baseline gate passes.
- `baseline_only=false` and `baseline_gate_enabled=false`:
  - robust branch always runs after baseline pretraining.

If your results suddenly differ, check `metrics.json -> flow_debug` and `baseline_gate.attack_training_executed` first.

## Quick Start

From repo root:

```bash
cd Wild-Diffusion
python toy/run_toy.py \
  --method-version v2 \
  --exp-name constrained_v2 \
  --steps 3000 \
  --batch-size 512 \
  --inner-steps 1 \
  --control-radius-kappa 0.15

# Gate behavior:
# attack chỉ chạy khi baseline pass gate.
# bật hard-fail khi gate fail:
#   --baseline-gate-error-on-fail
# tắt gate (không khuyến nghị):
#   --disable-baseline-gate

# Baseline-only debug (recommended before robust attack):
python toy/run_toy.py \
  --exp-name baseline_only_debug \
  --steps 2000 \
  --batch-size 1024 \
  --baseline-only \
  --debug-terminal-step 20

# Limited-data setting (recommended): fixed finite train set.
python toy/run_toy.py \
  --exp-name limited_data_2ppm \
  --train-points-per-mode 2 \
  --val-size 10000

# Stronger scarcity:
#   --train-points-per-mode 1
# Ablation back to population sampling (not limited-data):
#   --disable-limited-data
```

## One-Command Stage-A (Train + Eval)

Run baseline + robust for a seed list, then auto-summarize readiness for image experiments:

```bash
cd Wild-Diffusion

# Default: seeds=0,1,2 on CPU, steps=3000.
./toy/scripts/run_stageA_once_and_eval.sh

# Example override:
SEEDS=0,1,2 DEVICE=cpu STEPS=3000 BATCH_SIZE=512 \
KAPPA=0.15 OUTDIR=toy_outputs_stageA PREFIX=stageA_once \
./toy/scripts/run_stageA_once_and_eval.sh
```

Outputs:
- Per-run artifacts:
  - `toy_outputs_stageA/<prefix>_baseline_s<seed>/metrics.json`
  - `toy_outputs_stageA/<prefix>_robust_k<kappa_tag>_s<seed>/metrics.json`
- Aggregated evaluation:
  - `toy_outputs_stageA/<prefix>_summary_k<kappa_tag>.json`
  - `toy_outputs_stageA/<prefix>_summary_k<kappa_tag>.txt`

Notes:
- `BASELINE_GATE_HARD_FAIL=1` (default) adds `--baseline-gate-error-on-fail`.
- `RUN_CHECKS=1` (default) runs preflight checks; set `RUN_CHECKS=0` for faster smoke runs.
- You can tune evaluation thresholds via env vars:
  - `ATTACK_WIN_RATIO_MIN`, `ATTACK_MEAN_GAP_MAX`
  - `CLEAN_MEAN_GAP_MAX`, `CLEAN_TERMINAL_GAP_MAX`
  - `SAMPLE_AVG_DELTA_MAX`, `SAMPLE_P90_DELTA_MAX`
- `TAIL_SATURATION_MAX`, `MIN_SEED_PASS_RATIO`

## One-Command Single Run

Run one toy experiment (baseline or robust) with env vars:

```bash
cd Wild-Diffusion

# Robust single run (default mode=robust).
EXP_NAME=toy_robust_s0 DEVICE=cpu SEED=0 \
KAPPA=0.15 STEPS=3000 BATCH_SIZE=512 \
./toy/scripts/run_toy_once.sh

# Baseline-only single run.
RUN_MODE=baseline EXP_NAME=toy_baseline_s0 DEVICE=cpu SEED=0 \
STEPS=3000 BATCH_SIZE=512 \
./toy/scripts/run_toy_once.sh

# Add extra flags when needed:
# EXTRA_ARGS='--disable-baseline-gate --plot-stochastic-backward'
```

Output:
- `OUTDIR/EXP_NAME/metrics.json`
- `OUTDIR/EXP_NAME/debug_losses_and_recovery.png`
- `OUTDIR/EXP_NAME/forward_backward_baseline_attack.png`

## Image Backend

The same `toy/run_toy.py` protocol can now run on a small `ImageFolder` dataset.

Required folder layout:

```text
your_dataset_root/
  class_a/
    000.png
    001.png
  class_b/
    000.png
    001.png
```

Quick run with the image wrapper:

```bash
cd Wild-Diffusion

# Baseline-only smoke run on image data.
RUN_MODE=baseline \
DATASET_PATH=/abs/path/to/your_dataset_root \
EXP_NAME=image_baseline_smoke DEVICE=cpu \
IMAGE_TRAIN_SIZE=256 IMAGE_VAL_SIZE=256 \
STEPS=200 BATCH_SIZE=32 HIDDEN_DIM=64 \
./toy/scripts/run_image_once.sh

# Robust run on the same image dataset.
RUN_MODE=robust \
DATASET_PATH=/abs/path/to/your_dataset_root \
EXP_NAME=image_robust_smoke DEVICE=cpu \
IMAGE_TRAIN_SIZE=256 IMAGE_VAL_SIZE=256 \
STEPS=200 BATCH_SIZE=32 HIDDEN_DIM=64 \
KAPPA=0.15 OUTER_CLEAN_WEIGHT=1.0 OUTER_ATTACK_WEIGHT=0.5 \
./toy/scripts/run_image_once.sh
```

Notes:
- Image gate is backend-specific and currently checks:
  - generated image global std is not too small,
  - endpoint image global std is not too small,
  - endpoint recovery MSE mean is not too large.
- This is a minimal image sanity-check backend, not a replacement for full FID-based image evaluation.

## MNIST Backend

You can run MNIST directly (no `torchvision` requirement). The backend auto-downloads IDX files
to `~/.cache/wild_diffusion/mnist` (or `--dataset-path` if provided).

```bash
cd Wild-Diffusion

# EDM baseline on MNIST.
python toy/run_toy.py \
  --dataset-kind mnist \
  --model-kind image_conv \
  --diagnostics-kind image_basic \
  --image-channels 1 \
  --image-size 28 \
  --image-train-size 5000 \
  --image-val-size 2000 \
  --training-objective edm \
  --baseline-only

# Score-matching objective on MNIST.
python toy/run_toy.py \
  --dataset-kind mnist \
  --model-kind image_conv \
  --diagnostics-kind image_basic \
  --image-channels 1 \
  --image-size 28 \
  --image-train-size 5000 \
  --image-val-size 2000 \
  --training-objective score \
  --score-matching-weight-power 2.0 \
  --baseline-only
```

### One-Command EDM vs Score (+ attack, v2 path-level) on MNIST

```bash
cd Wild-Diffusion
python toy/scripts/compare_mnist_objectives.py \
  --outdir toy_outputs_mnist_compare \
  --prefix mnist_objcmp \
  --seeds 0,1 \
  --device auto \
  --steps 300 \
  --batch-size 128 \
  --kappa 0.15
```

Outputs:
- `toy_outputs_mnist_compare/mnist_objcmp_summary.json`
- `toy_outputs_mnist_compare/mnist_objcmp_summary.png`

### One-Command Baseline vs WILD on MNIST (EDM + Score)

```bash
cd Wild-Diffusion
python toy/scripts/compare_mnist_wild_objectives.py \
  --outdir toy_outputs_mnist_wild_compare \
  --prefix mnist_wildcmp \
  --seeds 0,1 \
  --device auto \
  --steps 300 \
  --batch-size 128 \
  --wild-update-interval 20 \
  --wild-cache-batches 4 \
  --wild-inner-steps 3 \
  --wild-step-size 0.05 \
  --wild-gamma 2.0
```

Outputs:
- `toy_outputs_mnist_wild_compare/mnist_wildcmp_summary.json`
- `toy_outputs_mnist_wild_compare/mnist_wildcmp_summary.png`

## One-Command Baseline vs WILD on Toy (EDM + Score)

```bash
cd Wild-Diffusion
python toy/scripts/compare_toy_wild_objectives.py \
  --outdir toy_outputs_wild_compare \
  --prefix toy_wildcmp \
  --seeds 0,1 \
  --device auto \
  --steps 800 \
  --batch-size 512 \
  --wild-update-interval 20 \
  --wild-cache-batches 4 \
  --wild-inner-steps 3 \
  --wild-step-size 0.05 \
  --wild-gamma 2.0
```

Outputs:
- `toy_outputs_wild_compare/toy_wildcmp_summary.json`
- `toy_outputs_wild_compare/toy_wildcmp_summary.png`

## Checks (before training)

By default `run_toy.py` runs:

- rollout sanity checks (shape + projected delta ratio),
- finite-difference gradient check for denoiser,
- finite-difference gradient check for control/adversary.

Disable checks only when needed:

```bash
python toy/run_toy.py --skip-checks
```

## Artifacts

Outputs are stored at `toy_outputs/<exp-name>/`:

- `forward_backward_baseline_attack.png`
  - row 1: forward baseline
  - row 2: backward baseline
  - row 3: forward attack
  - row 4: backward attack
  - uses `debug_terminal_step` (default `20`) as the terminal noise step for backward diagnostics.
  - by default uses deterministic backward (`plot_stochastic_backward=false`) for pairwise x0-recovery diagnostics.
- use `--plot-stochastic-backward` when you want stochastic generative reverse trajectories.
  - khi stochastic bật, baseline/attack reverse dùng **chung** noise schedule để so sánh công bằng.
- `debug_losses_and_recovery.png`
  - baseline/robust training losses
  - inner objective + delta-ratio debug
  - baseline denoise error on forward baseline vs forward attack
  - baseline x0 recovery MSE vs terminal step
- `metrics.json`
  - `dataset_debug`: xác nhận run có bật limited-data hay không, train-pool size, mode counts.
  - `baseline_gate`: trạng thái pass/fail và danh sách điều kiện fail.
  - `denoise_debug_train_pool` / `denoise_debug_heldout_pool`: train-vs-heldout denoise curves.
  - `generalization_debug`: gap `heldout - train` cho các curve chính (mean/terminal).
  - `objective_debug.baseline_sigma_counts`: histogram số mẫu train theo từng mức sigma (để debug coverage).
  - `objective_debug.robust_delta_norm_ratio_mean`: mức saturation trung bình của constraint trong train.
  - `constraint_debug.heldout_rollout_by_step`: `delta_ratio`, `frac_near_boundary`, `gap_norm` theo timestep.
  - `constraint_debug.attack_gap_windows_heldout`: attacked gap theo `low/mid/high noise` windows.
  - `sample_quality_debug.baseline_generated_mode_metrics` và `robust_generated_mode_metrics`.
  - `sample_quality_debug.*_to_train_min_dist`: khoảng cách nearest-train để theo dõi memorization.
  - `recovery_debug.bayes_posterior_mean_mse_at_terminal_sigma`: mốc Bayes cho pairwise hồi phục \(x_0\) tại `debug_terminal_step`.
  - `recovery_debug.baseline_to_bayes_mse_ratio_ref_terminal`: tỉ lệ baseline/Bayes để biết đang sai implementation hay gần giới hạn thông tin.
  - `recovery_debug.paired_reverse_delta_under_plot_setting`: chênh 2 nhánh reverse (ref vs attack) dưới cùng noise schedule khi plot stochastic bật.
  - `recovery_debug.paired_reverse_delta_deterministic`: chênh 2 nhánh reverse ở chế độ deterministic.
  - `recovery_debug.reverse_terminal_consistency`: assert `||bwd[:,terminal]-fwd[:,terminal]||` cho từng nhánh.

Notes:
- `sigma_data <= 0` means auto-estimate from toy clean-data scale.
- Default `sigma_max=2.0` is chosen for stable backward/x0-recovery diagnostics on toy.
- Optional: `--auto-log-normal-params` để auto-calibrate log-normal theo `[sigma_min, sigma_max]` khi cần debug coverage.
- Baseline gate mặc định bật (`baseline_gate_enabled=true`) để khóa attack nếu baseline chưa đạt chuẩn.
- `--reverse-terminal-assert-tol` (default `1e-8`): ngưỡng assert consistency ở terminal reverse.
- `--control-radius-kappa`: bán kính hard-constraint của control theo từng bước (`||delta_k|| <= kappa * delta_sigma_k`).
- EMA eval is optional (`use_ema_eval=false` by default). Bật EMA khi chạy dài; với run rất ngắn EMA có thể làm model đánh giá bị trễ.
- In this robust-denoising setting, pairwise recovery to the exact original sample becomes harder at large terminal noise. Use `baseline_x0_mse_vs_terminal_step_*` to verify this behavior.
- `baseline_plot_endpoint_mode_metrics_*` tells whether plotted backward endpoints are clustering near true modes.
