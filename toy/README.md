# Toy: Trajectory-Perturbed Robust Diffusion (Hard-Constrained v2)

This folder is a Torch-first toy implementation for the draft method in `pdfs/method.md`,
focused on one target only:

- robust min-max training with trajectory control under a **hard per-step constraint**,
- no closeness term in this version.

## Structure

- `run_toy.py`: thin entrypoint.
- `app/cli.py`: CLI parser -> `ToyConfig`.
- `app/experiment.py`: orchestration pipeline (train, gate, eval, save).
- `data_backends/provider.py`: dataset template layer (`dataset_kind` backend).
- `config.py`: dataclass config.
- `data.py`: 2D toy data generation (8-mode Gaussian ring).
- `models.py`: denoiser/control MLPs.
  - Denoiser now uses EDM preconditioning (`c_skip`, `c_out`, `c_in`, `c_noise`) to match EDM/WILD behavior.
- `diffusion.py`: discrete VE-like rollout with shared noise + per-step projected control.
- `objective.py`: weighted denoising loss + attack-only inner objective.
- `trainer.py`: baseline and robust training loops.
- `checks.py`: rollout sanity checks + finite-difference gradient checks.
- `metrics.py`: denoise/recovery debug metrics aligned with the draft objective.
- `plotting.py`: focused debug plots for forward/backward + objective losses.

## Template Extension

- Current backend: `dataset_kind=toy_gmm`.
- To add image tests later:
  - add a new builder in `data_backends/provider.py` that returns `sample_train_batch` / `sample_val_batch`;
  - keep trainer unchanged (it now supports dataset-agnostic batch samplers);
  - plug image-specific metrics in `app/experiment.py` (or split evaluator module).

## Objective Implemented (v2)

Inner maximization:

- maximize `train_loss` under hard constraint:
  - `delta_k = Proj_{||.|| <= kappa * delta_sigma_k}(delta_raw_k)`.

Outer minimization:

- minimize weighted clean + attacked denoising loss
  (`outer_loss = attack_weight * L_attack + clean_weight * L_clean`).
- default config currently uses `outer_clean_weight=0.0`, i.e. attack-only outer objective.

No `R_close` term is used.

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
  --exp-name constrained_v2 \
  --steps 1200 \
  --batch-size 512 \
  --inner-steps 1 \
  --control-radius-kappa 0.3

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
SEEDS=0,1,2 DEVICE=cpu STEPS=3000 BATCH_SIZE=1024 \
KAPPA=0.3 OUTDIR=toy_outputs_stageA PREFIX=stageA_once \
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
