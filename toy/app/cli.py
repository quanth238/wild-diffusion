import argparse
import math
from typing import Optional, Sequence

from ..config import ToyConfig
from ..mainline_baseline import (
    SUPPORTED_BASELINE_TRAIN_BACKENDS,
    resolve_baseline_train_batch_gpu,
    validate_mainline_baseline_request,
)
from ..versions.registry import SUPPORTED_METHOD_VERSIONS


def _validate_config(cfg: ToyConfig) -> None:
    """Fail-fast validation for impossible or degenerate training settings."""

    if cfg.method_version not in SUPPORTED_METHOD_VERSIONS:
        raise ValueError(
            f"--method-version must be one of {SUPPORTED_METHOD_VERSIONS}, got {cfg.method_version}"
        )
    if str(cfg.image_backbone).lower() not in ("conv", "songunet", "ddpmpp"):
        raise ValueError(
            f"--image-backbone must be one of ('conv', 'songunet', 'ddpmpp'), got {cfg.image_backbone}"
        )
    if str(cfg.image_backbone).lower() == "ddpmpp" and int(cfg.hidden_dim) != 128:
        raise ValueError(
            f"--hidden-dim must be 128 when --image-backbone=ddpmpp, got {cfg.hidden_dim}"
        )
    if str(cfg.baseline_train_backend).lower() not in SUPPORTED_BASELINE_TRAIN_BACKENDS:
        raise ValueError(
            f"--baseline-train-backend must be one of {SUPPORTED_BASELINE_TRAIN_BACKENDS}, "
            f"got {cfg.baseline_train_backend}"
        )
    validate_mainline_baseline_request(cfg)
    if str(cfg.amp_dtype).lower() not in ("auto", "off", "bf16", "bfloat16", "fp16", "float16", "half"):
        raise ValueError(
            "--amp-dtype must be one of "
            "('auto', 'off', 'bfloat16', 'float16'), got "
            f"{cfg.amp_dtype}"
        )
    if cfg.steps < 0:
        raise ValueError(f"--steps must be >= 0, got {cfg.steps}")
    if cfg.batch_size <= 0:
        raise ValueError(f"--batch-size must be > 0, got {cfg.batch_size}")
    if int(getattr(cfg, "baseline_train_batch_gpu", 0)) < 0:
        raise ValueError(
            "--baseline-train-batch-gpu must be >= 0, "
            f"got {cfg.baseline_train_batch_gpu}"
        )
    batch_gpu = resolve_baseline_train_batch_gpu(cfg)
    if batch_gpu is not None and batch_gpu > int(cfg.batch_size):
        raise ValueError(
            "--baseline-train-batch-gpu must be <= --batch-size, "
            f"got batch_gpu={batch_gpu}, batch_size={cfg.batch_size}"
        )
    if cfg.baseline_ckpt_path and not isinstance(cfg.baseline_ckpt_path, str):
        raise ValueError("--baseline-ckpt-path must be a string path.")
    if cfg.robust_resume_ckpt_path and not isinstance(cfg.robust_resume_ckpt_path, str):
        raise ValueError("--robust-resume-ckpt-path must be a string path.")
    if cfg.robust_save_ckpt_path and not isinstance(cfg.robust_save_ckpt_path, str):
        raise ValueError("--robust-save-ckpt-path must be a string path.")
    if cfg.weighted_compute_calibration_path and not isinstance(cfg.weighted_compute_calibration_path, str):
        raise ValueError("--weighted-compute-calibration-path must be a string path.")
    if cfg.weighted_inputgrad_alpha < 0:
        raise ValueError(f"--weighted-inputgrad-alpha must be >= 0, got {cfg.weighted_inputgrad_alpha}")
    if cfg.weighted_parambackward_beta < 0:
        raise ValueError(
            f"--weighted-parambackward-beta must be >= 0, got {cfg.weighted_parambackward_beta}"
        )
    weighted_explicit = (cfg.weighted_inputgrad_alpha > 0.0) or (cfg.weighted_parambackward_beta > 0.0)
    if weighted_explicit and not (
        cfg.weighted_inputgrad_alpha > 0.0 and cfg.weighted_parambackward_beta > 0.0
    ):
        raise ValueError(
            "Explicit weighted compute calibration requires both --weighted-inputgrad-alpha "
            "and --weighted-parambackward-beta to be positive."
        )
    if cfg.baseline_steps_override < 0:
        raise ValueError(f"--baseline-steps-override must be >= 0, got {cfg.baseline_steps_override}")
    if cfg.inner_steps < 0:
        raise ValueError(f"--inner-steps must be >= 0, got {cfg.inner_steps}")
    if cfg.attack_num_steps is not None and int(cfg.attack_num_steps) not in (1, 2):
        raise ValueError(f"--attack-num-steps must be one of (1, 2), got {cfg.attack_num_steps}")
    if cfg.training_objective not in ("edm", "score", "rf"):
        raise ValueError(
            f"--training-objective must be one of ('edm', 'score', 'rf'), got {cfg.training_objective}"
        )
    if cfg.score_matching_weight_power < 0:
        raise ValueError(
            f"--score-matching-weight-power must be >= 0, got {cfg.score_matching_weight_power}"
        )
    if str(cfg.rf_baseline_mode).lower() not in ("strong", "plain"):
        raise ValueError(
            f"--rf-baseline-mode must be one of ('strong', 'plain'), got {cfg.rf_baseline_mode}"
        )
    if not (0.0 < float(cfg.rf_stage1_fraction) < 1.0):
        raise ValueError(
            f"--rf-stage1-fraction must be in (0, 1), got {cfg.rf_stage1_fraction}"
        )
    if str(cfg.rf_reflow_t_distribution).lower() not in ("u_shaped", "uniform"):
        raise ValueError(
            "--rf-reflow-t-distribution must be one of ('u_shaped', 'uniform'), got "
            f"{cfg.rf_reflow_t_distribution}"
        )
    if str(cfg.rf_loss).lower() not in ("pseudo_huber", "mse"):
        raise ValueError(f"--rf-loss must be one of ('pseudo_huber', 'mse'), got {cfg.rf_loss}")
    if float(cfg.rf_pseudo_huber_delta) <= 0:
        raise ValueError(
            f"--rf-pseudo-huber-delta must be > 0, got {cfg.rf_pseudo_huber_delta}"
        )
    if str(cfg.rf_cdro_pair_source).lower() not in ("auto", "reflow", "data_noise"):
        raise ValueError(
            "--rf-cdro-pair-source must be one of ('auto', 'reflow', 'data_noise'), got "
            f"{cfg.rf_cdro_pair_source}"
        )
    if cfg.n_steps_path <= 0:
        raise ValueError(f"--n-steps-path must be > 0, got {cfg.n_steps_path}")
    if cfg.sigma_min <= 0:
        raise ValueError(f"--sigma-min must be > 0, got {cfg.sigma_min}")
    if cfg.sigma_max <= cfg.sigma_min:
        raise ValueError(
            f"--sigma-max must be > --sigma-min, got sigma_max={cfg.sigma_max}, sigma_min={cfg.sigma_min}"
        )
    if cfg.control_radius_kappa < 0:
        raise ValueError(f"--control-radius-kappa must be >= 0, got {cfg.control_radius_kappa}")
    if cfg.v21_rho < 0 or cfg.v21_rho > 1:
        raise ValueError(f"--v21-rho must be in [0, 1], got {cfg.v21_rho}")
    if cfg.cdro_step_size <= 0:
        raise ValueError(f"--cdro-step-size must be > 0, got {cfg.cdro_step_size}")
    if cfg.cdro_total_budget_rho < 0:
        raise ValueError(f"--cdro-total-budget-rho must be >= 0, got {cfg.cdro_total_budget_rho}")
    if cfg.cdro_time_horizon <= 0:
        raise ValueError(f"--cdro-time-horizon must be > 0, got {cfg.cdro_time_horizon}")
    if str(cfg.cdro_edm_ladder_mode).lower() not in (
        "deterministic_midpoint_quantile",
        "stochastic_stratified_quantile",
    ):
        raise ValueError(
            "--cdro-edm-ladder-mode must be one of "
            "('deterministic_midpoint_quantile', 'stochastic_stratified_quantile'), got "
            f"{cfg.cdro_edm_ladder_mode}"
        )
    if not (0.0 <= cfg.cdro_warmup_fraction <= 1.0):
        raise ValueError(f"--cdro-warmup-fraction must be in [0, 1], got {cfg.cdro_warmup_fraction}")
    if cfg.outer_attack_weight < 0 or cfg.outer_clean_weight < 0:
        raise ValueError(
            "outer loss weights must be non-negative, got "
            f"attack={cfg.outer_attack_weight}, clean={cfg.outer_clean_weight}"
        )
    if cfg.outer_attack_weight == 0 and cfg.outer_clean_weight == 0:
        raise ValueError("Both outer weights are zero; training objective is identically zero.")
    if cfg.wild_update_interval <= 0:
        raise ValueError(f"--wild-update-interval must be > 0, got {cfg.wild_update_interval}")
    if cfg.wild_cache_batches <= 0:
        raise ValueError(f"--wild-cache-batches must be > 0, got {cfg.wild_cache_batches}")
    if cfg.wild_inner_steps < 0:
        raise ValueError(f"--wild-inner-steps must be >= 0, got {cfg.wild_inner_steps}")
    if cfg.wild_step_size <= 0:
        raise ValueError(f"--wild-step-size must be > 0, got {cfg.wild_step_size}")
    if cfg.wild_gamma < 0:
        raise ValueError(f"--wild-gamma must be >= 0, got {cfg.wild_gamma}")
    if cfg.wild_delta_ratio_denom <= 0:
        raise ValueError(f"--wild-delta-ratio-denom must be > 0, got {cfg.wild_delta_ratio_denom}")
    if cfg.wild_sample_max <= cfg.wild_sample_min:
        raise ValueError(
            f"--wild-sample-max must be > --wild-sample-min, got {cfg.wild_sample_max} <= {cfg.wild_sample_min}"
        )
    if not (0.0 <= cfg.wdro_warmup_fraction < 1.0):
        raise ValueError(f"--wdro-warmup-fraction must be in [0, 1), got {cfg.wdro_warmup_fraction}")
    if cfg.wdro_refresh_epochs <= 0:
        raise ValueError(f"--wdro-refresh-epochs must be > 0, got {cfg.wdro_refresh_epochs}")
    if cfg.wdro_adv_prob < 0 or cfg.wdro_adv_prob > 1:
        raise ValueError(f"--wdro-adv-prob must be in [0, 1], got {cfg.wdro_adv_prob}")
    if cfg.wdro_attack_steps < 0:
        raise ValueError(f"--wdro-attack-steps must be >= 0, got {cfg.wdro_attack_steps}")
    if cfg.wdro_attack_step_size <= 0:
        raise ValueError(f"--wdro-attack-step-size must be > 0, got {cfg.wdro_attack_step_size}")
    if cfg.wdro_gamma < 0:
        raise ValueError(f"--wdro-gamma must be >= 0, got {cfg.wdro_gamma}")
    if cfg.wdro_sample_max <= cfg.wdro_sample_min:
        raise ValueError(
            f"--wdro-sample-max must be > --wdro-sample-min, got {cfg.wdro_sample_max} <= {cfg.wdro_sample_min}"
        )
    if cfg.dataset_kind == "image_folder":
        if not cfg.dataset_path:
            raise ValueError("--dataset-path must be provided when --dataset-kind=image_folder")
        if cfg.image_size <= 0:
            raise ValueError(f"--image-size must be > 0, got {cfg.image_size}")
        if cfg.image_channels not in (1, 3):
            raise ValueError(f"--image-channels must be 1 or 3, got {cfg.image_channels}")
        if cfg.image_train_size <= 0 or cfg.image_val_size <= 0:
            raise ValueError(
                f"--image-train-size and --image-val-size must be > 0, got {cfg.image_train_size}, {cfg.image_val_size}"
            )
    if cfg.dataset_kind == "mnist":
        if cfg.image_channels != 1:
            raise ValueError(
                f"--image-channels must be 1 for --dataset-kind=mnist, got {cfg.image_channels}"
            )
        if cfg.image_size <= 0:
            raise ValueError(f"--image-size must be > 0, got {cfg.image_size}")
        if cfg.mnist_use_percent_split:
            if cfg.mnist_train_percent <= 0 or cfg.mnist_train_percent > 100:
                raise ValueError(
                    f"--mnist-train-percent must be in (0, 100], got {cfg.mnist_train_percent}"
                )
            if cfg.mnist_val_percent <= 0 or cfg.mnist_val_percent > 100:
                raise ValueError(
                    f"--mnist-val-percent must be in (0, 100], got {cfg.mnist_val_percent}"
                )
        else:
            if cfg.image_train_size <= 0 or cfg.image_val_size <= 0:
                raise ValueError(
                    "--image-train-size and --image-val-size must be > 0 when "
                    "--disable-mnist-percent-split is active, got "
                    f"{cfg.image_train_size}, {cfg.image_val_size}"
                )

    # Keep validation strict on impossible settings only; do not emit runtime warnings
    # that might be misinterpreted as implementation errors.


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser for toy baseline and constrained robust training."""

    parser = argparse.ArgumentParser(
        description="Toy trajectory-perturbed robust diffusion (hard constrained control)."
    )
    parser.add_argument("--outdir", type=str, default=ToyConfig.outdir)
    parser.add_argument("--exp-name", type=str, default=ToyConfig.exp_name)
    parser.add_argument("--seed", type=int, default=ToyConfig.seed)
    parser.add_argument("--device", type=str, default=ToyConfig.device, choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--disable-tf32", action="store_true")
    parser.add_argument("--disable-cudnn-benchmark", action="store_true")
    parser.add_argument("--amp-dtype", type=str, default=ToyConfig.amp_dtype)
    parser.add_argument(
        "--method-version",
        type=str,
        default=ToyConfig.method_version,
        choices=list(SUPPORTED_METHOD_VERSIONS),
    )
    parser.add_argument("--dataset-kind", type=str, default=ToyConfig.dataset_kind)
    parser.add_argument("--model-kind", type=str, default=ToyConfig.model_kind)
    parser.add_argument(
        "--image-backbone",
        type=str,
        default=ToyConfig.image_backbone,
        choices=["conv", "songunet", "ddpmpp"],
    )
    parser.add_argument(
        "--baseline-train-backend",
        type=str,
        default=ToyConfig.baseline_train_backend,
        choices=list(SUPPORTED_BASELINE_TRAIN_BACKENDS),
    )
    parser.add_argument("--diagnostics-kind", type=str, default=ToyConfig.diagnostics_kind)
    parser.add_argument("--dataset-path", type=str, default=ToyConfig.dataset_path)
    parser.add_argument("--dataset-val-path", type=str, default=ToyConfig.dataset_val_path)
    parser.add_argument("--image-size", type=int, default=ToyConfig.image_size)
    parser.add_argument("--image-channels", type=int, default=ToyConfig.image_channels)
    parser.add_argument("--image-train-size", type=int, default=ToyConfig.image_train_size)
    parser.add_argument("--image-val-size", type=int, default=ToyConfig.image_val_size)
    parser.add_argument("--image-split-seed", type=int, default=ToyConfig.image_split_seed)
    parser.add_argument(
        "--mnist-use-percent-split",
        action="store_true",
        default=ToyConfig.mnist_use_percent_split,
    )
    parser.add_argument("--disable-mnist-percent-split", action="store_true")
    parser.add_argument("--mnist-train-percent", type=float, default=ToyConfig.mnist_train_percent)
    parser.add_argument("--mnist-val-percent", type=float, default=ToyConfig.mnist_val_percent)
    parser.add_argument("--image-gate-min-generated-std", type=float, default=ToyConfig.image_gate_min_generated_std)
    parser.add_argument("--image-gate-min-endpoint-std", type=float, default=ToyConfig.image_gate_min_endpoint_std)
    parser.add_argument(
        "--image-gate-max-endpoint-recovery-mse",
        type=float,
        default=ToyConfig.image_gate_max_endpoint_recovery_mse,
    )

    parser.add_argument("--steps", type=int, default=ToyConfig.steps)
    parser.add_argument("--batch-size", type=int, default=ToyConfig.batch_size)
    parser.add_argument("--baseline-train-batch-gpu", type=int, default=ToyConfig.baseline_train_batch_gpu)
    parser.add_argument("--log-every", type=int, default=ToyConfig.log_every)
    parser.add_argument("--eval-samples", type=int, default=ToyConfig.eval_samples)
    parser.add_argument("--debug-eval-batch", type=int, default=ToyConfig.debug_eval_batch)
    parser.add_argument("--debug-terminal-step", type=int, default=ToyConfig.debug_terminal_step)
    parser.add_argument("--plot-stochastic-backward", action="store_true", default=ToyConfig.plot_stochastic_backward)
    parser.add_argument("--plot-deterministic-backward", action="store_true")
    parser.add_argument("--baseline-only", action="store_true", default=ToyConfig.baseline_only)
    parser.add_argument("--compute-fid", action="store_true", default=ToyConfig.compute_fid)
    parser.add_argument("--fid-samples", type=int, default=ToyConfig.fid_samples)
    parser.add_argument("--fid-ref-path", type=str, default=ToyConfig.fid_ref_path)
    parser.add_argument("--fid-ref-policy", type=str, default=ToyConfig.fid_ref_policy)
    parser.add_argument(
        "--weighted-compute-calibration-path",
        type=str,
        default=ToyConfig.weighted_compute_calibration_path,
    )
    parser.add_argument("--weighted-inputgrad-alpha", type=float, default=ToyConfig.weighted_inputgrad_alpha)
    parser.add_argument(
        "--weighted-parambackward-beta",
        type=float,
        default=ToyConfig.weighted_parambackward_beta,
    )
    parser.add_argument("--eval-seed-offset-gate", type=int, default=ToyConfig.eval_seed_offset_gate)
    parser.add_argument("--eval-seed-offset-metrics", type=int, default=ToyConfig.eval_seed_offset_metrics)
    parser.add_argument("--disable-eval-shared-terminal-noise", action="store_true")
    parser.add_argument("--disable-eval-shared-reverse-noise", action="store_true")
    parser.add_argument("--use-ema-eval", action="store_true", default=ToyConfig.use_ema_eval)
    parser.add_argument("--disable-ema-eval", action="store_true")
    parser.add_argument("--ema-mode", type=str, default=ToyConfig.ema_mode, choices=["official", "fixed"])
    parser.add_argument("--ema-decay", type=float, default=ToyConfig.ema_decay)
    parser.add_argument("--ema-halflife-kimg", type=float, default=ToyConfig.ema_halflife_kimg)
    parser.add_argument("--ema-rampup-ratio", type=float, default=ToyConfig.ema_rampup_ratio)
    parser.add_argument("--disable-ema-rampup", action="store_true")
    parser.add_argument("--disable-baseline-ckpt", action="store_true")
    parser.add_argument("--baseline-ckpt-path", type=str, default=ToyConfig.baseline_ckpt_path)
    parser.add_argument("--baseline-ckpt-force-retrain", action="store_true", default=ToyConfig.baseline_ckpt_force_retrain)
    parser.add_argument("--disable-baseline-ckpt-strict-meta", action="store_true")
    parser.add_argument("--robust-resume-ckpt-path", type=str, default=ToyConfig.robust_resume_ckpt_path)
    parser.add_argument("--robust-save-ckpt-path", type=str, default=ToyConfig.robust_save_ckpt_path)
    parser.add_argument("--baseline-steps-override", type=int, default=ToyConfig.baseline_steps_override)
    parser.add_argument("--wandb", action="store_true", default=ToyConfig.wandb_enabled)
    parser.add_argument("--wandb-project", type=str, default=ToyConfig.wandb_project)
    parser.add_argument("--wandb-entity", type=str, default=ToyConfig.wandb_entity)
    parser.add_argument("--wandb-name", type=str, default=ToyConfig.wandb_name)
    parser.add_argument("--wandb-group", type=str, default=ToyConfig.wandb_group)
    parser.add_argument("--wandb-tags", type=str, default=ToyConfig.wandb_tags)
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default=ToyConfig.wandb_mode,
        choices=["online", "offline", "disabled"],
    )

    parser.add_argument("--hidden-dim", type=int, default=ToyConfig.hidden_dim)
    parser.add_argument("--lr-theta", type=float, default=ToyConfig.lr_theta)
    parser.add_argument("--lr-phi", type=float, default=ToyConfig.lr_phi)
    parser.add_argument("--inner-steps", type=int, default=ToyConfig.inner_steps)
    parser.add_argument("--attack-num-steps", type=int, default=ToyConfig.attack_num_steps, choices=[1, 2])
    parser.add_argument("--clip-phi-grad", type=float, default=ToyConfig.clip_phi_grad)
    parser.add_argument("--training-objective", type=str, default=ToyConfig.training_objective, choices=["edm", "score", "rf"])
    parser.add_argument("--score-matching-weight-power", type=float, default=ToyConfig.score_matching_weight_power)
    parser.add_argument(
        "--rf-baseline-mode",
        type=str,
        default=ToyConfig.rf_baseline_mode,
        choices=["strong", "plain"],
    )
    parser.add_argument("--rf-stage1-fraction", type=float, default=ToyConfig.rf_stage1_fraction)
    parser.add_argument(
        "--rf-reflow-t-distribution",
        type=str,
        default=ToyConfig.rf_reflow_t_distribution,
        choices=["u_shaped", "uniform"],
    )
    parser.add_argument("--rf-loss", type=str, default=ToyConfig.rf_loss, choices=["pseudo_huber", "mse"])
    parser.add_argument("--rf-pseudo-huber-delta", type=float, default=ToyConfig.rf_pseudo_huber_delta)
    parser.add_argument("--rf-edm-init-ckpt-path", type=str, default=ToyConfig.rf_edm_init_ckpt_path)
    parser.add_argument(
        "--rf-cdro-pair-source",
        type=str,
        default=ToyConfig.rf_cdro_pair_source,
        choices=["auto", "reflow", "data_noise"],
    )

    parser.add_argument("--n-modes", type=int, default=ToyConfig.n_modes)
    parser.add_argument("--mode-radius", type=float, default=ToyConfig.mode_radius)
    parser.add_argument("--data-std", type=float, default=ToyConfig.data_std)
    parser.add_argument("--limited-data-enabled", action="store_true", default=ToyConfig.limited_data_enabled)
    parser.add_argument("--disable-limited-data", action="store_true")
    parser.add_argument("--train-points-per-mode", type=int, default=ToyConfig.train_points_per_mode)
    parser.add_argument("--val-size", type=int, default=ToyConfig.val_size)

    parser.add_argument("--n-steps-path", type=int, default=ToyConfig.n_steps_path)
    parser.add_argument("--sigma-min", type=float, default=ToyConfig.sigma_min)
    parser.add_argument("--sigma-max", type=float, default=ToyConfig.sigma_max)
    parser.add_argument("--sigma-data", type=float, default=ToyConfig.sigma_data)
    parser.add_argument(
        "--use-log-normal-sigma-sampling",
        action="store_true",
        default=ToyConfig.use_log_normal_sigma_sampling,
    )
    parser.add_argument("--disable-log-normal-sigma-sampling", action="store_true")
    parser.add_argument("--auto-log-normal-params", action="store_true", default=ToyConfig.auto_log_normal_params)
    parser.add_argument("--disable-auto-log-normal-params", action="store_true")
    parser.add_argument("--p-mean", type=float, default=ToyConfig.p_mean)
    parser.add_argument("--p-std", type=float, default=ToyConfig.p_std)

    parser.add_argument("--lambda-energy", type=float, default=ToyConfig.lambda_energy)
    parser.add_argument("--cdro-step-size", type=float, default=ToyConfig.cdro_step_size)
    parser.add_argument("--cdro-total-budget-rho", type=float, default=ToyConfig.cdro_total_budget_rho)
    parser.add_argument("--cdro-time-horizon", type=float, default=ToyConfig.cdro_time_horizon)
    parser.add_argument(
        "--cdro-edm-ladder-mode",
        type=str,
        default=ToyConfig.cdro_edm_ladder_mode,
        choices=["deterministic_midpoint_quantile", "stochastic_stratified_quantile"],
    )
    parser.add_argument("--cdro-warmup-fraction", type=float, default=ToyConfig.cdro_warmup_fraction)
    parser.add_argument("--control-radius-kappa", type=float, default=ToyConfig.control_radius_kappa)
    parser.add_argument("--v21-rho", type=float, default=ToyConfig.v21_rho)
    parser.add_argument("--use-time-dependent-kappa", action="store_true", default=ToyConfig.use_time_dependent_kappa)
    parser.add_argument("--disable-time-dependent-kappa", action="store_true")
    parser.add_argument("--kappa-low-multiplier", type=float, default=ToyConfig.kappa_low_multiplier)
    parser.add_argument("--kappa-mid-multiplier", type=float, default=ToyConfig.kappa_mid_multiplier)
    parser.add_argument("--kappa-high-multiplier", type=float, default=ToyConfig.kappa_high_multiplier)
    parser.add_argument("--disable-kappa-preserve-l2-budget", action="store_true")
    parser.add_argument("--outer-attack-weight", type=float, default=ToyConfig.outer_attack_weight)
    parser.add_argument("--outer-clean-weight", type=float, default=ToyConfig.outer_clean_weight)
    parser.add_argument("--warmup-clean-steps", type=int, default=ToyConfig.warmup_clean_steps)
    parser.add_argument("--warmup-ramp-steps", type=int, default=ToyConfig.warmup_ramp_steps)
    parser.add_argument("--warmup-attack-weight-start", type=float, default=ToyConfig.warmup_attack_weight_start)
    parser.add_argument("--warmup-phi-lr-scale-start", type=float, default=ToyConfig.warmup_phi_lr_scale_start)
    parser.add_argument("--wild-update-interval", type=int, default=ToyConfig.wild_update_interval)
    parser.add_argument("--wild-cache-batches", type=int, default=ToyConfig.wild_cache_batches)
    parser.add_argument("--wild-inner-steps", type=int, default=ToyConfig.wild_inner_steps)
    parser.add_argument("--wild-step-size", type=float, default=ToyConfig.wild_step_size)
    parser.add_argument("--wild-gamma", type=float, default=ToyConfig.wild_gamma)
    parser.add_argument("--wild-fixed-noise-inner", action="store_true", default=ToyConfig.wild_fixed_noise_inner)
    parser.add_argument("--disable-wild-fixed-noise-inner", action="store_true")
    parser.add_argument("--wild-clamp-samples", action="store_true", default=ToyConfig.wild_clamp_samples)
    parser.add_argument("--wild-sample-min", type=float, default=ToyConfig.wild_sample_min)
    parser.add_argument("--wild-sample-max", type=float, default=ToyConfig.wild_sample_max)
    parser.add_argument("--wild-delta-ratio-denom", type=float, default=ToyConfig.wild_delta_ratio_denom)
    parser.add_argument("--wdro-warmup-fraction", type=float, default=ToyConfig.wdro_warmup_fraction)
    parser.add_argument("--wdro-refresh-epochs", type=float, default=ToyConfig.wdro_refresh_epochs)
    parser.add_argument("--wdro-adv-prob", type=float, default=ToyConfig.wdro_adv_prob)
    parser.add_argument("--wdro-attack-steps", type=int, default=ToyConfig.wdro_attack_steps)
    parser.add_argument("--wdro-attack-step-size", type=float, default=ToyConfig.wdro_attack_step_size)
    parser.add_argument("--wdro-gamma", type=float, default=ToyConfig.wdro_gamma)
    parser.add_argument("--wdro-clamp-samples", action="store_true", default=ToyConfig.wdro_clamp_samples)
    parser.add_argument("--wdro-sample-min", type=float, default=ToyConfig.wdro_sample_min)
    parser.add_argument("--wdro-sample-max", type=float, default=ToyConfig.wdro_sample_max)
    parser.add_argument("--collapse-diag-every", type=int, default=ToyConfig.collapse_diag_every)
    parser.add_argument("--collapse-gap-ratio-tol", type=float, default=ToyConfig.collapse_gap_ratio_tol)
    parser.add_argument("--collapse-delta-ratio-tol", type=float, default=ToyConfig.collapse_delta_ratio_tol)
    parser.add_argument("--constraint-saturation-threshold", type=float, default=ToyConfig.constraint_saturation_threshold)
    parser.add_argument("--collapse-v-l2-tol", type=float, default=None)
    parser.add_argument("--disable-collapse-diagnostics", action="store_true")
    parser.add_argument("--baseline-gate-enabled", action="store_true", default=ToyConfig.baseline_gate_enabled)
    parser.add_argument("--disable-baseline-gate", action="store_true")
    parser.add_argument("--baseline-gate-error-on-fail", action="store_true", default=ToyConfig.baseline_gate_error_on_fail)
    parser.add_argument("--baseline-gate-min-coverage", type=float, default=ToyConfig.baseline_gate_min_coverage)
    parser.add_argument(
        "--baseline-gate-max-generated-avg-min-dist",
        type=float,
        default=ToyConfig.baseline_gate_max_generated_avg_min_dist,
    )
    parser.add_argument(
        "--baseline-gate-max-generated-p90-min-dist",
        type=float,
        default=ToyConfig.baseline_gate_max_generated_p90_min_dist,
    )
    parser.add_argument(
        "--baseline-gate-max-endpoint-avg-min-dist",
        type=float,
        default=ToyConfig.baseline_gate_max_endpoint_avg_min_dist,
    )
    parser.add_argument(
        "--baseline-gate-max-endpoint-p90-min-dist",
        type=float,
        default=ToyConfig.baseline_gate_max_endpoint_p90_min_dist,
    )
    parser.add_argument("--reverse-terminal-assert-tol", type=float, default=ToyConfig.reverse_terminal_assert_tol)

    parser.add_argument("--run-checks", action="store_true", default=ToyConfig.run_checks)
    parser.add_argument("--skip-checks", action="store_true")
    parser.add_argument("--gradcheck-eps", type=float, default=ToyConfig.gradcheck_eps)
    parser.add_argument("--gradcheck-tol", type=float, default=ToyConfig.gradcheck_tol)
    return parser


def parse_toy_config(argv: Optional[Sequence[str]] = None) -> ToyConfig:
    """Parse CLI, resolve legacy flags, auto-fill sigma sampling params, and validate."""

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    args_dict = vars(args).copy()

    skip_checks = bool(args_dict.pop("skip_checks"))
    disable_lognorm = bool(args_dict.pop("disable_log_normal_sigma_sampling"))
    disable_auto_lognorm = bool(args_dict.pop("disable_auto_log_normal_params"))
    disable_tf32 = bool(args_dict.pop("disable_tf32"))
    disable_cudnn_benchmark = bool(args_dict.pop("disable_cudnn_benchmark"))
    disable_baseline_ckpt = bool(args_dict.pop("disable_baseline_ckpt"))
    disable_baseline_ckpt_strict_meta = bool(args_dict.pop("disable_baseline_ckpt_strict_meta"))
    disable_baseline_gate = bool(args_dict.pop("disable_baseline_gate"))
    disable_eval_shared_terminal_noise = bool(args_dict.pop("disable_eval_shared_terminal_noise"))
    disable_eval_shared_reverse_noise = bool(args_dict.pop("disable_eval_shared_reverse_noise"))
    force_det_plot = bool(args_dict.pop("plot_deterministic_backward"))
    disable_ema_eval = bool(args_dict.pop("disable_ema_eval"))
    disable_ema_rampup = bool(args_dict.pop("disable_ema_rampup"))
    disable_limited_data = bool(args_dict.pop("disable_limited_data"))
    disable_mnist_percent_split = bool(args_dict.pop("disable_mnist_percent_split"))
    disable_collapse_diagnostics = bool(args_dict.pop("disable_collapse_diagnostics"))
    disable_time_dependent_kappa = bool(args_dict.pop("disable_time_dependent_kappa"))
    disable_kappa_preserve_l2_budget = bool(args_dict.pop("disable_kappa_preserve_l2_budget"))
    disable_wild_fixed_noise_inner = bool(args_dict.pop("disable_wild_fixed_noise_inner"))
    collapse_v_l2_tol_legacy = args_dict.pop("collapse_v_l2_tol")
    args_dict["wandb_enabled"] = bool(args_dict.pop("wandb"))

    cfg = ToyConfig(**args_dict)
    if skip_checks:
        cfg.run_checks = False
    if disable_lognorm:
        cfg.use_log_normal_sigma_sampling = False
    if disable_auto_lognorm:
        cfg.auto_log_normal_params = False
    if disable_tf32:
        cfg.allow_tf32 = False
    if disable_cudnn_benchmark:
        cfg.cudnn_benchmark = False
    if disable_baseline_gate:
        cfg.baseline_gate_enabled = False
    if disable_baseline_ckpt:
        cfg.baseline_ckpt_enabled = False
    if disable_baseline_ckpt_strict_meta:
        cfg.baseline_ckpt_strict_meta = False
    if disable_eval_shared_terminal_noise:
        cfg.eval_use_shared_terminal_noise = False
    if disable_eval_shared_reverse_noise:
        cfg.eval_use_shared_reverse_noise = False
    if force_det_plot:
        cfg.plot_stochastic_backward = False
    if disable_ema_eval:
        cfg.use_ema_eval = False
    if disable_ema_rampup:
        cfg.ema_rampup_ratio = None
    if disable_limited_data:
        cfg.limited_data_enabled = False
    if disable_mnist_percent_split:
        cfg.mnist_use_percent_split = False
    if disable_collapse_diagnostics:
        cfg.collapse_diagnostics_enabled = False
    if disable_time_dependent_kappa:
        cfg.use_time_dependent_kappa = False
    if disable_kappa_preserve_l2_budget:
        cfg.kappa_preserve_l2_budget = False
    if disable_wild_fixed_noise_inner:
        cfg.wild_fixed_noise_inner = False
    if collapse_v_l2_tol_legacy is not None:
        cfg.collapse_delta_ratio_tol = float(collapse_v_l2_tol_legacy)
    if cfg.use_log_normal_sigma_sampling and cfg.auto_log_normal_params:
        log_min = math.log(cfg.sigma_min)
        log_max = math.log(cfg.sigma_max)
        cfg.p_mean = 0.5 * (log_min + log_max)
        cfg.p_std = max((log_max - log_min) / 6.0, 1e-3)
    _validate_config(cfg)
    return cfg
