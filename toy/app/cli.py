import argparse
import math
from typing import Optional, Sequence

from ..config import ToyConfig


def _validate_config(cfg: ToyConfig) -> None:
    """Fail-fast validation for impossible or degenerate training settings."""

    if cfg.steps <= 0:
        raise ValueError(f"--steps must be > 0, got {cfg.steps}")
    if cfg.batch_size <= 0:
        raise ValueError(f"--batch-size must be > 0, got {cfg.batch_size}")
    if cfg.inner_steps < 0:
        raise ValueError(f"--inner-steps must be >= 0, got {cfg.inner_steps}")
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
    if cfg.outer_attack_weight < 0 or cfg.outer_clean_weight < 0:
        raise ValueError(
            "outer loss weights must be non-negative, got "
            f"attack={cfg.outer_attack_weight}, clean={cfg.outer_clean_weight}"
        )
    if cfg.outer_attack_weight == 0 and cfg.outer_clean_weight == 0:
        raise ValueError("Both outer weights are zero; training objective is identically zero.")

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
    parser.add_argument("--device", type=str, default=ToyConfig.device, choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dataset-kind", type=str, default=ToyConfig.dataset_kind)

    parser.add_argument("--steps", type=int, default=ToyConfig.steps)
    parser.add_argument("--batch-size", type=int, default=ToyConfig.batch_size)
    parser.add_argument("--log-every", type=int, default=ToyConfig.log_every)
    parser.add_argument("--eval-samples", type=int, default=ToyConfig.eval_samples)
    parser.add_argument("--debug-eval-batch", type=int, default=ToyConfig.debug_eval_batch)
    parser.add_argument("--debug-terminal-step", type=int, default=ToyConfig.debug_terminal_step)
    parser.add_argument("--plot-stochastic-backward", action="store_true", default=ToyConfig.plot_stochastic_backward)
    parser.add_argument("--plot-deterministic-backward", action="store_true")
    parser.add_argument("--baseline-only", action="store_true", default=ToyConfig.baseline_only)
    parser.add_argument("--use-ema-eval", action="store_true", default=ToyConfig.use_ema_eval)
    parser.add_argument("--disable-ema-eval", action="store_true")
    parser.add_argument("--ema-decay", type=float, default=ToyConfig.ema_decay)

    parser.add_argument("--hidden-dim", type=int, default=ToyConfig.hidden_dim)
    parser.add_argument("--lr-theta", type=float, default=ToyConfig.lr_theta)
    parser.add_argument("--lr-phi", type=float, default=ToyConfig.lr_phi)
    parser.add_argument("--inner-steps", type=int, default=ToyConfig.inner_steps)
    parser.add_argument("--clip-phi-grad", type=float, default=ToyConfig.clip_phi_grad)

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
    parser.add_argument("--control-radius-kappa", type=float, default=ToyConfig.control_radius_kappa)
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
    disable_baseline_gate = bool(args_dict.pop("disable_baseline_gate"))
    force_det_plot = bool(args_dict.pop("plot_deterministic_backward"))
    disable_ema_eval = bool(args_dict.pop("disable_ema_eval"))
    disable_limited_data = bool(args_dict.pop("disable_limited_data"))
    disable_collapse_diagnostics = bool(args_dict.pop("disable_collapse_diagnostics"))
    disable_time_dependent_kappa = bool(args_dict.pop("disable_time_dependent_kappa"))
    disable_kappa_preserve_l2_budget = bool(args_dict.pop("disable_kappa_preserve_l2_budget"))
    collapse_v_l2_tol_legacy = args_dict.pop("collapse_v_l2_tol")

    cfg = ToyConfig(**args_dict)
    if skip_checks:
        cfg.run_checks = False
    if disable_lognorm:
        cfg.use_log_normal_sigma_sampling = False
    if disable_auto_lognorm:
        cfg.auto_log_normal_params = False
    if disable_baseline_gate:
        cfg.baseline_gate_enabled = False
    if force_det_plot:
        cfg.plot_stochastic_backward = False
    if disable_ema_eval:
        cfg.use_ema_eval = False
    if disable_limited_data:
        cfg.limited_data_enabled = False
    if disable_collapse_diagnostics:
        cfg.collapse_diagnostics_enabled = False
    if disable_time_dependent_kappa:
        cfg.use_time_dependent_kappa = False
    if disable_kappa_preserve_l2_budget:
        cfg.kappa_preserve_l2_budget = False
    if collapse_v_l2_tol_legacy is not None:
        cfg.collapse_delta_ratio_tol = float(collapse_v_l2_tol_legacy)
    if cfg.use_log_normal_sigma_sampling and cfg.auto_log_normal_params:
        log_min = math.log(cfg.sigma_min)
        log_max = math.log(cfg.sigma_max)
        cfg.p_mean = 0.5 * (log_min + log_max)
        cfg.p_std = max((log_max - log_min) / 6.0, 1e-3)
    _validate_config(cfg)
    return cfg
