#!/usr/bin/env python3
import argparse
import json
import math
import os
import sys

import torch

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from toy.checks import run_preflight_checks
    from toy.config import ToyConfig
    from toy.data import (
        build_circle_gmm_centers,
        build_fixed_gmm_dataset_per_mode,
        build_random_gmm_dataset,
        sample_from_pool,
        sample_gmm,
    )
    from toy.diffusion import build_sigma_levels, rollout_controlled_ve, sample_target_indices
    from toy.metrics import (
        compute_denoise_error_curves,
        compute_paired_reverse_delta_by_step,
        compute_path_mse_by_step,
        compute_x0_recovery_vs_terminal_step,
        estimate_bayes_posterior_mean_mse,
        evaluate_baseline_gate,
        evaluate_nearest_reference_distance,
        evaluate_mode_coverage,
    )
    from toy.models import ControlNet, ToyEDMDenoiser
    from toy.plotting import (
        plot_debug_losses,
        plot_forward_timestep_clouds,
    )
    from toy.trainer import (
        reverse_paths_from_terminal,
        train_baseline,
        train_trajectory_robust_energy,
    )
    from toy.utils import as_jsonable_metrics, ensure_dir, pick_device, set_seed, tensor_to_numpy
else:
    from .checks import run_preflight_checks
    from .config import ToyConfig
    from .data import (
        build_circle_gmm_centers,
        build_fixed_gmm_dataset_per_mode,
        build_random_gmm_dataset,
        sample_from_pool,
        sample_gmm,
    )
    from .diffusion import build_sigma_levels, rollout_controlled_ve, sample_target_indices
    from .metrics import (
        compute_denoise_error_curves,
        compute_paired_reverse_delta_by_step,
        compute_path_mse_by_step,
        compute_x0_recovery_vs_terminal_step,
        estimate_bayes_posterior_mean_mse,
        evaluate_baseline_gate,
        evaluate_nearest_reference_distance,
        evaluate_mode_coverage,
    )
    from .models import ControlNet, ToyEDMDenoiser
    from .plotting import (
        plot_debug_losses,
        plot_forward_timestep_clouds,
    )
    from .trainer import (
        reverse_paths_from_terminal,
        train_baseline,
        train_trajectory_robust_energy,
    )
    from .utils import as_jsonable_metrics, ensure_dir, pick_device, set_seed, tensor_to_numpy


def parse_args() -> ToyConfig:
    parser = argparse.ArgumentParser(description="Toy trajectory-perturbed robust diffusion (energy-only).")
    parser.add_argument("--outdir", type=str, default=ToyConfig.outdir)
    parser.add_argument("--exp-name", type=str, default=ToyConfig.exp_name)
    parser.add_argument("--seed", type=int, default=ToyConfig.seed)
    parser.add_argument("--device", type=str, default=ToyConfig.device, choices=["auto", "cpu", "cuda"])

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
    parser.add_argument("--use-log-normal-sigma-sampling", action="store_true", default=ToyConfig.use_log_normal_sigma_sampling)
    parser.add_argument("--disable-log-normal-sigma-sampling", action="store_true")
    parser.add_argument("--auto-log-normal-params", action="store_true", default=ToyConfig.auto_log_normal_params)
    parser.add_argument("--disable-auto-log-normal-params", action="store_true")
    parser.add_argument("--p-mean", type=float, default=ToyConfig.p_mean)
    parser.add_argument("--p-std", type=float, default=ToyConfig.p_std)

    parser.add_argument("--lambda-energy", type=float, default=ToyConfig.lambda_energy)
    parser.add_argument("--outer-attack-weight", type=float, default=ToyConfig.outer_attack_weight)
    parser.add_argument("--outer-clean-weight", type=float, default=ToyConfig.outer_clean_weight)
    parser.add_argument("--warmup-clean-steps", type=int, default=ToyConfig.warmup_clean_steps)
    parser.add_argument("--warmup-ramp-steps", type=int, default=ToyConfig.warmup_ramp_steps)
    parser.add_argument("--warmup-attack-weight-start", type=float, default=ToyConfig.warmup_attack_weight_start)
    parser.add_argument("--warmup-phi-lr-scale-start", type=float, default=ToyConfig.warmup_phi_lr_scale_start)
    parser.add_argument("--collapse-diag-every", type=int, default=ToyConfig.collapse_diag_every)
    parser.add_argument("--collapse-gap-ratio-tol", type=float, default=ToyConfig.collapse_gap_ratio_tol)
    parser.add_argument("--collapse-v-l2-tol", type=float, default=ToyConfig.collapse_v_l2_tol)
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
    parser.add_argument(
        "--reverse-terminal-assert-tol",
        type=float,
        default=ToyConfig.reverse_terminal_assert_tol,
    )

    parser.add_argument("--run-checks", action="store_true", default=ToyConfig.run_checks)
    parser.add_argument("--skip-checks", action="store_true")
    parser.add_argument("--gradcheck-eps", type=float, default=ToyConfig.gradcheck_eps)
    parser.add_argument("--gradcheck-tol", type=float, default=ToyConfig.gradcheck_tol)

    args = parser.parse_args()
    args_dict = vars(args).copy()
    skip_checks = bool(args_dict.pop("skip_checks"))
    disable_lognorm = bool(args_dict.pop("disable_log_normal_sigma_sampling"))
    disable_auto_lognorm = bool(args_dict.pop("disable_auto_log_normal_params"))
    disable_baseline_gate = bool(args_dict.pop("disable_baseline_gate"))
    force_det_plot = bool(args_dict.pop("plot_deterministic_backward"))
    disable_ema_eval = bool(args_dict.pop("disable_ema_eval"))
    disable_limited_data = bool(args_dict.pop("disable_limited_data"))
    disable_collapse_diagnostics = bool(args_dict.pop("disable_collapse_diagnostics"))
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
    if cfg.use_log_normal_sigma_sampling and cfg.auto_log_normal_params:
        log_min = math.log(cfg.sigma_min)
        log_max = math.log(cfg.sigma_max)
        cfg.p_mean = 0.5 * (log_min + log_max)
        cfg.p_std = max((log_max - log_min) / 6.0, 1e-3)
    return cfg


def estimate_sigma_data(centers, data_std: float) -> float:
    # EDM sigma_data is the expected per-dimension std of clean data.
    center_var = centers.var(dim=0, unbiased=False).mean()
    return float(torch.sqrt(center_var + data_std ** 2).item())


def summarize_series(values, tail: int = 200) -> dict:
    if len(values) == 0:
        return {"final": None, "min": None, "max": None, "mean_last": None}
    t = min(tail, len(values))
    return {
        "final": float(values[-1]),
        "min": float(min(values)),
        "max": float(max(values)),
        "mean_last": float(sum(values[-t:]) / t),
    }


def _mean(values) -> float:
    if len(values) == 0:
        return float("nan")
    return float(sum(values) / len(values))


def summarize_train_vs_val_curve_gaps(train_curves: dict, val_curves: dict) -> dict:
    keys = [
        "baseline_on_forward_baseline",
        "baseline_on_forward_attack",
        "robust_on_forward_baseline",
        "robust_on_forward_attack",
    ]
    out = {}
    for key in keys:
        tvals = train_curves.get(key, [])
        vvals = val_curves.get(key, [])
        if len(tvals) == 0 or len(vvals) == 0 or len(tvals) != len(vvals):
            out[key] = {
                "mean_gap_val_minus_train": None,
                "terminal_gap_val_minus_train": None,
            }
            continue
        out[key] = {
            "mean_gap_val_minus_train": float(_mean(vvals) - _mean(tvals)),
            "terminal_gap_val_minus_train": float(vvals[-1] - tvals[-1]),
        }
    return out


def compute_terminal_match_stats(reverse_paths: torch.Tensor, forward_paths: torch.Tensor) -> dict:
    # reverse_paths / forward_paths: [B, N+1, 2], terminal step is index -1.
    delta = reverse_paths[:, -1] - forward_paths[:, -1]
    l2 = torch.sqrt(delta.pow(2).sum(dim=1))
    return {
        "mean_l2": float(l2.mean().item()),
        "max_l2": float(l2.max().item()),
    }


def main() -> None:
    cfg = parse_args()
    device = pick_device(cfg.device)
    set_seed(cfg.seed)

    ensure_dir(cfg.outdir)
    exp_dir = os.path.join(cfg.outdir, cfg.exp_name)
    ensure_dir(exp_dir)
    print(f"[info] device={device}", flush=True)
    print(f"[info] exp_dir={exp_dir}", flush=True)

    centers = build_circle_gmm_centers(cfg.n_modes, cfg.mode_radius, device=device)
    if cfg.sigma_data <= 0:
        cfg.sigma_data = estimate_sigma_data(centers, cfg.data_std)
    print(f"[info] sigma_data={cfg.sigma_data:.6f}", flush=True)
    centers_np = tensor_to_numpy(centers)
    sigma_levels = build_sigma_levels(cfg.sigma_min, cfg.sigma_max, cfg.n_steps_path, device=device)

    train_pool = None
    train_pool_labels = None
    if cfg.limited_data_enabled:
        train_pool, train_pool_labels = build_fixed_gmm_dataset_per_mode(
            centers=centers,
            std=cfg.data_std,
            points_per_mode=cfg.train_points_per_mode,
            seed=cfg.seed,
        )
        print(
            "[info] limited_data_enabled=True "
            f"train_pool_size={train_pool.shape[0]} "
            f"({cfg.train_points_per_mode} points/mode x {cfg.n_modes} modes)",
            flush=True,
        )
    else:
        print("[info] limited_data_enabled=False (population sampling mode)", flush=True)

    val_pool, _ = build_random_gmm_dataset(
        total_samples=cfg.val_size,
        centers=centers,
        std=cfg.data_std,
        seed=cfg.seed + 1001,
    )
    print(f"[info] val_pool_size={val_pool.shape[0]}", flush=True)

    baseline = ToyEDMDenoiser(cfg.hidden_dim, sigma_data=cfg.sigma_data).to(device)
    robust = ToyEDMDenoiser(cfg.hidden_dim, sigma_data=cfg.sigma_data).to(device)
    control = ControlNet(cfg.hidden_dim).to(device)

    check_report = {}
    if cfg.run_checks:
        check_report = run_preflight_checks(robust, control, centers, sigma_levels, cfg)
        print("[check] preflight passed", flush=True)
        print(
            f"[check] grad_rel_err denoiser={check_report['gradcheck_denoiser']['relative_error']:.3e} "
            f"control={check_report['gradcheck_control']['relative_error']:.3e}",
            flush=True,
        )

    history_baseline, baseline_eval = train_baseline(
        baseline,
        centers,
        sigma_levels,
        cfg,
        train_pool=train_pool,
    )
    n_steps = int(sigma_levels.numel() - 1)
    gate_terminal_step = cfg.debug_terminal_step
    if gate_terminal_step < 1 or gate_terminal_step > n_steps:
        gate_terminal_step = n_steps
    sigma_levels_gate = sigma_levels[: gate_terminal_step + 1]

    # Baseline gate diagnostics (computed before attack training).
    x_gate = sample_from_pool(val_pool, cfg.debug_eval_batch)
    idx_gate = sample_target_indices(cfg.debug_eval_batch, sigma_levels)
    gate_roll = rollout_controlled_ve(
        x0=x_gate,
        target_indices=idx_gate,
        control_net=control,
        sigma_levels=sigma_levels,
        grad_through_control=False,
    )
    gate_ref_paths = gate_roll.states_ref[:, : gate_terminal_step + 1]
    gate_rev_det = reverse_paths_from_terminal(
        denoiser=baseline_eval,
        x_terminal=gate_ref_paths[:, -1],
        sigma_levels=sigma_levels_gate,
        stochastic=False,
    )
    gate_endpoint_mode_metrics = evaluate_mode_coverage(tensor_to_numpy(gate_rev_det[:, 0]), centers_np)
    gate_gen_paths = reverse_paths_from_terminal(
        denoiser=baseline_eval,
        x_terminal=torch.randn(cfg.eval_samples, 2, device=device) * sigma_levels[-1],
        sigma_levels=sigma_levels,
        stochastic=True,
    )
    gate_generated_mode_metrics = evaluate_mode_coverage(tensor_to_numpy(gate_gen_paths[:, 0]), centers_np)
    baseline_gate = evaluate_baseline_gate(
        generated_mode_metrics=gate_generated_mode_metrics,
        endpoint_mode_metrics=gate_endpoint_mode_metrics,
        min_coverage=cfg.baseline_gate_min_coverage,
        max_generated_avg_min_dist=cfg.baseline_gate_max_generated_avg_min_dist,
        max_generated_p90_min_dist=cfg.baseline_gate_max_generated_p90_min_dist,
        max_endpoint_avg_min_dist=cfg.baseline_gate_max_endpoint_avg_min_dist,
        max_endpoint_p90_min_dist=cfg.baseline_gate_max_endpoint_p90_min_dist,
    )

    attack_training_executed = False
    if cfg.baseline_only:
        with torch.no_grad():
            for p in control.parameters():
                p.zero_()
        robust = baseline_eval
        history_robust = {
            "outer_loss": [],
            "outer_loss_attack": [],
            "outer_loss_clean": [],
            "inner_obj": [],
            "energy": [],
            "sched_attack_weight": [],
            "sched_clean_weight": [],
            "sched_phi_lr_scale": [],
            "diag_step": [],
            "diag_inner_obj_current": [],
            "diag_inner_obj_zero": [],
            "diag_inner_obj_gap": [],
            "diag_inner_obj_gap_ratio": [],
            "diag_v_l2_mean": [],
            "diag_path_delta_mean": [],
            "diag_terminal_delta_mean": [],
        }
    elif cfg.baseline_gate_enabled and not baseline_gate["passed"]:
        reason_lines = [
            f"{c['name']}: value={c['value']:.4f} {c['op']} {c['threshold']:.4f} (FAIL)"
            for c in baseline_gate["failed_checks"]
        ]
        msg = "Baseline acceptance gate failed. Attack training skipped.\n  - " + "\n  - ".join(reason_lines)
        if cfg.baseline_gate_error_on_fail:
            raise RuntimeError(msg)
        print(f"[gate] {msg}", flush=True)
        with torch.no_grad():
            for p in control.parameters():
                p.zero_()
        robust = baseline_eval
        history_robust = {
            "outer_loss": [],
            "outer_loss_attack": [],
            "outer_loss_clean": [],
            "inner_obj": [],
            "energy": [],
            "sched_attack_weight": [],
            "sched_clean_weight": [],
            "sched_phi_lr_scale": [],
            "diag_step": [],
            "diag_inner_obj_current": [],
            "diag_inner_obj_zero": [],
            "diag_inner_obj_gap": [],
            "diag_inner_obj_gap_ratio": [],
            "diag_v_l2_mean": [],
            "diag_path_delta_mean": [],
            "diag_terminal_delta_mean": [],
        }
    else:
        attack_training_executed = True
        history_robust = train_trajectory_robust_energy(
            robust,
            control,
            centers,
            sigma_levels,
            cfg,
            train_pool=train_pool,
        )

    # Evaluate denoise curves on train-pool and held-out pools to expose overfitting under limited data.
    if train_pool is not None:
        x_train_eval = sample_from_pool(train_pool, cfg.debug_eval_batch)
    else:
        x_train_eval = sample_gmm(cfg.debug_eval_batch, centers, cfg.data_std)
    idx_train_eval = sample_target_indices(cfg.debug_eval_batch, sigma_levels)
    train_roll = rollout_controlled_ve(
        x0=x_train_eval,
        target_indices=idx_train_eval,
        control_net=control,
        sigma_levels=sigma_levels,
        grad_through_control=False,
    )
    denoise_error_curves_train = compute_denoise_error_curves(
        baseline_model=baseline_eval,
        robust_model=robust,
        x0=x_train_eval,
        states_ref=train_roll.states_ref,
        states_ctrl=train_roll.states_ctrl,
        sigma_levels=sigma_levels,
    )

    x_val_eval = sample_from_pool(val_pool, cfg.debug_eval_batch)
    idx_val_eval = sample_target_indices(cfg.debug_eval_batch, sigma_levels)
    val_roll = rollout_controlled_ve(
        x0=x_val_eval,
        target_indices=idx_val_eval,
        control_net=control,
        sigma_levels=sigma_levels,
        grad_through_control=False,
    )
    denoise_error_curves_val = compute_denoise_error_curves(
        baseline_model=baseline_eval,
        robust_model=robust,
        x0=x_val_eval,
        states_ref=val_roll.states_ref,
        states_ctrl=val_roll.states_ctrl,
        sigma_levels=sigma_levels,
    )
    denoise_error_curves = denoise_error_curves_val
    x_demo = x_val_eval
    demo_roll = val_roll
    n_steps = int(sigma_levels.numel() - 1)
    terminal_step = cfg.debug_terminal_step
    if terminal_step < 1 or terminal_step > n_steps:
        terminal_step = n_steps
    sigma_levels_plot = sigma_levels[: terminal_step + 1]
    ref_paths_plot = demo_roll.states_ref[:, : terminal_step + 1]
    ctrl_paths_plot = demo_roll.states_ctrl[:, : terminal_step + 1]
    shared_reverse_noise = None
    if cfg.plot_stochastic_backward:
        # Fair comparison: baseline/attack reverse use the same stochastic increments.
        shared_reverse_noise = torch.randn(
            terminal_step + 1,
            ref_paths_plot.shape[0],
            ref_paths_plot.shape[2],
            device=device,
            dtype=ref_paths_plot.dtype,
        )
    # Deterministic reverse for pairwise-recovery diagnostics.
    rev_baseline_from_ref = reverse_paths_from_terminal(
        denoiser=baseline_eval,
        x_terminal=ref_paths_plot[:, -1],
        sigma_levels=sigma_levels_plot,
        stochastic=False,
    )
    rev_baseline_from_attack = reverse_paths_from_terminal(
        denoiser=baseline_eval,
        x_terminal=ctrl_paths_plot[:, -1],
        sigma_levels=sigma_levels_plot,
        stochastic=False,
    )
    # Optional stochastic reverse for visualization of clustered generative behavior.
    rev_baseline_from_ref_plot = reverse_paths_from_terminal(
        denoiser=baseline_eval,
        x_terminal=ref_paths_plot[:, -1],
        sigma_levels=sigma_levels_plot,
        stochastic=cfg.plot_stochastic_backward,
        noise_schedule=shared_reverse_noise,
    )
    rev_baseline_from_attack_plot = reverse_paths_from_terminal(
        denoiser=baseline_eval,
        x_terminal=ctrl_paths_plot[:, -1],
        sigma_levels=sigma_levels_plot,
        stochastic=cfg.plot_stochastic_backward,
        noise_schedule=shared_reverse_noise,
    )
    paired_reverse_delta_plot = compute_paired_reverse_delta_by_step(
        reverse_paths_ref=rev_baseline_from_ref_plot,
        reverse_paths_attack=rev_baseline_from_attack_plot,
    )
    paired_reverse_delta_det = compute_paired_reverse_delta_by_step(
        reverse_paths_ref=rev_baseline_from_ref,
        reverse_paths_attack=rev_baseline_from_attack,
    )
    rev_baseline_from_ref_np = tensor_to_numpy(rev_baseline_from_ref_plot)
    rev_baseline_from_attack_np = tensor_to_numpy(rev_baseline_from_attack_plot)

    terminal_consistency = {
        "tolerance": float(cfg.reverse_terminal_assert_tol),
        "plot_ref": compute_terminal_match_stats(rev_baseline_from_ref_plot, ref_paths_plot),
        "plot_attack": compute_terminal_match_stats(rev_baseline_from_attack_plot, ctrl_paths_plot),
        "det_ref": compute_terminal_match_stats(rev_baseline_from_ref, ref_paths_plot),
        "det_attack": compute_terminal_match_stats(rev_baseline_from_attack, ctrl_paths_plot),
    }
    terminal_consistency["passed"] = bool(
        terminal_consistency["plot_ref"]["max_l2"] <= cfg.reverse_terminal_assert_tol
        and terminal_consistency["plot_attack"]["max_l2"] <= cfg.reverse_terminal_assert_tol
        and terminal_consistency["det_ref"]["max_l2"] <= cfg.reverse_terminal_assert_tol
        and terminal_consistency["det_attack"]["max_l2"] <= cfg.reverse_terminal_assert_tol
    )
    if not terminal_consistency["passed"]:
        raise RuntimeError(
            "Reverse terminal consistency check failed: "
            f"plot_ref.max_l2={terminal_consistency['plot_ref']['max_l2']:.3e}, "
            f"plot_attack.max_l2={terminal_consistency['plot_attack']['max_l2']:.3e}, "
            f"det_ref.max_l2={terminal_consistency['det_ref']['max_l2']:.3e}, "
            f"det_attack.max_l2={terminal_consistency['det_attack']['max_l2']:.3e}, "
            f"tol={cfg.reverse_terminal_assert_tol:.3e}"
        )

    recovery_ref_curve = compute_x0_recovery_vs_terminal_step(
        denoiser=baseline_eval,
        forward_paths=demo_roll.states_ref,
        sigma_levels=sigma_levels,
        reverse_fn=reverse_paths_from_terminal,
    )
    recovery_attack_curve = compute_x0_recovery_vs_terminal_step(
        denoiser=baseline_eval,
        forward_paths=demo_roll.states_ctrl,
        sigma_levels=sigma_levels,
        reverse_fn=reverse_paths_from_terminal,
    )
    bayes_terminal_mse = estimate_bayes_posterior_mean_mse(
        centers=centers_np,
        data_std=cfg.data_std,
        sigma=float(sigma_levels[terminal_step].item()),
        n_samples=50000,
        seed=cfg.seed,
    )
    baseline_gen_paths = reverse_paths_from_terminal(
        denoiser=baseline_eval,
        x_terminal=torch.randn(cfg.eval_samples, 2, device=device) * sigma_levels[-1],
        sigma_levels=sigma_levels,
        stochastic=True,
    )
    baseline_gen_np = tensor_to_numpy(baseline_gen_paths[:, 0])
    robust_gen_paths = reverse_paths_from_terminal(
        denoiser=robust,
        x_terminal=torch.randn(cfg.eval_samples, 2, device=device) * sigma_levels[-1],
        sigma_levels=sigma_levels,
        stochastic=True,
    )
    robust_gen_np = tensor_to_numpy(robust_gen_paths[:, 0])
    val_pool_np = tensor_to_numpy(val_pool)
    train_pool_np = tensor_to_numpy(train_pool) if train_pool is not None else None
    denoise_gap = summarize_train_vs_val_curve_gaps(denoise_error_curves_train, denoise_error_curves_val)
    diag_gap_ratio_summary = summarize_series(history_robust.get("diag_inner_obj_gap_ratio", []))
    diag_v_l2_summary = summarize_series(history_robust.get("diag_v_l2_mean", []))
    collapse_suspected = None
    if diag_gap_ratio_summary["mean_last"] is not None and diag_v_l2_summary["mean_last"] is not None:
        collapse_suspected = bool(
            abs(diag_gap_ratio_summary["mean_last"]) <= float(cfg.collapse_gap_ratio_tol)
            and diag_v_l2_summary["mean_last"] <= float(cfg.collapse_v_l2_tol)
        )

    metrics = {
        "dataset_debug": {
            "limited_data_enabled": bool(cfg.limited_data_enabled),
            "train_points_per_mode": int(cfg.train_points_per_mode),
            "train_pool_size": int(train_pool.shape[0]) if train_pool is not None else None,
            "val_pool_size": int(val_pool.shape[0]),
            "train_mode_counts": (
                [int(v) for v in torch.bincount(train_pool_labels, minlength=cfg.n_modes).detach().cpu().tolist()]
                if train_pool_labels is not None
                else None
            ),
        },
        "baseline_gate": {
            "enabled": bool(cfg.baseline_gate_enabled),
            "error_on_fail": bool(cfg.baseline_gate_error_on_fail),
            "attack_training_executed": bool(attack_training_executed),
            **baseline_gate,
        },
        "objective_debug": {
            "baseline_loss": summarize_series(history_baseline["loss"]),
            "baseline_sigma_counts": history_baseline.get("sigma_counts", []),
            "robust_outer_loss": summarize_series(history_robust["outer_loss"]),
            "robust_outer_loss_attack": summarize_series(history_robust.get("outer_loss_attack", [])),
            "robust_outer_loss_clean": summarize_series(history_robust.get("outer_loss_clean", [])),
            "robust_inner_obj": summarize_series(history_robust["inner_obj"]),
            "robust_energy": summarize_series(history_robust["energy"]),
            "robust_sched_attack_weight": summarize_series(history_robust.get("sched_attack_weight", [])),
            "robust_sched_clean_weight": summarize_series(history_robust.get("sched_clean_weight", [])),
            "robust_sched_phi_lr_scale": summarize_series(history_robust.get("sched_phi_lr_scale", [])),
            "diag_steps_count": len(history_robust.get("diag_step", [])),
            "diag_inner_obj_current": summarize_series(history_robust.get("diag_inner_obj_current", [])),
            "diag_inner_obj_zero": summarize_series(history_robust.get("diag_inner_obj_zero", [])),
            "diag_inner_obj_gap": summarize_series(history_robust.get("diag_inner_obj_gap", [])),
            "diag_inner_obj_gap_ratio": diag_gap_ratio_summary,
            "diag_v_l2_mean": diag_v_l2_summary,
            "diag_path_delta_mean": summarize_series(history_robust.get("diag_path_delta_mean", [])),
            "diag_terminal_delta_mean": summarize_series(history_robust.get("diag_terminal_delta_mean", [])),
            "collapse_gap_ratio_tol": float(cfg.collapse_gap_ratio_tol),
            "collapse_v_l2_tol": float(cfg.collapse_v_l2_tol),
            "collapse_suspected": collapse_suspected,
        },
        "denoise_debug": denoise_error_curves,
        "denoise_debug_train_pool": denoise_error_curves_train,
        "denoise_debug_heldout_pool": denoise_error_curves_val,
        "generalization_debug": denoise_gap,
        "recovery_debug": {
            "terminal_step_for_plot": int(terminal_step),
            "terminal_sigma_for_plot": float(sigma_levels[terminal_step].item()),
            "plot_stochastic_backward": bool(cfg.plot_stochastic_backward),
            "baseline_x0_mse_from_ref_terminal": float((rev_baseline_from_ref[:, 0] - x_demo).pow(2).sum(dim=1).mean().item()),
            "baseline_x0_mse_from_attack_terminal": float(
                (rev_baseline_from_attack[:, 0] - x_demo).pow(2).sum(dim=1).mean().item()
            ),
            "bayes_posterior_mean_mse_at_terminal_sigma": float(bayes_terminal_mse),
            "baseline_to_bayes_mse_ratio_ref_terminal": float(
                ((rev_baseline_from_ref[:, 0] - x_demo).pow(2).sum(dim=1).mean().item()) / max(bayes_terminal_mse, 1e-12)
            ),
            "baseline_plot_endpoint_mode_metrics_ref_terminal": evaluate_mode_coverage(
                tensor_to_numpy(rev_baseline_from_ref_plot[:, 0]), centers_np
            ),
            "baseline_plot_endpoint_mode_metrics_attack_terminal": evaluate_mode_coverage(
                tensor_to_numpy(rev_baseline_from_attack_plot[:, 0]), centers_np
            ),
            "baseline_path_mse_by_step_from_ref_terminal": compute_path_mse_by_step(rev_baseline_from_ref, ref_paths_plot),
            "baseline_path_mse_by_step_from_attack_terminal": compute_path_mse_by_step(
                rev_baseline_from_attack, ctrl_paths_plot
            ),
            "baseline_x0_mse_vs_terminal_step_forward_baseline": recovery_ref_curve,
            "baseline_x0_mse_vs_terminal_step_forward_attack": recovery_attack_curve,
            "paired_reverse_delta_under_plot_setting": paired_reverse_delta_plot,
            "paired_reverse_delta_deterministic": paired_reverse_delta_det,
            "reverse_terminal_consistency": terminal_consistency,
        },
        "sample_quality_debug": {
            "baseline_generated_mode_metrics": evaluate_mode_coverage(baseline_gen_np, centers_np),
            "robust_generated_mode_metrics": evaluate_mode_coverage(robust_gen_np, centers_np),
            "baseline_generated_to_train_min_dist": (
                evaluate_nearest_reference_distance(baseline_gen_np, train_pool_np) if train_pool_np is not None else None
            ),
            "robust_generated_to_train_min_dist": (
                evaluate_nearest_reference_distance(robust_gen_np, train_pool_np) if train_pool_np is not None else None
            ),
            "heldout_to_train_min_dist": (
                evaluate_nearest_reference_distance(val_pool_np, train_pool_np) if train_pool_np is not None else None
            ),
        },
    }
    if check_report:
        metrics["checks"] = check_report

    plot_forward_timestep_clouds(
        fwd_baseline_paths=tensor_to_numpy(ref_paths_plot),
        bwd_baseline_paths=rev_baseline_from_ref_np,
        fwd_attack_paths=tensor_to_numpy(ctrl_paths_plot),
        bwd_attack_paths=rev_baseline_from_attack_np,
        centers=centers_np,
        out_path=os.path.join(exp_dir, "forward_backward_baseline_attack.png"),
    )
    plot_debug_losses(
        history_baseline=history_baseline,
        history_robust=history_robust,
        denoise_curves=denoise_error_curves,
        recovery_ref_curve=recovery_ref_curve,
        recovery_attack_curve=recovery_attack_curve,
        out_path=os.path.join(exp_dir, "debug_losses_and_recovery.png"),
    )

    payload = {"config": vars(cfg), "metrics": as_jsonable_metrics(metrics)}
    with open(os.path.join(exp_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("[result] metrics summary", flush=True)
    print(f"  dataset_debug: {metrics['dataset_debug']}", flush=True)
    print(f"  objective_debug: {metrics['objective_debug']}", flush=True)
    print(f"  generalization_debug: {metrics['generalization_debug']}", flush=True)
    print(
        "  recovery_debug:"
        f" baseline_x0_mse_from_ref_terminal={metrics['recovery_debug']['baseline_x0_mse_from_ref_terminal']:.6f},"
        f" baseline_x0_mse_from_attack_terminal={metrics['recovery_debug']['baseline_x0_mse_from_attack_terminal']:.6f}",
        flush=True,
    )
    paired_plot = metrics["recovery_debug"]["paired_reverse_delta_under_plot_setting"]
    print(
        "  paired_reverse_delta(plot_setting):"
        f" x0_mean_l2={paired_plot['x0_step_mean_l2']:.6f},"
        f" terminal_mean_l2={paired_plot['terminal_step_mean_l2']:.6f}",
        flush=True,
    )
    term_chk = metrics["recovery_debug"]["reverse_terminal_consistency"]
    print(
        "  reverse_terminal_consistency:"
        f" passed={term_chk['passed']}"
        f" max_l2(plot_ref)={term_chk['plot_ref']['max_l2']:.3e}"
        f" max_l2(plot_attack)={term_chk['plot_attack']['max_l2']:.3e}",
        flush=True,
    )
    print(
        f"  baseline_gate: passed={metrics['baseline_gate']['passed']} "
        f"attack_training_executed={metrics['baseline_gate']['attack_training_executed']}",
        flush=True,
    )
    if not metrics["baseline_gate"]["passed"]:
        for failed in metrics["baseline_gate"]["failed_checks"]:
            print(
                "    - "
                f"{failed['name']}: value={failed['value']:.4f} {failed['op']} {failed['threshold']:.4f} (FAIL)",
                flush=True,
            )
    baseline_gen_mode = metrics["sample_quality_debug"]["baseline_generated_mode_metrics"]
    if baseline_gen_mode["avg_min_dist_to_mode"] > cfg.baseline_gate_max_generated_avg_min_dist:
        print(
            "  [warn] baseline generated samples are still ring-like "
            f"(avg_min_dist_to_mode={baseline_gen_mode['avg_min_dist_to_mode']:.3f} > "
            f"{cfg.baseline_gate_max_generated_avg_min_dist:.3f}).",
            flush=True,
        )
        print(
            "         Increase baseline training steps or use larger batch for sharper mode separation.",
            flush=True,
        )
    if check_report:
        print(f"  checks: {check_report}", flush=True)
    print(f"[result] artifacts saved to: {exp_dir}", flush=True)


if __name__ == "__main__":
    main()
