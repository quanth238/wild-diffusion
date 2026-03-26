import json
import os
from typing import Dict

import torch

from ..checks import run_preflight_checks
from ..data_backends.provider import DatasetBundle, build_dataset_bundle
from ..diagnostics_backends.provider import build_diagnostics_bundle
from ..shared.sigma import build_sigma_levels, sample_target_indices
from ..model_backends.provider import build_model_bundle
from .utils import (
    compute_terminal_match_stats,
    empty_robust_history,
    summarize_series,
    summarize_train_vs_val_curve_gaps,
)
from ..metrics import (
    compute_control_constraint_stats_by_step,
    compute_denoise_error_curves,
    compute_paired_reverse_delta_by_step,
    compute_path_mse_by_step,
    compute_x0_recovery_vs_terminal_step,
    evaluate_nearest_reference_distance,
    summarize_attack_gap_windows,
)
from ..trainer import reverse_paths_from_terminal, train_baseline
from ..utils import as_jsonable_metrics, ensure_dir, pick_device, set_seed, tensor_to_numpy
from ..versions.registry import resolve_method_module


def _print_dataset_info(cfg, dataset: DatasetBundle) -> None:
    """Print dataset regime summary (limited-data vs population sampling)."""

    print(f"[info] dataset_backend={dataset.name} data_shape={dataset.data_shape}", flush=True)
    if cfg.limited_data_enabled:
        if dataset.name == "toy_gmm":
            print(
                "[info] limited_data_enabled=True "
                f"train_pool_size={dataset.train_pool.shape[0]} "
                f"({cfg.train_points_per_mode} points/mode x {cfg.n_modes} modes)",
                flush=True,
            )
        else:
            print(
                "[info] limited_data_enabled=True "
                f"train_pool_size={dataset.train_pool.shape[0]}",
                flush=True,
            )
    else:
        print("[info] limited_data_enabled=False (population sampling mode)", flush=True)
    print(f"[info] val_pool_size={dataset.val_pool.shape[0]}", flush=True)
    if dataset.name == "image_folder":
        disjoint = dataset.metadata.get("train_val_disjoint_guarantee")
        if disjoint is False:
            print(
                "[warn] image_folder backend is using dataset_val_path mode; train/val disjointness is not guaranteed "
                "unless the two roots are independently curated.",
                flush=True,
            )


def _build_baseline_gate(
    cfg,
    diagnostics,
    baseline_eval,
    control,
    sigma_levels: torch.Tensor,
    kappa_by_step: torch.Tensor,
    dataset: DatasetBundle,
    method,
) -> Dict:
    """Compute baseline acceptance gate metrics before robust phase.

    Gate is evaluated from:
    - endpoint quality after forward(ref)->reverse,
    - pure generative quality from Gaussian terminal noise.
    """

    n_steps = int(sigma_levels.numel() - 1)
    gate_terminal_step = cfg.debug_terminal_step
    if gate_terminal_step < 1 or gate_terminal_step > n_steps:
        gate_terminal_step = n_steps
    sigma_levels_gate = sigma_levels[: gate_terminal_step + 1]

    x_gate = dataset.sample_val_batch(cfg.debug_eval_batch)
    idx_gate = sample_target_indices(cfg.debug_eval_batch, sigma_levels)
    gate_roll = method.rollout_controlled_ve(
        x0=x_gate,
        target_indices=idx_gate,
        control_net=control,
        sigma_levels=sigma_levels,
        grad_through_control=False,
        control_radius_kappa=cfg.control_radius_kappa,
        kappa_by_step=kappa_by_step,
    )
    gate_ref_paths = gate_roll.states_ref[:, : gate_terminal_step + 1]
    gate_rev_det = reverse_paths_from_terminal(
        denoiser=baseline_eval,
        x_terminal=gate_ref_paths[:, -1],
        sigma_levels=sigma_levels_gate,
        stochastic=False,
    )
    gate_endpoint_mode_metrics = dataset.evaluate_sample_metrics(tensor_to_numpy(gate_rev_det[:, 0]))
    gate_endpoint_recovery_mse = float((gate_rev_det[:, 0] - x_gate).reshape(x_gate.shape[0], -1).pow(2).mean().item())
    gate_gen_paths = reverse_paths_from_terminal(
        denoiser=baseline_eval,
        x_terminal=dataset.sample_terminal_batch(cfg.eval_samples, sigma_levels[-1]),
        sigma_levels=sigma_levels,
        stochastic=True,
    )
    gate_generated_mode_metrics = dataset.evaluate_sample_metrics(tensor_to_numpy(gate_gen_paths[:, 0]))
    baseline_gate = diagnostics.build_baseline_gate(
        cfg,
        {
            "generated_metrics": gate_generated_mode_metrics,
            "endpoint_metrics": gate_endpoint_mode_metrics,
            "endpoint_recovery_mse_mean": gate_endpoint_recovery_mse,
            "sigma_terminal": float(sigma_levels_gate[-1].item()),
        },
    )
    return baseline_gate


def _run_robust_phase(
    cfg,
    robust,
    control,
    centers,
    sigma_levels,
    dataset: DatasetBundle,
    baseline_gate: Dict,
    method,
):
    """Execute or skip robust training depending on baseline-only mode and gate status."""

    attack_training_executed = False
    if cfg.baseline_only:
        with torch.no_grad():
            for p in control.parameters():
                p.zero_()
        return False, empty_robust_history(), robust

    if cfg.baseline_gate_enabled and not baseline_gate["passed"]:
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
        return False, empty_robust_history(), robust

    attack_training_executed = True
    history_robust = method.train_trajectory_robust(
        robust,
        control,
        centers,
        sigma_levels,
        cfg,
        train_pool=dataset.train_pool,
        sample_train_batch_fn=dataset.sample_train_batch,
        sample_population_batch_fn=dataset.sample_population_batch,
    )
    return attack_training_executed, history_robust, robust


def run_experiment(cfg) -> dict:
    """End-to-end toy experiment pipeline.

    Steps:
    1) build data + sigma ladder,
    2) train baseline and evaluate gate,
    3) run robust phase if allowed,
    4) compute diagnostics/plots/metrics and persist outputs.
    """

    device = pick_device(cfg.device)
    set_seed(cfg.seed)

    ensure_dir(cfg.outdir)
    exp_dir = os.path.join(cfg.outdir, cfg.exp_name)
    ensure_dir(exp_dir)
    flow_mode = (
        "baseline_only"
        if cfg.baseline_only
        else ("robust_forced_no_gate" if not cfg.baseline_gate_enabled else "robust_with_gate")
    )
    method = resolve_method_module(cfg.method_version)
    if not getattr(method, "IMPLEMENTED", True):
        raise NotImplementedError(
            f"method_version='{cfg.method_version}' is marked IMPLEMENTED=False. "
            "Please implement its rollout/train API under toy/versions/<version>/."
        )
    print(f"[info] device={device}", flush=True)
    print(f"[info] exp_dir={exp_dir}", flush=True)
    print(
        "[info] flow_mode="
        f"{flow_mode} baseline_gate_enabled={cfg.baseline_gate_enabled} "
        f"outer_attack_weight={cfg.outer_attack_weight} outer_clean_weight={cfg.outer_clean_weight} "
        f"method_version={cfg.method_version}",
        flush=True,
    )

    dataset = build_dataset_bundle(cfg, device)
    diagnostics = build_diagnostics_bundle(cfg, dataset)
    centers = dataset.centers
    if cfg.sigma_data <= 0:
        cfg.sigma_data = dataset.estimate_sigma_data()
    print(f"[info] sigma_data={cfg.sigma_data:.6f}", flush=True)
    _print_dataset_info(cfg, dataset)

    sigma_levels = build_sigma_levels(cfg.sigma_min, cfg.sigma_max, cfg.n_steps_path, device=device)
    kappa_by_step = method.build_kappa_schedule(
        sigma_levels=sigma_levels,
        base_kappa=cfg.control_radius_kappa,
        use_time_dependent=cfg.use_time_dependent_kappa,
        low_multiplier=cfg.kappa_low_multiplier,
        mid_multiplier=cfg.kappa_mid_multiplier,
        high_multiplier=cfg.kappa_high_multiplier,
        preserve_l2_budget=cfg.kappa_preserve_l2_budget,
    ).to(device=sigma_levels.device, dtype=sigma_levels.dtype)

    model_bundle = build_model_bundle(cfg, dataset, sigma_data=cfg.sigma_data, device=device)
    baseline = model_bundle.baseline
    robust = model_bundle.robust
    control = model_bundle.control

    check_report = {}
    if cfg.run_checks:
        check_report = run_preflight_checks(
            robust,
            control,
            centers,
            sigma_levels,
            cfg,
            sample_batch_fn=dataset.sample_population_batch,
            method=method,
        )
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
        train_pool=dataset.train_pool,
        sample_train_batch_fn=dataset.sample_train_batch,
        sample_population_batch_fn=dataset.sample_population_batch,
    )
    robust.load_state_dict(baseline_eval.state_dict())
    baseline_gate = _build_baseline_gate(
        cfg=cfg,
        diagnostics=diagnostics,
        baseline_eval=baseline_eval,
        control=control,
        sigma_levels=sigma_levels,
        kappa_by_step=kappa_by_step,
        dataset=dataset,
        method=method,
    )

    attack_training_executed, history_robust, robust = _run_robust_phase(
        cfg=cfg,
        robust=robust,
        control=control,
        centers=centers,
        sigma_levels=sigma_levels,
        dataset=dataset,
        baseline_gate=baseline_gate,
        method=method,
    )
    if not attack_training_executed:
        robust = baseline_eval

    # Evaluate denoise curves on train-pool and held-out pools to expose overfitting under limited data.
    x_train_eval = dataset.sample_train_batch(cfg.debug_eval_batch)
    idx_train_eval = sample_target_indices(cfg.debug_eval_batch, sigma_levels)
    train_roll = method.rollout_controlled_ve(
        x0=x_train_eval,
        target_indices=idx_train_eval,
        control_net=control,
        sigma_levels=sigma_levels,
        grad_through_control=False,
        control_radius_kappa=cfg.control_radius_kappa,
        kappa_by_step=kappa_by_step,
    )
    denoise_error_curves_train = compute_denoise_error_curves(
        baseline_model=baseline_eval,
        robust_model=robust,
        x0=x_train_eval,
        states_ref=train_roll.states_ref,
        states_ctrl=train_roll.states_ctrl,
        sigma_levels=sigma_levels,
    )

    x_val_eval = dataset.sample_val_batch(cfg.debug_eval_batch)
    idx_val_eval = sample_target_indices(cfg.debug_eval_batch, sigma_levels)
    val_roll = method.rollout_controlled_ve(
        x0=x_val_eval,
        target_indices=idx_val_eval,
        control_net=control,
        sigma_levels=sigma_levels,
        grad_through_control=False,
        control_radius_kappa=cfg.control_radius_kappa,
        kappa_by_step=kappa_by_step,
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
            (terminal_step + 1, ref_paths_plot.shape[0], *ref_paths_plot.shape[2:]),
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
    bayes_terminal_mse = diagnostics.estimate_bayes_terminal_mse(
        dataset,
        cfg,
        sigma=float(sigma_levels[terminal_step].item()),
    )
    baseline_gen_paths = reverse_paths_from_terminal(
        denoiser=baseline_eval,
        x_terminal=dataset.sample_terminal_batch(cfg.eval_samples, sigma_levels[-1]),
        sigma_levels=sigma_levels,
        stochastic=True,
    )
    baseline_gen_np = tensor_to_numpy(baseline_gen_paths[:, 0])
    robust_gen_paths = reverse_paths_from_terminal(
        denoiser=robust,
        x_terminal=dataset.sample_terminal_batch(cfg.eval_samples, sigma_levels[-1]),
        sigma_levels=sigma_levels,
        stochastic=True,
    )
    robust_gen_np = tensor_to_numpy(robust_gen_paths[:, 0])
    val_pool_np = tensor_to_numpy(dataset.val_pool)
    train_pool_np = tensor_to_numpy(dataset.train_pool) if dataset.train_pool is not None else None
    enable_ref_dist = bool(dataset.metadata.get("enable_nearest_reference_distance", True))
    denoise_gap = summarize_train_vs_val_curve_gaps(denoise_error_curves_train, denoise_error_curves_val)
    attack_gap_windows_train = summarize_attack_gap_windows(denoise_error_curves_train)
    attack_gap_windows_val = summarize_attack_gap_windows(denoise_error_curves_val)
    constraint_stats_train = compute_control_constraint_stats_by_step(
        delta_path=train_roll.delta_path,
        sigma_levels=sigma_levels,
        control_radius_kappa=cfg.control_radius_kappa,
        kappa_by_step=kappa_by_step,
        states_ref=train_roll.states_ref,
        states_ctrl=train_roll.states_ctrl,
        saturation_threshold=cfg.constraint_saturation_threshold,
    )
    constraint_stats_val = compute_control_constraint_stats_by_step(
        delta_path=val_roll.delta_path,
        sigma_levels=sigma_levels,
        control_radius_kappa=cfg.control_radius_kappa,
        kappa_by_step=kappa_by_step,
        states_ref=val_roll.states_ref,
        states_ctrl=val_roll.states_ctrl,
        saturation_threshold=cfg.constraint_saturation_threshold,
    )
    diag_gap_ratio_summary = summarize_series(history_robust.get("diag_inner_obj_gap_ratio", []))
    diag_delta_ratio_summary = summarize_series(history_robust.get("diag_delta_norm_ratio_mean", []))
    collapse_suspected = None
    if diag_gap_ratio_summary["mean_last"] is not None and diag_delta_ratio_summary["mean_last"] is not None:
        collapse_suspected = bool(
            abs(diag_gap_ratio_summary["mean_last"]) <= float(cfg.collapse_gap_ratio_tol)
            and diag_delta_ratio_summary["mean_last"] <= float(cfg.collapse_delta_ratio_tol)
        )

    metrics = {
        "flow_debug": {
            "flow_mode": flow_mode,
            "baseline_only": bool(cfg.baseline_only),
            "baseline_gate_enabled": bool(cfg.baseline_gate_enabled),
            "baseline_gate_error_on_fail": bool(cfg.baseline_gate_error_on_fail),
            "outer_attack_weight": float(cfg.outer_attack_weight),
            "outer_clean_weight": float(cfg.outer_clean_weight),
            "inner_steps": int(cfg.inner_steps),
            "v1_dual_lambda_enabled": bool(cfg.v1_dual_lambda_enabled),
            "v1_energy_budget_rho": float(cfg.v1_energy_budget_rho),
            "v1_lambda_init": float(cfg.v1_lambda_init),
            "v1_lambda_lr": float(cfg.v1_lambda_lr),
            "v1_lambda_max": float(cfg.v1_lambda_max),
            "lambda_energy_fixed": float(cfg.lambda_energy),
            "method_version": cfg.method_version,
            "method_description": getattr(method, "DESCRIPTION", ""),
            "model_backend": model_bundle.name,
            "diagnostics_backend": diagnostics.name,
        },
        "dataset_debug": {
            "dataset_backend": dataset.name,
            "data_shape": list(dataset.data_shape),
            "limited_data_enabled": bool(cfg.limited_data_enabled),
            "train_points_per_mode": (
                int(cfg.train_points_per_mode) if dataset.name == "toy_gmm" else None
            ),
            "train_pool_size": int(dataset.train_pool.shape[0]) if dataset.train_pool is not None else None,
            "val_pool_size": int(dataset.val_pool.shape[0]),
            "train_mode_counts": (
                [
                    int(v)
                    for v in torch.bincount(
                        dataset.train_pool_labels,
                        minlength=int(dataset.metadata.get("num_classes", int(dataset.train_pool_labels.max().item()) + 1)),
                    )
                    .detach()
                    .cpu()
                    .tolist()
                ]
                if dataset.train_pool_labels is not None
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
            "robust_energy": summarize_series(history_robust.get("energy", [])),
            "robust_lambda_value": summarize_series(history_robust.get("lambda_value", [])),
            "robust_lambda_value_next": summarize_series(history_robust.get("lambda_value_next", [])),
            "robust_lambda_subgrad": summarize_series(history_robust.get("lambda_subgrad", [])),
            "robust_dual_surrogate": summarize_series(history_robust.get("dual_surrogate", [])),
            "robust_delta_norm_mean": summarize_series(history_robust.get("delta_norm_mean", [])),
            "robust_delta_norm_max": summarize_series(history_robust.get("delta_norm_max", [])),
            "robust_delta_norm_ratio_mean": summarize_series(history_robust.get("delta_norm_ratio_mean", [])),
            "robust_delta_norm_ratio_max": summarize_series(history_robust.get("delta_norm_ratio_max", [])),
            "robust_sched_attack_weight": summarize_series(history_robust.get("sched_attack_weight", [])),
            "robust_sched_clean_weight": summarize_series(history_robust.get("sched_clean_weight", [])),
            "robust_sched_phi_lr_scale": summarize_series(history_robust.get("sched_phi_lr_scale", [])),
            "diag_steps_count": len(history_robust.get("diag_step", [])),
            "diag_inner_obj_current": summarize_series(history_robust.get("diag_inner_obj_current", [])),
            "diag_inner_obj_zero": summarize_series(history_robust.get("diag_inner_obj_zero", [])),
            "diag_inner_obj_gap": summarize_series(history_robust.get("diag_inner_obj_gap", [])),
            "diag_inner_obj_gap_ratio": diag_gap_ratio_summary,
            "diag_delta_norm_mean": summarize_series(history_robust.get("diag_delta_norm_mean", [])),
            "diag_delta_norm_max": summarize_series(history_robust.get("diag_delta_norm_max", [])),
            "diag_delta_norm_ratio_mean": diag_delta_ratio_summary,
            "diag_delta_norm_ratio_max": summarize_series(history_robust.get("diag_delta_norm_ratio_max", [])),
            "diag_path_delta_mean": summarize_series(history_robust.get("diag_path_delta_mean", [])),
            "diag_terminal_delta_mean": summarize_series(history_robust.get("diag_terminal_delta_mean", [])),
            "collapse_gap_ratio_tol": float(cfg.collapse_gap_ratio_tol),
            "collapse_delta_ratio_tol": float(cfg.collapse_delta_ratio_tol),
            "collapse_suspected": collapse_suspected,
        },
        "denoise_debug": denoise_error_curves,
        "denoise_debug_train_pool": denoise_error_curves_train,
        "denoise_debug_heldout_pool": denoise_error_curves_val,
        "generalization_debug": denoise_gap,
        "constraint_debug": {
            "use_time_dependent_kappa": bool(cfg.use_time_dependent_kappa),
            "kappa_base": float(cfg.control_radius_kappa),
            "kappa_low_multiplier": float(cfg.kappa_low_multiplier),
            "kappa_mid_multiplier": float(cfg.kappa_mid_multiplier),
            "kappa_high_multiplier": float(cfg.kappa_high_multiplier),
            "kappa_preserve_l2_budget": bool(cfg.kappa_preserve_l2_budget),
            "kappa_by_step": [float(v.item()) for v in kappa_by_step.detach().cpu()],
            "train_rollout_by_step": constraint_stats_train,
            "heldout_rollout_by_step": constraint_stats_val,
            "attack_gap_windows_train": attack_gap_windows_train,
            "attack_gap_windows_heldout": attack_gap_windows_val,
        },
        "recovery_debug": {
            "terminal_step_for_plot": int(terminal_step),
            "terminal_sigma_for_plot": float(sigma_levels[terminal_step].item()),
            "plot_stochastic_backward": bool(cfg.plot_stochastic_backward),
            "baseline_x0_mse_from_ref_terminal": float(
                (rev_baseline_from_ref[:, 0] - x_demo).reshape(x_demo.shape[0], -1).pow(2).sum(dim=1).mean().item()
            ),
            "baseline_x0_mse_from_attack_terminal": float(
                (rev_baseline_from_attack[:, 0] - x_demo).reshape(x_demo.shape[0], -1).pow(2).sum(dim=1).mean().item()
            ),
            "bayes_posterior_mean_mse_at_terminal_sigma": (
                None if bayes_terminal_mse is None else float(bayes_terminal_mse)
            ),
            "baseline_to_bayes_mse_ratio_ref_terminal": (
                None
                if bayes_terminal_mse is None
                else float(
                    ((rev_baseline_from_ref[:, 0] - x_demo).reshape(x_demo.shape[0], -1).pow(2).sum(dim=1).mean().item())
                    / max(bayes_terminal_mse, 1e-12)
                )
            ),
            "baseline_plot_endpoint_mode_metrics_ref_terminal": dataset.evaluate_sample_metrics(
                tensor_to_numpy(rev_baseline_from_ref_plot[:, 0])
            ),
            "baseline_plot_endpoint_mode_metrics_attack_terminal": dataset.evaluate_sample_metrics(
                tensor_to_numpy(rev_baseline_from_attack_plot[:, 0])
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
            "baseline_generated_metrics": dataset.evaluate_sample_metrics(baseline_gen_np),
            "robust_generated_metrics": dataset.evaluate_sample_metrics(robust_gen_np),
            "baseline_generated_to_train_min_dist": (
                evaluate_nearest_reference_distance(baseline_gen_np, train_pool_np)
                if train_pool_np is not None and enable_ref_dist
                else None
            ),
            "robust_generated_to_train_min_dist": (
                evaluate_nearest_reference_distance(robust_gen_np, train_pool_np)
                if train_pool_np is not None and enable_ref_dist
                else None
            ),
            "heldout_to_train_min_dist": (
                evaluate_nearest_reference_distance(val_pool_np, train_pool_np)
                if train_pool_np is not None and enable_ref_dist
                else None
            ),
        },
    }
    if check_report:
        metrics["checks"] = check_report
    metrics["sample_quality_debug"]["baseline_generated_mode_metrics"] = metrics["sample_quality_debug"][
        "baseline_generated_metrics"
    ]
    metrics["sample_quality_debug"]["robust_generated_mode_metrics"] = metrics["sample_quality_debug"][
        "robust_generated_metrics"
    ]

    diagnostics.plot_forward_backward_debug(
        fwd_baseline_paths=tensor_to_numpy(ref_paths_plot),
        bwd_baseline_paths=rev_baseline_from_ref_np,
        fwd_attack_paths=tensor_to_numpy(ctrl_paths_plot),
        bwd_attack_paths=rev_baseline_from_attack_np,
        centers=dataset.metadata.get("centers_np"),
        out_path=os.path.join(exp_dir, "forward_backward_baseline_attack.png"),
    )
    diagnostics.plot_loss_debug(
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
    heldout_constraint = metrics["constraint_debug"]["heldout_rollout_by_step"]
    heldout_windows = metrics["constraint_debug"]["attack_gap_windows_heldout"]
    frac_sat = heldout_constraint["frac_near_boundary"]
    tail_start = max(len(frac_sat) * 2 // 3, 0)
    frac_sat_tail = float(sum(frac_sat[tail_start:]) / max(len(frac_sat[tail_start:]), 1))
    print(
        "  constraint_debug:"
        f" heldout_overall_win_ratio={heldout_windows['overall']['win_ratio']:.3f},"
        f" high_noise_win_ratio={heldout_windows['high_noise']['win_ratio']:.3f},"
        f" high_noise_mean_gap={heldout_windows['high_noise']['mean_attack_gap']:.6f},"
        f" frac_near_boundary_tail={frac_sat_tail:.3f}",
        flush=True,
    )
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
    for warning in diagnostics.summarize_warnings(cfg, metrics):
        print(f"  [warn] {warning}", flush=True)
    if check_report:
        print(f"  checks: {check_report}", flush=True)
    print(f"[result] artifacts saved to: {exp_dir}", flush=True)
    return payload
