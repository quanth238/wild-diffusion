import copy
from typing import Callable, Optional

import torch

from ...compute_accounting import append_denoiser_op_count_step, ensure_denoiser_op_count_history
from ...models import set_requires_grad
from ...shared.ema import init_ema_model, update_ema_model
from ...shared.objective import compute_training_loss, inner_objective_attack_only
from ...shared.runtime import autocast_context, resolve_amp_dtype
from ...shared.sigma import sample_target_indices, sample_target_indices_log_normal
from ...shared.train_utils import pathwise_l2, sample_train_batch
from ...utils import has_nan_or_inf, scalarize
from .diffusion import (
    build_constraint_radii_for_objective,
    build_transition_deltas_for_objective,
    rollout_controlled_ve,
    rollout_path_heuristic_attack,
)


def _control_transport_cost(control_path: torch.Tensor, transition_deltas: torch.Tensor) -> torch.Tensor:
    """Route-A discrete control cost: E[sum_k Delta_tau_k ||u_k||^2]."""

    flat = control_path.reshape(control_path.shape[0], control_path.shape[1], -1)
    control_sq = flat.pow(2).sum(dim=2)
    dt = transition_deltas.view(1, -1).to(device=control_sq.device, dtype=control_sq.dtype)
    return (control_sq * dt).sum(dim=1).mean()


def _path_average_training_loss(
    cfg,
    denoiser,
    states: torch.Tensor,
    x0: torch.Tensor,
    sigma_levels: torch.Tensor,
) -> torch.Tensor:
    """Average weighted denoise loss over all rollout timesteps k=1..N."""

    n_steps = int(sigma_levels.numel() - 1)
    if states.shape[1] != n_steps + 1:
        raise ValueError(f"states step dim must be {n_steps + 1}, got {states.shape[1]}")

    total_loss = None
    for step_idx in range(n_steps):
        sigma = torch.full(
            (x0.shape[0],),
            float(sigma_levels[step_idx + 1].item()),
            device=x0.device,
            dtype=x0.dtype,
        )
        loss_step = compute_training_loss(cfg, denoiser, states[:, step_idx + 1], x0, sigma)
        total_loss = loss_step if total_loss is None else (total_loss + loss_step)
    return total_loss / float(n_steps)


def _path_outer_loss_backward(
    *,
    cfg,
    denoiser,
    states_ctrl: torch.Tensor,
    states_ref: torch.Tensor,
    x0: torch.Tensor,
    sigma_levels: torch.Tensor,
    lambda_ctrl: float,
    lambda_ref: float,
    amp_dtype,
) -> tuple[float, float, float]:
    """Accumulate exact path-mean outer loss with per-timestep backward passes."""

    n_steps = int(sigma_levels.numel() - 1)
    if states_ctrl.shape[1] != n_steps + 1:
        raise ValueError(f"states_ctrl step dim must be {n_steps + 1}, got {states_ctrl.shape[1]}")
    if states_ref.shape[1] != n_steps + 1:
        raise ValueError(f"states_ref step dim must be {n_steps + 1}, got {states_ref.shape[1]}")

    attack_loss_total = 0.0
    clean_loss_total = 0.0
    outer_loss_total = 0.0

    for step_idx in range(n_steps):
        sigma = torch.full(
            (x0.shape[0],),
            float(sigma_levels[step_idx + 1].item()),
            device=x0.device,
            dtype=x0.dtype,
        )
        with autocast_context(sigma_levels.device, amp_dtype):
            if lambda_ctrl > 0.0:
                attack_loss_step = compute_training_loss(
                    cfg,
                    denoiser,
                    states_ctrl[:, step_idx + 1],
                    x0,
                    sigma,
                )
            else:
                with torch.no_grad():
                    attack_loss_step = compute_training_loss(
                        cfg,
                        denoiser,
                        states_ctrl[:, step_idx + 1],
                        x0,
                        sigma,
                    )

            if lambda_ref > 0.0:
                clean_loss_step = compute_training_loss(
                    cfg,
                    denoiser,
                    states_ref[:, step_idx + 1],
                    x0,
                    sigma,
                )
            else:
                clean_loss_step = torch.zeros((), device=x0.device, dtype=x0.dtype)

            chunk_outer = lambda_ctrl * attack_loss_step + lambda_ref * clean_loss_step
            chunk_outer = chunk_outer / float(n_steps)

        if has_nan_or_inf(chunk_outer):
            raise RuntimeError("NaN/Inf detected in cdro outer loss.")
        if chunk_outer.requires_grad:
            chunk_outer.backward()

        attack_loss_total += scalarize(attack_loss_step) / float(n_steps)
        clean_loss_total += scalarize(clean_loss_step) / float(n_steps)
        outer_loss_total += scalarize(chunk_outer)

    return attack_loss_total, clean_loss_total, outer_loss_total


def _path_batch_equiv_denoiser_evals(sigma_levels: torch.Tensor) -> float:
    """Count one denoiser eval over `B*T` path states as `T` batch-equivalent evals."""

    return float(max(int(sigma_levels.numel()) - 1, 0))


def _rollout_delta_diagnostics(roll, radius_by_step: torch.Tensor):
    """Summarize path/control magnitudes against the exact Route-A delta-space caps."""

    diff = roll.states_ctrl - roll.states_ref
    diff_l2 = pathwise_l2(diff)
    path_delta_mean = diff_l2.mean()
    terminal_delta_mean = diff_l2[:, -1].mean()

    delta_l2 = pathwise_l2(roll.delta_path)
    delta_norm_mean = delta_l2.mean()
    delta_norm_max = delta_l2.max()
    radius = radius_by_step.view(1, -1).to(device=delta_l2.device, dtype=delta_l2.dtype)
    delta_ratio = delta_l2 / radius.clamp_min(1e-8)
    delta_ratio_mean = delta_ratio.mean()
    delta_ratio_max = delta_ratio.max()
    return path_delta_mean, terminal_delta_mean, delta_norm_mean, delta_norm_max, delta_ratio_mean, delta_ratio_max


def _build_rollout_noise_schedules(
    *,
    x0: torch.Tensor,
    sigma_levels: torch.Tensor,
    antithetic_rollouts: bool,
) -> list[Optional[torch.Tensor]]:
    """Return one shared-noise schedule, optionally paired with its antithetic copy."""

    if not bool(antithetic_rollouts):
        return [None]
    n_steps = int(sigma_levels.numel() - 1)
    if n_steps <= 0:
        return [None]
    base_schedule = torch.randn((n_steps,) + tuple(x0.shape), device=x0.device, dtype=x0.dtype)
    return [base_schedule, -base_schedule]


def _path_clean_only_loss_backward(
    *,
    cfg,
    denoiser,
    states_ref: torch.Tensor,
    x0: torch.Tensor,
    sigma_levels: torch.Tensor,
    lambda_ref: float,
    amp_dtype,
) -> tuple[float, float, float]:
    """Accumulate clean-only path loss without the redundant attacked-path pass."""

    n_steps = int(sigma_levels.numel() - 1)
    if states_ref.shape[1] != n_steps + 1:
        raise ValueError(f"states_ref step dim must be {n_steps + 1}, got {states_ref.shape[1]}")

    clean_loss_total = 0.0
    outer_loss_total = 0.0

    for step_idx in range(n_steps):
        sigma = torch.full(
            (x0.shape[0],),
            float(sigma_levels[step_idx + 1].item()),
            device=x0.device,
            dtype=x0.dtype,
        )
        with autocast_context(sigma_levels.device, amp_dtype):
            clean_loss_step = compute_training_loss(
                cfg,
                denoiser,
                states_ref[:, step_idx + 1],
                x0,
                sigma,
            )
            chunk_outer = lambda_ref * clean_loss_step / float(n_steps)

        if has_nan_or_inf(chunk_outer):
            raise RuntimeError("NaN/Inf detected in cdro clean-only outer loss.")
        if chunk_outer.requires_grad:
            chunk_outer.backward()

        clean_loss_total += scalarize(clean_loss_step) / float(n_steps)
        outer_loss_total += scalarize(chunk_outer)

    return 0.0, clean_loss_total, outer_loss_total


def train_trajectory_robust_cdro(
    denoiser,
    control,
    centers,
    sigma_levels,
    cfg,
    train_pool: torch.Tensor = None,
    sample_train_batch_fn: Optional[Callable[[int], torch.Tensor]] = None,
    sample_population_batch_fn: Optional[Callable[[int], torch.Tensor]] = None,
    start_step: int = 0,
    history_state: Optional[dict] = None,
    optimizer_theta_state: Optional[dict] = None,
    ema_state_dict: Optional[dict] = None,
    return_state: bool = False,
):
    """Route-A CDRO training with u-space local caps and greedy denoiser-dependent attacks."""

    optimizer_theta = torch.optim.Adam(denoiser.parameters(), lr=cfg.lr_theta)
    if optimizer_theta_state is not None:
        optimizer_theta.load_state_dict(optimizer_theta_state)
        for state in optimizer_theta.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device=sigma_levels.device)

    history = history_state if history_state is not None else {
        "outer_loss": [],
        "outer_loss_attack": [],
        "outer_loss_clean": [],
        "inner_obj": [],
        "energy": [],
        "delta_norm_mean": [],
        "delta_norm_max": [],
        "delta_norm_ratio_mean": [],
        "delta_norm_ratio_max": [],
        "sched_attack_weight": [],
        "sched_clean_weight": [],
        "sched_phi_lr_scale": [],
        "diag_step": [],
        "diag_inner_obj_current": [],
        "diag_inner_obj_zero": [],
        "diag_inner_obj_gap": [],
        "diag_inner_obj_gap_ratio": [],
        "diag_delta_norm_mean": [],
        "diag_delta_norm_max": [],
        "diag_delta_norm_ratio_mean": [],
        "diag_delta_norm_ratio_max": [],
        "diag_path_delta_mean": [],
        "diag_terminal_delta_mean": [],
        "batch_equiv_denoiser_evals_step": [],
        "batch_equiv_denoiser_evals_attack_construction": [],
        "batch_equiv_denoiser_evals_attack_eval": [],
        "batch_equiv_denoiser_evals_clean_eval": [],
        "batch_equiv_denoiser_evals_cumulative": [],
    }
    for key in (
        "outer_loss",
        "outer_loss_attack",
        "outer_loss_clean",
        "inner_obj",
        "energy",
        "delta_norm_mean",
        "delta_norm_max",
        "delta_norm_ratio_mean",
        "delta_norm_ratio_max",
        "sched_attack_weight",
        "sched_clean_weight",
        "sched_phi_lr_scale",
        "diag_step",
        "diag_inner_obj_current",
        "diag_inner_obj_zero",
        "diag_inner_obj_gap",
        "diag_inner_obj_gap_ratio",
        "diag_delta_norm_mean",
        "diag_delta_norm_max",
        "diag_delta_norm_ratio_mean",
        "diag_delta_norm_ratio_max",
        "diag_path_delta_mean",
        "diag_terminal_delta_mean",
        "batch_equiv_denoiser_evals_step",
        "batch_equiv_denoiser_evals_attack_construction",
        "batch_equiv_denoiser_evals_attack_eval",
        "batch_equiv_denoiser_evals_clean_eval",
        "batch_equiv_denoiser_evals_cumulative",
    ):
        history.setdefault(key, [])
    ensure_denoiser_op_count_history(history)

    # Route-A CDRO does not optimize a persistent control policy network.
    set_requires_grad(control, False)
    with torch.no_grad():
        for p in control.parameters():
            p.zero_()

    step_size = float(cfg.cdro_step_size)
    total_budget = float(cfg.cdro_total_budget_rho)
    time_horizon = float(cfg.cdro_time_horizon)
    transition_deltas = build_transition_deltas_for_objective(
        cfg=cfg,
        sigma_levels=sigma_levels,
        time_horizon=time_horizon,
    ).to(device=sigma_levels.device, dtype=sigma_levels.dtype)
    path_batch_equiv_evals = _path_batch_equiv_denoiser_evals(sigma_levels)
    cumulative_batch_equiv_evals = (
        float(history["batch_equiv_denoiser_evals_cumulative"][-1])
        if history["batch_equiv_denoiser_evals_cumulative"]
        else 0.0
    )
    amp_dtype = resolve_amp_dtype(sigma_levels.device, getattr(cfg, "amp_dtype", "auto"))
    ema_model = init_ema_model(denoiser, cfg, ema_state_dict=ema_state_dict)
    radius_by_step = build_constraint_radii_for_objective(
        cfg=cfg,
        sigma_levels=sigma_levels,
        total_budget=total_budget,
        time_horizon=time_horizon,
    ).to(device=sigma_levels.device, dtype=sigma_levels.dtype)

    for step in range(int(start_step) + 1, int(cfg.steps) + 1):
        x0 = sample_train_batch(
            cfg,
            centers,
            train_pool=train_pool,
            sample_train_batch_fn=sample_train_batch_fn,
            sample_population_batch_fn=sample_population_batch_fn,
        )
        if cfg.use_log_normal_sigma_sampling:
            indices = sample_target_indices_log_normal(
                cfg.batch_size,
                sigma_levels,
                p_mean=cfg.p_mean,
                p_std=cfg.p_std,
            )
        else:
            indices = sample_target_indices(cfg.batch_size, sigma_levels)

        raw_lambda_ref = float(cfg.outer_clean_weight)
        raw_lambda_ctrl = float(cfg.outer_attack_weight)
        mixture_mass = raw_lambda_ref + raw_lambda_ctrl
        if mixture_mass <= 0.0:
            raise RuntimeError("CDRO continuation mixture must have positive total mass.")
        lambda_ref = raw_lambda_ref / mixture_mass
        lambda_ctrl = raw_lambda_ctrl / mixture_mass
        phi_lr_scale = 1.0 if lambda_ctrl > 0.0 else 0.0
        control_updates_enabled = lambda_ctrl > 0.0
        attack_enabled = bool(control_updates_enabled and cfg.inner_steps > 0)
        rollout_schedules = _build_rollout_noise_schedules(
            x0=x0,
            sigma_levels=sigma_levels,
            antithetic_rollouts=bool(getattr(cfg, "cdro_antithetic_rollouts", False)),
        )
        rollout_multiplier = float(len(rollout_schedules))
        attack_construction_units = 0.0
        rollouts = []

        set_requires_grad(denoiser, False)
        for eps_schedule in rollout_schedules:
            if attack_enabled:
                roll = rollout_path_heuristic_attack(
                    cfg=cfg,
                    x0=x0,
                    target_indices=indices,
                    attack_net=denoiser,
                    sigma_levels=sigma_levels,
                    inner_steps=int(cfg.inner_steps),
                    step_size=step_size,
                    total_budget=total_budget,
                    time_horizon=time_horizon,
                    eps_schedule=eps_schedule,
                )
            else:
                roll = rollout_controlled_ve(
                    x0=x0,
                    target_indices=indices,
                    control_net=None,
                    sigma_levels=sigma_levels,
                    grad_through_control=False,
                    control_radius_kappa=cfg.control_radius_kappa,
                    kappa_by_step=None,
                    total_budget=total_budget,
                    time_horizon=time_horizon,
                    eps_schedule=eps_schedule,
                    cfg=cfg,
                )
            rollouts.append(roll)
        if attack_enabled:
            attack_construction_units = (
                path_batch_equiv_evals * float(max(int(cfg.inner_steps), 0)) * rollout_multiplier
            )

        if attack_enabled:
            attack_loss_inner = None
            transport_inner = None
            for roll in rollouts:
                with autocast_context(sigma_levels.device, amp_dtype):
                    attack_loss_chunk = _path_average_training_loss(cfg, denoiser, roll.states_ctrl, x0, sigma_levels)
                transport_chunk = _control_transport_cost(roll.control_path, transition_deltas)
                attack_loss_inner = (
                    attack_loss_chunk if attack_loss_inner is None else (attack_loss_inner + attack_loss_chunk)
                )
                transport_inner = transport_chunk if transport_inner is None else (transport_inner + transport_chunk)
            attack_loss_inner = attack_loss_inner / rollout_multiplier
            transport_inner = transport_inner / rollout_multiplier
            inner_obj = inner_objective_attack_only(attack_loss_inner)
            if has_nan_or_inf(inner_obj):
                raise RuntimeError("NaN/Inf detected in cdro inner objective.")
        else:
            attack_loss_inner = torch.zeros((), device=x0.device, dtype=x0.dtype)
            transport_inner = None
            for roll in rollouts:
                transport_chunk = _control_transport_cost(roll.control_path, transition_deltas)
                transport_inner = transport_chunk if transport_inner is None else (transport_inner + transport_chunk)
            transport_inner = transport_inner / rollout_multiplier
            inner_obj = torch.zeros((), device=x0.device, dtype=x0.dtype)

        last_inner_obj = scalarize(inner_obj) if attack_enabled else 0.0
        last_transport = scalarize(transport_inner)
        delta_norm_mean_values = []
        delta_norm_max_values = []
        delta_ratio_mean_values = []
        delta_ratio_max_values = []
        radius = radius_by_step.view(1, -1)
        for roll in rollouts:
            delta_l2 = pathwise_l2(roll.delta_path)
            delta_ratio = delta_l2 / radius.to(device=delta_l2.device, dtype=delta_l2.dtype).clamp_min(1e-8)
            delta_norm_mean_values.append(scalarize(delta_l2.mean()))
            delta_norm_max_values.append(scalarize(delta_l2.max()))
            delta_ratio_mean_values.append(scalarize(delta_ratio.mean()))
            delta_ratio_max_values.append(scalarize(delta_ratio.max()))
        last_delta_norm_mean = sum(delta_norm_mean_values) / len(delta_norm_mean_values)
        last_delta_norm_max = sum(delta_norm_max_values) / len(delta_norm_max_values)
        last_delta_ratio_mean = sum(delta_ratio_mean_values) / len(delta_ratio_mean_values)
        last_delta_ratio_max = sum(delta_ratio_max_values) / len(delta_ratio_max_values)

        set_requires_grad(denoiser, True)
        optimizer_theta.zero_grad(set_to_none=True)
        outer_loss_attack_vals = []
        outer_loss_clean_vals = []
        outer_loss_vals = []
        if lambda_ctrl <= 0.0:
            lambda_ref_scaled = float(lambda_ref) / rollout_multiplier
            for roll in rollouts:
                outer_loss_attack_val, outer_loss_clean_val, outer_loss_val = _path_clean_only_loss_backward(
                    cfg=cfg,
                    denoiser=denoiser,
                    states_ref=roll.states_ref,
                    x0=x0,
                    sigma_levels=sigma_levels,
                    lambda_ref=lambda_ref_scaled,
                    amp_dtype=amp_dtype,
                )
                outer_loss_attack_vals.append(float(outer_loss_attack_val))
                outer_loss_clean_vals.append(float(outer_loss_clean_val))
                outer_loss_vals.append(float(outer_loss_val))
            attack_eval_units = 0.0
            clean_eval_units = path_batch_equiv_evals * rollout_multiplier if lambda_ref > 0.0 else 0.0
        else:
            lambda_ctrl_scaled = float(lambda_ctrl) / rollout_multiplier
            lambda_ref_scaled = float(lambda_ref) / rollout_multiplier
            for roll in rollouts:
                outer_loss_attack_val, outer_loss_clean_val, outer_loss_val = _path_outer_loss_backward(
                    cfg=cfg,
                    denoiser=denoiser,
                    states_ctrl=roll.states_ctrl,
                    states_ref=roll.states_ref,
                    x0=x0,
                    sigma_levels=sigma_levels,
                    lambda_ctrl=lambda_ctrl_scaled,
                    lambda_ref=lambda_ref_scaled,
                    amp_dtype=amp_dtype,
                )
                outer_loss_attack_vals.append(float(outer_loss_attack_val))
                outer_loss_clean_vals.append(float(outer_loss_clean_val))
                outer_loss_vals.append(float(outer_loss_val))
            attack_eval_units = path_batch_equiv_evals * 2.0 * rollout_multiplier
            clean_eval_units = path_batch_equiv_evals * rollout_multiplier if lambda_ref > 0.0 else 0.0
        optimizer_theta.step()
        update_ema_model(
            ema_model,
            denoiser,
            cfg,
            cur_nimg=float((step - 1) * int(cfg.batch_size)),
            batch_size=int(cfg.batch_size),
        )
        outer_loss_attack_val = sum(outer_loss_attack_vals) / len(outer_loss_attack_vals)
        outer_loss_clean_val = sum(outer_loss_clean_vals) / len(outer_loss_clean_vals)
        outer_loss_val = sum(outer_loss_vals) / len(outer_loss_vals)
        outer_loss_attack = torch.tensor(outer_loss_attack_val, device=x0.device, dtype=x0.dtype)
        outer_loss_clean = torch.tensor(outer_loss_clean_val, device=x0.device, dtype=x0.dtype)
        outer_loss = torch.tensor(outer_loss_val, device=x0.device, dtype=x0.dtype)
        step_batch_equiv_evals = attack_construction_units + attack_eval_units + clean_eval_units
        cumulative_batch_equiv_evals += step_batch_equiv_evals

        history["outer_loss"].append(scalarize(outer_loss))
        history["outer_loss_attack"].append(scalarize(outer_loss_attack))
        history["outer_loss_clean"].append(scalarize(outer_loss_clean))
        history["inner_obj"].append(last_inner_obj)
        history["energy"].append(last_transport)
        history["delta_norm_mean"].append(last_delta_norm_mean)
        history["delta_norm_max"].append(last_delta_norm_max)
        history["delta_norm_ratio_mean"].append(last_delta_ratio_mean)
        history["delta_norm_ratio_max"].append(last_delta_ratio_max)
        history["sched_attack_weight"].append(float(lambda_ctrl))
        history["sched_clean_weight"].append(float(lambda_ref))
        history["sched_phi_lr_scale"].append(float(phi_lr_scale))
        history["batch_equiv_denoiser_evals_step"].append(float(step_batch_equiv_evals))
        history["batch_equiv_denoiser_evals_attack_construction"].append(float(attack_construction_units))
        history["batch_equiv_denoiser_evals_attack_eval"].append(float(attack_eval_units))
        history["batch_equiv_denoiser_evals_clean_eval"].append(float(clean_eval_units))
        history["batch_equiv_denoiser_evals_cumulative"].append(float(cumulative_batch_equiv_evals))
        append_denoiser_op_count_step(
            history,
            n_fwd=float(path_batch_equiv_evals) * rollout_multiplier if attack_enabled else 0.0,
            n_fwd_inputgrad=float(attack_construction_units),
            n_fwd_parambackward=float(path_batch_equiv_evals)
            * float(int(lambda_ctrl > 0.0) + int(lambda_ref > 0.0))
            * rollout_multiplier,
        )

        run_diag = (
            bool(cfg.collapse_diagnostics_enabled)
            and (step % max(int(cfg.collapse_diag_every), 1) == 0 or step == 1 or step == int(cfg.steps))
        )
        if run_diag:
            if lambda_ctrl > 0.0 and cfg.inner_steps > 0:
                set_requires_grad(denoiser, False)
                roll_cur_diag = rollout_path_heuristic_attack(
                    cfg=cfg,
                    x0=x0,
                    target_indices=indices,
                    attack_net=denoiser,
                    sigma_levels=sigma_levels,
                    inner_steps=int(cfg.inner_steps),
                    step_size=step_size,
                    total_budget=total_budget,
                    time_horizon=time_horizon,
                )
                set_requires_grad(denoiser, True)
            else:
                with torch.no_grad():
                    roll_cur_diag = rollout_controlled_ve(
                        x0=x0,
                        target_indices=indices,
                        control_net=None,
                        sigma_levels=sigma_levels,
                        grad_through_control=False,
                        control_radius_kappa=cfg.control_radius_kappa,
                        kappa_by_step=None,
                        total_budget=total_budget,
                        time_horizon=time_horizon,
                        cfg=cfg,
                    )

            with torch.no_grad():
                with autocast_context(sigma_levels.device, amp_dtype):
                    attack_cur = _path_average_training_loss(cfg, denoiser, roll_cur_diag.states_ctrl, x0, sigma_levels)
                    inner_obj_cur = inner_objective_attack_only(attack_cur)

                roll_zero_diag = rollout_controlled_ve(
                    x0=x0,
                    target_indices=indices,
                    control_net=None,
                    sigma_levels=sigma_levels,
                    grad_through_control=False,
                    control_radius_kappa=cfg.control_radius_kappa,
                    kappa_by_step=None,
                    total_budget=total_budget,
                    time_horizon=time_horizon,
                    cfg=cfg,
                )
                with autocast_context(sigma_levels.device, amp_dtype):
                    attack_zero = _path_average_training_loss(cfg, denoiser, roll_zero_diag.states_ctrl, x0, sigma_levels)
                    inner_obj_zero = inner_objective_attack_only(attack_zero)
                gap = inner_obj_cur - inner_obj_zero
                gap_ratio = gap / (inner_obj_zero.abs() + 1e-8)
                (
                    path_delta_mean,
                    terminal_delta_mean,
                    delta_norm_mean,
                    delta_norm_max,
                    delta_ratio_mean,
                    delta_ratio_max,
                ) = _rollout_delta_diagnostics(
                    roll_cur_diag,
                    radius_by_step=radius_by_step,
                )

            history["diag_step"].append(int(step))
            history["diag_inner_obj_current"].append(scalarize(inner_obj_cur))
            history["diag_inner_obj_zero"].append(scalarize(inner_obj_zero))
            history["diag_inner_obj_gap"].append(scalarize(gap))
            history["diag_inner_obj_gap_ratio"].append(scalarize(gap_ratio))
            history["diag_delta_norm_mean"].append(scalarize(delta_norm_mean))
            history["diag_delta_norm_max"].append(scalarize(delta_norm_max))
            history["diag_delta_norm_ratio_mean"].append(scalarize(delta_ratio_mean))
            history["diag_delta_norm_ratio_max"].append(scalarize(delta_ratio_max))
            history["diag_path_delta_mean"].append(scalarize(path_delta_mean))
            history["diag_terminal_delta_mean"].append(scalarize(terminal_delta_mean))

        if step % cfg.log_every == 0:
            diag_msg = ""
            if len(history["diag_step"]) > 0:
                diag_msg = (
                    " "
                    f"diag_gap={history['diag_inner_obj_gap'][-1]:.6f} "
                    f"diag_gap_ratio={history['diag_inner_obj_gap_ratio'][-1]:.6f} "
                    f"diag_delta_ratio={history['diag_delta_norm_ratio_mean'][-1]:.6f} "
                    f"diag_path_delta={history['diag_path_delta_mean'][-1]:.6f}"
                )
            print(
                f"[cdro] step={step:05d} outer_loss={outer_loss.item():.6f} "
                f"attack={outer_loss_attack.item():.6f} clean={outer_loss_clean.item():.6f} "
                f"inner_obj={last_inner_obj:.6f} control_cost={last_transport:.6f} "
                f"delta_norm={last_delta_norm_mean:.6f} delta_ratio={last_delta_ratio_mean:.6f} "
                f"be_evals={step_batch_equiv_evals:.1f} be_evals_cum={cumulative_batch_equiv_evals:.1f} "
                f"lambda_ctrl={lambda_ctrl:.3f} lambda_ref={lambda_ref:.3f} phi_lr_scale={phi_lr_scale:.3f} "
                f"step_size={step_size:.6f} rho={total_budget:.4f} T={time_horizon:.4f}"
                f"{diag_msg}",
                flush=True,
            )

    if return_state:
        resume_robust_state_dict = copy.deepcopy(denoiser.state_dict())
        trainer_state = {
            "completed_steps": int(cfg.steps),
            "optimizer_theta_state": optimizer_theta.state_dict(),
            "ema_state_dict": None if ema_model is None else copy.deepcopy(ema_model.state_dict()),
            "resume_robust_state_dict": resume_robust_state_dict,
        }
        if ema_model is not None:
            denoiser.load_state_dict(ema_model.state_dict(), strict=True)
        return history, trainer_state
    if ema_model is not None:
        denoiser.load_state_dict(ema_model.state_dict(), strict=True)
    return history


def train_trajectory_robust_energy(
    denoiser,
    control,
    centers,
    sigma_levels,
    cfg,
    train_pool: torch.Tensor = None,
    sample_train_batch_fn: Optional[Callable[[int], torch.Tensor]] = None,
    sample_population_batch_fn: Optional[Callable[[int], torch.Tensor]] = None,
):
    """Backward-compatible alias."""

    return train_trajectory_robust_cdro(
        denoiser,
        control,
        centers,
        sigma_levels,
        cfg,
        train_pool=train_pool,
        sample_train_batch_fn=sample_train_batch_fn,
        sample_population_batch_fn=sample_population_batch_fn,
    )
