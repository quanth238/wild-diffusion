from typing import Callable, Optional

import torch

from ...compute_accounting import append_denoiser_op_count_step, ensure_denoiser_op_count_history
from ...models import set_requires_grad
from ...shared.objective import compute_training_loss, inner_objective_attack_only
from ...shared.runtime import autocast_context, resolve_amp_dtype
from ...shared.sigma import sample_target_indices, sample_target_indices_log_normal
from ...shared.train_utils import (
    pathwise_l2,
    robust_schedule,
    rollout_control_diagnostics,
    sample_train_batch,
    zero_control,
)
from ...utils import has_nan_or_inf, scalarize
from .diffusion import build_kappa_schedule, rollout_controlled_ve, rollout_path_heuristic_attack


def _path_transport_cost(states_ctrl: torch.Tensor, states_ref: torch.Tensor) -> torch.Tensor:
    """Mean path transport cost: E[sum_t 0.5 * ||x_ctrl_t - x_ref_t||^2]."""

    if states_ctrl.shape != states_ref.shape:
        raise ValueError(f"states_ctrl/states_ref shape mismatch: {states_ctrl.shape} vs {states_ref.shape}")
    diff = states_ctrl[:, 1:] - states_ref[:, 1:]
    diff_sq = diff.reshape(diff.shape[0], diff.shape[1], -1).pow(2).sum(dim=2)
    return 0.5 * diff_sq.sum(dim=1).mean()


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
    attack_weight: float,
    clean_weight: float,
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
            if attack_weight > 0.0:
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

            if clean_weight > 0.0:
                clean_loss_step = compute_training_loss(
                    cfg,
                    denoiser,
                    states_ref[:, step_idx + 1],
                    x0,
                    sigma,
                )
            else:
                clean_loss_step = torch.zeros((), device=x0.device, dtype=x0.dtype)

            chunk_outer = attack_weight * attack_loss_step + clean_weight * clean_loss_step
            chunk_outer = chunk_outer / float(n_steps)

        if has_nan_or_inf(chunk_outer):
            raise RuntimeError("NaN/Inf detected in v1.1 outer loss.")
        if chunk_outer.requires_grad:
            chunk_outer.backward()

        attack_loss_total += scalarize(attack_loss_step) / float(n_steps)
        clean_loss_total += scalarize(clean_loss_step) / float(n_steps)
        outer_loss_total += scalarize(chunk_outer)

    return attack_loss_total, clean_loss_total, outer_loss_total


def _path_batch_equiv_denoiser_evals(sigma_levels: torch.Tensor) -> float:
    """Count one denoiser eval over `B*T` path states as `T` batch-equivalent evals."""

    return float(max(int(sigma_levels.numel()) - 1, 0))


def train_trajectory_robust_constrained(
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
    return_state: bool = False,
):
    """v1.1 robust training: legacy path-heuristic attack (batch-local control ascent)."""

    optimizer_theta = torch.optim.Adam(denoiser.parameters(), lr=cfg.lr_theta)
    if optimizer_theta_state is not None:
        optimizer_theta.load_state_dict(optimizer_theta_state)
        for state in optimizer_theta.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device=sigma_levels.device)
    kappa_by_step = build_kappa_schedule(
        sigma_levels=sigma_levels,
        base_kappa=cfg.control_radius_kappa,
        use_time_dependent=cfg.use_time_dependent_kappa,
        low_multiplier=cfg.kappa_low_multiplier,
        mid_multiplier=cfg.kappa_mid_multiplier,
        high_multiplier=cfg.kappa_high_multiplier,
        preserve_l2_budget=cfg.kappa_preserve_l2_budget,
    ).to(device=sigma_levels.device, dtype=sigma_levels.dtype)
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

    # v1.1 path-heuristic does not learn a global control policy.
    set_requires_grad(control, False)
    with torch.no_grad():
        for p in control.parameters():
            p.zero_()

    step_size = float(cfg.v11_step_size)
    gamma = float(cfg.v11_transport_gamma)
    total_budget = float(cfg.v11_total_budget_rho)
    projection_mode = str(cfg.v11_projection_mode).lower()
    path_batch_equiv_evals = _path_batch_equiv_denoiser_evals(sigma_levels)
    cumulative_batch_equiv_evals = float(history["batch_equiv_denoiser_evals_cumulative"][-1]) if history["batch_equiv_denoiser_evals_cumulative"] else 0.0
    amp_dtype = resolve_amp_dtype(sigma_levels.device, getattr(cfg, "amp_dtype", "auto"))

    for step in range(int(start_step) + 1, cfg.steps + 1):
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

        clean_weight, attack_weight, phi_lr_scale, control_updates_enabled = robust_schedule(step, cfg)
        attack_enabled = bool(control_updates_enabled and attack_weight > 0.0 and cfg.inner_steps > 0)
        attack_construction_units = 0.0

        set_requires_grad(denoiser, False)
        if attack_enabled:
            roll = rollout_path_heuristic_attack(
                cfg=cfg,
                x0=x0,
                target_indices=indices,
                attack_net=denoiser,
                sigma_levels=sigma_levels,
                inner_steps=int(cfg.inner_steps),
                step_size=step_size,
                gamma=gamma,
                total_budget=total_budget,
                projection_mode=projection_mode,
                control_radius_kappa=cfg.control_radius_kappa,
                kappa_by_step=kappa_by_step,
            )
            attack_construction_units = path_batch_equiv_evals * float(max(int(cfg.inner_steps), 0))
        else:
            roll = rollout_controlled_ve(
                x0=x0,
                target_indices=indices,
                control_net=zero_control,
                sigma_levels=sigma_levels,
                grad_through_control=False,
                control_radius_kappa=cfg.control_radius_kappa,
                kappa_by_step=kappa_by_step,
                total_budget=total_budget,
                projection_mode=projection_mode,
            )

        with autocast_context(sigma_levels.device, amp_dtype):
            attack_loss_inner = _path_average_training_loss(cfg, denoiser, roll.states_ctrl, x0, sigma_levels)
            transport_inner = _path_transport_cost(roll.states_ctrl, roll.states_ref)
            inner_obj = inner_objective_attack_only(attack_loss_inner) - gamma * transport_inner
        if has_nan_or_inf(inner_obj):
            raise RuntimeError("NaN/Inf detected in v1.1 inner objective.")

        delta_l2 = pathwise_l2(roll.delta_path)
        last_inner_obj = scalarize(inner_obj) if attack_enabled else 0.0
        last_transport = scalarize(transport_inner)
        last_delta_norm_mean = scalarize(delta_l2.mean())
        last_delta_norm_max = scalarize(delta_l2.max())
        sigma_k = sigma_levels[:-1]
        sigma_next = sigma_levels[1:]
        delta_sigma = torch.sqrt((sigma_next.square() - sigma_k.square()).clamp_min(1e-8))
        radius = (kappa_by_step * delta_sigma).view(1, -1).to(device=delta_l2.device, dtype=delta_l2.dtype)
        delta_ratio = delta_l2 / radius.clamp_min(1e-8)
        last_delta_ratio_mean = scalarize(delta_ratio.mean())
        last_delta_ratio_max = scalarize(delta_ratio.max())

        set_requires_grad(denoiser, True)
        optimizer_theta.zero_grad(set_to_none=True)
        outer_loss_attack_val, outer_loss_clean_val, outer_loss_val = _path_outer_loss_backward(
            cfg=cfg,
            denoiser=denoiser,
            states_ctrl=roll.states_ctrl,
            states_ref=roll.states_ref,
            x0=x0,
            sigma_levels=sigma_levels,
            attack_weight=float(attack_weight),
            clean_weight=float(clean_weight),
            amp_dtype=amp_dtype,
        )
        optimizer_theta.step()
        outer_loss_attack = torch.tensor(outer_loss_attack_val, device=x0.device, dtype=x0.dtype)
        outer_loss_clean = torch.tensor(outer_loss_clean_val, device=x0.device, dtype=x0.dtype)
        outer_loss = torch.tensor(outer_loss_val, device=x0.device, dtype=x0.dtype)
        with autocast_context(sigma_levels.device, amp_dtype):
            transport_outer = _path_transport_cost(roll.states_ctrl, roll.states_ref)
        attack_eval_units = path_batch_equiv_evals * 2.0
        clean_eval_units = path_batch_equiv_evals if clean_weight > 0.0 else 0.0
        step_batch_equiv_evals = attack_construction_units + attack_eval_units + clean_eval_units
        cumulative_batch_equiv_evals += step_batch_equiv_evals

        history["outer_loss"].append(scalarize(outer_loss))
        history["outer_loss_attack"].append(scalarize(outer_loss_attack))
        history["outer_loss_clean"].append(scalarize(outer_loss_clean))
        history["inner_obj"].append(last_inner_obj)
        history["energy"].append(scalarize(transport_outer))
        history["delta_norm_mean"].append(last_delta_norm_mean)
        history["delta_norm_max"].append(last_delta_norm_max)
        history["delta_norm_ratio_mean"].append(last_delta_ratio_mean)
        history["delta_norm_ratio_max"].append(last_delta_ratio_max)
        history["sched_attack_weight"].append(float(attack_weight))
        history["sched_clean_weight"].append(float(clean_weight))
        history["sched_phi_lr_scale"].append(float(phi_lr_scale))
        history["batch_equiv_denoiser_evals_step"].append(float(step_batch_equiv_evals))
        history["batch_equiv_denoiser_evals_attack_construction"].append(float(attack_construction_units))
        history["batch_equiv_denoiser_evals_attack_eval"].append(float(attack_eval_units))
        history["batch_equiv_denoiser_evals_clean_eval"].append(float(clean_eval_units))
        history["batch_equiv_denoiser_evals_cumulative"].append(float(cumulative_batch_equiv_evals))
        append_denoiser_op_count_step(
            history,
            n_fwd=float(path_batch_equiv_evals) if attack_enabled else 0.0,
            n_fwd_inputgrad=float(attack_construction_units),
            n_fwd_parambackward=float(path_batch_equiv_evals)
            * float(int(attack_weight > 0.0) + int(clean_weight > 0.0)),
        )

        run_diag = (
            bool(cfg.collapse_diagnostics_enabled)
            and (step % max(int(cfg.collapse_diag_every), 1) == 0 or step == 1 or step == int(cfg.steps))
        )
        if run_diag:
            if attack_weight > 0.0 and cfg.inner_steps > 0:
                set_requires_grad(denoiser, False)
                roll_cur_diag = rollout_path_heuristic_attack(
                    cfg=cfg,
                    x0=x0,
                    target_indices=indices,
                    attack_net=denoiser,
                    sigma_levels=sigma_levels,
                    inner_steps=int(cfg.inner_steps),
                    step_size=step_size,
                    gamma=gamma,
                    total_budget=total_budget,
                    projection_mode=projection_mode,
                    control_radius_kappa=cfg.control_radius_kappa,
                    kappa_by_step=kappa_by_step,
                )
                set_requires_grad(denoiser, True)
            else:
                with torch.no_grad():
                    roll_cur_diag = rollout_controlled_ve(
                        x0=x0,
                        target_indices=indices,
                        control_net=zero_control,
                        sigma_levels=sigma_levels,
                        grad_through_control=False,
                        control_radius_kappa=cfg.control_radius_kappa,
                        kappa_by_step=kappa_by_step,
                        total_budget=total_budget,
                        projection_mode=projection_mode,
                    )

            with torch.no_grad():
                with autocast_context(sigma_levels.device, amp_dtype):
                    attack_cur = _path_average_training_loss(cfg, denoiser, roll_cur_diag.states_ctrl, x0, sigma_levels)
                    transport_cur = _path_transport_cost(roll_cur_diag.states_ctrl, roll_cur_diag.states_ref)
                    inner_obj_cur = inner_objective_attack_only(attack_cur) - gamma * transport_cur

                roll_zero_diag = rollout_controlled_ve(
                    x0=x0,
                    target_indices=indices,
                    control_net=zero_control,
                    sigma_levels=sigma_levels,
                    grad_through_control=False,
                    control_radius_kappa=cfg.control_radius_kappa,
                    kappa_by_step=kappa_by_step,
                    total_budget=total_budget,
                    projection_mode=projection_mode,
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
                ) = rollout_control_diagnostics(
                    roll_cur_diag,
                    sigma_levels,
                    kappa_by_step,
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
                f"[robust-v1.1] step={step:05d} outer_loss={outer_loss.item():.6f} "
                f"attack={outer_loss_attack.item():.6f} clean={outer_loss_clean.item():.6f} "
                f"inner_obj={last_inner_obj:.6f} transport={last_transport:.6f} "
                f"delta_norm={last_delta_norm_mean:.6f} delta_ratio={last_delta_ratio_mean:.6f} "
                f"be_evals={step_batch_equiv_evals:.1f} be_evals_cum={cumulative_batch_equiv_evals:.1f} "
                f"w_attack={attack_weight:.3f} w_clean={clean_weight:.3f} phi_lr_scale={phi_lr_scale:.3f} "
                f"step_size={step_size:.6f} gamma={gamma:.4f} budget={total_budget:.4f} mode={projection_mode}"
                f"{diag_msg}",
                flush=True,
            )

    if return_state:
        return history, {
            "completed_steps": int(cfg.steps),
            "optimizer_theta_state": optimizer_theta.state_dict(),
        }
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

    return train_trajectory_robust_constrained(
        denoiser,
        control,
        centers,
        sigma_levels,
        cfg,
        train_pool=train_pool,
        sample_train_batch_fn=sample_train_batch_fn,
        sample_population_batch_fn=sample_population_batch_fn,
    )
