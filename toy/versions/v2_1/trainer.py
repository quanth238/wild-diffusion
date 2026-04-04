from typing import Callable, Optional

import torch

from ...compute_accounting import append_denoiser_op_count_step, ensure_denoiser_op_count_history
from ...models import set_requires_grad
from ...shared.objective import compute_training_loss, inner_objective_attack_only
from ...shared.sigma import sample_target_indices, sample_target_indices_log_normal
from ...shared.train_utils import (
    pathwise_l2,
    robust_schedule,
    rollout_control_diagnostics,
    sample_train_batch,
    zero_control,
)
from ...utils import has_nan_or_inf, scalarize
from .diffusion import build_kappa_schedule, rollout_controlled_ve


def train_trajectory_robust_constrained(
    denoiser,
    control,
    centers,
    sigma_levels,
    cfg,
    train_pool: torch.Tensor = None,
    sample_train_batch_fn: Optional[Callable[[int], torch.Tensor]] = None,
    sample_population_batch_fn: Optional[Callable[[int], torch.Tensor]] = None,
):
    """Bilevel robust training with hard per-step control and non-Markovian ref rollout."""

    optimizer_theta = torch.optim.Adam(denoiser.parameters(), lr=cfg.lr_theta)
    optimizer_phi = torch.optim.Adam(control.parameters(), lr=cfg.lr_phi)
    kappa_by_step = build_kappa_schedule(
        sigma_levels=sigma_levels,
        base_kappa=cfg.control_radius_kappa,
        use_time_dependent=cfg.use_time_dependent_kappa,
        low_multiplier=cfg.kappa_low_multiplier,
        mid_multiplier=cfg.kappa_mid_multiplier,
        high_multiplier=cfg.kappa_high_multiplier,
        preserve_l2_budget=cfg.kappa_preserve_l2_budget,
    ).to(device=sigma_levels.device, dtype=sigma_levels.dtype)
    history = {
        "outer_loss": [],
        "outer_loss_attack": [],
        "outer_loss_clean": [],
        "inner_obj": [],
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
    }
    ensure_denoiser_op_count_history(history)

    for step in range(1, cfg.steps + 1):
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
        for group in optimizer_phi.param_groups:
            group["lr"] = float(cfg.lr_phi) * float(phi_lr_scale)

        set_requires_grad(denoiser, False)
        set_requires_grad(control, True)
        last_inner_obj = 0.0
        last_delta_norm_mean = 0.0
        last_delta_norm_max = 0.0
        last_delta_ratio_mean = 0.0
        last_delta_ratio_max = 0.0
        attack_construction_units = 0.0

        if control_updates_enabled and phi_lr_scale > 0.0 and cfg.inner_steps > 0:
            for _ in range(cfg.inner_steps):
                optimizer_phi.zero_grad(set_to_none=True)
                roll = rollout_controlled_ve(
                    x0=x0,
                    target_indices=indices,
                    control_net=control,
                    sigma_levels=sigma_levels,
                    grad_through_control=True,
                    control_radius_kappa=cfg.control_radius_kappa,
                    non_markov_rho=cfg.v21_rho,
                    kappa_by_step=kappa_by_step,
                )
                train_loss = compute_training_loss(cfg, denoiser, roll.x_target, x0, roll.sigma_target)
                inner_obj = inner_objective_attack_only(train_loss)
                if has_nan_or_inf(inner_obj):
                    raise RuntimeError("NaN/Inf detected in inner objective.")
                (-inner_obj).backward()
                if cfg.clip_phi_grad > 0:
                    torch.nn.utils.clip_grad_norm_(control.parameters(), cfg.clip_phi_grad)
                optimizer_phi.step()
                last_inner_obj = scalarize(inner_obj)
                delta_l2 = pathwise_l2(roll.delta_path)
                last_delta_norm_mean = scalarize(delta_l2.mean())
                last_delta_norm_max = scalarize(delta_l2.max())
                sigma_k = sigma_levels[:-1]
                sigma_next = sigma_levels[1:]
                delta_sigma = torch.sqrt((sigma_next.square() - sigma_k.square()).clamp_min(1e-8))
                radius = (kappa_by_step * delta_sigma).view(1, -1).to(
                    device=delta_l2.device, dtype=delta_l2.dtype
                )
                delta_ratio = delta_l2 / radius.clamp_min(1e-8)
                last_delta_ratio_mean = scalarize(delta_ratio.mean())
                last_delta_ratio_max = scalarize(delta_ratio.max())
            attack_construction_units = float(max(int(cfg.inner_steps), 0))

        set_requires_grad(denoiser, True)
        set_requires_grad(control, False)
        optimizer_theta.zero_grad(set_to_none=True)
        roll = rollout_controlled_ve(
            x0=x0,
            target_indices=indices,
            control_net=control,
            sigma_levels=sigma_levels,
            grad_through_control=False,
            control_radius_kappa=cfg.control_radius_kappa,
            non_markov_rho=cfg.v21_rho,
            kappa_by_step=kappa_by_step,
        )
        outer_loss_attack = compute_training_loss(cfg, denoiser, roll.x_target, x0, roll.sigma_target)
        outer_loss_clean = torch.zeros((), device=x0.device, dtype=x0.dtype)
        if clean_weight > 0.0:
            x_ref_target = roll.states_ref[torch.arange(x0.shape[0], device=x0.device), indices]
            outer_loss_clean = compute_training_loss(cfg, denoiser, x_ref_target, x0, roll.sigma_target)
        outer_loss = attack_weight * outer_loss_attack + clean_weight * outer_loss_clean
        if has_nan_or_inf(outer_loss):
            raise RuntimeError("NaN/Inf detected in outer loss.")
        outer_loss.backward()
        optimizer_theta.step()
        set_requires_grad(control, True)
        if not control_updates_enabled:
            delta_l2 = pathwise_l2(roll.delta_path)
            last_delta_norm_mean = scalarize(delta_l2.mean())
            last_delta_norm_max = scalarize(delta_l2.max())
            sigma_k = sigma_levels[:-1]
            sigma_next = sigma_levels[1:]
            delta_sigma = torch.sqrt((sigma_next.square() - sigma_k.square()).clamp_min(1e-8))
            radius = (kappa_by_step * delta_sigma).view(1, -1).to(
                device=delta_l2.device, dtype=delta_l2.dtype
            )
            delta_ratio = delta_l2 / radius.clamp_min(1e-8)
            last_delta_ratio_mean = scalarize(delta_ratio.mean())
            last_delta_ratio_max = scalarize(delta_ratio.max())

        history["outer_loss"].append(scalarize(outer_loss))
        history["outer_loss_attack"].append(scalarize(outer_loss_attack))
        history["outer_loss_clean"].append(scalarize(outer_loss_clean))
        history["inner_obj"].append(last_inner_obj)
        history["delta_norm_mean"].append(last_delta_norm_mean)
        history["delta_norm_max"].append(last_delta_norm_max)
        history["delta_norm_ratio_mean"].append(last_delta_ratio_mean)
        history["delta_norm_ratio_max"].append(last_delta_ratio_max)
        history["sched_attack_weight"].append(float(attack_weight))
        history["sched_clean_weight"].append(float(clean_weight))
        history["sched_phi_lr_scale"].append(float(phi_lr_scale))
        append_denoiser_op_count_step(
            history,
            n_fwd=0.0,
            n_fwd_inputgrad=float(attack_construction_units),
            n_fwd_parambackward=float(int(attack_weight > 0.0) + int(clean_weight > 0.0)),
        )

        run_diag = (
            bool(cfg.collapse_diagnostics_enabled)
            and (step % max(int(cfg.collapse_diag_every), 1) == 0 or step == 1 or step == int(cfg.steps))
        )
        if run_diag:
            with torch.no_grad():
                roll_cur_diag = rollout_controlled_ve(
                    x0=x0,
                    target_indices=indices,
                    control_net=control,
                    sigma_levels=sigma_levels,
                    grad_through_control=False,
                    control_radius_kappa=cfg.control_radius_kappa,
                    non_markov_rho=cfg.v21_rho,
                    kappa_by_step=kappa_by_step,
                )
                train_loss_cur = compute_training_loss(
                    cfg,
                    denoiser,
                    roll_cur_diag.x_target,
                    x0,
                    roll_cur_diag.sigma_target,
                )
                inner_obj_cur = inner_objective_attack_only(train_loss_cur)

                roll_zero_diag = rollout_controlled_ve(
                    x0=x0,
                    target_indices=indices,
                    control_net=zero_control,
                    sigma_levels=sigma_levels,
                    grad_through_control=False,
                    control_radius_kappa=cfg.control_radius_kappa,
                    non_markov_rho=cfg.v21_rho,
                    kappa_by_step=kappa_by_step,
                )
                train_loss_zero = compute_training_loss(
                    cfg,
                    denoiser,
                    roll_zero_diag.x_target,
                    x0,
                    roll_zero_diag.sigma_target,
                )
                inner_obj_zero = inner_objective_attack_only(train_loss_zero)
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
                f"[robust] step={step:05d} outer_loss={outer_loss.item():.6f} "
                f"attack={outer_loss_attack.item():.6f} clean={outer_loss_clean.item():.6f} "
                f"inner_obj={last_inner_obj:.6f} "
                f"delta_norm={last_delta_norm_mean:.6f} delta_ratio={last_delta_ratio_mean:.6f} "
                f"w_attack={attack_weight:.3f} w_clean={clean_weight:.3f} phi_lr_scale={phi_lr_scale:.3f}"
                f"{diag_msg}",
                flush=True,
            )

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
    """Backward-compatible alias to constrained robust training."""

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
