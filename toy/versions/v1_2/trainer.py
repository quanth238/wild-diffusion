from typing import Callable, Optional

import torch

from ...models import set_requires_grad
from ...shared.objective import compute_training_loss, inner_objective_attack_only
from ...shared.sigma import sample_target_indices, sample_target_indices_log_normal
from ...shared.train_utils import pathwise_l2, robust_schedule, sample_train_batch, zero_control
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
    x_noisy = states[:, 1:].reshape(x0.shape[0] * n_steps, *x0.shape[1:])
    x_clean = x0[:, None, ...].expand(x0.shape[0], n_steps, *x0.shape[1:]).reshape_as(x_noisy)
    sigma = sigma_levels[1:].view(1, n_steps).expand(x0.shape[0], n_steps).reshape(x0.shape[0] * n_steps)
    return compute_training_loss(cfg, denoiser, x_noisy, x_clean, sigma)


def _activation_scale(step: int, cfg) -> float:
    """Ramp robust control from 0 to 1 over configured step window."""

    start_step = max(int(cfg.v12_start_step), 0)
    ramp_steps = max(int(cfg.v12_ramp_steps), 0)
    if step <= start_step:
        return 0.0
    if ramp_steps <= 0:
        return 1.0
    progress = (step - start_step) / float(ramp_steps)
    return float(min(max(progress, 0.0), 1.0))


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
    """v1.2 robust training: v1.1 path attack + adaptive dual-lambda and sigma gating."""

    optimizer_theta = torch.optim.Adam(denoiser.parameters(), lr=cfg.lr_theta)
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
        "energy": [],
        "lambda_dual": [],
        "lambda_update": [],
        "robust_mix": [],
        "activation_scale": [],
        "delta_norm_ratio_mean": [],
        "delta_norm_ratio_max": [],
        "sched_attack_weight": [],
        "sched_clean_weight": [],
        "sched_phi_lr_scale": [],
        "transport_inner": [],
        "delta_norm_mean": [],
        "delta_norm_max": [],
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

    # v1.2 path-heuristic does not learn a global control policy.
    set_requires_grad(control, False)
    with torch.no_grad():
        for p in control.parameters():
            p.zero_()

    step_size = float(cfg.v12_step_size)
    lambda_dual = float(cfg.v12_lambda_init)
    rho_target = float(cfg.v12_rho_target)
    lambda_lr = float(cfg.v12_lambda_lr)

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

        _clean_sched, attack_sched, phi_lr_scale, control_updates_enabled = robust_schedule(step, cfg)
        activation = _activation_scale(step, cfg)
        robust_mix = float(cfg.v12_robust_mix) * activation
        attack_target = float(max(cfg.outer_attack_weight, 0.0))
        if attack_target > 0.0:
            sched_scale = float(attack_sched) / attack_target
            robust_mix = robust_mix * float(min(max(sched_scale, 0.0), 1.0))
        robust_mix = float(min(max(robust_mix, 0.0), 1.0))
        attack_enabled = bool(robust_mix > 0.0 and int(cfg.inner_steps) > 0 and control_updates_enabled)

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
                lambda_dual=lambda_dual,
                activation_scale=activation,
                max_delta=float(cfg.v12_max_delta),
                sigma_floor=float(cfg.v12_sigma_floor),
                sigma_cut=float(cfg.v12_sigma_cut),
                gate_power=float(cfg.v12_gate_power),
                delta_space=str(cfg.v12_delta_space).lower(),
                control_radius_kappa=cfg.control_radius_kappa,
                kappa_by_step=kappa_by_step,
            )
        else:
            roll = rollout_controlled_ve(
                x0=x0,
                target_indices=indices,
                control_net=zero_control,
                sigma_levels=sigma_levels,
                grad_through_control=False,
                control_radius_kappa=cfg.control_radius_kappa,
                kappa_by_step=kappa_by_step,
            )

        attack_loss_inner = _path_average_training_loss(cfg, denoiser, roll.states_ctrl, x0, sigma_levels)
        transport_inner = _path_transport_cost(roll.states_ctrl, roll.states_ref)
        inner_obj = inner_objective_attack_only(attack_loss_inner) - lambda_dual * transport_inner
        if has_nan_or_inf(inner_obj):
            raise RuntimeError("NaN/Inf detected in v1.2 inner objective.")

        lambda_delta = 0.0
        if attack_enabled:
            lambda_delta = lambda_lr * (float(transport_inner.item()) - rho_target)
            lambda_dual = float(max(0.0, lambda_dual + lambda_delta))

        delta_l2 = pathwise_l2(roll.delta_path)
        flat_dim = int(roll.delta_path.reshape(roll.delta_path.shape[0], roll.delta_path.shape[1], -1).shape[-1])
        l2_budget = float(cfg.v12_max_delta) * float(flat_dim) ** 0.5
        delta_ratio = delta_l2 / max(l2_budget, 1e-8)
        last_inner_obj = scalarize(inner_obj) if attack_enabled else 0.0
        last_transport = scalarize(transport_inner)
        last_delta_norm_mean = scalarize(delta_l2.mean())
        last_delta_norm_max = scalarize(delta_l2.max())
        last_delta_ratio_mean = scalarize(delta_ratio.mean())
        last_delta_ratio_max = scalarize(delta_ratio.max())

        set_requires_grad(denoiser, True)
        optimizer_theta.zero_grad(set_to_none=True)
        outer_loss_attack = _path_average_training_loss(cfg, denoiser, roll.states_ctrl, x0, sigma_levels)
        outer_loss_clean = _path_average_training_loss(cfg, denoiser, roll.states_ref, x0, sigma_levels)
        outer_loss = (1.0 - robust_mix) * outer_loss_clean + robust_mix * outer_loss_attack
        transport_outer = _path_transport_cost(roll.states_ctrl, roll.states_ref)
        if has_nan_or_inf(outer_loss):
            raise RuntimeError("NaN/Inf detected in v1.2 outer loss.")
        outer_loss.backward()
        optimizer_theta.step()

        history["outer_loss"].append(scalarize(outer_loss))
        history["outer_loss_attack"].append(scalarize(outer_loss_attack))
        history["outer_loss_clean"].append(scalarize(outer_loss_clean))
        history["inner_obj"].append(last_inner_obj)
        history["energy"].append(scalarize(transport_outer))
        history["lambda_dual"].append(float(lambda_dual))
        history["lambda_update"].append(float(lambda_delta))
        history["robust_mix"].append(float(robust_mix))
        history["activation_scale"].append(float(activation))
        history["delta_norm_ratio_mean"].append(last_delta_ratio_mean)
        history["delta_norm_ratio_max"].append(last_delta_ratio_max)
        history["sched_attack_weight"].append(float(robust_mix))
        history["sched_clean_weight"].append(float(1.0 - robust_mix))
        history["sched_phi_lr_scale"].append(float(phi_lr_scale))
        history["transport_inner"].append(float(last_transport))
        history["delta_norm_mean"].append(last_delta_norm_mean)
        history["delta_norm_max"].append(last_delta_norm_max)

        if step % cfg.log_every == 0:
            print(
                f"[robust-v1.2] step={step:05d} outer_loss={outer_loss.item():.6f} "
                f"attack={outer_loss_attack.item():.6f} clean={outer_loss_clean.item():.6f} "
                f"inner_obj={last_inner_obj:.6f} transport={last_transport:.6f} "
                f"delta_norm={last_delta_norm_mean:.6f} delta_ratio={last_delta_ratio_mean:.6f} "
                f"w_attack={robust_mix:.3f} w_clean={(1.0 - robust_mix):.3f} "
                f"lambda={lambda_dual:.6f} lambda_delta={lambda_delta:.6f} "
                f"activation={activation:.3f} phi_lr_scale={phi_lr_scale:.3f} step_size={step_size:.6f}",
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
