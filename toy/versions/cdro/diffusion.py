import math
from dataclasses import dataclass
from typing import Optional

import torch

from ...shared.objective import compute_training_loss, rf_time_levels_from_sigma_levels
from ...shared.runtime import autocast_context, resolve_amp_dtype


@dataclass
class RolloutResult:
    """Container for one Route-A CDRO rollout over the VE ladder."""

    x_target: torch.Tensor
    sigma_target: torch.Tensor
    states_ref: Optional[torch.Tensor] = None
    states_ctrl: Optional[torch.Tensor] = None
    delta_path: Optional[torch.Tensor] = None
    beta_path: Optional[torch.Tensor] = None


def _is_rf_objective(cfg) -> bool:
    return str(getattr(cfg, "training_objective", "edm")).lower() == "rf"


def project_l2_ball(delta_raw: torch.Tensor, radius: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Project each sample in `delta_raw` onto an L2 ball with per-sample radius."""

    flat = delta_raw.reshape(delta_raw.shape[0], -1)
    norm = flat.norm(dim=1, keepdim=True).clamp_min(eps)
    if radius.ndim == 1:
        radius = radius[:, None]
    scale = torch.minimum(torch.ones_like(norm), radius / norm)
    return (flat * scale).reshape_as(delta_raw)


def build_kappa_schedule(
    sigma_levels: torch.Tensor,
    base_kappa: float,
    use_time_dependent: bool = False,
    low_multiplier: float = 1.0,
    mid_multiplier: float = 1.0,
    high_multiplier: float = 1.0,
    preserve_l2_budget: bool = True,
) -> torch.Tensor:
    """Return per-transition control radius multipliers kappa_k for k->k+1."""

    n_steps = int(sigma_levels.numel() - 1)
    if n_steps <= 0:
        raise ValueError(f"sigma_levels must contain at least 2 values, got {sigma_levels.numel()}")

    kappa = torch.full((n_steps,), float(base_kappa), device=sigma_levels.device, dtype=sigma_levels.dtype)
    if not use_time_dependent:
        return kappa

    raw = torch.empty_like(kappa)
    one_third = max(n_steps // 3, 1)
    two_third = min(2 * one_third, n_steps)
    raw[:one_third] = float(low_multiplier)
    raw[one_third:two_third] = float(mid_multiplier)
    raw[two_third:] = float(high_multiplier)

    if preserve_l2_budget:
        denom = raw.square().sum().clamp_min(1e-12)
        scale = math.sqrt(float(n_steps) / float(denom.item()))
        raw = raw * scale

    return kappa * raw


def build_time_deltas(
    sigma_levels: torch.Tensor,
    time_horizon: float,
) -> torch.Tensor:
    """Build sigma-induced auxiliary-time deltas over the positive VE ladder.

    The clean anchor `sigma_0 = 0` is treated as a boundary condition, not as part of
    the log-sigma control clock. Consequently the first hop `0 -> sigma_min` receives
    zero sigma-time, while positive-noise hops are spaced by normalized log-sigma and
    scaled to `time_horizon`.
    """

    n_steps = int(sigma_levels.numel() - 1)
    if n_steps <= 0:
        raise ValueError(f"sigma_levels must contain at least 2 values, got {sigma_levels.numel()}")
    if time_horizon <= 0:
        raise ValueError(f"time_horizon must be > 0, got {time_horizon}")

    if n_steps == 1:
        return torch.full(
            (1,),
            float(time_horizon),
            device=sigma_levels.device,
            dtype=sigma_levels.dtype,
        )

    positive_sigma = sigma_levels[1:]
    if torch.any(positive_sigma <= 0):
        raise ValueError("sigma_levels[1:] must be strictly positive for the log-sigma CDRO clock.")

    dt = torch.zeros((n_steps,), device=sigma_levels.device, dtype=sigma_levels.dtype)
    log_sigma = positive_sigma.log()
    log_span = log_sigma[-1] - log_sigma[0]
    if float(log_span.abs().item()) <= 1e-12:
        dt[1:] = float(time_horizon) / float(n_steps - 1)
        return dt

    tau = float(time_horizon) * (log_sigma - log_sigma[0]) / log_span
    dt[1:] = tau[1:] - tau[:-1]
    return dt


def _build_time_deltas_for_objective(
    *,
    cfg,
    sigma_levels: torch.Tensor,
    time_horizon: float,
) -> torch.Tensor:
    """Build method time deltas for the configured generative family."""

    if _is_rf_objective(cfg):
        t_levels = rf_time_levels_from_sigma_levels(sigma_levels)
        dt = t_levels[1:] - t_levels[:-1]
        total = float(dt.sum().item())
        if total <= 0.0:
            raise ValueError("RF time grid must have positive total length.")
        return dt * (float(time_horizon) / total)
    return build_time_deltas(sigma_levels, time_horizon)


def _build_beta_budget_by_step(
    sigma_levels: torch.Tensor,
    total_budget: float,
    time_horizon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return sigma-time deltas and matching per-step beta-space budget shares."""

    dt = build_time_deltas(sigma_levels, time_horizon)
    total_tau = float(dt.sum().item())
    if total_tau <= 0:
        raise ValueError("sigma-induced CDRO time grid must have positive total length.")
    beta_budget_by_step = float(total_budget) * dt / total_tau
    return dt, beta_budget_by_step


def _build_beta_budget_by_step_for_objective(
    *,
    cfg,
    sigma_levels: torch.Tensor,
    total_budget: float,
    time_horizon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return family-aware time deltas and matching per-step beta-space budgets."""

    dt = _build_time_deltas_for_objective(cfg=cfg, sigma_levels=sigma_levels, time_horizon=time_horizon)
    total_tau = float(dt.sum().item())
    if total_tau <= 0:
        raise ValueError("CDRO time grid must have positive total length.")
    beta_budget_by_step = float(total_budget) * dt / total_tau
    return dt, beta_budget_by_step


def build_constraint_radii(
    sigma_levels: torch.Tensor,
    total_budget: float,
    time_horizon: float,
) -> torch.Tensor:
    """Exact Route-A local cap expressed in state-increment (`delta`) space."""

    n_steps = int(sigma_levels.numel() - 1)
    if n_steps <= 0:
        raise ValueError(f"sigma_levels must contain at least 2 values, got {sigma_levels.numel()}")
    if total_budget < 0:
        raise ValueError(f"total_budget must be >= 0, got {total_budget}")

    dt, beta_budget_by_step = _build_beta_budget_by_step(sigma_levels, total_budget, time_horizon)
    radius_sq = dt * beta_budget_by_step
    return torch.sqrt(radius_sq.clamp_min(0.0))


def build_constraint_radii_for_objective(
    *,
    cfg,
    sigma_levels: torch.Tensor,
    total_budget: float,
    time_horizon: float,
) -> torch.Tensor:
    """Exact Route-A local caps for the configured objective family."""

    n_steps = int(sigma_levels.numel() - 1)
    if n_steps <= 0:
        raise ValueError(f"sigma_levels must contain at least 2 values, got {sigma_levels.numel()}")
    if total_budget < 0:
        raise ValueError(f"total_budget must be >= 0, got {total_budget}")
    dt, beta_budget_by_step = _build_beta_budget_by_step_for_objective(
        cfg=cfg,
        sigma_levels=sigma_levels,
        total_budget=total_budget,
        time_horizon=time_horizon,
    )
    radius_sq = dt * beta_budget_by_step
    return torch.sqrt(radius_sq.clamp_min(0.0))


def _build_beta_radii_by_step(
    sigma_levels: torch.Tensor,
    total_budget: float,
    time_horizon: float,
) -> torch.Tensor:
    """Per-step beta-space radii with ||beta_k||^2 <= rho * Delta tau_k / T_tau."""

    _, beta_budget_by_step = _build_beta_budget_by_step(sigma_levels, total_budget, time_horizon)
    return torch.sqrt(beta_budget_by_step.clamp_min(0.0))


def _build_beta_radii_by_step_for_objective(
    *,
    cfg,
    sigma_levels: torch.Tensor,
    total_budget: float,
    time_horizon: float,
) -> torch.Tensor:
    """Per-step beta-space radii for the configured objective family."""

    _, beta_budget_by_step = _build_beta_budget_by_step_for_objective(
        cfg=cfg,
        sigma_levels=sigma_levels,
        total_budget=total_budget,
        time_horizon=time_horizon,
    )
    return torch.sqrt(beta_budget_by_step.clamp_min(0.0))


def _as_sigma_batch(value: torch.Tensor, batch_size: int, x_ref: torch.Tensor) -> torch.Tensor:
    """Expand scalar sigma to `[B]` in the same device/dtype as inputs."""

    return torch.full((batch_size,), float(value.item()), device=x_ref.device, dtype=x_ref.dtype)


def _l2_normalize_per_sample(value: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Return per-sample L2-normalized directions with zero fallback for zero gradients."""

    flat = value.reshape(value.shape[0], -1)
    norm = flat.norm(p=2, dim=1, keepdim=True).clamp_min(eps)
    return (flat / norm).reshape_as(value)


def _rf_terminal_noise(
    *,
    x0: torch.Tensor,
    sigma_levels: torch.Tensor,
    eps_schedule: Optional[torch.Tensor],
) -> torch.Tensor:
    """Resolve one terminal-noise sample per batch element for RF forward paths."""

    if eps_schedule is None:
        terminal_eps = torch.randn_like(x0)
    elif eps_schedule.ndim == x0.ndim + 1:
        terminal_eps = eps_schedule[-1].to(device=x0.device, dtype=x0.dtype)
    else:
        terminal_eps = eps_schedule.to(device=x0.device, dtype=x0.dtype)
    return terminal_eps * float(sigma_levels[-1].item())


def rollout_path_heuristic_attack(
    *,
    cfg,
    x0: torch.Tensor,
    target_indices: torch.Tensor,
    attack_net,
    sigma_levels: torch.Tensor,
    inner_steps: int,
    step_size: float,
    total_budget: float,
    time_horizon: float,
    eps_schedule: Optional[torch.Tensor] = None,
) -> RolloutResult:
    """Greedy Route-A CDRO attack with sigma-time-weighted local beta caps."""

    batch_size = x0.shape[0]
    n_steps = int(sigma_levels.numel() - 1)
    if n_steps <= 0:
        raise ValueError(f"sigma_levels must contain at least 2 values, got {sigma_levels.numel()}")
    if inner_steps < 0:
        raise ValueError(f"inner_steps must be >= 0, got {inner_steps}")
    if step_size <= 0:
        raise ValueError(f"step_size must be > 0, got {step_size}")
    if total_budget < 0:
        raise ValueError(f"total_budget must be >= 0, got {total_budget}")
    if time_horizon <= 0:
        raise ValueError(f"time_horizon must be > 0, got {time_horizon}")

    dt = _build_time_deltas_for_objective(cfg=cfg, sigma_levels=sigma_levels, time_horizon=time_horizon)
    sqrt_dt = torch.sqrt(dt.clamp_min(0.0))
    beta_radius_by_step = _build_beta_radii_by_step_for_objective(
        cfg=cfg,
        sigma_levels=sigma_levels,
        total_budget=total_budget,
        time_horizon=time_horizon,
    )
    amp_dtype = resolve_amp_dtype(x0.device, getattr(cfg, "amp_dtype", "auto"))
    rf_objective = _is_rf_objective(cfg)
    z_terminal = (
        _rf_terminal_noise(x0=x0, sigma_levels=sigma_levels, eps_schedule=eps_schedule)
        if rf_objective
        else None
    )

    x_ref = x0.detach()
    x_ctrl = x0.detach()
    path_ref = [x_ref]
    path_ctrl = [x_ctrl]
    path_delta = []
    path_beta = []

    was_training = attack_net.training
    attack_net.eval()

    for k in range(n_steps):
        sigma_k = sigma_levels[k]
        sigma_next = sigma_levels[k + 1]
        if rf_objective:
            delta_t = float(dt[k].item())
            base_increment = delta_t * (z_terminal - x0)
        else:
            delta_sigma = torch.sqrt((sigma_next.square() - sigma_k.square()).clamp_min(1e-8))
            eps = torch.randn_like(x0) if eps_schedule is None else eps_schedule[k]
            base_increment = delta_sigma * eps
        reference_state = (x_ref + base_increment).detach()
        x_nominal_next = (x_ctrl + base_increment).detach()
        sigma_batch = _as_sigma_batch(sigma_next, batch_size, x0)

        beta = torch.zeros_like(x_ctrl)
        step_sqrt_dt = float(sqrt_dt[k].item())
        step_beta_radius = float(beta_radius_by_step[k].item())
        if step_sqrt_dt > 0.0 and step_beta_radius > 0.0 and int(inner_steps) == 1:
            # With one inner step, solve the linearized local-cap problem exactly:
            # max_{||beta|| <= sqrt(rho_k)} <grad, beta> where rho_k = rho * Delta tau_k / T_tau.
            beta = beta.requires_grad_(True)
            candidate = x_nominal_next + step_sqrt_dt * beta
            with autocast_context(x0.device, amp_dtype):
                step_loss = compute_training_loss(cfg, attack_net, candidate, x0, sigma_batch)
            grad = torch.autograd.grad(step_loss, beta)[0]
            grad_unit = _l2_normalize_per_sample(grad)
            beta_radius = torch.full((batch_size,), step_beta_radius, device=x0.device, dtype=x0.dtype)
            beta = (grad_unit.reshape(batch_size, -1) * beta_radius[:, None]).reshape_as(grad).detach()
        elif step_sqrt_dt > 0.0 and step_beta_radius > 0.0:
            beta_radius = torch.full((batch_size,), step_beta_radius, device=x0.device, dtype=x0.dtype)
            for _ in range(int(inner_steps)):
                beta.requires_grad_(True)
                candidate = x_nominal_next + step_sqrt_dt * beta
                with autocast_context(x0.device, amp_dtype):
                    step_loss = compute_training_loss(cfg, attack_net, candidate, x0, sigma_batch)
                grad = torch.autograd.grad(step_loss, beta)[0]

                grad_unit = _l2_normalize_per_sample(grad)
                step = float(step_size) * grad_unit
                beta = (beta + step).detach()
                beta = project_l2_ball(beta, beta_radius).detach()

        delta_effective = step_sqrt_dt * beta
        candidate_final = (x_nominal_next + delta_effective).detach()
        x_ref = reference_state
        x_ctrl = candidate_final
        path_ref.append(x_ref)
        path_ctrl.append(x_ctrl)
        path_delta.append(delta_effective.detach())
        path_beta.append(beta.detach())

    if was_training:
        attack_net.train()

    states_ctrl = torch.stack(path_ctrl, dim=1)
    states_ref = torch.stack(path_ref, dim=1)
    delta_path = torch.stack(path_delta, dim=1)
    beta_path = torch.stack(path_beta, dim=1)
    x_target = states_ctrl[torch.arange(batch_size, device=x0.device), target_indices]
    sigma_target = sigma_levels[target_indices]
    return RolloutResult(
        x_target=x_target,
        sigma_target=sigma_target,
        states_ref=states_ref,
        states_ctrl=states_ctrl,
        delta_path=delta_path,
        beta_path=beta_path,
    )


def rollout_controlled_ve(
    x0: torch.Tensor,
    target_indices: torch.Tensor,
    control_net,
    sigma_levels: torch.Tensor,
    grad_through_control: bool,
    control_radius_kappa: float,
    kappa_by_step: Optional[torch.Tensor] = None,
    eps_schedule: Optional[torch.Tensor] = None,
    total_budget: Optional[float] = None,
    time_horizon: float = 1.0,
    cfg=None,
) -> RolloutResult:
    """Reference VE rollout for CDRO.

    CDRO training uses a denoiser-dependent beta-space attack during rollout construction,
    not a persistent learned control policy. For API compatibility we expose the shared-noise
    reference rollout with zero additional control increments.
    """

    del control_net, grad_through_control, control_radius_kappa, kappa_by_step, total_budget

    batch_size = x0.shape[0]
    n_steps = int(sigma_levels.numel() - 1)
    if n_steps <= 0:
        raise ValueError(f"sigma_levels must contain at least 2 values, got {sigma_levels.numel()}")
    rf_objective = bool(cfg is not None and _is_rf_objective(cfg))
    if cfg is not None:
        _ = _build_time_deltas_for_objective(cfg=cfg, sigma_levels=sigma_levels, time_horizon=time_horizon)
    else:
        _ = build_time_deltas(sigma_levels, time_horizon)
    z_terminal = (
        _rf_terminal_noise(x0=x0, sigma_levels=sigma_levels, eps_schedule=eps_schedule)
        if rf_objective
        else None
    )

    x_ref = x0
    path_ref = [x_ref]
    path_ctrl = [x_ref]
    path_delta = []
    path_beta = []

    for k in range(n_steps):
        sigma_k = sigma_levels[k]
        sigma_next = sigma_levels[k + 1]
        if rf_objective:
            t_levels = rf_time_levels_from_sigma_levels(sigma_levels).to(device=x0.device, dtype=x0.dtype)
            delta_t = float((t_levels[k + 1] - t_levels[k]).item())
            x_ref = x_ref + delta_t * (z_terminal - x0)
        else:
            delta_sigma = torch.sqrt((sigma_next.square() - sigma_k.square()).clamp_min(1e-8))
            eps = torch.randn_like(x0) if eps_schedule is None else eps_schedule[k]
            x_ref = x_ref + delta_sigma * eps
        path_ref.append(x_ref)
        path_ctrl.append(x_ref)
        path_delta.append(torch.zeros_like(x_ref))
        path_beta.append(torch.zeros_like(x_ref))

    states_ref = torch.stack(path_ref, dim=1)
    states_ctrl = torch.stack(path_ctrl, dim=1)
    delta_path = torch.stack(path_delta, dim=1)
    beta_path = torch.stack(path_beta, dim=1)
    x_target = states_ctrl[torch.arange(batch_size, device=x0.device), target_indices]
    sigma_target = sigma_levels[target_indices]

    return RolloutResult(
        x_target=x_target,
        sigma_target=sigma_target,
        states_ref=states_ref,
        states_ctrl=states_ctrl,
        delta_path=delta_path,
        beta_path=beta_path,
    )
