import math
from dataclasses import dataclass
from typing import Optional

import torch

from ...shared.objective import compute_training_loss, rf_time_levels_from_sigma_levels
from ...shared.runtime import autocast_context, resolve_amp_dtype
from ...shared.sigma import build_log_sigma_quantile_ladder
from ...utils import batch_scalar_like

DETERMINISTIC_MIDPOINT_QUANTILE_LADDER = "deterministic_midpoint_quantile"
STOCHASTIC_STRATIFIED_QUANTILE_LADDER = "stochastic_stratified_quantile"


@dataclass
class RolloutResult:
    """Container for one Route-A CDRO rollout over the VE ladder."""

    x_target: torch.Tensor
    sigma_target: torch.Tensor
    x_left: Optional[torch.Tensor] = None
    x_right: Optional[torch.Tensor] = None
    states_ref: Optional[torch.Tensor] = None
    states_ctrl: Optional[torch.Tensor] = None
    delta_path: Optional[torch.Tensor] = None
    control_path: Optional[torch.Tensor] = None


def _is_rf_objective(cfg) -> bool:
    return str(getattr(cfg, "training_objective", "edm")).lower() == "rf"


def resolve_cdro_edm_ladder_mode(cfg) -> str:
    """Resolve the CDRO-EDM continuation ladder construction mode."""

    mode = str(
        getattr(cfg, "cdro_edm_ladder_mode", DETERMINISTIC_MIDPOINT_QUANTILE_LADDER)
    ).strip().lower()
    if mode in ("deterministic", "midpoint", "midpoint_quantile", DETERMINISTIC_MIDPOINT_QUANTILE_LADDER):
        return DETERMINISTIC_MIDPOINT_QUANTILE_LADDER
    if mode in ("stochastic", "stratified", "stochastic_stratified", STOCHASTIC_STRATIFIED_QUANTILE_LADDER):
        return STOCHASTIC_STRATIFIED_QUANTILE_LADDER
    raise ValueError(
        "--cdro-edm-ladder-mode must be one of "
        f"('{DETERMINISTIC_MIDPOINT_QUANTILE_LADDER}', '{STOCHASTIC_STRATIFIED_QUANTILE_LADDER}'), got {mode}"
    )


def _num_path_steps(sigma_levels: torch.Tensor) -> int:
    if sigma_levels.ndim not in (1, 2):
        raise ValueError(f"sigma_levels must be rank 1 or 2, got shape={tuple(sigma_levels.shape)}")
    return int(sigma_levels.shape[-1] - 1)


def _step_scalar_vector(values: torch.Tensor, step_idx: int, *, batch_size: int, reference: torch.Tensor) -> torch.Tensor:
    if values.ndim == 1:
        return torch.full(
            (batch_size,),
            float(values[step_idx].item()),
            device=reference.device,
            dtype=reference.dtype,
        )
    if values.ndim == 2:
        return values[:, step_idx].to(device=reference.device, dtype=reference.dtype)
    raise ValueError(f"values must be rank 1 or 2, got shape={tuple(values.shape)}")


def _path_step_sigma_batch(sigma_levels: torch.Tensor, step_idx: int, x_ref: torch.Tensor) -> torch.Tensor:
    return _step_scalar_vector(sigma_levels, step_idx, batch_size=x_ref.shape[0], reference=x_ref)


def _step_scalar_like(values: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if values.ndim != 1:
        raise ValueError(f"Expected rank-1 per-sample values, got shape={tuple(values.shape)}")
    return batch_scalar_like(values, reference).to(device=reference.device, dtype=reference.dtype)


def _gather_target_sigmas(sigma_levels: torch.Tensor, target_indices: torch.Tensor, x_ref: torch.Tensor) -> torch.Tensor:
    if sigma_levels.ndim == 1:
        return sigma_levels[target_indices].to(device=x_ref.device, dtype=x_ref.dtype)
    if sigma_levels.ndim == 2:
        batch_index = torch.arange(target_indices.shape[0], device=target_indices.device)
        return sigma_levels[batch_index, target_indices].to(device=x_ref.device, dtype=x_ref.dtype)
    raise ValueError(f"sigma_levels must be rank 1 or 2, got shape={tuple(sigma_levels.shape)}")


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

    n_steps = _num_path_steps(sigma_levels)
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


def build_tau_levels(
    sigma_levels: torch.Tensor,
    time_horizon: float,
    *,
    sigma_min: float,
    sigma_max: float,
) -> torch.Tensor:
    """Build the positive continuation tau ladder from the actual sigma ladder."""

    n_steps = _num_path_steps(sigma_levels)
    if n_steps <= 0:
        raise ValueError(f"sigma_levels must contain at least 2 ladder points, got shape={tuple(sigma_levels.shape)}")
    if time_horizon <= 0:
        raise ValueError(f"time_horizon must be > 0, got {time_horizon}")
    if sigma_min <= 0.0 or sigma_max <= 0.0:
        raise ValueError(f"sigma_min and sigma_max must be positive, got {sigma_min}, {sigma_max}")
    if sigma_min >= sigma_max:
        raise ValueError(f"sigma_min must be < sigma_max, got {sigma_min} >= {sigma_max}")

    positive_sigma = sigma_levels[..., 1:]
    if torch.any(positive_sigma <= 0):
        raise ValueError("sigma_levels[1:] must be strictly positive for the log-sigma CDRO clock.")
    if positive_sigma.shape[-1] > 1 and not torch.all(positive_sigma[..., 1:] > positive_sigma[..., :-1]):
        raise ValueError("Continuation sigma ladder must be strictly increasing.")

    log_span = math.log(float(sigma_max)) - math.log(float(sigma_min))
    if log_span <= 0.0:
        raise ValueError("Log-sigma continuation span must be positive.")

    tau = float(time_horizon) * (positive_sigma.log() - math.log(float(sigma_min))) / log_span
    if torch.any(tau <= 0):
        raise ValueError("Continuation tau ladder must be strictly positive.")
    if tau.shape[-1] > 1 and not torch.all(tau[..., 1:] > tau[..., :-1]):
        raise ValueError("Continuation tau ladder must be strictly increasing.")
    return tau


def build_time_deltas(
    sigma_levels: torch.Tensor,
    time_horizon: float,
    *,
    sigma_min: float,
    sigma_max: float,
) -> torch.Tensor:
    """Build strictly positive transition sizes aligned to the positive continuation ladder."""

    tau = build_tau_levels(
        sigma_levels,
        time_horizon,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
    )
    dt = torch.empty_like(tau)
    dt[..., 0] = tau[..., 0]
    if tau.shape[-1] > 1:
        dt[..., 1:] = tau[..., 1:] - tau[..., :-1]
    if torch.any(dt <= 0):
        raise ValueError("All continuation transition sizes Delta_tau_k must be strictly positive.")
    return dt


def _build_rf_path_deltas(sigma_levels: torch.Tensor, *, sigma_max: Optional[float] = None) -> torch.Tensor:
    """Return physical straight-path RF increments over the normalized t-grid."""

    if sigma_max is None:
        t_levels = rf_time_levels_from_sigma_levels(sigma_levels)
    else:
        sigma_max_value = max(float(sigma_max), 1e-8)
        t_levels = (sigma_levels / sigma_max_value).clamp(0.0, 1.0)
    dt = t_levels[..., 1:] - t_levels[..., :-1]
    total = dt.sum(dim=-1) if dt.ndim == 2 else dt.sum()
    if torch.any(total <= 0):
        raise ValueError("RF time grid must have positive total length.")
    return dt


def build_transition_deltas_for_objective(
    *,
    cfg,
    sigma_levels: torch.Tensor,
    time_horizon: float,
) -> torch.Tensor:
    """Build method time deltas for the configured generative family."""

    if _is_rf_objective(cfg):
        dt = _build_rf_path_deltas(
            sigma_levels,
            sigma_max=float(getattr(cfg, "sigma_max", float(sigma_levels[..., -1].max().item()))),
        )
        total = dt.sum(dim=-1, keepdim=True) if dt.ndim == 2 else dt.sum()
        return dt * (float(time_horizon) / total)
    ladder_mode = resolve_cdro_edm_ladder_mode(cfg)
    if ladder_mode == STOCHASTIC_STRATIFIED_QUANTILE_LADDER:
        return build_time_deltas(
            sigma_levels,
            time_horizon,
            sigma_min=float(getattr(cfg, "sigma_min", 0.0)),
            sigma_max=float(getattr(cfg, "sigma_max", 0.0)),
        )
    ladder = _build_checked_warmup_quantile_ladder(cfg=cfg, sigma_levels=sigma_levels)
    z_edges = ladder.log_sigma_edges
    if z_edges is None:
        raise ValueError("Deterministic warmup-quantile cell edges are required for edge-based Delta_tau.")
    sigma_min = float(getattr(cfg, "sigma_min", 0.0))
    sigma_max = float(getattr(cfg, "sigma_max", 0.0))
    log_span = math.log(sigma_max) - math.log(sigma_min)
    if log_span <= 0.0:
        raise ValueError("Log-sigma continuation span must be positive.")
    tau_edges = float(time_horizon) * (z_edges - math.log(sigma_min)) / log_span
    dt = tau_edges[1:] - tau_edges[:-1]
    if torch.any(dt <= 0):
        raise ValueError("All continuation transition sizes Delta_tau_k must be strictly positive.")
    return dt


def _build_checked_warmup_quantile_ladder(
    *,
    cfg,
    sigma_levels: torch.Tensor,
    atol: float = 1e-6,
    rtol: float = 1e-5,
):
    """Build the config-implied warmup ladder and assert it matches runtime sigma levels."""

    if sigma_levels.ndim != 1:
        raise ValueError("Deterministic midpoint quantile ladders currently require shared rank-1 sigma_levels.")
    n_steps = _num_path_steps(sigma_levels)
    ladder = build_log_sigma_quantile_ladder(
        float(getattr(cfg, "sigma_min", 0.0)),
        float(getattr(cfg, "sigma_max", 0.0)),
        n_steps,
        p_mean=float(getattr(cfg, "p_mean", -1.2)),
        p_std=float(getattr(cfg, "p_std", 1.2)),
        device=sigma_levels.device,
        dtype=sigma_levels.dtype,
    )
    expected = ladder.sigma_levels
    if sigma_levels.shape != expected.shape:
        raise ValueError(
            "Runtime sigma_levels shape does not match the warmup-quantile ladder implied by cfg: "
            f"got {tuple(sigma_levels.shape)} vs expected {tuple(expected.shape)}."
        )
    if not torch.allclose(sigma_levels, expected, atol=float(atol), rtol=float(rtol)):
        diff = (sigma_levels - expected).abs()
        flat_idx = int(diff.reshape(-1).argmax().item())
        max_abs = float(diff.reshape(-1)[flat_idx].item())
        expected_flat = expected.reshape(-1)
        rel_base = expected_flat.abs().clamp_min(torch.finfo(expected.dtype).eps)
        max_rel = float((diff.reshape(-1)[flat_idx] / rel_base[flat_idx]).item())
        raise ValueError(
            "Runtime sigma_levels is not the midpoint ladder implied by the configured warmup quantile law. "
            f"max_abs_diff={max_abs:.3e} max_rel_diff={max_rel:.3e} at flat_index={flat_idx}."
        )
    return ladder


def _build_time_deltas_for_objective(
    *,
    cfg,
    sigma_levels: torch.Tensor,
    time_horizon: float,
) -> torch.Tensor:
    """Backward-compatible alias for family-aware continuation transition sizes."""

    return build_transition_deltas_for_objective(cfg=cfg, sigma_levels=sigma_levels, time_horizon=time_horizon)


def _build_control_radius(
    total_budget: float,
    time_horizon: float,
) -> float:
    """Return the shared per-step Euclidean projection radius in u-space."""

    if total_budget < 0:
        raise ValueError(f"total_budget must be >= 0, got {total_budget}")
    if time_horizon <= 0:
        raise ValueError(f"time_horizon must be > 0, got {time_horizon}")
    if total_budget == 0:
        return 0.0
    return math.sqrt(float(total_budget) / float(time_horizon))


def build_constraint_radii(
    sigma_levels: torch.Tensor,
    total_budget: float,
    time_horizon: float,
    *,
    sigma_min: float,
    sigma_max: float,
) -> torch.Tensor:
    """Exact Route-A local cap expressed in state-increment (`delta`) space."""

    n_steps = _num_path_steps(sigma_levels)
    if n_steps <= 0:
        raise ValueError(f"sigma_levels must contain at least 2 ladder points, got shape={tuple(sigma_levels.shape)}")
    if total_budget < 0:
        raise ValueError(f"total_budget must be >= 0, got {total_budget}")

    dt = build_time_deltas(
        sigma_levels,
        time_horizon,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
    )
    control_radius = _build_control_radius(total_budget, time_horizon)
    return dt * float(control_radius)


def build_constraint_radii_for_objective(
    *,
    cfg,
    sigma_levels: torch.Tensor,
    total_budget: float,
    time_horizon: float,
) -> torch.Tensor:
    """Exact Route-A local delta caps implied by the u-space projection rule."""

    n_steps = _num_path_steps(sigma_levels)
    if n_steps <= 0:
        raise ValueError(f"sigma_levels must contain at least 2 ladder points, got shape={tuple(sigma_levels.shape)}")
    if total_budget < 0:
        raise ValueError(f"total_budget must be >= 0, got {total_budget}")
    dt = build_transition_deltas_for_objective(cfg=cfg, sigma_levels=sigma_levels, time_horizon=time_horizon)
    control_radius = _build_control_radius(total_budget, time_horizon)
    return dt * float(control_radius)


def _as_sigma_batch(value: torch.Tensor, batch_size: int, x_ref: torch.Tensor) -> torch.Tensor:
    """Expand scalar sigma or cast per-sample sigma to `[B]`."""

    if value.ndim == 0:
        return torch.full((batch_size,), float(value.item()), device=x_ref.device, dtype=x_ref.dtype)
    if value.ndim == 1 and value.shape[0] == batch_size:
        return value.to(device=x_ref.device, dtype=x_ref.dtype)
    raise ValueError(f"value must be scalar or [batch], got shape={tuple(value.shape)}")


def _l2_normalize_per_sample(value: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Return per-sample L2-normalized directions with zero fallback for zero gradients."""

    flat = value.reshape(value.shape[0], -1)
    norm = flat.norm(p=2, dim=1, keepdim=True).clamp_min(eps)
    return (flat / norm).reshape_as(value)


def _rf_pair_left_source(
    *,
    x0: torch.Tensor,
    sigma_levels: torch.Tensor,
    eps_schedule: Optional[torch.Tensor],
    rf_pair_left: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Resolve the RF source endpoint x_L for the active pair law."""

    del sigma_levels
    if rf_pair_left is not None:
        if tuple(rf_pair_left.shape) != tuple(x0.shape):
            raise ValueError(
                f"rf_pair_left must match x0 shape {tuple(x0.shape)}, got {tuple(rf_pair_left.shape)}"
            )
        return rf_pair_left.to(device=x0.device, dtype=x0.dtype)
    if eps_schedule is None:
        terminal_eps = torch.randn_like(x0)
    elif eps_schedule.ndim == x0.ndim + 1:
        terminal_eps = eps_schedule[-1].to(device=x0.device, dtype=x0.dtype)
    else:
        terminal_eps = eps_schedule.to(device=x0.device, dtype=x0.dtype)
    # Straight-path RF stage-1 pairs use x_L = z ~ N(0, I).
    return terminal_eps


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
    rf_pair_left: Optional[torch.Tensor] = None,
) -> RolloutResult:
    """Greedy Route-A CDRO attack with literal u-space drift controls."""

    batch_size = x0.shape[0]
    n_steps = _num_path_steps(sigma_levels)
    if n_steps <= 0:
        raise ValueError(f"sigma_levels must contain at least 2 ladder points, got shape={tuple(sigma_levels.shape)}")
    if inner_steps < 0:
        raise ValueError(f"inner_steps must be >= 0, got {inner_steps}")
    if step_size <= 0:
        raise ValueError(f"step_size must be > 0, got {step_size}")
    if total_budget < 0:
        raise ValueError(f"total_budget must be >= 0, got {total_budget}")
    if time_horizon <= 0:
        raise ValueError(f"time_horizon must be > 0, got {time_horizon}")

    dt = build_transition_deltas_for_objective(cfg=cfg, sigma_levels=sigma_levels, time_horizon=time_horizon)
    control_radius = _build_control_radius(total_budget, time_horizon)
    amp_dtype = resolve_amp_dtype(x0.device, getattr(cfg, "amp_dtype", "auto"))
    rf_objective = _is_rf_objective(cfg)
    rf_path_dt = (
        _build_rf_path_deltas(
            sigma_levels,
            sigma_max=float(getattr(cfg, "sigma_max", float(sigma_levels[..., -1].max().item()))),
        )
        if rf_objective
        else None
    )
    x_left = (
        _rf_pair_left_source(
            x0=x0,
            sigma_levels=sigma_levels,
            eps_schedule=eps_schedule,
            rf_pair_left=rf_pair_left,
        )
        if rf_objective
        else None
    )

    x_ref = x_left.detach() if rf_objective else x0.detach()
    x_ctrl = x_left.detach() if rf_objective else x0.detach()
    path_ref = [x_ref]
    path_ctrl = [x_ctrl]
    path_delta = []
    path_control = []

    was_training = attack_net.training
    attack_net.eval()

    for k in range(n_steps):
        sigma_k = _path_step_sigma_batch(sigma_levels, k, x0)
        sigma_next = _path_step_sigma_batch(sigma_levels, k + 1, x0)
        if rf_objective:
            # The reference path follows the normalized RF straight path.
            # The auxiliary CDRO clock only scales control magnitudes and budgets.
            delta_t = _step_scalar_vector(rf_path_dt, k, batch_size=batch_size, reference=x0)
            base_increment = _step_scalar_like(delta_t, x0) * (x0 - x_left)
        else:
            variance_increment = sigma_next.square() - sigma_k.square()
            if float(variance_increment.min().item()) < -1e-12:
                raise ValueError("VE variance increments must be nonnegative.")
            delta_sigma = torch.sqrt(variance_increment.clamp_min(0.0))
            eps = torch.randn_like(x0) if eps_schedule is None else eps_schedule[k]
            base_increment = _step_scalar_like(delta_sigma, x0) * eps
        reference_state = (x_ref + base_increment).detach()
        x_nominal_next = (x_ctrl + base_increment).detach()
        sigma_batch = _as_sigma_batch(sigma_next, batch_size, x0)

        control = torch.zeros_like(x_ctrl)
        step_delta_tau = _step_scalar_vector(dt, k, batch_size=batch_size, reference=x0)
        step_delta_tau_value = float(step_delta_tau.max().item())
        step_delta_tau_full = _step_scalar_like(step_delta_tau, x0)
        if step_delta_tau_value > 0.0 and control_radius > 0.0 and int(inner_steps) == 1:
            # With one inner step, solve the linearized local-cap problem exactly:
            # max_{||u|| <= sqrt(rho / T_tau)} <grad, u> under the local u-space cap.
            control = control.requires_grad_(True)
            candidate = x_nominal_next + step_delta_tau_full * control
            with autocast_context(x0.device, amp_dtype):
                step_loss = compute_training_loss(
                    cfg,
                    attack_net,
                    candidate,
                    x0,
                    sigma_batch,
                    x_left=x_left,
                    x_right=x0,
                )
            grad = torch.autograd.grad(step_loss, control)[0]
            grad_unit = _l2_normalize_per_sample(grad)
            control_cap = torch.full((batch_size,), control_radius, device=x0.device, dtype=x0.dtype)
            control = (grad_unit.reshape(batch_size, -1) * control_cap[:, None]).reshape_as(grad).detach()
        elif step_delta_tau_value > 0.0 and control_radius > 0.0:
            control_cap = torch.full((batch_size,), control_radius, device=x0.device, dtype=x0.dtype)
            for _ in range(int(inner_steps)):
                control.requires_grad_(True)
                candidate = x_nominal_next + step_delta_tau_full * control
                with autocast_context(x0.device, amp_dtype):
                    step_loss = compute_training_loss(
                        cfg,
                        attack_net,
                        candidate,
                        x0,
                        sigma_batch,
                        x_left=x_left,
                        x_right=x0,
                    )
                grad = torch.autograd.grad(step_loss, control)[0]

                grad_unit = _l2_normalize_per_sample(grad)
                step = float(step_size) * grad_unit
                control = (control + step).detach()
                control = project_l2_ball(control, control_cap).detach()

        delta_effective = step_delta_tau_full * control
        candidate_final = (x_nominal_next + delta_effective).detach()
        x_ref = reference_state
        x_ctrl = candidate_final
        path_ref.append(x_ref)
        path_ctrl.append(x_ctrl)
        path_delta.append(delta_effective.detach())
        path_control.append(control.detach())

    if was_training:
        attack_net.train()

    states_ctrl = torch.stack(path_ctrl, dim=1)
    states_ref = torch.stack(path_ref, dim=1)
    delta_path = torch.stack(path_delta, dim=1)
    control_path = torch.stack(path_control, dim=1)
    x_target = states_ctrl[torch.arange(batch_size, device=x0.device), target_indices]
    sigma_target = _gather_target_sigmas(sigma_levels, target_indices, x0)
    return RolloutResult(
        x_target=x_target,
        sigma_target=sigma_target,
        x_left=x_left,
        x_right=x0 if rf_objective else None,
        states_ref=states_ref,
        states_ctrl=states_ctrl,
        delta_path=delta_path,
        control_path=control_path,
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
    rf_pair_left: Optional[torch.Tensor] = None,
) -> RolloutResult:
    """Reference VE rollout for CDRO.

    CDRO training uses a denoiser-dependent u-space attack during rollout construction,
    not a persistent learned control policy. For API compatibility we expose the shared-noise
    reference rollout with zero additional control increments.
    """

    del control_net, grad_through_control, control_radius_kappa, kappa_by_step, total_budget

    batch_size = x0.shape[0]
    n_steps = _num_path_steps(sigma_levels)
    if n_steps <= 0:
        raise ValueError(f"sigma_levels must contain at least 2 ladder points, got shape={tuple(sigma_levels.shape)}")
    rf_objective = bool(cfg is not None and _is_rf_objective(cfg))
    rf_path_dt = (
        _build_rf_path_deltas(
            sigma_levels,
            sigma_max=float(getattr(cfg, "sigma_max", float(sigma_levels[..., -1].max().item()))),
        )
        if rf_objective
        else None
    )
    x_left = (
        _rf_pair_left_source(
            x0=x0,
            sigma_levels=sigma_levels,
            eps_schedule=eps_schedule,
            rf_pair_left=rf_pair_left,
        )
        if rf_objective
        else None
    )

    x_ref = x_left if rf_objective else x0
    path_ref = [x_ref]
    path_ctrl = [x_ref]
    path_delta = []
    path_control = []

    for k in range(n_steps):
        sigma_k = _path_step_sigma_batch(sigma_levels, k, x0)
        sigma_next = _path_step_sigma_batch(sigma_levels, k + 1, x0)
        if rf_objective:
            delta_t = _step_scalar_vector(rf_path_dt, k, batch_size=batch_size, reference=x0)
            x_ref = x_ref + _step_scalar_like(delta_t, x0) * (x0 - x_left)
        else:
            variance_increment = sigma_next.square() - sigma_k.square()
            if float(variance_increment.min().item()) < -1e-12:
                raise ValueError("VE variance increments must be nonnegative.")
            delta_sigma = torch.sqrt(variance_increment.clamp_min(0.0))
            eps = torch.randn_like(x0) if eps_schedule is None else eps_schedule[k]
            x_ref = x_ref + _step_scalar_like(delta_sigma, x0) * eps
        path_ref.append(x_ref)
        path_ctrl.append(x_ref)
        path_delta.append(torch.zeros_like(x_ref))
        path_control.append(torch.zeros_like(x_ref))

    states_ref = torch.stack(path_ref, dim=1)
    states_ctrl = torch.stack(path_ctrl, dim=1)
    delta_path = torch.stack(path_delta, dim=1)
    control_path = torch.stack(path_control, dim=1)
    x_target = states_ctrl[torch.arange(batch_size, device=x0.device), target_indices]
    sigma_target = _gather_target_sigmas(sigma_levels, target_indices, x0)

    return RolloutResult(
        x_target=x_target,
        sigma_target=sigma_target,
        x_left=x_left,
        x_right=x0 if rf_objective else None,
        states_ref=states_ref,
        states_ctrl=states_ctrl,
        delta_path=delta_path,
        control_path=control_path,
    )
