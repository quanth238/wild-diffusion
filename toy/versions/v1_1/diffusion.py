import math
from dataclasses import dataclass
from typing import Optional

import torch

from ...shared.objective import compute_training_loss
from ...shared.runtime import autocast_context, resolve_amp_dtype


@dataclass
class RolloutResult:
    """Container for one controlled forward rollout over the VE ladder."""

    x_target: torch.Tensor
    sigma_target: torch.Tensor
    states_ref: Optional[torch.Tensor] = None
    states_ctrl: Optional[torch.Tensor] = None
    delta_path: Optional[torch.Tensor] = None


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


def _project_to_remaining_budget(
    candidate: torch.Tensor,
    reference_state: torch.Tensor,
    used_transport_cost: torch.Tensor,
    total_budget: float,
) -> torch.Tensor:
    """Project candidate state onto per-sample remaining budget ball."""

    displacement = candidate - reference_state
    flat = displacement.reshape(displacement.shape[0], -1)
    d_norm = flat.norm(dim=1)
    remaining = torch.clamp(torch.full_like(d_norm, float(total_budget)) - used_transport_cost, min=0.0)
    max_norm = torch.sqrt(torch.clamp(2.0 * remaining, min=0.0))
    scale = torch.ones_like(d_norm)
    overspent = d_norm > max_norm
    scale[overspent] = max_norm[overspent] / d_norm[overspent].clamp_min(1e-12)
    projected = reference_state + displacement * scale.view(-1, *([1] * (displacement.ndim - 1)))
    return projected


def _project_to_step_budget(
    candidate: torch.Tensor,
    reference_state: torch.Tensor,
    step_budget: float,
    exact: bool,
) -> torch.Tensor:
    """Project candidate state to step budget rho/T (clip or exact mode)."""

    displacement = candidate - reference_state
    flat = displacement.reshape(displacement.shape[0], -1)
    d_norm = flat.norm(dim=1)
    target_norm = torch.full_like(d_norm, math.sqrt(max(float(step_budget), 0.0) * 2.0))
    scale = torch.ones_like(d_norm)
    positive = d_norm > 1e-12
    if exact:
        scale[positive] = target_norm[positive] / d_norm[positive]
        scale[~positive] = 0.0
    else:
        overspent = d_norm > target_norm
        scale[overspent] = target_norm[overspent] / d_norm[overspent]
    projected = reference_state + displacement * scale.view(-1, *([1] * (displacement.ndim - 1)))
    return projected


def _as_sigma_batch(value: torch.Tensor, batch_size: int, x_ref: torch.Tensor) -> torch.Tensor:
    """Expand scalar sigma to `[B]` in the same device/dtype as inputs."""

    return torch.full((batch_size,), float(value.item()), device=x_ref.device, dtype=x_ref.dtype)


def rollout_path_heuristic_attack(
    *,
    cfg,
    x0: torch.Tensor,
    target_indices: torch.Tensor,
    attack_net,
    sigma_levels: torch.Tensor,
    inner_steps: int,
    step_size: float,
    gamma: float,
    total_budget: Optional[float],
    projection_mode: str,
    control_radius_kappa: float,
    kappa_by_step: Optional[torch.Tensor] = None,
    eps_schedule: Optional[torch.Tensor] = None,
) -> RolloutResult:
    """Legacy CDRO-v1.1 path-heuristic attack (per-step local control ascent)."""

    batch_size = x0.shape[0]
    n_steps = int(sigma_levels.numel() - 1)
    mode = str(projection_mode).lower()
    if n_steps <= 0:
        raise ValueError(f"sigma_levels must contain at least 2 values, got {sigma_levels.numel()}")
    if inner_steps < 0:
        raise ValueError(f"inner_steps must be >= 0, got {inner_steps}")
    if step_size <= 0:
        raise ValueError(f"step_size must be > 0, got {step_size}")
    if kappa_by_step is not None:
        if kappa_by_step.numel() != n_steps:
            raise ValueError(f"kappa_by_step must have {n_steps} elements, got {kappa_by_step.numel()}")
        kappa_by_step = kappa_by_step.to(device=x0.device, dtype=x0.dtype)

    if mode in ("global_remaining", "step_clip", "step_exact") and total_budget is None:
        raise ValueError(
            "total_budget must be set for projection_mode in "
            "('global_remaining', 'step_clip', 'step_exact')."
        )
    amp_dtype = resolve_amp_dtype(x0.device, getattr(cfg, "amp_dtype", "auto"))

    x_ref = x0.detach()
    prev_adv = x0.detach()
    path_ref = [x_ref]
    path_ctrl = [prev_adv]
    path_delta = []
    used_transport_cost = torch.zeros(batch_size, device=x0.device, dtype=x0.dtype)
    per_step_budget = None
    if total_budget is not None:
        per_step_budget = float(total_budget) / float(max(n_steps, 1))

    was_training = attack_net.training
    attack_net.eval()

    for k in range(n_steps):
        sigma_k = sigma_levels[k]
        sigma_next = sigma_levels[k + 1]
        delta_sigma = torch.sqrt((sigma_next.square() - sigma_k.square()).clamp_min(1e-8))
        if eps_schedule is None:
            eps = torch.randn_like(x0)
        else:
            eps = eps_schedule[k]

        base_increment = delta_sigma * eps
        reference_state = (x_ref + base_increment).detach()
        x_nominal_next = (prev_adv + base_increment).detach()
        sigma_batch = _as_sigma_batch(sigma_next, batch_size, x0)
        control = torch.zeros_like(prev_adv)

        for _ in range(int(inner_steps)):
            control.requires_grad_(True)
            candidate = x_nominal_next + control
            with autocast_context(x0.device, amp_dtype):
                step_loss = compute_training_loss(cfg, attack_net, candidate, x0, sigma_batch)
                displacement = candidate - reference_state
                transport = 0.5 * displacement.reshape(batch_size, -1).pow(2).sum(dim=1).mean()
                objective = step_loss - float(gamma) * transport
            grad = torch.autograd.grad(objective, control)[0]
            control = (control + float(step_size) * grad).detach()

            candidate = (x_nominal_next + control).detach()
            if mode == "global_remaining":
                candidate = _project_to_remaining_budget(
                    candidate=candidate,
                    reference_state=reference_state,
                    used_transport_cost=used_transport_cost,
                    total_budget=float(total_budget),
                ).detach()
                control = (candidate - x_nominal_next).detach()
            elif mode == "step_clip":
                candidate = _project_to_step_budget(
                    candidate=candidate,
                    reference_state=reference_state,
                    step_budget=float(per_step_budget),
                    exact=False,
                ).detach()
                control = (candidate - x_nominal_next).detach()
            elif mode == "step_exact":
                candidate = _project_to_step_budget(
                    candidate=candidate,
                    reference_state=reference_state,
                    step_budget=float(per_step_budget),
                    exact=True,
                ).detach()
                control = (candidate - x_nominal_next).detach()
            elif mode == "kappa_clip":
                step_kappa = float(control_radius_kappa) if kappa_by_step is None else float(kappa_by_step[k].item())
                radius = torch.full((batch_size,), step_kappa * float(delta_sigma), device=x0.device, dtype=x0.dtype)
                control = project_l2_ball(control, radius).detach()
            elif mode == "none":
                pass
            else:
                raise ValueError(
                    f"Unknown projection_mode='{projection_mode}'. "
                    "Expected one of: global_remaining, step_clip, step_exact, kappa_clip, none."
                )

        candidate_final = (x_nominal_next + control).detach()
        if inner_steps == 0:
            if mode == "global_remaining":
                candidate_final = _project_to_remaining_budget(
                    candidate=candidate_final,
                    reference_state=reference_state,
                    used_transport_cost=used_transport_cost,
                    total_budget=float(total_budget),
                ).detach()
            elif mode == "step_clip":
                candidate_final = _project_to_step_budget(
                    candidate=candidate_final,
                    reference_state=reference_state,
                    step_budget=float(per_step_budget),
                    exact=False,
                ).detach()
            elif mode == "step_exact":
                candidate_final = _project_to_step_budget(
                    candidate=candidate_final,
                    reference_state=reference_state,
                    step_budget=float(per_step_budget),
                    exact=True,
                ).detach()
            elif mode == "kappa_clip":
                step_kappa = float(control_radius_kappa) if kappa_by_step is None else float(kappa_by_step[k].item())
                radius = torch.full((batch_size,), step_kappa * float(delta_sigma), device=x0.device, dtype=x0.dtype)
                control = project_l2_ball(control, radius).detach()
                candidate_final = (x_nominal_next + control).detach()
            elif mode == "none":
                pass
            else:
                raise ValueError(
                    f"Unknown projection_mode='{projection_mode}'. "
                    "Expected one of: global_remaining, step_clip, step_exact, kappa_clip, none."
                )

        if mode == "global_remaining":
            step_cost = 0.5 * (candidate_final - reference_state).reshape(batch_size, -1).pow(2).sum(dim=1)
            used_transport_cost = (used_transport_cost + step_cost).detach()

        delta_effective = (candidate_final - x_nominal_next).detach()
        x_ref = reference_state
        prev_adv = candidate_final
        path_ref.append(x_ref)
        path_ctrl.append(prev_adv)
        path_delta.append(delta_effective)

    if was_training:
        attack_net.train()

    states_ctrl = torch.stack(path_ctrl, dim=1)
    states_ref = torch.stack(path_ref, dim=1)
    delta_path = torch.stack(path_delta, dim=1)
    x_target = states_ctrl[torch.arange(batch_size, device=x0.device), target_indices]
    sigma_target = sigma_levels[target_indices]
    return RolloutResult(
        x_target=x_target,
        sigma_target=sigma_target,
        states_ref=states_ref,
        states_ctrl=states_ctrl,
        delta_path=delta_path,
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
    projection_mode: str = "kappa_clip",
) -> RolloutResult:
    """Roll out a reference and controlled VE process with configurable projection."""

    batch_size = x0.shape[0]
    n_steps = sigma_levels.numel() - 1
    mode = str(projection_mode).lower()

    if kappa_by_step is not None:
        if kappa_by_step.numel() != n_steps:
            raise ValueError(f"kappa_by_step must have {n_steps} elements, got {kappa_by_step.numel()}")
        kappa_by_step = kappa_by_step.to(device=x0.device, dtype=x0.dtype)

    x_ref = x0
    x_ctrl = x0
    path_ref = [x_ref]
    path_ctrl = [x_ctrl]
    path_delta = []

    used_transport_cost = torch.zeros(batch_size, device=x0.device, dtype=x0.dtype)
    per_step_budget = None
    if total_budget is not None:
        per_step_budget = float(total_budget) / float(max(n_steps, 1))

    for k in range(n_steps):
        sigma_k = sigma_levels[k]
        sigma_next = sigma_levels[k + 1]
        delta_sigma = torch.sqrt((sigma_next.square() - sigma_k.square()).clamp_min(1e-8))
        if eps_schedule is None:
            eps = torch.randn_like(x0)
        else:
            eps = eps_schedule[k]

        x_ref_next = x_ref + delta_sigma * eps
        gap = x_ctrl - x_ref
        sigma_batch = torch.full((batch_size,), sigma_k, device=x0.device, dtype=x0.dtype)

        if grad_through_control:
            delta_raw = control_net(x_ref, gap, sigma_batch)
        else:
            with torch.no_grad():
                delta_raw = control_net(x_ref, gap, sigma_batch)

        x_nominal_next = x_ctrl + delta_sigma * eps

        if mode == "kappa_clip":
            step_kappa = float(control_radius_kappa) if kappa_by_step is None else float(kappa_by_step[k].item())
            radius = torch.full((batch_size,), step_kappa * float(delta_sigma), device=x0.device, dtype=x0.dtype)
            delta_step = project_l2_ball(delta_raw, radius)
            x_ctrl_next = x_nominal_next + delta_step
        else:
            candidate = x_nominal_next + delta_raw
            if mode == "global_remaining":
                if total_budget is None:
                    raise ValueError("total_budget must be set for projection_mode='global_remaining'")
                x_ctrl_next = _project_to_remaining_budget(
                    candidate=candidate,
                    reference_state=x_ref_next,
                    used_transport_cost=used_transport_cost,
                    total_budget=float(total_budget),
                )
            elif mode == "step_clip":
                if per_step_budget is None:
                    raise ValueError("total_budget must be set for projection_mode='step_clip'")
                x_ctrl_next = _project_to_step_budget(
                    candidate=candidate,
                    reference_state=x_ref_next,
                    step_budget=per_step_budget,
                    exact=False,
                )
            elif mode == "step_exact":
                if per_step_budget is None:
                    raise ValueError("total_budget must be set for projection_mode='step_exact'")
                x_ctrl_next = _project_to_step_budget(
                    candidate=candidate,
                    reference_state=x_ref_next,
                    step_budget=per_step_budget,
                    exact=True,
                )
            elif mode == "none":
                x_ctrl_next = candidate
            else:
                raise ValueError(
                    f"Unknown projection_mode='{projection_mode}'. "
                    "Expected one of: global_remaining, step_clip, step_exact, kappa_clip, none."
                )

            if mode == "global_remaining":
                displacement = x_ctrl_next - x_ref_next
                step_cost = 0.5 * displacement.reshape(batch_size, -1).pow(2).sum(dim=1)
                used_transport_cost = (used_transport_cost + step_cost).detach()

        delta_effective = x_ctrl_next - x_nominal_next
        x_ref = x_ref_next
        x_ctrl = x_ctrl_next

        path_ref.append(x_ref)
        path_ctrl.append(x_ctrl)
        path_delta.append(delta_effective)

    states_ctrl = torch.stack(path_ctrl, dim=1)
    states_ref = torch.stack(path_ref, dim=1)
    delta_path = torch.stack(path_delta, dim=1)
    x_target = states_ctrl[torch.arange(batch_size, device=x0.device), target_indices]
    sigma_target = sigma_levels[target_indices]

    return RolloutResult(
        x_target=x_target,
        sigma_target=sigma_target,
        states_ref=states_ref,
        states_ctrl=states_ctrl,
        delta_path=delta_path,
    )
