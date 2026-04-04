import math
from dataclasses import dataclass
from typing import Optional

import torch

from ...shared.objective import compute_training_loss
from ...shared.runtime import autocast_context, resolve_amp_dtype
from ..v1_1.diffusion import build_kappa_schedule


@dataclass
class RolloutResult:
    """Container for one Route-A CDRO rollout over the VE ladder."""

    x_target: torch.Tensor
    sigma_target: torch.Tensor
    states_ref: Optional[torch.Tensor] = None
    states_ctrl: Optional[torch.Tensor] = None
    delta_path: Optional[torch.Tensor] = None
    beta_path: Optional[torch.Tensor] = None


def project_l2_ball(delta_raw: torch.Tensor, radius: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Project each sample in `delta_raw` onto an L2 ball with per-sample radius."""

    flat = delta_raw.reshape(delta_raw.shape[0], -1)
    norm = flat.norm(dim=1, keepdim=True).clamp_min(eps)
    if radius.ndim == 1:
        radius = radius[:, None]
    scale = torch.minimum(torch.ones_like(norm), radius / norm)
    return (flat * scale).reshape_as(delta_raw)


def build_time_deltas(
    sigma_levels: torch.Tensor,
    time_horizon: float,
) -> torch.Tensor:
    """Build a uniform physical-time grid used by the Route-A beta parameterization."""

    n_steps = int(sigma_levels.numel() - 1)
    if n_steps <= 0:
        raise ValueError(f"sigma_levels must contain at least 2 values, got {sigma_levels.numel()}")
    if time_horizon <= 0:
        raise ValueError(f"time_horizon must be > 0, got {time_horizon}")
    return torch.full(
        (n_steps,),
        float(time_horizon) / float(n_steps),
        device=sigma_levels.device,
        dtype=sigma_levels.dtype,
    )


def build_constraint_radii(
    sigma_levels: torch.Tensor,
    total_budget: float,
    time_horizon: float,
) -> torch.Tensor:
    """Exact Route-A uniform local cap expressed in state-increment (`delta`) space."""

    n_steps = int(sigma_levels.numel() - 1)
    if n_steps <= 0:
        raise ValueError(f"sigma_levels must contain at least 2 values, got {sigma_levels.numel()}")
    if total_budget < 0:
        raise ValueError(f"total_budget must be >= 0, got {total_budget}")

    dt = build_time_deltas(sigma_levels, time_horizon)
    weight = float(total_budget) / float(n_steps)
    radius_sq = dt * weight
    return torch.sqrt(radius_sq.clamp_min(0.0))


def _beta_radius(
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    n_steps: int,
    total_budget: float,
) -> torch.Tensor:
    """Uniform local cap radius in beta-space: ||beta_k||^2 <= rho / K."""

    return torch.full(
        (batch_size,),
        math.sqrt(max(float(total_budget), 0.0) / float(max(n_steps, 1))),
        device=device,
        dtype=dtype,
    )


def _as_sigma_batch(value: torch.Tensor, batch_size: int, x_ref: torch.Tensor) -> torch.Tensor:
    """Expand scalar sigma to `[B]` in the same device/dtype as inputs."""

    return torch.full((batch_size,), float(value.item()), device=x_ref.device, dtype=x_ref.dtype)


def _l2_normalize_per_sample(value: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Return per-sample L2-normalized directions with zero fallback for zero gradients."""

    flat = value.reshape(value.shape[0], -1)
    norm = flat.norm(p=2, dim=1, keepdim=True).clamp_min(eps)
    return (flat / norm).reshape_as(value)


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
    """Greedy Route-A CDRO attack: beta-space local ascent with exact uniform local caps."""

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

    dt = build_time_deltas(sigma_levels, time_horizon)
    sqrt_dt = torch.sqrt(dt.clamp_min(1e-12))
    beta_radius = _beta_radius(
        batch_size=batch_size,
        device=x0.device,
        dtype=x0.dtype,
        n_steps=n_steps,
        total_budget=total_budget,
    )
    amp_dtype = resolve_amp_dtype(x0.device, getattr(cfg, "amp_dtype", "auto"))

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
        delta_sigma = torch.sqrt((sigma_next.square() - sigma_k.square()).clamp_min(1e-8))
        eps = torch.randn_like(x0) if eps_schedule is None else eps_schedule[k]

        base_increment = delta_sigma * eps
        reference_state = (x_ref + base_increment).detach()
        x_nominal_next = (x_ctrl + base_increment).detach()
        sigma_batch = _as_sigma_batch(sigma_next, batch_size, x0)

        beta = torch.zeros_like(x_ctrl)
        if int(inner_steps) == 1:
            # With one inner step, solve the linearized local-cap problem exactly:
            # max_{||beta|| <= sqrt(rho/K)} <grad, beta>.
            beta = beta.requires_grad_(True)
            candidate = x_nominal_next + float(sqrt_dt[k].item()) * beta
            with autocast_context(x0.device, amp_dtype):
                step_loss = compute_training_loss(cfg, attack_net, candidate, x0, sigma_batch)
            grad = torch.autograd.grad(step_loss, beta)[0]
            grad_unit = _l2_normalize_per_sample(grad)
            beta = (grad_unit.reshape(batch_size, -1) * beta_radius[:, None]).reshape_as(grad).detach()
        else:
            for _ in range(int(inner_steps)):
                beta.requires_grad_(True)
                candidate = x_nominal_next + float(sqrt_dt[k].item()) * beta
                with autocast_context(x0.device, amp_dtype):
                    step_loss = compute_training_loss(cfg, attack_net, candidate, x0, sigma_batch)
                grad = torch.autograd.grad(step_loss, beta)[0]

                grad_unit = _l2_normalize_per_sample(grad)
                step = float(step_size) * grad_unit
                beta = (beta + step).detach()
                beta = project_l2_ball(beta, beta_radius).detach()

        delta_effective = float(sqrt_dt[k].item()) * beta
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
    _ = build_time_deltas(sigma_levels, time_horizon)

    x_ref = x0
    path_ref = [x_ref]
    path_ctrl = [x_ref]
    path_delta = []
    path_beta = []

    for k in range(n_steps):
        sigma_k = sigma_levels[k]
        sigma_next = sigma_levels[k + 1]
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
