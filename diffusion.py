import math
from dataclasses import dataclass
from typing import Optional

import torch


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


def rollout_controlled_ve(
    x0: torch.Tensor,
    target_indices: torch.Tensor,
    control_net,
    sigma_levels: torch.Tensor,
    grad_through_control: bool,
    control_radius_kappa: float,
    non_markov_rho: float = 0.8,
    kappa_by_step: Optional[torch.Tensor] = None,
    eps_schedule: Optional[torch.Tensor] = None,
) -> RolloutResult:
    """Roll out a coupled forward process with non-Markovian ref increments.

    The update is parameterized by rho in [0,1]:
      z_{k+1} = rho * z_k + sqrt(1-rho^2) * eps_k
      x_{k+1} = x0 + sigma_{k+1} * z_{k+1}
    where z_k = (x_k - x0) / sigma_k when sigma_k > 0.
    """

    batch_size = x0.shape[0]
    n_steps = sigma_levels.numel() - 1

    if kappa_by_step is not None:
        if kappa_by_step.numel() != n_steps:
            raise ValueError(f"kappa_by_step must have {n_steps} elements, got {kappa_by_step.numel()}")
        kappa_by_step = kappa_by_step.to(device=x0.device, dtype=x0.dtype)

    x_ref = x0
    x_ctrl = x0

    path_ref = [x_ref]
    path_ctrl = [x_ctrl]
    path_delta = []

    rho = float(non_markov_rho)
    rho = max(0.0, min(1.0, rho))
    rho_stochastic_scale = math.sqrt(max(1.0 - rho * rho, 0.0))

    for k in range(n_steps):
        sigma_k = sigma_levels[k]
        sigma_next = sigma_levels[k + 1]
        delta_sigma = torch.sqrt((sigma_next.square() - sigma_k.square()).clamp_min(1e-8))
        if eps_schedule is None:
            eps = torch.randn_like(x0)
        else:
            eps = eps_schedule[k]

        # Non-Markovian increment with explicit dependence on x0 and previous x_ref.
        if float(sigma_k.item()) < 1e-7:
            step_increment = sigma_next * eps
        else:
            directed_term = (x_ref - x0) * (rho * (sigma_next / sigma_k) - 1.0)
            stochastic_term = sigma_next * rho_stochastic_scale * eps
            step_increment = directed_term + stochastic_term

        x_ref_next = x_ref + step_increment
        gap = x_ctrl - x_ref
        sigma_batch = torch.full((batch_size,), sigma_k, device=x0.device, dtype=x0.dtype)

        if grad_through_control:
            delta_raw = control_net(x_ref, gap, sigma_batch)
        else:
            with torch.no_grad():
                delta_raw = control_net(x_ref, gap, sigma_batch)

        step_kappa = float(control_radius_kappa) if kappa_by_step is None else float(kappa_by_step[k].item())
        radius = torch.full((batch_size,), step_kappa * float(delta_sigma), device=x0.device, dtype=x0.dtype)
        delta = project_l2_ball(delta_raw, radius)
        # Apply the same base increment to keep ref/ctrl aligned before adding control delta.
        x_ctrl_next = x_ctrl + step_increment + delta

        x_ref = x_ref_next
        x_ctrl = x_ctrl_next

        path_ref.append(x_ref)
        path_ctrl.append(x_ctrl)
        path_delta.append(delta)

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
