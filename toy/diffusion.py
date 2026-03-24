import math
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class RolloutResult:
    """Container for one controlled forward rollout over the VE ladder.

    `x_target` and `sigma_target` are aligned by sampled target index i:
      x_target[b] = states_ctrl[b, i_b], sigma_target[b] = sigma_levels[i_b].
    """

    x_target: torch.Tensor
    sigma_target: torch.Tensor
    states_ref: Optional[torch.Tensor] = None
    states_ctrl: Optional[torch.Tensor] = None
    delta_path: Optional[torch.Tensor] = None


def build_sigma_levels(sigma_min: float, sigma_max: float, n_steps: int, device: torch.device) -> torch.Tensor:
    """Build VE noise ladder [sigma_0, ..., sigma_N] with sigma_0 = 0.

    For k=0..N-1, forward increments use:
      Delta_sigma_k = sqrt(sigma_{k+1}^2 - sigma_k^2),
      x_{k+1} = x_k + Delta_sigma_k * eps_k.
    """

    return torch.cat(
        [
            torch.zeros(1, device=device),
            torch.logspace(math.log10(sigma_min), math.log10(sigma_max), n_steps, device=device),
        ]
    )


def sample_target_indices(batch_size: int, sigma_levels: torch.Tensor) -> torch.Tensor:
    """Uniformly sample training target steps i in {1, ..., N}."""

    return torch.randint(1, sigma_levels.numel(), (batch_size,), device=sigma_levels.device)


def sample_target_indices_log_normal(
    batch_size: int,
    sigma_levels: torch.Tensor,
    p_mean: float = -1.2,
    p_std: float = 1.2,
) -> torch.Tensor:
    """EDM-style sigma sampling with discretization to the nearest ladder step.

    Continuous draw:
      sigma ~ LogNormal(p_mean, p_std).
    Then map to nearest sigma_levels[1:] index for discrete-path training.
    """

    rnd = torch.randn(batch_size, device=sigma_levels.device)
    sigma = torch.exp(rnd * p_std + p_mean).clamp(min=float(sigma_levels[1]), max=float(sigma_levels[-1]))
    d = (sigma.unsqueeze(1) - sigma_levels[1:].unsqueeze(0)).abs()
    nearest = torch.argmin(d, dim=1)
    return nearest + 1


def project_l2_ball(delta_raw: torch.Tensor, radius: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Project each sample in `delta_raw` onto an L2 ball with per-sample radius.

    Implements:
      Pi_{||v||_2 <= r}(u) = min(1, r / ||u||_2) * u
    """

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
    """Return per-transition control radius multipliers kappa_k for k->k+1.

    If time-dependent mode is enabled, kappa is piecewise-scaled over
    low/mid/high noise regions. With `preserve_l2_budget=True`, rescale
    multipliers so that sum_k kappa_k^2 matches the flat-kappa baseline.
    """

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
        # Keep sum_k kappa_k^2 constant versus the uniform-kappa baseline.
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
    kappa_by_step: Optional[torch.Tensor] = None,
    eps_schedule: Optional[torch.Tensor] = None,
) -> RolloutResult:
    """Roll out a reference and a controlled VE forward process using shared noise.

    Reference path:
      x_ref_{k+1} = x_ref_k + Delta_sigma_k * eps_k
    Controlled path:
      x_ctrl_{k+1} = x_ctrl_k + Delta_sigma_k * eps_k + delta_k

    where:
      delta_k = Pi_{||.|| <= kappa_k * Delta_sigma_k}(
                   control_net(x_ref_k, x_ctrl_k - x_ref_k, sigma_k)
               )

    The projection enforces the hard per-step innovation-kernel constraint.
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

    for k in range(n_steps):
        sigma_k = sigma_levels[k]
        sigma_next = sigma_levels[k + 1]
        delta_sigma = torch.sqrt((sigma_next.square() - sigma_k.square()).clamp_min(1e-8))
        if eps_schedule is None:
            eps = torch.randn_like(x0)
        else:
            eps = eps_schedule[k]

        # Reference forward update.
        x_ref_next = x_ref + delta_sigma * eps
        gap = x_ctrl - x_ref
        sigma_batch = torch.full((batch_size,), sigma_k, device=x0.device, dtype=x0.dtype)

        if grad_through_control:
            delta_raw = control_net(x_ref, gap, sigma_batch)
        else:
            with torch.no_grad():
                delta_raw = control_net(x_ref, gap, sigma_batch)

        # Hard per-step innovation-kernel ball: ||delta_k|| <= kappa * delta_sigma_k
        step_kappa = float(control_radius_kappa) if kappa_by_step is None else float(kappa_by_step[k].item())
        radius = torch.full((batch_size,), step_kappa * float(delta_sigma), device=x0.device, dtype=x0.dtype)
        delta = project_l2_ball(delta_raw, radius)
        x_ctrl_next = x_ctrl + delta_sigma * eps + delta

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
