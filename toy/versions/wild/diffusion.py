from typing import Optional

import torch

from ..v2.diffusion import RolloutResult, build_kappa_schedule


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
    """API-compatible rollout for WILD version.

    WILD robust training in this toy backend acts on sample-space WDRO surrogates,
    not on path-level control deltas. For compatibility with existing diagnostics,
    we expose a reference rollout where controlled and reference trajectories are
    identical and per-step deltas are all zeros.
    """

    del control_net, grad_through_control, control_radius_kappa, kappa_by_step

    batch_size = x0.shape[0]
    n_steps = int(sigma_levels.numel() - 1)
    if n_steps <= 0:
        raise ValueError(f"sigma_levels must contain at least 2 values, got {sigma_levels.numel()}")

    x_ref = x0
    path_ref = [x_ref]
    path_ctrl = [x_ref]
    path_delta = []

    if eps_schedule is not None and eps_schedule.shape[0] < n_steps:
        raise ValueError(f"eps_schedule first dim must be >= {n_steps}, got {eps_schedule.shape[0]}")

    for k in range(n_steps):
        sigma_k = sigma_levels[k]
        sigma_next = sigma_levels[k + 1]
        delta_sigma = torch.sqrt((sigma_next.square() - sigma_k.square()).clamp_min(1e-8))
        eps = torch.randn_like(x0) if eps_schedule is None else eps_schedule[k]
        x_ref = x_ref + delta_sigma * eps
        path_ref.append(x_ref)
        path_ctrl.append(x_ref)
        path_delta.append(torch.zeros_like(x_ref))

    states_ref = torch.stack(path_ref, dim=1)
    states_ctrl = torch.stack(path_ctrl, dim=1)
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
