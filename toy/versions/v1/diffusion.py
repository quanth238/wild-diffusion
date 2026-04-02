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
    """Roll out a soft-penalized VE forward process with shared reference noise.

    v1 uses no hard projection. The control network output is interpreted as a
    per-step drift coefficient u_k and converted to a state increment by:
      delta_k = u_k * Delta_sigma_k.
    """

    del control_radius_kappa, kappa_by_step  # kept in signature for API compatibility

    batch_size = x0.shape[0]
    n_steps = sigma_levels.numel() - 1

    x_ref = x0
    x_ctrl = x0

    path_ref = [x_ref]
    path_ctrl = [x_ctrl]
    path_delta = []

    if eps_schedule is not None and eps_schedule.shape[0] < n_steps:
        raise ValueError(f"eps_schedule first dim must be >= {n_steps}, got {eps_schedule.shape[0]}")

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
            drift = control_net(x_ref, gap, sigma_batch)
        else:
            with torch.no_grad():
                drift = control_net(x_ref, gap, sigma_batch)

        delta = delta_sigma * drift
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
