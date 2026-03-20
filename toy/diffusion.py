import math
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class RolloutResult:
    x_target: torch.Tensor
    sigma_target: torch.Tensor
    energy: torch.Tensor
    states_ref: Optional[torch.Tensor] = None
    states_ctrl: Optional[torch.Tensor] = None


def build_sigma_levels(sigma_min: float, sigma_max: float, n_steps: int, device: torch.device) -> torch.Tensor:
    return torch.cat(
        [
            torch.zeros(1, device=device),
            torch.logspace(math.log10(sigma_min), math.log10(sigma_max), n_steps, device=device),
        ]
    )


def sample_target_indices(batch_size: int, sigma_levels: torch.Tensor) -> torch.Tensor:
    # Valid corruption levels: 1..N where sigma_levels[0] = 0.
    return torch.randint(1, sigma_levels.numel(), (batch_size,), device=sigma_levels.device)


def sample_target_indices_log_normal(
    batch_size: int,
    sigma_levels: torch.Tensor,
    p_mean: float = -1.2,
    p_std: float = 1.2,
) -> torch.Tensor:
    # EDM-style sigma ~ LogNormal(p_mean, p_std), then snap to nearest discrete ladder level.
    rnd = torch.randn(batch_size, device=sigma_levels.device)
    sigma = torch.exp(rnd * p_std + p_mean).clamp(min=float(sigma_levels[1]), max=float(sigma_levels[-1]))
    d = (sigma.unsqueeze(1) - sigma_levels[1:].unsqueeze(0)).abs()
    nearest = torch.argmin(d, dim=1)
    return nearest + 1


def rollout_controlled_ve(
    x0: torch.Tensor,
    target_indices: torch.Tensor,
    control_net,
    sigma_levels: torch.Tensor,
    grad_through_control: bool,
    eps_schedule: Optional[torch.Tensor] = None,
) -> RolloutResult:
    batch_size = x0.shape[0]
    n_steps = sigma_levels.numel() - 1
    dt = 1.0 / float(n_steps)
    sqrt_dt = math.sqrt(dt)

    x_ref = x0
    x_ctrl = x0
    energy = torch.zeros(batch_size, device=x0.device)

    path_ref = [x_ref]
    path_ctrl = [x_ctrl]

    for k in range(n_steps):
        sigma_k = sigma_levels[k]
        sigma_next = sigma_levels[k + 1]
        delta_sigma = torch.sqrt((sigma_next.square() - sigma_k.square()).clamp_min(1e-8))
        if eps_schedule is None:
            eps = torch.randn_like(x0)
        else:
            eps = eps_schedule[k]

        # Base VE-like corruption increment.
        x_ref = x_ref + delta_sigma * eps
        t_value = k / float(max(n_steps - 1, 1))
        t = torch.full((batch_size,), t_value, device=x0.device)

        if grad_through_control:
            v = control_net(x_ctrl, t)
        else:
            with torch.no_grad():
                v = control_net(x_ctrl, t)

        # Control enters through diffusion channel:
        # dX = g(t)(dW + v dt) -> in discrete form, control increment scales as sqrt(dt).
        control_increment = delta_sigma * sqrt_dt * v
        x_ctrl = x_ctrl + delta_sigma * eps + control_increment
        energy = energy + v.pow(2).sum(dim=1) * dt

        path_ref.append(x_ref)
        path_ctrl.append(x_ctrl)

    states_ctrl = torch.stack(path_ctrl, dim=1)
    states_ref = torch.stack(path_ref, dim=1)
    x_target = states_ctrl[torch.arange(batch_size, device=x0.device), target_indices]

    sigma_target = sigma_levels[target_indices]
    return RolloutResult(
        x_target=x_target,
        sigma_target=sigma_target,
        energy=energy.mean(),
        states_ref=states_ref,
        states_ctrl=states_ctrl,
    )
