from __future__ import annotations

import torch

from toy_2d.sde_models import ReverseGaussianModel
from toy_2d.vp_sde import VPSchedule


def compute_reverse_teacher_forcing_loss(
    *,
    model: ReverseGaussianModel,
    paths: torch.Tensor,
    schedule: VPSchedule,
) -> tuple[torch.Tensor, dict[str, float]]:
    if paths.ndim != 3:
        raise ValueError(f"paths must have shape [batch, steps + 1, data_dim], got {tuple(paths.shape)}")
    if paths.shape[1] != schedule.num_steps + 1:
        raise ValueError(
            f"paths has {paths.shape[1]} states but schedule expects {schedule.num_steps + 1}"
        )

    terminal_states = paths[:, -1, :]
    terminal_nll = model.terminal_nll(terminal_states)
    hidden_state = model.init_reverse_hidden(terminal_states)
    step_losses: list[torch.Tensor] = []

    for step_idx in range(schedule.num_steps - 1, -1, -1):
        next_states = paths[:, step_idx + 1, :]
        current_states = paths[:, step_idx, :]
        mean = model.reverse_mean(
            next_states=next_states,
            time_value=schedule.times[step_idx : step_idx + 1],
            hidden_state=hidden_state,
        )
        sigma_sq = schedule.reverse_noise_scales[step_idx].square().clamp_min(1e-8)
        step_loss = (current_states - mean).square().sum(dim=1) / (2.0 * sigma_sq)
        step_losses.append(step_loss)
        hidden_state = model.update_hidden(
            current_states=current_states,
            time_value=schedule.times[step_idx : step_idx + 1],
            hidden_state=hidden_state,
        )

    if step_losses:
        step_loss_tensor = torch.stack(step_losses, dim=1).sum(dim=1)
    else:
        step_loss_tensor = torch.zeros_like(terminal_nll)
    total_loss = terminal_nll + step_loss_tensor
    metrics = {
        "terminal_nll": float(terminal_nll.mean().item()),
        "transition_nll": float(step_loss_tensor.mean().item()),
        "reverse_nll": float(total_loss.mean().item()),
    }
    return total_loss, metrics


@torch.no_grad()
def sample_reverse_paths(
    *,
    model: ReverseGaussianModel,
    schedule: VPSchedule,
    num_samples: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    data_dim = model.data_dim
    paths = torch.empty(num_samples, schedule.num_steps + 1, data_dim, device=device, dtype=dtype)
    terminal_states = model.sample_terminal(num_samples=num_samples, device=device, dtype=dtype)
    paths[:, -1, :] = terminal_states
    hidden_state = model.init_reverse_hidden(terminal_states)

    for step_idx in range(schedule.num_steps - 1, -1, -1):
        mean = model.reverse_mean(
            next_states=paths[:, step_idx + 1, :],
            time_value=schedule.times[step_idx : step_idx + 1],
            hidden_state=hidden_state,
        )
        noise = torch.randn(num_samples, data_dim, device=device, dtype=dtype)
        current_states = mean + schedule.reverse_noise_scales[step_idx] * noise
        paths[:, step_idx, :] = current_states
        hidden_state = model.update_hidden(
            current_states=current_states,
            time_value=schedule.times[step_idx : step_idx + 1],
            hidden_state=hidden_state,
        )

    return paths
