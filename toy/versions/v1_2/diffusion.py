from typing import Optional

import torch

from ...shared.objective import compute_training_loss
from ..v1_1.diffusion import RolloutResult, build_kappa_schedule, rollout_controlled_ve


def _as_sigma_batch(value: torch.Tensor, batch_size: int, x_ref: torch.Tensor) -> torch.Tensor:
    """Expand scalar sigma to `[B]` in the same device/dtype as inputs."""

    return torch.full((batch_size,), float(value.item()), device=x_ref.device, dtype=x_ref.dtype)


def _sigma_gate(
    sigma: torch.Tensor,
    *,
    activation_scale: float,
    sigma_floor: float,
    sigma_cut: float,
    gate_power: float,
) -> torch.Tensor:
    """CDRO-style sigma gating for robust perturbations."""

    sigma_cut_safe = max(float(sigma_cut), 1e-8)
    gate = (1.0 - sigma / sigma_cut_safe).clamp(min=0.0, max=1.0)
    sigma_floor_safe = max(float(sigma_floor), 0.0)
    if sigma_floor_safe > 0.0:
        if sigma_floor_safe >= sigma_cut_safe:
            raise ValueError(
                f"sigma_floor ({sigma_floor_safe}) must be smaller than sigma_cut ({sigma_cut_safe})"
            )
        ramp = ((sigma - sigma_floor_safe) / (sigma_cut_safe - sigma_floor_safe)).clamp(min=0.0, max=1.0)
        gate = gate * ramp
    gate = gate * float(max(activation_scale, 0.0))
    return gate.pow(float(gate_power))


def _delta_scale(sigma: torch.Tensor, gate: torch.Tensor, delta_space: str) -> torch.Tensor:
    """Scale raw perturbation according to chosen parameterization."""

    if delta_space == "noise":
        return gate * sigma
    if delta_space == "image":
        return gate
    raise ValueError(f"Unsupported delta_space='{delta_space}'. Expected one of: image, noise.")


def _project_l2_rms_ball(delta_raw: torch.Tensor, radius: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Project each sample onto a per-sample RMS-L2 ball."""

    flat = delta_raw.reshape(delta_raw.shape[0], -1)
    rms = (flat.square().mean(dim=1, keepdim=True) + eps).sqrt()
    if radius.ndim == 1:
        radius = radius[:, None]
    scale = torch.minimum(torch.ones_like(rms), radius / rms)
    return (flat * scale).reshape_as(delta_raw)


def _transport_cost(
    *,
    applied_delta: torch.Tensor,
    sigma: torch.Tensor,
    sigma_data: float,
    delta_space: str,
) -> torch.Tensor:
    """Compute per-sample transport cost used by the dual update."""

    if delta_space == "noise":
        return applied_delta.square().reshape(applied_delta.shape[0], -1).mean(dim=1)
    denom = sigma.square() + float(sigma_data) ** 2
    num = applied_delta.square().reshape(applied_delta.shape[0], -1)
    return (num / denom.view(-1, 1).clamp_min(1e-8)).mean(dim=1)


def rollout_path_heuristic_attack(
    *,
    cfg,
    x0: torch.Tensor,
    target_indices: torch.Tensor,
    attack_net,
    sigma_levels: torch.Tensor,
    inner_steps: int,
    step_size: float,
    lambda_dual: float,
    activation_scale: float,
    max_delta: float,
    sigma_floor: float,
    sigma_cut: float,
    gate_power: float,
    delta_space: str,
    control_radius_kappa: float,
    kappa_by_step: Optional[torch.Tensor] = None,
    eps_schedule: Optional[torch.Tensor] = None,
) -> RolloutResult:
    """v1.2 path-heuristic attack with adaptive-dual objective and sigma gating."""

    del control_radius_kappa, kappa_by_step  # kept for API compatibility with other versions

    batch_size = x0.shape[0]
    n_steps = int(sigma_levels.numel() - 1)
    if n_steps <= 0:
        raise ValueError(f"sigma_levels must contain at least 2 values, got {sigma_levels.numel()}")
    if inner_steps < 0:
        raise ValueError(f"inner_steps must be >= 0, got {inner_steps}")
    if step_size <= 0:
        raise ValueError(f"step_size must be > 0, got {step_size}")
    if max_delta <= 0:
        raise ValueError(f"max_delta must be > 0, got {max_delta}")

    delta_space = str(delta_space).lower()
    if delta_space not in ("image", "noise"):
        raise ValueError(f"Unsupported delta_space='{delta_space}'. Expected one of: image, noise.")

    x_ref = x0.detach()
    prev_adv = x0.detach()
    path_ref = [x_ref]
    path_ctrl = [prev_adv]
    path_delta = []

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

        gate = _sigma_gate(
            sigma_batch,
            activation_scale=activation_scale,
            sigma_floor=sigma_floor,
            sigma_cut=sigma_cut,
            gate_power=gate_power,
        )
        scale = _delta_scale(sigma_batch, gate, delta_space)
        scale_view = scale.view(batch_size, *([1] * (x0.ndim - 1)))

        delta_raw = torch.zeros_like(prev_adv)
        for _ in range(int(inner_steps)):
            delta_raw.requires_grad_(True)
            applied_delta = scale_view * delta_raw
            candidate = x_nominal_next + applied_delta
            step_loss = compute_training_loss(cfg, attack_net, candidate, x0, sigma_batch)
            transport = _transport_cost(
                applied_delta=applied_delta,
                sigma=sigma_batch,
                sigma_data=float(cfg.sigma_data),
                delta_space=delta_space,
            ).mean()
            objective = step_loss - float(lambda_dual) * transport
            grad = torch.autograd.grad(objective, delta_raw)[0]

            grad_flat = grad.reshape(batch_size, -1)
            grad_norm = grad_flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
            step = float(step_size) * (grad_flat / grad_norm).reshape_as(delta_raw)
            delta_raw = (delta_raw + step).detach()

            if delta_space == "noise":
                radius = torch.full((batch_size,), float(max_delta), device=x0.device, dtype=x0.dtype)
                radius = radius / scale.clamp_min(1e-8)
            else:
                radius = torch.full((batch_size,), float(max_delta), device=x0.device, dtype=x0.dtype)
            delta_raw = _project_l2_rms_ball(delta_raw, radius).detach()

        delta_effective = scale_view * delta_raw
        candidate_final = (x_nominal_next + delta_effective).detach()
        x_ref = reference_state
        prev_adv = candidate_final
        path_ref.append(x_ref)
        path_ctrl.append(prev_adv)
        path_delta.append(delta_effective.detach())

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

