from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CausalBatchStats:
    mean_delta_norm: float
    max_delta_norm: float
    mean_transport_cost: float
    max_transport_cost: float
    mean_total_transport_cost: float
    max_total_transport_cost: float
    target_total_budget: float | None


def causal_wdro_loss(
    *,
    train_net,
    clean_points: torch.Tensor,
    p_mean: float,
    p_std: float,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    path_steps: int,
    inner_steps: int,
    step_size: float,
    gamma: float,
    total_budget: float | None = None,
    exact_budget_split: bool = False,
    shared_noise: bool = False,
    sigma_schedule: str = "karras_grid",
    attack_net=None,
) -> tuple[torch.Tensor, CausalBatchStats]:
    sigmas = sample_path_sigmas(
        batch_size=clean_points.shape[0],
        num_steps=path_steps,
        p_mean=p_mean,
        p_std=p_std,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        rho=rho,
        device=clean_points.device,
        dtype=clean_points.dtype,
        schedule=sigma_schedule,
    )
    reference_path = build_forward_path(clean_points=clean_points, sigmas=sigmas, shared_noise=shared_noise)
    final_path = solve_causal_path_attack(
        attack_net=attack_net if attack_net is not None else train_net,
        clean_points=clean_points,
        reference_path=reference_path,
        sigmas=sigmas,
        inner_steps=inner_steps,
        step_size=step_size,
        gamma=gamma,
        total_budget=total_budget,
        exact_budget_split=exact_budget_split,
    )
    outer_loss = _per_time_losses(net=train_net, clean_points=clean_points, adv_path=final_path, sigmas=sigmas).mean()
    displacement = (final_path - reference_path).detach()
    norms = displacement.norm(dim=2)
    per_step_transport_cost = 0.5 * displacement.square().sum(dim=2)
    total_transport_cost = per_step_transport_cost.sum(dim=1)
    stats = CausalBatchStats(
        mean_delta_norm=float(norms.mean().item()),
        max_delta_norm=float(norms.max().item()),
        mean_transport_cost=float(per_step_transport_cost.mean().item()),
        max_transport_cost=float(per_step_transport_cost.max().item()),
        mean_total_transport_cost=float(total_transport_cost.mean().item()),
        max_total_transport_cost=float(total_transport_cost.max().item()),
        target_total_budget=total_budget,
    )
    return outer_loss, stats


def sample_path_sigmas(
    *,
    batch_size: int,
    num_steps: int,
    p_mean: float,
    p_std: float,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    device: torch.device,
    dtype: torch.dtype,
    schedule: str,
) -> torch.Tensor:
    if num_steps < 1:
        raise ValueError("num_steps must be at least 1.")
    if schedule == "edm_random":
        random_sigmas = sample_edm_random_sigmas(
            batch_size=batch_size,
            num_steps=num_steps,
            p_mean=p_mean,
            p_std=p_std,
            device=device,
            dtype=dtype,
        )
        return torch.sort(random_sigmas, dim=1).values
    if schedule == "edm_quantiles":
        shared = build_edm_quantile_sigmas(
            num_steps=num_steps,
            p_mean=p_mean,
            p_std=p_std,
            device=device,
            dtype=dtype,
        )
        shared = torch.sort(shared).values
        return shared.view(1, num_steps).expand(batch_size, num_steps)
    if schedule != "karras_grid":
        raise ValueError(f"Unsupported sigma schedule: {schedule}")
    shared = build_forward_karras_sigmas(
        num_steps=num_steps,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        rho=rho,
        device=device,
        dtype=dtype,
    )
    return shared.view(1, num_steps).expand(batch_size, num_steps)


def sample_edm_random_sigmas(
    *,
    batch_size: int,
    num_steps: int,
    p_mean: float,
    p_std: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    rnd_normal = torch.randn(batch_size, num_steps, device=device, dtype=dtype)
    return torch.exp(rnd_normal * p_std + p_mean)


def build_edm_quantile_sigmas(
    *,
    num_steps: int,
    p_mean: float,
    p_std: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    quantiles = 1.0 - (torch.arange(num_steps, device=device, dtype=dtype) + 0.5) / num_steps
    normal_quantiles = torch.erfinv(2.0 * quantiles - 1.0) * (2.0**0.5)
    return torch.exp(normal_quantiles * p_std + p_mean)


def build_forward_karras_sigmas(
    *,
    num_steps: int,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    ramp = torch.linspace(0.0, 1.0, num_steps, device=device, dtype=dtype)
    min_inv_rho = sigma_min ** (1.0 / rho)
    max_inv_rho = sigma_max ** (1.0 / rho)
    descending = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return torch.flip(descending, dims=[0])


def build_forward_path(*, clean_points: torch.Tensor, sigmas: torch.Tensor, shared_noise: bool) -> torch.Tensor:
    batch_size, data_dim = clean_points.shape
    sigma_matrix = expand_sigmas(sigmas=sigmas, batch_size=batch_size)
    num_steps = sigma_matrix.shape[1]
    sigma_prev = torch.cat(
        [torch.zeros(batch_size, 1, device=clean_points.device, dtype=clean_points.dtype), sigma_matrix[:, :-1]],
        dim=1,
    )
    increment_scale = torch.sqrt(torch.clamp(sigma_matrix.square() - sigma_prev.square(), min=0.0))

    if shared_noise:
        eps = torch.randn(batch_size, 1, data_dim, device=clean_points.device, dtype=clean_points.dtype)
        eps = eps.expand(batch_size, num_steps, data_dim)
    else:
        eps = torch.randn(batch_size, num_steps, data_dim, device=clean_points.device, dtype=clean_points.dtype)

    increments = increment_scale[:, :, None] * eps
    return clean_points[:, None, :] + torch.cumsum(increments, dim=1)


def solve_causal_path_attack(
    *,
    attack_net,
    clean_points: torch.Tensor,
    reference_path: torch.Tensor,
    sigmas: torch.Tensor,
    inner_steps: int,
    step_size: float,
    gamma: float,
    total_budget: float | None = None,
    exact_budget_split: bool = False,
) -> torch.Tensor:
    batch_size = clean_points.shape[0]
    sigma_matrix = expand_sigmas(sigmas=sigmas, batch_size=batch_size)
    base_increments = torch.empty_like(reference_path)
    base_increments[:, 0, :] = reference_path[:, 0, :] - clean_points
    base_increments[:, 1:, :] = reference_path[:, 1:, :] - reference_path[:, :-1, :]

    adv_states: list[torch.Tensor] = []
    prev_adv = clean_points.detach()
    used_transport_cost = torch.zeros(batch_size, device=clean_points.device, dtype=clean_points.dtype)
    per_step_budget = None
    if total_budget is not None and reference_path.shape[1] > 0:
        per_step_budget = float(total_budget) / float(reference_path.shape[1])

    was_training = attack_net.training
    attack_net.eval()

    for step_idx in range(reference_path.shape[1]):
        sigma_step = sigma_matrix[:, step_idx]
        base_increment = base_increments[:, step_idx, :].detach()
        reference_state = reference_path[:, step_idx, :].detach()
        control = torch.zeros_like(prev_adv)

        # Solve the control sequentially so u_i only uses the current path prefix.
        # With exact_budget_split=False, this is the same WDRO quadratic transport
        # penalty, just applied at each attacked time along the path.
        for _ in range(inner_steps):
            control.requires_grad_(True)
            candidate = prev_adv + base_increment + control
            step_loss = _single_time_losses(
                net=attack_net,
                clean_points=clean_points,
                adv_points=candidate,
                sigma=sigma_step,
            ).mean()
            displacement = candidate - reference_state
            transport = 0.5 * displacement.square().sum(dim=1).mean()
            objective = step_loss - gamma * transport
            grad = torch.autograd.grad(objective, control)[0]
            control = (control + step_size * grad).detach()
            if total_budget is not None:
                candidate = prev_adv + base_increment + control
                if exact_budget_split and per_step_budget is not None:
                    candidate = project_to_step_budget(
                        candidate=candidate,
                        reference_state=reference_state,
                        step_budget=per_step_budget,
                        exact=True,
                    )
                else:
                    candidate = project_to_remaining_budget(
                        candidate=candidate,
                        reference_state=reference_state,
                        used_transport_cost=used_transport_cost,
                        total_budget=float(total_budget),
                    )
                control = (candidate - (prev_adv + base_increment)).detach()

        prev_adv = (prev_adv + base_increment + control).detach()
        used_transport_cost = used_transport_cost + 0.5 * (prev_adv - reference_state).square().sum(dim=1)
        adv_states.append(prev_adv)

    if was_training:
        attack_net.train()
    return torch.stack(adv_states, dim=1)


def project_to_remaining_budget(
    *,
    candidate: torch.Tensor,
    reference_state: torch.Tensor,
    used_transport_cost: torch.Tensor,
    total_budget: float,
) -> torch.Tensor:
    remaining_budget = torch.clamp(candidate.new_full(used_transport_cost.shape, total_budget) - used_transport_cost, min=0.0)
    displacement = candidate - reference_state
    displacement_flat = displacement.view(displacement.shape[0], -1)
    displacement_norm = displacement_flat.norm(dim=1)
    max_norm = torch.sqrt(torch.clamp(2.0 * remaining_budget, min=0.0))
    scale = torch.ones_like(displacement_norm)
    overspent = displacement_norm > max_norm
    scale[overspent] = max_norm[overspent] / displacement_norm[overspent].clamp_min(1e-12)
    return reference_state + displacement * scale.view(-1, 1)


def project_to_step_budget(
    *,
    candidate: torch.Tensor,
    reference_state: torch.Tensor,
    step_budget: float,
    exact: bool,
) -> torch.Tensor:
    displacement = candidate - reference_state
    displacement_flat = displacement.view(displacement.shape[0], -1)
    displacement_norm = displacement_flat.norm(dim=1)
    target_norm = torch.full_like(displacement_norm, float(max(step_budget, 0.0) * 2.0) ** 0.5)
    scale = torch.ones_like(displacement_norm)
    positive = displacement_norm > 1e-12
    if exact:
        scale[positive] = target_norm[positive] / displacement_norm[positive]
        scale[~positive] = 0.0
    else:
        overspent = displacement_norm > target_norm
        scale[overspent] = target_norm[overspent] / displacement_norm[overspent]
    return reference_state + displacement * scale.view(-1, 1)


def _per_time_losses(*, net, clean_points: torch.Tensor, adv_path: torch.Tensor, sigmas: torch.Tensor) -> torch.Tensor:
    batch_size, num_steps, data_dim = adv_path.shape
    sigma_matrix = expand_sigmas(sigmas=sigmas, batch_size=batch_size)
    flat_inputs = adv_path.reshape(batch_size * num_steps, data_dim)
    flat_sigmas = sigma_matrix.reshape(batch_size * num_steps)
    flat_targets = clean_points[:, None, :].expand(batch_size, num_steps, data_dim).reshape(batch_size * num_steps, data_dim)

    sigma = flat_sigmas.view(-1, 1)
    sigma_data_sq = net.sigma_data ** 2
    weight = (sigma.square() + sigma_data_sq) / (sigma * net.sigma_data).square()
    denoised = net(flat_inputs, flat_sigmas)
    sq_error = (denoised - flat_targets).square().mean(dim=1)
    return (weight.squeeze(1) * sq_error).view(batch_size, num_steps)


def _single_time_losses(*, net, clean_points: torch.Tensor, adv_points: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    sigma_batch = sigma.view(-1)
    sigma_col = sigma_batch.view(-1, 1)
    sigma_data_sq = net.sigma_data ** 2
    weight = (sigma_col.square() + sigma_data_sq) / (sigma_col * net.sigma_data).square()
    denoised = net(adv_points, sigma_batch)
    sq_error = (denoised - clean_points).square().mean(dim=1)
    return weight.squeeze(1) * sq_error


def expand_sigmas(*, sigmas: torch.Tensor, batch_size: int) -> torch.Tensor:
    if sigmas.ndim == 1:
        return sigmas.view(1, -1).expand(batch_size, sigmas.shape[0])
    if sigmas.ndim == 2 and sigmas.shape[0] == batch_size:
        return sigmas
    raise ValueError(f"sigmas must have shape [steps] or [batch, steps], got {tuple(sigmas.shape)}")
