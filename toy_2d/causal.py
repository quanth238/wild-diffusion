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


def causal_wdro_loss(
    *,
    net,
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
    shared_noise: bool = False,
    sigma_schedule: str = "karras_grid",
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
        net=net,
        clean_points=clean_points,
        reference_path=reference_path,
        sigmas=sigmas,
        inner_steps=inner_steps,
        step_size=step_size,
        gamma=gamma,
    )
    outer_loss = _per_time_losses(net=net, clean_points=clean_points, adv_path=final_path, sigmas=sigmas).mean()
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
    net,
    clean_points: torch.Tensor,
    reference_path: torch.Tensor,
    sigmas: torch.Tensor,
    inner_steps: int,
    step_size: float,
    gamma: float,
) -> torch.Tensor:
    batch_size = clean_points.shape[0]
    sigma_matrix = expand_sigmas(sigmas=sigmas, batch_size=batch_size)
    base_increments = torch.empty_like(reference_path)
    base_increments[:, 0, :] = reference_path[:, 0, :] - clean_points
    base_increments[:, 1:, :] = reference_path[:, 1:, :] - reference_path[:, :-1, :]

    adv_states: list[torch.Tensor] = []
    prev_adv = clean_points.detach()

    for step_idx in range(reference_path.shape[1]):
        sigma_step = sigma_matrix[:, step_idx]
        base_increment = base_increments[:, step_idx, :].detach()
        reference_state = reference_path[:, step_idx, :].detach()
        control = torch.zeros_like(prev_adv)

        # Solve the control sequentially so u_i only uses the current path prefix.
        # The penalty uses accumulated path displacement so the total path budget
        # can be matched directly against x0-only WDRO.
        for _ in range(inner_steps):
            control.requires_grad_(True)
            candidate = prev_adv + base_increment + control
            step_loss = _single_time_losses(
                net=net,
                clean_points=clean_points,
                adv_points=candidate,
                sigma=sigma_step,
            ).mean()
            displacement = candidate - reference_state
            transport = 0.5 * displacement.square().sum(dim=1).mean()
            objective = step_loss - gamma * transport
            grad = torch.autograd.grad(objective, control)[0]
            grad_flat = grad.view(grad.shape[0], -1)
            grad_norm = grad_flat.norm(dim=1, keepdim=True).clamp_min(1e-12)
            normalized_grad = grad / grad_norm.view(-1, 1)
            control = (control + step_size * normalized_grad).detach()

        prev_adv = (prev_adv + base_increment + control).detach()
        adv_states.append(prev_adv)

    return torch.stack(adv_states, dim=1)


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
