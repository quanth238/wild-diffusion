from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from toy_2d.cdro import CdroConfig


def solve_path_heuristic_cdro_attack(
    *,
    attack_net,
    clean_points: torch.Tensor,
    reference_path: torch.Tensor,
    sigmas: torch.Tensor,
    config: "CdroConfig",
) -> torch.Tensor:
    batch_size = clean_points.shape[0]
    sigma_matrix = _expand_sigmas(sigmas=sigmas, batch_size=batch_size)
    base_increments = torch.empty_like(reference_path)
    base_increments[:, 0, :] = reference_path[:, 0, :] - clean_points
    base_increments[:, 1:, :] = reference_path[:, 1:, :] - reference_path[:, :-1, :]

    adv_states: list[torch.Tensor] = []
    prev_adv = clean_points.detach()
    used_transport_cost = torch.zeros(batch_size, device=clean_points.device, dtype=clean_points.dtype)
    per_step_budget = None
    if config.total_budget is not None and reference_path.shape[1] > 0:
        per_step_budget = float(config.total_budget) / float(reference_path.shape[1])

    was_training = attack_net.training
    attack_net.eval()

    for step_idx in range(reference_path.shape[1]):
        sigma_step = sigma_matrix[:, step_idx]
        base_increment = base_increments[:, step_idx, :].detach()
        reference_state = reference_path[:, step_idx, :].detach()
        control = torch.zeros_like(prev_adv)

        # This is the current legacy path heuristic: optimize one control at a time
        # using only the current path prefix, then project back into the budget set.
        for _ in range(config.inner_steps):
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
            objective = step_loss - config.gamma * transport
            grad = torch.autograd.grad(objective, control)[0]
            control = (control + config.step_size * grad).detach()
            if config.total_budget is not None:
                candidate = prev_adv + base_increment + control
                if config.exact_budget_split and per_step_budget is not None:
                    candidate = _project_to_step_budget(
                        candidate=candidate,
                        reference_state=reference_state,
                        step_budget=per_step_budget,
                        exact=True,
                    )
                else:
                    candidate = _project_to_remaining_budget(
                        candidate=candidate,
                        reference_state=reference_state,
                        used_transport_cost=used_transport_cost,
                        total_budget=float(config.total_budget),
                    )
                control = (candidate - (prev_adv + base_increment)).detach()

        prev_adv = (prev_adv + base_increment + control).detach()
        used_transport_cost = used_transport_cost + 0.5 * (prev_adv - reference_state).square().sum(dim=1)
        adv_states.append(prev_adv)

    if was_training:
        attack_net.train()
    return torch.stack(adv_states, dim=1)


def _project_to_remaining_budget(
    *,
    candidate: torch.Tensor,
    reference_state: torch.Tensor,
    used_transport_cost: torch.Tensor,
    total_budget: float,
) -> torch.Tensor:
    remaining_budget = torch.clamp(
        candidate.new_full(used_transport_cost.shape, total_budget) - used_transport_cost,
        min=0.0,
    )
    displacement = candidate - reference_state
    displacement_flat = displacement.view(displacement.shape[0], -1)
    displacement_norm = displacement_flat.norm(dim=1)
    max_norm = torch.sqrt(torch.clamp(2.0 * remaining_budget, min=0.0))
    scale = torch.ones_like(displacement_norm)
    overspent = displacement_norm > max_norm
    scale[overspent] = max_norm[overspent] / displacement_norm[overspent].clamp_min(1e-12)
    return reference_state + displacement * scale.view(-1, 1)


def _project_to_step_budget(
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


def _single_time_losses(*, net, clean_points: torch.Tensor, adv_points: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    sigma_batch = sigma.view(-1)
    sigma_col = sigma_batch.view(-1, 1)
    sigma_data_sq = net.sigma_data ** 2
    weight = (sigma_col.square() + sigma_data_sq) / (sigma_col * net.sigma_data).square()
    denoised = net(adv_points, sigma_batch)
    sq_error = (denoised - clean_points).square().mean(dim=1)
    return weight.squeeze(1) * sq_error


def _expand_sigmas(*, sigmas: torch.Tensor, batch_size: int) -> torch.Tensor:
    if sigmas.ndim == 1:
        return sigmas.view(1, -1).expand(batch_size, sigmas.shape[0])
    if sigmas.ndim == 2 and sigmas.shape[0] == batch_size:
        return sigmas
    raise ValueError(f"sigmas must have shape [steps] or [batch, steps], got {tuple(sigmas.shape)}")
