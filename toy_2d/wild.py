from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class WdroRefreshResult:
    combined_points: torch.Tensor
    adv_original: torch.Tensor | None
    adv_generated: torch.Tensor | None
    mean_l2_shift: float
    max_l2_shift: float
    mean_transport_cost: float
    max_transport_cost: float


def wdro_attack(
    points: torch.Tensor,
    model,
    loss_fn,
    *,
    gamma: float,
    step_size: float,
    iters: int,
    clamp_min: torch.Tensor,
    clamp_max: torch.Tensor,
) -> torch.Tensor:
    clamp_min = clamp_min.to(device=points.device, dtype=points.dtype).view(1, -1)
    clamp_max = clamp_max.to(device=points.device, dtype=points.dtype).view(1, -1)
    x_adv = points.detach().clone().requires_grad_(True)
    was_training = model.training
    model.eval()

    for _ in range(iters):
        with torch.enable_grad():
            loss_cls = loss_fn(model, x_adv).mean()
            delta = (x_adv - points).view(points.shape[0], -1)
            transport_cost = 0.5 * delta.square().sum(dim=1).mean()
            objective = loss_cls - gamma * transport_cost
        grad = torch.autograd.grad(objective, x_adv)[0]
        x_adv = (x_adv + step_size * grad).detach()
        x_adv = torch.maximum(torch.minimum(x_adv, clamp_max), clamp_min)
        x_adv.requires_grad_(True)

    if was_training:
        model.train()
    return x_adv.detach()


def build_wdro_dataset(
    *,
    base_points: torch.Tensor,
    model,
    loss_fn,
    batch_size: int,
    gamma: float,
    step_size: float,
    iters: int,
    p_adv: float,
    clamp_min: torch.Tensor,
    clamp_max: torch.Tensor,
    device: torch.device,
    rng: np.random.Generator,
    debug_adv_points: int = 256,
) -> WdroRefreshResult:
    combined_batches: list[torch.Tensor] = []
    adv_original: list[torch.Tensor] = []
    adv_generated: list[torch.Tensor] = []
    debug_left = max(int(debug_adv_points), 0)
    l2_shifts: list[torch.Tensor] = []
    transport_costs: list[torch.Tensor] = []

    for start in range(0, base_points.shape[0], batch_size):
        batch = base_points[start : start + batch_size].to(device)
        if p_adv >= 1.0 or rng.random() < p_adv:
            adv_batch = wdro_attack(
                batch,
                model,
                loss_fn,
                gamma=gamma,
                step_size=step_size,
                iters=iters,
                clamp_min=clamp_min,
                clamp_max=clamp_max,
            )
            combined_batches.append(batch.cpu())
            combined_batches.append(adv_batch.cpu())
            delta = adv_batch - batch
            l2_shifts.append(delta.norm(dim=1).cpu())
            transport_costs.append((0.5 * delta.square().sum(dim=1)).cpu())

            if debug_left > 0:
                take = min(debug_left, batch.shape[0])
                adv_original.append(batch[:take].cpu())
                adv_generated.append(adv_batch[:take].cpu())
                debug_left -= take
        else:
            combined_batches.append(batch.cpu())

    return WdroRefreshResult(
        combined_points=torch.cat(combined_batches, dim=0),
        adv_original=torch.cat(adv_original, dim=0) if adv_original else None,
        adv_generated=torch.cat(adv_generated, dim=0) if adv_generated else None,
        mean_l2_shift=float(torch.cat(l2_shifts).mean().item()) if l2_shifts else 0.0,
        max_l2_shift=float(torch.cat(l2_shifts).max().item()) if l2_shifts else 0.0,
        mean_transport_cost=float(torch.cat(transport_costs).mean().item()) if transport_costs else 0.0,
        max_transport_cost=float(torch.cat(transport_costs).max().item()) if transport_costs else 0.0,
    )
