from __future__ import annotations

import torch


def make_sigma_schedule(
    *,
    num_steps: int,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    device: torch.device,
) -> torch.Tensor:
    if num_steps < 2:
        raise ValueError("num_steps must be at least 2.")
    ramp = torch.linspace(0.0, 1.0, num_steps, device=device)
    min_inv_rho = sigma_min ** (1.0 / rho)
    max_inv_rho = sigma_max ** (1.0 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return torch.cat([sigmas, torch.zeros(1, device=device)])


@torch.no_grad()
def sample_edm(
    model,
    *,
    num_samples: int,
    data_dim: int,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    num_steps: int,
    device: torch.device,
) -> torch.Tensor:
    samples, _, _ = sample_edm_trajectory(
        model,
        num_samples=num_samples,
        data_dim=data_dim,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        rho=rho,
        num_steps=num_steps,
        device=device,
    )
    return samples


@torch.no_grad()
def sample_edm_trajectory(
    model,
    *,
    data_dim: int,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    num_steps: int,
    device: torch.device,
    num_samples: int | None = None,
    initial_points: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
    sigmas = make_sigma_schedule(
        num_steps=num_steps,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        rho=rho,
        device=device,
    )

    if initial_points is not None:
        if initial_points.ndim != 2 or initial_points.shape[1] != data_dim:
            raise ValueError(
                f"initial_points must have shape [batch, {data_dim}], got {tuple(initial_points.shape)}"
            )
        x_next = initial_points.to(device=device)
    else:
        if num_samples is None:
            raise ValueError("num_samples is required when initial_points is not provided.")
        x_next = torch.randn(num_samples, data_dim, device=device) * sigmas[0]

    trajectory = [x_next.detach().cpu()]

    for step_idx in range(num_steps):
        sigma_cur = sigmas[step_idx]
        sigma_next = sigmas[step_idx + 1]
        sigma_batch = torch.full((x_next.shape[0],), float(sigma_cur), device=device)
        denoised = model(x_next, sigma_batch)
        d_cur = (x_next - denoised) / sigma_cur
        x_next = x_next + (sigma_next - sigma_cur) * d_cur
        trajectory.append(x_next.detach().cpu())

    return x_next, trajectory, sigmas.detach().cpu()
