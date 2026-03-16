from __future__ import annotations

import torch


def mmd_rbf(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    bandwidths: tuple[float, ...] = (0.2, 0.5, 1.0, 2.0, 5.0),
) -> float:
    xx = torch.cdist(x, x).square()
    yy = torch.cdist(y, y).square()
    xy = torch.cdist(x, y).square()

    kernel_xx = 0.0
    kernel_yy = 0.0
    kernel_xy = 0.0
    for bandwidth in bandwidths:
        gamma = 1.0 / (2.0 * bandwidth * bandwidth)
        kernel_xx = kernel_xx + torch.exp(-gamma * xx)
        kernel_yy = kernel_yy + torch.exp(-gamma * yy)
        kernel_xy = kernel_xy + torch.exp(-gamma * xy)

    m = x.shape[0]
    n = y.shape[0]
    mmd = kernel_xx.sum() / (m * m) + kernel_yy.sum() / (n * n) - 2.0 * kernel_xy.sum() / (m * n)
    return float(mmd.item())


def sliced_wasserstein(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    num_projections: int = 128,
    seed: int = 0,
) -> float:
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same number of samples.")

    generator = torch.Generator(device=x.device)
    generator.manual_seed(seed)
    directions = torch.randn(num_projections, x.shape[1], generator=generator, device=x.device, dtype=x.dtype)
    directions = directions / directions.norm(dim=1, keepdim=True).clamp(min=1e-12)

    proj_x = (x @ directions.T).sort(dim=0).values
    proj_y = (y @ directions.T).sort(dim=0).values
    distance = (proj_x - proj_y).abs().mean()
    return float(distance.item())

