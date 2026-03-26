import torch

from ..utils import batch_scalar_like


def edm_weight(sigma: torch.Tensor, sigma_data: float) -> torch.Tensor:
    """EDM per-sample weighting for x0-regression."""

    sigma2 = sigma.square()
    sigma_data2 = sigma_data ** 2
    return (sigma2 + sigma_data2) / (sigma * sigma_data).square().clamp_min(1e-8)


def weighted_denoise_loss(
    denoiser,
    x_noisy: torch.Tensor,
    x_clean: torch.Tensor,
    sigma: torch.Tensor,
    sigma_data: float,
) -> torch.Tensor:
    """Compute weighted denoising objective E[w(sigma) * ||x0_hat - x0||^2]."""

    pred = denoiser(x_noisy, sigma)
    w = batch_scalar_like(edm_weight(sigma, sigma_data), x_clean)
    return (w * (pred - x_clean).pow(2)).reshape(x_clean.shape[0], -1).sum(dim=1).mean()


def inner_objective_attack_only(train_loss: torch.Tensor) -> torch.Tensor:
    """Inner objective J(phi): maximize denoising loss on controlled trajectories."""

    return train_loss

