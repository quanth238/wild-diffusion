import torch

from ..utils import batch_scalar_like


def edm_weight(sigma: torch.Tensor, sigma_data: float) -> torch.Tensor:
    """EDM per-sample weighting for x0-regression."""

    sigma2 = sigma.square()
    sigma_data2 = sigma_data ** 2
    return (sigma2 + sigma_data2) / (sigma * sigma_data).square().clamp_min(1e-8)


def score_matching_weight(sigma: torch.Tensor, weight_power: float = 2.0) -> torch.Tensor:
    """Per-sample weight lambda(sigma)=sigma^p for score matching."""

    sigma = sigma.clamp_min(1e-8)
    return sigma.pow(float(weight_power))


def predict_score(
    denoiser,
    x_noisy: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """Predict score field s_theta(x,sigma) with a model-specific fast path."""

    if hasattr(denoiser, "predict_score"):
        return denoiser.predict_score(x_noisy, sigma)
    sigma2 = batch_scalar_like(sigma.clamp_min(1e-6).square(), x_noisy)
    x0_pred = denoiser(x_noisy, sigma)
    return (x0_pred - x_noisy) / sigma2.clamp_min(1e-8)


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


def weighted_score_matching_loss(
    denoiser,
    x_noisy: torch.Tensor,
    x_clean: torch.Tensor,
    sigma: torch.Tensor,
    score_weight_power: float = 2.0,
) -> torch.Tensor:
    """Compute E[lambda(sigma) * ||s_theta(x_sigma,sigma) - s_target||^2]."""

    sigma = sigma.clamp_min(1e-6)
    pred_score = predict_score(denoiser, x_noisy, sigma)
    target_score = -(x_noisy - x_clean) / batch_scalar_like(sigma.square(), x_noisy).clamp_min(1e-8)
    w = batch_scalar_like(score_matching_weight(sigma, score_weight_power), x_clean)
    return (w * (pred_score - target_score).pow(2)).reshape(x_clean.shape[0], -1).sum(dim=1).mean()


def compute_training_loss(
    cfg,
    denoiser,
    x_noisy: torch.Tensor,
    x_clean: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """Dispatch train loss by objective kind: EDM x0 regression or score matching."""

    objective = str(getattr(cfg, "training_objective", "edm")).lower()
    if objective == "edm":
        return weighted_denoise_loss(denoiser, x_noisy, x_clean, sigma, cfg.sigma_data)
    if objective == "score":
        return weighted_score_matching_loss(
            denoiser,
            x_noisy,
            x_clean,
            sigma,
            score_weight_power=float(getattr(cfg, "score_matching_weight_power", 2.0)),
        )
    raise ValueError(f"Unsupported training objective '{objective}'. Expected one of: edm, score.")


def inner_objective_attack_only(train_loss: torch.Tensor) -> torch.Tensor:
    """Inner objective J(phi): maximize denoising loss on controlled trajectories."""

    return train_loss
