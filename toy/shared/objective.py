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


def rf_time_from_sigma(
    sigma: torch.Tensor,
    sigma_max: float,
) -> torch.Tensor:
    """Map the existing positive sigma ladder onto normalized RF time in [0, 1]."""

    sigma_max_value = max(float(sigma_max), 1e-8)
    return (sigma / sigma_max_value).clamp(0.0, 1.0)


def rf_time_levels_from_sigma_levels(
    sigma_levels: torch.Tensor,
) -> torch.Tensor:
    """Build normalized RF time levels from the existing sigma ladder."""

    if sigma_levels.numel() < 2:
        raise ValueError(f"sigma_levels must contain at least 2 values, got {sigma_levels.numel()}")
    sigma_max_value = max(float(sigma_levels[-1].item()), 1e-8)
    return (sigma_levels / sigma_max_value).clamp(0.0, 1.0)


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


def predict_velocity(
    denoiser,
    x_t: torch.Tensor,
    sigma: torch.Tensor,
    sigma_max: float,
) -> torch.Tensor:
    """Predict RF velocity field v_theta(x_t, t) with a model-specific fast path."""

    if hasattr(denoiser, "predict_velocity"):
        return denoiser.predict_velocity(x_t, sigma)
    t = batch_scalar_like(rf_time_from_sigma(sigma, sigma_max).clamp_min(1e-6), x_t)
    x0_pred = denoiser(x_t, sigma)
    return (x_t - x0_pred) / t.clamp_min(1e-8)


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


def weighted_rectified_flow_loss(
    denoiser,
    x_t: torch.Tensor,
    x_clean: torch.Tensor,
    sigma: torch.Tensor,
    sigma_max: float,
) -> torch.Tensor:
    """Compute RF regression loss E[||v_theta(x_t,t) - (x_t - x0)/t||^2]."""

    t = rf_time_from_sigma(sigma, sigma_max).clamp_min(1e-6)
    pred_velocity = predict_velocity(denoiser, x_t, sigma, sigma_max=sigma_max)
    target_velocity = (x_t - x_clean) / batch_scalar_like(t, x_t).clamp_min(1e-8)
    return (pred_velocity - target_velocity).pow(2).reshape(x_clean.shape[0], -1).sum(dim=1).mean()


def build_training_state(
    *,
    cfg,
    x_clean: torch.Tensor,
    sigma: torch.Tensor,
    sample_terminal_batch_fn=None,
) -> torch.Tensor:
    """Construct the objective-specific training state x_t from clean samples."""

    objective = str(getattr(cfg, "training_objective", "edm")).lower()
    if objective in ("edm", "score"):
        return x_clean + batch_scalar_like(sigma, x_clean) * torch.randn_like(x_clean)
    if objective == "rf":
        terminal_sigma = float(getattr(cfg, "sigma_max", 1.0))
        if sample_terminal_batch_fn is None:
            x_terminal = torch.randn_like(x_clean) * terminal_sigma
        else:
            x_terminal = sample_terminal_batch_fn(x_clean.shape[0], terminal_sigma).to(
                device=x_clean.device,
                dtype=x_clean.dtype,
            )
        t = batch_scalar_like(rf_time_from_sigma(sigma, terminal_sigma), x_clean)
        return (1.0 - t) * x_clean + t * x_terminal
    raise ValueError(f"Unsupported training objective '{objective}'. Expected one of: edm, score, rf.")


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
    if objective == "rf":
        return weighted_rectified_flow_loss(
            denoiser,
            x_noisy,
            x_clean,
            sigma,
            sigma_max=float(getattr(cfg, "sigma_max", 1.0)),
        )
    raise ValueError(f"Unsupported training objective '{objective}'. Expected one of: edm, score, rf.")


def inner_objective_attack_only(train_loss: torch.Tensor) -> torch.Tensor:
    """Inner objective J(phi): maximize denoising loss on controlled trajectories."""

    return train_loss
