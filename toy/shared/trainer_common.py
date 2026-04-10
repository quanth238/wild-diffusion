from typing import Callable, Optional

import torch

from ..compute_accounting import append_denoiser_op_count_step, ensure_denoiser_op_count_history
from .ema import init_ema_model, update_ema_model
from ..shared.runtime import autocast_context, resolve_amp_dtype
from ..utils import batch_scalar_like, has_nan_or_inf, scalarize
from .objective import build_training_state, compute_training_loss, weighted_denoise_loss
from .reverse import sample_reverse_paths
from .sigma import sample_target_indices, sample_target_indices_log_normal
from .train_utils import sample_train_batch


def train_baseline(
    denoiser,
    centers,
    sigma_levels,
    cfg,
    train_pool: torch.Tensor = None,
    sample_train_batch_fn: Optional[Callable[[int], torch.Tensor]] = None,
    sample_population_batch_fn: Optional[Callable[[int], torch.Tensor]] = None,
    sample_terminal_batch_fn: Optional[Callable[[int, float], torch.Tensor]] = None,
):
    """Train baseline model theta under the configured generative objective."""

    optimizer = torch.optim.Adam(denoiser.parameters(), lr=cfg.lr_theta)
    history = {"loss": [], "proxy_weighted_denoise_loss": []}
    ensure_denoiser_op_count_history(history)
    sigma_counts = torch.zeros(sigma_levels.numel() - 1, device=sigma_levels.device, dtype=torch.long)
    amp_dtype = resolve_amp_dtype(sigma_levels.device, getattr(cfg, "amp_dtype", "auto"))

    ema_model = init_ema_model(denoiser, cfg)

    for step in range(1, cfg.steps + 1):
        x0 = sample_train_batch(
            cfg,
            centers,
            train_pool=train_pool,
            sample_train_batch_fn=sample_train_batch_fn,
            sample_population_batch_fn=sample_population_batch_fn,
        )
        if cfg.use_log_normal_sigma_sampling:
            indices = sample_target_indices_log_normal(
                cfg.batch_size,
                sigma_levels,
                p_mean=cfg.p_mean,
                p_std=cfg.p_std,
            )
        else:
            indices = sample_target_indices(cfg.batch_size, sigma_levels)
        sigma_counts += torch.bincount(indices - 1, minlength=sigma_counts.numel())
        sigma = sigma_levels[indices]
        x_noisy = build_training_state(
            cfg=cfg,
            x_clean=x0,
            sigma=sigma,
            sample_terminal_batch_fn=sample_terminal_batch_fn,
        )

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(sigma_levels.device, amp_dtype):
            loss = compute_training_loss(cfg, denoiser, x_noisy, x0, sigma)
        if has_nan_or_inf(loss):
            raise RuntimeError("NaN/Inf detected in baseline loss.")
        loss.backward()
        optimizer.step()
        update_ema_model(
            ema_model,
            denoiser,
            cfg,
            cur_nimg=float((step - 1) * int(cfg.batch_size)),
            batch_size=int(cfg.batch_size),
        )
        history["loss"].append(scalarize(loss))
        if str(getattr(cfg, "training_objective", "edm")).lower() == "edm":
            proxy_loss = loss
        else:
            with torch.no_grad():
                with autocast_context(sigma_levels.device, amp_dtype):
                    proxy_loss = weighted_denoise_loss(denoiser, x_noisy, x0, sigma, cfg.sigma_data)
        history["proxy_weighted_denoise_loss"].append(scalarize(proxy_loss))
        append_denoiser_op_count_step(
            history,
            n_fwd=0.0,
            n_fwd_inputgrad=0.0,
            n_fwd_parambackward=1.0,
        )

        if step % cfg.log_every == 0:
            if str(getattr(cfg, "training_objective", "edm")).lower() == "edm":
                print(f"[baseline] step={step:05d} loss={loss.item():.6f}", flush=True)
            else:
                print(
                    f"[baseline] step={step:05d} loss={loss.item():.6f} "
                    f"proxy_weighted_denoise={proxy_loss.item():.6f}",
                    flush=True,
                )

    history["sigma_counts"] = [int(v) for v in sigma_counts.detach().cpu().tolist()]
    eval_model = ema_model if ema_model is not None else denoiser
    return history, eval_model


@torch.no_grad()
def sample_with_denoiser(denoiser, sigma_levels, n_samples, device, sample_terminal_batch_fn=None):
    """Sample x0 by running reverse paths from Gaussian terminal noise."""

    states = sample_reverse_paths(
        denoiser,
        sigma_levels,
        n_samples,
        device,
        stochastic=True,
        sample_terminal_batch_fn=sample_terminal_batch_fn,
    )
    return states[:, 0]
