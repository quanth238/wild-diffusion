import copy
from typing import Callable, Optional

import torch

from ...app.utils import empty_robust_history
from ...compute_accounting import append_denoiser_op_count_step, ensure_denoiser_op_count_history
from ...models import set_requires_grad
from ...shared.objective import compute_training_loss
from ...shared.runtime import autocast_context, resolve_amp_dtype
from ...shared.sigma import sample_target_indices, sample_target_indices_log_normal
from ...shared.train_utils import sample_train_batch
from ...utils import batch_scalar_like, has_nan_or_inf, scalarize


def train_trajectory_robust_clean(
    denoiser,
    control,
    centers,
    sigma_levels,
    cfg,
    train_pool: torch.Tensor = None,
    sample_train_batch_fn: Optional[Callable[[int], torch.Tensor]] = None,
    sample_population_batch_fn: Optional[Callable[[int], torch.Tensor]] = None,
    start_step: int = 0,
    history_state: Optional[dict] = None,
    optimizer_theta_state: Optional[dict] = None,
    ema_state_dict: Optional[dict] = None,
    return_state: bool = False,
):
    """Continue the baseline EDM objective from an initialized checkpoint."""

    del control
    if str(getattr(cfg, "training_objective", "edm")).lower() == "rf":
        raise NotImplementedError(
            "method_version='clean' does not yet implement rectified-flow continuation states. "
            "Use the shared baseline RF path or add a clean RF continuation implementation first."
        )

    optimizer_theta = torch.optim.Adam(denoiser.parameters(), lr=cfg.lr_theta)
    if optimizer_theta_state is not None:
        optimizer_theta.load_state_dict(optimizer_theta_state)
        for state in optimizer_theta.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device=sigma_levels.device)

    history = history_state if history_state is not None else empty_robust_history()
    for key in empty_robust_history():
        history.setdefault(key, [])
    ensure_denoiser_op_count_history(history)
    cumulative_batch_equiv_evals = (
        float(history["batch_equiv_denoiser_evals_cumulative"][-1])
        if history["batch_equiv_denoiser_evals_cumulative"]
        else 0.0
    )
    amp_dtype = resolve_amp_dtype(sigma_levels.device, getattr(cfg, "amp_dtype", "auto"))

    ema_model = None
    if cfg.use_ema_eval:
        ema_model = copy.deepcopy(denoiser).eval()
        set_requires_grad(ema_model, False)
        if ema_state_dict is not None:
            ema_model.load_state_dict(ema_state_dict, strict=True)

    for step in range(int(start_step) + 1, cfg.steps + 1):
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
        sigma = sigma_levels[indices]
        x_noisy = x0 + batch_scalar_like(sigma, x0) * torch.randn_like(x0)

        optimizer_theta.zero_grad(set_to_none=True)
        with autocast_context(sigma_levels.device, amp_dtype):
            loss = compute_training_loss(cfg, denoiser, x_noisy, x0, sigma)
        if has_nan_or_inf(loss):
            raise RuntimeError("NaN/Inf detected in clean continuation loss.")
        loss.backward()
        optimizer_theta.step()

        if ema_model is not None:
            with torch.no_grad():
                for p_ema, p in zip(ema_model.parameters(), denoiser.parameters()):
                    p_ema.mul_(cfg.ema_decay).add_(p, alpha=1.0 - cfg.ema_decay)

        cumulative_batch_equiv_evals += 1.0
        loss_scalar = scalarize(loss)
        history["outer_loss"].append(loss_scalar)
        history["outer_loss_attack"].append(0.0)
        history["outer_loss_clean"].append(loss_scalar)
        history["inner_obj"].append(0.0)
        history["energy"].append(0.0)
        history["delta_norm_mean"].append(0.0)
        history["delta_norm_max"].append(0.0)
        history["delta_norm_ratio_mean"].append(0.0)
        history["delta_norm_ratio_max"].append(0.0)
        history["sched_attack_weight"].append(0.0)
        history["sched_clean_weight"].append(1.0)
        history["sched_phi_lr_scale"].append(0.0)
        history["batch_equiv_denoiser_evals_step"].append(1.0)
        history["batch_equiv_denoiser_evals_attack_construction"].append(0.0)
        history["batch_equiv_denoiser_evals_attack_eval"].append(0.0)
        history["batch_equiv_denoiser_evals_clean_eval"].append(1.0)
        history["batch_equiv_denoiser_evals_cumulative"].append(float(cumulative_batch_equiv_evals))
        append_denoiser_op_count_step(
            history,
            n_fwd=0.0,
            n_fwd_inputgrad=0.0,
            n_fwd_parambackward=1.0,
        )

        if step % cfg.log_every == 0:
            print(
                f"[clean] step={step:05d} loss={loss.item():.6f} "
                f"be_evals=1.0 be_evals_cum={cumulative_batch_equiv_evals:.1f}",
                flush=True,
            )

    trainer_state = None
    if return_state:
        resume_model_state = copy.deepcopy(denoiser.state_dict())
        trainer_state = {
            "completed_steps": int(cfg.steps),
            "optimizer_theta_state": optimizer_theta.state_dict(),
            "ema_state_dict": None if ema_model is None else copy.deepcopy(ema_model.state_dict()),
            "resume_robust_state_dict": resume_model_state,
        }

    if ema_model is not None:
        denoiser.load_state_dict(ema_model.state_dict(), strict=True)

    if return_state:
        return history, trainer_state
    return history
