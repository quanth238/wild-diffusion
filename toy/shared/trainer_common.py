import copy
import math
import time
from typing import Callable, Optional

import torch

from ..compute_accounting import append_denoiser_op_count_step, ensure_denoiser_op_count_history
from .ema import init_ema_model, update_ema_model
from ..shared.runtime import autocast_context, resolve_amp_dtype
from ..utils import has_nan_or_inf, scalarize
from .objective import (
    build_rectified_flow_state,
    build_training_state,
    compute_training_loss,
    terminal_prior_scale_from_family,
    weighted_denoise_loss,
)
from .reverse import (
    generated_data_path_index_from_denoiser,
    reverse_paths_from_terminal,
    resolve_ve_sampler_mode,
    sample_rectified_flow_paths_from_source,
    sample_reverse_paths,
)
from .sigma import (
    assign_sigmas_to_nearest_levels,
    resolve_rf_stage_t_distribution,
    sample_sigmas_log_normal,
    sample_target_indices,
)
from .train_utils import sample_train_batch


def _is_strong_rf_baseline(cfg) -> bool:
    return (
        str(getattr(cfg, "training_objective", "edm")).strip().lower() == "rf"
        and str(getattr(cfg, "rf_baseline_mode", "strong")).strip().lower() == "strong"
    )


def _rf_ema_cfg(cfg):
    """Use EMA for strong RF even when legacy EDM runs leave EMA disabled."""

    ema_cfg = copy.copy(cfg)
    ema_cfg.use_ema_eval = True
    return ema_cfg


def _resolve_rf_stage_steps(total_steps: int, stage1_fraction: float) -> tuple[int, int]:
    total_steps = max(int(total_steps), 0)
    del stage1_fraction
    return 0, total_steps


def _sample_rf_t(
    batch_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    distribution: str,
) -> torch.Tensor:
    mode = str(distribution).strip().lower()
    u = torch.rand(int(batch_size), device=device, dtype=dtype)
    if mode == "uniform":
        t = u
    elif mode == "u_shaped":
        # Beta(1/2, 1/2), sampled by inverse CDF.
        t = torch.sin(0.5 * math.pi * u).square()
    else:
        raise ValueError(
            f"Unsupported RF timestep distribution '{distribution}'. Expected one of: uniform, u_shaped."
        )
    return t.clamp(1e-5, 1.0 - 1e-5)


def _sample_terminal_like(
    x_template: torch.Tensor,
    *,
    sample_terminal_batch_fn: Optional[Callable[[int, float], torch.Tensor]] = None,
) -> torch.Tensor:
    if sample_terminal_batch_fn is None:
        return torch.randn_like(x_template)
    return sample_terminal_batch_fn(x_template.shape[0], 1.0).to(
        device=x_template.device,
        dtype=x_template.dtype,
    )


def _sample_teacher_terminal_like(
    teacher,
    x_template: torch.Tensor,
    sigma_levels: torch.Tensor,
    *,
    sample_terminal_batch_fn: Optional[Callable[[int, float], torch.Tensor]] = None,
) -> tuple[torch.Tensor, float]:
    terminal_scale = terminal_prior_scale_from_family(
        getattr(teacher, "generative_family", ""),
        float(sigma_levels[-1].item()),
    )
    if sample_terminal_batch_fn is None:
        x_terminal = torch.randn_like(x_template) * float(terminal_scale)
    else:
        x_terminal = sample_terminal_batch_fn(x_template.shape[0], float(terminal_scale)).to(
            device=x_template.device,
            dtype=x_template.dtype,
        )
    return x_terminal, float(max(float(terminal_scale), 1e-8))


def resolve_rf_teacher_ve_sampler_mode(cfg) -> str:
    """Resolve the VE sampler used when an EDM teacher emits RF reflow pairs."""

    return resolve_ve_sampler_mode(
        stochastic=True,
        sampler_mode=str(getattr(cfg, "rf_edm_teacher_sampler", "edm_heun")),
    )


def resolve_rf_teacher_pair_sampling_mode(
    teacher,
    cfg=None,
    *,
    ve_sampler_mode: Optional[str] = None,
) -> str:
    """Human-readable teacher pair sampler mode for RF diagnostics."""

    family = str(getattr(teacher, "generative_family", "")).strip().lower()
    if family == "rectified_flow":
        return "deterministic_rf_path"
    if ve_sampler_mode is None:
        ve_sampler_mode = resolve_rf_teacher_ve_sampler_mode(cfg)
    return resolve_ve_sampler_mode(stochastic=True, sampler_mode=ve_sampler_mode)


@torch.no_grad()
def generate_reflow_pairs(
    teacher,
    sigma_levels: torch.Tensor,
    x_template: torch.Tensor,
    *,
    sample_terminal_batch_fn: Optional[Callable[[int, float], torch.Tensor]] = None,
    ve_sampler_mode: str = "edm_heun",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate one-round reflow pairs `(x_left, x_right)` from a frozen teacher."""

    was_training = teacher.training
    teacher.eval()
    try:
        teacher_family = str(getattr(teacher, "generative_family", "")).strip().lower()
        if teacher_family == "rectified_flow":
            x_left = _sample_terminal_like(
                x_template,
                sample_terminal_batch_fn=sample_terminal_batch_fn,
            )
            paths = sample_rectified_flow_paths_from_source(
                denoiser=teacher,
                x_source=x_left,
                sigma_levels=sigma_levels,
            )
        else:
            x_terminal, terminal_scale = _sample_teacher_terminal_like(
                teacher,
                x_template,
                sigma_levels,
                sample_terminal_batch_fn=sample_terminal_batch_fn,
            )
            resolved_ve_sampler_mode = resolve_ve_sampler_mode(
                stochastic=True,
                sampler_mode=ve_sampler_mode,
            )
            paths = reverse_paths_from_terminal(
                denoiser=teacher,
                x_terminal=x_terminal,
                sigma_levels=sigma_levels,
                stochastic=resolved_ve_sampler_mode == "ancestral_stochastic",
                sampler_mode=resolved_ve_sampler_mode,
            )
            x_left = x_terminal / float(terminal_scale)
        generated_idx = generated_data_path_index_from_denoiser(teacher)
        x_generated = paths[:, generated_idx].detach()
    finally:
        if was_training:
            teacher.train()
    return x_left.detach(), x_generated


def _train_rf_pair_stage(
    *,
    denoiser,
    optimizer,
    ema_model,
    ema_cfg,
    centers,
    sigma_levels,
    cfg,
    history,
    sigma_counts,
    amp_dtype,
    stage_name: str,
    num_steps: int,
    global_step_offset: int,
    t_distribution: str,
    train_pool: torch.Tensor = None,
    sample_train_batch_fn: Optional[Callable[[int], torch.Tensor]] = None,
    sample_population_batch_fn: Optional[Callable[[int], torch.Tensor]] = None,
    sample_terminal_batch_fn: Optional[Callable[[int, float], torch.Tensor]] = None,
    reflow_teacher=None,
    reflow_teacher_sigma_levels: Optional[torch.Tensor] = None,
):
    """Train one RF stage on explicit straight-path pairs."""

    for local_step in range(1, int(num_steps) + 1):
        step_t0 = time.perf_counter()
        global_step = int(global_step_offset) + int(local_step)
        x_template = sample_train_batch(
            cfg,
            centers,
            train_pool=train_pool,
            sample_train_batch_fn=sample_train_batch_fn,
            sample_population_batch_fn=sample_population_batch_fn,
        )
        if reflow_teacher is None:
            x_right = x_template
            x_left = _sample_terminal_like(x_template, sample_terminal_batch_fn=sample_terminal_batch_fn)
            reflow_fwd_units = 0.0
        else:
            teacher_sigma_levels = reflow_teacher_sigma_levels if reflow_teacher_sigma_levels is not None else sigma_levels
            x_left, x_right = generate_reflow_pairs(
                reflow_teacher,
                teacher_sigma_levels,
                x_template,
                sample_terminal_batch_fn=sample_terminal_batch_fn,
                ve_sampler_mode=resolve_rf_teacher_ve_sampler_mode(cfg),
            )
            reflow_fwd_units = float(max(int(teacher_sigma_levels.numel()) - 1, 0))

        t = _sample_rf_t(
            int(x_template.shape[0]),
            device=sigma_levels.device,
            dtype=sigma_levels.dtype,
            distribution=t_distribution,
        )
        sigma = t * float(getattr(cfg, "sigma_max", float(sigma_levels[-1].item())))
        indices = assign_sigmas_to_nearest_levels(sigma, sigma_levels)
        sigma_counts += torch.bincount(indices - 1, minlength=sigma_counts.numel())
        x_t = build_rectified_flow_state(x_left, x_right, t)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(sigma_levels.device, amp_dtype):
            loss = compute_training_loss(cfg, denoiser, x_t, x_right, sigma, x_left=x_left, x_right=x_right)
        if has_nan_or_inf(loss):
            raise RuntimeError(f"NaN/Inf detected in {stage_name} RF baseline loss.")
        loss.backward()
        optimizer.step()
        update_ema_model(
            ema_model,
            denoiser,
            ema_cfg,
            cur_nimg=float((global_step - 1) * int(cfg.batch_size)),
            batch_size=int(cfg.batch_size),
        )

        history["loss"].append(scalarize(loss))
        with torch.no_grad():
            with autocast_context(sigma_levels.device, amp_dtype):
                proxy_loss = weighted_denoise_loss(denoiser, x_t, x_right, sigma, cfg.sigma_data)
        history["proxy_weighted_denoise_loss"].append(scalarize(proxy_loss))
        history.setdefault("rf_stage", []).append(stage_name)
        history.setdefault("rf_t_mean", []).append(float(t.detach().mean().item()))
        history.setdefault("rf_reflow_pair_batches", []).append(int(reflow_teacher is not None))
        append_denoiser_op_count_step(
            history,
            n_fwd=float(reflow_fwd_units),
            n_fwd_inputgrad=0.0,
            n_fwd_parambackward=1.0,
        )

        if global_step % cfg.log_every == 0:
            print(
                f"[baseline-rf:{stage_name}] step={global_step:05d} "
                f"loss={loss.item():.6f} proxy_weighted_denoise={proxy_loss.item():.6f} "
                f"t_mean={float(t.detach().mean().item()):.4f}",
                flush=True,
                )

        history.setdefault("rf_step_wall_clock_sec", []).append(float(time.perf_counter() - step_t0))

    return ema_model


def _train_strong_rf_baseline(
    denoiser,
    centers,
    sigma_levels,
    cfg,
    train_pool: torch.Tensor = None,
    sample_train_batch_fn: Optional[Callable[[int], torch.Tensor]] = None,
    sample_population_batch_fn: Optional[Callable[[int], torch.Tensor]] = None,
    sample_terminal_batch_fn: Optional[Callable[[int, float], torch.Tensor]] = None,
    rf_teacher_model=None,
    rf_teacher_sigma_levels: Optional[torch.Tensor] = None,
):
    """Train the public RF baseline as one explicit teacher-pair reflow round."""

    history = {
        "loss": [],
        "proxy_weighted_denoise_loss": [],
        "rf_stage": [],
        "rf_t_mean": [],
        "rf_reflow_pair_batches": [],
        "rf_step_wall_clock_sec": [],
    }
    ensure_denoiser_op_count_history(history)
    sigma_counts = torch.zeros(sigma_levels.numel() - 1, device=sigma_levels.device, dtype=torch.long)
    amp_dtype = resolve_amp_dtype(sigma_levels.device, getattr(cfg, "amp_dtype", "auto"))
    ema_cfg = _rf_ema_cfg(cfg)
    stage1_steps = 0
    reflow_steps = max(int(cfg.steps), 0)
    rf_reflow_t_distribution = resolve_rf_stage_t_distribution(
        "rf_reflow",
        reflow_distribution=str(getattr(cfg, "rf_reflow_t_distribution", "u_shaped")),
    )
    history["rf_stage1_steps"] = int(stage1_steps)
    history["rf_reflow_steps"] = int(reflow_steps)
    history["rf_stage1_t_distribution_resolved"] = ""
    history["rf_reflow_t_distribution_resolved"] = str(rf_reflow_t_distribution)
    history["rf_eval_t_distribution_resolved"] = str(rf_reflow_t_distribution)
    history["rf_pair_source_resolved"] = "shared_edm_teacher_reflow"

    optimizer = torch.optim.Adam(denoiser.parameters(), lr=cfg.lr_theta)
    ema_model = init_ema_model(denoiser, ema_cfg)
    reflow_teacher = rf_teacher_model if rf_teacher_model is not None else copy.deepcopy(ema_model if ema_model is not None else denoiser).eval()
    teacher_sigma_levels = rf_teacher_sigma_levels if rf_teacher_sigma_levels is not None else sigma_levels
    if reflow_steps > 0:
        ema_model = _train_rf_pair_stage(
            denoiser=denoiser,
            optimizer=optimizer,
            ema_model=ema_model,
            ema_cfg=ema_cfg,
            centers=centers,
            sigma_levels=sigma_levels,
            cfg=cfg,
            history=history,
            sigma_counts=sigma_counts,
            amp_dtype=amp_dtype,
            stage_name="rf_reflow",
            num_steps=reflow_steps,
            global_step_offset=0,
            t_distribution=rf_reflow_t_distribution,
            train_pool=train_pool,
            sample_train_batch_fn=sample_train_batch_fn,
            sample_population_batch_fn=sample_population_batch_fn,
            sample_terminal_batch_fn=sample_terminal_batch_fn,
            reflow_teacher=reflow_teacher,
            reflow_teacher_sigma_levels=teacher_sigma_levels,
        )

    history["sigma_counts"] = [int(v) for v in sigma_counts.detach().cpu().tolist()]
    history["rf_public_baseline"] = "RF"
    history["rf_teacher_family_resolved"] = str(getattr(reflow_teacher, "generative_family", ""))
    history["rf_teacher_pair_n_steps_path_resolved"] = int(max(int(teacher_sigma_levels.numel()) - 1, 0))
    history["rf_teacher_pair_sampling_mode_resolved"] = resolve_rf_teacher_pair_sampling_mode(
        reflow_teacher,
        cfg,
    )
    eval_model = ema_model if ema_model is not None else denoiser
    return history, eval_model


def train_baseline(
    denoiser,
    centers,
    sigma_levels,
    cfg,
    train_pool: torch.Tensor = None,
    sample_train_batch_fn: Optional[Callable[[int], torch.Tensor]] = None,
    sample_population_batch_fn: Optional[Callable[[int], torch.Tensor]] = None,
    sample_terminal_batch_fn: Optional[Callable[[int, float], torch.Tensor]] = None,
    rf_teacher_model=None,
    rf_teacher_sigma_levels: Optional[torch.Tensor] = None,
):
    """Train baseline model theta under the configured generative objective."""

    if _is_strong_rf_baseline(cfg):
        return _train_strong_rf_baseline(
            denoiser,
            centers,
            sigma_levels,
            cfg,
            train_pool=train_pool,
            sample_train_batch_fn=sample_train_batch_fn,
            sample_population_batch_fn=sample_population_batch_fn,
            sample_terminal_batch_fn=sample_terminal_batch_fn,
            rf_teacher_model=rf_teacher_model,
            rf_teacher_sigma_levels=rf_teacher_sigma_levels,
        )

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
            sigma = sample_sigmas_log_normal(
                cfg.batch_size,
                sigma_min=float(cfg.sigma_min),
                sigma_max=float(cfg.sigma_max),
                device=sigma_levels.device,
                p_mean=cfg.p_mean,
                p_std=cfg.p_std,
                dtype=sigma_levels.dtype,
            )
            indices = assign_sigmas_to_nearest_levels(sigma, sigma_levels)
        else:
            indices = sample_target_indices(cfg.batch_size, sigma_levels)
            sigma = sigma_levels[indices]
        sigma_counts += torch.bincount(indices - 1, minlength=sigma_counts.numel())
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
    return states[:, generated_data_path_index_from_denoiser(denoiser)]
