import copy
import math
import time
from typing import Callable, Optional

import torch

from ...app.utils import empty_robust_history
from ...compute_accounting import append_denoiser_op_count_step, ensure_denoiser_op_count_history
from ...shared.ema import init_ema_model, update_ema_model
from ...shared.objective import build_rectified_flow_state, compute_training_loss
from ...shared.runtime import autocast_context, resolve_amp_dtype
from ...shared.sigma import (
    resolve_rf_stage_t_distribution,
    sample_target_indices,
    sample_target_indices_log_normal,
)
from ...shared.trainer_common import generate_reflow_pairs
from ...shared.train_utils import sample_train_batch
from ...utils import batch_scalar_like, has_nan_or_inf, scalarize


def _is_rf_objective(cfg) -> bool:
    return str(getattr(cfg, "training_objective", "edm")).strip().lower() == "rf"


def _rf_ema_cfg(cfg):
    """Use EMA for clean RF continuation even when legacy configs disable it."""

    ema_cfg = copy.copy(cfg)
    ema_cfg.use_ema_eval = True
    return ema_cfg


def _resolve_rf_stage_planning_total_steps(cfg) -> int:
    override = int(getattr(cfg, "rf_continuation_total_steps_override", 0) or 0)
    if override > 0:
        return int(override)
    return int(cfg.steps)


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
        t = torch.sin(0.5 * math.pi * u).square()
    else:
        raise ValueError(
            f"Unsupported RF timestep distribution '{distribution}'. Expected one of: uniform, u_shaped."
        )
    return t.clamp(1e-5, 1.0 - 1e-5)


def _train_trajectory_robust_clean_rf(
    denoiser,
    centers,
    sigma_levels,
    cfg,
    *,
    train_pool: torch.Tensor = None,
    sample_train_batch_fn: Optional[Callable[[int], torch.Tensor]] = None,
    sample_population_batch_fn: Optional[Callable[[int], torch.Tensor]] = None,
    start_step: int = 0,
    history_state: Optional[dict] = None,
    optimizer_theta_state: Optional[dict] = None,
    ema_state_dict: Optional[dict] = None,
    rf_reflow_teacher_state_dict: Optional[dict] = None,
    rf_teacher_model=None,
    rf_teacher_sigma_levels: Optional[torch.Tensor] = None,
    return_state: bool = False,
):
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
    for key in (
        "rf_stage",
        "rf_t_distribution",
        "rf_reflow_pair_fwd_units",
        "rf_step_wall_clock_sec",
    ):
        history.setdefault(key, [])

    cumulative_batch_equiv_evals = (
        float(history["batch_equiv_denoiser_evals_cumulative"][-1])
        if history["batch_equiv_denoiser_evals_cumulative"]
        else 0.0
    )
    amp_dtype = resolve_amp_dtype(sigma_levels.device, getattr(cfg, "amp_dtype", "auto"))
    ema_cfg = _rf_ema_cfg(cfg)
    ema_model = init_ema_model(denoiser, ema_cfg, ema_state_dict=ema_state_dict)

    rf_stage_planning_total_steps = _resolve_rf_stage_planning_total_steps(cfg)
    stage1_steps = 0
    reflow_steps = max(int(rf_stage_planning_total_steps), 0)
    rf_reflow_t_distribution = resolve_rf_stage_t_distribution(
        "rf_reflow",
        reflow_distribution=str(getattr(cfg, "rf_reflow_t_distribution", "u_shaped")),
    )
    history["rf_stage1_steps"] = int(stage1_steps)
    history["rf_reflow_steps"] = int(reflow_steps)
    history["rf_stage_planning_total_steps"] = int(rf_stage_planning_total_steps)
    history["rf_stage1_t_distribution_resolved"] = ""
    history["rf_reflow_t_distribution_resolved"] = str(rf_reflow_t_distribution)
    history["rf_eval_t_distribution_resolved"] = str(rf_reflow_t_distribution)
    history["rf_pair_source_resolved"] = "shared_edm_teacher_reflow"
    teacher_sigma_levels = rf_teacher_sigma_levels if rf_teacher_sigma_levels is not None else sigma_levels

    rf_pair_teacher = None
    if rf_teacher_model is not None:
        rf_pair_teacher = rf_teacher_model.eval()
    elif rf_reflow_teacher_state_dict is not None:
        rf_pair_teacher = copy.deepcopy(denoiser).eval()
        rf_pair_teacher.load_state_dict(rf_reflow_teacher_state_dict, strict=True)
    if rf_pair_teacher is not None:
        history.setdefault("rf_reflow_teacher_refresh_step", 0)
        history["rf_teacher_family_resolved"] = str(getattr(rf_pair_teacher, "generative_family", ""))
        history["rf_teacher_pair_n_steps_path_resolved"] = int(max(int(teacher_sigma_levels.numel()) - 1, 0))

    for step in range(int(start_step) + 1, int(cfg.steps) + 1):
        step_t0 = time.perf_counter()
        reflow_pair_fwd_units = 0.0
        x_right = sample_train_batch(
            cfg,
            centers,
            train_pool=train_pool,
            sample_train_batch_fn=sample_train_batch_fn,
            sample_population_batch_fn=sample_population_batch_fn,
        )
        if rf_pair_teacher is None:
            teacher_source = ema_model if ema_model is not None else denoiser
            rf_pair_teacher = copy.deepcopy(teacher_source).eval()
            history.setdefault("rf_reflow_teacher_refresh_step", max(int(step - 1), 0))
            history["rf_teacher_family_resolved"] = str(getattr(rf_pair_teacher, "generative_family", ""))
            history["rf_teacher_pair_n_steps_path_resolved"] = int(max(int(teacher_sigma_levels.numel()) - 1, 0))
        x_left, x_right = generate_reflow_pairs(
            rf_pair_teacher,
            teacher_sigma_levels,
            x_right,
        )
        reflow_pair_fwd_units = float(max(int(teacher_sigma_levels.numel()) - 1, 0))
        t_distribution = rf_reflow_t_distribution
        stage_name = "rf_reflow"

        t = _sample_rf_t(
            int(x_right.shape[0]),
            device=sigma_levels.device,
            dtype=sigma_levels.dtype,
            distribution=t_distribution,
        )
        sigma = t * float(getattr(cfg, "sigma_max", float(sigma_levels[-1].item())))
        x_state = build_rectified_flow_state(x_left, x_right, t)

        optimizer_theta.zero_grad(set_to_none=True)
        with autocast_context(sigma_levels.device, amp_dtype):
            loss = compute_training_loss(
                cfg,
                denoiser,
                x_state,
                x_right,
                sigma,
                x_left=x_left,
                x_right=x_right,
            )
        if has_nan_or_inf(loss):
            raise RuntimeError("NaN/Inf detected in clean RF continuation loss.")
        loss.backward()
        optimizer_theta.step()

        update_ema_model(
            ema_model,
            denoiser,
            ema_cfg,
            cur_nimg=float((step - 1) * int(cfg.batch_size)),
            batch_size=int(cfg.batch_size),
        )

        step_batch_equiv = 1.0 + reflow_pair_fwd_units
        cumulative_batch_equiv_evals += step_batch_equiv
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
        history["batch_equiv_denoiser_evals_step"].append(float(step_batch_equiv))
        history["batch_equiv_denoiser_evals_attack_construction"].append(0.0)
        history["batch_equiv_denoiser_evals_attack_eval"].append(0.0)
        history["batch_equiv_denoiser_evals_clean_eval"].append(1.0)
        history["batch_equiv_denoiser_evals_cumulative"].append(float(cumulative_batch_equiv_evals))
        history["rf_stage"].append(str(stage_name))
        history["rf_t_distribution"].append(str(t_distribution))
        history["rf_reflow_pair_fwd_units"].append(float(reflow_pair_fwd_units))
        history["rf_step_wall_clock_sec"].append(float(time.perf_counter() - step_t0))
        append_denoiser_op_count_step(
            history,
            n_fwd=float(reflow_pair_fwd_units),
            n_fwd_inputgrad=0.0,
            n_fwd_parambackward=1.0,
        )

        if step % cfg.log_every == 0:
            print(
                f"[clean-rf] step={step:05d} stage={stage_name} loss={loss.item():.6f} "
                f"reflow_fwd={reflow_pair_fwd_units:.1f} "
                f"be_evals={step_batch_equiv:.1f} be_evals_cum={cumulative_batch_equiv_evals:.1f}",
                flush=True,
            )

    trainer_state = None
    if return_state:
        resume_model_state = copy.deepcopy(denoiser.state_dict())
        trainer_state = {
            "completed_steps": int(cfg.steps),
            "optimizer_theta_state": optimizer_theta.state_dict(),
            "ema_state_dict": None if ema_model is None else copy.deepcopy(ema_model.state_dict()),
            "rf_teacher_state_dict": None if rf_pair_teacher is None else copy.deepcopy(rf_pair_teacher.state_dict()),
            "rf_teacher_training_objective": (
                "rf"
                if rf_pair_teacher is None
                else (
                    "rf"
                    if str(getattr(rf_pair_teacher, "generative_family", "")).lower() == "rectified_flow"
                    else "edm"
                )
            ),
            "resume_robust_state_dict": resume_model_state,
        }

    if ema_model is not None:
        denoiser.load_state_dict(ema_model.state_dict(), strict=True)

    if return_state:
        return history, trainer_state
    return history


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
    rf_reflow_teacher_state_dict: Optional[dict] = None,
    rf_teacher_model=None,
    rf_teacher_sigma_levels: Optional[torch.Tensor] = None,
    return_state: bool = False,
):
    """Continue the baseline EDM objective from an initialized checkpoint."""

    del control
    if _is_rf_objective(cfg):
        return _train_trajectory_robust_clean_rf(
            denoiser,
            centers,
            sigma_levels,
            cfg,
            train_pool=train_pool,
            sample_train_batch_fn=sample_train_batch_fn,
            sample_population_batch_fn=sample_population_batch_fn,
            start_step=start_step,
            history_state=history_state,
            optimizer_theta_state=optimizer_theta_state,
            ema_state_dict=ema_state_dict,
            rf_reflow_teacher_state_dict=rf_reflow_teacher_state_dict,
            rf_teacher_model=rf_teacher_model,
            rf_teacher_sigma_levels=rf_teacher_sigma_levels,
            return_state=return_state,
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

    ema_model = init_ema_model(denoiser, cfg, ema_state_dict=ema_state_dict)

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

        update_ema_model(
            ema_model,
            denoiser,
            cfg,
            cur_nimg=float((step - 1) * int(cfg.batch_size)),
            batch_size=int(cfg.batch_size),
        )

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
