import copy
import math
import time
from typing import Callable, Dict, Optional, Tuple

import torch

from ...app.utils import empty_robust_history
from ...compute_accounting import append_denoiser_op_count_step, ensure_denoiser_op_count_history
from ...models import set_requires_grad
from ...shared.ema import init_ema_model, update_ema_model
from ...shared.objective import build_rectified_flow_state, compute_training_loss
from ...shared.runtime import autocast_context, resolve_amp_dtype
from ...shared.sigma import resolve_rf_stage_t_distribution, sample_target_indices, sample_target_indices_log_normal
from ...shared.trainer_common import (
    generate_reflow_pairs,
    resolve_rf_teacher_pair_sampling_mode,
    resolve_rf_teacher_ve_sampler_mode,
)
from ...utils import batch_scalar_like, has_nan_or_inf, scalarize


def _sample_sigma_indices(batch_size: int, sigma_levels: torch.Tensor, cfg) -> torch.Tensor:
    if cfg.use_log_normal_sigma_sampling:
        return sample_target_indices_log_normal(
            batch_size,
            sigma_levels,
            p_mean=cfg.p_mean,
            p_std=cfg.p_std,
        )
    return sample_target_indices(batch_size, sigma_levels)


def _transport_cost(x_adv: torch.Tensor, x_clean: torch.Tensor) -> torch.Tensor:
    return (x_adv - x_clean).reshape(x_clean.shape[0], -1).norm(p=2, dim=1).mean()


def _is_rf_objective(cfg) -> bool:
    return str(getattr(cfg, "training_objective", "edm")).strip().lower() == "rf"


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


def _rf_teacher_pair_sampling_mode(cfg, teacher) -> str:
    return resolve_rf_teacher_pair_sampling_mode(teacher, cfg)


def _wdro_attack_batch(
    denoiser,
    x_clean: torch.Tensor,
    sigma_levels: torch.Tensor,
    cfg,
    amp_dtype,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    attack_steps = max(int(cfg.wdro_attack_steps), 0)
    if attack_steps == 0:
        return x_clean.detach().clone(), {"attack_loss": 0.0, "transport_cost": 0.0}

    gamma = float(cfg.wdro_gamma)
    step_size = float(cfg.wdro_attack_step_size)
    x_adv = x_clean.detach().clone()
    last_attack_loss = torch.zeros((), device=x_clean.device, dtype=x_clean.dtype)
    last_transport = torch.zeros((), device=x_clean.device, dtype=x_clean.dtype)

    for _ in range(attack_steps):
        x_adv.requires_grad_(True)
        with autocast_context(x_clean.device, amp_dtype):
            if _is_rf_objective(cfg):
                t_distribution = resolve_rf_stage_t_distribution(
                    "rf_stage1",
                    reflow_distribution=str(getattr(cfg, "rf_reflow_t_distribution", "u_shaped")),
                )
                t = _sample_rf_t(
                    x_adv.shape[0],
                    device=x_adv.device,
                    dtype=sigma_levels.dtype,
                    distribution=t_distribution,
                )
                sigma = t * float(getattr(cfg, "sigma_max", float(sigma_levels[-1].item())))
                x_left = torch.randn_like(x_adv)
                x_state = build_rectified_flow_state(x_left, x_adv, t)
                attack_loss = compute_training_loss(
                    cfg,
                    denoiser,
                    x_state,
                    x_adv,
                    sigma,
                    x_left=x_left,
                    x_right=x_adv,
                )
            else:
                sigma_idx = _sample_sigma_indices(x_adv.shape[0], sigma_levels, cfg)
                sigma = sigma_levels[sigma_idx]
                x_state = x_adv + batch_scalar_like(sigma, x_adv) * torch.randn_like(x_adv)
                attack_loss = compute_training_loss(cfg, denoiser, x_state, x_adv, sigma)
            transport = _transport_cost(x_adv, x_clean)
            objective = attack_loss - gamma * transport
        if has_nan_or_inf(objective):
            raise RuntimeError("NaN/Inf detected in WDRO attack objective.")
        grad_x = torch.autograd.grad(objective, x_adv, only_inputs=True)[0]
        with torch.no_grad():
            x_adv = x_adv + step_size * grad_x
            if bool(cfg.wdro_clamp_samples):
                x_adv.clamp_(float(cfg.wdro_sample_min), float(cfg.wdro_sample_max))
        x_adv = x_adv.detach()
        last_attack_loss = attack_loss.detach()
        last_transport = transport.detach()

    return x_adv, {
        "attack_loss": scalarize(last_attack_loss),
        "transport_cost": scalarize(last_transport),
    }


def _wdro_attack_rf_pair_batch(
    denoiser,
    x_left: torch.Tensor,
    x_clean: torch.Tensor,
    sigma_levels: torch.Tensor,
    cfg,
    amp_dtype,
    *,
    t_distribution: str,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    attack_steps = max(int(cfg.wdro_attack_steps), 0)
    if attack_steps == 0:
        return x_clean.detach().clone(), {"attack_loss": 0.0, "transport_cost": 0.0}

    gamma = float(cfg.wdro_gamma)
    step_size = float(cfg.wdro_attack_step_size)
    x_adv = x_clean.detach().clone()
    last_attack_loss = torch.zeros((), device=x_clean.device, dtype=x_clean.dtype)
    last_transport = torch.zeros((), device=x_clean.device, dtype=x_clean.dtype)

    for _ in range(attack_steps):
        x_adv.requires_grad_(True)
        with autocast_context(x_clean.device, amp_dtype):
            t = _sample_rf_t(
                x_adv.shape[0],
                device=x_adv.device,
                dtype=sigma_levels.dtype,
                distribution=t_distribution,
            )
            sigma = t * float(getattr(cfg, "sigma_max", float(sigma_levels[-1].item())))
            x_state = build_rectified_flow_state(x_left, x_adv, t)
            attack_loss = compute_training_loss(
                cfg,
                denoiser,
                x_state,
                x_adv,
                sigma,
                x_left=x_left,
                x_right=x_adv,
            )
            transport = _transport_cost(x_adv, x_clean)
            objective = attack_loss - gamma * transport
        if has_nan_or_inf(objective):
            raise RuntimeError("NaN/Inf detected in WDRO-RF pair attack objective.")
        grad_x = torch.autograd.grad(objective, x_adv, only_inputs=True)[0]
        with torch.no_grad():
            x_adv = x_adv + step_size * grad_x
            if bool(cfg.wdro_clamp_samples):
                x_adv.clamp_(float(cfg.wdro_sample_min), float(cfg.wdro_sample_max))
        x_adv = x_adv.detach()
        last_attack_loss = attack_loss.detach()
        last_transport = transport.detach()

    return x_adv, {
        "attack_loss": scalarize(last_attack_loss),
        "transport_cost": scalarize(last_transport),
    }


def _build_combined_pool(
    denoiser,
    train_pool: torch.Tensor,
    sigma_levels: torch.Tensor,
    cfg,
    amp_dtype,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    batch_size = max(int(cfg.batch_size), 1)
    chunks = []
    attack_batches = 0
    adv_examples = 0
    attack_loss_vals = []
    transport_vals = []

    set_requires_grad(denoiser, False)
    for start in range(0, int(train_pool.shape[0]), batch_size):
        x_batch = train_pool[start : start + batch_size]
        chunks.append(x_batch.detach())
        if torch.rand((), device=x_batch.device).item() < float(cfg.wdro_adv_prob):
            x_adv, stats = _wdro_attack_batch(denoiser, x_batch, sigma_levels, cfg, amp_dtype)
            chunks.append(x_adv.detach())
            attack_batches += 1
            adv_examples += int(x_adv.shape[0])
            attack_loss_vals.append(float(stats["attack_loss"]))
            transport_vals.append(float(stats["transport_cost"]))
    set_requires_grad(denoiser, True)

    combined = torch.cat(chunks, dim=0)
    return combined, {
        "attack_batches": int(attack_batches),
        "adv_examples": int(adv_examples),
        "dataset_size": int(combined.shape[0]),
        "attack_loss_mean": float(sum(attack_loss_vals) / len(attack_loss_vals)) if attack_loss_vals else 0.0,
        "transport_cost_mean": float(sum(transport_vals) / len(transport_vals)) if transport_vals else 0.0,
        "attack_construction_units": float(attack_batches * max(int(cfg.wdro_attack_steps), 0)),
    }


def _sample_train_like_batch(
    *,
    train_pool: torch.Tensor,
    batch_size: int,
    sample_train_batch_fn: Optional[Callable[[int], torch.Tensor]],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if callable(sample_train_batch_fn):
        batch = sample_train_batch_fn(int(batch_size))
        return batch.to(device=device, dtype=dtype)
    idx = torch.randint(0, int(train_pool.shape[0]), (int(batch_size),), device=device)
    return train_pool[idx].to(device=device, dtype=dtype)


def _train_trajectory_robust_wdro_rf(
    denoiser,
    sigma_levels,
    cfg,
    *,
    train_pool: torch.Tensor,
    sample_train_batch_fn: Optional[Callable[[int], torch.Tensor]] = None,
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
        "wdro_refresh_step",
        "wdro_dataset_size",
        "wdro_adv_examples",
        "wdro_attack_batches",
        "wdro_attack_loss",
        "wdro_transport_cost",
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
    ema_model = init_ema_model(denoiser, cfg, ema_state_dict=ema_state_dict)

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
        set_requires_grad(rf_pair_teacher, False)
    elif rf_reflow_teacher_state_dict is not None:
        rf_pair_teacher = copy.deepcopy(denoiser).eval()
        rf_pair_teacher.load_state_dict(rf_reflow_teacher_state_dict, strict=True)
        set_requires_grad(rf_pair_teacher, False)
    if rf_pair_teacher is not None:
        history.setdefault("rf_reflow_teacher_refresh_step", 0)
        history["rf_teacher_family_resolved"] = str(getattr(rf_pair_teacher, "generative_family", ""))
        history["rf_teacher_pair_n_steps_path_resolved"] = int(max(int(teacher_sigma_levels.numel()) - 1, 0))
        history["rf_teacher_pair_sampling_mode_resolved"] = _rf_teacher_pair_sampling_mode(cfg, rf_pair_teacher)

    for step in range(int(start_step) + 1, int(cfg.steps) + 1):
        step_t0 = time.perf_counter()
        reflow_pair_fwd_units = 0.0
        if rf_pair_teacher is None:
            teacher_source = ema_model if ema_model is not None else denoiser
            rf_pair_teacher = copy.deepcopy(teacher_source).eval()
            set_requires_grad(rf_pair_teacher, False)
            history.setdefault("rf_reflow_teacher_refresh_step", max(int(step - 1), 0))
            history["rf_teacher_family_resolved"] = str(getattr(rf_pair_teacher, "generative_family", ""))
            history["rf_teacher_pair_n_steps_path_resolved"] = int(max(int(teacher_sigma_levels.numel()) - 1, 0))
            history["rf_teacher_pair_sampling_mode_resolved"] = _rf_teacher_pair_sampling_mode(cfg, rf_pair_teacher)
        x_template = _sample_train_like_batch(
            train_pool=train_pool,
            batch_size=int(cfg.batch_size),
            sample_train_batch_fn=sample_train_batch_fn,
            device=sigma_levels.device,
            dtype=sigma_levels.dtype,
        )
        x_left, x_right_clean = generate_reflow_pairs(
            rf_pair_teacher,
            teacher_sigma_levels,
            x_template,
            ve_sampler_mode=resolve_rf_teacher_ve_sampler_mode(cfg),
        )
        reflow_pair_fwd_units = float(max(int(teacher_sigma_levels.numel()) - 1, 0))
        attack_applied = bool(
            int(cfg.wdro_attack_steps) > 0 and torch.rand((), device=x_template.device).item() < float(cfg.wdro_adv_prob)
        )
        attack_stats = {"attack_loss": 0.0, "transport_cost": 0.0}
        attack_construction_units = 0.0
        if attack_applied:
            x_right, attack_stats = _wdro_attack_rf_pair_batch(
                denoiser,
                x_left,
                x_right_clean,
                sigma_levels,
                cfg,
                amp_dtype,
                t_distribution=rf_reflow_t_distribution,
            )
            attack_construction_units = float(max(int(cfg.wdro_attack_steps), 0))
        else:
            x_right = x_right_clean
        history["wdro_refresh_step"].append(int(step))
        history["wdro_dataset_size"].append(int(x_right.shape[0]))
        history["wdro_adv_examples"].append(int(x_right.shape[0]) if attack_applied else 0)
        history["wdro_attack_batches"].append(int(attack_applied))
        history["wdro_attack_loss"].append(float(attack_stats["attack_loss"]))
        history["wdro_transport_cost"].append(float(attack_stats["transport_cost"]))
        t_distribution = rf_reflow_t_distribution
        stage_name = "rf_reflow"
        stage_energy = float(attack_stats["transport_cost"])
        stage_dual_surrogate = float(attack_stats["attack_loss"])

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
            raise RuntimeError("NaN/Inf detected in WDRO-RF outer loss.")
        loss.backward()
        optimizer_theta.step()

        update_ema_model(
            ema_model,
            denoiser,
            cfg,
            cur_nimg=float((step - 1) * int(cfg.batch_size)),
            batch_size=int(cfg.batch_size),
        )

        step_batch_equiv = 1.0 + attack_construction_units + reflow_pair_fwd_units
        cumulative_batch_equiv_evals += step_batch_equiv
        loss_scalar = scalarize(loss)
        history["outer_loss"].append(loss_scalar)
        history["outer_loss_attack"].append(0.0)
        history["outer_loss_clean"].append(loss_scalar)
        history["inner_obj"].append(0.0)
        history["energy"].append(float(stage_energy))
        history["lambda_value"].append(float(cfg.wdro_gamma))
        history["lambda_value_next"].append(float(cfg.wdro_gamma))
        history["lambda_subgrad"].append(0.0)
        history["dual_surrogate"].append(float(stage_dual_surrogate))
        history["delta_norm_mean"].append(0.0)
        history["delta_norm_max"].append(0.0)
        history["delta_norm_ratio_mean"].append(0.0)
        history["delta_norm_ratio_max"].append(0.0)
        history["sched_attack_weight"].append(1.0)
        history["sched_clean_weight"].append(1.0)
        history["sched_phi_lr_scale"].append(0.0)
        history["batch_equiv_denoiser_evals_step"].append(float(step_batch_equiv))
        history["batch_equiv_denoiser_evals_attack_construction"].append(float(attack_construction_units))
        history["batch_equiv_denoiser_evals_attack_eval"].append(0.0)
        history["batch_equiv_denoiser_evals_clean_eval"].append(1.0)
        history["rf_stage"].append(str(stage_name))
        history["rf_t_distribution"].append(str(t_distribution))
        history["rf_reflow_pair_fwd_units"].append(float(reflow_pair_fwd_units))
        history["rf_step_wall_clock_sec"].append(float(time.perf_counter() - step_t0))
        history["batch_equiv_denoiser_evals_cumulative"].append(float(cumulative_batch_equiv_evals))
        append_denoiser_op_count_step(
            history,
            n_fwd=float(reflow_pair_fwd_units),
            n_fwd_inputgrad=float(attack_construction_units),
            n_fwd_parambackward=1.0,
        )

        if step % int(cfg.log_every) == 0:
            print(
                f"[wdro-rf] step={step:05d} stage={stage_name} loss={loss.item():.6f} "
                f"adv_examples={int(x_right.shape[0]) if attack_applied else 0} "
                f"reflow_fwd={reflow_pair_fwd_units:.1f} "
                f"be_evals={step_batch_equiv:.1f} be_evals_cum={cumulative_batch_equiv_evals:.1f}",
                flush=True,
            )

    trainer_state = None
    if return_state:
        resume_robust_state_dict = copy.deepcopy(denoiser.state_dict())
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
            "resume_robust_state_dict": resume_robust_state_dict,
        }

    if ema_model is not None:
        denoiser.load_state_dict(ema_model.state_dict(), strict=True)

    if return_state:
        return history, trainer_state
    return history


def train_trajectory_robust_wdro(
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
    del control, centers, sample_population_batch_fn
    if train_pool is None:
        raise RuntimeError("WDRO trainer requires a finite train_pool so it can rebuild the augmented dataset.")
    if train_pool.device != sigma_levels.device:
        train_pool = train_pool.to(device=sigma_levels.device)
    if _is_rf_objective(cfg):
        return _train_trajectory_robust_wdro_rf(
            denoiser,
            sigma_levels,
            cfg,
            train_pool=train_pool,
            sample_train_batch_fn=sample_train_batch_fn,
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
    for key in (
        "wdro_refresh_step",
        "wdro_dataset_size",
        "wdro_adv_examples",
        "wdro_attack_batches",
        "wdro_attack_loss",
        "wdro_transport_cost",
    ):
        history.setdefault(key, [])

    cumulative_batch_equiv_evals = (
        float(history["batch_equiv_denoiser_evals_cumulative"][-1])
        if history["batch_equiv_denoiser_evals_cumulative"]
        else 0.0
    )
    amp_dtype = resolve_amp_dtype(sigma_levels.device, getattr(cfg, "amp_dtype", "auto"))

    ema_model = init_ema_model(denoiser, cfg, ema_state_dict=ema_state_dict)

    refresh_interval_steps = max(
        int(math.ceil(float(cfg.wdro_refresh_epochs) * float(train_pool.shape[0]) / float(max(int(cfg.batch_size), 1)))),
        1,
    )
    combined_pool = None
    combined_stats = {
        "attack_batches": 0,
        "adv_examples": 0,
        "dataset_size": int(train_pool.shape[0]),
        "attack_loss_mean": 0.0,
        "transport_cost_mean": 0.0,
        "attack_construction_units": 0.0,
    }

    for step in range(int(start_step) + 1, int(cfg.steps) + 1):
        refresh_now = combined_pool is None or step == 1 or ((step - 1) % refresh_interval_steps == 0)
        refresh_attack_construction_units = 0.0
        if refresh_now:
            combined_pool, combined_stats = _build_combined_pool(
                denoiser=denoiser,
                train_pool=train_pool,
                sigma_levels=sigma_levels,
                cfg=cfg,
                amp_dtype=amp_dtype,
            )
            refresh_attack_construction_units = float(combined_stats["attack_construction_units"])
            history["wdro_refresh_step"].append(int(step))
            history["wdro_dataset_size"].append(int(combined_stats["dataset_size"]))
            history["wdro_adv_examples"].append(int(combined_stats["adv_examples"]))
            history["wdro_attack_batches"].append(int(combined_stats["attack_batches"]))
            history["wdro_attack_loss"].append(float(combined_stats["attack_loss_mean"]))
            history["wdro_transport_cost"].append(float(combined_stats["transport_cost_mean"]))

        idx = torch.randint(0, int(combined_pool.shape[0]), (int(cfg.batch_size),), device=combined_pool.device)
        x0 = combined_pool[idx]
        sigma_idx = _sample_sigma_indices(int(cfg.batch_size), sigma_levels, cfg)
        sigma = sigma_levels[sigma_idx]
        x_noisy = x0 + batch_scalar_like(sigma, x0) * torch.randn_like(x0)

        optimizer_theta.zero_grad(set_to_none=True)
        with autocast_context(sigma_levels.device, amp_dtype):
            loss = compute_training_loss(cfg, denoiser, x_noisy, x0, sigma)
        if has_nan_or_inf(loss):
            raise RuntimeError("NaN/Inf detected in WDRO outer loss.")
        loss.backward()
        optimizer_theta.step()

        update_ema_model(
            ema_model,
            denoiser,
            cfg,
            cur_nimg=float((step - 1) * int(cfg.batch_size)),
            batch_size=int(cfg.batch_size),
        )

        step_batch_equiv = 1.0 + refresh_attack_construction_units
        cumulative_batch_equiv_evals += step_batch_equiv
        loss_scalar = scalarize(loss)
        history["outer_loss"].append(loss_scalar)
        history["outer_loss_attack"].append(0.0)
        history["outer_loss_clean"].append(loss_scalar)
        history["inner_obj"].append(0.0)
        history["energy"].append(float(combined_stats["transport_cost_mean"]))
        history["lambda_value"].append(float(cfg.wdro_gamma))
        history["lambda_value_next"].append(float(cfg.wdro_gamma))
        history["lambda_subgrad"].append(0.0)
        history["dual_surrogate"].append(float(combined_stats["attack_loss_mean"]))
        history["delta_norm_mean"].append(0.0)
        history["delta_norm_max"].append(0.0)
        history["delta_norm_ratio_mean"].append(0.0)
        history["delta_norm_ratio_max"].append(0.0)
        history["sched_attack_weight"].append(1.0)
        history["sched_clean_weight"].append(1.0)
        history["sched_phi_lr_scale"].append(0.0)
        history["batch_equiv_denoiser_evals_step"].append(float(step_batch_equiv))
        history["batch_equiv_denoiser_evals_attack_construction"].append(float(refresh_attack_construction_units))
        history["batch_equiv_denoiser_evals_attack_eval"].append(0.0)
        history["batch_equiv_denoiser_evals_clean_eval"].append(1.0)
        history["batch_equiv_denoiser_evals_cumulative"].append(float(cumulative_batch_equiv_evals))
        append_denoiser_op_count_step(
            history,
            n_fwd=0.0,
            n_fwd_inputgrad=float(refresh_attack_construction_units),
            n_fwd_parambackward=1.0,
        )

        if step % int(cfg.log_every) == 0:
            print(
                f"[wdro] step={step:05d} loss={loss.item():.6f} "
                f"pool={int(combined_stats['dataset_size'])} adv_examples={int(combined_stats['adv_examples'])} "
                f"be_evals={step_batch_equiv:.1f} be_evals_cum={cumulative_batch_equiv_evals:.1f}",
                flush=True,
            )

    trainer_state = None
    if return_state:
        resume_robust_state_dict = copy.deepcopy(denoiser.state_dict())
        trainer_state = {
            "completed_steps": int(cfg.steps),
            "optimizer_theta_state": optimizer_theta.state_dict(),
            "ema_state_dict": None if ema_model is None else copy.deepcopy(ema_model.state_dict()),
            "resume_robust_state_dict": resume_robust_state_dict,
        }

    if ema_model is not None:
        denoiser.load_state_dict(ema_model.state_dict(), strict=True)

    if return_state:
        return history, trainer_state
    return history
