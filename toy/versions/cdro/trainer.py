import copy
import time
from typing import Callable, Optional

import torch

from ...compute_accounting import append_denoiser_op_count_step, ensure_denoiser_op_count_history
from ...models import set_requires_grad
from ...shared.ema import init_ema_model, update_ema_model
from ...shared.objective import compute_training_loss, inner_objective_attack_only
from ...shared.rf_stage import resolve_rf_cdro_stage_steps
from ...shared.runtime import autocast_context, resolve_amp_dtype
from ...shared.trainer_common import (
    generate_reflow_pairs,
    resolve_rf_teacher_pair_sampling_mode,
    resolve_rf_teacher_ve_sampler_mode,
)
from ...shared.sigma import (
    build_rf_stage_time_quantile_levels,
    resolve_rf_stage_t_distribution,
    resolve_rf_teacher_n_steps_path,
    sample_log_sigma_stratified_quantile_ladder,
    sample_log_sigma_stratified_quantile_ladder_batch,
    sample_rf_stage_time_stratified_levels_batch,
    sample_rf_stage_time_stratified_levels,
    sample_target_indices,
    sample_target_indices_log_normal,
)
from ...shared.train_utils import pathwise_l2, sample_train_batch
from ...utils import has_nan_or_inf, scalarize
from .diffusion import (
    STOCHASTIC_STRATIFIED_QUANTILE_LADDER,
    build_constraint_radii_for_objective,
    build_transition_deltas_for_objective,
    resolve_cdro_edm_ladder_mode,
    rollout_controlled_ve,
    rollout_path_heuristic_attack,
)


def _control_transport_cost(control_path: torch.Tensor, transition_deltas: torch.Tensor) -> torch.Tensor:
    """Route-A discrete control cost: E[sum_k Delta_tau_k ||u_k||^2]."""

    flat = control_path.reshape(control_path.shape[0], control_path.shape[1], -1)
    control_sq = flat.pow(2).sum(dim=2)
    if transition_deltas.ndim == 1:
        dt = transition_deltas.view(1, -1).to(device=control_sq.device, dtype=control_sq.dtype)
    elif transition_deltas.ndim == 2:
        if transition_deltas.shape != control_sq.shape:
            raise ValueError(
                "Per-example transition_deltas must match control_path step shape, got "
                f"{tuple(transition_deltas.shape)} vs {tuple(control_sq.shape)}"
            )
        dt = transition_deltas.to(device=control_sq.device, dtype=control_sq.dtype)
    else:
        raise ValueError(f"transition_deltas must be rank 1 or 2, got shape={tuple(transition_deltas.shape)}")
    return (control_sq * dt).sum(dim=1).mean()


def _is_rf_objective(cfg) -> bool:
    return str(getattr(cfg, "training_objective", "edm")).strip().lower() == "rf"


def _sample_cdro_edm_step_ladder(cfg, sigma_levels: torch.Tensor, *, batch_size: int) -> torch.Tensor:
    """Sample one CDRO-EDM stratified ladder with the same support and step count."""

    ladder_kwargs = {
        "sigma_min": float(getattr(cfg, "sigma_min", 0.0)),
        "sigma_max": float(getattr(cfg, "sigma_max", 0.0)),
        "n_steps": int(sigma_levels.shape[-1] - 1),
        "p_mean": float(getattr(cfg, "p_mean", -1.2)),
        "p_std": float(getattr(cfg, "p_std", 1.2)),
        "device": sigma_levels.device,
        "dtype": sigma_levels.dtype,
    }
    if bool(getattr(cfg, "cdro_per_example_sigma_ladders", True)):
        ladder = sample_log_sigma_stratified_quantile_ladder_batch(
            batch_size=int(batch_size),
            **ladder_kwargs,
        )
    else:
        ladder = sample_log_sigma_stratified_quantile_ladder(**ladder_kwargs)
    return ladder.sigma_levels


def _snapshot_tensor_values(value: torch.Tensor) -> list[float]:
    snapshot = value.detach().cpu()
    if snapshot.ndim > 1:
        snapshot = snapshot[0]
    return [float(v.item()) for v in snapshot.reshape(-1)]


def _rf_teacher_pair_sampling_mode(cfg, teacher) -> str:
    return resolve_rf_teacher_pair_sampling_mode(teacher, cfg)


def _path_n_steps(sigma_levels: torch.Tensor) -> int:
    if sigma_levels.ndim not in (1, 2):
        raise ValueError(f"sigma_levels must be rank 1 or 2, got shape={tuple(sigma_levels.shape)}")
    return int(sigma_levels.shape[-1] - 1)


def _path_step_sigma_batch(sigma_levels: torch.Tensor, step_idx: int, x0: torch.Tensor) -> torch.Tensor:
    if sigma_levels.ndim == 1:
        return torch.full(
            (x0.shape[0],),
            float(sigma_levels[step_idx].item()),
            device=x0.device,
            dtype=x0.dtype,
        )
    if sigma_levels.ndim == 2:
        return sigma_levels[:, step_idx].to(device=x0.device, dtype=x0.dtype)
    raise ValueError(f"sigma_levels must be rank 1 or 2, got shape={tuple(sigma_levels.shape)}")


def _resolve_attack_num_steps(cfg) -> tuple[int, str]:
    """Resolve the CDRO inner attack count with legacy `inner_steps` fallback."""

    explicit = getattr(cfg, "attack_num_steps", None)
    if explicit is not None:
        resolved = int(explicit)
        if resolved not in (1, 2):
            raise ValueError(f"attack_num_steps must be one of (1, 2), got {explicit}")
        return resolved, "attack_num_steps"
    legacy = int(getattr(cfg, "inner_steps", 1))
    if legacy <= 0:
        return 0, "inner_steps_legacy"
    return legacy, "inner_steps_legacy"


def _resolve_rf_cdro_pair_source(cfg) -> str:
    mode = str(getattr(cfg, "rf_cdro_pair_source", "auto")).strip().lower()
    if mode in ("auto", "staged"):
        return "reflow"
    if mode not in ("reflow", "data_noise"):
        raise ValueError(f"Unsupported rf_cdro_pair_source='{mode}'.")
    return mode


def _resolve_rf_stage_planning_total_steps(cfg) -> int:
    override = int(getattr(cfg, "rf_continuation_total_steps_override", 0) or 0)
    if override > 0:
        return int(override)
    return int(cfg.steps)


def _build_rf_cdro_stage_grid_info(
    cfg,
    *,
    device: torch.device,
    dtype: torch.dtype,
    time_horizon: float,
    stage_name: str,
) -> dict:
    """Build the deterministic RF solver/reference grid for one robust stage."""

    distribution = resolve_rf_stage_t_distribution(
        stage_name,
        reflow_distribution=str(getattr(cfg, "rf_reflow_t_distribution", "u_shaped")),
    )
    sigma_levels = build_rf_stage_time_quantile_levels(
        float(getattr(cfg, "sigma_max", 1.0)),
        int(resolve_rf_teacher_n_steps_path(cfg)),
        device=device,
        stage_name=stage_name,
        reflow_distribution=distribution,
    ).to(device=device, dtype=dtype)
    transition_deltas = build_transition_deltas_for_objective(
        cfg=cfg,
        sigma_levels=sigma_levels,
        time_horizon=time_horizon,
    ).to(device=device, dtype=dtype)
    radius_by_step = build_constraint_radii_for_objective(
        cfg=cfg,
        sigma_levels=sigma_levels,
        total_budget=float(getattr(cfg, "cdro_total_budget_rho", 0.0)),
        time_horizon=time_horizon,
    ).to(device=device, dtype=dtype)
    return {
        "stage_name": str(stage_name),
        "distribution": str(distribution),
        "sigma_levels": sigma_levels,
        "transition_deltas": transition_deltas,
        "radius_by_step": radius_by_step,
    }


def _sample_rf_cdro_stage_grid_info(
    cfg,
    *,
    device: torch.device,
    dtype: torch.dtype,
    time_horizon: float,
    stage_name: str,
    batch_size: int,
) -> dict:
    """Sample one stratified RF rollout grid aligned to the clean stage law."""

    distribution = resolve_rf_stage_t_distribution(
        stage_name,
        reflow_distribution=str(getattr(cfg, "rf_reflow_t_distribution", "u_shaped")),
    )
    if bool(getattr(cfg, "cdro_per_example_sigma_ladders", True)):
        sigma_levels = sample_rf_stage_time_stratified_levels_batch(
            float(getattr(cfg, "sigma_max", 1.0)),
            int(resolve_rf_teacher_n_steps_path(cfg)),
            device=device,
            batch_size=int(batch_size),
            stage_name=stage_name,
            reflow_distribution=distribution,
            dtype=dtype,
        ).to(device=device, dtype=dtype)
    else:
        sigma_levels = sample_rf_stage_time_stratified_levels(
            float(getattr(cfg, "sigma_max", 1.0)),
            int(resolve_rf_teacher_n_steps_path(cfg)),
            device=device,
            stage_name=stage_name,
            reflow_distribution=distribution,
            dtype=dtype,
        ).to(device=device, dtype=dtype)
    transition_deltas = build_transition_deltas_for_objective(
        cfg=cfg,
        sigma_levels=sigma_levels,
        time_horizon=time_horizon,
    ).to(device=device, dtype=dtype)
    radius_by_step = build_constraint_radii_for_objective(
        cfg=cfg,
        sigma_levels=sigma_levels,
        total_budget=float(getattr(cfg, "cdro_total_budget_rho", 0.0)),
        time_horizon=time_horizon,
    ).to(device=device, dtype=dtype)
    return {
        "stage_name": str(stage_name),
        "distribution": str(distribution),
        "sigma_levels": sigma_levels,
        "transition_deltas": transition_deltas,
        "radius_by_step": radius_by_step,
    }


def _path_average_training_loss(
    cfg,
    denoiser,
    states: torch.Tensor,
    x0: torch.Tensor,
    sigma_levels: torch.Tensor,
    *,
    x_left: Optional[torch.Tensor] = None,
    x_right: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Average weighted denoise loss over all rollout timesteps k=1..N."""

    n_steps = _path_n_steps(sigma_levels)
    if states.shape[1] != n_steps + 1:
        raise ValueError(f"states step dim must be {n_steps + 1}, got {states.shape[1]}")

    total_loss = None
    for step_idx in range(n_steps):
        sigma = _path_step_sigma_batch(sigma_levels, step_idx + 1, x0)
        loss_step = compute_training_loss(
            cfg,
            denoiser,
            states[:, step_idx + 1],
            x0,
            sigma,
            x_left=x_left,
            x_right=x_right,
        )
        total_loss = loss_step if total_loss is None else (total_loss + loss_step)
    return total_loss / float(n_steps)


def _path_outer_loss_backward(
    *,
    cfg,
    denoiser,
    states_ctrl: torch.Tensor,
    states_ref: torch.Tensor,
    x0: torch.Tensor,
    x_left: Optional[torch.Tensor],
    x_right: Optional[torch.Tensor],
    sigma_levels: torch.Tensor,
    lambda_ctrl: float,
    lambda_ref: float,
    amp_dtype,
) -> tuple[float, float, float]:
    """Accumulate exact path-mean outer loss with per-timestep backward passes."""

    n_steps = _path_n_steps(sigma_levels)
    if states_ctrl.shape[1] != n_steps + 1:
        raise ValueError(f"states_ctrl step dim must be {n_steps + 1}, got {states_ctrl.shape[1]}")
    if states_ref.shape[1] != n_steps + 1:
        raise ValueError(f"states_ref step dim must be {n_steps + 1}, got {states_ref.shape[1]}")

    attack_loss_total = 0.0
    clean_loss_total = 0.0
    outer_loss_total = 0.0

    for step_idx in range(n_steps):
        sigma = _path_step_sigma_batch(sigma_levels, step_idx + 1, x0)
        with autocast_context(sigma_levels.device, amp_dtype):
            if lambda_ctrl > 0.0:
                attack_loss_step = compute_training_loss(
                    cfg,
                    denoiser,
                    states_ctrl[:, step_idx + 1],
                    x0,
                    sigma,
                    x_left=x_left,
                    x_right=x_right,
                )
            else:
                with torch.no_grad():
                    attack_loss_step = compute_training_loss(
                        cfg,
                        denoiser,
                        states_ctrl[:, step_idx + 1],
                        x0,
                        sigma,
                        x_left=x_left,
                        x_right=x_right,
                    )

            if lambda_ref > 0.0:
                clean_loss_step = compute_training_loss(
                    cfg,
                    denoiser,
                    states_ref[:, step_idx + 1],
                    x0,
                    sigma,
                    x_left=x_left,
                    x_right=x_right,
                )
            else:
                clean_loss_step = torch.zeros((), device=x0.device, dtype=x0.dtype)

            chunk_outer = lambda_ctrl * attack_loss_step + lambda_ref * clean_loss_step
            chunk_outer = chunk_outer / float(n_steps)

        if has_nan_or_inf(chunk_outer):
            raise RuntimeError("NaN/Inf detected in cdro outer loss.")
        if chunk_outer.requires_grad:
            chunk_outer.backward()

        attack_loss_total += scalarize(attack_loss_step) / float(n_steps)
        clean_loss_total += scalarize(clean_loss_step) / float(n_steps)
        outer_loss_total += scalarize(chunk_outer)

    return attack_loss_total, clean_loss_total, outer_loss_total


def _path_batch_equiv_denoiser_evals(sigma_levels: torch.Tensor) -> float:
    """Count one denoiser eval over `B*T` path states as `T` batch-equivalent evals."""

    return float(max(_path_n_steps(sigma_levels), 0))


def _rollout_delta_diagnostics(roll, radius_by_step: torch.Tensor):
    """Summarize path/control magnitudes against the exact Route-A delta-space caps."""

    diff = roll.states_ctrl - roll.states_ref
    diff_l2 = pathwise_l2(diff)
    path_delta_mean = diff_l2.mean()
    terminal_delta_mean = diff_l2[:, -1].mean()

    delta_l2 = pathwise_l2(roll.delta_path)
    delta_norm_mean = delta_l2.mean()
    delta_norm_max = delta_l2.max()
    if radius_by_step.ndim == 1:
        radius = radius_by_step.view(1, -1).to(device=delta_l2.device, dtype=delta_l2.dtype)
    elif radius_by_step.ndim == 2:
        radius = radius_by_step.to(device=delta_l2.device, dtype=delta_l2.dtype)
    else:
        raise ValueError(f"radius_by_step must be rank 1 or 2, got shape={tuple(radius_by_step.shape)}")
    delta_ratio = delta_l2 / radius.clamp_min(1e-8)
    delta_ratio_mean = delta_ratio.mean()
    delta_ratio_max = delta_ratio.max()
    return path_delta_mean, terminal_delta_mean, delta_norm_mean, delta_norm_max, delta_ratio_mean, delta_ratio_max


def _path_reference_only_loss_backward(
    *,
    cfg,
    denoiser,
    states_ref: torch.Tensor,
    x0: torch.Tensor,
    x_left: Optional[torch.Tensor],
    x_right: Optional[torch.Tensor],
    sigma_levels: torch.Tensor,
    lambda_ctrl_equiv: float,
    lambda_ref_equiv: float,
    amp_dtype,
) -> tuple[float, float, float]:
    """Accumulate one reference-path loss while preserving branch-weight accounting."""

    n_steps = _path_n_steps(sigma_levels)
    if states_ref.shape[1] != n_steps + 1:
        raise ValueError(f"states_ref step dim must be {n_steps + 1}, got {states_ref.shape[1]}")

    total_mass = float(lambda_ctrl_equiv) + float(lambda_ref_equiv)
    if total_mass <= 0.0:
        return 0.0, 0.0, 0.0

    attack_loss_total = 0.0
    clean_loss_total = 0.0
    outer_loss_total = 0.0

    for step_idx in range(n_steps):
        sigma = _path_step_sigma_batch(sigma_levels, step_idx + 1, x0)
        with autocast_context(sigma_levels.device, amp_dtype):
            clean_loss_step = compute_training_loss(
                cfg,
                denoiser,
                states_ref[:, step_idx + 1],
                x0,
                sigma,
                x_left=x_left,
                x_right=x_right,
            )
            chunk_outer = total_mass * clean_loss_step / float(n_steps)

        if has_nan_or_inf(chunk_outer):
            raise RuntimeError("NaN/Inf detected in cdro reference-only outer loss.")
        if chunk_outer.requires_grad:
            chunk_outer.backward()

        loss_value = scalarize(clean_loss_step) / float(n_steps)
        if lambda_ctrl_equiv > 0.0:
            attack_loss_total += loss_value
        if lambda_ref_equiv > 0.0:
            clean_loss_total += loss_value
        outer_loss_total += scalarize(chunk_outer)

    return attack_loss_total, clean_loss_total, outer_loss_total


def train_trajectory_robust_cdro(
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
    """Route-A CDRO training with u-space local caps and greedy denoiser-dependent attacks."""

    optimizer_theta = torch.optim.Adam(denoiser.parameters(), lr=cfg.lr_theta)
    if optimizer_theta_state is not None:
        optimizer_theta.load_state_dict(optimizer_theta_state)
        for state in optimizer_theta.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device=sigma_levels.device)

    history = history_state if history_state is not None else {
        "outer_loss": [],
        "outer_loss_attack": [],
        "outer_loss_clean": [],
        "inner_obj": [],
        "energy": [],
        "delta_norm_mean": [],
        "delta_norm_max": [],
        "delta_norm_ratio_mean": [],
        "delta_norm_ratio_max": [],
        "sched_attack_weight": [],
        "sched_clean_weight": [],
        "sched_phi_lr_scale": [],
        "diag_step": [],
        "diag_inner_obj_current": [],
        "diag_inner_obj_zero": [],
        "diag_inner_obj_gap": [],
        "diag_inner_obj_gap_ratio": [],
        "diag_delta_norm_mean": [],
        "diag_delta_norm_max": [],
        "diag_delta_norm_ratio_mean": [],
        "diag_delta_norm_ratio_max": [],
        "diag_path_delta_mean": [],
        "diag_terminal_delta_mean": [],
        "batch_equiv_denoiser_evals_step": [],
        "batch_equiv_denoiser_evals_attack_construction": [],
        "batch_equiv_denoiser_evals_attack_eval": [],
        "batch_equiv_denoiser_evals_clean_eval": [],
        "rf_reflow_pair_fwd_units": [],
        "rf_step_wall_clock_sec": [],
        "batch_equiv_denoiser_evals_cumulative": [],
    }
    for key in (
        "outer_loss",
        "outer_loss_attack",
        "outer_loss_clean",
        "inner_obj",
        "energy",
        "delta_norm_mean",
        "delta_norm_max",
        "delta_norm_ratio_mean",
        "delta_norm_ratio_max",
        "sched_attack_weight",
        "sched_clean_weight",
        "sched_phi_lr_scale",
        "diag_step",
        "diag_inner_obj_current",
        "diag_inner_obj_zero",
        "diag_inner_obj_gap",
        "diag_inner_obj_gap_ratio",
        "diag_delta_norm_mean",
        "diag_delta_norm_max",
        "diag_delta_norm_ratio_mean",
        "diag_delta_norm_ratio_max",
        "diag_path_delta_mean",
        "diag_terminal_delta_mean",
        "batch_equiv_denoiser_evals_step",
        "batch_equiv_denoiser_evals_attack_construction",
        "batch_equiv_denoiser_evals_attack_eval",
        "batch_equiv_denoiser_evals_clean_eval",
        "rf_reflow_pair_fwd_units",
        "rf_step_wall_clock_sec",
        "batch_equiv_denoiser_evals_cumulative",
    ):
        history.setdefault(key, [])
    ensure_denoiser_op_count_history(history)

    # Route-A CDRO does not optimize a persistent control policy network.
    set_requires_grad(control, False)
    with torch.no_grad():
        for p in control.parameters():
            p.zero_()

    step_size = float(cfg.cdro_step_size)
    total_budget = float(cfg.cdro_total_budget_rho)
    time_horizon = float(cfg.cdro_time_horizon)
    path_batch_equiv_evals = _path_batch_equiv_denoiser_evals(sigma_levels)
    cumulative_batch_equiv_evals = (
        float(history["batch_equiv_denoiser_evals_cumulative"][-1])
        if history["batch_equiv_denoiser_evals_cumulative"]
        else 0.0
    )
    attack_num_steps, attack_num_steps_source = _resolve_attack_num_steps(cfg)
    amp_dtype = resolve_amp_dtype(sigma_levels.device, getattr(cfg, "amp_dtype", "auto"))
    ema_model = init_ema_model(denoiser, cfg, ema_state_dict=ema_state_dict)
    rf_pair_source = _resolve_rf_cdro_pair_source(cfg) if _is_rf_objective(cfg) else "data_noise"
    cdro_edm_ladder_mode = resolve_cdro_edm_ladder_mode(cfg)
    use_stochastic_edm_ladders = bool(
        not _is_rf_objective(cfg) and cdro_edm_ladder_mode == STOCHASTIC_STRATIFIED_QUANTILE_LADDER
    )
    history["rf_pair_source_resolved"] = (
        "shared_edm_teacher_reflow" if (_is_rf_objective(cfg) and rf_pair_source != "data_noise") else rf_pair_source
    )
    history["attack_num_steps_resolved"] = int(attack_num_steps)
    history["attack_num_steps_source"] = str(attack_num_steps_source)
    history["control_u_radius"] = float((total_budget / time_horizon) ** 0.5 if time_horizon > 0.0 else 0.0)
    history["cdro_edm_ladder_mode_resolved"] = str(cdro_edm_ladder_mode)
    history["rf_cdro_rollout_grid_mode_resolved"] = "stochastic_stratified" if _is_rf_objective(cfg) else ""
    rf_pair_teacher = None
    rf_stage1_steps = 0
    rf_reflow_steps = 0
    rf_stage_grids = {}
    teacher_pair_sigma_levels = None
    transition_deltas = build_transition_deltas_for_objective(
        cfg=cfg,
        sigma_levels=sigma_levels,
        time_horizon=time_horizon,
    ).to(device=sigma_levels.device, dtype=sigma_levels.dtype)
    radius_by_step = build_constraint_radii_for_objective(
        cfg=cfg,
        sigma_levels=sigma_levels,
        total_budget=total_budget,
        time_horizon=time_horizon,
    ).to(device=sigma_levels.device, dtype=sigma_levels.dtype)
    if _is_rf_objective(cfg):
        rf_stage_planning_total_steps = _resolve_rf_stage_planning_total_steps(cfg)
        rf_stage1_steps, rf_reflow_steps = resolve_rf_cdro_stage_steps(
            int(rf_stage_planning_total_steps),
            0.0,
            rf_pair_source,
        )
        rf_stage_grids = {
            "rf_stage1": _build_rf_cdro_stage_grid_info(
                cfg,
                device=sigma_levels.device,
                dtype=sigma_levels.dtype,
                time_horizon=time_horizon,
                stage_name="rf_stage1",
            ),
            "rf_reflow": _build_rf_cdro_stage_grid_info(
                cfg,
                device=sigma_levels.device,
                dtype=sigma_levels.dtype,
                time_horizon=time_horizon,
                stage_name="rf_reflow",
            ),
        }
        teacher_pair_sigma_levels = (
            rf_teacher_sigma_levels.to(device=sigma_levels.device, dtype=sigma_levels.dtype)
            if rf_teacher_sigma_levels is not None
            else build_rf_stage_time_quantile_levels(
                float(getattr(cfg, "sigma_max", 1.0)),
                int(resolve_rf_teacher_n_steps_path(cfg)),
                device=sigma_levels.device,
                stage_name="rf_reflow",
                reflow_distribution=str(getattr(cfg, "rf_reflow_t_distribution", "u_shaped")),
            ).to(device=sigma_levels.device, dtype=sigma_levels.dtype)
        )
        rf_eval_stage = "rf_reflow" if int(rf_reflow_steps) > 0 else "rf_stage1"
        transition_deltas = rf_stage_grids[rf_eval_stage]["transition_deltas"]
        radius_by_step = rf_stage_grids[rf_eval_stage]["radius_by_step"]
        history["transition_deltas"] = []
        history["transition_deltas_reference"] = _snapshot_tensor_values(transition_deltas)
        history["rf_stage1_steps"] = int(rf_stage1_steps)
        history["rf_reflow_steps"] = int(rf_reflow_steps)
        history["rf_stage_planning_total_steps"] = int(rf_stage_planning_total_steps)
        history.setdefault("rf_stage", [])
        history.setdefault("rf_t_distribution", [])
        history["rf_stage1_t_distribution_resolved"] = ""
        history["rf_reflow_t_distribution_resolved"] = str(rf_stage_grids["rf_reflow"]["distribution"])
        history["rf_eval_t_distribution_resolved"] = str(rf_stage_grids[rf_eval_stage]["distribution"])
        history["rf_stage_transition_deltas"] = {
            stage_name: [float(v.item()) for v in stage_info["transition_deltas"].detach().cpu()]
            for stage_name, stage_info in rf_stage_grids.items()
        }
        history["rf_teacher_pair_n_steps_path_resolved"] = int(
            max(int(teacher_pair_sigma_levels.numel()) - 1, 0)
        )
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
            history["rf_teacher_pair_sampling_mode_resolved"] = _rf_teacher_pair_sampling_mode(cfg, rf_pair_teacher)
    else:
        if use_stochastic_edm_ladders:
            history["transition_deltas"] = []
            history["transition_deltas_reference"] = _snapshot_tensor_values(transition_deltas)
        else:
            history["transition_deltas"] = _snapshot_tensor_values(transition_deltas)

    print(
        f"[cdro] resolved attack_num_steps={attack_num_steps} "
        f"source={attack_num_steps_source} pair_source={rf_pair_source} "
        f"control_u_radius={history['control_u_radius']:.6f} "
        f"edm_ladder_mode={cdro_edm_ladder_mode}"
        + (
            ""
            if not _is_rf_objective(cfg)
            else " "
            + "rf_stage1_t_distribution="
            + str(history.get("rf_stage1_t_distribution_resolved", ""))
            + " "
            + "rf_reflow_t_distribution="
            + str(history.get("rf_reflow_t_distribution_resolved", ""))
        ),
        flush=True,
    )

    for step in range(int(start_step) + 1, int(cfg.steps) + 1):
        step_t0 = time.perf_counter()
        x_data = sample_train_batch(
            cfg,
            centers,
            train_pool=train_pool,
            sample_train_batch_fn=sample_train_batch_fn,
            sample_population_batch_fn=sample_population_batch_fn,
        )
        x0 = x_data
        rf_pair_left = None
        current_rf_stage = None
        reflow_pair_fwd_units = 0.0
        current_sigma_levels = sigma_levels
        current_transition_deltas = transition_deltas
        current_radius_by_step = radius_by_step
        if _is_rf_objective(cfg):
            if step <= int(rf_stage1_steps):
                rf_pair_left = torch.randn_like(x_data)
                x0 = x_data
                current_rf_stage = "rf_stage1"
            else:
                if rf_pair_teacher is None:
                    teacher_source = ema_model if ema_model is not None else denoiser
                    rf_pair_teacher = copy.deepcopy(teacher_source).eval()
                    set_requires_grad(rf_pair_teacher, False)
                    history.setdefault("rf_reflow_teacher_refresh_step", int(step - 1))
                    history["rf_teacher_pair_sampling_mode_resolved"] = _rf_teacher_pair_sampling_mode(cfg, rf_pair_teacher)
                rf_pair_left, x0 = generate_reflow_pairs(
                    rf_pair_teacher,
                    teacher_pair_sigma_levels,
                    x_data,
                    sample_terminal_batch_fn=None,
                    ve_sampler_mode=resolve_rf_teacher_ve_sampler_mode(cfg),
                )
                reflow_pair_fwd_units = float(max(int(teacher_pair_sigma_levels.numel()) - 1, 0))
                current_rf_stage = "rf_reflow"
            stage_grid = _sample_rf_cdro_stage_grid_info(
                cfg,
                device=sigma_levels.device,
                dtype=sigma_levels.dtype,
                time_horizon=time_horizon,
                stage_name=current_rf_stage,
                batch_size=x0.shape[0],
            )
            current_sigma_levels = stage_grid["sigma_levels"]
            current_transition_deltas = stage_grid["transition_deltas"]
            current_radius_by_step = stage_grid["radius_by_step"]
            history["rf_stage"].append(current_rf_stage)
            history["rf_t_distribution"].append(str(stage_grid["distribution"]))
            if "stochastic_ladder_first_sigma_levels" not in history:
                history["stochastic_ladder_first_sigma_levels"] = _snapshot_tensor_values(current_sigma_levels)
                history["stochastic_ladder_first_transition_deltas"] = _snapshot_tensor_values(current_transition_deltas)
            history["stochastic_ladder_last_sigma_levels"] = _snapshot_tensor_values(current_sigma_levels)
            history["stochastic_ladder_last_transition_deltas"] = _snapshot_tensor_values(current_transition_deltas)
        elif use_stochastic_edm_ladders:
            current_sigma_levels = _sample_cdro_edm_step_ladder(cfg, sigma_levels, batch_size=x0.shape[0])
            current_transition_deltas = build_transition_deltas_for_objective(
                cfg=cfg,
                sigma_levels=current_sigma_levels,
                time_horizon=time_horizon,
            ).to(device=current_sigma_levels.device, dtype=current_sigma_levels.dtype)
            current_radius_by_step = build_constraint_radii_for_objective(
                cfg=cfg,
                sigma_levels=current_sigma_levels,
                total_budget=total_budget,
                time_horizon=time_horizon,
            ).to(device=current_sigma_levels.device, dtype=current_sigma_levels.dtype)
            if "stochastic_ladder_first_sigma_levels" not in history:
                history["stochastic_ladder_first_sigma_levels"] = _snapshot_tensor_values(current_sigma_levels)
                history["stochastic_ladder_first_transition_deltas"] = _snapshot_tensor_values(
                    current_transition_deltas
                )
            history["stochastic_ladder_last_sigma_levels"] = _snapshot_tensor_values(current_sigma_levels)
            history["stochastic_ladder_last_transition_deltas"] = _snapshot_tensor_values(current_transition_deltas)
        if _is_rf_objective(cfg):
            indices = sample_target_indices(x0.shape[0], current_sigma_levels)
        elif cfg.use_log_normal_sigma_sampling:
            indices = sample_target_indices_log_normal(
                x0.shape[0],
                current_sigma_levels,
                p_mean=cfg.p_mean,
                p_std=cfg.p_std,
            )
        else:
            indices = sample_target_indices(x0.shape[0], current_sigma_levels)

        raw_lambda_ref = float(cfg.outer_clean_weight)
        raw_lambda_ctrl = float(cfg.outer_attack_weight)
        mixture_mass = raw_lambda_ref + raw_lambda_ctrl
        if mixture_mass <= 0.0:
            raise RuntimeError("CDRO continuation mixture must have positive total mass.")
        lambda_ref = raw_lambda_ref / mixture_mass
        lambda_ctrl = raw_lambda_ctrl / mixture_mass
        attack_path_enabled = bool(lambda_ctrl > 0.0 and attack_num_steps > 0 and total_budget > 0.0)
        reference_path_weight = float(lambda_ref + (lambda_ctrl if not attack_path_enabled else 0.0))
        phi_lr_scale = 1.0 if attack_path_enabled else 0.0
        control_updates_enabled = attack_path_enabled
        rollout_schedules = [None]
        rollout_multiplier = 1.0
        attack_construction_units = 0.0
        rollouts = []

        set_requires_grad(denoiser, False)
        for eps_schedule in rollout_schedules:
            if attack_path_enabled:
                roll = rollout_path_heuristic_attack(
                    cfg=cfg,
                    x0=x0,
                    target_indices=indices,
                    attack_net=denoiser,
                    sigma_levels=current_sigma_levels,
                    inner_steps=int(attack_num_steps),
                    step_size=step_size,
                    total_budget=total_budget,
                    time_horizon=time_horizon,
                    eps_schedule=eps_schedule,
                    rf_pair_left=rf_pair_left,
                )
            else:
                roll = rollout_controlled_ve(
                    x0=x0,
                    target_indices=indices,
                    control_net=None,
                    sigma_levels=current_sigma_levels,
                    grad_through_control=False,
                    control_radius_kappa=cfg.control_radius_kappa,
                    kappa_by_step=None,
                    total_budget=total_budget,
                    time_horizon=time_horizon,
                    eps_schedule=eps_schedule,
                    cfg=cfg,
                    rf_pair_left=rf_pair_left,
                )
            rollouts.append(roll)
        if attack_path_enabled:
            attack_construction_units = (
                path_batch_equiv_evals * float(max(int(attack_num_steps), 0)) * rollout_multiplier
            )

        if attack_path_enabled:
            attack_loss_inner = None
            transport_inner = None
            for roll in rollouts:
                with autocast_context(sigma_levels.device, amp_dtype):
                    attack_loss_chunk = _path_average_training_loss(
                        cfg,
                        denoiser,
                        roll.states_ctrl,
                        x0,
                        current_sigma_levels,
                        x_left=roll.x_left,
                        x_right=roll.x_right,
                    )
                transport_chunk = _control_transport_cost(roll.control_path, current_transition_deltas)
                attack_loss_inner = (
                    attack_loss_chunk if attack_loss_inner is None else (attack_loss_inner + attack_loss_chunk)
                )
                transport_inner = transport_chunk if transport_inner is None else (transport_inner + transport_chunk)
            attack_loss_inner = attack_loss_inner / rollout_multiplier
            transport_inner = transport_inner / rollout_multiplier
            inner_obj = inner_objective_attack_only(attack_loss_inner)
            if has_nan_or_inf(inner_obj):
                raise RuntimeError("NaN/Inf detected in cdro inner objective.")
        else:
            attack_loss_inner = torch.zeros((), device=x0.device, dtype=x0.dtype)
            transport_inner = None
            for roll in rollouts:
                transport_chunk = _control_transport_cost(roll.control_path, current_transition_deltas)
                transport_inner = transport_chunk if transport_inner is None else (transport_inner + transport_chunk)
            transport_inner = transport_inner / rollout_multiplier
            inner_obj = torch.zeros((), device=x0.device, dtype=x0.dtype)

        last_inner_obj = scalarize(inner_obj) if attack_path_enabled else 0.0
        last_transport = scalarize(transport_inner)
        delta_norm_mean_values = []
        delta_norm_max_values = []
        delta_ratio_mean_values = []
        delta_ratio_max_values = []
        if current_radius_by_step.ndim == 1:
            radius = current_radius_by_step.view(1, -1)
        elif current_radius_by_step.ndim == 2:
            radius = current_radius_by_step
        else:
            raise ValueError(
                f"current_radius_by_step must be rank 1 or 2, got shape={tuple(current_radius_by_step.shape)}"
            )
        for roll in rollouts:
            delta_l2 = pathwise_l2(roll.delta_path)
            delta_ratio = delta_l2 / radius.to(device=delta_l2.device, dtype=delta_l2.dtype).clamp_min(1e-8)
            delta_norm_mean_values.append(scalarize(delta_l2.mean()))
            delta_norm_max_values.append(scalarize(delta_l2.max()))
            delta_ratio_mean_values.append(scalarize(delta_ratio.mean()))
            delta_ratio_max_values.append(scalarize(delta_ratio.max()))
        last_delta_norm_mean = sum(delta_norm_mean_values) / len(delta_norm_mean_values)
        last_delta_norm_max = sum(delta_norm_max_values) / len(delta_norm_max_values)
        last_delta_ratio_mean = sum(delta_ratio_mean_values) / len(delta_ratio_mean_values)
        last_delta_ratio_max = sum(delta_ratio_max_values) / len(delta_ratio_max_values)

        set_requires_grad(denoiser, True)
        optimizer_theta.zero_grad(set_to_none=True)
        outer_loss_attack_vals = []
        outer_loss_clean_vals = []
        outer_loss_vals = []
        if not attack_path_enabled:
            lambda_ctrl_scaled = float(lambda_ctrl) / rollout_multiplier
            lambda_ref_scaled = float(lambda_ref) / rollout_multiplier
            for roll in rollouts:
                outer_loss_attack_val, outer_loss_clean_val, outer_loss_val = _path_reference_only_loss_backward(
                    cfg=cfg,
                    denoiser=denoiser,
                    states_ref=roll.states_ref,
                    x0=x0,
                    x_left=roll.x_left,
                    x_right=roll.x_right,
                    sigma_levels=current_sigma_levels,
                    lambda_ctrl_equiv=lambda_ctrl_scaled,
                    lambda_ref_equiv=lambda_ref_scaled,
                    amp_dtype=amp_dtype,
                )
                outer_loss_attack_vals.append(float(outer_loss_attack_val))
                outer_loss_clean_vals.append(float(outer_loss_clean_val))
                outer_loss_vals.append(float(outer_loss_val))
            attack_eval_units = 0.0
            clean_eval_units = path_batch_equiv_evals * rollout_multiplier if reference_path_weight > 0.0 else 0.0
        else:
            lambda_ctrl_scaled = float(lambda_ctrl) / rollout_multiplier
            lambda_ref_scaled = float(lambda_ref) / rollout_multiplier
            for roll in rollouts:
                outer_loss_attack_val, outer_loss_clean_val, outer_loss_val = _path_outer_loss_backward(
                    cfg=cfg,
                    denoiser=denoiser,
                    states_ctrl=roll.states_ctrl,
                    states_ref=roll.states_ref,
                    x0=x0,
                    x_left=roll.x_left,
                    x_right=roll.x_right,
                    sigma_levels=current_sigma_levels,
                    lambda_ctrl=lambda_ctrl_scaled,
                    lambda_ref=lambda_ref_scaled,
                    amp_dtype=amp_dtype,
                )
                outer_loss_attack_vals.append(float(outer_loss_attack_val))
                outer_loss_clean_vals.append(float(outer_loss_clean_val))
                outer_loss_vals.append(float(outer_loss_val))
            attack_eval_units = path_batch_equiv_evals * 2.0 * rollout_multiplier
            clean_eval_units = path_batch_equiv_evals * rollout_multiplier if lambda_ref > 0.0 else 0.0
        optimizer_theta.step()
        update_ema_model(
            ema_model,
            denoiser,
            cfg,
            cur_nimg=float((step - 1) * int(cfg.batch_size)),
            batch_size=int(cfg.batch_size),
        )
        outer_loss_attack_val = sum(outer_loss_attack_vals) / len(outer_loss_attack_vals)
        outer_loss_clean_val = sum(outer_loss_clean_vals) / len(outer_loss_clean_vals)
        outer_loss_val = sum(outer_loss_vals) / len(outer_loss_vals)
        outer_loss_attack = torch.tensor(outer_loss_attack_val, device=x0.device, dtype=x0.dtype)
        outer_loss_clean = torch.tensor(outer_loss_clean_val, device=x0.device, dtype=x0.dtype)
        outer_loss = torch.tensor(outer_loss_val, device=x0.device, dtype=x0.dtype)
        step_batch_equiv_evals = attack_construction_units + attack_eval_units + clean_eval_units + reflow_pair_fwd_units
        cumulative_batch_equiv_evals += step_batch_equiv_evals

        history["outer_loss"].append(scalarize(outer_loss))
        history["outer_loss_attack"].append(scalarize(outer_loss_attack))
        history["outer_loss_clean"].append(scalarize(outer_loss_clean))
        history["inner_obj"].append(last_inner_obj)
        history["energy"].append(last_transport)
        history["delta_norm_mean"].append(last_delta_norm_mean)
        history["delta_norm_max"].append(last_delta_norm_max)
        history["delta_norm_ratio_mean"].append(last_delta_ratio_mean)
        history["delta_norm_ratio_max"].append(last_delta_ratio_max)
        history["sched_attack_weight"].append(float(lambda_ctrl))
        history["sched_clean_weight"].append(float(lambda_ref))
        history["sched_phi_lr_scale"].append(float(phi_lr_scale))
        history["batch_equiv_denoiser_evals_step"].append(float(step_batch_equiv_evals))
        history["batch_equiv_denoiser_evals_attack_construction"].append(float(attack_construction_units))
        history["batch_equiv_denoiser_evals_attack_eval"].append(float(attack_eval_units))
        history["batch_equiv_denoiser_evals_clean_eval"].append(float(clean_eval_units))
        history["rf_reflow_pair_fwd_units"].append(float(reflow_pair_fwd_units))
        history["batch_equiv_denoiser_evals_cumulative"].append(float(cumulative_batch_equiv_evals))
        effective_outer_branches = (
            float(int(lambda_ctrl > 0.0) + int(lambda_ref > 0.0))
            if attack_path_enabled
            else float(int(reference_path_weight > 0.0))
        )
        append_denoiser_op_count_step(
            history,
            n_fwd=float(reflow_pair_fwd_units),
            n_fwd_inputgrad=float(attack_construction_units),
            n_fwd_parambackward=float(path_batch_equiv_evals) * effective_outer_branches * rollout_multiplier,
        )
        history["rf_step_wall_clock_sec"].append(float(time.perf_counter() - step_t0))

        run_diag = (
            bool(cfg.collapse_diagnostics_enabled)
            and (step % max(int(cfg.collapse_diag_every), 1) == 0 or step == 1 or step == int(cfg.steps))
        )
        if run_diag:
            if attack_path_enabled:
                set_requires_grad(denoiser, False)
                roll_cur_diag = rollout_path_heuristic_attack(
                    cfg=cfg,
                    x0=x0,
                    target_indices=indices,
                    attack_net=denoiser,
                    sigma_levels=current_sigma_levels,
                    inner_steps=int(attack_num_steps),
                    step_size=step_size,
                    total_budget=total_budget,
                    time_horizon=time_horizon,
                    rf_pair_left=rf_pair_left,
                )
                set_requires_grad(denoiser, True)
            else:
                with torch.no_grad():
                    roll_cur_diag = rollout_controlled_ve(
                        x0=x0,
                        target_indices=indices,
                        control_net=None,
                        sigma_levels=current_sigma_levels,
                        grad_through_control=False,
                        control_radius_kappa=cfg.control_radius_kappa,
                        kappa_by_step=None,
                        total_budget=total_budget,
                        time_horizon=time_horizon,
                        cfg=cfg,
                        rf_pair_left=rf_pair_left,
                    )

            with torch.no_grad():
                with autocast_context(sigma_levels.device, amp_dtype):
                    attack_cur = _path_average_training_loss(
                        cfg,
                        denoiser,
                        roll_cur_diag.states_ctrl,
                        x0,
                        current_sigma_levels,
                        x_left=roll_cur_diag.x_left,
                        x_right=roll_cur_diag.x_right,
                    )
                    inner_obj_cur = inner_objective_attack_only(attack_cur)

                roll_zero_diag = rollout_controlled_ve(
                    x0=x0,
                    target_indices=indices,
                    control_net=None,
                    sigma_levels=current_sigma_levels,
                    grad_through_control=False,
                    control_radius_kappa=cfg.control_radius_kappa,
                    kappa_by_step=None,
                    total_budget=total_budget,
                    time_horizon=time_horizon,
                    cfg=cfg,
                    rf_pair_left=rf_pair_left,
                )
                with autocast_context(sigma_levels.device, amp_dtype):
                    attack_zero = _path_average_training_loss(
                        cfg,
                        denoiser,
                        roll_zero_diag.states_ctrl,
                        x0,
                        current_sigma_levels,
                        x_left=roll_zero_diag.x_left,
                        x_right=roll_zero_diag.x_right,
                    )
                    inner_obj_zero = inner_objective_attack_only(attack_zero)
                gap = inner_obj_cur - inner_obj_zero
                gap_ratio = gap / (inner_obj_zero.abs() + 1e-8)
                (
                    path_delta_mean,
                    terminal_delta_mean,
                    delta_norm_mean,
                    delta_norm_max,
                    delta_ratio_mean,
                    delta_ratio_max,
                ) = _rollout_delta_diagnostics(
                    roll_cur_diag,
                    radius_by_step=current_radius_by_step,
                )

            history["diag_step"].append(int(step))
            history["diag_inner_obj_current"].append(scalarize(inner_obj_cur))
            history["diag_inner_obj_zero"].append(scalarize(inner_obj_zero))
            history["diag_inner_obj_gap"].append(scalarize(gap))
            history["diag_inner_obj_gap_ratio"].append(scalarize(gap_ratio))
            history["diag_delta_norm_mean"].append(scalarize(delta_norm_mean))
            history["diag_delta_norm_max"].append(scalarize(delta_norm_max))
            history["diag_delta_norm_ratio_mean"].append(scalarize(delta_ratio_mean))
            history["diag_delta_norm_ratio_max"].append(scalarize(delta_ratio_max))
            history["diag_path_delta_mean"].append(scalarize(path_delta_mean))
            history["diag_terminal_delta_mean"].append(scalarize(terminal_delta_mean))

        if step % cfg.log_every == 0:
            diag_msg = ""
            if len(history["diag_step"]) > 0:
                diag_msg = (
                    " "
                    f"diag_gap={history['diag_inner_obj_gap'][-1]:.6f} "
                    f"diag_gap_ratio={history['diag_inner_obj_gap_ratio'][-1]:.6f} "
                    f"diag_delta_ratio={history['diag_delta_norm_ratio_mean'][-1]:.6f} "
                    f"diag_path_delta={history['diag_path_delta_mean'][-1]:.6f}"
                )
            print(
                f"[cdro] step={step:05d} outer_loss={outer_loss.item():.6f} "
                f"attack={outer_loss_attack.item():.6f} clean={outer_loss_clean.item():.6f} "
                f"inner_obj={last_inner_obj:.6f} control_cost={last_transport:.6f} "
                f"delta_norm={last_delta_norm_mean:.6f} delta_ratio={last_delta_ratio_mean:.6f} "
                f"be_evals={step_batch_equiv_evals:.1f} be_evals_cum={cumulative_batch_equiv_evals:.1f} "
                f"lambda_ctrl={lambda_ctrl:.3f} lambda_ref={lambda_ref:.3f} phi_lr_scale={phi_lr_scale:.3f} "
                f"step_size={step_size:.6f} rho={total_budget:.4f} T={time_horizon:.4f}"
                f"{diag_msg}",
                flush=True,
            )

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
        return history, trainer_state
    if ema_model is not None:
        denoiser.load_state_dict(ema_model.state_dict(), strict=True)
    return history


def train_trajectory_robust_energy(
    denoiser,
    control,
    centers,
    sigma_levels,
    cfg,
    train_pool: torch.Tensor = None,
    sample_train_batch_fn: Optional[Callable[[int], torch.Tensor]] = None,
    sample_population_batch_fn: Optional[Callable[[int], torch.Tensor]] = None,
):
    """Backward-compatible alias."""

    return train_trajectory_robust_cdro(
        denoiser,
        control,
        centers,
        sigma_levels,
        cfg,
        train_pool=train_pool,
        sample_train_batch_fn=sample_train_batch_fn,
        sample_population_batch_fn=sample_population_batch_fn,
    )
