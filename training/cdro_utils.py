import math
from dataclasses import dataclass
from typing import Optional, Union

import torch


DETERMINISTIC_MIDPOINT_QUANTILE_LADDER = "deterministic_midpoint_quantile"
STOCHASTIC_STRATIFIED_QUANTILE_LADDER = "stochastic_stratified_quantile"


@dataclass
class CDRORollout:
    states_ref: torch.Tensor
    states_ctrl: torch.Tensor
    delta_path: torch.Tensor
    control_path: torch.Tensor
    sigma_levels: torch.Tensor
    transition_deltas: torch.Tensor
    radius_by_step: torch.Tensor


def resolve_cdro_edm_ladder_mode(mode: str) -> str:
    value = str(mode or DETERMINISTIC_MIDPOINT_QUANTILE_LADDER).strip().lower()
    if value in (
        "deterministic",
        "midpoint",
        "midpoint_quantile",
        DETERMINISTIC_MIDPOINT_QUANTILE_LADDER,
    ):
        return DETERMINISTIC_MIDPOINT_QUANTILE_LADDER
    if value in (
        "stochastic",
        "stratified",
        "stochastic_stratified",
        STOCHASTIC_STRATIFIED_QUANTILE_LADDER,
    ):
        return STOCHASTIC_STRATIFIED_QUANTILE_LADDER
    raise ValueError(
        "cdro_edm_ladder_mode must be one of "
        f"{DETERMINISTIC_MIDPOINT_QUANTILE_LADDER!r} or {STOCHASTIC_STRATIFIED_QUANTILE_LADDER!r}, "
        f"got {mode!r}"
    )


def _truncated_log_sigma_cdf_bounds(
    *,
    sigma_min: float,
    sigma_max: float,
    p_mean: float,
    p_std: float,
) -> tuple[torch.distributions.Normal, torch.Tensor, torch.Tensor]:
    if sigma_min <= 0.0 or sigma_max <= 0.0:
        raise ValueError(f"sigma_min and sigma_max must be positive, got {sigma_min}, {sigma_max}")
    if sigma_min >= sigma_max:
        raise ValueError(f"sigma_min must be < sigma_max, got {sigma_min} >= {sigma_max}")
    if p_std <= 0.0:
        raise ValueError(f"p_std must be > 0, got {p_std}")
    normal = torch.distributions.Normal(
        loc=torch.tensor(float(p_mean), dtype=torch.float64),
        scale=torch.tensor(float(p_std), dtype=torch.float64),
    )
    z_min = torch.tensor(math.log(float(sigma_min)), dtype=torch.float64)
    z_max = torch.tensor(math.log(float(sigma_max)), dtype=torch.float64)
    cdf_min = normal.cdf(z_min)
    cdf_max = normal.cdf(z_max)
    if float((cdf_max - cdf_min).item()) <= 1e-12:
        raise ValueError("Truncated warmup-law support has zero mass on the configured sigma range.")
    return normal, cdf_min, cdf_max


def build_log_sigma_quantile_ladder(
    *,
    sigma_min: float,
    sigma_max: float,
    n_steps: int,
    p_mean: float,
    p_std: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if int(n_steps) <= 0:
        raise ValueError(f"n_steps must be > 0, got {n_steps}")
    normal, cdf_min, cdf_max = _truncated_log_sigma_cdf_bounds(
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        p_mean=p_mean,
        p_std=p_std,
    )
    mass = cdf_max - cdf_min
    probs_mid = (torch.arange(1, int(n_steps) + 1, dtype=torch.float64) - 0.5) / float(n_steps)
    trunc_mid_cdf = cdf_min + probs_mid * mass
    z_mid = normal.icdf(trunc_mid_cdf.clamp(min=1e-12, max=1.0 - 1e-12))
    z_min = torch.tensor(math.log(float(sigma_min)), dtype=torch.float64)
    z_max = torch.tensor(math.log(float(sigma_max)), dtype=torch.float64)
    if int(n_steps) == 1:
        z_edges = torch.stack([z_min, z_max])
    else:
        probs_edge = torch.arange(1, int(n_steps), dtype=torch.float64) / float(n_steps)
        trunc_edge_cdf = cdf_min + probs_edge * mass
        z_inner = normal.icdf(trunc_edge_cdf.clamp(min=1e-12, max=1.0 - 1e-12))
        z_edges = torch.cat([z_min.view(1), z_inner, z_max.view(1)])
    sigma_positive = torch.exp(z_mid)
    sigma_levels = torch.cat([torch.zeros(1, dtype=torch.float64), sigma_positive]).to(device=device, dtype=dtype)
    return sigma_levels, z_edges.to(device=device, dtype=dtype)


def sample_log_sigma_stratified_quantile_ladder(
    *,
    sigma_min: float,
    sigma_max: float,
    n_steps: int,
    p_mean: float,
    p_std: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if int(n_steps) <= 0:
        raise ValueError(f"n_steps must be > 0, got {n_steps}")
    normal, cdf_min, cdf_max = _truncated_log_sigma_cdf_bounds(
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        p_mean=p_mean,
        p_std=p_std,
    )
    mass = cdf_max - cdf_min
    jitter = torch.empty(int(n_steps), dtype=torch.float64).uniform_(1e-12, 1.0 - 1e-12)
    strata = torch.arange(0, int(n_steps), dtype=torch.float64)
    probs = (strata + jitter) / float(n_steps)
    trunc_cdf = cdf_min + probs * mass
    z_nodes = normal.icdf(trunc_cdf.clamp(min=1e-12, max=1.0 - 1e-12))
    sigma_positive = torch.exp(z_nodes)
    sigma_levels = torch.cat([torch.zeros(1, dtype=torch.float64), sigma_positive]).to(device=device, dtype=dtype)
    return sigma_levels


def sample_log_sigma_stratified_quantile_ladder_batch(
    *,
    sigma_min: float,
    sigma_max: float,
    n_steps: int,
    batch_size: int,
    p_mean: float,
    p_std: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if int(n_steps) <= 0:
        raise ValueError(f"n_steps must be > 0, got {n_steps}")
    if int(batch_size) <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    normal, cdf_min, cdf_max = _truncated_log_sigma_cdf_bounds(
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        p_mean=p_mean,
        p_std=p_std,
    )
    mass = cdf_max - cdf_min
    jitter = torch.empty((int(batch_size), int(n_steps)), dtype=torch.float64).uniform_(1e-12, 1.0 - 1e-12)
    strata = torch.arange(0, int(n_steps), dtype=torch.float64).view(1, int(n_steps))
    probs = (strata + jitter) / float(n_steps)
    trunc_cdf = cdf_min + probs * mass
    z_nodes = normal.icdf(trunc_cdf.clamp(min=1e-12, max=1.0 - 1e-12))
    sigma_positive = torch.exp(z_nodes)
    if torch.any(sigma_positive <= 0):
        raise ValueError("Per-example stratified continuation ladders must be strictly positive.")
    if sigma_positive.shape[1] > 1 and not torch.all(sigma_positive[:, 1:] > sigma_positive[:, :-1]):
        raise ValueError("Per-example stratified continuation ladders must be strictly increasing.")
    sigma_levels = torch.cat(
        [
            torch.zeros((int(batch_size), 1), dtype=torch.float64),
            sigma_positive,
        ],
        dim=1,
    ).to(device=device, dtype=dtype)
    return sigma_levels


def build_transition_deltas(
    *,
    sigma_levels: torch.Tensor,
    sigma_min: float,
    sigma_max: float,
    time_horizon: float,
    ladder_mode: str,
    log_sigma_edges: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if float(time_horizon) <= 0.0:
        raise ValueError(f"time_horizon must be > 0, got {time_horizon}")
    if ladder_mode == STOCHASTIC_STRATIFIED_QUANTILE_LADDER:
        positive_sigma = sigma_levels[..., 1:]
        log_span = math.log(float(sigma_max)) - math.log(float(sigma_min))
        tau = float(time_horizon) * (positive_sigma.log() - math.log(float(sigma_min))) / log_span
        dt = torch.empty_like(tau)
        dt[..., 0] = tau[..., 0]
        if tau.shape[-1] > 1:
            dt[..., 1:] = tau[..., 1:] - tau[..., :-1]
        return dt
    if log_sigma_edges is None:
        raise ValueError("Deterministic CDRO ladders require log_sigma_edges.")
    log_span = math.log(float(sigma_max)) - math.log(float(sigma_min))
    tau_edges = float(time_horizon) * (log_sigma_edges - math.log(float(sigma_min))) / log_span
    return tau_edges[1:] - tau_edges[:-1]


def build_constraint_radii(
    *,
    transition_deltas: torch.Tensor,
    total_budget_rho: float,
    time_horizon: float,
) -> torch.Tensor:
    if float(total_budget_rho) < 0.0:
        raise ValueError(f"total_budget_rho must be >= 0, got {total_budget_rho}")
    if float(time_horizon) <= 0.0:
        raise ValueError(f"time_horizon must be > 0, got {time_horizon}")
    control_radius = math.sqrt(float(total_budget_rho) / float(time_horizon)) if float(total_budget_rho) > 0.0 else 0.0
    return transition_deltas * float(control_radius)


def project_l2_ball(delta_raw: torch.Tensor, radius: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    flat = delta_raw.reshape(delta_raw.shape[0], -1)
    norm = flat.norm(dim=1, keepdim=True).clamp_min(eps)
    radius = radius.reshape(-1, 1).to(device=flat.device, dtype=flat.dtype)
    scale = torch.minimum(torch.ones_like(norm), radius / norm)
    return (flat * scale).reshape_as(delta_raw)


def l2_normalize_per_sample(value: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    flat = value.reshape(value.shape[0], -1)
    norm = flat.norm(p=2, dim=1, keepdim=True).clamp_min(eps)
    return (flat / norm).reshape_as(value)


def weighted_edm_loss_per_pixel(
    *,
    net,
    x_noisy: torch.Tensor,
    x_clean: torch.Tensor,
    sigma: torch.Tensor,
    sigma_data: float,
    labels=None,
    augment_labels=None,
) -> torch.Tensor:
    sigma = sigma.to(torch.float32)
    weight = (sigma.square() + float(sigma_data) ** 2) / (sigma * float(sigma_data)).square().clamp_min(1e-8)
    pred = net(x_noisy, sigma, labels, augment_labels=augment_labels)
    return weight.reshape(-1, 1, 1, 1) * (pred - x_clean).square()


def reduce_per_sample(loss: torch.Tensor) -> torch.Tensor:
    return loss.reshape(loss.shape[0], -1).sum(dim=1)


def control_transport_cost(control_path: torch.Tensor, transition_deltas: torch.Tensor) -> torch.Tensor:
    flat = control_path.reshape(control_path.shape[0], control_path.shape[1], -1)
    control_sq = flat.pow(2).sum(dim=2)
    if transition_deltas.ndim == 1:
        dt = transition_deltas.view(1, -1)
    elif transition_deltas.ndim == 2:
        dt = transition_deltas
    else:
        raise ValueError(
            "transition_deltas must have shape [steps] or [batch, steps], "
            f"got {tuple(transition_deltas.shape)}"
        )
    dt = dt.to(device=control_sq.device, dtype=control_sq.dtype)
    return (control_sq * dt).sum(dim=1)


def pathwise_l2(path_tensor: torch.Tensor) -> torch.Tensor:
    return path_tensor.reshape(path_tensor.shape[0], path_tensor.shape[1], -1).norm(dim=2)


def build_cdro_ladder(
    *,
    n_steps_path: int,
    sigma_min: float,
    sigma_max: float,
    p_mean: float,
    p_std: float,
    total_budget_rho: float,
    time_horizon: float,
    ladder_mode: str,
    batch_size: Optional[int],
    per_example_sigma_ladders: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    resolved_mode = resolve_cdro_edm_ladder_mode(ladder_mode)
    if resolved_mode == STOCHASTIC_STRATIFIED_QUANTILE_LADDER and bool(per_example_sigma_ladders):
        if batch_size is None or int(batch_size) <= 0:
            raise ValueError(
                "Per-example stochastic CDRO ladders require a positive batch_size, "
                f"got {batch_size}"
            )
        sigma_levels = sample_log_sigma_stratified_quantile_ladder_batch(
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            n_steps=n_steps_path,
            batch_size=int(batch_size),
            p_mean=p_mean,
            p_std=p_std,
            device=device,
            dtype=dtype,
        )
        log_sigma_edges = None
    elif resolved_mode == STOCHASTIC_STRATIFIED_QUANTILE_LADDER:
        sigma_levels = sample_log_sigma_stratified_quantile_ladder(
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            n_steps=n_steps_path,
            p_mean=p_mean,
            p_std=p_std,
            device=device,
            dtype=dtype,
        )
        log_sigma_edges = None
    else:
        sigma_levels, log_sigma_edges = build_log_sigma_quantile_ladder(
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            n_steps=n_steps_path,
            p_mean=p_mean,
            p_std=p_std,
            device=device,
            dtype=dtype,
        )
    transition_deltas = build_transition_deltas(
        sigma_levels=sigma_levels,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        time_horizon=time_horizon,
        ladder_mode=resolved_mode,
        log_sigma_edges=log_sigma_edges,
    ).to(device=device, dtype=dtype)
    radius_by_step = build_constraint_radii(
        transition_deltas=transition_deltas,
        total_budget_rho=total_budget_rho,
        time_horizon=time_horizon,
    ).to(device=device, dtype=dtype)
    return sigma_levels, transition_deltas, radius_by_step


def _path_step_value(values: torch.Tensor, step_idx: int) -> torch.Tensor:
    if values.ndim == 1:
        return values[step_idx]
    if values.ndim == 2:
        return values[:, step_idx]
    raise ValueError(f"Expected rank-1 or rank-2 path values, got shape={tuple(values.shape)}")


def _batch_vector(value: Union[torch.Tensor, float], *, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return torch.full((batch_size,), float(value.item()), device=device, dtype=dtype)
        if value.ndim == 1 and int(value.shape[0]) == int(batch_size):
            return value.to(device=device, dtype=dtype)
        raise ValueError(
            "Expected scalar tensor or per-example vector with length batch_size, "
            f"got shape={tuple(value.shape)} for batch_size={batch_size}"
        )
    return torch.full((batch_size,), float(value), device=device, dtype=dtype)


def _step_scalar_like(value: Union[torch.Tensor, float], reference: torch.Tensor) -> torch.Tensor:
    batch_vector = _batch_vector(
        value,
        batch_size=int(reference.shape[0]),
        device=reference.device,
        dtype=reference.dtype,
    )
    return batch_vector.view(reference.shape[0], *([1] * (reference.ndim - 1)))


def rollout_path_heuristic_attack(
    *,
    attack_net,
    x0: torch.Tensor,
    sigma_levels: torch.Tensor,
    transition_deltas: torch.Tensor,
    radius_by_step: torch.Tensor,
    inner_steps: int,
    step_size: float,
    sigma_data: float,
    labels=None,
    augment_labels=None,
) -> CDRORollout:
    batch_size = x0.shape[0]
    n_steps = int(sigma_levels.shape[-1] - 1)
    x_ref = x0.detach()
    x_ctrl = x0.detach()
    path_ref = [x_ref]
    path_ctrl = [x_ctrl]
    path_delta = []
    path_control = []

    was_training = attack_net.training
    attack_net.eval()
    try:
        for step_idx in range(n_steps):
            sigma_k = _path_step_value(sigma_levels, step_idx)
            sigma_next = _path_step_value(sigma_levels, step_idx + 1)
            variance_increment = sigma_next.square() - sigma_k.square()
            if float(variance_increment.min().item()) < -1e-12:
                raise ValueError("VE variance increments must be nonnegative.")
            delta_sigma = torch.sqrt(variance_increment.clamp_min(0.0))
            base_increment = _step_scalar_like(delta_sigma, x0) * torch.randn_like(x0)
            reference_state = (x_ref + base_increment).detach()
            x_nominal_next = (x_ctrl + base_increment).detach()
            sigma_batch = _batch_vector(
                sigma_next,
                batch_size=batch_size,
                device=x0.device,
                dtype=torch.float32,
            )

            control = torch.zeros_like(x_ctrl)
            delta_tau = _batch_vector(
                _path_step_value(transition_deltas, step_idx),
                batch_size=batch_size,
                device=x0.device,
                dtype=x0.dtype,
            )
            delta_tau_full = _step_scalar_like(delta_tau, x0)
            radius_step = _batch_vector(
                _path_step_value(radius_by_step, step_idx),
                batch_size=batch_size,
                device=x0.device,
                dtype=x0.dtype,
            )
            control_cap = torch.where(
                delta_tau > 0.0,
                radius_step / delta_tau.clamp_min(1e-12),
                torch.zeros_like(radius_step),
            )
            if float(delta_tau.max().item()) > 0.0 and torch.any(control_cap > 0.0) and int(inner_steps) > 0:
                if int(inner_steps) == 1:
                    control = control.requires_grad_(True)
                    candidate = x_nominal_next + delta_tau_full * control
                    loss_step = weighted_edm_loss_per_pixel(
                        net=attack_net,
                        x_noisy=candidate,
                        x_clean=x0,
                        sigma=sigma_batch,
                        sigma_data=sigma_data,
                        labels=labels,
                        augment_labels=augment_labels,
                    )
                    grad = torch.autograd.grad(reduce_per_sample(loss_step).mean(), control, only_inputs=True)[0]
                    grad_unit = l2_normalize_per_sample(grad)
                    control = (grad_unit.reshape(batch_size, -1) * control_cap[:, None]).reshape_as(grad).detach()
                else:
                    for _ in range(int(inner_steps)):
                        control = control.requires_grad_(True)
                        candidate = x_nominal_next + delta_tau_full * control
                        loss_step = weighted_edm_loss_per_pixel(
                            net=attack_net,
                            x_noisy=candidate,
                            x_clean=x0,
                            sigma=sigma_batch,
                            sigma_data=sigma_data,
                            labels=labels,
                            augment_labels=augment_labels,
                        )
                        grad = torch.autograd.grad(reduce_per_sample(loss_step).mean(), control, only_inputs=True)[0]
                        control = project_l2_ball(
                            control.detach() + float(step_size) * l2_normalize_per_sample(grad),
                            control_cap,
                        ).detach()

            delta_effective = delta_tau_full * control
            x_ref = reference_state
            x_ctrl = (x_nominal_next + delta_effective).detach()
            path_ref.append(x_ref)
            path_ctrl.append(x_ctrl)
            path_delta.append(delta_effective)
            path_control.append(control.detach())
    finally:
        if was_training:
            attack_net.train()

    return CDRORollout(
        states_ref=torch.stack(path_ref, dim=1),
        states_ctrl=torch.stack(path_ctrl, dim=1),
        delta_path=torch.stack(path_delta, dim=1),
        control_path=torch.stack(path_control, dim=1),
        sigma_levels=sigma_levels,
        transition_deltas=transition_deltas,
        radius_by_step=radius_by_step,
    )
