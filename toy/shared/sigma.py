import math
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class LogSigmaQuantileLadder:
    """Shared representation for a warmup-quantile VE continuation ladder."""

    sigma_levels: torch.Tensor
    log_sigma_edges: Optional[torch.Tensor] = None
    log_sigma_nodes: Optional[torch.Tensor] = None
    quantile_nodes: Optional[torch.Tensor] = None


def build_sigma_levels(sigma_min: float, sigma_max: float, n_steps: int, device: torch.device) -> torch.Tensor:
    """Build VE noise ladder [sigma_0, ..., sigma_N] with sigma_0 = 0."""

    return torch.cat(
        [
            torch.zeros(1, device=device),
            torch.logspace(math.log10(sigma_min), math.log10(sigma_max), n_steps, device="cpu").to(device),
        ]
    )


def build_rf_time_quantile_levels(
    sigma_max: float,
    n_steps: int,
    device: torch.device,
    *,
    distribution: str = "u_shaped",
) -> torch.Tensor:
    """Build a right-endpoint RF solver/continuation ladder.

    The returned tensor still uses the repo's sigma-shaped API, with
    `sigma = t * sigma_max`; unlike EDM ladders, these values are RF clock
    coordinates rather than noise scales. The positive levels `sigma_levels[1:]`
    are right-endpoint quantiles `F^{-1}(k / N)` of the chosen RF time law,
    preserving endpoint-inclusive solver/reference rollouts.
    """

    if n_steps <= 0:
        raise ValueError(f"n_steps must be > 0, got {n_steps}")
    sigma_max_value = float(sigma_max)
    if sigma_max_value <= 0.0:
        raise ValueError(f"sigma_max must be > 0 for RF time levels, got {sigma_max}")
    probs = torch.linspace(0.0, 1.0, int(n_steps) + 1, device=device)
    mode = str(distribution).strip().lower()
    if mode == "u_shaped":
        # Inverse CDF of Beta(1/2, 1/2): F^{-1}(u) = sin^2(pi u / 2).
        t_levels = torch.sin(0.5 * math.pi * probs).square()
    elif mode == "uniform":
        t_levels = probs
    else:
        raise ValueError(
            f"Unsupported RF time distribution '{distribution}'. Expected one of: u_shaped, uniform."
        )
    return t_levels * sigma_max_value


def sample_rf_time_stratified_levels(
    sigma_max: float,
    n_steps: int,
    device: torch.device,
    *,
    distribution: str = "u_shaped",
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Sample one stratified RF time ladder from the stage time law.

    The positive levels are sampled from open quantile cells
    `((k-1)/N, k/N)`, so the resulting grid matches the baseline RF
    random-time law in stratified Monte Carlo form rather than the
    endpoint-inclusive deterministic quantile grid.
    """

    if n_steps <= 0:
        raise ValueError(f"n_steps must be > 0, got {n_steps}")
    sigma_max_value = float(sigma_max)
    if sigma_max_value <= 0.0:
        raise ValueError(f"sigma_max must be > 0 for RF time levels, got {sigma_max}")
    out_dtype = torch.float32 if dtype is None else dtype
    jitter = torch.empty(int(n_steps), dtype=torch.float64, device=device).uniform_(1e-12, 1.0 - 1e-12)
    strata = torch.arange(0, int(n_steps), dtype=torch.float64, device=device)
    probs = (strata + jitter) / float(n_steps)
    mode = str(distribution).strip().lower()
    if mode == "u_shaped":
        t_positive = torch.sin(0.5 * math.pi * probs).square()
    elif mode == "uniform":
        t_positive = probs
    else:
        raise ValueError(
            f"Unsupported RF time distribution '{distribution}'. Expected one of: u_shaped, uniform."
        )
    if t_positive.numel() > 1 and not torch.all(t_positive[1:] > t_positive[:-1]):
        raise ValueError("Stratified RF time ladder must be strictly increasing.")
    sigma_levels = torch.cat([torch.zeros(1, dtype=torch.float64, device=device), t_positive * sigma_max_value]).to(
        device=device,
        dtype=out_dtype,
    )
    positive_out = sigma_levels[1:]
    if positive_out.numel() > 1 and not torch.all(positive_out[1:] > positive_out[:-1]):
        raise ValueError("Stratified RF time ladder must remain strictly increasing after dtype conversion.")
    return sigma_levels


def sample_rf_time_stratified_levels_batch(
    sigma_max: float,
    n_steps: int,
    device: torch.device,
    *,
    batch_size: int,
    distribution: str = "u_shaped",
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Sample one independent stratified RF time ladder per batch element."""

    if n_steps <= 0:
        raise ValueError(f"n_steps must be > 0, got {n_steps}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    sigma_max_value = float(sigma_max)
    if sigma_max_value <= 0.0:
        raise ValueError(f"sigma_max must be > 0 for RF time levels, got {sigma_max}")
    out_dtype = torch.float32 if dtype is None else dtype
    jitter = torch.empty((int(batch_size), int(n_steps)), dtype=torch.float64, device=device).uniform_(
        1e-12,
        1.0 - 1e-12,
    )
    strata = torch.arange(0, int(n_steps), dtype=torch.float64, device=device).view(1, int(n_steps))
    probs = (strata + jitter) / float(n_steps)
    mode = str(distribution).strip().lower()
    if mode == "u_shaped":
        t_positive = torch.sin(0.5 * math.pi * probs).square()
    elif mode == "uniform":
        t_positive = probs
    else:
        raise ValueError(
            f"Unsupported RF time distribution '{distribution}'. Expected one of: u_shaped, uniform."
        )
    if t_positive.shape[1] > 1 and not torch.all(t_positive[:, 1:] > t_positive[:, :-1]):
        raise ValueError("Per-example stratified RF time ladders must be strictly increasing.")
    sigma_levels = torch.cat(
        [
            torch.zeros((int(batch_size), 1), dtype=torch.float64, device=device),
            t_positive * sigma_max_value,
        ],
        dim=1,
    ).to(device=device, dtype=out_dtype)
    positive_out = sigma_levels[:, 1:]
    if positive_out.shape[1] > 1 and not torch.all(positive_out[:, 1:] > positive_out[:, :-1]):
        raise ValueError("Per-example stratified RF time ladders must remain strictly increasing after dtype conversion.")
    return sigma_levels


def resolve_rf_stage_t_distribution(
    stage_name: str,
    *,
    reflow_distribution: str = "u_shaped",
) -> str:
    """Resolve the clean/robust RF timestep law used by a named training stage."""

    stage = str(stage_name).strip().lower()
    if stage in ("rf_stage1", "stage1", "data_noise"):
        return "uniform"
    if stage in ("rf_reflow", "reflow"):
        mode = str(reflow_distribution).strip().lower()
        if mode not in ("u_shaped", "uniform"):
            raise ValueError(
                f"Unsupported RF reflow timestep distribution '{reflow_distribution}'. "
                "Expected one of: uniform, u_shaped."
            )
        return mode
    raise ValueError(
        f"Unsupported RF stage '{stage_name}'. Expected one of: rf_stage1, rf_reflow, stage1, reflow, data_noise."
    )


def resolve_rf_teacher_n_steps_path(cfg) -> int:
    """Resolve the shared RF teacher/reflow grid step count."""

    override = int(getattr(cfg, "rf_teacher_n_steps_path", 0) or 0)
    if override > 0:
        return override
    return max(int(getattr(cfg, "n_steps_path", 1)), 1)


def resolve_rf_eval_n_steps_path(cfg) -> int:
    """Resolve the shared RF evaluation/FID grid step count."""

    override = int(getattr(cfg, "rf_eval_n_steps_path", 0) or 0)
    if override > 0:
        return override
    return resolve_rf_teacher_n_steps_path(cfg)


def build_rf_stage_time_quantile_levels(
    sigma_max: float,
    n_steps: int,
    device: torch.device,
    *,
    stage_name: str,
    reflow_distribution: str = "u_shaped",
) -> torch.Tensor:
    """Build the RF time grid aligned to the clean timestep law of a named stage."""

    distribution = resolve_rf_stage_t_distribution(
        stage_name,
        reflow_distribution=reflow_distribution,
    )
    return build_rf_time_quantile_levels(
        sigma_max,
        n_steps,
        device,
        distribution=distribution,
    )


def sample_rf_stage_time_stratified_levels(
    sigma_max: float,
    n_steps: int,
    device: torch.device,
    *,
    stage_name: str,
    reflow_distribution: str = "u_shaped",
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Sample a stratified RF time ladder aligned to the clean stage law."""

    distribution = resolve_rf_stage_t_distribution(
        stage_name,
        reflow_distribution=reflow_distribution,
    )
    return sample_rf_time_stratified_levels(
        sigma_max,
        n_steps,
        device,
        distribution=distribution,
        dtype=dtype,
    )


def sample_rf_stage_time_stratified_levels_batch(
    sigma_max: float,
    n_steps: int,
    device: torch.device,
    *,
    batch_size: int,
    stage_name: str,
    reflow_distribution: str = "u_shaped",
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Sample one stratified RF time ladder per batch element aligned to the clean stage law."""

    distribution = resolve_rf_stage_t_distribution(
        stage_name,
        reflow_distribution=reflow_distribution,
    )
    return sample_rf_time_stratified_levels_batch(
        sigma_max,
        n_steps,
        device,
        batch_size=batch_size,
        distribution=distribution,
        dtype=dtype,
    )


def _truncated_log_sigma_cdf_bounds(
    *,
    sigma_min: float,
    sigma_max: float,
    p_mean: float,
    p_std: float,
) -> tuple[torch.distributions.Normal, torch.Tensor, torch.Tensor]:
    """Return the warmup-law normal and its truncated CDF bounds in log-sigma space."""

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
        raise ValueError("Truncated warmup-law support has zero mass on [sigma_min, sigma_max].")
    return normal, cdf_min, cdf_max


def sample_sigmas_log_normal(
    batch_size: int,
    *,
    sigma_min: float,
    sigma_max: float,
    device: torch.device,
    p_mean: float = -1.2,
    p_std: float = 1.2,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Sample continuous EDM warmup sigmas from the clamped log-normal law."""

    out_dtype = torch.float32 if dtype is None else dtype
    rnd = torch.randn(batch_size, device=device, dtype=out_dtype)
    sigma = torch.exp(rnd * float(p_std) + float(p_mean))
    return sigma.clamp(min=float(sigma_min), max=float(sigma_max))


def assign_sigmas_to_nearest_levels(sigmas: torch.Tensor, sigma_levels: torch.Tensor) -> torch.Tensor:
    """Map continuous sigmas to the nearest positive ladder point for diagnostics only."""

    if sigma_levels.ndim == 1:
        if sigma_levels.numel() < 2:
            raise ValueError(f"sigma_levels must contain at least 2 values, got {sigma_levels.numel()}")
        d = (sigmas.unsqueeze(1) - sigma_levels[1:].unsqueeze(0)).abs()
        nearest = torch.argmin(d, dim=1)
        return nearest + 1
    if sigma_levels.ndim == 2:
        if sigma_levels.shape[1] < 2:
            raise ValueError(
                "Per-example sigma_levels must contain at least 2 ladder points, "
                f"got shape={tuple(sigma_levels.shape)}"
            )
        if sigmas.ndim != 1 or sigmas.shape[0] != sigma_levels.shape[0]:
            raise ValueError(
                "Per-example sigma snapping expects sigmas with shape [batch], got "
                f"{tuple(sigmas.shape)} for sigma_levels {tuple(sigma_levels.shape)}"
            )
        d = (sigmas.unsqueeze(1) - sigma_levels[:, 1:]).abs()
        nearest = torch.argmin(d, dim=1)
        return nearest + 1
    raise ValueError(f"sigma_levels must be rank 1 or 2, got shape={tuple(sigma_levels.shape)}")


def build_sigma_levels_from_warmup_quantiles(
    sigma_min: float,
    sigma_max: float,
    n_steps: int,
    device: torch.device,
    p_mean: float = -1.2,
    p_std: float = 1.2,
) -> torch.Tensor:
    """Build the positive continuation ladder from warmup-law midpoint quantiles."""

    return build_log_sigma_quantile_ladder(
        sigma_min,
        sigma_max,
        n_steps,
        p_mean=p_mean,
        p_std=p_std,
        device=device,
        dtype=torch.float32,
    ).sigma_levels


def build_log_sigma_quantile_ladder(
    sigma_min: float,
    sigma_max: float,
    n_steps: int,
    *,
    p_mean: float = -1.2,
    p_std: float = 1.2,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> LogSigmaQuantileLadder:
    """Build midpoint states and exact cell edges from one shared quantile ladder."""

    if n_steps <= 0:
        raise ValueError(f"n_steps must be > 0, got {n_steps}")
    normal, cdf_min, cdf_max = _truncated_log_sigma_cdf_bounds(
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        p_mean=p_mean,
        p_std=p_std,
    )
    out_device = torch.device("cpu") if device is None else device
    out_dtype = torch.float32 if dtype is None else dtype

    mass = cdf_max - cdf_min
    probs_mid = (torch.arange(1, n_steps + 1, dtype=torch.float64) - 0.5) / float(n_steps)
    trunc_mid_cdf = cdf_min + probs_mid * mass
    z_mid = normal.icdf(trunc_mid_cdf.clamp(min=1e-12, max=1.0 - 1e-12))

    z_min = torch.tensor(math.log(float(sigma_min)), dtype=torch.float64)
    z_max = torch.tensor(math.log(float(sigma_max)), dtype=torch.float64)
    if n_steps == 1:
        z_edges = torch.stack([z_min, z_max])
    else:
        probs_edge = torch.arange(1, n_steps, dtype=torch.float64) / float(n_steps)
        trunc_edge_cdf = cdf_min + probs_edge * mass
        z_inner = normal.icdf(trunc_edge_cdf.clamp(min=1e-12, max=1.0 - 1e-12))
        z_edges = torch.cat([z_min.view(1), z_inner, z_max.view(1)])

    sigma_positive = torch.exp(z_mid)
    if sigma_positive.numel() > 1 and not torch.all(sigma_positive[1:] > sigma_positive[:-1]):
        raise ValueError("Warmup-quantile continuation ladder must be strictly increasing.")
    if not torch.all(z_edges[1:] > z_edges[:-1]):
        raise ValueError("Warmup-quantile cell edges must be strictly increasing.")

    sigma_levels = torch.cat(
        [
            torch.zeros(1, dtype=torch.float64),
            sigma_positive,
        ]
    ).to(device=out_device, dtype=out_dtype)
    log_sigma_edges = z_edges.to(device=out_device, dtype=out_dtype)
    return LogSigmaQuantileLadder(
        sigma_levels=sigma_levels,
        log_sigma_edges=log_sigma_edges,
        log_sigma_nodes=z_mid.to(device=out_device, dtype=out_dtype),
        quantile_nodes=probs_mid.to(device=out_device, dtype=out_dtype),
    )


def sample_log_sigma_stratified_quantile_ladder(
    sigma_min: float,
    sigma_max: float,
    n_steps: int,
    *,
    p_mean: float = -1.2,
    p_std: float = 1.2,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> LogSigmaQuantileLadder:
    """Sample one stratified inverse-CDF ladder from the truncated warmup log-sigma law."""

    if n_steps <= 0:
        raise ValueError(f"n_steps must be > 0, got {n_steps}")
    normal, cdf_min, cdf_max = _truncated_log_sigma_cdf_bounds(
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        p_mean=p_mean,
        p_std=p_std,
    )
    out_device = torch.device("cpu") if device is None else device
    out_dtype = torch.float32 if dtype is None else dtype

    mass = cdf_max - cdf_min
    # Open-cell jitter avoids exact boundary samples, so Delta_tau_1 stays positive.
    jitter = torch.empty(int(n_steps), dtype=torch.float64).uniform_(1e-12, 1.0 - 1e-12)
    strata = torch.arange(0, int(n_steps), dtype=torch.float64)
    probs = (strata + jitter) / float(n_steps)
    trunc_cdf = cdf_min + probs * mass
    z_nodes = normal.icdf(trunc_cdf.clamp(min=1e-12, max=1.0 - 1e-12))
    sigma_positive = torch.exp(z_nodes)
    if torch.any(sigma_positive <= 0):
        raise ValueError("Stratified warmup-quantile continuation ladder must be strictly positive.")
    if sigma_positive.numel() > 1 and not torch.all(sigma_positive[1:] > sigma_positive[:-1]):
        raise ValueError("Stratified warmup-quantile continuation ladder must be strictly increasing.")

    sigma_levels = torch.cat(
        [
            torch.zeros(1, dtype=torch.float64),
            sigma_positive,
        ]
    ).to(device=out_device, dtype=out_dtype)
    positive_out = sigma_levels[1:]
    if torch.any(positive_out <= 0):
        raise ValueError("Stratified warmup-quantile continuation ladder must remain positive after dtype conversion.")
    if positive_out.numel() > 1 and not torch.all(positive_out[1:] > positive_out[:-1]):
        raise ValueError(
            "Stratified warmup-quantile continuation ladder must remain strictly increasing after dtype conversion."
        )

    return LogSigmaQuantileLadder(
        sigma_levels=sigma_levels,
        log_sigma_nodes=z_nodes.to(device=out_device, dtype=out_dtype),
        quantile_nodes=probs.to(device=out_device, dtype=out_dtype),
    )


def sample_log_sigma_stratified_quantile_ladder_batch(
    sigma_min: float,
    sigma_max: float,
    n_steps: int,
    *,
    batch_size: int,
    p_mean: float = -1.2,
    p_std: float = 1.2,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> LogSigmaQuantileLadder:
    """Sample one independent stratified inverse-CDF ladder per batch element."""

    if n_steps <= 0:
        raise ValueError(f"n_steps must be > 0, got {n_steps}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    normal, cdf_min, cdf_max = _truncated_log_sigma_cdf_bounds(
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        p_mean=p_mean,
        p_std=p_std,
    )
    out_device = torch.device("cpu") if device is None else device
    out_dtype = torch.float32 if dtype is None else dtype

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
    ).to(device=out_device, dtype=out_dtype)
    positive_out = sigma_levels[:, 1:]
    if torch.any(positive_out <= 0):
        raise ValueError("Per-example stratified continuation ladders must remain positive after dtype conversion.")
    if positive_out.shape[1] > 1 and not torch.all(positive_out[:, 1:] > positive_out[:, :-1]):
        raise ValueError(
            "Per-example stratified continuation ladders must remain strictly increasing after dtype conversion."
        )

    return LogSigmaQuantileLadder(
        sigma_levels=sigma_levels,
        log_sigma_nodes=z_nodes.to(device=out_device, dtype=out_dtype),
        quantile_nodes=probs.to(device=out_device, dtype=out_dtype),
    )


def build_log_sigma_quantile_cell_edges(
    sigma_min: float,
    sigma_max: float,
    n_steps: int,
    *,
    p_mean: float = -1.2,
    p_std: float = 1.2,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Build exact log-sigma cell edges for the midpoint-quantile continuation grid."""

    ladder = build_log_sigma_quantile_ladder(
        sigma_min,
        sigma_max,
        n_steps,
        p_mean=p_mean,
        p_std=p_std,
        device=device,
        dtype=dtype,
    )
    if ladder.log_sigma_edges is None:
        raise RuntimeError("Midpoint warmup-quantile ladder did not return log-sigma cell edges.")
    return ladder.log_sigma_edges


def sample_target_indices(batch_size: int, sigma_levels: torch.Tensor) -> torch.Tensor:
    """Uniformly sample training target steps i in {1, ..., N}."""

    if sigma_levels.ndim not in (1, 2):
        raise ValueError(f"sigma_levels must be rank 1 or 2, got shape={tuple(sigma_levels.shape)}")
    return torch.randint(1, sigma_levels.shape[-1], (batch_size,), device=sigma_levels.device)


def sample_target_indices_log_normal(
    batch_size: int,
    sigma_levels: torch.Tensor,
    p_mean: float = -1.2,
    p_std: float = 1.2,
) -> torch.Tensor:
    """EDM-style sigma sampling snapped to the nearest continuation ladder point."""

    if sigma_levels.ndim == 1:
        sigma_min_value = float(sigma_levels[1].item())
        sigma_max_value = float(sigma_levels[-1].item())
    elif sigma_levels.ndim == 2:
        sigma_min_value = float(sigma_levels[:, 1].min().item())
        sigma_max_value = float(sigma_levels[:, -1].max().item())
    else:
        raise ValueError(f"sigma_levels must be rank 1 or 2, got shape={tuple(sigma_levels.shape)}")
    sigma = sample_sigmas_log_normal(
        batch_size,
        sigma_min=sigma_min_value,
        sigma_max=sigma_max_value,
        device=sigma_levels.device,
        p_mean=p_mean,
        p_std=p_std,
        dtype=sigma_levels.dtype,
    )
    return assign_sigmas_to_nearest_levels(sigma, sigma_levels)
