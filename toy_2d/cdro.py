from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Mapping

import torch

# Keep this module as the stable CDRO integration layer.
# New CDRO variants should register a solver here instead of changing the
# training loop, plotting code, or benchmark plumbing again.
DEFAULT_CDRO_IMPLEMENTATION = "path_heuristic"


@dataclass(frozen=True)
class CdroConfig:
    p_mean: float
    p_std: float
    sigma_min: float
    sigma_max: float
    rho: float
    path_steps: int
    inner_steps: int
    step_size: float
    gamma: float
    total_budget: float | None = None
    exact_budget_split: bool = False
    sigma_schedule: str = "karras_grid"
    implementation: str = DEFAULT_CDRO_IMPLEMENTATION

    @classmethod
    def from_args(cls, args) -> "CdroConfig":
        return cls(
            p_mean=float(args.p_mean),
            p_std=float(args.p_std),
            sigma_min=float(args.sigma_min),
            sigma_max=float(args.sigma_max),
            rho=float(args.rho),
            path_steps=int(args.cdro_path_steps),
            inner_steps=int(args.cdro_inner_steps),
            step_size=float(args.cdro_step_size),
            gamma=float(args.cdro_gamma),
            total_budget=_maybe_float(args.cdro_total_budget),
            exact_budget_split=bool(args.cdro_exact_budget_split),
            sigma_schedule=str(args.cdro_sigma_schedule),
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "CdroConfig":
        return cls(
            p_mean=float(payload["p_mean"]),
            p_std=float(payload["p_std"]),
            sigma_min=float(payload["sigma_min"]),
            sigma_max=float(payload["sigma_max"]),
            rho=float(payload["rho"]),
            path_steps=int(payload["cdro_path_steps"]),
            inner_steps=int(payload["cdro_inner_steps"]),
            step_size=float(payload["cdro_step_size"]),
            gamma=float(payload["cdro_gamma"]),
            total_budget=_maybe_float(payload.get("cdro_total_budget")),
            exact_budget_split=bool(payload.get("cdro_exact_budget_split", False)),
            sigma_schedule=str(payload.get("cdro_sigma_schedule", "karras_grid")),
            implementation=str(payload.get("cdro_implementation", DEFAULT_CDRO_IMPLEMENTATION)),
        )

    def with_total_budget(self, total_budget: float | None) -> "CdroConfig":
        return replace(self, total_budget=total_budget)


@dataclass
class CdroBatchStats:
    mean_delta_norm: float
    max_delta_norm: float
    mean_transport_cost: float
    max_transport_cost: float
    mean_total_transport_cost: float
    max_total_transport_cost: float
    target_total_budget: float | None


@dataclass
class CdroAttackResult:
    config: CdroConfig
    sigmas: torch.Tensor
    reference_path: torch.Tensor
    adv_path: torch.Tensor
    stats: CdroBatchStats


CdroAttackSolver = Callable[..., torch.Tensor]
_CDRO_SOLVERS: dict[str, CdroAttackSolver] = {}


def register_cdro_solver(name: str, solver: CdroAttackSolver) -> None:
    normalized_name = str(name).strip()
    if not normalized_name:
        raise ValueError("CDRO solver name must be non-empty.")
    _CDRO_SOLVERS[normalized_name] = solver


def list_cdro_implementations() -> tuple[str, ...]:
    return tuple(sorted(_CDRO_SOLVERS))


def get_cdro_solver(name: str) -> CdroAttackSolver:
    try:
        return _CDRO_SOLVERS[name]
    except KeyError as exc:
        available = ", ".join(list_cdro_implementations()) or "<none>"
        raise ValueError(f"Unsupported CDRO implementation: {name}. Available: {available}") from exc


def cdro_loss(
    *,
    train_net,
    clean_points: torch.Tensor,
    config: CdroConfig,
    shared_noise: bool = False,
    attack_net=None,
) -> tuple[torch.Tensor, CdroBatchStats]:
    attack_result = solve_cdro_attack(
        attack_net=attack_net if attack_net is not None else train_net,
        clean_points=clean_points,
        config=config,
        shared_noise=shared_noise,
    )
    outer_loss = _per_time_losses(
        net=train_net,
        clean_points=clean_points,
        adv_path=attack_result.adv_path,
        sigmas=attack_result.sigmas,
    ).mean()
    return outer_loss, attack_result.stats


def solve_cdro_attack(
    *,
    attack_net,
    clean_points: torch.Tensor,
    config: CdroConfig,
    shared_noise: bool = False,
) -> CdroAttackResult:
    sigmas, reference_path = build_cdro_reference_path(
        clean_points=clean_points,
        config=config,
        shared_noise=shared_noise,
    )
    solver = get_cdro_solver(config.implementation)
    adv_path = solver(
        attack_net=attack_net,
        clean_points=clean_points,
        reference_path=reference_path,
        sigmas=sigmas,
        config=config,
    )
    return CdroAttackResult(
        config=config,
        sigmas=sigmas,
        reference_path=reference_path,
        adv_path=adv_path,
        stats=summarize_cdro_attack(
            reference_path=reference_path,
            adv_path=adv_path,
            target_total_budget=config.total_budget,
        ),
    )


def build_cdro_reference_path(
    *,
    clean_points: torch.Tensor,
    config: CdroConfig,
    shared_noise: bool,
    sigmas: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    sigma_values = sigmas if sigmas is not None else build_cdro_sigmas(clean_points=clean_points, config=config)
    return sigma_values, build_forward_path(clean_points=clean_points, sigmas=sigma_values, shared_noise=shared_noise)


def build_cdro_sigmas(*, clean_points: torch.Tensor, config: CdroConfig) -> torch.Tensor:
    return sample_path_sigmas(
        batch_size=clean_points.shape[0],
        num_steps=config.path_steps,
        p_mean=config.p_mean,
        p_std=config.p_std,
        sigma_min=config.sigma_min,
        sigma_max=config.sigma_max,
        rho=config.rho,
        device=clean_points.device,
        dtype=clean_points.dtype,
        schedule=config.sigma_schedule,
    )


def summarize_cdro_attack(
    *,
    reference_path: torch.Tensor,
    adv_path: torch.Tensor,
    target_total_budget: float | None,
) -> CdroBatchStats:
    displacement = (adv_path - reference_path).detach()
    norms = displacement.norm(dim=2)
    per_step_transport_cost = 0.5 * displacement.square().sum(dim=2)
    total_transport_cost = per_step_transport_cost.sum(dim=1)
    return CdroBatchStats(
        mean_delta_norm=float(norms.mean().item()),
        max_delta_norm=float(norms.max().item()),
        mean_transport_cost=float(per_step_transport_cost.mean().item()),
        max_transport_cost=float(per_step_transport_cost.max().item()),
        mean_total_transport_cost=float(total_transport_cost.mean().item()),
        max_total_transport_cost=float(total_transport_cost.max().item()),
        target_total_budget=target_total_budget,
    )


def sample_path_sigmas(
    *,
    batch_size: int,
    num_steps: int,
    p_mean: float,
    p_std: float,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    device: torch.device,
    dtype: torch.dtype,
    schedule: str,
) -> torch.Tensor:
    if num_steps < 1:
        raise ValueError("num_steps must be at least 1.")
    if schedule == "edm_random":
        random_sigmas = sample_edm_random_sigmas(
            batch_size=batch_size,
            num_steps=num_steps,
            p_mean=p_mean,
            p_std=p_std,
            device=device,
            dtype=dtype,
        )
        return torch.sort(random_sigmas, dim=1).values
    if schedule == "edm_quantiles":
        shared = build_edm_quantile_sigmas(
            num_steps=num_steps,
            p_mean=p_mean,
            p_std=p_std,
            device=device,
            dtype=dtype,
        )
        shared = torch.sort(shared).values
        return shared.view(1, num_steps).expand(batch_size, num_steps)
    if schedule != "karras_grid":
        raise ValueError(f"Unsupported sigma schedule: {schedule}")
    shared = build_forward_karras_sigmas(
        num_steps=num_steps,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        rho=rho,
        device=device,
        dtype=dtype,
    )
    return shared.view(1, num_steps).expand(batch_size, num_steps)


def sample_edm_random_sigmas(
    *,
    batch_size: int,
    num_steps: int,
    p_mean: float,
    p_std: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    rnd_normal = torch.randn(batch_size, num_steps, device=device, dtype=dtype)
    return torch.exp(rnd_normal * p_std + p_mean)


def build_edm_quantile_sigmas(
    *,
    num_steps: int,
    p_mean: float,
    p_std: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    quantiles = 1.0 - (torch.arange(num_steps, device=device, dtype=dtype) + 0.5) / num_steps
    normal_quantiles = torch.erfinv(2.0 * quantiles - 1.0) * (2.0**0.5)
    return torch.exp(normal_quantiles * p_std + p_mean)


def build_forward_karras_sigmas(
    *,
    num_steps: int,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    ramp = torch.linspace(0.0, 1.0, num_steps, device=device, dtype=dtype)
    min_inv_rho = sigma_min ** (1.0 / rho)
    max_inv_rho = sigma_max ** (1.0 / rho)
    descending = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return torch.flip(descending, dims=[0])


def build_forward_path(*, clean_points: torch.Tensor, sigmas: torch.Tensor, shared_noise: bool) -> torch.Tensor:
    batch_size, data_dim = clean_points.shape
    sigma_matrix = expand_sigmas(sigmas=sigmas, batch_size=batch_size)
    num_steps = sigma_matrix.shape[1]
    sigma_prev = torch.cat(
        [torch.zeros(batch_size, 1, device=clean_points.device, dtype=clean_points.dtype), sigma_matrix[:, :-1]],
        dim=1,
    )
    increment_scale = torch.sqrt(torch.clamp(sigma_matrix.square() - sigma_prev.square(), min=0.0))

    if shared_noise:
        eps = torch.randn(batch_size, 1, data_dim, device=clean_points.device, dtype=clean_points.dtype)
        eps = eps.expand(batch_size, num_steps, data_dim)
    else:
        eps = torch.randn(batch_size, num_steps, data_dim, device=clean_points.device, dtype=clean_points.dtype)

    increments = increment_scale[:, :, None] * eps
    return clean_points[:, None, :] + torch.cumsum(increments, dim=1)


def _per_time_losses(*, net, clean_points: torch.Tensor, adv_path: torch.Tensor, sigmas: torch.Tensor) -> torch.Tensor:
    batch_size, num_steps, data_dim = adv_path.shape
    sigma_matrix = expand_sigmas(sigmas=sigmas, batch_size=batch_size)
    flat_inputs = adv_path.reshape(batch_size * num_steps, data_dim)
    flat_sigmas = sigma_matrix.reshape(batch_size * num_steps)
    flat_targets = clean_points[:, None, :].expand(batch_size, num_steps, data_dim).reshape(batch_size * num_steps, data_dim)

    sigma = flat_sigmas.view(-1, 1)
    sigma_data_sq = net.sigma_data ** 2
    weight = (sigma.square() + sigma_data_sq) / (sigma * net.sigma_data).square()
    denoised = net(flat_inputs, flat_sigmas)
    sq_error = (denoised - flat_targets).square().mean(dim=1)
    return (weight.squeeze(1) * sq_error).view(batch_size, num_steps)


def expand_sigmas(*, sigmas: torch.Tensor, batch_size: int) -> torch.Tensor:
    if sigmas.ndim == 1:
        return sigmas.view(1, -1).expand(batch_size, sigmas.shape[0])
    if sigmas.ndim == 2 and sigmas.shape[0] == batch_size:
        return sigmas
    raise ValueError(f"sigmas must have shape [steps] or [batch, steps], got {tuple(sigmas.shape)}")


def _maybe_float(value) -> float | None:
    if value is None:
        return None
    return float(value)


from toy_2d.cdro_heuristic import solve_path_heuristic_cdro_attack

register_cdro_solver(DEFAULT_CDRO_IMPLEMENTATION, solve_path_heuristic_cdro_attack)
