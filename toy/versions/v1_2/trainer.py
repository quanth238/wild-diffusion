from typing import Callable, Optional

import torch

from ...models import set_requires_grad
from ...shared.objective import inner_objective_attack_only
from ...shared.train_utils import sample_train_batch
from ...utils import batch_scalar_like, has_nan_or_inf, scalarize


def _activation_scale(step: int, cfg) -> float:
    """CDRO activation gate schedule (toy-step analogue of kimg start/ramp)."""

    start_step = max(int(cfg.v12_start_step), 0)
    ramp_steps = max(int(cfg.v12_ramp_steps), 0)
    if step <= start_step:
        return 0.0
    if ramp_steps <= 0:
        return 1.0
    progress = (step - start_step) / float(max(ramp_steps, 1))
    return float(min(max(progress, 0.0), 1.0))


def _sample_sigma(cfg, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Sample per-sample EDM sigma ~ LogNormal(p_mean, p_std)."""

    rnd = torch.randn(batch_size, device=device, dtype=dtype)
    return torch.exp(rnd * float(cfg.p_std) + float(cfg.p_mean))


def _edm_weight(sigma: torch.Tensor, sigma_data: float) -> torch.Tensor:
    """EDM weighting used by CDRO-EDM loss."""

    sigma2 = sigma.square()
    return (sigma2 + float(sigma_data) ** 2) / (sigma * float(sigma_data)).square().clamp_min(1e-8)


def _cdro_inner_attack_loss(
    *,
    denoiser,
    x_noisy: torch.Tensor,
    x_clean: torch.Tensor,
    sigma: torch.Tensor,
    sigma_data: float,
) -> torch.Tensor:
    """Inner loss exactly as EDMLossCDRO.build_batch: mean over all tensor elements."""

    pred = denoiser(x_noisy, sigma)
    w = batch_scalar_like(_edm_weight(sigma, sigma_data), x_clean)
    return (w * (pred - x_clean).pow(2)).mean()


def _cdro_outer_loss_scalar(
    *,
    denoiser,
    x_noisy: torch.Tensor,
    x_clean: torch.Tensor,
    sigma: torch.Tensor,
    sigma_data: float,
) -> torch.Tensor:
    """Outer scalar loss equivalent to training_loop(loss_tensor.sum()/batch)."""

    pred = denoiser(x_noisy, sigma)
    w = batch_scalar_like(_edm_weight(sigma, sigma_data), x_clean)
    loss_tensor = w * (pred - x_clean).pow(2)
    return loss_tensor.reshape(x_clean.shape[0], -1).sum(dim=1).mean()


def _sigma_gate(
    sigma: torch.Tensor,
    *,
    activation_scale: float,
    sigma_floor: float,
    sigma_cut: float,
    gate_power: float,
) -> torch.Tensor:
    """Sigma gate from CDRO-EDM: high-noise steps receive weaker/no attack."""

    sigma_cut_safe = max(float(sigma_cut), 1e-8)
    gate = (1.0 - sigma / sigma_cut_safe).clamp(min=0.0, max=1.0)
    sigma_floor_safe = max(float(sigma_floor), 0.0)
    if sigma_floor_safe > 0.0:
        if sigma_floor_safe >= sigma_cut_safe:
            raise ValueError(
                f"sigma_floor ({sigma_floor_safe}) must be smaller than sigma_cut ({sigma_cut_safe})"
            )
        ramp = ((sigma - sigma_floor_safe) / (sigma_cut_safe - sigma_floor_safe)).clamp(min=0.0, max=1.0)
        gate = gate * ramp
    gate = gate * float(max(activation_scale, 0.0))
    return gate.pow(float(max(gate_power, 1e-8)))


def _delta_scale(sigma: torch.Tensor, gate: torch.Tensor, delta_space: str) -> torch.Tensor:
    """Convert raw delta parameterization to applied perturbation scale."""

    if delta_space == "noise":
        return gate * sigma
    if delta_space == "image":
        return gate
    raise ValueError(f"Unsupported delta_space='{delta_space}'. Expected one of: image, noise.")


def _applied_delta(delta_raw: torch.Tensor, sigma: torch.Tensor, gate: torch.Tensor, delta_space: str) -> torch.Tensor:
    scale = _delta_scale(sigma, gate, delta_space)
    return batch_scalar_like(scale, delta_raw) * delta_raw


def _delta_radius(*, sigma: torch.Tensor, gate: torch.Tensor, max_delta: float, delta_space: str) -> torch.Tensor:
    """RMS-L2 radius used to project raw delta."""

    if delta_space == "noise":
        scale = _delta_scale(sigma, gate, delta_space).clamp_min(1e-8)
        return torch.full_like(sigma, float(max_delta)) / scale
    return torch.full_like(sigma, float(max_delta))


def _project_l2_rms_ball(delta_raw: torch.Tensor, radius: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Project each sample onto an RMS-L2 ball with per-sample radius."""

    flat = delta_raw.reshape(delta_raw.shape[0], -1)
    rms = (flat.square().mean(dim=1, keepdim=True) + eps).sqrt()
    if radius.ndim == 1:
        radius = radius[:, None]
    scale = torch.minimum(torch.ones_like(rms), radius / rms)
    return (flat * scale).reshape_as(delta_raw)


def _transport_cost(
    *,
    delta_raw: torch.Tensor,
    sigma: torch.Tensor,
    gate: torch.Tensor,
    sigma_data: float,
    delta_space: str,
) -> torch.Tensor:
    """Per-sample transport cost used by CDRO dual update."""

    applied = _applied_delta(delta_raw, sigma, gate, delta_space)
    if delta_space == "noise":
        return applied.square().reshape(applied.shape[0], -1).mean(dim=1)
    denom = sigma.square() + float(sigma_data) ** 2
    numer = applied.square().reshape(applied.shape[0], -1)
    return (numer / denom.view(-1, 1).clamp_min(1e-8)).mean(dim=1)


def _build_cdro_batch(
    *,
    cfg,
    denoiser,
    x_clean: torch.Tensor,
    lambda_dual: float,
    activation_scale: float,
) -> dict:
    """Build one CDRO-EDM batch: inner attack on base_noisy using fixed sigma/noise."""

    batch_size = x_clean.shape[0]
    sigma = _sample_sigma(cfg, batch_size, device=x_clean.device, dtype=x_clean.dtype)
    noise = torch.randn_like(x_clean)
    base_noisy = x_clean + batch_scalar_like(sigma, x_clean) * noise

    delta_space = str(cfg.v12_delta_space).lower()
    gate = _sigma_gate(
        sigma,
        activation_scale=activation_scale,
        sigma_floor=float(cfg.v12_sigma_floor),
        sigma_cut=float(cfg.v12_sigma_cut),
        gate_power=float(cfg.v12_gate_power),
    )
    radius = _delta_radius(
        sigma=sigma,
        gate=gate,
        max_delta=float(cfg.v12_max_delta),
        delta_space=delta_space,
    )

    delta_raw = torch.zeros_like(base_noisy)
    adv_steps = max(int(cfg.inner_steps), 0)
    step_size = float(cfg.v12_step_size)
    inner_obj_last = torch.zeros((), device=x_clean.device, dtype=x_clean.dtype)
    attack_loss_last = torch.zeros((), device=x_clean.device, dtype=x_clean.dtype)
    transport_last = torch.zeros((), device=x_clean.device, dtype=x_clean.dtype)

    for _ in range(adv_steps):
        delta_raw.requires_grad_(True)
        perturbed = base_noisy + _applied_delta(delta_raw, sigma, gate, delta_space)
        attack_loss = _cdro_inner_attack_loss(
            denoiser=denoiser,
            x_noisy=perturbed,
            x_clean=x_clean,
            sigma=sigma,
            sigma_data=float(cfg.sigma_data),
        )
        transport = _transport_cost(
            delta_raw=delta_raw,
            sigma=sigma,
            gate=gate,
            sigma_data=float(cfg.sigma_data),
            delta_space=delta_space,
        ).mean()
        inner_obj = inner_objective_attack_only(attack_loss) - float(lambda_dual) * transport
        grad_delta = torch.autograd.grad(inner_obj, delta_raw, only_inputs=True)[0]

        grad_flat = grad_delta.reshape(batch_size, -1)
        grad_norm = grad_flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
        step = float(step_size) * (grad_flat / grad_norm).reshape_as(delta_raw)
        delta_raw = (delta_raw + step).detach()
        delta_raw = _project_l2_rms_ball(delta_raw, radius).detach()

        inner_obj_last = inner_obj.detach()
        attack_loss_last = attack_loss.detach()
        transport_last = transport.detach()

    applied_delta = _applied_delta(delta_raw, sigma, gate, delta_space).detach()
    transport_per = _transport_cost(
        delta_raw=delta_raw.detach(),
        sigma=sigma.detach(),
        gate=gate.detach(),
        sigma_data=float(cfg.sigma_data),
        delta_space=delta_space,
    )
    if adv_steps == 0:
        attack_loss_last = _cdro_inner_attack_loss(
            denoiser=denoiser,
            x_noisy=base_noisy,
            x_clean=x_clean,
            sigma=sigma,
            sigma_data=float(cfg.sigma_data),
        ).detach()
        transport_last = transport_per.mean().detach()
        inner_obj_last = (inner_objective_attack_only(attack_loss_last) - float(lambda_dual) * transport_last).detach()

    raw_rms = delta_raw.detach().reshape(batch_size, -1).square().mean(dim=1).sqrt()
    applied_rms = applied_delta.reshape(batch_size, -1).square().mean(dim=1).sqrt()
    return {
        "sigma": sigma.detach(),
        "base_noisy": base_noisy.detach(),
        "gate": gate.detach(),
        "delta_raw": delta_raw.detach(),
        "applied_delta": applied_delta.detach(),
        "transport_per": transport_per.detach(),
        "inner_obj": inner_obj_last.detach(),
        "attack_loss": attack_loss_last.detach(),
        "transport_mean": transport_last.detach(),
        "raw_rms": raw_rms.detach(),
        "applied_rms": applied_rms.detach(),
    }


def train_trajectory_robust_constrained(
    denoiser,
    control,
    centers,
    sigma_levels,
    cfg,
    train_pool: torch.Tensor = None,
    sample_train_batch_fn: Optional[Callable[[int], torch.Tensor]] = None,
    sample_population_batch_fn: Optional[Callable[[int], torch.Tensor]] = None,
):
    """v1.2 robust training aligned to EDMLossCDRO (single-sigma adversarial batch)."""

    optimizer_theta = torch.optim.Adam(denoiser.parameters(), lr=cfg.lr_theta)
    if str(getattr(cfg, "training_objective", "edm")).lower() != "edm":
        raise ValueError("v1.2 (cdroedm-aligned) requires training_objective='edm'.")
    history = {
        "outer_loss": [],
        "outer_loss_attack": [],
        "outer_loss_clean": [],
        "inner_obj": [],
        "energy": [],
        "lambda_value": [],
        "lambda_value_next": [],
        "lambda_subgrad": [],
        "dual_surrogate": [],
        "lambda_dual": [],
        "lambda_update": [],
        "robust_mix": [],
        "activation_scale": [],
        "delta_norm_ratio_mean": [],
        "delta_norm_ratio_max": [],
        "sched_attack_weight": [],
        "sched_clean_weight": [],
        "sched_phi_lr_scale": [],
        "transport_inner": [],
        "delta_norm_mean": [],
        "delta_norm_max": [],
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
    }

    # CDRO-EDM does not optimize a persistent control policy network.
    del sigma_levels
    set_requires_grad(control, False)
    with torch.no_grad():
        for p in control.parameters():
            p.zero_()

    lambda_dual = float(cfg.v12_lambda_init)
    rho_target = float(cfg.v12_rho_target)
    lambda_lr = float(cfg.v12_lambda_lr)
    robust_mix = float(min(max(cfg.v12_robust_mix, 0.0), 1.0))
    max_delta = float(cfg.v12_max_delta)
    if max_delta <= 0.0:
        raise ValueError(f"v12_max_delta must be > 0, got {max_delta}")

    for step in range(1, cfg.steps + 1):
        x0 = sample_train_batch(
            cfg,
            centers,
            train_pool=train_pool,
            sample_train_batch_fn=sample_train_batch_fn,
            sample_population_batch_fn=sample_population_batch_fn,
        )
        activation = _activation_scale(step, cfg)
        set_requires_grad(denoiser, False)
        lambda_prev = float(lambda_dual)
        attack = _build_cdro_batch(
            cfg=cfg,
            denoiser=denoiser,
            x_clean=x0,
            lambda_dual=lambda_prev,
            activation_scale=activation,
        )
        set_requires_grad(denoiser, True)

        attack_loss_inner = attack["attack_loss"]
        transport_inner = attack["transport_mean"]
        inner_obj = inner_objective_attack_only(attack_loss_inner) - float(lambda_prev) * transport_inner
        if has_nan_or_inf(inner_obj):
            raise RuntimeError("NaN/Inf detected in v1.2 inner objective.")

        lambda_delta = float(lambda_lr) * (float(transport_inner.item()) - float(rho_target))
        lambda_dual = float(max(0.0, float(lambda_prev) + float(lambda_delta)))

        delta_rms = attack["applied_rms"]
        delta_ratio = delta_rms / max(max_delta, 1e-8)
        last_inner_obj = scalarize(inner_obj)
        last_transport = scalarize(transport_inner)
        last_delta_norm_mean = scalarize(delta_rms.mean())
        last_delta_norm_max = scalarize(delta_rms.max())
        last_delta_ratio_mean = scalarize(delta_ratio.mean())
        last_delta_ratio_max = scalarize(delta_ratio.max())

        optimizer_theta.zero_grad(set_to_none=True)
        sigma = attack["sigma"]
        base_noisy = attack["base_noisy"]
        applied_delta = attack["applied_delta"]
        outer_loss_clean = _cdro_outer_loss_scalar(
            denoiser=denoiser,
            x_noisy=base_noisy,
            x_clean=x0,
            sigma=sigma,
            sigma_data=float(cfg.sigma_data),
        )
        outer_loss_attack = _cdro_outer_loss_scalar(
            denoiser=denoiser,
            x_noisy=base_noisy + applied_delta,
            x_clean=x0,
            sigma=sigma,
            sigma_data=float(cfg.sigma_data),
        )
        outer_loss = (1.0 - robust_mix) * outer_loss_clean + robust_mix * outer_loss_attack
        transport_outer = attack["transport_per"].mean()
        if has_nan_or_inf(outer_loss):
            raise RuntimeError("NaN/Inf detected in v1.2 outer loss.")
        outer_loss.backward()
        optimizer_theta.step()

        history["outer_loss"].append(scalarize(outer_loss))
        history["outer_loss_attack"].append(scalarize(outer_loss_attack))
        history["outer_loss_clean"].append(scalarize(outer_loss_clean))
        history["inner_obj"].append(last_inner_obj)
        history["energy"].append(scalarize(transport_outer))
        history["lambda_value"].append(float(lambda_prev))
        history["lambda_value_next"].append(float(lambda_dual))
        history["lambda_subgrad"].append(float(last_transport - rho_target))
        history["dual_surrogate"].append(float(last_inner_obj))
        history["lambda_dual"].append(float(lambda_dual))
        history["lambda_update"].append(float(lambda_delta))
        history["robust_mix"].append(float(robust_mix))
        history["activation_scale"].append(float(activation))
        history["delta_norm_ratio_mean"].append(last_delta_ratio_mean)
        history["delta_norm_ratio_max"].append(last_delta_ratio_max)
        history["sched_attack_weight"].append(float(robust_mix))
        history["sched_clean_weight"].append(float(1.0 - robust_mix))
        history["sched_phi_lr_scale"].append(1.0)
        history["transport_inner"].append(float(last_transport))
        history["delta_norm_mean"].append(last_delta_norm_mean)
        history["delta_norm_max"].append(last_delta_norm_max)

        if step % cfg.log_every == 0:
            print(
                f"[robust-v1.2] step={step:05d} outer_loss={outer_loss.item():.6f} "
                f"attack={outer_loss_attack.item():.6f} clean={outer_loss_clean.item():.6f} "
                f"inner_obj={last_inner_obj:.6f} transport={last_transport:.6f} "
                f"delta_norm={last_delta_norm_mean:.6f} delta_ratio={last_delta_ratio_mean:.6f} "
                f"w_attack={robust_mix:.3f} w_clean={(1.0 - robust_mix):.3f} "
                f"lambda={lambda_dual:.6f} lambda_delta={lambda_delta:.6f} "
                f"activation={activation:.3f} step_size={float(cfg.v12_step_size):.6f}",
                flush=True,
            )

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

    return train_trajectory_robust_constrained(
        denoiser,
        control,
        centers,
        sigma_levels,
        cfg,
        train_pool=train_pool,
        sample_train_batch_fn=sample_train_batch_fn,
        sample_population_batch_fn=sample_population_batch_fn,
    )
