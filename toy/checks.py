import copy
from dataclasses import asdict
from typing import Dict, Tuple

import torch

from .data import sample_gmm
from .diffusion import build_kappa_schedule, rollout_controlled_ve, sample_target_indices
from .models import set_requires_grad
from .objective import inner_objective_attack_only, weighted_denoise_loss


def _finite_difference_scalar(loss_fn, param: torch.Tensor, idx: Tuple[int, ...], eps: float) -> float:
    """Central-difference gradient estimate d loss / d param[idx] with adaptive step scale."""

    with torch.no_grad():
        orig = param[idx].item()
    for scale in [1.0, 10.0, 100.0]:
        h = eps * scale
        with torch.no_grad():
            param[idx] = orig + h
        plus = float(loss_fn().item())
        with torch.no_grad():
            param[idx] = orig - h
        minus = float(loss_fn().item())
        with torch.no_grad():
            param[idx] = orig
        if abs(plus - minus) > 1e-10:
            return (plus - minus) / (2.0 * h)
    return 0.0


def _relative_error(a: float, b: float) -> float:
    """Symmetric relative error used by gradient checks."""

    return abs(a - b) / (abs(a) + abs(b) + 1e-8)


def _argmax_abs_index(tensor: torch.Tensor) -> Tuple[int, ...]:
    """Pick the index with largest absolute value for stable finite-difference check."""

    flat_idx = int(torch.argmax(torch.abs(tensor)).item())
    return tuple(torch.unravel_index(torch.tensor(flat_idx), tensor.shape))


def sanity_check_rollout(control, centers, sigma_levels, cfg) -> Dict[str, float]:
    """Quick rollout sanity: shape consistency + control magnitude diagnostics."""

    x0 = sample_gmm(64, centers, cfg.data_std)
    idx = sample_target_indices(64, sigma_levels)
    kappa_by_step = build_kappa_schedule(
        sigma_levels=sigma_levels,
        base_kappa=cfg.control_radius_kappa,
        use_time_dependent=cfg.use_time_dependent_kappa,
        low_multiplier=cfg.kappa_low_multiplier,
        mid_multiplier=cfg.kappa_mid_multiplier,
        high_multiplier=cfg.kappa_high_multiplier,
        preserve_l2_budget=cfg.kappa_preserve_l2_budget,
    )
    roll = rollout_controlled_ve(
        x0,
        idx,
        control,
        sigma_levels,
        grad_through_control=False,
        control_radius_kappa=cfg.control_radius_kappa,
        kappa_by_step=kappa_by_step,
    )
    assert roll.x_target.shape == x0.shape
    assert roll.sigma_target.shape == idx.shape
    assert roll.states_ref is not None and roll.states_ctrl is not None and roll.delta_path is not None
    mean_dev = (roll.states_ctrl - roll.states_ref).pow(2).sum(dim=2).sqrt().mean().item()
    delta_l2 = torch.sqrt(roll.delta_path.pow(2).sum(dim=2))
    sigma_k = sigma_levels[:-1]
    sigma_next = sigma_levels[1:]
    delta_sigma = torch.sqrt((sigma_next.square() - sigma_k.square()).clamp_min(1e-8))
    radius = (kappa_by_step * delta_sigma).view(1, -1).to(device=delta_l2.device, dtype=delta_l2.dtype)
    delta_ratio = delta_l2 / radius.clamp_min(1e-8)
    return {
        "rollout_mean_path_deviation": float(mean_dev),
        "rollout_delta_norm_mean": float(delta_l2.mean().item()),
        "rollout_delta_ratio_mean": float(delta_ratio.mean().item()),
        "rollout_delta_ratio_max": float(delta_ratio.max().item()),
    }


def gradient_check_denoiser(denoiser, control, centers, sigma_levels, cfg) -> Dict[str, float]:
    """Compare autodiff vs finite-difference gradient for denoiser objective."""

    denoiser.zero_grad(set_to_none=True)
    set_requires_grad(denoiser, True)
    set_requires_grad(control, False)

    x0 = sample_gmm(32, centers, cfg.data_std)
    idx = sample_target_indices(32, sigma_levels)
    eps_schedule = torch.randn(sigma_levels.numel() - 1, x0.shape[0], x0.shape[1], device=x0.device)
    kappa_by_step = build_kappa_schedule(
        sigma_levels=sigma_levels,
        base_kappa=cfg.control_radius_kappa,
        use_time_dependent=cfg.use_time_dependent_kappa,
        low_multiplier=cfg.kappa_low_multiplier,
        mid_multiplier=cfg.kappa_mid_multiplier,
        high_multiplier=cfg.kappa_high_multiplier,
        preserve_l2_budget=cfg.kappa_preserve_l2_budget,
    )

    def loss_fn():
        roll = rollout_controlled_ve(
            x0,
            idx,
            control,
            sigma_levels,
            grad_through_control=False,
            control_radius_kappa=cfg.control_radius_kappa,
            kappa_by_step=kappa_by_step,
            eps_schedule=eps_schedule,
        )
        return weighted_denoise_loss(denoiser, roll.x_target, x0, roll.sigma_target, cfg.sigma_data)

    loss = loss_fn()
    loss.backward()

    param = next(denoiser.parameters())
    index = _argmax_abs_index(param.grad)
    autodiff = float(param.grad[index].item())
    numerical = _finite_difference_scalar(loss_fn, param, index, cfg.gradcheck_eps)
    rel_err = _relative_error(autodiff, numerical)

    return {
        "index": [int(i) for i in index],
        "autodiff": autodiff,
        "numerical": numerical,
        "relative_error": rel_err,
    }


def gradient_check_control(denoiser, control, centers, sigma_levels, cfg) -> Dict[str, float]:
    """Compare autodiff vs finite-difference gradient for inner control objective."""

    control.zero_grad(set_to_none=True)
    set_requires_grad(denoiser, False)
    set_requires_grad(control, True)

    x0 = sample_gmm(32, centers, cfg.data_std)
    idx = sample_target_indices(32, sigma_levels)
    eps_schedule = torch.randn(sigma_levels.numel() - 1, x0.shape[0], x0.shape[1], device=x0.device)
    kappa_by_step = build_kappa_schedule(
        sigma_levels=sigma_levels,
        base_kappa=cfg.control_radius_kappa,
        use_time_dependent=cfg.use_time_dependent_kappa,
        low_multiplier=cfg.kappa_low_multiplier,
        mid_multiplier=cfg.kappa_mid_multiplier,
        high_multiplier=cfg.kappa_high_multiplier,
        preserve_l2_budget=cfg.kappa_preserve_l2_budget,
    )

    def inner_fn():
        roll = rollout_controlled_ve(
            x0,
            idx,
            control,
            sigma_levels,
            grad_through_control=True,
            control_radius_kappa=cfg.control_radius_kappa,
            kappa_by_step=kappa_by_step,
            eps_schedule=eps_schedule,
        )
        train_loss = weighted_denoise_loss(denoiser, roll.x_target, x0, roll.sigma_target, cfg.sigma_data)
        return inner_objective_attack_only(train_loss)

    def training_inner_loss():
        # Same scalar used during training update: minimize negative inner objective.
        return -inner_fn()

    inner_obj = training_inner_loss()
    inner_obj.backward()

    param = next(control.parameters())
    index = _argmax_abs_index(param.grad)
    autodiff = float(param.grad[index].item())
    numerical = _finite_difference_scalar(training_inner_loss, param, index, cfg.gradcheck_eps)
    rel_err = _relative_error(autodiff, numerical)
    return {
        "index": [int(i) for i in index],
        "autodiff": autodiff,
        "numerical": numerical,
        "relative_error": rel_err,
    }


def run_preflight_checks(denoiser, control, centers, sigma_levels, cfg) -> Dict:
    """Numerical preflight checks run on CPU/float64 copies before training."""

    # Run checks on CPU/float64 copies for numerical stability.
    denoiser_chk = copy.deepcopy(denoiser).to(device="cpu", dtype=torch.float64)
    control_chk = copy.deepcopy(control).to(device="cpu", dtype=torch.float64)
    centers_chk = centers.detach().to(device="cpu", dtype=torch.float64)
    sigma_levels_chk = sigma_levels.detach().to(device="cpu", dtype=torch.float64)

    report = {
        "rollout": sanity_check_rollout(control_chk, centers_chk, sigma_levels_chk, cfg),
        "gradcheck_denoiser": gradient_check_denoiser(denoiser_chk, control_chk, centers_chk, sigma_levels_chk, cfg),
        "gradcheck_control": gradient_check_control(denoiser_chk, control_chk, centers_chk, sigma_levels_chk, cfg),
    }

    if report["gradcheck_denoiser"]["relative_error"] > cfg.gradcheck_tol:
        raise RuntimeError(
            f"Denoiser gradient check failed: rel_err={report['gradcheck_denoiser']['relative_error']:.4e} > tol={cfg.gradcheck_tol:.4e}"
        )
    if report["gradcheck_control"]["relative_error"] > cfg.gradcheck_tol:
        raise RuntimeError(
            f"Control gradient check failed: rel_err={report['gradcheck_control']['relative_error']:.4e} > tol={cfg.gradcheck_tol:.4e}"
        )
    return report
