import copy
from typing import Dict, Tuple

import torch

from .data import sample_gmm
from .models import set_requires_grad
from .shared.objective import inner_objective_attack_only, weighted_denoise_loss
from .shared.sigma import sample_target_indices
from .versions.registry import resolve_method_module


def _sample_check_batch(cfg, batch_size: int, centers=None, sample_batch_fn=None) -> torch.Tensor:
    """Sample x0 for numerical checks from backend callback or toy GMM fallback."""

    if sample_batch_fn is not None:
        return sample_batch_fn(batch_size)
    if centers is None:
        raise ValueError("centers must be provided when no sample_batch_fn is available for checks.")
    return sample_gmm(batch_size, centers, cfg.data_std)


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


def sanity_check_rollout(control, centers, sigma_levels, cfg, sample_batch_fn=None, method=None) -> Dict[str, float]:
    """Quick rollout sanity: shape consistency + control magnitude diagnostics."""

    if method is None:
        method = resolve_method_module(cfg.method_version)
    x0 = _sample_check_batch(cfg, 64, centers=centers, sample_batch_fn=sample_batch_fn)
    idx = sample_target_indices(64, sigma_levels)
    kappa_by_step = method.build_kappa_schedule(
        sigma_levels=sigma_levels,
        base_kappa=cfg.control_radius_kappa,
        use_time_dependent=cfg.use_time_dependent_kappa,
        low_multiplier=cfg.kappa_low_multiplier,
        mid_multiplier=cfg.kappa_mid_multiplier,
        high_multiplier=cfg.kappa_high_multiplier,
        preserve_l2_budget=cfg.kappa_preserve_l2_budget,
    )
    roll = method.rollout_controlled_ve(
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
    flat_dev = (roll.states_ctrl - roll.states_ref).reshape(roll.states_ref.shape[0], roll.states_ref.shape[1], -1)
    mean_dev = flat_dev.pow(2).sum(dim=2).sqrt().mean().item()
    delta_l2 = roll.delta_path.reshape(roll.delta_path.shape[0], roll.delta_path.shape[1], -1).pow(2).sum(dim=2).sqrt()
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


def gradient_check_denoiser(
    denoiser,
    control,
    centers,
    sigma_levels,
    cfg,
    sample_batch_fn=None,
    method=None,
) -> Dict[str, float]:
    """Compare autodiff vs finite-difference gradient for denoiser objective."""

    if method is None:
        method = resolve_method_module(cfg.method_version)
    denoiser.zero_grad(set_to_none=True)
    set_requires_grad(denoiser, True)
    set_requires_grad(control, False)

    x0 = _sample_check_batch(cfg, 32, centers=centers, sample_batch_fn=sample_batch_fn)
    idx = sample_target_indices(32, sigma_levels)
    eps_schedule = torch.randn((sigma_levels.numel() - 1, x0.shape[0], *x0.shape[1:]), device=x0.device)
    kappa_by_step = method.build_kappa_schedule(
        sigma_levels=sigma_levels,
        base_kappa=cfg.control_radius_kappa,
        use_time_dependent=cfg.use_time_dependent_kappa,
        low_multiplier=cfg.kappa_low_multiplier,
        mid_multiplier=cfg.kappa_mid_multiplier,
        high_multiplier=cfg.kappa_high_multiplier,
        preserve_l2_budget=cfg.kappa_preserve_l2_budget,
    )

    def loss_fn():
        roll = method.rollout_controlled_ve(
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


def gradient_check_control(
    denoiser,
    control,
    centers,
    sigma_levels,
    cfg,
    sample_batch_fn=None,
    method=None,
) -> Dict[str, float]:
    """Compare autodiff vs finite-difference gradient for inner control objective."""

    if method is None:
        method = resolve_method_module(cfg.method_version)
    control.zero_grad(set_to_none=True)
    set_requires_grad(denoiser, False)
    set_requires_grad(control, True)

    x0 = _sample_check_batch(cfg, 32, centers=centers, sample_batch_fn=sample_batch_fn)
    idx = sample_target_indices(32, sigma_levels)
    eps_schedule = torch.randn((sigma_levels.numel() - 1, x0.shape[0], *x0.shape[1:]), device=x0.device)
    kappa_by_step = method.build_kappa_schedule(
        sigma_levels=sigma_levels,
        base_kappa=cfg.control_radius_kappa,
        use_time_dependent=cfg.use_time_dependent_kappa,
        low_multiplier=cfg.kappa_low_multiplier,
        mid_multiplier=cfg.kappa_mid_multiplier,
        high_multiplier=cfg.kappa_high_multiplier,
        preserve_l2_budget=cfg.kappa_preserve_l2_budget,
    )

    def inner_fn():
        roll = method.rollout_controlled_ve(
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


def run_preflight_checks(denoiser, control, centers, sigma_levels, cfg, sample_batch_fn=None, method=None) -> Dict:
    """Numerical preflight checks run on CPU/float64 copies before training."""

    if method is None:
        method = resolve_method_module(cfg.method_version)
    if not getattr(method, "IMPLEMENTED", True):
        raise NotImplementedError(
            f"method_version='{cfg.method_version}' preflight checks are unavailable: method scaffold is not implemented."
        )

    # Run checks on CPU/float64 copies for numerical stability.
    denoiser_chk = copy.deepcopy(denoiser).to(device="cpu", dtype=torch.float64)
    control_chk = copy.deepcopy(control).to(device="cpu", dtype=torch.float64)
    centers_chk = None if centers is None else centers.detach().to(device="cpu", dtype=torch.float64)
    sigma_levels_chk = sigma_levels.detach().to(device="cpu", dtype=torch.float64)

    def sample_batch_chk(batch_size: int) -> torch.Tensor:
        if sample_batch_fn is None:
            return _sample_check_batch(cfg, batch_size, centers=centers_chk, sample_batch_fn=None)
        return sample_batch_fn(batch_size).detach().to(device="cpu", dtype=torch.float64)

    report = {
        "rollout": sanity_check_rollout(
            control_chk,
            centers_chk,
            sigma_levels_chk,
            cfg,
            sample_batch_fn=sample_batch_chk,
            method=method,
        ),
        "gradcheck_denoiser": gradient_check_denoiser(
            denoiser_chk,
            control_chk,
            centers_chk,
            sigma_levels_chk,
            cfg,
            sample_batch_fn=sample_batch_chk,
            method=method,
        ),
        "gradcheck_control": gradient_check_control(
            denoiser_chk,
            control_chk,
            centers_chk,
            sigma_levels_chk,
            cfg,
            sample_batch_fn=sample_batch_chk,
            method=method,
        ),
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
