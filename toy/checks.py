import copy
from dataclasses import asdict
from typing import Dict, Tuple

import torch

from .data import sample_gmm
from .diffusion import rollout_controlled_ve, sample_target_indices
from .models import set_requires_grad
from .objective import inner_objective_energy_only, weighted_denoise_loss


def _finite_difference_scalar(loss_fn, param: torch.Tensor, idx: Tuple[int, ...], eps: float) -> float:
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
    return abs(a - b) / (abs(a) + abs(b) + 1e-8)


def _argmax_abs_index(tensor: torch.Tensor) -> Tuple[int, ...]:
    flat_idx = int(torch.argmax(torch.abs(tensor)).item())
    return tuple(torch.unravel_index(torch.tensor(flat_idx), tensor.shape))


def sanity_check_rollout(control, centers, sigma_levels, cfg) -> Dict[str, float]:
    x0 = sample_gmm(64, centers, cfg.data_std)
    idx = sample_target_indices(64, sigma_levels)
    roll = rollout_controlled_ve(x0, idx, control, sigma_levels, grad_through_control=False)
    assert roll.x_target.shape == x0.shape
    assert roll.sigma_target.shape == idx.shape
    assert roll.states_ref is not None and roll.states_ctrl is not None
    assert torch.all(roll.energy >= 0)
    mean_dev = (roll.states_ctrl - roll.states_ref).pow(2).sum(dim=2).sqrt().mean().item()
    return {
        "rollout_energy_mean": float(roll.energy.item()),
        "rollout_mean_path_deviation": float(mean_dev),
    }


def gradient_check_denoiser(denoiser, control, centers, sigma_levels, cfg) -> Dict[str, float]:
    denoiser.zero_grad(set_to_none=True)
    set_requires_grad(denoiser, True)
    set_requires_grad(control, False)

    x0 = sample_gmm(32, centers, cfg.data_std)
    idx = sample_target_indices(32, sigma_levels)
    eps_schedule = torch.randn(sigma_levels.numel() - 1, x0.shape[0], x0.shape[1], device=x0.device)

    def loss_fn():
        roll = rollout_controlled_ve(
            x0,
            idx,
            control,
            sigma_levels,
            grad_through_control=False,
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
    control.zero_grad(set_to_none=True)
    set_requires_grad(denoiser, False)
    set_requires_grad(control, True)

    x0 = sample_gmm(32, centers, cfg.data_std)
    idx = sample_target_indices(32, sigma_levels)
    eps_schedule = torch.randn(sigma_levels.numel() - 1, x0.shape[0], x0.shape[1], device=x0.device)

    def inner_fn():
        roll = rollout_controlled_ve(
            x0,
            idx,
            control,
            sigma_levels,
            grad_through_control=True,
            eps_schedule=eps_schedule,
        )
        train_loss = weighted_denoise_loss(denoiser, roll.x_target, x0, roll.sigma_target, cfg.sigma_data)
        return inner_objective_energy_only(train_loss, roll.energy, cfg.lambda_energy)

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
