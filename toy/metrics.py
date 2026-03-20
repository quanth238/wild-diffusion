from typing import Callable, Dict

import numpy as np
import torch


def evaluate_mode_coverage(samples: np.ndarray, centers: np.ndarray) -> Dict[str, float]:
    diff = samples[:, None, :] - centers[None, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    nearest = np.argmin(dists, axis=1)
    counts = np.bincount(nearest, minlength=centers.shape[0])
    min_dist = np.min(dists, axis=1)

    return {
        "coverage_ratio": float((counts > 0).sum() / centers.shape[0]),
        "covered_modes": int((counts > 0).sum()),
        "avg_min_dist_to_mode": float(min_dist.mean()),
        "p50_min_dist_to_mode": float(np.quantile(min_dist, 0.50)),
        "p90_min_dist_to_mode": float(np.quantile(min_dist, 0.90)),
        "p99_min_dist_to_mode": float(np.quantile(min_dist, 0.99)),
        "std_x": float(samples[:, 0].std()),
        "std_y": float(samples[:, 1].std()),
    }


def evaluate_nearest_reference_distance(samples: np.ndarray, reference: np.ndarray) -> Dict[str, float]:
    if samples.ndim != 2 or reference.ndim != 2:
        raise ValueError(
            "samples and reference must be rank-2 arrays [N, D], "
            f"got {samples.shape} and {reference.shape}"
        )
    diff = samples[:, None, :] - reference[None, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    min_dist = np.min(dists, axis=1)
    return {
        "avg_min_dist": float(min_dist.mean()),
        "p50_min_dist": float(np.quantile(min_dist, 0.50)),
        "p90_min_dist": float(np.quantile(min_dist, 0.90)),
        "p99_min_dist": float(np.quantile(min_dist, 0.99)),
    }


def estimate_bayes_posterior_mean_mse(
    centers: np.ndarray,
    data_std: float,
    sigma: float,
    n_samples: int = 100000,
    seed: int = 0,
) -> float:
    rng = np.random.default_rng(seed)
    n_modes = centers.shape[0]
    mode_ids = rng.integers(0, n_modes, size=n_samples)
    x0 = centers[mode_ids] + rng.normal(scale=data_std, size=(n_samples, 2))
    x = x0 + rng.normal(scale=sigma, size=(n_samples, 2))

    s2 = data_std ** 2 + sigma ** 2
    diff = x[:, None, :] - centers[None, :, :]
    logw = -np.sum(diff ** 2, axis=2) / (2.0 * s2)
    logw = logw - np.max(logw, axis=1, keepdims=True)
    w = np.exp(logw)
    w = w / np.sum(w, axis=1, keepdims=True)

    a = data_std ** 2 / (data_std ** 2 + sigma ** 2)
    comp_mean = a * x[:, None, :] + (1.0 - a) * centers[None, :, :]
    mu = np.sum(w[:, :, None] * comp_mean, axis=1)
    return float(np.mean(np.sum((mu - x0) ** 2, axis=1)))


def evaluate_baseline_gate(
    generated_mode_metrics: Dict[str, float],
    endpoint_mode_metrics: Dict[str, float],
    min_coverage: float,
    max_generated_avg_min_dist: float,
    max_generated_p90_min_dist: float,
    max_endpoint_avg_min_dist: float,
    max_endpoint_p90_min_dist: float,
) -> Dict:
    checks = []

    checks.append(
        {
            "name": "coverage_ratio",
            "value": float(generated_mode_metrics["coverage_ratio"]),
            "op": ">=",
            "threshold": float(min_coverage),
            "pass": bool(generated_mode_metrics["coverage_ratio"] >= min_coverage),
        }
    )
    checks.append(
        {
            "name": "generated_avg_min_dist_to_mode",
            "value": float(generated_mode_metrics["avg_min_dist_to_mode"]),
            "op": "<=",
            "threshold": float(max_generated_avg_min_dist),
            "pass": bool(generated_mode_metrics["avg_min_dist_to_mode"] <= max_generated_avg_min_dist),
        }
    )
    checks.append(
        {
            "name": "generated_p90_min_dist_to_mode",
            "value": float(generated_mode_metrics["p90_min_dist_to_mode"]),
            "op": "<=",
            "threshold": float(max_generated_p90_min_dist),
            "pass": bool(generated_mode_metrics["p90_min_dist_to_mode"] <= max_generated_p90_min_dist),
        }
    )
    checks.append(
        {
            "name": "endpoint_avg_min_dist_to_mode",
            "value": float(endpoint_mode_metrics["avg_min_dist_to_mode"]),
            "op": "<=",
            "threshold": float(max_endpoint_avg_min_dist),
            "pass": bool(endpoint_mode_metrics["avg_min_dist_to_mode"] <= max_endpoint_avg_min_dist),
        }
    )
    checks.append(
        {
            "name": "endpoint_p90_min_dist_to_mode",
            "value": float(endpoint_mode_metrics["p90_min_dist_to_mode"]),
            "op": "<=",
            "threshold": float(max_endpoint_p90_min_dist),
            "pass": bool(endpoint_mode_metrics["p90_min_dist_to_mode"] <= max_endpoint_p90_min_dist),
        }
    )

    failed = [c for c in checks if not c["pass"]]
    return {
        "passed": bool(len(failed) == 0),
        "checks": checks,
        "failed_checks": failed,
    }


@torch.no_grad()
def compute_denoise_error_curves(
    baseline_model,
    robust_model,
    x0: torch.Tensor,
    states_ref: torch.Tensor,
    states_ctrl: torch.Tensor,
    sigma_levels: torch.Tensor,
) -> Dict[str, list]:
    n_steps = int(sigma_levels.numel() - 1)
    curves = {
        "step": [],
        "sigma": [],
        "baseline_on_forward_baseline": [],
        "baseline_on_forward_attack": [],
        "robust_on_forward_baseline": [],
        "robust_on_forward_attack": [],
    }

    for k in range(1, n_steps + 1):
        sigma_k = torch.full((x0.shape[0],), sigma_levels[k], device=x0.device)
        ref_k = states_ref[:, k]
        ctrl_k = states_ctrl[:, k]

        pred_b_ref = baseline_model(ref_k, sigma_k)
        pred_b_ctrl = baseline_model(ctrl_k, sigma_k)
        pred_r_ref = robust_model(ref_k, sigma_k)
        pred_r_ctrl = robust_model(ctrl_k, sigma_k)

        err_b_ref = (pred_b_ref - x0).pow(2).sum(dim=1).mean()
        err_b_ctrl = (pred_b_ctrl - x0).pow(2).sum(dim=1).mean()
        err_r_ref = (pred_r_ref - x0).pow(2).sum(dim=1).mean()
        err_r_ctrl = (pred_r_ctrl - x0).pow(2).sum(dim=1).mean()

        curves["step"].append(k)
        curves["sigma"].append(float(sigma_levels[k].item()))
        curves["baseline_on_forward_baseline"].append(float(err_b_ref.item()))
        curves["baseline_on_forward_attack"].append(float(err_b_ctrl.item()))
        curves["robust_on_forward_baseline"].append(float(err_r_ref.item()))
        curves["robust_on_forward_attack"].append(float(err_r_ctrl.item()))

    return curves


@torch.no_grad()
def compute_path_mse_by_step(reverse_paths: torch.Tensor, forward_paths: torch.Tensor) -> list:
    # Both tensors are [B, N+1, 2], indexed by noise-step k.
    mse = (reverse_paths - forward_paths).pow(2).sum(dim=2).mean(dim=0)
    return [float(v.item()) for v in mse]


@torch.no_grad()
def compute_paired_reverse_delta_by_step(
    reverse_paths_ref: torch.Tensor,
    reverse_paths_attack: torch.Tensor,
) -> Dict[str, list]:
    # Both tensors are [B, N+1, 2], indexed by reverse noise-step k.
    if reverse_paths_ref.shape != reverse_paths_attack.shape:
        raise ValueError(
            "reverse path tensors must have same shape, got "
            f"{tuple(reverse_paths_ref.shape)} vs {tuple(reverse_paths_attack.shape)}"
        )
    delta = reverse_paths_attack - reverse_paths_ref
    l2 = torch.sqrt(delta.pow(2).sum(dim=2))  # [B, N+1]
    mse = delta.pow(2).sum(dim=2)  # [B, N+1]
    n_steps = reverse_paths_ref.shape[1]
    step = list(range(n_steps))

    return {
        "step": [int(s) for s in step],
        "mean_l2": [float(v.item()) for v in l2.mean(dim=0)],
        "mean_mse": [float(v.item()) for v in mse.mean(dim=0)],
        "p50_l2": [float(v.item()) for v in torch.quantile(l2, 0.50, dim=0)],
        "p90_l2": [float(v.item()) for v in torch.quantile(l2, 0.90, dim=0)],
        "x0_step_mean_l2": float(l2[:, 0].mean().item()),
        "x0_step_mean_mse": float(mse[:, 0].mean().item()),
        "terminal_step_mean_l2": float(l2[:, -1].mean().item()),
        "terminal_step_mean_mse": float(mse[:, -1].mean().item()),
    }


@torch.no_grad()
def compute_x0_recovery_vs_terminal_step(
    denoiser,
    forward_paths: torch.Tensor,
    sigma_levels: torch.Tensor,
    reverse_fn: Callable,
) -> Dict[str, list]:
    # For each terminal step k, start reverse from x_k and measure x0 recovery MSE.
    n_steps = int(sigma_levels.numel() - 1)
    x0 = forward_paths[:, 0]
    out = {"step": [], "sigma": [], "x0_mse": []}
    for k in range(1, n_steps + 1):
        rev_k = reverse_fn(denoiser=denoiser, x_terminal=forward_paths[:, k], sigma_levels=sigma_levels[: k + 1])
        x0_mse = (rev_k[:, 0] - x0).pow(2).sum(dim=1).mean()
        out["step"].append(k)
        out["sigma"].append(float(sigma_levels[k].item()))
        out["x0_mse"].append(float(x0_mse.item()))
    return out
