from typing import Callable, Dict

import numpy as np
import torch

from .utils import per_sample_l2, per_sample_squared_l2


def evaluate_mode_coverage(samples: np.ndarray, centers: np.ndarray) -> Dict[str, float]:
    """Nearest-mode coverage and distance quality metrics for generated 2D samples."""

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
    """Distance from each sample to its nearest reference sample (memorization proxy)."""

    samples = samples.reshape(samples.shape[0], -1)
    reference = reference.reshape(reference.shape[0], -1)
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
    """Monte-Carlo Bayes lower bound for x0 recovery on toy GMM under Gaussian noise.

    For x = x0 + N(0, sigma^2 I), compute E||E[x0|x] - x0||^2.
    """

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
    """Evaluate baseline acceptance gate before enabling robust attack training."""

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
    """Per-sigma denoising MSE curves for baseline/robust models on ref/attack paths."""

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

        err_b_ref = per_sample_squared_l2(pred_b_ref - x0).mean()
        err_b_ctrl = per_sample_squared_l2(pred_b_ctrl - x0).mean()
        err_r_ref = per_sample_squared_l2(pred_r_ref - x0).mean()
        err_r_ctrl = per_sample_squared_l2(pred_r_ctrl - x0).mean()

        curves["step"].append(k)
        curves["sigma"].append(float(sigma_levels[k].item()))
        curves["baseline_on_forward_baseline"].append(float(err_b_ref.item()))
        curves["baseline_on_forward_attack"].append(float(err_b_ctrl.item()))
        curves["robust_on_forward_baseline"].append(float(err_r_ref.item()))
        curves["robust_on_forward_attack"].append(float(err_r_ctrl.item()))

    return curves


@torch.no_grad()
def compute_path_mse_by_step(reverse_paths: torch.Tensor, forward_paths: torch.Tensor) -> list:
    """Compute per-step E||x_rev_k - x_fwd_k||^2 along aligned trajectories."""

    delta = reverse_paths - forward_paths
    mse = delta.reshape(delta.shape[0], delta.shape[1], -1).pow(2).sum(dim=2).mean(dim=0)
    return [float(v.item()) for v in mse]


@torch.no_grad()
def compute_paired_reverse_delta_by_step(
    reverse_paths_ref: torch.Tensor,
    reverse_paths_attack: torch.Tensor,
) -> Dict[str, list]:
    """Compare reverse paths from attacked vs reference terminals using shared settings."""

    if reverse_paths_ref.shape != reverse_paths_attack.shape:
        raise ValueError(
            "reverse path tensors must have same shape, got "
            f"{tuple(reverse_paths_ref.shape)} vs {tuple(reverse_paths_attack.shape)}"
        )
    delta = reverse_paths_attack - reverse_paths_ref
    flat = delta.reshape(delta.shape[0], delta.shape[1], -1)
    mse = flat.pow(2).sum(dim=2)
    l2 = torch.sqrt(mse)
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
    """For each terminal level k, measure x0 recovery MSE after reverse rollout."""

    # For each terminal step k, start reverse from x_k and measure x0 recovery MSE.
    n_steps = int(sigma_levels.numel() - 1)
    x0 = forward_paths[:, 0]
    out = {"step": [], "sigma": [], "x0_mse": []}
    for k in range(1, n_steps + 1):
        rev_k = reverse_fn(denoiser=denoiser, x_terminal=forward_paths[:, k], sigma_levels=sigma_levels[: k + 1])
        x0_mse = per_sample_squared_l2(rev_k[:, 0] - x0).mean()
        out["step"].append(k)
        out["sigma"].append(float(sigma_levels[k].item()))
        out["x0_mse"].append(float(x0_mse.item()))
    return out


@torch.no_grad()
def compute_control_constraint_stats_by_step(
    delta_path: torch.Tensor,
    sigma_levels: torch.Tensor,
    control_radius_kappa: float,
    kappa_by_step: torch.Tensor = None,
    states_ref: torch.Tensor = None,
    states_ctrl: torch.Tensor = None,
    saturation_threshold: float = 0.98,
) -> Dict[str, list]:
    """Summarize per-step control norms against hard radius kappa_k * Delta_sigma_k."""

    # delta_path: [B, N, D], step k corresponds to transition k->k+1.
    n_steps = int(sigma_levels.numel() - 1)
    if delta_path.shape[1] != n_steps:
        raise ValueError(f"delta_path step dim must be {n_steps}, got {delta_path.shape[1]}")

    sigma_k = sigma_levels[:-1]
    sigma_next = sigma_levels[1:]
    delta_sigma = torch.sqrt((sigma_next.square() - sigma_k.square()).clamp_min(1e-8))  # [N]
    if kappa_by_step is not None:
        if kappa_by_step.numel() != n_steps:
            raise ValueError(f"kappa_by_step must have {n_steps} elements, got {kappa_by_step.numel()}")
        kappa_eff = kappa_by_step.to(device=delta_sigma.device, dtype=delta_sigma.dtype).reshape(n_steps)
    else:
        kappa_eff = torch.full((n_steps,), float(control_radius_kappa), device=delta_sigma.device, dtype=delta_sigma.dtype)
    radius = kappa_eff * delta_sigma  # [N]

    delta_l2 = delta_path.reshape(delta_path.shape[0], delta_path.shape[1], -1).pow(2).sum(dim=2).sqrt()
    ratio = delta_l2 / radius.view(1, n_steps).clamp_min(1e-8)
    near_boundary = (ratio >= float(saturation_threshold)).to(delta_l2.dtype)

    out = {
        "step": [int(k) for k in range(1, n_steps + 1)],
        "sigma": [float(v.item()) for v in sigma_levels[1:]],
        "effective_kappa": [float(v.item()) for v in kappa_eff],
        "delta_norm_mean": [float(v.item()) for v in delta_l2.mean(dim=0)],
        "delta_norm_p90": [float(v.item()) for v in torch.quantile(delta_l2, 0.90, dim=0)],
        "radius": [float(v.item()) for v in radius],
        "delta_ratio_mean": [float(v.item()) for v in ratio.mean(dim=0)],
        "delta_ratio_p90": [float(v.item()) for v in torch.quantile(ratio, 0.90, dim=0)],
        "frac_near_boundary": [float(v.item()) for v in near_boundary.mean(dim=0)],
        "saturation_threshold": float(saturation_threshold),
    }

    if states_ref is not None and states_ctrl is not None:
        # states_*: [B, N+1, D], compare at noise index 1..N
        gap = states_ctrl[:, 1:] - states_ref[:, 1:]
        gap_l2 = per_sample_l2(gap.reshape(-1, *gap.shape[2:])).reshape(gap.shape[0], gap.shape[1])
        out["gap_norm_mean"] = [float(v.item()) for v in gap_l2.mean(dim=0)]
        out["gap_norm_p90"] = [float(v.item()) for v in torch.quantile(gap_l2, 0.90, dim=0)]

    return out


def summarize_attack_gap_windows(curves: Dict[str, list]) -> Dict[str, Dict[str, float]]:
    """Aggregate robust-vs-baseline attack gaps over low/mid/high sigma windows."""

    # Uses heldout denoise curves:
    # gap_k = robust_on_forward_attack[k] - baseline_on_forward_attack[k]
    step = curves["step"]
    b_att = curves["baseline_on_forward_attack"]
    r_att = curves["robust_on_forward_attack"]
    n = len(step)
    if n == 0 or len(b_att) != n or len(r_att) != n:
        return {}

    gaps = [float(r - b) for r, b in zip(r_att, b_att)]
    wins = [1.0 if g < 0.0 else 0.0 for g in gaps]

    one_third = max(n // 3, 1)
    idx_low = list(range(0, one_third))
    idx_mid = list(range(one_third, min(2 * one_third, n)))
    idx_high = list(range(min(2 * one_third, n), n))
    if len(idx_mid) == 0:
        idx_mid = idx_low
    if len(idx_high) == 0:
        idx_high = idx_mid

    def agg(indices):
        vals = [gaps[i] for i in indices]
        w = [wins[i] for i in indices]
        return {
            "num_steps": int(len(indices)),
            "mean_attack_gap": float(sum(vals) / len(vals)),
            "win_steps": int(sum(1 for i in indices if gaps[i] < 0.0)),
            "win_ratio": float(sum(w) / len(w)),
            "step_start": int(step[indices[0]]),
            "step_end": int(step[indices[-1]]),
        }

    return {
        "low_noise": agg(idx_low),    # early steps, lower sigma
        "mid_noise": agg(idx_mid),
        "high_noise": agg(idx_high),  # later steps, higher sigma
        "overall": agg(list(range(n))),
    }
