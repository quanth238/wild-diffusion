from typing import Dict, List

import torch

from ..compute_accounting import (
    DENOISER_OP_COUNT_CUMULATIVE_KEYS,
    DENOISER_OP_COUNT_STEP_KEYS,
    DENOISER_OP_COUNTS_RECORDED_KEY,
)
from ..utils import per_sample_l2


def estimate_sigma_data(centers: torch.Tensor, data_std: float) -> float:
    """Estimate EDM `sigma_data` as per-dimension std of clean toy distribution.

    sigma_data^2 ~= Var(mode_center) + data_std^2
    """

    center_var = centers.var(dim=0, unbiased=False).mean()
    return float(torch.sqrt(center_var + data_std ** 2).item())


def summarize_series(values: List[float], tail: int = 200) -> dict:
    """Return compact tail statistics for a scalar training curve."""

    if len(values) == 0:
        return {"final": None, "min": None, "max": None, "mean_last": None}
    t = min(tail, len(values))
    return {
        "final": float(values[-1]),
        "min": float(min(values)),
        "max": float(max(values)),
        "mean_last": float(sum(values[-t:]) / t),
    }


def _mean(values: List[float]) -> float:
    """Return arithmetic mean; NaN when list is empty."""

    if len(values) == 0:
        return float("nan")
    return float(sum(values) / len(values))


def summarize_train_vs_val_curve_gaps(train_curves: Dict[str, List[float]], val_curves: Dict[str, List[float]]) -> dict:
    """Summarize overfitting signal via (val - train) gap on denoise curves."""

    keys = [
        "baseline_on_forward_baseline",
        "baseline_on_forward_attack",
        "robust_on_forward_baseline",
        "robust_on_forward_attack",
    ]
    out = {}
    for key in keys:
        tvals = train_curves.get(key, [])
        vvals = val_curves.get(key, [])
        if len(tvals) == 0 or len(vvals) == 0 or len(tvals) != len(vvals):
            out[key] = {
                "mean_gap_val_minus_train": None,
                "terminal_gap_val_minus_train": None,
            }
            continue
        out[key] = {
            "mean_gap_val_minus_train": float(_mean(vvals) - _mean(tvals)),
            "terminal_gap_val_minus_train": float(vvals[-1] - tvals[-1]),
        }
    return out


def compute_terminal_match_stats(reverse_paths: torch.Tensor, forward_paths: torch.Tensor) -> dict:
    """Terminal consistency check: compare reverse[-1] and forward[-1] in L2."""

    delta = reverse_paths[:, -1] - forward_paths[:, -1]
    l2 = per_sample_l2(delta)
    return {
        "mean_l2": float(l2.mean().item()),
        "max_l2": float(l2.max().item()),
    }


def empty_robust_history() -> dict:
    """Shape-stable empty robust history used when robust phase is skipped."""

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
        "transport_inner": [],
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
        "wild_inner_attack_loss": [],
        "wild_inner_transport_cost": [],
        "wild_inner_sigma_mean": [],
        "wild_refresh_step": [],
        "wild_cache_size": [],
        "wdro_refresh_step": [],
        "wdro_dataset_size": [],
        "wdro_adv_examples": [],
        "wdro_attack_batches": [],
        "wdro_attack_loss": [],
        "wdro_transport_cost": [],
        "batch_equiv_denoiser_evals_step": [],
        "batch_equiv_denoiser_evals_attack_construction": [],
        "batch_equiv_denoiser_evals_attack_eval": [],
        "batch_equiv_denoiser_evals_clean_eval": [],
        "batch_equiv_denoiser_evals_cumulative": [],
    }
    history[DENOISER_OP_COUNTS_RECORDED_KEY] = False
    for key in DENOISER_OP_COUNT_STEP_KEYS + DENOISER_OP_COUNT_CUMULATIVE_KEYS:
        history[key] = []
    return history
