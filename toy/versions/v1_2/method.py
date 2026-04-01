from .diffusion import build_kappa_schedule, rollout_controlled_ve, rollout_path_heuristic_attack
from .trainer import train_trajectory_robust_constrained

NAME = "v1.2"
IMPLEMENTED = True
DESCRIPTION = (
    "CDRO-EDM-inspired upgrade of v1.1: path-heuristic attack with adaptive dual-lambda "
    "transport regularization, sigma-gated perturbations, and robust/clean outer-loss mixing."
)


def train_trajectory_robust(*args, **kwargs):
    return train_trajectory_robust_constrained(*args, **kwargs)


def rollout_eval(
    *,
    cfg,
    x0,
    target_indices,
    attack_net,
    sigma_levels,
    control_radius_kappa,
    kappa_by_step=None,
):
    """Evaluation rollout for v1.2 using denoiser-dependent path-heuristic attack."""

    return rollout_path_heuristic_attack(
        cfg=cfg,
        x0=x0,
        target_indices=target_indices,
        attack_net=attack_net,
        sigma_levels=sigma_levels,
        inner_steps=int(cfg.inner_steps),
        step_size=float(cfg.v12_step_size),
        lambda_dual=float(cfg.v12_lambda_init),
        activation_scale=1.0,
        max_delta=float(cfg.v12_max_delta),
        sigma_floor=float(cfg.v12_sigma_floor),
        sigma_cut=float(cfg.v12_sigma_cut),
        gate_power=float(cfg.v12_gate_power),
        delta_space=str(cfg.v12_delta_space).lower(),
        control_radius_kappa=float(control_radius_kappa),
        kappa_by_step=kappa_by_step,
    )
