from .diffusion import (
    build_constraint_radii as build_constraint_radii_impl,
    build_kappa_schedule,
    rollout_controlled_ve,
    rollout_path_heuristic_attack,
)
from .trainer import train_trajectory_robust_cdro

NAME = "cdro"
IMPLEMENTED = True
DESCRIPTION = (
    "Route-A CDRO path attack: beta-space greedy ascent with exact uniform local caps "
    "on the Jiang-style additive-drift cost, no learned attack policy."
)


def train_trajectory_robust(*args, **kwargs):
    return train_trajectory_robust_cdro(*args, **kwargs)


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
    """Evaluation rollout for CDRO using the denoiser-dependent beta-space attack."""

    del control_radius_kappa, kappa_by_step
    return rollout_path_heuristic_attack(
        cfg=cfg,
        x0=x0,
        target_indices=target_indices,
        attack_net=attack_net,
        sigma_levels=sigma_levels,
        inner_steps=int(cfg.inner_steps),
        step_size=float(cfg.cdro_step_size),
        total_budget=float(cfg.cdro_total_budget_rho),
        time_horizon=float(cfg.cdro_time_horizon),
    )


def build_constraint_radii_for_method(*, cfg, sigma_levels):
    """Method-scoped helper used by experiment metrics for exact CDRO cap statistics."""

    return build_constraint_radii_impl(
        sigma_levels=sigma_levels,
        total_budget=float(cfg.cdro_total_budget_rho),
        time_horizon=float(cfg.cdro_time_horizon),
    )


build_constraint_radii = build_constraint_radii_for_method
