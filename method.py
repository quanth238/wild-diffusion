from .diffusion import build_kappa_schedule, rollout_controlled_ve, rollout_path_heuristic_attack
from .trainer import train_trajectory_robust_constrained

NAME = "v1.1"
IMPLEMENTED = True
DESCRIPTION = (
    "Path-heuristic robust surrogate: gamma-penalized path attack "
    "with explicit budget projection (global_remaining/step_clip/step_exact)."
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
    """Evaluation rollout for v1.1 using denoiser-dependent path-heuristic attack."""

    return rollout_path_heuristic_attack(
        cfg=cfg,
        x0=x0,
        target_indices=target_indices,
        attack_net=attack_net,
        sigma_levels=sigma_levels,
        inner_steps=int(cfg.inner_steps),
        step_size=float(cfg.v11_step_size),
        gamma=float(cfg.v11_transport_gamma),
        total_budget=float(cfg.v11_total_budget_rho),
        projection_mode=str(cfg.v11_projection_mode).lower(),
        control_radius_kappa=float(control_radius_kappa),
        kappa_by_step=kappa_by_step,
    )
