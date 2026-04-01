from ...shared.train_utils import zero_control
from .diffusion import build_kappa_schedule, rollout_controlled_ve
from .trainer import train_trajectory_robust_constrained

NAME = "v1.2"
IMPLEMENTED = True
DESCRIPTION = (
    "CDRO-EDM aligned training (single-sigma adversarial batch): inner ascent over delta on "
    "noisy samples, sigma-gated perturbations, dual-lambda transport control, and clean/robust mix."
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
    """CDRO-EDM has no path control policy; eval uses reference rollout (delta=0)."""

    del cfg, attack_net
    return rollout_controlled_ve(
        x0=x0,
        target_indices=target_indices,
        control_net=zero_control,
        sigma_levels=sigma_levels,
        grad_through_control=False,
        control_radius_kappa=float(control_radius_kappa),
        kappa_by_step=kappa_by_step,
    )
