from .diffusion import build_kappa_schedule, rollout_controlled_ve
from .trainer import train_trajectory_robust_energy

NAME = "v1"
IMPLEMENTED = True
DESCRIPTION = (
    "Energy-only robust surrogate (closeness disabled): "
    "inner max attack-loss - lambda*C_energy, with optional dual-lambda update "
    "for min_{theta,lambda>=0} rho*lambda + U(theta,lambda)."
)


def train_trajectory_robust(*args, **kwargs):
    return train_trajectory_robust_energy(*args, **kwargs)
