from .diffusion import build_kappa_schedule, rollout_controlled_ve
from .trainer import train_trajectory_robust_constrained

NAME = "v2.1"
IMPLEMENTED = True
DESCRIPTION = (
    "Hard per-step constrained control with non-Markovian reference increments "
    "(rho-controlled correlated rollout)."
)


def train_trajectory_robust(*args, **kwargs):
    return train_trajectory_robust_constrained(*args, **kwargs)
