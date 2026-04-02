from .diffusion import build_kappa_schedule, rollout_controlled_ve
from .trainer import train_trajectory_robust_constrained

NAME = "v2"
IMPLEMENTED = True
DESCRIPTION = "Hard per-step constrained control (projection onto local L2 ball)."


def train_trajectory_robust(*args, **kwargs):
    return train_trajectory_robust_constrained(*args, **kwargs)

