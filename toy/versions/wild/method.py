from .diffusion import build_kappa_schedule, rollout_controlled_ve
from .trainer import train_trajectory_robust_wild

NAME = "wild"
IMPLEMENTED = True
DESCRIPTION = (
    "WILD-style WDRO surrogate training: sample-level inner maximization "
    "on loss(x') - gamma * 0.5||x'-x||^2 with interval cache refresh."
)


def train_trajectory_robust(*args, **kwargs):
    return train_trajectory_robust_wild(*args, **kwargs)
