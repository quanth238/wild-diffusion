from ..cdro.diffusion import build_kappa_schedule, rollout_controlled_ve
from .trainer import train_trajectory_robust_wdro

NAME = "wdro"
IMPLEMENTED = True
DESCRIPTION = (
    "Paper-style WDRO training with a clean warmup stage followed by periodic "
    "dataset augmentation using worst-case samples."
)


def train_trajectory_robust(*args, **kwargs):
    return train_trajectory_robust_wdro(*args, **kwargs)
