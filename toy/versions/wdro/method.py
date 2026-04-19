from ..cdro.diffusion import build_kappa_schedule, rollout_controlled_ve
from .trainer import train_trajectory_robust_wdro

NAME = "wdro"
IMPLEMENTED = True
DESCRIPTION = (
    "Paper-style WDRO training with periodic dataset augmentation using worst-case "
    "samples; RF mode starts from the shared EDM warm start and runs RF stage-1 plus reflow."
)


def train_trajectory_robust(*args, **kwargs):
    return train_trajectory_robust_wdro(*args, **kwargs)
