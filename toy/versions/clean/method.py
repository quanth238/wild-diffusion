from ..wild.diffusion import build_kappa_schedule, rollout_controlled_ve
from .trainer import train_trajectory_robust_clean

NAME = "clean"
IMPLEMENTED = True
DESCRIPTION = "Baseline EDM continuation from a loaded checkpoint (no control rollout training)."


def train_trajectory_robust(*args, **kwargs):
    return train_trajectory_robust_clean(*args, **kwargs)
