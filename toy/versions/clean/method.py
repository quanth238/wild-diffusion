from ..cdro.diffusion import build_kappa_schedule, rollout_controlled_ve
from .trainer import train_trajectory_robust_clean

NAME = "clean"
IMPLEMENTED = True
DESCRIPTION = (
    "Baseline continuation path: EDM continues from a checkpoint, and RF mode continues from the shared EDM "
    "branch via the public --baseline-only path or explicit continuation lineage."
)


def train_trajectory_robust(*args, **kwargs):
    return train_trajectory_robust_clean(*args, **kwargs)
