"""Backward-compatible trainer exports.

Common trainer logic is in toy/shared/trainer_common.py.
Current robust implementation lives in toy/versions/v2/trainer.py.
"""

from .shared.reverse import (
    reverse_paths_from_terminal,
    reverse_posterior_mean as _reverse_posterior_mean,
    reverse_posterior_std as _reverse_posterior_std,
    sample_reverse_paths,
)
from .shared.trainer_common import sample_with_denoiser, train_baseline
from .versions.v2.trainer import train_trajectory_robust_constrained, train_trajectory_robust_energy

__all__ = [
    "_reverse_posterior_mean",
    "_reverse_posterior_std",
    "train_baseline",
    "train_trajectory_robust_constrained",
    "train_trajectory_robust_energy",
    "sample_with_denoiser",
    "sample_reverse_paths",
    "reverse_paths_from_terminal",
]

