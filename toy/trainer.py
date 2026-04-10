"""Backward-compatible trainer exports.

Common trainer logic is in toy/shared/trainer_common.py.
Current robust implementation lives in method-specific trainers under toy/versions/.
"""

from .shared.reverse import (
    reverse_paths_from_terminal,
    reverse_posterior_mean as _reverse_posterior_mean,
    reverse_posterior_std as _reverse_posterior_std,
    sample_reverse_paths,
)
from .shared.trainer_common import sample_with_denoiser, train_baseline
from .versions.cdro.trainer import train_trajectory_robust_energy

# Preserve the legacy symbol for older imports that still expect a constrained
# robust trainer entrypoint at the top level.
train_trajectory_robust_constrained = train_trajectory_robust_energy

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
