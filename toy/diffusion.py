"""Backward-compatible diffusion exports.

Primary implementation now lives in:
- toy/shared/sigma.py
- toy/versions/v2/diffusion.py
"""

from .shared.sigma import build_sigma_levels, sample_target_indices, sample_target_indices_log_normal
from .versions.v2.diffusion import RolloutResult, build_kappa_schedule, project_l2_ball, rollout_controlled_ve

__all__ = [
    "RolloutResult",
    "build_sigma_levels",
    "sample_target_indices",
    "sample_target_indices_log_normal",
    "project_l2_ball",
    "build_kappa_schedule",
    "rollout_controlled_ve",
]
