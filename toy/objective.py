"""Backward-compatible objective exports.

Primary implementation now lives in toy/shared/objective.py.
"""

from .shared.objective import edm_weight, inner_objective_attack_only, weighted_denoise_loss

__all__ = [
    "edm_weight",
    "weighted_denoise_loss",
    "inner_objective_attack_only",
]
