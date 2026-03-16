from __future__ import annotations

# Keep causal WDRO on the same scale as standard WDRO so comparisons stay meaningful.
WDRO_CORE_DEFAULTS = {
    "k": 5,
    "step_size": 0.05,
    "gamma": 1.0,
    "p_adv": 1.0,
}

WDRO_CORE_TUNING_SPACE = {
    "k": [3, 5, 8, 10],
    "step_size": [0.02, 0.05, 0.1, 0.2],
    "gamma": [0.1, 0.3, 1.0, 3.0],
}

CAUSAL_WDRO_DEFAULTS = {
    "path_steps": 16,
    "inner_steps": 2,
    # With accumulated path cost, a smaller discrete path keeps the runtime
    # manageable while still matching WDRO-scale total budgets.
    "step_size": 5e-4,
    "gamma": 10.0,
    "sigma_schedule": "karras_grid",
}

CAUSAL_WDRO_TUNING_SPACE = {
    "causal_path_steps": [8, 16, 24],
    "causal_inner_steps": [1, 2, 3],
    "causal_step_size": [2e-4, 5e-4, 1e-3],
    "causal_gamma": [3.0, 10.0, 30.0],
    "causal_sigma_schedule": [CAUSAL_WDRO_DEFAULTS["sigma_schedule"]],
}
