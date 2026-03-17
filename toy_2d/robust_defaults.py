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
    "warmup_epochs": 5,
    "path_steps": 8,
    "inner_steps": 2,
    "step_size": 5e-4,
    "gamma": 0.0,
    "total_budget": 1e-4,
    "budget_mode": "match_wdro",
    "exact_budget_split": True,
    "sigma_schedule": "edm_quantiles",
}

CAUSAL_WDRO_TUNING_SPACE = {
    "causal_warmup_epochs": [0, 5, 10],
    "causal_path_steps": [4, 8, 12, 16],
    "causal_inner_steps": [1, 2],
    "causal_step_size": [2e-4, 5e-4, 1e-3, 2e-3],
    "causal_gamma": [CAUSAL_WDRO_DEFAULTS["gamma"]],
    "causal_budget_mode": [CAUSAL_WDRO_DEFAULTS["budget_mode"]],
    "causal_exact_budget_split": [CAUSAL_WDRO_DEFAULTS["exact_budget_split"]],
    "causal_sigma_schedule": ["edm_quantiles", "karras_grid"],
}
