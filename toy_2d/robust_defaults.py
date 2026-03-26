from __future__ import annotations

# Keep CDRO on the same scale as standard WDRO so comparisons stay meaningful.
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

CDRO_DEFAULTS = {
    "warmup_epochs": 5,
    "path_steps": 8,
    "inner_steps": 2,
    "step_size": 5e-4,
    "gamma": WDRO_CORE_DEFAULTS["gamma"],
    "total_budget": 1e-4,
    "budget_mode": "match_wdro",
    "exact_budget_split": True,
    "sigma_schedule": "edm_quantiles",
}

CDRO_TUNING_SPACE = {
    "cdro_warmup_epochs": [0, 5, 10],
    "cdro_path_steps": [4, 8, 12, 16],
    "cdro_inner_steps": [1, 2],
    "cdro_step_size": [2e-4, 5e-4, 1e-3, 2e-3],
    "cdro_gamma": list(WDRO_CORE_TUNING_SPACE["gamma"]),
    "cdro_budget_mode": [CDRO_DEFAULTS["budget_mode"]],
    "cdro_exact_budget_split": [CDRO_DEFAULTS["exact_budget_split"]],
    "cdro_sigma_schedule": ["edm_quantiles", "karras_grid"],
}

CDRO_MARKOV_DEFAULTS = {
    "score_lr": 1e-3,
    "control_lr": 2e-4,
    "lambda_lr": 2e-2,
    "lambda_init": 0.1,
    "lambda_min": 0.02,
    "control_radius": 0.05,
    "warmup_epochs": 8,
    "adversary_steps": 1,
    "score_steps": 8,
    "terminal_momentum": 0.95,
    "num_steps": 12,
    "total_time": 1.0,
    "beta_min": 0.2,
    "beta_max": 6.0,
    "score_weight_schedule": "uniform",
    "score_hidden_dim": 128,
    "score_depth": 4,
    "control_arch": "mlp",
    "control_hidden_dim": 64,
    "control_depth": 3,
    "control_scale": 0.5,
}
