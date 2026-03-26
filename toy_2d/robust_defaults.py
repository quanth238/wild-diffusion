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
    "adversary_stop_epoch": 0,
    "score_steps": 8,
    "terminal_momentum": 0.95,
    "terminal_sampler": "gaussian",
    "terminal_buffer_size": 4096,
    "terminal_jitter_scale": 0.0,
    "num_steps": 12,
    "total_time": 1.0,
    "beta_min": 0.2,
    "beta_max": 6.0,
    "score_weight_schedule": "uniform",
    "score_hidden_dim": 128,
    "score_depth": 4,
    "score_arch": "precond",
    "control_arch": "mlp",
    "control_hidden_dim": 64,
    "control_depth": 3,
    "control_scale": 0.5,
    "sigma_data": 0.5,
    "reverse_solver": "heun",
    "reverse_noise_scale": 1.0,
    "reverse_tail_noise_scale": 1.0,
    "reverse_deterministic_tail_steps": 0,
    "budget_mode": "fixed",
    "budget_estimate_batch_size": 256,
    "budget_scale": 1.0,
    "budget_ema_decay": 0.9,
    "budget_max_ratio": 4.0,
    "budget_schedule": "constant",
    "budget_frontload_power": 2.0,
    "budget_frontload_floor": 0.25,
    "reference_wdro_k": WDRO_CORE_DEFAULTS["k"],
    "reference_wdro_step_size": WDRO_CORE_DEFAULTS["step_size"],
    "reference_wdro_gamma": WDRO_CORE_DEFAULTS["gamma"],
}
