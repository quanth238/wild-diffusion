from dataclasses import dataclass


@dataclass
class ToyConfig:
    outdir: str = "toy_outputs"
    exp_name: str = "constrained_v2"
    seed: int = 0
    device: str = "auto"  # auto|cpu|cuda
    method_version: str = "v2"  # v2 implemented; v1 is scaffold-only in /toy.
    dataset_kind: str = "toy_gmm"  # extension point: add image dataset backends later.
    model_kind: str = "auto"  # auto|toy_mlp|...
    diagnostics_kind: str = "auto"  # auto|toy_gmm|...
    dataset_path: str = ""
    dataset_val_path: str = ""
    image_size: int = 32
    image_channels: int = 3
    image_train_size: int = 2000
    image_val_size: int = 2000
    image_split_seed: int = 0
    image_gate_min_generated_std: float = 0.10
    image_gate_min_endpoint_std: float = 0.10
    image_gate_max_endpoint_recovery_mse: float = 0.30

    # Training.
    steps: int = 3000
    batch_size: int = 512
    log_every: int = 200
    eval_samples: int = 6000
    debug_eval_batch: int = 256
    debug_terminal_step: int = 20  # terminal step used for the 4-row forward/backward debug plot.
    plot_stochastic_backward: bool = False
    baseline_only: bool = False
    use_ema_eval: bool = False
    ema_decay: float = 0.995

    # Model.
    hidden_dim: int = 128
    lr_theta: float = 1e-3
    lr_phi: float = 5e-4
    inner_steps: int = 1
    clip_phi_grad: float = 1.0
    training_objective: str = "edm"  # edm|score
    score_matching_weight_power: float = 2.0  # lambda(sigma)=sigma^p in score objective

    # Data.
    n_modes: int = 8
    mode_radius: float = 2.0
    data_std: float = 0.08
    limited_data_enabled: bool = True
    train_points_per_mode: int = 2
    val_size: int = 10000

    # Diffusion ladder.
    n_steps_path: int = 24
    sigma_min: float = 0.002
    sigma_max: float = 2.0  # Keep terminal noise moderate for better x0-recovery diagnostics.
    sigma_data: float = -1.0  # <=0: auto-estimate from toy data scale.
    use_log_normal_sigma_sampling: bool = True
    auto_log_normal_params: bool = False
    p_mean: float = -1.2
    p_std: float = 1.2

    # Constrained robust objective (v2): hard per-step control radius.
    # Deprecated in v2 (kept for backward CLI compatibility only).
    lambda_energy: float = 0.2
    # v1 dual-lambda options (energy-only robust surrogate):
    # min_{theta, lambda>=0} rho*lambda + sup_u [L_attack(theta,u) - lambda*C_energy(u)].
    v1_dual_lambda_enabled: bool = True
    v1_energy_budget_rho: float = 0.02
    v1_lambda_init: float = 1.0
    v1_lambda_lr: float = 5e-4
    v1_lambda_max: float = 100.0
    control_radius_kappa: float = 0.15
    # v2.1 non-Markovian reference update parameter in [0,1].
    v21_rho: float = 0.8
    use_time_dependent_kappa: bool = False
    kappa_low_multiplier: float = 1.0
    kappa_mid_multiplier: float = 1.0
    kappa_high_multiplier: float = 1.0
    kappa_preserve_l2_budget: bool = True
    outer_attack_weight: float = 0.5
    outer_clean_weight: float = 1.0
    warmup_clean_steps: int = 900
    warmup_ramp_steps: int = 600
    warmup_attack_weight_start: float = 0.0
    warmup_phi_lr_scale_start: float = 0.0
    collapse_diagnostics_enabled: bool = True
    collapse_diag_every: int = 100
    collapse_gap_ratio_tol: float = 0.05
    collapse_delta_ratio_tol: float = 0.05
    constraint_saturation_threshold: float = 0.98

    # WILD-style sample-level WDRO surrogate options (method-version=wild).
    wild_update_interval: int = 20
    wild_cache_batches: int = 4
    wild_inner_steps: int = 3
    wild_step_size: float = 0.05
    wild_gamma: float = 2.0
    wild_fixed_noise_inner: bool = True
    wild_clamp_samples: bool = False
    wild_sample_min: float = -1.0
    wild_sample_max: float = 1.0
    wild_delta_ratio_denom: float = 1.0

    # Baseline acceptance gate (must pass before attack training).
    baseline_gate_enabled: bool = True
    baseline_gate_error_on_fail: bool = False
    baseline_gate_min_coverage: float = 1.0
    baseline_gate_max_generated_avg_min_dist: float = 0.30
    baseline_gate_max_generated_p90_min_dist: float = 0.45
    baseline_gate_max_endpoint_avg_min_dist: float = 0.35
    baseline_gate_max_endpoint_p90_min_dist: float = 0.60

    compute_fid: bool = False
    fid_samples: int = 2000
    # Fair/faithful evaluation controls.
    # - gate seed is scoped to gate diagnostics only (RNG state restored afterward).
    # - metrics seed is scoped to post-training diagnostics/FID generation only.
    eval_seed_offset_gate: int = 10000
    eval_seed_offset_metrics: int = 20000
    eval_use_shared_terminal_noise: bool = True
    eval_use_shared_reverse_noise: bool = True

    # Reverse terminal consistency check:
    # at k=terminal, reverse state must match the forward terminal state.
    reverse_terminal_assert_tol: float = 1e-8

    # Checks.
    run_checks: bool = True
    gradcheck_eps: float = 1e-4
    gradcheck_tol: float = 5e-2
