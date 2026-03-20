from dataclasses import dataclass


@dataclass
class ToyConfig:
    outdir: str = "toy_outputs"
    exp_name: str = "energy_only_v1"
    seed: int = 0
    device: str = "auto"  # auto|cpu|cuda

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

    # Energy-only robust objective.
    lambda_energy: float = 0.2
    outer_attack_weight: float = 1.0
    outer_clean_weight: float = 0.0
    warmup_clean_steps: int = 0
    warmup_ramp_steps: int = 0
    warmup_attack_weight_start: float = 0.0
    warmup_phi_lr_scale_start: float = 0.0
    collapse_diagnostics_enabled: bool = True
    collapse_diag_every: int = 100
    collapse_gap_ratio_tol: float = 0.05
    collapse_v_l2_tol: float = 0.05

    # Baseline acceptance gate (must pass before attack training).
    baseline_gate_enabled: bool = True
    baseline_gate_error_on_fail: bool = False
    baseline_gate_min_coverage: float = 1.0
    baseline_gate_max_generated_avg_min_dist: float = 0.30
    baseline_gate_max_generated_p90_min_dist: float = 0.45
    baseline_gate_max_endpoint_avg_min_dist: float = 0.35
    baseline_gate_max_endpoint_p90_min_dist: float = 0.60

    # Reverse terminal consistency check:
    # at k=terminal, reverse state must match the forward terminal state.
    reverse_terminal_assert_tol: float = 1e-8

    # Checks.
    run_checks: bool = True
    gradcheck_eps: float = 1e-4
    gradcheck_tol: float = 5e-2
