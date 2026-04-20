from dataclasses import dataclass
from typing import Optional


@dataclass
class ToyConfig:
    outdir: str = "toy_outputs"
    exp_name: str = "simpsons_mnist_rgb_cdro"
    seed: int = 0
    device: str = "auto"  # auto|cpu|cuda
    allow_tf32: bool = True
    cudnn_benchmark: bool = True
    amp_dtype: str = "auto"  # auto|off|bfloat16|float16
    method_version: str = "cdro"  # one of supported versions in toy/versions/registry.py
    dataset_kind: str = "image_folder"
    model_kind: str = "auto"  # auto|toy_mlp|...
    image_backbone: str = "conv"  # conv|songunet|ddpmpp
    baseline_train_backend: str = "toy"  # toy|mainline
    baseline_train_batch_gpu: int = 0  # 0=auto, else per-GPU microbatch for mainline baseline training
    diagnostics_kind: str = "auto"  # auto|toy_gmm|...
    dataset_path: str = "toy_data/simpsons_mnist_rgb/imagefolder/train"
    dataset_val_path: str = "toy_data/simpsons_mnist_rgb/imagefolder/test"
    image_size: int = 28
    image_channels: int = 3
    image_train_size: int = 400
    image_val_size: int = 2000
    image_split_seed: int = 0
    # MNIST limited-data controls (for fair low-data protocols):
    # - when enabled, train/val subset sizes are resolved by percent of full split.
    # - keep image_*_size as backward-compatible fallback by disabling this mode.
    mnist_use_percent_split: bool = True
    mnist_train_percent: float = 20.0
    mnist_val_percent: float = 100.0
    image_gate_min_generated_std: float = 0.10
    image_gate_min_endpoint_std: float = 0.10
    image_gate_max_endpoint_recovery_mse: float = 0.30

    # Training.
    steps: int = 3000
    batch_size: int = 256
    log_every: int = 200
    eval_samples: int = 6000
    debug_eval_batch: int = 256
    debug_terminal_step: int = 20  # terminal step used for the 4-row forward/backward debug plot.
    plot_stochastic_backward: bool = False
    baseline_only: bool = False
    use_ema_eval: bool = True
    ema_mode: str = "official"  # official|fixed
    ema_decay: float = 0.999  # only used when ema_mode='fixed'
    ema_halflife_kimg: float = 500.0
    ema_rampup_ratio: Optional[float] = 0.05
    fid_ref_path: str = "toy_data/simpsons_mnist_rgb/fid_refs/simpsons_mnist_rgb_test_28x28.npz"
    fid_ref_policy: str = "auto"
    weighted_compute_calibration_path: str = ""
    weighted_inputgrad_alpha: float = 0.0
    weighted_parambackward_beta: float = 0.0
    # Baseline checkpoint cache for fair cross-version comparison:
    # - if enabled and checkpoint exists: load baseline (skip baseline retrain).
    # - else: train baseline once, then save it for reuse by later runs.
    baseline_ckpt_enabled: bool = True
    baseline_ckpt_path: str = ""
    baseline_ckpt_force_retrain: bool = False
    baseline_ckpt_strict_meta: bool = True
    robust_resume_ckpt_path: str = ""
    robust_save_ckpt_path: str = ""
    baseline_steps_override: int = 0
    wandb_enabled: bool = False
    wandb_project: str = "GM-CDRO"
    wandb_entity: str = "lechibachh"
    wandb_name: str = ""
    wandb_group: str = ""
    wandb_tags: str = ""
    wandb_mode: str = "online"

    # Model.
    hidden_dim: int = 64
    lr_theta: float = 1e-3
    lr_phi: float = 5e-4
    inner_steps: int = 1
    attack_num_steps: Optional[int] = None  # None -> use legacy inner_steps, else {1,2} for CDRO attack steps
    clip_phi_grad: float = 1.0
    training_objective: str = "edm"  # edm|score|rf
    score_matching_weight_power: float = 2.0  # lambda(sigma)=sigma^p in score objective
    # Rectified Flow baseline controls. Public RF runs follow the RF++-style
    # explicit reflow protocol: start from the shared EDM checkpoint, use that
    # checkpoint as the frozen teacher, and train the RF-family student directly
    # on teacher-generated pairs.
    rf_baseline_mode: str = "strong"  # strong|plain
    rf_reflow_t_distribution: str = "u_shaped"  # u_shaped|uniform
    rf_loss: str = "pseudo_huber"  # pseudo_huber|mse
    rf_pseudo_huber_delta: float = 0.1
    rf_edm_init_ckpt_path: str = ""  # required shared EDM warm-start for RF / CDRO-RF / Wild-Diffusion-RF
    rf_continuation_total_steps_override: int = 0  # stable RF continuation budget for resumed checkpoint sweeps
    rf_cdro_pair_source: str = "auto"  # auto|reflow|data_noise; auto now resolves to explicit reflow
    rf_teacher_n_steps_path: int = 40  # shared RF reflow teacher grid; RF++-style default
    rf_eval_n_steps_path: int = 9  # shared RF eval/FID grid; fast RF-family inference default

    # Data.
    n_modes: int = 8
    mode_radius: float = 2.0
    data_std: float = 0.08
    limited_data_enabled: bool = True
    train_points_per_mode: int = 2
    val_size: int = 10000

    # Diffusion ladder.
    n_steps_path: int = 32
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
    # CDRO Route-A options (beta-space greedy attack with sigma-time local caps).
    cdro_step_size: float = 0.02
    cdro_total_budget_rho: float = 0.001
    cdro_time_horizon: float = 1.0  # Total span of the sigma-induced auxiliary clock.
    cdro_edm_ladder_mode: str = "stochastic_stratified_quantile"  # deterministic_midpoint_quantile|stochastic_stratified_quantile
    cdro_per_example_sigma_ladders: bool = True
    cdro_eval_stochastic_ladders: bool = False
    # Reference WDRO warmup step fraction. CDRO converts this into a target
    # weighted-compute warmup share, then solves for the baseline warmup steps
    # needed to match that share under CDRO's more expensive robust steps.
    cdro_warmup_fraction: float = 0.2
    control_radius_kappa: float = 0.15
    # v2.1 non-Markovian reference update parameter in [0,1].
    v21_rho: float = 0.8
    use_time_dependent_kappa: bool = False
    kappa_low_multiplier: float = 1.0
    kappa_mid_multiplier: float = 1.0
    kappa_high_multiplier: float = 1.0
    kappa_preserve_l2_budget: bool = True
    outer_attack_weight: float = 1.0
    outer_clean_weight: float = 0.0
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
    wdro_warmup_fraction: float = 0.05
    wdro_refresh_epochs: float = 100.0
    wdro_adv_prob: float = 0.3
    wdro_attack_steps: int = 2
    wdro_attack_step_size: float = 1e-3
    wdro_gamma: float = 1.0
    wdro_clamp_samples: bool = True
    wdro_sample_min: float = -1.0
    wdro_sample_max: float = 1.0

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
