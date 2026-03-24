import copy
from typing import Callable, Optional

import torch

from .data import sample_from_pool, sample_gmm
from .diffusion import build_kappa_schedule, rollout_controlled_ve, sample_target_indices, sample_target_indices_log_normal
from .models import set_requires_grad
from .objective import inner_objective_attack_only, weighted_denoise_loss
from .utils import has_nan_or_inf, scalarize


def _reverse_posterior_mean(x: torch.Tensor, x0_pred: torch.Tensor, sigma: torch.Tensor, sigma_prev: torch.Tensor) -> torch.Tensor:
    """Posterior mean for VE step k->k-1 under x = x0 + sigma * eps.

    Uses:
      E[x_{k-1} | x_k, x0_hat] = alpha * x_k + (1 - alpha) * x0_hat
      alpha = sigma_{k-1}^2 / sigma_k^2
    """

    sigma2 = sigma.square().clamp_min(1e-12)
    alpha = sigma_prev.square() / sigma2
    return alpha.unsqueeze(1) * x + (1.0 - alpha).unsqueeze(1) * x0_pred


def _reverse_posterior_std(sigma: torch.Tensor, sigma_prev: torch.Tensor) -> torch.Tensor:
    """Posterior std for VE reverse transition k->k-1.

    Var[x_{k-1} | x_k, x0] = sigma_{k-1}^2 * (1 - sigma_{k-1}^2 / sigma_k^2)
    """

    sigma2 = sigma.square().clamp_min(1e-12)
    var = sigma_prev.square() * (1.0 - sigma_prev.square() / sigma2)
    return torch.sqrt(var.clamp_min(0.0))


def _sample_train_batch(
    cfg,
    centers,
    train_pool: torch.Tensor = None,
    sample_train_batch_fn: Optional[Callable[[int], torch.Tensor]] = None,
) -> torch.Tensor:
    """Uniform train-batch loader abstraction.

    Priority:
    1) dataset-specific sampler callback,
    2) finite train pool (limited-data regime),
    3) on-the-fly toy GMM population sampling.
    """

    if sample_train_batch_fn is not None:
        return sample_train_batch_fn(cfg.batch_size)
    if train_pool is None:
        return sample_gmm(cfg.batch_size, centers, cfg.data_std)
    return sample_from_pool(train_pool, cfg.batch_size)


def _robust_schedule(step: int, cfg):
    """Return current outer weights and inner-LR scale for warmup/ramp.

    Output:
      (clean_weight, attack_weight, phi_lr_scale, control_updates_enabled)
    """

    warmup_clean_steps = max(int(cfg.warmup_clean_steps), 0)
    warmup_ramp_steps = max(int(cfg.warmup_ramp_steps), 0)
    clean_weight = float(cfg.outer_clean_weight)
    attack_target = float(cfg.outer_attack_weight)
    attack_start = float(cfg.warmup_attack_weight_start)
    phi_lr_scale_start = float(cfg.warmup_phi_lr_scale_start)

    if step <= warmup_clean_steps:
        return clean_weight, attack_start, 0.0, False
    if warmup_ramp_steps > 0 and step <= (warmup_clean_steps + warmup_ramp_steps):
        progress = (step - warmup_clean_steps) / float(warmup_ramp_steps)
        attack_weight = attack_start + progress * (attack_target - attack_start)
        phi_lr_scale = phi_lr_scale_start + progress * (1.0 - phi_lr_scale_start)
        return clean_weight, attack_weight, phi_lr_scale, attack_weight > 0.0
    return clean_weight, attack_target, 1.0, attack_target > 0.0


def _rollout_control_diagnostics(roll, sigma_levels: torch.Tensor, kappa_by_step: torch.Tensor):
    """Summarize path/control magnitudes and constraint saturation statistics."""

    diff = roll.states_ctrl - roll.states_ref
    diff_l2 = torch.sqrt(diff.pow(2).sum(dim=2))
    path_delta_mean = diff_l2.mean()
    terminal_delta_mean = diff_l2[:, -1].mean()

    delta_l2 = torch.sqrt(roll.delta_path.pow(2).sum(dim=2))
    delta_norm_mean = delta_l2.mean()
    delta_norm_max = delta_l2.max()
    sigma_k = sigma_levels[:-1]
    sigma_next = sigma_levels[1:]
    delta_sigma = torch.sqrt((sigma_next.square() - sigma_k.square()).clamp_min(1e-8))
    radius = (kappa_by_step * delta_sigma).view(1, -1).to(
        device=roll.delta_path.device, dtype=roll.delta_path.dtype
    )
    delta_ratio = delta_l2 / radius.clamp_min(1e-8)
    delta_ratio_mean = delta_ratio.mean()
    delta_ratio_max = delta_ratio.max()
    return path_delta_mean, terminal_delta_mean, delta_norm_mean, delta_norm_max, delta_ratio_mean, delta_ratio_max


def _zero_control(x_ref: torch.Tensor, gap: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Reference control used for diagnostics: delta_k = 0 for all steps."""

    del gap, sigma
    return torch.zeros_like(x_ref)


def train_baseline(
    denoiser,
    centers,
    sigma_levels,
    cfg,
    train_pool: torch.Tensor = None,
    sample_train_batch_fn: Optional[Callable[[int], torch.Tensor]] = None,
):
    """Train baseline denoiser theta on standard EDM weighted denoising loss."""

    optimizer = torch.optim.Adam(denoiser.parameters(), lr=cfg.lr_theta)
    history = {"loss": []}
    sigma_counts = torch.zeros(sigma_levels.numel() - 1, device=sigma_levels.device, dtype=torch.long)

    ema_model = None
    if cfg.use_ema_eval:
        ema_model = copy.deepcopy(denoiser).eval()
        set_requires_grad(ema_model, False)

    for step in range(1, cfg.steps + 1):
        x0 = _sample_train_batch(
            cfg,
            centers,
            train_pool=train_pool,
            sample_train_batch_fn=sample_train_batch_fn,
        )
        if cfg.use_log_normal_sigma_sampling:
            indices = sample_target_indices_log_normal(
                cfg.batch_size,
                sigma_levels,
                p_mean=cfg.p_mean,
                p_std=cfg.p_std,
            )
        else:
            indices = sample_target_indices(cfg.batch_size, sigma_levels)
        sigma_counts += torch.bincount(indices - 1, minlength=sigma_counts.numel())
        sigma = sigma_levels[indices]
        x_noisy = x0 + sigma.unsqueeze(1) * torch.randn_like(x0)

        optimizer.zero_grad(set_to_none=True)
        loss = weighted_denoise_loss(denoiser, x_noisy, x0, sigma, cfg.sigma_data)
        if has_nan_or_inf(loss):
            raise RuntimeError("NaN/Inf detected in baseline loss.")
        loss.backward()
        optimizer.step()
        if ema_model is not None:
            with torch.no_grad():
                for p_ema, p in zip(ema_model.parameters(), denoiser.parameters()):
                    p_ema.mul_(cfg.ema_decay).add_(p, alpha=1.0 - cfg.ema_decay)
        history["loss"].append(scalarize(loss))

        if step % cfg.log_every == 0:
            print(f"[baseline] step={step:05d} loss={loss.item():.6f}", flush=True)

    history["sigma_counts"] = [int(v) for v in sigma_counts.detach().cpu().tolist()]
    eval_model = ema_model if ema_model is not None else denoiser
    return history, eval_model


def train_trajectory_robust_constrained(
    denoiser,
    control,
    centers,
    sigma_levels,
    cfg,
    train_pool: torch.Tensor = None,
    sample_train_batch_fn: Optional[Callable[[int], torch.Tensor]] = None,
):
    """Bilevel robust training with hard per-step control constraint.

    Inner (ascent in phi):
      maximize J(phi; theta) = L_denoise(theta, rollout_phi(x0))

    Outer (descent in theta):
      minimize attack_weight * L_attack + clean_weight * L_clean

    with projected control per step:
      ||delta_k||_2 <= kappa_k * Delta_sigma_k.
    """

    optimizer_theta = torch.optim.Adam(denoiser.parameters(), lr=cfg.lr_theta)
    optimizer_phi = torch.optim.Adam(control.parameters(), lr=cfg.lr_phi)
    kappa_by_step = build_kappa_schedule(
        sigma_levels=sigma_levels,
        base_kappa=cfg.control_radius_kappa,
        use_time_dependent=cfg.use_time_dependent_kappa,
        low_multiplier=cfg.kappa_low_multiplier,
        mid_multiplier=cfg.kappa_mid_multiplier,
        high_multiplier=cfg.kappa_high_multiplier,
        preserve_l2_budget=cfg.kappa_preserve_l2_budget,
    ).to(device=sigma_levels.device, dtype=sigma_levels.dtype)
    history = {
        "outer_loss": [],
        "outer_loss_attack": [],
        "outer_loss_clean": [],
        "inner_obj": [],
        "delta_norm_mean": [],
        "delta_norm_max": [],
        "delta_norm_ratio_mean": [],
        "delta_norm_ratio_max": [],
        "sched_attack_weight": [],
        "sched_clean_weight": [],
        "sched_phi_lr_scale": [],
        "diag_step": [],
        "diag_inner_obj_current": [],
        "diag_inner_obj_zero": [],
        "diag_inner_obj_gap": [],
        "diag_inner_obj_gap_ratio": [],
        "diag_delta_norm_mean": [],
        "diag_delta_norm_max": [],
        "diag_delta_norm_ratio_mean": [],
        "diag_delta_norm_ratio_max": [],
        "diag_path_delta_mean": [],
        "diag_terminal_delta_mean": [],
    }

    for step in range(1, cfg.steps + 1):
        x0 = _sample_train_batch(
            cfg,
            centers,
            train_pool=train_pool,
            sample_train_batch_fn=sample_train_batch_fn,
        )
        if cfg.use_log_normal_sigma_sampling:
            indices = sample_target_indices_log_normal(
                cfg.batch_size,
                sigma_levels,
                p_mean=cfg.p_mean,
                p_std=cfg.p_std,
            )
        else:
            indices = sample_target_indices(cfg.batch_size, sigma_levels)

        clean_weight, attack_weight, phi_lr_scale, control_updates_enabled = _robust_schedule(step, cfg)
        for group in optimizer_phi.param_groups:
            group["lr"] = float(cfg.lr_phi) * float(phi_lr_scale)

        # Inner ascent on control network.
        set_requires_grad(denoiser, False)
        set_requires_grad(control, True)
        last_inner_obj = 0.0
        last_delta_norm_mean = 0.0
        last_delta_norm_max = 0.0
        last_delta_ratio_mean = 0.0
        last_delta_ratio_max = 0.0

        if control_updates_enabled and phi_lr_scale > 0.0 and cfg.inner_steps > 0:
            for _ in range(cfg.inner_steps):
                optimizer_phi.zero_grad(set_to_none=True)
                roll = rollout_controlled_ve(
                    x0=x0,
                    target_indices=indices,
                    control_net=control,
                    sigma_levels=sigma_levels,
                    grad_through_control=True,
                    control_radius_kappa=cfg.control_radius_kappa,
                    kappa_by_step=kappa_by_step,
                )
                train_loss = weighted_denoise_loss(denoiser, roll.x_target, x0, roll.sigma_target, cfg.sigma_data)
                inner_obj = inner_objective_attack_only(train_loss)
                if has_nan_or_inf(inner_obj):
                    raise RuntimeError("NaN/Inf detected in inner objective.")
                (-inner_obj).backward()
                if cfg.clip_phi_grad > 0:
                    torch.nn.utils.clip_grad_norm_(control.parameters(), cfg.clip_phi_grad)
                optimizer_phi.step()
                last_inner_obj = scalarize(inner_obj)
                delta_l2 = torch.sqrt(roll.delta_path.pow(2).sum(dim=2))
                last_delta_norm_mean = scalarize(delta_l2.mean())
                last_delta_norm_max = scalarize(delta_l2.max())
                sigma_k = sigma_levels[:-1]
                sigma_next = sigma_levels[1:]
                delta_sigma = torch.sqrt((sigma_next.square() - sigma_k.square()).clamp_min(1e-8))
                radius = (kappa_by_step * delta_sigma).view(1, -1).to(
                    device=delta_l2.device, dtype=delta_l2.dtype
                )
                delta_ratio = delta_l2 / radius.clamp_min(1e-8)
                last_delta_ratio_mean = scalarize(delta_ratio.mean())
                last_delta_ratio_max = scalarize(delta_ratio.max())

        # Outer descent on denoiser.
        set_requires_grad(denoiser, True)
        set_requires_grad(control, False)
        optimizer_theta.zero_grad(set_to_none=True)
        roll = rollout_controlled_ve(
            x0=x0,
            target_indices=indices,
            control_net=control,
            sigma_levels=sigma_levels,
            grad_through_control=False,
            control_radius_kappa=cfg.control_radius_kappa,
            kappa_by_step=kappa_by_step,
        )
        outer_loss_attack = weighted_denoise_loss(denoiser, roll.x_target, x0, roll.sigma_target, cfg.sigma_data)
        outer_loss_clean = torch.zeros((), device=x0.device, dtype=x0.dtype)
        if clean_weight > 0.0:
            x_ref_target = roll.states_ref[torch.arange(x0.shape[0], device=x0.device), indices]
            outer_loss_clean = weighted_denoise_loss(denoiser, x_ref_target, x0, roll.sigma_target, cfg.sigma_data)
        outer_loss = attack_weight * outer_loss_attack + clean_weight * outer_loss_clean
        if has_nan_or_inf(outer_loss):
            raise RuntimeError("NaN/Inf detected in outer loss.")
        outer_loss.backward()
        optimizer_theta.step()
        set_requires_grad(control, True)
        if not control_updates_enabled:
            delta_l2 = torch.sqrt(roll.delta_path.pow(2).sum(dim=2))
            last_delta_norm_mean = scalarize(delta_l2.mean())
            last_delta_norm_max = scalarize(delta_l2.max())
            sigma_k = sigma_levels[:-1]
            sigma_next = sigma_levels[1:]
            delta_sigma = torch.sqrt((sigma_next.square() - sigma_k.square()).clamp_min(1e-8))
            radius = (kappa_by_step * delta_sigma).view(1, -1).to(
                device=delta_l2.device, dtype=delta_l2.dtype
            )
            delta_ratio = delta_l2 / radius.clamp_min(1e-8)
            last_delta_ratio_mean = scalarize(delta_ratio.mean())
            last_delta_ratio_max = scalarize(delta_ratio.max())

        history["outer_loss"].append(scalarize(outer_loss))
        history["outer_loss_attack"].append(scalarize(outer_loss_attack))
        history["outer_loss_clean"].append(scalarize(outer_loss_clean))
        history["inner_obj"].append(last_inner_obj)
        history["delta_norm_mean"].append(last_delta_norm_mean)
        history["delta_norm_max"].append(last_delta_norm_max)
        history["delta_norm_ratio_mean"].append(last_delta_ratio_mean)
        history["delta_norm_ratio_max"].append(last_delta_ratio_max)
        history["sched_attack_weight"].append(float(attack_weight))
        history["sched_clean_weight"].append(float(clean_weight))
        history["sched_phi_lr_scale"].append(float(phi_lr_scale))

        run_diag = (
            bool(cfg.collapse_diagnostics_enabled)
            and (step % max(int(cfg.collapse_diag_every), 1) == 0 or step == 1 or step == int(cfg.steps))
        )
        if run_diag:
            with torch.no_grad():
                roll_cur_diag = rollout_controlled_ve(
                    x0=x0,
                    target_indices=indices,
                    control_net=control,
                    sigma_levels=sigma_levels,
                    grad_through_control=False,
                    control_radius_kappa=cfg.control_radius_kappa,
                    kappa_by_step=kappa_by_step,
                )
                train_loss_cur = weighted_denoise_loss(
                    denoiser,
                    roll_cur_diag.x_target,
                    x0,
                    roll_cur_diag.sigma_target,
                    cfg.sigma_data,
                )
                inner_obj_cur = inner_objective_attack_only(train_loss_cur)

                roll_zero_diag = rollout_controlled_ve(
                    x0=x0,
                    target_indices=indices,
                    control_net=_zero_control,
                    sigma_levels=sigma_levels,
                    grad_through_control=False,
                    control_radius_kappa=cfg.control_radius_kappa,
                    kappa_by_step=kappa_by_step,
                )
                train_loss_zero = weighted_denoise_loss(
                    denoiser,
                    roll_zero_diag.x_target,
                    x0,
                    roll_zero_diag.sigma_target,
                    cfg.sigma_data,
                )
                inner_obj_zero = inner_objective_attack_only(train_loss_zero)
                gap = inner_obj_cur - inner_obj_zero
                gap_ratio = gap / (inner_obj_zero.abs() + 1e-8)
                (
                    path_delta_mean,
                    terminal_delta_mean,
                    delta_norm_mean,
                    delta_norm_max,
                    delta_ratio_mean,
                    delta_ratio_max,
                ) = _rollout_control_diagnostics(
                    roll_cur_diag,
                    sigma_levels,
                    kappa_by_step,
                )

            history["diag_step"].append(int(step))
            history["diag_inner_obj_current"].append(scalarize(inner_obj_cur))
            history["diag_inner_obj_zero"].append(scalarize(inner_obj_zero))
            history["diag_inner_obj_gap"].append(scalarize(gap))
            history["diag_inner_obj_gap_ratio"].append(scalarize(gap_ratio))
            history["diag_delta_norm_mean"].append(scalarize(delta_norm_mean))
            history["diag_delta_norm_max"].append(scalarize(delta_norm_max))
            history["diag_delta_norm_ratio_mean"].append(scalarize(delta_ratio_mean))
            history["diag_delta_norm_ratio_max"].append(scalarize(delta_ratio_max))
            history["diag_path_delta_mean"].append(scalarize(path_delta_mean))
            history["diag_terminal_delta_mean"].append(scalarize(terminal_delta_mean))

        if step % cfg.log_every == 0:
            diag_msg = ""
            if len(history["diag_step"]) > 0:
                diag_msg = (
                    " "
                    f"diag_gap={history['diag_inner_obj_gap'][-1]:.6f} "
                    f"diag_gap_ratio={history['diag_inner_obj_gap_ratio'][-1]:.6f} "
                    f"diag_delta_ratio={history['diag_delta_norm_ratio_mean'][-1]:.6f} "
                    f"diag_path_delta={history['diag_path_delta_mean'][-1]:.6f}"
                )
            print(
                f"[robust] step={step:05d} outer_loss={outer_loss.item():.6f} "
                f"attack={outer_loss_attack.item():.6f} clean={outer_loss_clean.item():.6f} "
                f"inner_obj={last_inner_obj:.6f} "
                f"delta_norm={last_delta_norm_mean:.6f} delta_ratio={last_delta_ratio_mean:.6f} "
                f"w_attack={attack_weight:.3f} w_clean={clean_weight:.3f} phi_lr_scale={phi_lr_scale:.3f}"
                f"{diag_msg}",
                flush=True,
            )

    return history


def train_trajectory_robust_energy(
    denoiser,
    control,
    centers,
    sigma_levels,
    cfg,
    train_pool: torch.Tensor = None,
    sample_train_batch_fn: Optional[Callable[[int], torch.Tensor]] = None,
):
    """Backward-compatible alias to constrained robust training."""

    # Backward-compatible alias; v2 now uses constrained control (no energy penalty).
    return train_trajectory_robust_constrained(
        denoiser,
        control,
        centers,
        sigma_levels,
        cfg,
        train_pool=train_pool,
        sample_train_batch_fn=sample_train_batch_fn,
    )


@torch.no_grad()
def sample_with_denoiser(denoiser, sigma_levels, n_samples, device):
    """Sample x0 by running reverse paths from Gaussian terminal noise."""

    states = sample_reverse_paths(denoiser, sigma_levels, n_samples, device, stochastic=True)
    return states[:, 0]


@torch.no_grad()
def sample_reverse_paths(denoiser, sigma_levels, n_samples, device, stochastic: bool = True, noise_schedule: torch.Tensor = None):
    """Generate full reverse trajectories from sigma_N to sigma_0.

    Start from x_N ~ N(0, sigma_N^2 I), then iterate posterior update with
    x0_hat = denoiser(x_k, sigma_k). If `stochastic=False`, use posterior mean only.
    """

    x = torch.randn(n_samples, 2, device=device) * sigma_levels[-1]
    n_steps = sigma_levels.numel() - 1
    states = [None for _ in range(n_steps + 1)]
    states[n_steps] = x

    if noise_schedule is not None:
        if noise_schedule.shape[0] < (n_steps + 1):
            raise ValueError(
                f"noise_schedule first dim must be >= {n_steps + 1}, got {noise_schedule.shape[0]}"
            )
        if noise_schedule.shape[1] != n_samples or noise_schedule.shape[2] != x.shape[1]:
            raise ValueError(
                "noise_schedule must have shape [>=N+1, n_samples, data_dim], "
                f"got {tuple(noise_schedule.shape)}"
            )

    for k in range(n_steps, 0, -1):
        sigma = torch.full((n_samples,), sigma_levels[k], device=device, dtype=x.dtype)
        sigma_next = torch.full((n_samples,), sigma_levels[k - 1], device=device, dtype=x.dtype)
        x0_pred = denoiser(x, sigma)
        mean = _reverse_posterior_mean(x, x0_pred, sigma, sigma_next)
        if stochastic and k > 1:
            std = _reverse_posterior_std(sigma, sigma_next)
            if noise_schedule is None:
                eps = torch.randn_like(x)
            else:
                eps = noise_schedule[k].to(device=x.device, dtype=x.dtype)
            x = mean + std.unsqueeze(1) * eps
        else:
            x = mean
        states[k - 1] = x
    return torch.stack(states, dim=1)


@torch.no_grad()
def reverse_paths_from_terminal(
    denoiser,
    x_terminal,
    sigma_levels,
    stochastic: bool = False,
    noise_schedule: torch.Tensor = None,
):
    """Reverse trajectories from provided terminal states x_k at the final index.

    Returns tensor indexed by ladder index (0..N): index N is input terminal
    state, index 0 is recovered clean estimate after reverse rollout.
    """

    n_steps = sigma_levels.numel() - 1
    x = x_terminal.clone()
    states = [None for _ in range(n_steps + 1)]
    states[n_steps] = x

    if noise_schedule is not None:
        if noise_schedule.shape[0] < (n_steps + 1):
            raise ValueError(
                f"noise_schedule first dim must be >= {n_steps + 1}, got {noise_schedule.shape[0]}"
            )
        if noise_schedule.shape[1] != x.shape[0] or noise_schedule.shape[2] != x.shape[1]:
            raise ValueError(
                "noise_schedule must have shape [>=N+1, batch, data_dim], "
                f"got {tuple(noise_schedule.shape)}"
            )

    for k in range(n_steps, 0, -1):
        sigma = torch.full((x.shape[0],), sigma_levels[k], device=x.device, dtype=x.dtype)
        sigma_prev = torch.full((x.shape[0],), sigma_levels[k - 1], device=x.device, dtype=x.dtype)
        x0_pred = denoiser(x, sigma)
        mean = _reverse_posterior_mean(x, x0_pred, sigma, sigma_prev)
        if stochastic and k > 1:
            std = _reverse_posterior_std(sigma, sigma_prev)
            if noise_schedule is None:
                eps = torch.randn_like(x)
            else:
                eps = noise_schedule[k].to(device=x.device, dtype=x.dtype)
            x = mean + std.unsqueeze(1) * eps
        else:
            x = mean
        states[k - 1] = x

    return torch.stack(states, dim=1)
