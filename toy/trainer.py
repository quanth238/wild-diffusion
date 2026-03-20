import copy
import math

import torch

from .data import sample_from_pool, sample_gmm
from .diffusion import rollout_controlled_ve, sample_target_indices, sample_target_indices_log_normal
from .models import set_requires_grad
from .objective import inner_objective_energy_only, weighted_denoise_loss
from .utils import has_nan_or_inf, scalarize


def _reverse_posterior_mean(x: torch.Tensor, x0_pred: torch.Tensor, sigma: torch.Tensor, sigma_prev: torch.Tensor) -> torch.Tensor:
    sigma2 = sigma.square().clamp_min(1e-12)
    alpha = sigma_prev.square() / sigma2
    return alpha.unsqueeze(1) * x + (1.0 - alpha).unsqueeze(1) * x0_pred


def _reverse_posterior_std(sigma: torch.Tensor, sigma_prev: torch.Tensor) -> torch.Tensor:
    sigma2 = sigma.square().clamp_min(1e-12)
    var = sigma_prev.square() * (1.0 - sigma_prev.square() / sigma2)
    return torch.sqrt(var.clamp_min(0.0))


def _sample_train_batch(cfg, centers, train_pool: torch.Tensor = None) -> torch.Tensor:
    if train_pool is None:
        return sample_gmm(cfg.batch_size, centers, cfg.data_std)
    return sample_from_pool(train_pool, cfg.batch_size)


def _robust_schedule(step: int, cfg):
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


def _rollout_control_diagnostics(states_ref: torch.Tensor, states_ctrl: torch.Tensor, sigma_levels: torch.Tensor):
    # states_*: [B, N+1, D], indexed by noise step k.
    n_steps = int(sigma_levels.numel() - 1)
    dt = 1.0 / float(n_steps)
    sqrt_dt = math.sqrt(dt)

    diff = states_ctrl - states_ref
    diff_l2 = torch.sqrt(diff.pow(2).sum(dim=2))
    path_delta_mean = diff_l2.mean()
    terminal_delta_mean = diff_l2[:, -1].mean()

    # control_increment_k = (x_ctrl_{k+1}-x_ref_{k+1}) - (x_ctrl_k-x_ref_k)
    control_increment = diff[:, 1:] - diff[:, :-1]  # [B, N, D]
    sigma_k = sigma_levels[:-1]
    sigma_next = sigma_levels[1:]
    delta_sigma = torch.sqrt((sigma_next.square() - sigma_k.square()).clamp_min(1e-8))  # [N]
    scale = (delta_sigma * sqrt_dt).view(1, n_steps, 1).to(device=control_increment.device, dtype=control_increment.dtype)
    v_est = control_increment / scale
    v_l2_mean = torch.sqrt(v_est.pow(2).sum(dim=2)).mean()
    return path_delta_mean, terminal_delta_mean, v_l2_mean


def _zero_control(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    del t
    return torch.zeros_like(x)


def train_baseline(denoiser, centers, sigma_levels, cfg, train_pool: torch.Tensor = None):
    optimizer = torch.optim.Adam(denoiser.parameters(), lr=cfg.lr_theta)
    history = {"loss": []}
    sigma_counts = torch.zeros(sigma_levels.numel() - 1, device=sigma_levels.device, dtype=torch.long)

    ema_model = None
    if cfg.use_ema_eval:
        ema_model = copy.deepcopy(denoiser).eval()
        set_requires_grad(ema_model, False)

    for step in range(1, cfg.steps + 1):
        x0 = _sample_train_batch(cfg, centers, train_pool=train_pool)
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


def train_trajectory_robust_energy(denoiser, control, centers, sigma_levels, cfg, train_pool: torch.Tensor = None):
    optimizer_theta = torch.optim.Adam(denoiser.parameters(), lr=cfg.lr_theta)
    optimizer_phi = torch.optim.Adam(control.parameters(), lr=cfg.lr_phi)
    history = {
        "outer_loss": [],
        "outer_loss_attack": [],
        "outer_loss_clean": [],
        "inner_obj": [],
        "energy": [],
        "sched_attack_weight": [],
        "sched_clean_weight": [],
        "sched_phi_lr_scale": [],
        "diag_step": [],
        "diag_inner_obj_current": [],
        "diag_inner_obj_zero": [],
        "diag_inner_obj_gap": [],
        "diag_inner_obj_gap_ratio": [],
        "diag_v_l2_mean": [],
        "diag_path_delta_mean": [],
        "diag_terminal_delta_mean": [],
    }

    # Keep ControlNet's random initialization. Zero-initializing all layers causes
    # gradient starvation in hidden layers for this MLP (only last-layer bias gets
    # non-zero gradients), which can falsely look like adversary collapse.

    for step in range(1, cfg.steps + 1):
        x0 = _sample_train_batch(cfg, centers, train_pool=train_pool)
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
        last_energy = 0.0

        if control_updates_enabled and phi_lr_scale > 0.0 and cfg.inner_steps > 0:
            for _ in range(cfg.inner_steps):
                optimizer_phi.zero_grad(set_to_none=True)
                roll = rollout_controlled_ve(
                    x0=x0,
                    target_indices=indices,
                    control_net=control,
                    sigma_levels=sigma_levels,
                    grad_through_control=True,
                )
                train_loss = weighted_denoise_loss(denoiser, roll.x_target, x0, roll.sigma_target, cfg.sigma_data)
                inner_obj = inner_objective_energy_only(train_loss, roll.energy, cfg.lambda_energy)
                if has_nan_or_inf(inner_obj):
                    raise RuntimeError("NaN/Inf detected in inner objective.")
                (-inner_obj).backward()
                if cfg.clip_phi_grad > 0:
                    torch.nn.utils.clip_grad_norm_(control.parameters(), cfg.clip_phi_grad)
                optimizer_phi.step()
                last_inner_obj = scalarize(inner_obj)
                last_energy = scalarize(roll.energy)

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
            last_energy = scalarize(roll.energy)

        history["outer_loss"].append(scalarize(outer_loss))
        history["outer_loss_attack"].append(scalarize(outer_loss_attack))
        history["outer_loss_clean"].append(scalarize(outer_loss_clean))
        history["inner_obj"].append(last_inner_obj)
        history["energy"].append(last_energy)
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
                )
                train_loss_cur = weighted_denoise_loss(
                    denoiser,
                    roll_cur_diag.x_target,
                    x0,
                    roll_cur_diag.sigma_target,
                    cfg.sigma_data,
                )
                inner_obj_cur = inner_objective_energy_only(train_loss_cur, roll_cur_diag.energy, cfg.lambda_energy)

                roll_zero_diag = rollout_controlled_ve(
                    x0=x0,
                    target_indices=indices,
                    control_net=_zero_control,
                    sigma_levels=sigma_levels,
                    grad_through_control=False,
                )
                train_loss_zero = weighted_denoise_loss(
                    denoiser,
                    roll_zero_diag.x_target,
                    x0,
                    roll_zero_diag.sigma_target,
                    cfg.sigma_data,
                )
                inner_obj_zero = inner_objective_energy_only(train_loss_zero, roll_zero_diag.energy, cfg.lambda_energy)
                gap = inner_obj_cur - inner_obj_zero
                gap_ratio = gap / (inner_obj_zero.abs() + 1e-8)
                path_delta_mean, terminal_delta_mean, v_l2_mean = _rollout_control_diagnostics(
                    roll_cur_diag.states_ref, roll_cur_diag.states_ctrl, sigma_levels
                )

            history["diag_step"].append(int(step))
            history["diag_inner_obj_current"].append(scalarize(inner_obj_cur))
            history["diag_inner_obj_zero"].append(scalarize(inner_obj_zero))
            history["diag_inner_obj_gap"].append(scalarize(gap))
            history["diag_inner_obj_gap_ratio"].append(scalarize(gap_ratio))
            history["diag_v_l2_mean"].append(scalarize(v_l2_mean))
            history["diag_path_delta_mean"].append(scalarize(path_delta_mean))
            history["diag_terminal_delta_mean"].append(scalarize(terminal_delta_mean))

        if step % cfg.log_every == 0:
            diag_msg = ""
            if len(history["diag_step"]) > 0:
                diag_msg = (
                    " "
                    f"diag_gap={history['diag_inner_obj_gap'][-1]:.6f} "
                    f"diag_gap_ratio={history['diag_inner_obj_gap_ratio'][-1]:.6f} "
                    f"diag_v_l2={history['diag_v_l2_mean'][-1]:.6f} "
                    f"diag_path_delta={history['diag_path_delta_mean'][-1]:.6f}"
                )
            print(
                f"[robust] step={step:05d} outer_loss={outer_loss.item():.6f} "
                f"attack={outer_loss_attack.item():.6f} clean={outer_loss_clean.item():.6f} "
                f"inner_obj={last_inner_obj:.6f} energy={last_energy:.6f} "
                f"w_attack={attack_weight:.3f} w_clean={clean_weight:.3f} phi_lr_scale={phi_lr_scale:.3f}"
                f"{diag_msg}",
                flush=True,
            )

    return history


@torch.no_grad()
def sample_with_denoiser(denoiser, sigma_levels, n_samples, device):
    states = sample_reverse_paths(denoiser, sigma_levels, n_samples, device, stochastic=True)
    return states[:, 0]


@torch.no_grad()
def sample_reverse_paths(denoiser, sigma_levels, n_samples, device, stochastic: bool = True, noise_schedule: torch.Tensor = None):
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
    # Returns states indexed by sigma index k (0..N), where k=N is noisy terminal and k=0 is recovered clean.
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
