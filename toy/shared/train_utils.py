from typing import Callable, Optional

import torch

from ..data import sample_from_pool, sample_gmm


def sample_train_batch(
    cfg,
    centers=None,
    train_pool: torch.Tensor = None,
    sample_train_batch_fn: Optional[Callable[[int], torch.Tensor]] = None,
    sample_population_batch_fn: Optional[Callable[[int], torch.Tensor]] = None,
) -> torch.Tensor:
    """Uniform train-batch loader abstraction."""

    if sample_train_batch_fn is not None:
        return sample_train_batch_fn(cfg.batch_size)
    if train_pool is None:
        if sample_population_batch_fn is not None:
            return sample_population_batch_fn(cfg.batch_size)
        if centers is None:
            raise ValueError("centers must be provided when no dataset-specific batch sampler is available.")
        return sample_gmm(cfg.batch_size, centers, cfg.data_std)
    return sample_from_pool(train_pool, cfg.batch_size)


def robust_schedule(step: int, cfg):
    """Return current outer weights and inner-LR scale for warmup/ramp."""

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


def pathwise_l2(path_tensor: torch.Tensor) -> torch.Tensor:
    """L2 norm over feature axes for tensors shaped [B, T, ...]."""

    flat = path_tensor.reshape(path_tensor.shape[0], path_tensor.shape[1], -1)
    return torch.sqrt(flat.pow(2).sum(dim=2))


def rollout_control_diagnostics(roll, sigma_levels: torch.Tensor, kappa_by_step: torch.Tensor):
    """Summarize path/control magnitudes and constraint saturation statistics."""

    diff = roll.states_ctrl - roll.states_ref
    diff_l2 = pathwise_l2(diff)
    path_delta_mean = diff_l2.mean()
    terminal_delta_mean = diff_l2[:, -1].mean()

    delta_l2 = pathwise_l2(roll.delta_path)
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


def zero_control(x_ref: torch.Tensor, gap: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Reference control used for diagnostics: delta_k = 0 for all steps."""

    del gap, sigma
    return torch.zeros_like(x_ref)

