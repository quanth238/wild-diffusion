from typing import Callable, Dict, Optional, Tuple

import torch

from ...compute_accounting import append_denoiser_op_count_step, ensure_denoiser_op_count_history
from ...models import set_requires_grad
from ...shared.objective import compute_training_loss, inner_objective_attack_only
from ...shared.runtime import autocast_context, resolve_amp_dtype
from ...shared.sigma import sample_target_indices, sample_target_indices_log_normal
from ...shared.train_utils import robust_schedule, sample_train_batch
from ...utils import batch_scalar_like, has_nan_or_inf, scalarize


def _sample_sigma_indices(batch_size: int, sigma_levels: torch.Tensor, cfg) -> torch.Tensor:
    """Sample target sigma indices using configured uniform/log-normal policy."""

    if cfg.use_log_normal_sigma_sampling:
        return sample_target_indices_log_normal(
            batch_size,
            sigma_levels,
            p_mean=cfg.p_mean,
            p_std=cfg.p_std,
        )
    return sample_target_indices(batch_size, sigma_levels)


def _transport_cost(x_adv: torch.Tensor, x_clean: torch.Tensor) -> torch.Tensor:
    """Quadratic transport penalty c(x',x)=0.5||x'-x||^2 averaged over batch."""

    return 0.5 * (x_adv - x_clean).reshape(x_clean.shape[0], -1).pow(2).sum(dim=1).mean()


def _sample_clean_batch(
    cfg,
    centers,
    train_pool: torch.Tensor,
    sample_train_batch_fn: Optional[Callable[[int], torch.Tensor]],
    sample_population_batch_fn: Optional[Callable[[int], torch.Tensor]],
) -> torch.Tensor:
    """Sample one clean mini-batch using shared backend abstraction."""

    return sample_train_batch(
        cfg,
        centers,
        train_pool=train_pool,
        sample_train_batch_fn=sample_train_batch_fn,
        sample_population_batch_fn=sample_population_batch_fn,
    )


def _build_wild_adversarial_batch(
    denoiser,
    x_clean: torch.Tensor,
    sigma_levels: torch.Tensor,
    cfg,
    amp_dtype,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Approximate argmax_x' loss(x') - gamma*c(x',x) via K gradient-ascent steps."""

    inner_steps = max(int(cfg.wild_inner_steps), 0)
    if inner_steps == 0:
        zeros = {
            "inner_obj": 0.0,
            "attack_loss": 0.0,
            "transport_cost": 0.0,
            "sigma_mean": 0.0,
        }
        return x_clean.detach().clone(), zeros

    gamma = float(cfg.wild_gamma)
    step_size = float(cfg.wild_step_size)

    sigma_idx = _sample_sigma_indices(x_clean.shape[0], sigma_levels, cfg)
    sigma = sigma_levels[sigma_idx]
    fixed_eps = torch.randn_like(x_clean) if bool(cfg.wild_fixed_noise_inner) else None

    x_adv = x_clean.detach().clone()
    last_inner_obj = torch.zeros((), device=x_clean.device, dtype=x_clean.dtype)
    last_attack_loss = torch.zeros((), device=x_clean.device, dtype=x_clean.dtype)
    last_transport = torch.zeros((), device=x_clean.device, dtype=x_clean.dtype)

    for _ in range(inner_steps):
        x_adv.requires_grad_(True)
        eps = fixed_eps if fixed_eps is not None else torch.randn_like(x_clean)
        x_noisy = x_adv + batch_scalar_like(sigma, x_adv) * eps

        with autocast_context(x_clean.device, amp_dtype):
            attack_loss = compute_training_loss(cfg, denoiser, x_noisy, x_adv, sigma)
            transport = _transport_cost(x_adv, x_clean)
            inner_obj = inner_objective_attack_only(attack_loss) - gamma * transport
        if has_nan_or_inf(inner_obj):
            raise RuntimeError("NaN/Inf detected in WILD inner objective.")

        grad_x = torch.autograd.grad(inner_obj, x_adv, only_inputs=True)[0]
        with torch.no_grad():
            x_adv = x_adv + step_size * grad_x
            if bool(cfg.wild_clamp_samples):
                x_adv.clamp_(float(cfg.wild_sample_min), float(cfg.wild_sample_max))
        x_adv = x_adv.detach()

        last_inner_obj = inner_obj.detach()
        last_attack_loss = attack_loss.detach()
        last_transport = transport.detach()

    stats = {
        "inner_obj": scalarize(last_inner_obj),
        "attack_loss": scalarize(last_attack_loss),
        "transport_cost": scalarize(last_transport),
        "sigma_mean": float(sigma.detach().mean().item()),
    }
    return x_adv, stats


def train_trajectory_robust_wild(
    denoiser,
    control,
    centers,
    sigma_levels,
    cfg,
    train_pool: torch.Tensor = None,
    sample_train_batch_fn: Optional[Callable[[int], torch.Tensor]] = None,
    sample_population_batch_fn: Optional[Callable[[int], torch.Tensor]] = None,
    start_step: int = 0,
    history_state: Optional[dict] = None,
    optimizer_theta_state: Optional[dict] = None,
    return_state: bool = False,
):
    """WILD-style robust training with interval-refreshed sample-level WDRO surrogates."""

    del control

    optimizer_theta = torch.optim.Adam(denoiser.parameters(), lr=cfg.lr_theta)
    if optimizer_theta_state is not None:
        optimizer_theta.load_state_dict(optimizer_theta_state)
        for state in optimizer_theta.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device=sigma_levels.device)
    history = history_state if history_state is not None else {
        "outer_loss": [],
        "outer_loss_attack": [],
        "outer_loss_clean": [],
        "inner_obj": [],
        "energy": [],
        "lambda_value": [],
        "lambda_value_next": [],
        "lambda_subgrad": [],
        "dual_surrogate": [],
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
        "wild_inner_attack_loss": [],
        "wild_inner_transport_cost": [],
        "wild_inner_sigma_mean": [],
        "wild_refresh_step": [],
        "wild_cache_size": [],
        "batch_equiv_denoiser_evals_step": [],
        "batch_equiv_denoiser_evals_attack_construction": [],
        "batch_equiv_denoiser_evals_attack_eval": [],
        "batch_equiv_denoiser_evals_clean_eval": [],
        "batch_equiv_denoiser_evals_cumulative": [],
    }
    ensure_denoiser_op_count_history(history)
    for key in (
        "outer_loss",
        "outer_loss_attack",
        "outer_loss_clean",
        "inner_obj",
        "energy",
        "lambda_value",
        "lambda_value_next",
        "lambda_subgrad",
        "dual_surrogate",
        "delta_norm_mean",
        "delta_norm_max",
        "delta_norm_ratio_mean",
        "delta_norm_ratio_max",
        "sched_attack_weight",
        "sched_clean_weight",
        "sched_phi_lr_scale",
        "diag_step",
        "diag_inner_obj_current",
        "diag_inner_obj_zero",
        "diag_inner_obj_gap",
        "diag_inner_obj_gap_ratio",
        "diag_delta_norm_mean",
        "diag_delta_norm_max",
        "diag_delta_norm_ratio_mean",
        "diag_delta_norm_ratio_max",
        "diag_path_delta_mean",
        "diag_terminal_delta_mean",
        "wild_inner_attack_loss",
        "wild_inner_transport_cost",
        "wild_inner_sigma_mean",
        "wild_refresh_step",
        "wild_cache_size",
        "batch_equiv_denoiser_evals_step",
        "batch_equiv_denoiser_evals_attack_construction",
        "batch_equiv_denoiser_evals_attack_eval",
        "batch_equiv_denoiser_evals_clean_eval",
        "batch_equiv_denoiser_evals_cumulative",
    ):
        history.setdefault(key, [])

    cache_clean = None
    cache_adv = None
    cache_inner_obj = 0.0
    cache_attack_loss = 0.0
    cache_transport = 0.0
    cache_sigma_mean = 0.0

    refresh_interval = max(int(cfg.wild_update_interval), 1)
    cache_batches = max(int(cfg.wild_cache_batches), 1)
    ratio_denom = max(float(cfg.wild_delta_ratio_denom), 1e-8)
    cumulative_batch_equiv_evals = float(history["batch_equiv_denoiser_evals_cumulative"][-1]) if history["batch_equiv_denoiser_evals_cumulative"] else 0.0
    amp_dtype = resolve_amp_dtype(sigma_levels.device, getattr(cfg, "amp_dtype", "auto"))

    for step in range(int(start_step) + 1, cfg.steps + 1):
        clean_weight, attack_weight, phi_lr_scale, control_updates_enabled = robust_schedule(step, cfg)
        attack_construction_units = 0.0

        need_refresh = cache_clean is None or ((step - 1) % refresh_interval == 0)
        if need_refresh:
            clean_chunks = []
            adv_chunks = []
            inner_stats = []

            set_requires_grad(denoiser, False)
            for _ in range(cache_batches):
                x_clean_chunk = _sample_clean_batch(
                    cfg,
                    centers,
                    train_pool=train_pool,
                    sample_train_batch_fn=sample_train_batch_fn,
                    sample_population_batch_fn=sample_population_batch_fn,
                )
                if control_updates_enabled and attack_weight > 0.0 and int(cfg.wild_inner_steps) > 0:
                    x_adv_chunk, stats = _build_wild_adversarial_batch(
                        denoiser,
                        x_clean_chunk,
                        sigma_levels,
                        cfg,
                        amp_dtype,
                    )
                    attack_construction_units += float(max(int(cfg.wild_inner_steps), 0))
                else:
                    x_adv_chunk = x_clean_chunk.detach().clone()
                    stats = {
                        "inner_obj": 0.0,
                        "attack_loss": 0.0,
                        "transport_cost": 0.0,
                        "sigma_mean": 0.0,
                    }
                clean_chunks.append(x_clean_chunk)
                adv_chunks.append(x_adv_chunk)
                inner_stats.append(stats)
            set_requires_grad(denoiser, True)

            cache_clean = torch.cat(clean_chunks, dim=0)
            cache_adv = torch.cat(adv_chunks, dim=0)
            cache_inner_obj = float(sum(s["inner_obj"] for s in inner_stats) / len(inner_stats))
            cache_attack_loss = float(sum(s["attack_loss"] for s in inner_stats) / len(inner_stats))
            cache_transport = float(sum(s["transport_cost"] for s in inner_stats) / len(inner_stats))
            cache_sigma_mean = float(sum(s["sigma_mean"] for s in inner_stats) / len(inner_stats))

            history["wild_refresh_step"].append(int(step))
            history["wild_cache_size"].append(int(cache_clean.shape[0]))

        select_idx = torch.randint(0, cache_clean.shape[0], (cfg.batch_size,), device=cache_clean.device)
        x_clean = cache_clean[select_idx]
        x_adv = cache_adv[select_idx]

        sigma_idx = _sample_sigma_indices(cfg.batch_size, sigma_levels, cfg)
        sigma = sigma_levels[sigma_idx]

        x_noisy_clean = x_clean + batch_scalar_like(sigma, x_clean) * torch.randn_like(x_clean)
        x_noisy_adv = x_adv + batch_scalar_like(sigma, x_adv) * torch.randn_like(x_adv)

        optimizer_theta.zero_grad(set_to_none=True)
        with autocast_context(sigma_levels.device, amp_dtype):
            outer_loss_clean = compute_training_loss(cfg, denoiser, x_noisy_clean, x_clean, sigma)
            outer_loss_attack = compute_training_loss(cfg, denoiser, x_noisy_adv, x_adv, sigma)
            outer_loss = clean_weight * outer_loss_clean + attack_weight * outer_loss_attack
        if has_nan_or_inf(outer_loss):
            raise RuntimeError("NaN/Inf detected in WILD outer loss.")

        outer_loss.backward()
        optimizer_theta.step()
        attack_eval_units = 1.0
        clean_eval_units = 1.0
        step_batch_equiv_evals = attack_construction_units + attack_eval_units + clean_eval_units
        cumulative_batch_equiv_evals += step_batch_equiv_evals

        delta_l2 = (x_adv - x_clean).reshape(x_clean.shape[0], -1).pow(2).sum(dim=1).sqrt()
        delta_norm_mean = scalarize(delta_l2.mean())
        delta_norm_max = scalarize(delta_l2.max())
        delta_ratio_mean = delta_norm_mean / ratio_denom
        delta_ratio_max = delta_norm_max / ratio_denom

        history["outer_loss"].append(scalarize(outer_loss))
        history["outer_loss_attack"].append(scalarize(outer_loss_attack))
        history["outer_loss_clean"].append(scalarize(outer_loss_clean))
        history["inner_obj"].append(float(cache_inner_obj))
        history["energy"].append(float(cache_transport))
        history["lambda_value"].append(float(cfg.wild_gamma))
        history["lambda_value_next"].append(float(cfg.wild_gamma))
        history["lambda_subgrad"].append(0.0)
        history["dual_surrogate"].append(float(cache_inner_obj))
        history["delta_norm_mean"].append(delta_norm_mean)
        history["delta_norm_max"].append(delta_norm_max)
        history["delta_norm_ratio_mean"].append(float(delta_ratio_mean))
        history["delta_norm_ratio_max"].append(float(delta_ratio_max))
        history["sched_attack_weight"].append(float(attack_weight))
        history["sched_clean_weight"].append(float(clean_weight))
        history["sched_phi_lr_scale"].append(float(phi_lr_scale))
        history["wild_inner_attack_loss"].append(float(cache_attack_loss))
        history["wild_inner_transport_cost"].append(float(cache_transport))
        history["wild_inner_sigma_mean"].append(float(cache_sigma_mean))
        history["batch_equiv_denoiser_evals_step"].append(float(step_batch_equiv_evals))
        history["batch_equiv_denoiser_evals_attack_construction"].append(float(attack_construction_units))
        history["batch_equiv_denoiser_evals_attack_eval"].append(float(attack_eval_units))
        history["batch_equiv_denoiser_evals_clean_eval"].append(float(clean_eval_units))
        history["batch_equiv_denoiser_evals_cumulative"].append(float(cumulative_batch_equiv_evals))
        append_denoiser_op_count_step(
            history,
            n_fwd=0.0,
            n_fwd_inputgrad=float(attack_construction_units),
            n_fwd_parambackward=float(int(attack_weight > 0.0) + int(clean_weight > 0.0)),
        )

        if step % cfg.log_every == 0:
            print(
                f"[wild] step={step:05d} outer_loss={outer_loss.item():.6f} "
                f"attack={outer_loss_attack.item():.6f} clean={outer_loss_clean.item():.6f} "
                f"inner_obj={cache_inner_obj:.6f} transport={cache_transport:.6f} "
                f"delta_norm={delta_norm_mean:.6f} "
                f"be_evals={step_batch_equiv_evals:.1f} be_evals_cum={cumulative_batch_equiv_evals:.1f} "
                f"w_attack={attack_weight:.3f} w_clean={clean_weight:.3f}",
                flush=True,
            )

    if return_state:
        return history, {
            "completed_steps": int(cfg.steps),
            "optimizer_theta_state": optimizer_theta.state_dict(),
        }
    return history
