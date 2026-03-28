from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from toy_2d.cdro_markov import (
    MarkovControlGRU,
    MarkovControlMLP,
    MarkovPrecondScoreMLP,
    MarkovScoreMLP,
    TerminalStats,
    build_markov_schedule,
    build_score_wdro_dataset,
    compute_control_cost,
    compute_score_matching_loss,
    estimate_markov_wdro_proxy_budget,
    rollout_markov_forward,
    sample_reverse_chain,
    set_module_grad,
    update_ema,
    update_terminal_stats,
)
from toy_2d.datasets import build_dataset
from toy_2d.metrics import mmd_rbf, sliced_wasserstein
from toy_2d.plotting import (
    save_adversarial_debug,
    save_markov_score_training_curves,
    save_process_snapshots,
    save_scatter_comparison,
)
from toy_2d.robust_defaults import WDRO_CORE_DEFAULTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Markov score-based CDRO toy model.")
    parser.add_argument("--outdir", type=Path, default=Path("toy-runs") / "cdro_markov")
    parser.add_argument("--method", type=str, default="cdro_markov", choices=("baseline_score", "wdro_score", "cdro_markov"))
    parser.add_argument("--dataset", type=str, default="two_moons")
    parser.add_argument("--num-samples", type=int, default=4096)
    parser.add_argument("--noise", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--torch-num-threads", type=int, default=0)
    parser.add_argument("--standardize", action="store_true", default=True)
    parser.add_argument("--no-standardize", dest="standardize", action="store_false")

    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--score-lr", type=float, default=1e-3)
    parser.add_argument("--control-lr", type=float, default=2e-4)
    parser.add_argument("--lambda-lr", type=float, default=2e-2)
    parser.add_argument("--lambda-init", type=float, default=0.1)
    parser.add_argument("--lambda-min", type=float, default=0.02)
    parser.add_argument("--control-radius", type=float, default=0.05)
    parser.add_argument("--budget-mode", type=str, default="fixed", choices=("fixed", "match_wdro_proxy"))
    parser.add_argument("--reference-wdro-k", type=int, default=WDRO_CORE_DEFAULTS["k"])
    parser.add_argument("--reference-wdro-step-size", type=float, default=WDRO_CORE_DEFAULTS["step_size"])
    parser.add_argument("--reference-wdro-gamma", type=float, default=WDRO_CORE_DEFAULTS["gamma"])
    parser.add_argument("--budget-estimate-batch-size", type=int, default=256)
    parser.add_argument("--budget-scale", type=float, default=1.0)
    parser.add_argument("--budget-ema-decay", type=float, default=0.9)
    parser.add_argument("--budget-max-ratio", type=float, default=4.0)
    parser.add_argument("--budget-schedule", type=str, default="constant", choices=("constant", "frontload"))
    parser.add_argument("--budget-frontload-power", type=float, default=2.0)
    parser.add_argument("--budget-frontload-floor", type=float, default=0.25)
    parser.add_argument("--min-budget-utilization", type=float, default=0.0)
    parser.add_argument("--budget-utilization-patience", type=int, default=0)
    parser.add_argument("--budget-utilization-start-epoch", type=int, default=0)
    parser.add_argument("--max-budget-utilization", type=float, default=0.0)
    parser.add_argument("--high-budget-utilization-patience", type=int, default=0)
    parser.add_argument("--high-budget-utilization-start-epoch", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--warmup-epochs", type=int, default=8)
    parser.add_argument("--adversary-steps", type=int, default=1)
    parser.add_argument("--adversary-stop-epoch", type=int, default=0)
    parser.add_argument(
        "--disable-control-after-stop",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--score-steps", type=int, default=8)
    parser.add_argument("--terminal-momentum", type=float, default=0.95)
    parser.add_argument("--terminal-sampler", type=str, default="gaussian", choices=("gaussian", "replay"))
    parser.add_argument("--terminal-buffer-size", type=int, default=4096)
    parser.add_argument("--terminal-jitter-scale", type=float, default=0.0)
    parser.add_argument("--reverse-noise-scale", type=float, default=1.0)
    parser.add_argument("--reverse-tail-noise-scale", type=float, default=1.0)
    parser.add_argument("--reverse-deterministic-tail-steps", type=int, default=0)

    parser.add_argument("--num-steps", type=int, default=12)
    parser.add_argument("--total-time", type=float, default=1.0)
    parser.add_argument(
        "--sde-family",
        type=str,
        default="vp_linear",
        choices=("vp_linear", "vp_cosine", "ve_geometric"),
    )
    parser.add_argument("--beta-min", type=float, default=0.2)
    parser.add_argument("--beta-max", type=float, default=6.0)
    parser.add_argument("--ve-sigma-min", type=float, default=0.01)
    parser.add_argument("--ve-sigma-max", type=float, default=3.0)
    parser.add_argument("--cosine-s", type=float, default=0.008)
    parser.add_argument(
        "--score-weight-schedule",
        type=str,
        default="uniform",
        choices=("uniform", "sigma_sq", "inv_sigma_sq"),
    )

    parser.add_argument("--score-hidden-dim", type=int, default=128)
    parser.add_argument("--score-depth", type=int, default=4)
    parser.add_argument("--score-arch", type=str, default="precond", choices=("raw", "precond"))
    parser.add_argument("--control-hidden-dim", type=int, default=64)
    parser.add_argument("--control-depth", type=int, default=3)
    parser.add_argument("--control-arch", type=str, default="mlp", choices=("mlp", "gru"))
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--control-scale", type=float, default=0.5)
    parser.add_argument("--sigma-data", type=float, default=0.5)
    parser.add_argument("--reverse-solver", type=str, default="heun", choices=("euler", "heun"))
    parser.add_argument("--reverse-control-scale", type=float, default=1.0)
    parser.add_argument("--wdro-k", type=int, default=WDRO_CORE_DEFAULTS["k"])
    parser.add_argument("--wdro-step-size", type=float, default=WDRO_CORE_DEFAULTS["step_size"])
    parser.add_argument("--wdro-gamma", type=float, default=WDRO_CORE_DEFAULTS["gamma"])
    parser.add_argument("--wdro-p-adv", type=float, default=WDRO_CORE_DEFAULTS["p_adv"])
    parser.add_argument("--wdro-warmup-epochs", type=int, default=25)
    parser.add_argument("--wdro-refresh-every", type=int, default=10)
    parser.add_argument("--wdro-debug-points", type=int, default=256)

    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--num-eval-samples", type=int, default=2048)
    parser.add_argument("--metric-samples", type=int, default=1024)
    parser.add_argument("--num-snapshot-steps", type=int, default=6)
    parser.add_argument("--diagnostic-batch-size", type=int, default=256)
    parser.add_argument(
        "--log-reverse-ablation",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--save-eval-checkpoints", action="store_true")
    parser.add_argument("--fast-tuning", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.torch_num_threads > 0:
        torch.set_num_threads(int(args.torch_num_threads))
        try:
            torch.set_num_interop_threads(max(1, min(4, int(args.torch_num_threads))))
        except RuntimeError:
            pass
    device = resolve_device(args.device)
    set_seed(args.seed)

    args.outdir.mkdir(parents=True, exist_ok=True)
    save_json(args.outdir / "config.json", vars(args))

    dataset = build_dataset(
        name=args.dataset,
        num_samples=args.num_samples,
        seed=args.seed,
        noise=args.noise,
        standardize=args.standardize,
    )
    base_points = dataset.train_points.clone()
    current_points = base_points.clone()
    rng = np.random.default_rng(args.seed + 1)

    schedule = build_markov_schedule(
        num_steps=args.num_steps,
        total_time=args.total_time,
        beta_min=args.beta_min,
        beta_max=args.beta_max,
        sde_family=args.sde_family,
        device=device,
        dtype=dataset.train_points.dtype,
        weight_schedule=args.score_weight_schedule,
        ve_sigma_min=args.ve_sigma_min,
        ve_sigma_max=args.ve_sigma_max,
        cosine_s=args.cosine_s,
    )
    diagnostic_batch_size = min(max(int(args.diagnostic_batch_size), 1), int(dataset.train_points.shape[0]))
    diagnostic_points = dataset.train_points[:diagnostic_batch_size].to(device)
    diagnostic_noise = torch.randn(
        diagnostic_batch_size,
        args.num_steps,
        dataset.train_points.shape[1],
        generator=torch.Generator().manual_seed(args.seed + 12345),
        dtype=dataset.train_points.dtype,
    ).to(device)

    cdro_has_adversary = args.method == "cdro_markov" and args.adversary_steps > 0 and args.control_scale > 0.0
    score_model = build_score_model(args=args, device=device)
    score_ema = copy.deepcopy(score_model).eval()

    control_model = build_control_model(args=args, device=device) if cdro_has_adversary else None
    if control_model is not None:
        initialize_control_head(control_model)
    control_ema = copy.deepcopy(control_model).eval() if control_model is not None else None

    score_optimizer = torch.optim.Adam(score_model.parameters(), lr=args.score_lr)
    control_optimizer = (
        torch.optim.Adam(control_model.parameters(), lr=args.control_lr) if control_model is not None else None
    )
    lambda_value = float(args.lambda_init) if control_model is not None else 0.0
    terminal_stats: TerminalStats | None = None

    history: list[dict] = []
    eval_history: list[dict] = []
    best_swd = float("inf")
    best_epoch: int | None = None
    best_eval: dict | None = None
    budget_state: dict[str, float | None] = {"raw": None, "smoothed": None}
    latest_adv_stats: dict | None = None
    latest_adv_snapshot: tuple[np.ndarray, np.ndarray] | None = None
    latest_adv_plot_stats: tuple[float, float] | None = None
    low_budget_utilization_streak = 0
    high_budget_utilization_streak = 0
    adversary_disabled_epoch: int | None = None

    start_time = time.time()
    active_epochs = max(args.epochs - args.warmup_epochs, 1)
    for epoch_idx in range(args.epochs):
        epoch = epoch_idx + 1
        if args.method == "wdro_score" and should_refresh(
            epoch_idx=epoch_idx,
            warmup=args.wdro_warmup_epochs,
            refresh_every=args.wdro_refresh_every,
        ):
            refresh = build_score_wdro_dataset(
                base_points=base_points,
                score_net=score_ema,
                schedule=schedule,
                batch_size=args.batch_size,
                gamma=args.wdro_gamma,
                step_size=args.wdro_step_size,
                iters=args.wdro_k,
                p_adv=args.wdro_p_adv,
                clamp_min=dataset.bounds_min.to(device),
                clamp_max=dataset.bounds_max.to(device),
                device=device,
                rng=rng,
                debug_adv_points=args.wdro_debug_points,
            )
            current_points = refresh.combined_points
            latest_adv_stats = {
                "mean_l2_shift": refresh.mean_l2_shift,
                "max_l2_shift": refresh.max_l2_shift,
                "mean_transport_cost": refresh.mean_transport_cost,
                "max_transport_cost": refresh.max_transport_cost,
                "mean_total_transport_cost": refresh.mean_transport_cost,
                "max_total_transport_cost": refresh.max_transport_cost,
            }
            if refresh.adv_original is not None and refresh.adv_generated is not None:
                orig_np = dataset.destandardize(refresh.adv_original).cpu().numpy()
                adv_np = dataset.destandardize(refresh.adv_generated).cpu().numpy()
                latest_adv_snapshot = (orig_np, adv_np)
                shifts = np.linalg.norm(adv_np - orig_np, axis=1)
                latest_adv_plot_stats = (float(shifts.mean()), float(shifts.max()))
        elif args.method != "wdro_score":
            current_points = base_points
        loader = build_loader(
            points=current_points,
            batch_size=args.batch_size,
            seed=args.seed + epoch_idx,
            pin_memory=device.type == "cuda",
        )
        control_present = (
            cdro_has_adversary
            and epoch_idx >= args.warmup_epochs
            and args.adversary_steps > 0
            and adversary_disabled_epoch is None
        )
        if (
            control_present
            and args.disable_control_after_stop
            and args.adversary_stop_epoch > 0
            and epoch > args.adversary_stop_epoch
        ):
            control_present = False
        adversary_update_active = control_present and (
            args.adversary_stop_epoch <= 0 or (epoch_idx + 1) <= args.adversary_stop_epoch
        )
        raw_budget_estimate, base_total_budget, target_total_budget = resolve_target_total_budget(
            args=args,
            epoch=epoch,
            total_epochs=args.epochs,
            active_epochs=active_epochs,
            score_model=score_ema,
            schedule=schedule,
            dataset=dataset,
            device=device,
            budget_state=budget_state,
        )

        epoch_score_losses: list[float] = []
        epoch_control_costs: list[float] = []
        epoch_adv_values: list[float] = []
        epoch_control_mean_norms: list[float] = []
        epoch_control_max_norms: list[float] = []
        epoch_score_grad_norms: list[float] = []
        epoch_control_grad_norms: list[float] = []

        for (batch,) in loader:
            batch = batch.to(device, non_blocking=device.type == "cuda")

            if adversary_update_active:
                if control_model is None or control_optimizer is None:
                    raise RuntimeError("CDRO Markov adversary updates require a control model and optimizer.")
                set_module_grad(score_model, False)
                set_module_grad(control_model, True)
                score_model.eval()
                control_model.train()
                for _ in range(args.adversary_steps):
                    control_optimizer.zero_grad(set_to_none=True)
                    rollout = rollout_markov_forward(
                        clean_points=batch,
                        control_net=control_model,
                        schedule=schedule,
                        zero_control=False,
                    )
                    score_loss = compute_score_matching_loss(score_net=score_model, rollout=rollout, schedule=schedule)
                    control_cost = compute_control_cost(rollout=rollout, schedule=schedule)
                    objective = score_loss - lambda_value * control_cost
                    (-objective).backward()
                    epoch_control_grad_norms.append(float(compute_grad_l2_norm(control_model.parameters())))
                    clip_gradients(control_model.parameters(), grad_clip=args.grad_clip)
                    control_optimizer.step()
                    lambda_value = max(
                        float(args.lambda_min),
                        lambda_value - args.lambda_lr * (float(target_total_budget) - float(control_cost.detach().item())),
                    )
                    epoch_control_costs.append(float(control_cost.detach().item()))
                    epoch_adv_values.append(float(objective.detach().item()))
                    control_norms = rollout.controls.detach().norm(dim=2)
                    epoch_control_mean_norms.append(float(control_norms.mean().item()))
                    epoch_control_max_norms.append(float(control_norms.max().item()))
                if control_ema is not None:
                    update_ema(ema=control_ema, model=control_model, decay=args.ema_decay)
                set_module_grad(score_model, True)

            if control_model is not None:
                set_module_grad(control_model, False)
            set_module_grad(score_model, True)
            score_model.train()
            for _ in range(args.score_steps):
                score_optimizer.zero_grad(set_to_none=True)
                rollout = rollout_markov_forward(
                    clean_points=batch,
                    control_net=control_model if control_present else None,
                    schedule=schedule,
                    zero_control=not control_present,
                )
                score_loss = compute_score_matching_loss(score_net=score_model, rollout=rollout, schedule=schedule)
                score_loss.backward()
                epoch_score_grad_norms.append(float(compute_grad_l2_norm(score_model.parameters())))
                clip_gradients(score_model.parameters(), grad_clip=args.grad_clip)
                score_optimizer.step()
                update_ema(ema=score_ema, model=score_model, decay=args.ema_decay)
                epoch_score_losses.append(float(score_loss.detach().item()))

            with torch.no_grad():
                terminal_rollout = rollout_markov_forward(
                    clean_points=batch,
                    control_net=control_ema if control_present else None,
                    schedule=schedule,
                    zero_control=not control_present,
                )
                terminal_stats = update_terminal_stats(
                    terminal_states=terminal_rollout.states[:, -1, :],
                    current=terminal_stats,
                    momentum=args.terminal_momentum,
                    replay_max_size=args.terminal_buffer_size,
                )

            if control_model is not None:
                set_module_grad(control_model, True)

        epoch_metrics = {
            "epoch": epoch_idx + 1,
            "score_loss": float(np.mean(epoch_score_losses)) if epoch_score_losses else float("nan"),
            "score_loss_std": float(np.std(epoch_score_losses)) if epoch_score_losses else 0.0,
            "control_cost": float(np.mean(epoch_control_costs)) if epoch_control_costs else 0.0,
            "control_cost_std": float(np.std(epoch_control_costs)) if epoch_control_costs else 0.0,
            "adversary_value": float(np.mean(epoch_adv_values)) if epoch_adv_values else 0.0,
            "adversary_value_std": float(np.std(epoch_adv_values)) if epoch_adv_values else 0.0,
            "mean_control_norm": float(np.mean(epoch_control_mean_norms)) if epoch_control_mean_norms else 0.0,
            "max_control_norm": float(np.max(epoch_control_max_norms)) if epoch_control_max_norms else 0.0,
            "mean_score_grad_norm": float(np.mean(epoch_score_grad_norms)) if epoch_score_grad_norms else 0.0,
            "max_score_grad_norm": float(np.max(epoch_score_grad_norms)) if epoch_score_grad_norms else 0.0,
            "mean_control_grad_norm": float(np.mean(epoch_control_grad_norms)) if epoch_control_grad_norms else 0.0,
            "max_control_grad_norm": float(np.max(epoch_control_grad_norms)) if epoch_control_grad_norms else 0.0,
            "target_total_budget": float(target_total_budget),
            "base_total_budget": float(base_total_budget),
            "raw_budget_estimate": (None if raw_budget_estimate is None else float(raw_budget_estimate)),
            "lambda_value": float(lambda_value),
            "control_active": bool(control_present),
            "adversary_update_active": bool(adversary_update_active),
            "dataset_size": int(current_points.shape[0]),
            "method": args.method,
            "low_budget_utilization_streak": int(low_budget_utilization_streak),
            "high_budget_utilization_streak": int(high_budget_utilization_streak),
            "adversary_disabled_epoch": adversary_disabled_epoch,
        }
        if latest_adv_stats is not None:
            epoch_metrics["mean_l2_shift"] = latest_adv_stats["mean_l2_shift"]
            epoch_metrics["max_l2_shift"] = latest_adv_stats["max_l2_shift"]
            epoch_metrics["mean_transport_cost"] = latest_adv_stats["mean_transport_cost"]
            epoch_metrics["max_transport_cost"] = latest_adv_stats["max_transport_cost"]
            epoch_metrics["total_transport_cost"] = latest_adv_stats["mean_total_transport_cost"]
            epoch_metrics["max_total_transport_cost"] = latest_adv_stats["max_total_transport_cost"]
        epoch_metrics["budget_utilization"] = (
            float(epoch_metrics["control_cost"] / target_total_budget) if target_total_budget > 0 else 0.0
        )
        if (
            args.method == "cdro_markov"
            and adversary_disabled_epoch is None
            and adversary_update_active
        ):
            utilization = float(epoch_metrics["budget_utilization"])
            low_trigger_enabled = args.min_budget_utilization > 0.0 and args.budget_utilization_patience > 0
            high_trigger_enabled = args.max_budget_utilization > 0.0 and args.high_budget_utilization_patience > 0

            if low_trigger_enabled:
                low_start_epoch = max(args.budget_utilization_start_epoch, args.warmup_epochs + 1)
                if epoch >= low_start_epoch and utilization < float(args.min_budget_utilization):
                    low_budget_utilization_streak += 1
                else:
                    low_budget_utilization_streak = 0
            else:
                low_budget_utilization_streak = 0

            if high_trigger_enabled:
                high_start_epoch = max(args.high_budget_utilization_start_epoch, args.warmup_epochs + 1)
                if epoch >= high_start_epoch and utilization > float(args.max_budget_utilization):
                    high_budget_utilization_streak += 1
                else:
                    high_budget_utilization_streak = 0
            else:
                high_budget_utilization_streak = 0

            low_triggered = low_trigger_enabled and (
                low_budget_utilization_streak >= int(args.budget_utilization_patience)
            )
            high_triggered = high_trigger_enabled and (
                high_budget_utilization_streak >= int(args.high_budget_utilization_patience)
            )
            if low_triggered or high_triggered:
                adversary_disabled_epoch = epoch
        else:
            low_budget_utilization_streak = 0
            high_budget_utilization_streak = 0
        epoch_metrics["low_budget_utilization_streak"] = int(low_budget_utilization_streak)
        epoch_metrics["high_budget_utilization_streak"] = int(high_budget_utilization_streak)
        epoch_metrics["adversary_disabled_epoch"] = adversary_disabled_epoch
        epoch_metrics["adversary_disabled_due_to_low_utilization"] = (
            adversary_disabled_epoch is not None
            and epoch >= adversary_disabled_epoch
            and low_budget_utilization_streak >= int(args.budget_utilization_patience)
        )
        epoch_metrics["adversary_disabled_due_to_high_utilization"] = (
            adversary_disabled_epoch is not None
            and epoch >= adversary_disabled_epoch
            and high_budget_utilization_streak >= int(args.high_budget_utilization_patience)
        )
        if terminal_stats is not None:
            epoch_metrics["terminal_var_mean"] = float(terminal_stats.var.mean().item())
            epoch_metrics["terminal_var_min"] = float(terminal_stats.var.min().item())
            epoch_metrics["terminal_replay_size"] = int(terminal_stats.replay.shape[0]) if terminal_stats.replay is not None else 0
        epoch_metrics["score_param_norm"] = float(compute_param_l2_norm(score_model.parameters()))
        epoch_metrics["score_ema_param_norm"] = float(compute_param_l2_norm(score_ema.parameters()))
        epoch_metrics["control_param_norm"] = (
            float(compute_param_l2_norm(control_model.parameters())) if control_model is not None else 0.0
        )
        epoch_metrics["control_ema_param_norm"] = (
            float(compute_param_l2_norm(control_ema.parameters()))
            if control_ema is not None
            else 0.0
        )
        history.append(epoch_metrics)

        if epoch_metrics["epoch"] == 1 or epoch_metrics["epoch"] % args.eval_every == 0 or epoch_metrics["epoch"] == args.epochs:
            if terminal_stats is None:
                raise RuntimeError("terminal_stats must be initialized before evaluation.")
            eval_metrics = evaluate_model(
                dataset=dataset,
                score_model=score_ema,
                control_model=control_ema if control_present else None,
                terminal_stats=terminal_stats,
                schedule=schedule,
                outdir=args.outdir,
                epoch=epoch_metrics["epoch"],
                device=device,
                num_eval_samples=args.num_eval_samples,
                metric_samples=args.metric_samples,
                seed=args.seed,
                zero_control=not control_present,
                num_snapshot_steps=args.num_snapshot_steps,
                method_name=args.method,
                reverse_solver=args.reverse_solver,
                terminal_sampler=args.terminal_sampler,
                terminal_jitter_scale=args.terminal_jitter_scale,
                reverse_noise_scale=args.reverse_noise_scale,
                reverse_control_scale=args.reverse_control_scale,
                reverse_tail_noise_scale=args.reverse_tail_noise_scale,
                reverse_deterministic_tail_steps=args.reverse_deterministic_tail_steps,
                diagnostic_points=diagnostic_points,
                diagnostic_noise=diagnostic_noise,
                log_reverse_ablation=args.log_reverse_ablation,
                fast_tuning=args.fast_tuning,
            )
            eval_metrics.update(epoch_metrics)
            history[-1].update(
                {
                    f"eval_{key}": value
                    for key, value in eval_metrics.items()
                    if key != "epoch" and isinstance(value, (int, float)) and not isinstance(value, bool)
                }
            )
            eval_history.append(eval_metrics)
            append_jsonl(args.outdir / "metrics.jsonl", eval_metrics)
            if not args.fast_tuning:
                save_markov_score_training_curves(
                    path=args.outdir / "plots" / "training_curves.png",
                    history=history,
                )

            if args.save_eval_checkpoints and not args.fast_tuning:
                save_checkpoint(
                    path=args.outdir / "checkpoints" / f"checkpoint_epoch_{epoch_metrics['epoch']:04d}.pt",
                    score_model=score_ema,
                    control_model=control_ema,
                    schedule=schedule,
                    terminal_stats=terminal_stats,
                    args=args,
                    dataset=dataset,
                    epoch=epoch_metrics["epoch"],
                    metrics=eval_metrics,
                )
            if latest_adv_snapshot is not None and latest_adv_plot_stats is not None and not args.fast_tuning:
                save_adversarial_debug(
                    path=args.outdir / "plots" / f"adv_epoch_{epoch_metrics['epoch']:04d}.png",
                    original_points=latest_adv_snapshot[0],
                    adversarial_points=latest_adv_snapshot[1],
                    title=f"{args.method} refresh at epoch {epoch_metrics['epoch']}",
                    mean_l2_shift=latest_adv_plot_stats[0],
                    max_l2_shift=latest_adv_plot_stats[1],
                )

            if eval_metrics["sliced_wasserstein"] < best_swd:
                best_swd = eval_metrics["sliced_wasserstein"]
                best_epoch = epoch_metrics["epoch"]
                best_eval = dict(eval_metrics)
                if not args.fast_tuning:
                    save_checkpoint(
                        path=args.outdir / "checkpoint_best.pt",
                        score_model=score_ema,
                        control_model=control_ema,
                        schedule=schedule,
                        terminal_stats=terminal_stats,
                        args=args,
                        dataset=dataset,
                        epoch=epoch_metrics["epoch"],
                        metrics=eval_metrics,
                    )

    active_history = [row for row in history if row.get("adversary_update_active", False)]
    nontrivial_active_history = [
        row
        for row in active_history
        if float(row.get("mean_control_norm", 0.0)) > 1e-6
        or float(row.get("control_cost", 0.0)) > 1e-10
    ]
    active_eval_history = [row for row in eval_history if row.get("adversary_update_active", False)]

    total_minutes = (time.time() - start_time) / 60.0
    final_summary = {
        "dataset": args.dataset,
        "epochs": args.epochs,
        "runtime_minutes": total_minutes,
        "final_loss": history[-1]["score_loss"] if history else None,
        "best_sliced_wasserstein": best_swd,
        "best_epoch": best_epoch,
        "best_eval": best_eval,
        "last_eval": eval_history[-1] if eval_history else None,
        "adversary_update_epochs": len(active_history),
        "nontrivial_adversary_epochs": len(nontrivial_active_history),
        "adversary_update_epoch_fraction": (
            float(len(active_history) / len(history)) if history else 0.0
        ),
        "nontrivial_adversary_epoch_fraction": (
            float(len(nontrivial_active_history) / len(active_history)) if active_history else 0.0
        ),
        "active_budget_utilization_mean": (
            float(np.mean([row.get("budget_utilization", 0.0) for row in active_history]))
            if active_history
            else 0.0
        ),
        "active_budget_utilization_max": (
            float(np.max([row.get("budget_utilization", 0.0) for row in active_history]))
            if active_history
            else 0.0
        ),
        "active_control_cost_mean": (
            float(np.mean([row.get("control_cost", 0.0) for row in active_history]))
            if active_history
            else 0.0
        ),
        "active_control_norm_mean": (
            float(np.mean([row.get("mean_control_norm", 0.0) for row in active_history]))
            if active_history
            else 0.0
        ),
        "active_control_norm_max": (
            float(np.max([row.get("max_control_norm", 0.0) for row in active_history]))
            if active_history
            else 0.0
        ),
        "active_diag_score_loss_gap_mean": (
            float(np.mean([row.get("diag_score_loss_gap", 0.0) for row in active_eval_history]))
            if active_eval_history
            else 0.0
        ),
    }
    save_json(args.outdir / "summary.json", final_summary)
    if terminal_stats is None:
        raise RuntimeError("terminal_stats must be initialized before saving the final checkpoint.")
    if not args.fast_tuning:
        save_checkpoint(
            path=args.outdir / "checkpoint_last.pt",
            score_model=score_ema,
            control_model=control_ema,
            schedule=schedule,
            terminal_stats=terminal_stats,
            args=args,
            dataset=dataset,
            epoch=args.epochs,
            metrics=eval_history[-1] if eval_history else None,
        )
    print(json.dumps(final_summary, indent=2))


@torch.no_grad()
def evaluate_model(
    *,
    dataset,
    score_model,
    control_model,
    terminal_stats: TerminalStats,
    schedule,
    outdir: Path,
    epoch: int,
    device: torch.device,
    num_eval_samples: int,
    metric_samples: int,
    seed: int,
    zero_control: bool,
    num_snapshot_steps: int,
    method_name: str,
    reverse_solver: str,
    terminal_sampler: str,
    terminal_jitter_scale: float,
    reverse_noise_scale: float,
    reverse_control_scale: float,
    reverse_tail_noise_scale: float,
    reverse_deterministic_tail_steps: int,
    diagnostic_points: torch.Tensor,
    diagnostic_noise: torch.Tensor,
    log_reverse_ablation: bool,
    fast_tuning: bool = False,
) -> dict:
    score_model.eval()
    if control_model is not None:
        control_model.eval()

    shared_reverse_randomness = build_reverse_sampling_randomness(
        schedule=schedule,
        terminal_stats=terminal_stats,
        num_samples=num_eval_samples,
        data_dim=2,
        terminal_sampler=terminal_sampler,
        terminal_jitter_scale=terminal_jitter_scale,
        seed=seed + epoch + 100_000,
        device=device,
        dtype=torch.float32,
    )
    generated_std, reverse_states = sample_reverse_chain(
        score_net=score_model,
        control_net=control_model,
        schedule=schedule,
        terminal_stats=terminal_stats,
        num_samples=num_eval_samples,
        data_dim=2,
        device=device,
        zero_control=zero_control,
        collect_states=not fast_tuning,
        solver=reverse_solver,
        terminal_sampler=terminal_sampler,
        terminal_jitter_scale=terminal_jitter_scale,
        reverse_noise_scale=reverse_noise_scale,
        reverse_control_scale=reverse_control_scale,
        reverse_tail_noise_scale=reverse_tail_noise_scale,
        reverse_deterministic_tail_steps=reverse_deterministic_tail_steps,
        terminal_replay_indices=shared_reverse_randomness["terminal_replay_indices"],
        terminal_draw_noise=shared_reverse_randomness["terminal_draw_noise"],
        step_noises=shared_reverse_randomness["step_noises"],
    )
    generated = dataset.destandardize(generated_std.cpu()).cpu()
    real_all = torch.from_numpy(dataset.raw_points)

    metric_count = min(metric_samples, num_eval_samples, real_all.shape[0])
    generator = torch.Generator()
    generator.manual_seed(seed + epoch)
    real_idx = torch.randperm(real_all.shape[0], generator=generator)[:metric_count]
    fake_idx = torch.randperm(generated.shape[0], generator=generator)[:metric_count]
    real_metric = real_all[real_idx]
    fake_metric = generated[fake_idx]

    metrics = {
        "epoch": epoch,
        "mmd_rbf": mmd_rbf(real_metric, fake_metric),
        "sliced_wasserstein": sliced_wasserstein(real_metric, fake_metric, seed=seed + epoch),
        "terminal_sampler": terminal_sampler,
        "terminal_jitter_scale": float(terminal_jitter_scale),
        "reverse_noise_scale": float(reverse_noise_scale),
        "reverse_control_scale": float(reverse_control_scale),
        "reverse_tail_noise_scale": float(reverse_tail_noise_scale),
        "reverse_deterministic_tail_steps": int(reverse_deterministic_tail_steps),
        "reverse_ablation_shared_randomness": bool(log_reverse_ablation and control_model is not None and not zero_control),
    }

    generated_stats = compute_point_cloud_stats(fake_metric)
    real_stats = compute_point_cloud_stats(real_metric)
    metrics.update(
        {
            "generated_mean_norm": generated_stats["mean_norm"],
            "generated_std_mean": generated_stats["std_mean"],
            "generated_cov_trace": generated_stats["cov_trace"],
            "generated_cov_logdet": generated_stats["cov_logdet"],
            "real_mean_norm": real_stats["mean_norm"],
            "real_std_mean": real_stats["std_mean"],
            "real_cov_trace": real_stats["cov_trace"],
            "real_cov_logdet": real_stats["cov_logdet"],
            "sample_mean_gap": float(torch.linalg.norm(fake_metric.mean(dim=0) - real_metric.mean(dim=0)).item()),
            "sample_std_gap": float(
                torch.linalg.norm(
                    fake_metric.std(dim=0, unbiased=False) - real_metric.std(dim=0, unbiased=False)
                ).item()
            ),
        }
    )

    nominal_rollout = rollout_markov_forward(
        clean_points=diagnostic_points,
        control_net=None,
        schedule=schedule,
        zero_control=True,
        noise_override=diagnostic_noise,
    )
    nominal_diag_score_loss = float(
        compute_score_matching_loss(score_net=score_model, rollout=nominal_rollout, schedule=schedule).item()
    )
    nominal_diag_control_cost = float(compute_control_cost(rollout=nominal_rollout, schedule=schedule).item())
    metrics["diag_nominal_score_loss"] = nominal_diag_score_loss
    metrics["diag_nominal_control_cost"] = nominal_diag_control_cost

    if control_model is not None and not zero_control:
        controlled_rollout = rollout_markov_forward(
            clean_points=diagnostic_points,
            control_net=control_model,
            schedule=schedule,
            zero_control=False,
            noise_override=diagnostic_noise,
        )
        controlled_diag_score_loss = float(
            compute_score_matching_loss(score_net=score_model, rollout=controlled_rollout, schedule=schedule).item()
        )
        controlled_diag_control_cost = float(
            compute_control_cost(rollout=controlled_rollout, schedule=schedule).item()
        )
        controlled_norms = controlled_rollout.controls.norm(dim=2)
        reference_drift = (
            schedule.drift_coeff.view(1, -1, 1) * controlled_rollout.states[:, :-1, :]
        )
        reference_drift_norms = reference_drift.norm(dim=2)
        metrics["diag_controlled_score_loss"] = controlled_diag_score_loss
        metrics["diag_controlled_control_cost"] = controlled_diag_control_cost
        metrics["diag_score_loss_gap"] = controlled_diag_score_loss - nominal_diag_score_loss
        metrics["diag_control_cost_gap"] = controlled_diag_control_cost - nominal_diag_control_cost
        metrics["diag_control_norm_mean"] = float(controlled_norms.mean().item())
        metrics["diag_control_norm_max"] = float(controlled_norms.max().item())
        metrics["diag_reference_drift_norm_mean"] = float(reference_drift_norms.mean().item())
        metrics["diag_control_to_drift_ratio"] = float(
            controlled_norms.mean().item() / max(reference_drift_norms.mean().item(), 1e-8)
        )
    else:
        metrics["diag_controlled_score_loss"] = nominal_diag_score_loss
        metrics["diag_controlled_control_cost"] = nominal_diag_control_cost
        metrics["diag_score_loss_gap"] = 0.0
        metrics["diag_control_cost_gap"] = 0.0
        metrics["diag_control_norm_mean"] = 0.0
        metrics["diag_control_norm_max"] = 0.0
        metrics["diag_reference_drift_norm_mean"] = 0.0
        metrics["diag_control_to_drift_ratio"] = 0.0

    if log_reverse_ablation and control_model is not None and not zero_control:
        if float(reverse_control_scale) == 0.0:
            metrics["reverse_no_control_mmd_rbf"] = metrics["mmd_rbf"]
            metrics["reverse_no_control_sliced_wasserstein"] = metrics["sliced_wasserstein"]
            metrics["reverse_control_gain_mmd"] = 0.0
            metrics["reverse_control_gain_swd"] = 0.0
        else:
            generated_no_control_std, _ = sample_reverse_chain(
                score_net=score_model,
                control_net=control_model,
                schedule=schedule,
                terminal_stats=terminal_stats,
                num_samples=num_eval_samples,
                data_dim=2,
                device=device,
                zero_control=True,
                collect_states=False,
                solver=reverse_solver,
                terminal_sampler=terminal_sampler,
                terminal_jitter_scale=terminal_jitter_scale,
                reverse_noise_scale=reverse_noise_scale,
                reverse_control_scale=reverse_control_scale,
                reverse_tail_noise_scale=reverse_tail_noise_scale,
                reverse_deterministic_tail_steps=reverse_deterministic_tail_steps,
                terminal_replay_indices=shared_reverse_randomness["terminal_replay_indices"],
                terminal_draw_noise=shared_reverse_randomness["terminal_draw_noise"],
                step_noises=shared_reverse_randomness["step_noises"],
            )
            generated_no_control = dataset.destandardize(generated_no_control_std.cpu()).cpu()
            fake_metric_no_control = generated_no_control[fake_idx]
            reverse_no_control_mmd = mmd_rbf(real_metric, fake_metric_no_control)
            reverse_no_control_swd = sliced_wasserstein(real_metric, fake_metric_no_control, seed=seed + epoch + 999)
            metrics["reverse_no_control_mmd_rbf"] = reverse_no_control_mmd
            metrics["reverse_no_control_sliced_wasserstein"] = reverse_no_control_swd
            metrics["reverse_control_gain_mmd"] = reverse_no_control_mmd - metrics["mmd_rbf"]
            metrics["reverse_control_gain_swd"] = reverse_no_control_swd - metrics["sliced_wasserstein"]
    else:
        metrics["reverse_no_control_mmd_rbf"] = metrics["mmd_rbf"]
        metrics["reverse_no_control_sliced_wasserstein"] = metrics["sliced_wasserstein"]
        metrics["reverse_control_gain_mmd"] = 0.0
        metrics["reverse_control_gain_swd"] = 0.0

    if not fast_tuning:
        save_scatter_comparison(
            path=outdir / "plots" / f"samples_epoch_{epoch:04d}.png",
            real_points=real_metric.numpy(),
            generated_points=fake_metric.numpy(),
            title=f"{dataset.name} | {format_markov_method_name(method_name)} | epoch {epoch}",
        )
        np.savez(
            outdir / "samples_latest.npz",
            real=real_metric.numpy(),
            generated=fake_metric.numpy(),
            generated_full=generated.numpy(),
        )

        snapshot_indices = select_snapshot_indices(total_count=len(reverse_states), num_snapshots=num_snapshot_steps)
        reverse_snapshots = [dataset.destandardize(reverse_states[idx]).numpy() for idx in snapshot_indices]
        reverse_titles = [f"reverse {idx}/{len(reverse_states) - 1}" for idx in snapshot_indices]
        save_process_snapshots(
            path=outdir / "plots" / f"reverse_process_epoch_{epoch:04d}.png",
            rows=[
                {
                    "row_title": "Reverse",
                    "snapshots": reverse_snapshots,
                    "titles": reverse_titles,
                    "color": "#2ca02c",
                }
            ],
            title=f"{format_markov_method_name(method_name)} reverse process | epoch {epoch}",
        )
    return metrics


def build_reverse_sampling_randomness(
    *,
    schedule,
    terminal_stats: TerminalStats,
    num_samples: int,
    data_dim: int,
    terminal_sampler: str,
    terminal_jitter_scale: float,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor | None]:
    generator = torch.Generator()
    generator.manual_seed(int(seed))

    terminal_replay_indices: torch.Tensor | None = None
    terminal_draw_noise: torch.Tensor | None = None
    if terminal_sampler == "replay" and terminal_stats.replay is not None and terminal_stats.replay.shape[0] > 0:
        terminal_replay_indices = torch.randint(
            0,
            terminal_stats.replay.shape[0],
            (num_samples,),
            generator=generator,
            dtype=torch.int64,
        ).to(device=device)
        if terminal_jitter_scale > 0.0:
            terminal_draw_noise = torch.randn(num_samples, data_dim, generator=generator, dtype=dtype).to(device=device)
    else:
        terminal_draw_noise = torch.randn(num_samples, data_dim, generator=generator, dtype=dtype).to(device=device)

    step_noises = torch.randn(schedule.dt.shape[0], num_samples, data_dim, generator=generator, dtype=dtype).to(
        device=device
    )
    return {
        "terminal_replay_indices": terminal_replay_indices,
        "terminal_draw_noise": terminal_draw_noise,
        "step_noises": step_noises,
    }


def initialize_control_head(module) -> None:
    final_layer = None
    if hasattr(module, "backbone"):
        final_layer = module.backbone[-1]
    elif hasattr(module, "head"):
        final_layer = module.head[-1]
    if isinstance(final_layer, torch.nn.Linear):
        torch.nn.init.zeros_(final_layer.weight)
        torch.nn.init.zeros_(final_layer.bias)


def build_control_model(*, args: argparse.Namespace, device: torch.device):
    kwargs = {
        "data_dim": 2,
        "hidden_dim": args.control_hidden_dim,
        "depth": args.control_depth,
        "embedding_dim": args.embedding_dim,
        "control_scale": args.control_scale,
    }
    if args.control_arch == "gru":
        return MarkovControlGRU(**kwargs).to(device)
    return MarkovControlMLP(**kwargs).to(device)


def build_score_model(*, args: argparse.Namespace, device: torch.device):
    kwargs = {
        "data_dim": 2,
        "hidden_dim": args.score_hidden_dim,
        "depth": args.score_depth,
        "embedding_dim": args.embedding_dim,
    }
    if args.score_arch == "precond":
        return MarkovPrecondScoreMLP(**kwargs, sigma_data=args.sigma_data).to(device)
    return MarkovScoreMLP(**kwargs).to(device)


def format_markov_method_name(method: str) -> str:
    if method == "baseline_score":
        return "Baseline Score"
    if method == "wdro_score":
        return "WDRO Score"
    if method == "cdro_markov":
        return "CDRO Markov"
    return method.replace("_", " ").title()


def resolve_target_total_budget(
    *,
    args: argparse.Namespace,
    epoch: int,
    total_epochs: int,
    active_epochs: int,
    score_model,
    schedule,
    dataset,
    device: torch.device,
    budget_state: dict[str, float | None],
) -> tuple[float | None, float, float]:
    if args.method != "cdro_markov" or args.adversary_steps <= 0 or args.control_scale <= 0.0:
        return None, 0.0, 0.0
    raw_budget: float | None = None
    if epoch <= args.warmup_epochs:
        base_budget = float(args.control_radius)
        budget_state["raw"] = None
        budget_state["smoothed"] = None
        target_budget = apply_budget_schedule(
            base_budget=base_budget,
            epoch=epoch,
            total_epochs=total_epochs,
            warmup_epochs=args.warmup_epochs,
            active_epochs=active_epochs,
            schedule_name="constant",
            frontload_power=float(args.budget_frontload_power),
            frontload_floor=float(args.budget_frontload_floor),
        )
        return raw_budget, base_budget, target_budget

    if args.budget_mode == "match_wdro_proxy":
        estimate_batch = min(
            max(int(args.budget_estimate_batch_size), 1),
            int(dataset.train_points.shape[0]),
        )
        estimate_points = dataset.train_points[:estimate_batch].to(device)
        raw_budget = estimate_markov_wdro_proxy_budget(
            estimate_points,
            score_model,
            schedule=schedule,
            gamma=float(args.reference_wdro_gamma),
            step_size=float(args.reference_wdro_step_size),
            iters=int(args.reference_wdro_k),
            clamp_min=dataset.bounds_min.to(device),
            clamp_max=dataset.bounds_max.to(device),
        )
        raw_budget *= float(args.budget_scale)
        previous = budget_state.get("smoothed")
        if previous is None:
            smoothed = raw_budget
        else:
            max_ratio = max(float(args.budget_max_ratio), 1.0)
            clipped = min(max(raw_budget, previous / max_ratio), previous * max_ratio)
            smoothed = float(args.budget_ema_decay) * previous + (1.0 - float(args.budget_ema_decay)) * clipped
        budget_state["raw"] = raw_budget
        budget_state["smoothed"] = smoothed
        base_budget = float(smoothed)
    else:
        base_budget = float(args.control_radius)
        budget_state["raw"] = base_budget
        budget_state["smoothed"] = base_budget
    target_budget = apply_budget_schedule(
        base_budget=base_budget,
        epoch=epoch,
        total_epochs=total_epochs,
        warmup_epochs=args.warmup_epochs,
        active_epochs=active_epochs,
        schedule_name=args.budget_schedule,
        frontload_power=float(args.budget_frontload_power),
        frontload_floor=float(args.budget_frontload_floor),
    )
    return raw_budget, base_budget, target_budget


def apply_budget_schedule(
    *,
    base_budget: float,
    epoch: int,
    total_epochs: int,
    warmup_epochs: int,
    active_epochs: int,
    schedule_name: str,
    frontload_power: float,
    frontload_floor: float,
) -> float:
    if schedule_name == "constant":
        return float(base_budget)
    if schedule_name != "frontload":
        raise ValueError(f"Unsupported budget schedule: {schedule_name}")
    if epoch <= warmup_epochs or active_epochs <= 1:
        return float(base_budget)

    frontload_floor = min(max(frontload_floor, 0.0), 1.0)
    active_idx = min(max(epoch - warmup_epochs, 1), active_epochs)
    progress = (active_idx - 1) / max(active_epochs - 1, 1)
    current_weight = frontload_floor + (1.0 - frontload_floor) * ((1.0 - progress) ** max(frontload_power, 0.0))
    grid = np.linspace(0.0, 1.0, num=active_epochs, dtype=np.float64)
    weights = frontload_floor + (1.0 - frontload_floor) * np.power(1.0 - grid, max(frontload_power, 0.0))
    normalized_weight = current_weight / float(weights.mean())
    return float(base_budget * normalized_weight)


def clip_gradients(parameters, *, grad_clip: float) -> None:
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(list(parameters), grad_clip)


def compute_grad_l2_norm(parameters) -> float:
    total = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        grad_norm = float(param.grad.detach().norm().item())
        total += grad_norm * grad_norm
    return math.sqrt(total)


def compute_param_l2_norm(parameters) -> float:
    total = 0.0
    for param in parameters:
        param_norm = float(param.detach().norm().item())
        total += param_norm * param_norm
    return math.sqrt(total)


def compute_point_cloud_stats(points: torch.Tensor) -> dict[str, float]:
    centered = points - points.mean(dim=0, keepdim=True)
    if points.shape[0] > 1:
        cov = centered.t().matmul(centered) / float(points.shape[0] - 1)
    else:
        cov = torch.zeros(points.shape[1], points.shape[1], dtype=points.dtype, device=points.device)
    cov_trace = float(torch.trace(cov).item())
    stabilized = cov + 1e-6 * torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
    sign, logabsdet = torch.linalg.slogdet(stabilized)
    cov_logdet = float(logabsdet.item()) if float(sign.item()) > 0 else float("nan")
    return {
        "mean_norm": float(points.norm(dim=1).mean().item()),
        "std_mean": float(points.std(dim=0, unbiased=False).mean().item()),
        "cov_trace": cov_trace,
        "cov_logdet": cov_logdet,
    }


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def should_refresh(*, epoch_idx: int, warmup: int, refresh_every: int) -> bool:
    if refresh_every <= 0:
        return False
    if epoch_idx < warmup:
        return False
    return (epoch_idx - warmup) % refresh_every == 0


def build_loader(*, points: torch.Tensor, batch_size: int, seed: int, pin_memory: bool) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    dataset = TensorDataset(points)
    effective_batch_size = points.shape[0] if batch_size <= 0 else min(batch_size, points.shape[0])
    return DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        generator=generator,
        drop_last=False,
        pin_memory=pin_memory,
    )


def select_snapshot_indices(*, total_count: int, num_snapshots: int) -> list[int]:
    if total_count < 1:
        return []
    if num_snapshots <= 1 or total_count == 1:
        return [0]

    raw = np.linspace(0, total_count - 1, num=min(num_snapshots, total_count))
    indices: list[int] = []
    for idx in np.rint(raw).astype(int).tolist():
        if not indices or idx != indices[-1]:
            indices.append(idx)
    return indices


def save_checkpoint(
    *,
    path: Path,
    score_model,
    control_model,
    schedule,
    terminal_stats: TerminalStats,
    args: argparse.Namespace,
    dataset,
    epoch: int,
    metrics: dict | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "score_model_state": score_model.state_dict(),
            "control_model_state": (control_model.state_dict() if control_model is not None else None),
            "schedule": schedule.to_dict(),
            "terminal_stats": {
                "mean": terminal_stats.mean.detach().cpu(),
                "var": terminal_stats.var.detach().cpu(),
                "replay": (terminal_stats.replay.detach().cpu() if terminal_stats.replay is not None else None),
            },
            "config": vars(args),
            "dataset": {
                "name": dataset.name,
                "mean": dataset.mean,
                "std": dataset.std,
                "bounds_min": dataset.bounds_min,
                "bounds_max": dataset.bounds_max,
            },
            "epoch": epoch,
            "metrics": metrics,
        },
        path,
    )


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=_json_default) + "\n")


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


if __name__ == "__main__":
    main()
