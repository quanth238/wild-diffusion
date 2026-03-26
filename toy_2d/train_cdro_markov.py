from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from toy_2d.cdro_markov import (
    MarkovControlGRU,
    MarkovControlMLP,
    MarkovScoreMLP,
    TerminalStats,
    build_markov_vp_schedule,
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
    save_markov_score_training_curves,
    save_process_snapshots,
    save_scatter_comparison,
)
from toy_2d.robust_defaults import WDRO_CORE_DEFAULTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Markov score-based CDRO toy model.")
    parser.add_argument("--outdir", type=Path, default=Path("toy-runs") / "cdro_markov")
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
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--warmup-epochs", type=int, default=8)
    parser.add_argument("--adversary-steps", type=int, default=1)
    parser.add_argument("--adversary-stop-epoch", type=int, default=0)
    parser.add_argument("--score-steps", type=int, default=8)
    parser.add_argument("--terminal-momentum", type=float, default=0.95)

    parser.add_argument("--num-steps", type=int, default=12)
    parser.add_argument("--total-time", type=float, default=1.0)
    parser.add_argument("--beta-min", type=float, default=0.2)
    parser.add_argument("--beta-max", type=float, default=6.0)
    parser.add_argument(
        "--score-weight-schedule",
        type=str,
        default="uniform",
        choices=("uniform", "sigma_sq", "inv_sigma_sq"),
    )

    parser.add_argument("--score-hidden-dim", type=int, default=128)
    parser.add_argument("--score-depth", type=int, default=4)
    parser.add_argument("--control-hidden-dim", type=int, default=64)
    parser.add_argument("--control-depth", type=int, default=3)
    parser.add_argument("--control-arch", type=str, default="mlp", choices=("mlp", "gru"))
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--control-scale", type=float, default=0.5)

    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--num-eval-samples", type=int, default=2048)
    parser.add_argument("--metric-samples", type=int, default=1024)
    parser.add_argument("--num-snapshot-steps", type=int, default=6)
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

    schedule = build_markov_vp_schedule(
        num_steps=args.num_steps,
        total_time=args.total_time,
        beta_min=args.beta_min,
        beta_max=args.beta_max,
        device=device,
        dtype=dataset.train_points.dtype,
        weight_schedule=args.score_weight_schedule,
    )

    score_model = MarkovScoreMLP(
        data_dim=2,
        hidden_dim=args.score_hidden_dim,
        depth=args.score_depth,
        embedding_dim=args.embedding_dim,
    ).to(device)
    score_ema = copy.deepcopy(score_model).eval()

    control_model = build_control_model(args=args, device=device)
    initialize_control_head(control_model)
    control_ema = copy.deepcopy(control_model).eval()

    score_optimizer = torch.optim.Adam(score_model.parameters(), lr=args.score_lr)
    control_optimizer = torch.optim.Adam(control_model.parameters(), lr=args.control_lr)
    lambda_value = float(args.lambda_init)
    terminal_stats: TerminalStats | None = None

    history: list[dict] = []
    eval_history: list[dict] = []
    best_swd = float("inf")
    best_epoch: int | None = None
    best_eval: dict | None = None
    budget_state: dict[str, float | None] = {"raw": None, "smoothed": None}

    start_time = time.time()
    active_epochs = max(args.epochs - args.warmup_epochs, 1)
    for epoch_idx in range(args.epochs):
        epoch = epoch_idx + 1
        loader = build_loader(
            points=dataset.train_points,
            batch_size=args.batch_size,
            seed=args.seed + epoch_idx,
            pin_memory=device.type == "cuda",
        )
        control_present = epoch_idx >= args.warmup_epochs and args.adversary_steps > 0
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

        for (batch,) in loader:
            batch = batch.to(device, non_blocking=device.type == "cuda")

            if adversary_update_active:
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
                update_ema(ema=control_ema, model=control_model, decay=args.ema_decay)
                set_module_grad(score_model, True)

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
                )

            set_module_grad(control_model, True)

        epoch_metrics = {
            "epoch": epoch_idx + 1,
            "score_loss": float(np.mean(epoch_score_losses)) if epoch_score_losses else float("nan"),
            "control_cost": float(np.mean(epoch_control_costs)) if epoch_control_costs else 0.0,
            "adversary_value": float(np.mean(epoch_adv_values)) if epoch_adv_values else 0.0,
            "mean_control_norm": float(np.mean(epoch_control_mean_norms)) if epoch_control_mean_norms else 0.0,
            "max_control_norm": float(np.max(epoch_control_max_norms)) if epoch_control_max_norms else 0.0,
            "target_total_budget": float(target_total_budget),
            "base_total_budget": float(base_total_budget),
            "raw_budget_estimate": (None if raw_budget_estimate is None else float(raw_budget_estimate)),
            "lambda_value": float(lambda_value),
            "control_active": bool(control_present),
            "adversary_update_active": bool(adversary_update_active),
        }
        epoch_metrics["budget_utilization"] = (
            float(epoch_metrics["control_cost"] / target_total_budget) if target_total_budget > 0 else 0.0
        )
        if terminal_stats is not None:
            epoch_metrics["terminal_var_mean"] = float(terminal_stats.var.mean().item())
            epoch_metrics["terminal_var_min"] = float(terminal_stats.var.min().item())
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
                fast_tuning=args.fast_tuning,
            )
            eval_metrics.update(epoch_metrics)
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
    fast_tuning: bool = False,
) -> dict:
    score_model.eval()
    if control_model is not None:
        control_model.eval()

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
    }

    if not fast_tuning:
        save_scatter_comparison(
            path=outdir / "plots" / f"samples_epoch_{epoch:04d}.png",
            real_points=real_metric.numpy(),
            generated_points=fake_metric.numpy(),
            title=f"{dataset.name} | Markov CDRO | epoch {epoch}",
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
            title=f"Markov CDRO reverse process | epoch {epoch}",
        )
    return metrics


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


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
            "control_model_state": control_model.state_dict(),
            "schedule": schedule.to_dict(),
            "terminal_stats": {
                "mean": terminal_stats.mean.detach().cpu(),
                "var": terminal_stats.var.detach().cpu(),
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
