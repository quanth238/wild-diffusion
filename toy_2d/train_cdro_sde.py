from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from toy_2d.datasets import available_datasets, build_dataset
from toy_2d.plotting import (
    save_adversarial_debug,
    save_cdro_sde_forward_curves,
    save_process_snapshots,
)
from toy_2d.sde_models import CausalAdversaryGRU, CausalPredictorGRU
from toy_2d.vp_sde import (
    VPSchedule,
    build_vp_schedule,
    compute_control_cost,
    compute_vp_reference_target,
    compute_weighted_path_mse,
    sample_reference_paths,
)


@dataclass
class PolicyRollout:
    adv_path: torch.Tensor
    controls: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the toy VP-SDE causal-DRO forward model.")
    parser.add_argument("--outdir", type=Path, default=Path("toy-runs") / "cdro_sde_forward")
    parser.add_argument("--dataset", type=str, default="two_moons", choices=available_datasets())
    parser.add_argument("--num-samples", type=int, default=2048)
    parser.add_argument("--noise", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--standardize", action="store_true", default=True)
    parser.add_argument("--no-standardize", dest="standardize", action="store_false")

    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr-theta", type=float, default=1e-3)
    parser.add_argument("--lr-phi", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--adv-steps", type=int, default=3)
    parser.add_argument("--rho-budget", type=float, default=0.05)
    parser.add_argument("--lambda-init", type=float, default=1.0)
    parser.add_argument("--lambda-lr", type=float, default=0.05)
    parser.add_argument("--lambda-min", type=float, default=0.0)

    parser.add_argument("--num-steps", type=int, default=12)
    parser.add_argument("--total-time", type=float, default=1.0)
    parser.add_argument("--beta-start", type=float, default=0.5)
    parser.add_argument("--beta-end", type=float, default=10.0)
    parser.add_argument("--weight-mode", type=str, default="uniform", choices=("uniform", "noise_rate", "sqrt_noise_rate"))
    parser.add_argument("--reverse-noise-scale", type=float, default=0.5)

    parser.add_argument("--time-embedding-dim", type=int, default=16)
    parser.add_argument("--predictor-hidden-dim", type=int, default=96)
    parser.add_argument("--predictor-layers", type=int, default=1)
    parser.add_argument("--policy-hidden-dim", type=int, default=48)
    parser.add_argument("--control-scale", type=float, default=0.5)

    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--debug-points", type=int, default=128)
    parser.add_argument("--num-plot-paths", type=int, default=64)
    parser.add_argument("--replay-samples", type=int, default=4096)
    parser.add_argument("--save-eval-checkpoints", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
    schedule = build_vp_schedule(
        num_steps=args.num_steps,
        total_time=args.total_time,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        reverse_noise_scale=args.reverse_noise_scale,
        weight_mode=args.weight_mode,
    ).to(device=device, dtype=torch.float32)

    predictor = CausalPredictorGRU(
        data_dim=2,
        hidden_dim=args.predictor_hidden_dim,
        time_embedding_dim=args.time_embedding_dim,
        num_layers=args.predictor_layers,
    ).to(device)
    policy = CausalAdversaryGRU(
        data_dim=2,
        hidden_dim=args.policy_hidden_dim,
        time_embedding_dim=args.time_embedding_dim,
        control_scale=args.control_scale,
    ).to(device)

    theta_optimizer = torch.optim.Adam(
        predictor.parameters(),
        lr=args.lr_theta,
        weight_decay=args.weight_decay,
    )
    phi_optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=args.lr_phi,
        weight_decay=args.weight_decay,
    )

    lambda_value = max(float(args.lambda_init), float(args.lambda_min))
    history: list[dict] = []
    best_predictor_loss = float("inf")
    best_epoch: int | None = None
    best_eval: dict | None = None
    debug_points = select_debug_points(
        train_points=dataset.train_points,
        count=args.debug_points,
        seed=args.seed,
    ).to(device)

    start_time = time.time()
    for epoch_idx in range(args.epochs):
        epoch = epoch_idx + 1
        loader = build_loader(points=dataset.train_points, batch_size=args.batch_size, seed=args.seed + epoch_idx)
        epoch_predictor_losses: list[float] = []
        epoch_control_costs: list[float] = []

        predictor.train()
        policy.train()
        for (batch_x0,) in loader:
            batch_x0 = batch_x0.to(device)
            alpha_path, reference_noise = sample_reference_paths(x0=batch_x0, schedule=schedule)

            if args.adv_steps > 0:
                set_requires_grad(predictor, False)
                predictor.eval()
                for _ in range(args.adv_steps):
                    phi_optimizer.zero_grad(set_to_none=True)
                    rollout = rollout_causal_policy(
                        policy=policy,
                        alpha_path=alpha_path,
                        schedule=schedule,
                    )
                    targets = compute_vp_reference_target(
                        reference_noise=reference_noise,
                        controls=rollout.controls,
                        schedule=schedule,
                    )
                    predictions = predictor(
                        path_states=rollout.adv_path[:, :-1, :],
                        times=schedule.times[:-1],
                    )
                    path_loss = compute_weighted_path_mse(
                        predictions=predictions,
                        targets=targets,
                        schedule=schedule,
                    )
                    control_cost = compute_control_cost(controls=rollout.controls, schedule=schedule)
                    objective = path_loss.mean() - lambda_value * control_cost.mean()
                    phi_loss = -objective
                    phi_loss.backward()
                    clip_gradients(policy.parameters(), args.grad_clip)
                    phi_optimizer.step()
                set_requires_grad(predictor, True)
                predictor.train()

            rollout = rollout_causal_policy(
                policy=policy,
                alpha_path=alpha_path,
                schedule=schedule,
            )
            targets = compute_vp_reference_target(
                reference_noise=reference_noise,
                controls=rollout.controls,
                schedule=schedule,
            )
            control_cost = compute_control_cost(controls=rollout.controls.detach(), schedule=schedule)

            theta_optimizer.zero_grad(set_to_none=True)
            predictions = predictor(
                path_states=rollout.adv_path[:, :-1, :].detach(),
                times=schedule.times[:-1],
            )
            predictor_loss = compute_weighted_path_mse(
                predictions=predictions,
                targets=targets.detach(),
                schedule=schedule,
            ).mean()
            predictor_loss.backward()
            clip_gradients(predictor.parameters(), args.grad_clip)
            theta_optimizer.step()

            epoch_predictor_losses.append(float(predictor_loss.item()))
            epoch_control_costs.append(float(control_cost.mean().item()))
            lambda_value = max(
                float(args.lambda_min),
                float(lambda_value - args.lambda_lr * (args.rho_budget - float(control_cost.mean().item()))),
            )

        epoch_metrics = {
            "epoch": epoch,
            "predictor_loss": float(np.mean(epoch_predictor_losses)) if epoch_predictor_losses else float("nan"),
            "control_cost": float(np.mean(epoch_control_costs)) if epoch_control_costs else float("nan"),
            "lambda_value": float(lambda_value),
        }

        if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs:
            eval_metrics = evaluate_forward_epoch(
                dataset=dataset,
                predictor=predictor,
                policy=policy,
                schedule=schedule,
                debug_points=debug_points,
                outdir=args.outdir,
                epoch=epoch,
                lambda_value=lambda_value,
                num_plot_paths=args.num_plot_paths,
            )
            epoch_metrics.update(eval_metrics)
            append_jsonl(args.outdir / "metrics.jsonl", epoch_metrics)
            history.append(dict(epoch_metrics))
            save_cdro_sde_forward_curves(
                path=args.outdir / "plots" / "training_curves.png",
                history=history,
            )

            if epoch_metrics["predictor_loss"] < best_predictor_loss:
                best_predictor_loss = epoch_metrics["predictor_loss"]
                best_epoch = epoch
                best_eval = dict(epoch_metrics)
                save_checkpoint(
                    path=args.outdir / "checkpoint_best.pt",
                    predictor=predictor,
                    policy=policy,
                    args=args,
                    dataset=dataset,
                    schedule=schedule,
                    epoch=epoch,
                    lambda_value=lambda_value,
                    metrics=epoch_metrics,
                )

            if args.save_eval_checkpoints:
                save_checkpoint(
                    path=args.outdir / "checkpoints" / f"checkpoint_epoch_{epoch:04d}.pt",
                    predictor=predictor,
                    policy=policy,
                    args=args,
                    dataset=dataset,
                    schedule=schedule,
                    epoch=epoch,
                    lambda_value=lambda_value,
                    metrics=epoch_metrics,
                )

    buffer_payload = export_robust_buffer(
        dataset=dataset,
        policy=policy,
        schedule=schedule,
        num_samples=args.replay_samples,
        batch_size=args.batch_size,
        device=device,
        seed=args.seed + 101,
    )
    robust_buffer_path = args.outdir / "robust_buffer.pt"
    torch.save(buffer_payload, robust_buffer_path)

    final_summary = {
        "dataset": args.dataset,
        "epochs": args.epochs,
        "runtime_minutes": (time.time() - start_time) / 60.0,
        "lambda_value": float(lambda_value),
        "final_predictor_loss": history[-1]["predictor_loss"] if history else epoch_metrics["predictor_loss"],
        "final_control_cost": history[-1]["control_cost"] if history else epoch_metrics["control_cost"],
        "best_predictor_loss": best_predictor_loss,
        "best_epoch": best_epoch,
        "best_eval": best_eval,
        "last_eval": history[-1] if history else epoch_metrics,
        "robust_buffer_path": str(robust_buffer_path),
    }
    save_json(args.outdir / "summary.json", final_summary)
    save_checkpoint(
        path=args.outdir / "checkpoint_last.pt",
        predictor=predictor,
        policy=policy,
        args=args,
        dataset=dataset,
        schedule=schedule,
        epoch=args.epochs,
        lambda_value=lambda_value,
        metrics=history[-1] if history else epoch_metrics,
    )
    print(json.dumps(final_summary, indent=2))


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loader(*, points: torch.Tensor, batch_size: int, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(TensorDataset(points), batch_size=batch_size, shuffle=True, generator=generator, drop_last=False)


def select_debug_points(*, train_points: torch.Tensor, count: int, seed: int) -> torch.Tensor:
    generator = torch.Generator()
    generator.manual_seed(seed)
    num_points = min(int(count), int(train_points.shape[0]))
    indices = torch.randperm(train_points.shape[0], generator=generator)[:num_points]
    return train_points[indices].clone()


def rollout_causal_policy(
    *,
    policy: CausalAdversaryGRU,
    alpha_path: torch.Tensor,
    schedule: VPSchedule,
) -> PolicyRollout:
    batch_size, num_states, data_dim = alpha_path.shape
    if num_states != schedule.num_steps + 1:
        raise ValueError(f"alpha_path has {num_states} states but schedule expects {schedule.num_steps + 1}")
    adv_path = torch.empty_like(alpha_path)
    controls = torch.empty(batch_size, schedule.num_steps, data_dim, device=alpha_path.device, dtype=alpha_path.dtype)
    adv_path[:, 0, :] = alpha_path[:, 0, :]
    hidden_state = policy.init_hidden(batch_size=batch_size, device=alpha_path.device, dtype=alpha_path.dtype)

    for step_idx in range(schedule.num_steps):
        current_state = adv_path[:, step_idx, :]
        control, hidden_state = policy.step(
            current_state=current_state,
            time_value=schedule.times[step_idx : step_idx + 1],
            hidden_state=hidden_state,
        )
        adv_path[:, step_idx + 1, :] = current_state + (alpha_path[:, step_idx + 1, :] - alpha_path[:, step_idx, :]) + control * schedule.dt[step_idx]
        controls[:, step_idx, :] = control

    return PolicyRollout(adv_path=adv_path, controls=controls)


@torch.no_grad()
def evaluate_forward_epoch(
    *,
    dataset,
    predictor: CausalPredictorGRU,
    policy: CausalAdversaryGRU,
    schedule: VPSchedule,
    debug_points: torch.Tensor,
    outdir: Path,
    epoch: int,
    lambda_value: float,
    num_plot_paths: int,
) -> dict:
    predictor.eval()
    policy.eval()

    alpha_path, reference_noise = sample_reference_paths(x0=debug_points, schedule=schedule)
    rollout = rollout_causal_policy(policy=policy, alpha_path=alpha_path, schedule=schedule)
    targets = compute_vp_reference_target(
        reference_noise=reference_noise,
        controls=rollout.controls,
        schedule=schedule,
    )
    predictions = predictor(path_states=rollout.adv_path[:, :-1, :], times=schedule.times[:-1])

    predictor_loss = compute_weighted_path_mse(
        predictions=predictions,
        targets=targets,
        schedule=schedule,
    )
    control_cost = compute_control_cost(controls=rollout.controls, schedule=schedule)
    shift = torch.linalg.vector_norm(rollout.adv_path[:, -1, :] - alpha_path[:, -1, :], dim=1)

    save_adversarial_debug(
        path=outdir / "plots" / f"controlled_endpoint_epoch_{epoch:04d}.png",
        original_points=dataset.destandardize(alpha_path[:, -1, :].cpu()).numpy(),
        adversarial_points=dataset.destandardize(rollout.adv_path[:, -1, :].cpu()).numpy(),
        title=f"VP controlled terminal state | epoch {epoch}",
        mean_l2_shift=float(shift.mean().item()),
        max_l2_shift=float(shift.max().item()),
    )
    save_process_snapshots(
        path=outdir / "plots" / f"path_process_epoch_{epoch:04d}.png",
        rows=build_path_rows(
            dataset=dataset,
            reference_path=alpha_path[:num_plot_paths].cpu(),
            controlled_path=rollout.adv_path[:num_plot_paths].cpu(),
        ),
        title=f"{dataset.name} | reference vs controlled forward path | epoch {epoch}",
    )

    return {
        "predictor_loss": float(predictor_loss.mean().item()),
        "control_cost": float(control_cost.mean().item()),
        "lambda_value": float(lambda_value),
        "mean_control_norm": float(torch.linalg.vector_norm(rollout.controls, dim=2).mean().item()),
        "max_control_norm": float(torch.linalg.vector_norm(rollout.controls, dim=2).max().item()),
        "mean_target_norm": float(torch.linalg.vector_norm(targets, dim=2).mean().item()),
        "mean_terminal_shift": float(shift.mean().item()),
        "max_terminal_shift": float(shift.max().item()),
    }


def build_path_rows(
    *,
    dataset,
    reference_path: torch.Tensor,
    controlled_path: torch.Tensor,
) -> list[dict]:
    snapshot_indices = select_snapshot_indices(total_count=reference_path.shape[1], num_snapshots=min(5, reference_path.shape[1]))
    rows = []
    for name, path, color in (
        ("Reference", reference_path, "#1f77b4"),
        ("Controlled", controlled_path, "#d62728"),
    ):
        snapshots = [dataset.destandardize(path[:, idx, :]).numpy() for idx in snapshot_indices]
        titles = [f"step {idx}/{path.shape[1] - 1}" for idx in snapshot_indices]
        rows.append(
            {
                "row_title": name,
                "snapshots": snapshots,
                "titles": titles,
                "color": color,
            }
        )
    return rows


def select_snapshot_indices(*, total_count: int, num_snapshots: int) -> list[int]:
    if total_count <= 0:
        return []
    if num_snapshots <= 1 or total_count == 1:
        return [0]
    raw = np.linspace(0, total_count - 1, num=min(num_snapshots, total_count))
    indices: list[int] = []
    for idx in np.rint(raw).astype(int).tolist():
        if not indices or idx != indices[-1]:
            indices.append(idx)
    return indices


@torch.no_grad()
def export_robust_buffer(
    *,
    dataset,
    policy: CausalAdversaryGRU,
    schedule: VPSchedule,
    num_samples: int,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> dict:
    policy.eval()
    generator = torch.Generator()
    generator.manual_seed(seed)

    adv_paths: list[torch.Tensor] = []
    reference_paths: list[torch.Tensor] = []
    controls: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    reference_noise: list[torch.Tensor] = []

    remaining = int(num_samples)
    total_points = int(dataset.train_points.shape[0])
    while remaining > 0:
        count = min(int(batch_size), remaining)
        indices = torch.randint(total_points, size=(count,), generator=generator)
        x0 = dataset.train_points[indices].to(device)
        alpha_path, xi = sample_reference_paths(x0=x0, schedule=schedule)
        rollout = rollout_causal_policy(policy=policy, alpha_path=alpha_path, schedule=schedule)
        tau = compute_vp_reference_target(
            reference_noise=xi,
            controls=rollout.controls,
            schedule=schedule,
        )
        adv_paths.append(rollout.adv_path.cpu())
        reference_paths.append(alpha_path.cpu())
        controls.append(rollout.controls.cpu())
        targets.append(tau.cpu())
        reference_noise.append(xi.cpu())
        remaining -= count

    return {
        "adv_paths": torch.cat(adv_paths, dim=0),
        "reference_paths": torch.cat(reference_paths, dim=0),
        "controls": torch.cat(controls, dim=0),
        "targets": torch.cat(targets, dim=0),
        "reference_noise": torch.cat(reference_noise, dim=0),
        "dataset": serialize_dataset_metadata(dataset),
        "schedule": serialize_schedule(schedule),
    }


def set_requires_grad(module: torch.nn.Module, flag: bool) -> None:
    for param in module.parameters():
        param.requires_grad_(flag)


def clip_gradients(parameters, max_norm: float) -> None:
    if max_norm <= 0:
        return
    torch.nn.utils.clip_grad_norm_(list(parameters), max_norm)


def save_checkpoint(
    *,
    path: Path,
    predictor: CausalPredictorGRU,
    policy: CausalAdversaryGRU,
    args: argparse.Namespace,
    dataset,
    schedule: VPSchedule,
    epoch: int,
    lambda_value: float,
    metrics: dict | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "predictor_state": predictor.state_dict(),
            "policy_state": policy.state_dict(),
            "args": vars(args),
            "dataset": serialize_dataset_metadata(dataset),
            "schedule": serialize_schedule(schedule),
            "epoch": epoch,
            "lambda_value": float(lambda_value),
            "metrics": metrics,
        },
        path,
    )


def serialize_schedule(schedule: VPSchedule) -> dict:
    return {
        "times": schedule.times.cpu(),
        "dt": schedule.dt.cpu(),
        "noise_rates": schedule.noise_rates.cpu(),
        "weights": schedule.weights.cpu(),
        "reverse_noise_scales": schedule.reverse_noise_scales.cpu(),
    }


def serialize_dataset_metadata(dataset) -> dict:
    return {
        "name": dataset.name,
        "mean": dataset.mean.cpu(),
        "std": dataset.std.cpu(),
        "bounds_min": dataset.bounds_min.cpu(),
        "bounds_max": dataset.bounds_max.cpu(),
    }


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
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return float(value.item())
        return value.tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


if __name__ == "__main__":
    main()
