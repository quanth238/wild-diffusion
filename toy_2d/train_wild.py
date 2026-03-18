from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from toy_2d.datasets import available_datasets, build_dataset
from toy_2d.causal import build_forward_path, causal_wdro_loss, sample_path_sigmas, solve_causal_path_attack
from toy_2d.losses import EDMLoss2D
from toy_2d.metrics import mmd_rbf, sliced_wasserstein
from toy_2d.model import EDMPrecondMLP
from toy_2d.plotting import (
    compute_plot_bounds,
    save_adversarial_debug,
    save_process_snapshots,
    save_scatter_comparison,
    save_training_curves,
)
from toy_2d.robust_defaults import CAUSAL_WDRO_DEFAULTS, WDRO_CORE_DEFAULTS
from toy_2d.sampler import sample_edm, sample_edm_trajectory
from toy_2d.wild import build_wdro_dataset, estimate_wdro_transport_budget


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a toy 2D WILD-style diffusion baseline on CPU or GPU.")
    parser.add_argument("--outdir", type=Path, default=Path("toy-runs") / "wild_2d")
    parser.add_argument("--method", type=str, default="wdro", choices=("baseline", "wdro", "causal_wdro"))
    parser.add_argument("--dataset", type=str, default="eight_gaussians", choices=available_datasets())
    parser.add_argument("--num-samples", type=int, default=4096)
    parser.add_argument("--noise", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--standardize", action="store_true", default=True)
    parser.add_argument("--no-standardize", dest="standardize", action="store_false")

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--ema-decay", type=float, default=0.995)

    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--embedding-dim", type=int, default=32)

    parser.add_argument("--p-mean", type=float, default=-1.2)
    parser.add_argument("--p-std", type=float, default=1.2)
    parser.add_argument("--sigma-data", type=float, default=0.5)
    parser.add_argument("--sigma-min", type=float, default=0.002)
    parser.add_argument("--sigma-max", type=float, default=10.0)
    parser.add_argument("--rho", type=float, default=7.0)
    parser.add_argument("--sampler-steps", type=int, default=40)

    parser.add_argument("--wdro-warmup-epochs", type=int, default=25)
    parser.add_argument("--wdro-refresh-every", type=int, default=10)
    parser.add_argument("--wdro-k", type=int, default=WDRO_CORE_DEFAULTS["k"])
    parser.add_argument("--wdro-step-size", type=float, default=WDRO_CORE_DEFAULTS["step_size"])
    parser.add_argument("--wdro-gamma", type=float, default=WDRO_CORE_DEFAULTS["gamma"])
    parser.add_argument("--wdro-p-adv", type=float, default=WDRO_CORE_DEFAULTS["p_adv"])
    parser.add_argument("--wdro-debug-points", type=int, default=256)

    parser.add_argument("--causal-path-steps", type=int, default=CAUSAL_WDRO_DEFAULTS["path_steps"])
    parser.add_argument("--causal-warmup-epochs", type=int, default=CAUSAL_WDRO_DEFAULTS["warmup_epochs"])
    parser.add_argument("--causal-inner-steps", type=int, default=CAUSAL_WDRO_DEFAULTS["inner_steps"])
    parser.add_argument("--causal-step-size", type=float, default=CAUSAL_WDRO_DEFAULTS["step_size"])
    parser.add_argument("--causal-gamma", type=float, default=CAUSAL_WDRO_DEFAULTS["gamma"])
    parser.add_argument("--causal-total-budget", type=float, default=CAUSAL_WDRO_DEFAULTS["total_budget"])
    parser.add_argument(
        "--causal-budget-mode",
        type=str,
        default=CAUSAL_WDRO_DEFAULTS["budget_mode"],
        choices=("fixed", "match_wdro"),
    )
    parser.add_argument(
        "--causal-exact-budget-split",
        action=argparse.BooleanOptionalAction,
        default=CAUSAL_WDRO_DEFAULTS["exact_budget_split"],
    )
    parser.add_argument("--causal-reference-wdro-k", type=int, default=None)
    parser.add_argument("--causal-reference-wdro-step-size", type=float, default=None)
    parser.add_argument("--causal-reference-wdro-gamma", type=float, default=None)
    parser.add_argument("--causal-debug-snapshots", type=int, default=0)
    parser.add_argument("--causal-debug-points", type=int, default=256)
    parser.add_argument(
        "--causal-sigma-schedule",
        type=str,
        default=CAUSAL_WDRO_DEFAULTS["sigma_schedule"],
        choices=("edm_random", "edm_quantiles", "karras_grid"),
    )

    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--num-eval-samples", type=int, default=2048)
    parser.add_argument("--metric-samples", type=int, default=1024)
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

    model = EDMPrecondMLP(
        data_dim=2,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        embedding_dim=args.embedding_dim,
        sigma_data=args.sigma_data,
    ).to(device)
    ema = copy.deepcopy(model).eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = EDMLoss2D(p_mean=args.p_mean, p_std=args.p_std, sigma_data=args.sigma_data)

    base_points = dataset.train_points.clone()
    current_points = base_points.clone()
    clamp_min = dataset.bounds_min.to(device)
    clamp_max = dataset.bounds_max.to(device)
    rng = np.random.default_rng(args.seed + 1)

    loss_history: list[float] = []
    eval_history: list[dict] = []
    latest_adv_snapshot: tuple[np.ndarray, np.ndarray] | None = None
    latest_adv_stats: dict | None = None
    latest_adv_plot_stats: tuple[float, float] | None = None
    latest_causal_stats: dict | None = None
    best_swd = float("inf")
    best_epoch: int | None = None
    best_eval: dict | None = None

    start_time = time.time()
    for epoch_idx in range(args.epochs):
        epoch = epoch_idx + 1
        if args.method == "wdro" and should_refresh(
            epoch_idx=epoch_idx,
            warmup=args.wdro_warmup_epochs,
            refresh_every=args.wdro_refresh_every,
        ):
            refresh = build_wdro_dataset(
                base_points=base_points,
                model=ema,
                loss_fn=loss_fn,
                batch_size=args.batch_size,
                gamma=args.wdro_gamma,
                step_size=args.wdro_step_size,
                iters=args.wdro_k,
                p_adv=args.wdro_p_adv,
                clamp_min=clamp_min,
                clamp_max=clamp_max,
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

        loader = build_loader(points=current_points, batch_size=args.batch_size, seed=args.seed + epoch_idx)
        mean_loss, train_stats = train_one_epoch(
            model=model,
            ema=ema,
            optimizer=optimizer,
            loader=loader,
            loss_fn=loss_fn,
            device=device,
            grad_clip=args.grad_clip,
            ema_decay=args.ema_decay,
            method=args.method,
            causal_attack_active=epoch_idx >= args.causal_warmup_epochs,
            causal_kwargs={
                "warmup_epochs": args.causal_warmup_epochs,
                "sigma_min": args.sigma_min,
                "sigma_max": args.sigma_max,
                "rho": args.rho,
                "p_mean": args.p_mean,
                "p_std": args.p_std,
                "path_steps": args.causal_path_steps,
                "inner_steps": args.causal_inner_steps,
                "step_size": args.causal_step_size,
                "gamma": args.causal_gamma,
                "total_budget": args.causal_total_budget,
                "budget_mode": args.causal_budget_mode,
                "exact_budget_split": args.causal_exact_budget_split,
                "sigma_schedule": args.causal_sigma_schedule,
                "reference_wdro_k": args.causal_reference_wdro_k if args.causal_reference_wdro_k is not None else args.wdro_k,
                "reference_wdro_step_size": (
                    args.causal_reference_wdro_step_size
                    if args.causal_reference_wdro_step_size is not None
                    else args.wdro_step_size
                ),
                "reference_wdro_gamma": (
                    args.causal_reference_wdro_gamma if args.causal_reference_wdro_gamma is not None else args.wdro_gamma
                ),
                "clamp_min": clamp_min,
                "clamp_max": clamp_max,
            },
        )
        loss_history.append(mean_loss)
        if args.method == "causal_wdro":
            latest_causal_stats = train_stats

        if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs:
            metrics = evaluate(
                dataset=dataset,
                model=ema,
                outdir=args.outdir,
                epoch=epoch,
                device=device,
                num_eval_samples=args.num_eval_samples,
                metric_samples=args.metric_samples,
                sigma_min=args.sigma_min,
                sigma_max=args.sigma_max,
                rho=args.rho,
                sampler_steps=args.sampler_steps,
                seed=args.seed,
                causal_debug=build_causal_debug_config(args=args) if args.method == "causal_wdro" else None,
            )
            metrics["train_loss"] = mean_loss
            metrics["dataset_size"] = int(current_points.shape[0])
            if latest_adv_stats is not None:
                metrics["mean_l2_shift"] = latest_adv_stats["mean_l2_shift"]
                metrics["max_l2_shift"] = latest_adv_stats["max_l2_shift"]
                metrics["mean_transport_cost"] = latest_adv_stats["mean_transport_cost"]
                metrics["max_transport_cost"] = latest_adv_stats["max_transport_cost"]
                metrics["total_transport_cost"] = latest_adv_stats["mean_total_transport_cost"]
                metrics["max_total_transport_cost"] = latest_adv_stats["max_total_transport_cost"]
                metrics["wdro_mean_l2_shift"] = latest_adv_stats["mean_l2_shift"]
                metrics["wdro_max_l2_shift"] = latest_adv_stats["max_l2_shift"]
                metrics["wdro_mean_transport_cost"] = latest_adv_stats["mean_transport_cost"]
                metrics["wdro_max_transport_cost"] = latest_adv_stats["max_transport_cost"]
                metrics["wdro_total_transport_cost"] = latest_adv_stats["mean_total_transport_cost"]
                metrics["wdro_max_total_transport_cost"] = latest_adv_stats["max_total_transport_cost"]
            if latest_causal_stats is not None:
                metrics["causal_mean_l2_shift"] = latest_causal_stats["mean_l2_shift"]
                metrics["causal_max_l2_shift"] = latest_causal_stats["max_l2_shift"]
                metrics["causal_mean_transport_cost"] = latest_causal_stats["mean_transport_cost"]
                metrics["causal_max_transport_cost"] = latest_causal_stats["max_transport_cost"]
                metrics["causal_total_transport_cost"] = latest_causal_stats["mean_total_transport_cost"]
                metrics["causal_max_total_transport_cost"] = latest_causal_stats["max_total_transport_cost"]
                metrics["causal_target_total_budget"] = latest_causal_stats["target_total_budget"]
                metrics["mean_l2_shift"] = latest_causal_stats["mean_l2_shift"]
                metrics["max_l2_shift"] = latest_causal_stats["max_l2_shift"]
                metrics["mean_transport_cost"] = latest_causal_stats["mean_transport_cost"]
                metrics["max_transport_cost"] = latest_causal_stats["max_transport_cost"]
                metrics["total_transport_cost"] = latest_causal_stats["mean_total_transport_cost"]
                metrics["max_total_transport_cost"] = latest_causal_stats["max_total_transport_cost"]
            metrics["method"] = args.method
            eval_history.append(metrics)
            append_jsonl(args.outdir / "metrics.jsonl", metrics)
            if args.save_eval_checkpoints:
                save_checkpoint(
                    path=args.outdir / "checkpoints" / f"checkpoint_epoch_{epoch:04d}.pt",
                    model=ema,
                    args=args,
                    dataset=dataset,
                    epoch=epoch,
                    metrics=metrics,
                )

            if latest_adv_snapshot is not None and latest_adv_plot_stats is not None:
                save_adversarial_debug(
                    path=args.outdir / "plots" / f"adv_epoch_{epoch:04d}.png",
                    original_points=latest_adv_snapshot[0],
                    adversarial_points=latest_adv_snapshot[1],
                    title=f"WDRO refresh at epoch {epoch}",
                    mean_l2_shift=latest_adv_plot_stats[0],
                    max_l2_shift=latest_adv_plot_stats[1],
                )
            save_training_curves(
                path=args.outdir / "plots" / "training_curves.png",
                loss_history=loss_history,
                eval_history=eval_history,
            )

            if metrics["sliced_wasserstein"] < best_swd:
                best_swd = metrics["sliced_wasserstein"]
                best_epoch = epoch
                best_eval = dict(metrics)
                save_checkpoint(
                    path=args.outdir / "checkpoint_best.pt",
                    model=ema,
                    args=args,
                    dataset=dataset,
                    epoch=epoch,
                    metrics=metrics,
                )

    total_minutes = (time.time() - start_time) / 60.0
    final_summary = {
        "dataset": args.dataset,
        "epochs": args.epochs,
        "runtime_minutes": total_minutes,
        "final_loss": loss_history[-1],
        "best_sliced_wasserstein": best_swd,
        "best_epoch": best_epoch,
        "best_eval": best_eval,
        "last_eval": eval_history[-1] if eval_history else None,
    }
    save_json(args.outdir / "summary.json", final_summary)
    save_checkpoint(
        path=args.outdir / "checkpoint_last.pt",
        model=ema,
        args=args,
        dataset=dataset,
        epoch=args.epochs,
        metrics=eval_history[-1] if eval_history else None,
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


def should_refresh(*, epoch_idx: int, warmup: int, refresh_every: int) -> bool:
    if refresh_every <= 0:
        return False
    if epoch_idx < warmup:
        return False
    return (epoch_idx - warmup) % refresh_every == 0


def build_loader(*, points: torch.Tensor, batch_size: int, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    dataset = TensorDataset(points)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator, drop_last=False)


def train_one_epoch(
    *,
    model,
    ema,
    optimizer,
    loader: DataLoader,
    loss_fn,
    device: torch.device,
    grad_clip: float,
    ema_decay: float,
    method: str,
    causal_attack_active: bool,
    causal_kwargs: dict,
) -> tuple[float, dict | None]:
    model.train()
    losses: list[float] = []
    causal_mean_norms: list[float] = []
    causal_max_norms: list[float] = []
    causal_mean_costs: list[float] = []
    causal_max_costs: list[float] = []
    causal_mean_total_costs: list[float] = []
    causal_max_total_costs: list[float] = []
    causal_target_budgets: list[float] = []
    for (batch,) in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        if method == "causal_wdro" and causal_attack_active:
            total_budget = normalize_causal_total_budget(
                total_budget=causal_kwargs["total_budget"],
                budget_mode=causal_kwargs["budget_mode"],
            )
            if causal_kwargs["budget_mode"] == "match_wdro":
                total_budget = estimate_wdro_transport_budget(
                    batch,
                    ema,
                    loss_fn,
                    gamma=float(causal_kwargs["reference_wdro_gamma"]),
                    step_size=float(causal_kwargs["reference_wdro_step_size"]),
                    iters=int(causal_kwargs["reference_wdro_k"]),
                    clamp_min=causal_kwargs["clamp_min"],
                    clamp_max=causal_kwargs["clamp_max"],
                )
            loss, stats = causal_wdro_loss(
                train_net=model,
                attack_net=ema,
                clean_points=batch,
                p_mean=causal_kwargs["p_mean"],
                p_std=causal_kwargs["p_std"],
                sigma_min=causal_kwargs["sigma_min"],
                sigma_max=causal_kwargs["sigma_max"],
                rho=causal_kwargs["rho"],
                path_steps=causal_kwargs["path_steps"],
                inner_steps=causal_kwargs["inner_steps"],
                step_size=causal_kwargs["step_size"],
                gamma=causal_kwargs["gamma"],
                total_budget=total_budget,
                exact_budget_split=bool(causal_kwargs["exact_budget_split"]),
                sigma_schedule=causal_kwargs["sigma_schedule"],
            )
            causal_mean_norms.append(stats.mean_delta_norm)
            causal_max_norms.append(stats.max_delta_norm)
            causal_mean_costs.append(stats.mean_transport_cost)
            causal_max_costs.append(stats.max_transport_cost)
            causal_mean_total_costs.append(stats.mean_total_transport_cost)
            causal_max_total_costs.append(stats.max_total_transport_cost)
            if stats.target_total_budget is not None:
                causal_target_budgets.append(float(stats.target_total_budget))
        else:
            loss = loss_fn(model, batch).mean()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        update_ema(ema=ema, model=model, decay=ema_decay)
        losses.append(float(loss.item()))
    stats = None
    if method == "causal_wdro" and causal_mean_norms:
        stats = {
            "mean_l2_shift": float(np.mean(causal_mean_norms)),
            "max_l2_shift": float(np.max(causal_max_norms)),
            "mean_transport_cost": float(np.mean(causal_mean_costs)),
            "max_transport_cost": float(np.max(causal_max_costs)),
            "mean_total_transport_cost": float(np.mean(causal_mean_total_costs)),
            "max_total_transport_cost": float(np.max(causal_max_total_costs)),
            "target_total_budget": float(np.mean(causal_target_budgets)) if causal_target_budgets else None,
        }
    return float(np.mean(losses)) if losses else float("nan"), stats


@torch.no_grad()
def update_ema(*, ema, model, decay: float) -> None:
    for ema_param, model_param in zip(ema.parameters(), model.parameters()):
        ema_param.data.mul_(decay).add_(model_param.data, alpha=1.0 - decay)
    for ema_buffer, model_buffer in zip(ema.buffers(), model.buffers()):
        ema_buffer.copy_(model_buffer)


def evaluate(
    *,
    dataset,
    model,
    outdir: Path,
    epoch: int,
    device: torch.device,
    num_eval_samples: int,
    metric_samples: int,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    sampler_steps: int,
    seed: int,
    causal_debug: dict | None,
) -> dict:
    model.eval()
    generated_std = sample_edm(
        model,
        num_samples=num_eval_samples,
        data_dim=2,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        rho=rho,
        num_steps=sampler_steps,
        device=device,
    ).cpu()
    generated = dataset.destandardize(generated_std).cpu()

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

    save_scatter_comparison(
        path=outdir / "plots" / f"samples_epoch_{epoch:04d}.png",
        real_points=real_metric.numpy(),
        generated_points=fake_metric.numpy(),
        title=f"{dataset.name} | epoch {epoch}",
    )
    np.savez(
        outdir / "samples_latest.npz",
        real=real_metric.numpy(),
        generated=fake_metric.numpy(),
        generated_full=generated.numpy(),
    )
    if causal_debug is not None and causal_debug["num_snapshots"] > 0:
        save_causal_debug_process(
            dataset=dataset,
            model=model,
            outdir=outdir,
            epoch=epoch,
            device=device,
            sampler_steps=sampler_steps,
            seed=seed,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            rho=rho,
            causal_debug=causal_debug,
        )
    return metrics


def build_causal_debug_config(*, args: argparse.Namespace) -> dict:
    return {
        "num_snapshots": args.causal_debug_snapshots,
        "num_points": args.causal_debug_points,
        "p_mean": args.p_mean,
        "p_std": args.p_std,
        "path_steps": args.causal_path_steps,
        "inner_steps": args.causal_inner_steps,
        "step_size": args.causal_step_size,
        "gamma": args.causal_gamma,
        "total_budget": args.causal_total_budget,
        "budget_mode": args.causal_budget_mode,
        "exact_budget_split": args.causal_exact_budget_split,
        "sigma_schedule": args.causal_sigma_schedule,
        "reference_wdro_k": args.causal_reference_wdro_k if args.causal_reference_wdro_k is not None else args.wdro_k,
        "reference_wdro_step_size": (
            args.causal_reference_wdro_step_size if args.causal_reference_wdro_step_size is not None else args.wdro_step_size
        ),
        "reference_wdro_gamma": (
            args.causal_reference_wdro_gamma if args.causal_reference_wdro_gamma is not None else args.wdro_gamma
        ),
    }


def save_causal_debug_process(
    *,
    dataset,
    model,
    outdir: Path,
    epoch: int,
    device: torch.device,
    sampler_steps: int,
    seed: int,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    causal_debug: dict,
) -> None:
    num_points = min(int(causal_debug["num_points"]), int(dataset.train_points.shape[0]))
    if num_points <= 0:
        return

    generator = torch.Generator()
    generator.manual_seed(seed)
    indices = torch.randperm(dataset.train_points.shape[0], generator=generator)[:num_points]
    clean_points = dataset.train_points[indices].to(device)

    sigmas = sample_path_sigmas(
        batch_size=clean_points.shape[0],
        num_steps=int(causal_debug["path_steps"]),
        p_mean=float(causal_debug["p_mean"]),
        p_std=float(causal_debug["p_std"]),
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        rho=rho,
        device=device,
        dtype=clean_points.dtype,
        schedule=str(causal_debug["sigma_schedule"]),
    )
    reference_path = build_forward_path(clean_points=clean_points, sigmas=sigmas, shared_noise=False)
    total_budget = float(causal_debug["total_budget"])
    total_budget = normalize_causal_total_budget(
        total_budget=total_budget,
        budget_mode=str(causal_debug["budget_mode"]),
    )
    if causal_debug["budget_mode"] == "match_wdro":
        total_budget = estimate_wdro_transport_budget(
            clean_points,
            model,
            EDMLoss2D(
                p_mean=float(causal_debug["p_mean"]),
                p_std=float(causal_debug["p_std"]),
                sigma_data=float(model.sigma_data),
            ),
            gamma=float(causal_debug["reference_wdro_gamma"]),
            step_size=float(causal_debug["reference_wdro_step_size"]),
            iters=int(causal_debug["reference_wdro_k"]),
            clamp_min=dataset.bounds_min.to(device),
            clamp_max=dataset.bounds_max.to(device),
        )
    adv_path = solve_causal_path_attack(
        attack_net=model,
        clean_points=clean_points,
        reference_path=reference_path,
        sigmas=sigmas,
        inner_steps=int(causal_debug["inner_steps"]),
        step_size=float(causal_debug["step_size"]),
        gamma=float(causal_debug["gamma"]),
        total_budget=total_budget,
        exact_budget_split=bool(causal_debug["exact_budget_split"]),
    )

    sigma_labels = torch.cat(
        [
            torch.zeros(1, device=device, dtype=clean_points.dtype),
            sigmas.mean(dim=0),
        ]
    ).cpu()
    forward_states = [clean_points.detach().cpu()] + [adv_path[:, step_idx, :].detach().cpu() for step_idx in range(adv_path.shape[1])]
    forward_indices = select_snapshot_indices(total_count=len(forward_states), num_snapshots=int(causal_debug["num_snapshots"]))
    forward_snapshots = [dataset.destandardize(forward_states[idx]).numpy() for idx in forward_indices]
    forward_titles = [
        f"step {idx}/{len(forward_states) - 1}\nσ={float(sigma_labels[idx]):.3f}" for idx in forward_indices
    ]

    _, reverse_states, reverse_sigmas = sample_edm_trajectory(
        model,
        data_dim=2,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        rho=rho,
        num_steps=sampler_steps,
        device=device,
        initial_points=adv_path[:, -1, :].detach(),
    )
    reverse_indices = select_snapshot_indices(total_count=len(reverse_states), num_snapshots=int(causal_debug["num_snapshots"]))
    reverse_snapshots = [dataset.destandardize(reverse_states[idx]).numpy() for idx in reverse_indices]
    reverse_titles = [
        f"step {idx}/{len(reverse_states) - 1}\nσ={float(reverse_sigmas[idx]):.3f}" for idx in reverse_indices
    ]

    save_process_snapshots(
        path=outdir / "plots" / f"causal_process_epoch_{epoch:04d}.png",
        rows=[
            {
                "row_title": "Noising",
                "snapshots": forward_snapshots,
                "titles": forward_titles,
                "color": "#d62728",
            },
            {
                "row_title": "Denoising",
                "snapshots": reverse_snapshots,
                "titles": reverse_titles,
                "color": "#2ca02c",
            },
        ],
        title=f"Causal WDRO debug | epoch {epoch}",
    )

    early_forward = [dataset.destandardize(state).numpy() for state in forward_states[: min(3, len(forward_states))]]
    save_process_snapshots(
        path=outdir / "plots" / f"causal_process_zoom_start_epoch_{epoch:04d}.png",
        rows=[
            {
                "row_title": "Noising",
                "snapshots": forward_snapshots,
                "titles": forward_titles,
                "color": "#d62728",
            },
            {
                "row_title": "Denoising",
                "snapshots": reverse_snapshots,
                "titles": reverse_titles,
                "color": "#2ca02c",
            },
        ],
        title=f"Causal WDRO debug (zoom near start) | epoch {epoch}",
        bounds=compute_plot_bounds(*early_forward),
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


def save_checkpoint(*, path: Path, model, args: argparse.Namespace, dataset, epoch: int, metrics: dict | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
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


def normalize_causal_total_budget(*, total_budget: float | None, budget_mode: str) -> float | None:
    if budget_mode != "fixed":
        return total_budget
    if total_budget is None:
        return None
    if float(total_budget) <= 0.0:
        return None
    return float(total_budget)


if __name__ == "__main__":
    main()
