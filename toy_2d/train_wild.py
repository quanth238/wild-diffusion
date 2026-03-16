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
from toy_2d.causal import causal_wdro_loss
from toy_2d.losses import EDMLoss2D
from toy_2d.metrics import mmd_rbf, sliced_wasserstein
from toy_2d.model import EDMPrecondMLP
from toy_2d.plotting import save_adversarial_debug, save_scatter_comparison, save_training_curves
from toy_2d.robust_defaults import CAUSAL_WDRO_DEFAULTS, WDRO_CORE_DEFAULTS
from toy_2d.sampler import sample_edm
from toy_2d.wild import build_wdro_dataset


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
    parser.add_argument("--causal-inner-steps", type=int, default=CAUSAL_WDRO_DEFAULTS["inner_steps"])
    parser.add_argument("--causal-step-size", type=float, default=CAUSAL_WDRO_DEFAULTS["step_size"])
    parser.add_argument("--causal-gamma", type=float, default=CAUSAL_WDRO_DEFAULTS["gamma"])
    parser.add_argument(
        "--causal-sigma-schedule",
        type=str,
        default=CAUSAL_WDRO_DEFAULTS["sigma_schedule"],
        choices=("edm_random", "edm_quantiles", "karras_grid"),
    )

    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--num-eval-samples", type=int, default=2048)
    parser.add_argument("--metric-samples", type=int, default=1024)
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
            causal_kwargs={
                "sigma_min": args.sigma_min,
                "sigma_max": args.sigma_max,
                "rho": args.rho,
                "p_mean": args.p_mean,
                "p_std": args.p_std,
                "path_steps": args.causal_path_steps,
                "inner_steps": args.causal_inner_steps,
                "step_size": args.causal_step_size,
                "gamma": args.causal_gamma,
                "sigma_schedule": args.causal_sigma_schedule,
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
                metrics["mean_l2_shift"] = latest_causal_stats["mean_l2_shift"]
                metrics["max_l2_shift"] = latest_causal_stats["max_l2_shift"]
                metrics["mean_transport_cost"] = latest_causal_stats["mean_transport_cost"]
                metrics["max_transport_cost"] = latest_causal_stats["max_transport_cost"]
                metrics["total_transport_cost"] = latest_causal_stats["mean_total_transport_cost"]
                metrics["max_total_transport_cost"] = latest_causal_stats["max_total_transport_cost"]
            metrics["method"] = args.method
            eval_history.append(metrics)
            append_jsonl(args.outdir / "metrics.jsonl", metrics)

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
    for (batch,) in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        if method == "causal_wdro":
            loss, stats = causal_wdro_loss(net=model, clean_points=batch, **causal_kwargs)
            causal_mean_norms.append(stats.mean_delta_norm)
            causal_max_norms.append(stats.max_delta_norm)
            causal_mean_costs.append(stats.mean_transport_cost)
            causal_max_costs.append(stats.max_transport_cost)
            causal_mean_total_costs.append(stats.mean_total_transport_cost)
            causal_max_total_costs.append(stats.max_total_transport_cost)
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
    return metrics


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


if __name__ == "__main__":
    main()
