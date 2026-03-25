from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from toy_2d.datasets import build_dataset
from toy_2d.metrics import mmd_rbf, sliced_wasserstein
from toy_2d.plotting import (
    save_cdro_sde_reverse_curves,
    save_process_snapshots,
    save_scatter_comparison,
)
from toy_2d.sde_models import ReverseGaussianModel
from toy_2d.sde_reverse import compute_reverse_teacher_forcing_loss, sample_reverse_paths
from toy_2d.vp_sde import deserialize_vp_schedule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the reverse sampler for the toy CDRO VP-SDE model.")
    parser.add_argument("--forward-run-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--robust-buffer", type=Path, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--val-fraction", type=float, default=0.1)

    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--time-embedding-dim", type=int, default=16)

    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--num-eval-samples", type=int, default=2048)
    parser.add_argument("--metric-samples", type=int, default=1024)
    parser.add_argument("--num-plot-paths", type=int, default=64)
    parser.add_argument("--save-eval-checkpoints", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    set_seed(args.seed)

    args.outdir = args.outdir or args.forward_run_dir / "reverse_model"
    args.robust_buffer = args.robust_buffer or args.forward_run_dir / "robust_buffer.pt"
    args.outdir.mkdir(parents=True, exist_ok=True)
    save_json(args.outdir / "config.json", vars(args))

    forward_checkpoint = torch.load(args.forward_run_dir / "checkpoint_last.pt", map_location="cpu", weights_only=False)
    forward_args = argparse.Namespace(**forward_checkpoint["args"])
    dataset = build_dataset(
        name=forward_args.dataset,
        num_samples=forward_args.num_samples,
        seed=forward_args.seed,
        noise=forward_args.noise,
        standardize=forward_args.standardize,
    )
    schedule = deserialize_vp_schedule(
        payload=forward_checkpoint["schedule"],
        device=device,
        dtype=torch.float32,
    )

    buffer_payload = torch.load(args.robust_buffer, map_location="cpu", weights_only=False)
    robust_paths = buffer_payload["adv_paths"].to(torch.float32)
    train_paths, val_paths = split_paths(paths=robust_paths, val_fraction=args.val_fraction, seed=args.seed)

    model = ReverseGaussianModel(
        data_dim=int(robust_paths.shape[2]),
        hidden_dim=args.hidden_dim,
        time_embedding_dim=args.time_embedding_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loader = build_loader(paths=train_paths, batch_size=args.batch_size, seed=args.seed)
    val_loader = build_loader(paths=val_paths, batch_size=args.batch_size, seed=args.seed + 17, shuffle=False)

    history: list[dict] = []
    best_swd = float("inf")
    best_epoch: int | None = None
    best_eval: dict | None = None
    start_time = time.time()

    for epoch_idx in range(args.epochs):
        epoch = epoch_idx + 1
        train_losses: list[float] = []

        model.train()
        for (batch_paths,) in train_loader:
            batch_paths = batch_paths.to(device)
            optimizer.zero_grad(set_to_none=True)
            total_loss, _ = compute_reverse_teacher_forcing_loss(
                model=model,
                paths=batch_paths,
                schedule=schedule,
            )
            loss = total_loss.mean()
            loss.backward()
            clip_gradients(model.parameters(), args.grad_clip)
            optimizer.step()
            train_losses.append(float(loss.item()))

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)) if train_losses else float("nan"),
        }

        if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs:
            eval_metrics = evaluate_reverse_epoch(
                dataset=dataset,
                model=model,
                schedule=schedule,
                val_loader=val_loader,
                outdir=args.outdir,
                epoch=epoch,
                device=device,
                num_eval_samples=args.num_eval_samples,
                metric_samples=args.metric_samples,
                num_plot_paths=args.num_plot_paths,
                seed=args.seed,
            )
            epoch_metrics.update(eval_metrics)
            history.append(dict(epoch_metrics))
            append_jsonl(args.outdir / "metrics.jsonl", epoch_metrics)
            save_cdro_sde_reverse_curves(
                path=args.outdir / "plots" / "training_curves.png",
                history=history,
            )

            if epoch_metrics["sliced_wasserstein"] < best_swd:
                best_swd = epoch_metrics["sliced_wasserstein"]
                best_epoch = epoch
                best_eval = dict(epoch_metrics)
                save_checkpoint(
                    path=args.outdir / "checkpoint_best.pt",
                    model=model,
                    args=args,
                    forward_args=forward_args,
                    dataset=forward_checkpoint["dataset"],
                    schedule=forward_checkpoint["schedule"],
                    epoch=epoch,
                    metrics=epoch_metrics,
                )

            if args.save_eval_checkpoints:
                save_checkpoint(
                    path=args.outdir / "checkpoints" / f"checkpoint_epoch_{epoch:04d}.pt",
                    model=model,
                    args=args,
                    forward_args=forward_args,
                    dataset=forward_checkpoint["dataset"],
                    schedule=forward_checkpoint["schedule"],
                    epoch=epoch,
                    metrics=epoch_metrics,
                )

    final_summary = {
        "dataset": forward_args.dataset,
        "epochs": args.epochs,
        "runtime_minutes": (time.time() - start_time) / 60.0,
        "best_sliced_wasserstein": best_swd,
        "best_epoch": best_epoch,
        "best_eval": best_eval,
        "last_eval": history[-1] if history else None,
    }
    save_json(args.outdir / "summary.json", final_summary)
    save_checkpoint(
        path=args.outdir / "checkpoint_last.pt",
        model=model,
        args=args,
        forward_args=forward_args,
        dataset=forward_checkpoint["dataset"],
        schedule=forward_checkpoint["schedule"],
        epoch=args.epochs,
        metrics=history[-1] if history else None,
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


def split_paths(*, paths: torch.Tensor, val_fraction: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    total = int(paths.shape[0])
    if total < 2 or val_fraction <= 0.0:
        return paths, paths
    generator = torch.Generator()
    generator.manual_seed(seed)
    indices = torch.randperm(total, generator=generator)
    val_count = min(max(int(round(total * val_fraction)), 1), total - 1)
    val_idx = indices[:val_count]
    train_idx = indices[val_count:]
    return paths[train_idx], paths[val_idx]


def build_loader(*, paths: torch.Tensor, batch_size: int, seed: int, shuffle: bool = True) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(TensorDataset(paths), batch_size=batch_size, shuffle=shuffle, generator=generator, drop_last=False)


@torch.no_grad()
def evaluate_reverse_epoch(
    *,
    dataset,
    model: ReverseGaussianModel,
    schedule,
    val_loader: DataLoader,
    outdir: Path,
    epoch: int,
    device: torch.device,
    num_eval_samples: int,
    metric_samples: int,
    num_plot_paths: int,
    seed: int,
) -> dict:
    model.eval()

    val_losses: list[float] = []
    terminal_nll_values: list[float] = []
    transition_nll_values: list[float] = []
    for (batch_paths,) in val_loader:
        batch_paths = batch_paths.to(device)
        total_loss, metrics = compute_reverse_teacher_forcing_loss(
            model=model,
            paths=batch_paths,
            schedule=schedule,
        )
        val_losses.append(float(total_loss.mean().item()))
        terminal_nll_values.append(metrics["terminal_nll"])
        transition_nll_values.append(metrics["transition_nll"])

    generated_paths = sample_reverse_paths(
        model=model,
        schedule=schedule,
        num_samples=num_eval_samples,
        device=device,
        dtype=torch.float32,
    ).cpu()
    generated_samples = dataset.destandardize(generated_paths[:, 0, :]).cpu()

    real_all = torch.from_numpy(dataset.raw_points)
    metric_count = min(metric_samples, num_eval_samples, int(real_all.shape[0]))
    generator = torch.Generator()
    generator.manual_seed(seed + epoch)
    real_idx = torch.randperm(real_all.shape[0], generator=generator)[:metric_count]
    fake_idx = torch.randperm(generated_samples.shape[0], generator=generator)[:metric_count]
    real_metric = real_all[real_idx]
    fake_metric = generated_samples[fake_idx]

    save_scatter_comparison(
        path=outdir / "plots" / f"samples_epoch_{epoch:04d}.png",
        real_points=real_metric.numpy(),
        generated_points=fake_metric.numpy(),
        title=f"{dataset.name} | reverse samples | epoch {epoch}",
    )
    np.savez(
        outdir / "samples_latest.npz",
        generated=generated_samples.numpy(),
        generated_paths=generated_paths.numpy(),
    )
    save_process_snapshots(
        path=outdir / "plots" / f"reverse_process_epoch_{epoch:04d}.png",
        rows=build_reverse_rows(
            dataset=dataset,
            generated_paths=generated_paths[:num_plot_paths],
        ),
        title=f"{dataset.name} | learned reverse process | epoch {epoch}",
    )

    return {
        "val_loss": float(np.mean(val_losses)) if val_losses else float("nan"),
        "terminal_nll": float(np.mean(terminal_nll_values)) if terminal_nll_values else float("nan"),
        "transition_nll": float(np.mean(transition_nll_values)) if transition_nll_values else float("nan"),
        "sliced_wasserstein": sliced_wasserstein(real_metric, fake_metric, seed=seed + epoch),
        "mmd_rbf": mmd_rbf(real_metric, fake_metric),
    }


def build_reverse_rows(
    *,
    dataset,
    generated_paths: torch.Tensor,
) -> list[dict]:
    snapshot_indices = select_snapshot_indices(total_count=generated_paths.shape[1], num_snapshots=min(5, generated_paths.shape[1]))
    ordered_indices = list(reversed(snapshot_indices))
    snapshots = [dataset.destandardize(generated_paths[:, idx, :]).numpy() for idx in ordered_indices]
    titles = [f"state {idx}" for idx in ordered_indices]
    return [
        {
            "row_title": "Reverse samples",
            "snapshots": snapshots,
            "titles": titles,
            "color": "#2ca02c",
        }
    ]


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


def clip_gradients(parameters, max_norm: float) -> None:
    if max_norm <= 0:
        return
    torch.nn.utils.clip_grad_norm_(list(parameters), max_norm)


def save_checkpoint(
    *,
    path: Path,
    model: ReverseGaussianModel,
    args: argparse.Namespace,
    forward_args: argparse.Namespace,
    dataset: dict,
    schedule: dict,
    epoch: int,
    metrics: dict | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "args": vars(args),
            "forward_args": vars(forward_args),
            "dataset": dataset,
            "schedule": schedule,
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
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return float(value.item())
        return value.tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


if __name__ == "__main__":
    main()
