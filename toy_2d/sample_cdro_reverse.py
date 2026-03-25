from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from toy_2d.datasets import build_dataset
from toy_2d.metrics import mmd_rbf, sliced_wasserstein
from toy_2d.plotting import save_process_snapshots, save_scatter_comparison
from toy_2d.sde_models import ReverseGaussianModel
from toy_2d.sde_reverse import sample_reverse_paths
from toy_2d.vp_sde import deserialize_vp_schedule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample from the learned toy CDRO reverse model.")
    parser.add_argument("--reverse-run-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-mode", type=str, default="best", choices=("best", "last"))
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=2048)
    parser.add_argument("--metric-samples", type=int, default=1024)
    parser.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-plot-paths", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    set_seed(args.seed)

    checkpoint_path = resolve_checkpoint_path(
        checkpoint=args.checkpoint,
        reverse_run_dir=args.reverse_run_dir,
        checkpoint_mode=args.checkpoint_mode,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    reverse_args = argparse.Namespace(**checkpoint["args"])
    forward_args = argparse.Namespace(**checkpoint["forward_args"])

    dataset = build_dataset(
        name=forward_args.dataset,
        num_samples=forward_args.num_samples,
        seed=forward_args.seed,
        noise=forward_args.noise,
        standardize=forward_args.standardize,
    )
    schedule = deserialize_vp_schedule(
        payload=checkpoint["schedule"],
        device=device,
        dtype=torch.float32,
    )
    model = ReverseGaussianModel(
        data_dim=int(checkpoint["dataset"]["mean"].shape[0]),
        hidden_dim=reverse_args.hidden_dim,
        time_embedding_dim=reverse_args.time_embedding_dim,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    args.outdir.mkdir(parents=True, exist_ok=True)
    generated_paths = sample_reverse_paths(
        model=model,
        schedule=schedule,
        num_samples=args.num_samples,
        device=device,
        dtype=torch.float32,
    ).cpu()
    generated_samples = dataset.destandardize(generated_paths[:, 0, :]).cpu()

    real_all = torch.from_numpy(dataset.raw_points)
    metric_count = min(args.metric_samples, args.num_samples, int(real_all.shape[0]))
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    real_idx = torch.randperm(real_all.shape[0], generator=generator)[:metric_count]
    fake_idx = torch.randperm(generated_samples.shape[0], generator=generator)[:metric_count]
    real_metric = real_all[real_idx]
    fake_metric = generated_samples[fake_idx]

    save_scatter_comparison(
        path=args.outdir / "samples.png",
        real_points=real_metric.numpy(),
        generated_points=fake_metric.numpy(),
        title=f"{dataset.name} | learned reverse samples",
    )
    save_process_snapshots(
        path=args.outdir / "reverse_process.png",
        rows=build_reverse_rows(
            dataset=dataset,
            generated_paths=generated_paths[: args.num_plot_paths],
        ),
        title=f"{dataset.name} | learned reverse process",
    )
    np.savez(
        args.outdir / "samples_latest.npz",
        generated=generated_samples.numpy(),
        generated_paths=generated_paths.numpy(),
    )

    summary = {
        "checkpoint": str(checkpoint_path),
        "dataset": forward_args.dataset,
        "num_samples": args.num_samples,
        "sliced_wasserstein": sliced_wasserstein(real_metric, fake_metric, seed=args.seed),
        "mmd_rbf": mmd_rbf(real_metric, fake_metric),
    }
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def resolve_checkpoint_path(
    *,
    checkpoint: Path | None,
    reverse_run_dir: Path | None,
    checkpoint_mode: str,
) -> Path:
    if checkpoint is not None:
        return checkpoint
    if reverse_run_dir is None:
        raise ValueError("Provide either --checkpoint or --reverse-run-dir.")
    if checkpoint_mode == "best":
        return reverse_run_dir / "checkpoint_best.pt"
    return reverse_run_dir / "checkpoint_last.pt"


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


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
