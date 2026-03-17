from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from toy_2d.causal import build_forward_path, sample_path_sigmas, solve_causal_path_attack
from toy_2d.datasets import build_dataset
from toy_2d.losses import EDMLoss2D
from toy_2d.model import EDMPrecondMLP
from toy_2d.plotting import compute_plot_bounds, save_process_snapshots
from toy_2d.sampler import sample_edm_trajectory
from toy_2d.wild import estimate_wdro_transport_budget, wdro_attack


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export shared-scale process comparison plots.")
    parser.add_argument("--baseline-checkpoint", type=Path, default=None)
    parser.add_argument("--wdro-checkpoint", type=Path, default=None)
    parser.add_argument("--causal-checkpoint", type=Path, default=None)
    parser.add_argument("--baseline-run-dir", type=Path, default=None)
    parser.add_argument("--wdro-run-dir", type=Path, default=None)
    parser.add_argument("--causal-run-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-mode", type=str, default="last", choices=("best", "last", "fixed"))
    parser.add_argument("--fixed-epoch", type=int, default=None)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--num-points", type=int, default=256)
    parser.add_argument("--num-snapshots", type=int, default=5)
    parser.add_argument("--zoom-forward-steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu", choices=("cpu", "cuda", "auto"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)

    baseline_bundle = load_checkpoint_bundle(
        resolve_checkpoint_path(
            explicit_path=args.baseline_checkpoint,
            run_dir=args.baseline_run_dir,
            checkpoint_mode=args.checkpoint_mode,
            fixed_epoch=args.fixed_epoch,
        ),
        device=device,
    )
    wdro_bundle = load_checkpoint_bundle(
        resolve_checkpoint_path(
            explicit_path=args.wdro_checkpoint,
            run_dir=args.wdro_run_dir,
            checkpoint_mode=args.checkpoint_mode,
            fixed_epoch=args.fixed_epoch,
        ),
        device=device,
    )
    causal_bundle = load_checkpoint_bundle(
        resolve_checkpoint_path(
            explicit_path=args.causal_checkpoint,
            run_dir=args.causal_run_dir,
            checkpoint_mode=args.checkpoint_mode,
            fixed_epoch=args.fixed_epoch,
        ),
        device=device,
    )

    dataset = build_dataset(
        name=baseline_bundle["config"]["dataset"],
        num_samples=int(baseline_bundle["config"]["num_samples"]),
        seed=int(baseline_bundle["config"]["seed"]),
        noise=float(baseline_bundle["config"]["noise"]),
        standardize=bool(baseline_bundle["config"]["standardize"]),
    )

    clean_points = select_clean_points(
        dataset=dataset,
        num_points=args.num_points,
        seed=args.seed,
        device=device,
    )

    causal_cfg = causal_bundle["config"]
    sigmas = sample_path_sigmas(
        batch_size=clean_points.shape[0],
        num_steps=int(causal_cfg["causal_path_steps"]),
        p_mean=float(causal_cfg["p_mean"]),
        p_std=float(causal_cfg["p_std"]),
        sigma_min=float(causal_cfg["sigma_min"]),
        sigma_max=float(causal_cfg["sigma_max"]),
        rho=float(causal_cfg["rho"]),
        device=device,
        dtype=clean_points.dtype,
        schedule=str(causal_cfg["causal_sigma_schedule"]),
    )

    torch.manual_seed(args.seed)
    reference_path = build_forward_path(clean_points=clean_points, sigmas=sigmas, shared_noise=False)
    baseline_path = reference_path

    wdro_attack_points = wdro_attack(
        clean_points,
        wdro_bundle["model"],
        wdro_bundle["loss_fn"],
        gamma=float(wdro_bundle["config"]["wdro_gamma"]),
        step_size=float(wdro_bundle["config"]["wdro_step_size"]),
        iters=int(wdro_bundle["config"]["wdro_k"]),
        clamp_min=dataset.bounds_min.to(device),
        clamp_max=dataset.bounds_max.to(device),
    )
    wdro_path = reference_path + (wdro_attack_points - clean_points)[:, None, :]

    causal_path = solve_causal_path_attack(
        attack_net=causal_bundle["model"],
        clean_points=clean_points,
        reference_path=reference_path,
        sigmas=sigmas,
        inner_steps=int(causal_cfg["causal_inner_steps"]),
        step_size=float(causal_cfg["causal_step_size"]),
        gamma=float(causal_cfg["causal_gamma"]),
        total_budget=resolve_causal_budget(
            clean_points=clean_points,
            causal_bundle=causal_bundle,
            dataset=dataset,
        ),
        exact_budget_split=bool(causal_cfg.get("causal_exact_budget_split", False)),
    )

    baseline_reverse = sample_reverse(
        model=baseline_bundle["model"],
        initial_points=baseline_path[:, -1, :],
        config=baseline_bundle["config"],
        device=device,
    )
    wdro_reverse = sample_reverse(
        model=wdro_bundle["model"],
        initial_points=wdro_path[:, -1, :],
        config=wdro_bundle["config"],
        device=device,
    )
    causal_reverse = sample_reverse(
        model=causal_bundle["model"],
        initial_points=causal_path[:, -1, :],
        config=causal_bundle["config"],
        device=device,
    )

    sigma_labels = torch.cat(
        [torch.zeros(1, device=device, dtype=clean_points.dtype), sigmas.mean(dim=0)]
    ).cpu()

    forward_indices = select_snapshot_indices(total_count=baseline_path.shape[1] + 1, num_snapshots=args.num_snapshots)
    reverse_indices = select_snapshot_indices(
        total_count=len(baseline_reverse["trajectory"]),
        num_snapshots=args.num_snapshots,
    )

    rows = build_rows(
        dataset=dataset,
        clean_points=clean_points,
        wdro_start=wdro_attack_points,
        baseline_path=baseline_path,
        wdro_path=wdro_path,
        causal_path=causal_path,
        baseline_reverse=baseline_reverse,
        wdro_reverse=wdro_reverse,
        causal_reverse=causal_reverse,
        sigma_labels=sigma_labels,
        forward_indices=forward_indices,
        reverse_indices=reverse_indices,
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    save_process_snapshots(
        path=args.outdir / "process_comparison_full.png",
        rows=rows,
        title=f"{dataset.name} | Baseline vs WDRO vs Causal WDRO",
    )
    save_process_snapshots(
        path=args.outdir / "process_comparison_step_autofit.png",
        rows=rows,
        title=f"{dataset.name} | Baseline vs WDRO vs Causal WDRO (per-step auto-fit by stage)",
        bounds_mode="per_group_column",
    )

    early_forward_arrays = []
    early_forward_arrays.append(dataset.destandardize(clean_points.detach().cpu()).numpy())
    early_forward_arrays.append(dataset.destandardize(wdro_attack_points.detach().cpu()).numpy())
    for path_tensor in (baseline_path, wdro_path, causal_path):
        for step_idx in range(min(args.zoom_forward_steps, path_tensor.shape[1])):
            early_forward_arrays.append(dataset.destandardize(path_tensor[:, step_idx, :].detach().cpu()).numpy())
    save_process_snapshots(
        path=args.outdir / "process_comparison_zoom_start.png",
        rows=rows,
        title=f"{dataset.name} | Baseline vs WDRO vs Causal WDRO (zoom near start)",
        bounds=compute_plot_bounds(*early_forward_arrays),
    )


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def resolve_checkpoint_path(
    *,
    explicit_path: Path | None,
    run_dir: Path | None,
    checkpoint_mode: str,
    fixed_epoch: int | None,
) -> Path:
    if explicit_path is not None:
        return explicit_path
    if run_dir is None:
        raise ValueError("Provide either an explicit checkpoint path or a run directory.")
    if checkpoint_mode == "best":
        return run_dir / "checkpoint_best.pt"
    if checkpoint_mode == "last":
        return run_dir / "checkpoint_last.pt"
    if checkpoint_mode == "fixed":
        if fixed_epoch is None:
            raise ValueError("--fixed-epoch is required when --checkpoint-mode fixed.")
        return run_dir / "checkpoints" / f"checkpoint_epoch_{fixed_epoch:04d}.pt"
    raise ValueError(f"Unsupported checkpoint_mode: {checkpoint_mode}")


def load_checkpoint_bundle(path: Path, *, device: torch.device) -> dict:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = EDMPrecondMLP(
        data_dim=2,
        hidden_dim=int(config["hidden_dim"]),
        depth=int(config["depth"]),
        embedding_dim=int(config["embedding_dim"]),
        sigma_data=float(config["sigma_data"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    loss_fn = EDMLoss2D(
        p_mean=float(config["p_mean"]),
        p_std=float(config["p_std"]),
        sigma_data=float(config["sigma_data"]),
    )
    return {
        "checkpoint": checkpoint,
        "config": config,
        "model": model,
        "loss_fn": loss_fn,
    }


def select_clean_points(*, dataset, num_points: int, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator()
    generator.manual_seed(seed)
    count = min(int(num_points), int(dataset.train_points.shape[0]))
    indices = torch.randperm(dataset.train_points.shape[0], generator=generator)[:count]
    return dataset.train_points[indices].to(device)


def sample_reverse(*, model, initial_points: torch.Tensor, config: dict, device: torch.device) -> dict:
    _, trajectory, sigmas = sample_edm_trajectory(
        model,
        data_dim=2,
        sigma_min=float(config["sigma_min"]),
        sigma_max=float(config["sigma_max"]),
        rho=float(config["rho"]),
        num_steps=int(config["sampler_steps"]),
        device=device,
        initial_points=initial_points.detach(),
    )
    return {"trajectory": trajectory, "sigmas": sigmas}


def resolve_causal_budget(*, clean_points: torch.Tensor, causal_bundle: dict, dataset) -> float | None:
    config = causal_bundle["config"]
    budget_mode = str(config.get("causal_budget_mode", "fixed"))
    if budget_mode == "match_wdro":
        return estimate_wdro_transport_budget(
            clean_points,
            causal_bundle["model"],
            causal_bundle["loss_fn"],
            gamma=float(config.get("causal_reference_wdro_gamma", config["wdro_gamma"])),
            step_size=float(config.get("causal_reference_wdro_step_size", config["wdro_step_size"])),
            iters=int(config.get("causal_reference_wdro_k", config["wdro_k"])),
            clamp_min=dataset.bounds_min.to(clean_points.device),
            clamp_max=dataset.bounds_max.to(clean_points.device),
        )
    if config.get("causal_total_budget") is None:
        return None
    return float(config["causal_total_budget"])


def build_rows(
    *,
    dataset,
    clean_points: torch.Tensor,
    wdro_start: torch.Tensor,
    baseline_path: torch.Tensor,
    wdro_path: torch.Tensor,
    causal_path: torch.Tensor,
    baseline_reverse: dict,
    wdro_reverse: dict,
    causal_reverse: dict,
    sigma_labels: torch.Tensor,
    forward_indices: list[int],
    reverse_indices: list[int],
) -> list[dict]:
    method_paths = {
        "Baseline": baseline_path,
        "WDRO": wdro_path,
        "Causal WDRO": causal_path,
    }
    method_starts = {
        "Baseline": clean_points.detach().cpu(),
        "WDRO": wdro_start.detach().cpu(),
        "Causal WDRO": clean_points.detach().cpu(),
    }
    reverse_paths = {
        "Baseline": baseline_reverse,
        "WDRO": wdro_reverse,
        "Causal WDRO": causal_reverse,
    }
    colors = {
        "Baseline": "#1f77b4",
        "WDRO": "#ff7f0e",
        "Causal WDRO": "#d62728",
    }

    rows: list[dict] = []
    for method_name in ("Baseline", "WDRO", "Causal WDRO"):
        path_tensor = method_paths[method_name]
        forward_stack = [method_starts[method_name]] + [
            path_tensor[:, step_idx, :].detach().cpu() for step_idx in range(path_tensor.shape[1])
        ]
        forward_snapshots = [dataset.destandardize(forward_stack[idx]).numpy() for idx in forward_indices]
        forward_titles = [
            f"step {idx}/{len(forward_stack) - 1}\nσ={float(sigma_labels[idx]):.3f}" for idx in forward_indices
        ]
        rows.append(
            {
                "row_title": f"{method_name}\nNoising",
                "snapshots": forward_snapshots,
                "titles": forward_titles,
                "color": colors[method_name],
                "bounds_group": "noising",
            }
        )

        reverse_info = reverse_paths[method_name]
        reverse_snapshots = [
            dataset.destandardize(reverse_info["trajectory"][idx]).numpy() for idx in reverse_indices
        ]
        reverse_titles = [
            f"step {idx}/{len(reverse_info['trajectory']) - 1}\nσ={float(reverse_info['sigmas'][idx]):.3f}"
            for idx in reverse_indices
        ]
        rows.append(
            {
                "row_title": f"{method_name}\nDenoising",
                "snapshots": reverse_snapshots,
                "titles": reverse_titles,
                "color": "#2ca02c",
                "bounds_group": "denoising",
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


if __name__ == "__main__":
    main()
