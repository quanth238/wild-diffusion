from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from toy_2d.cdro import CdroConfig, solve_cdro_attack
from toy_2d.cdro_markov import (
    MarkovControlGRU,
    MarkovControlMLP,
    MarkovPrecondScoreMLP,
    MarkovScoreMLP,
    MarkovVPSchedule,
    TerminalStats,
    rollout_markov_forward,
    sample_reverse_chain,
)
from toy_2d.datasets import build_dataset
from toy_2d.export_process_comparison import (
    load_checkpoint_bundle as load_edm_checkpoint_bundle,
    resolve_cdro_budget,
    resolve_checkpoint_path,
    sample_reverse as sample_edm_reverse,
    select_clean_points,
    select_snapshot_indices,
)
from toy_2d.plotting import save_process_snapshots
from toy_2d.wild import wdro_attack


ALL_METHODS = ["baseline", "wdro", "cdro", "baseline_score", "wdro_score", "cdro_markov"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export true noising/denoising process comparisons for all methods.")
    parser.add_argument("--comparison-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--fraction-tags", nargs="*", default=["100pct"])
    parser.add_argument("--methods", nargs="+", default=ALL_METHODS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-mode", choices=("best", "last"), default="best")
    parser.add_argument("--out-subdir", type=str, default="process_comparisons")
    parser.add_argument("--num-points", type=int, default=256)
    parser.add_argument("--num-snapshots", type=int, default=5)
    parser.add_argument("--device", type=str, default="auto", choices=("cpu", "cuda", "auto"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    datasets = args.datasets or discover_datasets(args.comparison_dir, args.methods, args.seed)
    for dataset_name in datasets:
        for fraction_tag in args.fraction_tags:
            export_dataset_fraction(
                comparison_dir=args.comparison_dir,
                dataset_name=dataset_name,
                fraction_tag=fraction_tag,
                methods=args.methods,
                seed=args.seed,
                checkpoint_mode=args.checkpoint_mode,
                out_subdir=args.out_subdir,
                num_points=args.num_points,
                num_snapshots=args.num_snapshots,
                device=device,
            )


def export_dataset_fraction(
    *,
    comparison_dir: Path,
    dataset_name: str,
    fraction_tag: str,
    methods: list[str],
    seed: int,
    checkpoint_mode: str,
    out_subdir: str,
    num_points: int,
    num_snapshots: int,
    device: torch.device,
) -> None:
    run_dirs = {
        method: comparison_dir / dataset_name / fraction_tag / method / f"seed{seed}"
        for method in methods
    }
    existing = {method: path for method, path in run_dirs.items() if path.is_dir()}
    if not existing:
        return

    reference_run = next(iter(existing.values()))
    dataset = build_dataset_from_run(reference_run)
    clean_points = select_clean_points(dataset=dataset, num_points=num_points, seed=seed, device=device)

    noising_rows: list[dict] = []
    denoising_rows: list[dict] = []

    edm_rows = build_edm_rows(
        run_dirs=existing,
        clean_points=clean_points,
        dataset=dataset,
        seed=seed,
        checkpoint_mode=checkpoint_mode,
        num_snapshots=num_snapshots,
        device=device,
    )
    noising_rows.extend(edm_rows["noising"])
    denoising_rows.extend(edm_rows["denoising"])

    score_rows = build_markov_rows(
        run_dirs=existing,
        clean_points=clean_points,
        dataset=dataset,
        seed=seed,
        checkpoint_mode=checkpoint_mode,
        num_snapshots=num_snapshots,
        device=device,
    )
    noising_rows.extend(score_rows["noising"])
    denoising_rows.extend(score_rows["denoising"])

    if not noising_rows or not denoising_rows:
        return

    outdir = comparison_dir / out_subdir / dataset_name
    outdir.mkdir(parents=True, exist_ok=True)
    title_base = f"{dataset_name} | {fraction_tag} | seed {seed} | {checkpoint_mode}"
    save_process_snapshots(
        path=outdir / f"{fraction_tag}_noising_process_{checkpoint_mode}_seed{seed}.png",
        rows=noising_rows,
        title=f"{title_base} | Noising Process",
        bounds_mode="per_column",
    )
    save_process_snapshots(
        path=outdir / f"{fraction_tag}_denoising_{checkpoint_mode}_seed{seed}.png",
        rows=denoising_rows,
        title=f"{title_base} | Denoising",
        bounds_mode="per_column",
    )


def build_dataset_from_run(run_dir: Path):
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    return build_dataset(
        name=config["dataset"],
        num_samples=int(config["num_samples"]),
        seed=int(config["seed"]),
        noise=float(config.get("noise", 0.08)),
        standardize=bool(config.get("standardize", True)),
    )


def build_edm_rows(
    *,
    run_dirs: dict[str, Path],
    clean_points: torch.Tensor,
    dataset,
    seed: int,
    checkpoint_mode: str,
    num_snapshots: int,
    device: torch.device,
) -> dict[str, list[dict]]:
    rows = {"noising": [], "denoising": []}
    edm_methods = [name for name in ("baseline", "wdro", "cdro") if name in run_dirs]
    if not edm_methods:
        return rows

    baseline_bundle = load_edm_bundle_if_present(run_dirs.get("baseline"), checkpoint_mode, device=device)
    wdro_bundle = load_edm_bundle_if_present(run_dirs.get("wdro"), checkpoint_mode, device=device)
    cdro_bundle = load_edm_bundle_if_present(run_dirs.get("cdro"), checkpoint_mode, device=device)
    if cdro_bundle is None:
        return rows

    torch.manual_seed(seed)
    cdro_cfg = CdroConfig.from_dict(cdro_bundle["config"])
    cdro_result = solve_cdro_attack(
        attack_net=cdro_bundle["model"],
        clean_points=clean_points,
        config=cdro_cfg.with_total_budget(
            resolve_cdro_budget(clean_points=clean_points, cdro_bundle=cdro_bundle, dataset=dataset)
        ),
        shared_noise=False,
    )
    reference_path = cdro_result.reference_path
    sigma_labels = torch.cat(
        [torch.zeros(1, device=device, dtype=clean_points.dtype), cdro_result.sigmas.mean(dim=0)]
    ).cpu()

    method_forward_paths: dict[str, torch.Tensor] = {}
    if baseline_bundle is not None:
        method_forward_paths["baseline"] = reference_path
    if wdro_bundle is not None:
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
        method_forward_paths["wdro"] = reference_path + (wdro_attack_points - clean_points)[:, None, :]
    method_forward_paths["cdro"] = cdro_result.adv_path

    for method in ("baseline", "wdro", "cdro"):
        if method not in method_forward_paths:
            continue
        path_tensor = method_forward_paths[method]
        forward_states = [clean_points.detach().cpu()] + [path_tensor[:, idx, :].detach().cpu() for idx in range(path_tensor.shape[1])]
        forward_indices = select_snapshot_indices(total_count=len(forward_states), num_snapshots=num_snapshots)
        forward_snapshots = [dataset.destandardize(forward_states[idx]).numpy() for idx in forward_indices]
        forward_titles = [f"step {idx}/{len(forward_states) - 1}\nσ={float(sigma_labels[idx]):.3f}" for idx in forward_indices]
        rows["noising"].append(
            {
                "row_title": pretty_method_name(method),
                "snapshots": forward_snapshots,
                "titles": forward_titles,
                "color": method_color(method),
            }
        )

        bundle = {"baseline": baseline_bundle, "wdro": wdro_bundle, "cdro": cdro_bundle}[method]
        if bundle is None:
            continue
        reverse = sample_edm_reverse(
            model=bundle["model"],
            initial_points=path_tensor[:, -1, :],
            config=bundle["config"],
            device=device,
        )
        reverse_indices = select_snapshot_indices(total_count=len(reverse["trajectory"]), num_snapshots=num_snapshots)
        reverse_snapshots = [dataset.destandardize(reverse["trajectory"][idx]).numpy() for idx in reverse_indices]
        reverse_titles = [
            f"step {idx}/{len(reverse['trajectory']) - 1}\nσ={float(reverse['sigmas'][idx]):.3f}"
            for idx in reverse_indices
        ]
        rows["denoising"].append(
            {
                "row_title": pretty_method_name(method),
                "snapshots": reverse_snapshots,
                "titles": reverse_titles,
                "color": method_color(method),
            }
        )
    return rows


def build_markov_rows(
    *,
    run_dirs: dict[str, Path],
    clean_points: torch.Tensor,
    dataset,
    seed: int,
    checkpoint_mode: str,
    num_snapshots: int,
    device: torch.device,
) -> dict[str, list[dict]]:
    del dataset
    rows = {"noising": [], "denoising": []}
    methods = [name for name in ("baseline_score", "wdro_score", "cdro_markov") if name in run_dirs]
    for method in methods:
        bundle = load_markov_bundle(
            run_dir=run_dirs[method],
            checkpoint_mode=checkpoint_mode,
            device=device,
        )
        dataset_local = build_dataset_from_run(run_dirs[method])
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        noise_override = torch.randn(
            clean_points.shape[0],
            bundle["schedule"].dt.shape[0],
            clean_points.shape[1],
            generator=generator,
            device=device,
            dtype=clean_points.dtype,
        )
        zero_control = method != "cdro_markov"
        rollout = rollout_markov_forward(
            clean_points=clean_points,
            control_net=bundle["control_model"],
            schedule=bundle["schedule"],
            zero_control=zero_control,
            noise_override=noise_override,
        )
        sigma_labels = torch.cat(
            [
                torch.zeros(1, dtype=clean_points.dtype).cpu(),
                bundle["schedule"].step_sigma.detach().cpu(),
            ]
        )
        forward_states = [rollout.states[:, idx, :].detach().cpu() for idx in range(rollout.states.shape[1])]
        forward_indices = select_snapshot_indices(total_count=len(forward_states), num_snapshots=num_snapshots)
        forward_snapshots = [dataset_local.destandardize(forward_states[idx]).numpy() for idx in forward_indices]
        forward_titles = [f"step {idx}/{len(forward_states) - 1}\nσ={float(sigma_labels[idx]):.3f}" for idx in forward_indices]
        rows["noising"].append(
            {
                "row_title": pretty_method_name(method),
                "snapshots": forward_snapshots,
                "titles": forward_titles,
                "color": method_color(method),
            }
        )

        reverse_samples, reverse_states = sample_reverse_chain(
            score_net=bundle["score_model"],
            control_net=bundle["control_model"],
            schedule=bundle["schedule"],
            terminal_stats=bundle["terminal_stats"],
            num_samples=clean_points.shape[0],
            data_dim=clean_points.shape[1],
            device=device,
            zero_control=zero_control,
            collect_states=True,
            solver=bundle["config"].get("reverse_solver", "euler"),
            terminal_sampler=bundle["config"].get("terminal_sampler", "gaussian"),
            terminal_jitter_scale=float(bundle["config"].get("terminal_jitter_scale", 0.0)),
            reverse_noise_scale=float(bundle["config"].get("reverse_noise_scale", 1.0)),
            reverse_control_scale=float(bundle["config"].get("reverse_control_scale", 1.0)),
            reverse_tail_noise_scale=float(bundle["config"].get("reverse_tail_noise_scale", 1.0)),
            reverse_deterministic_tail_steps=int(bundle["config"].get("reverse_deterministic_tail_steps", 0)),
        )
        del reverse_samples
        reverse_indices = select_snapshot_indices(total_count=len(reverse_states), num_snapshots=num_snapshots)
        reverse_snapshots = [dataset_local.destandardize(reverse_states[idx]).numpy() for idx in reverse_indices]
        reverse_step_sigmas = torch.flip(bundle["schedule"].step_sigma.detach().cpu(), dims=[0])
        reverse_titles = []
        for idx in reverse_indices:
            sigma_value = 0.0
            if idx < len(reverse_states) - 1 and idx < reverse_step_sigmas.shape[0]:
                sigma_value = float(reverse_step_sigmas[idx])
            reverse_titles.append(f"step {idx}/{len(reverse_states) - 1}\nσ={sigma_value:.3f}")
        rows["denoising"].append(
            {
                "row_title": pretty_method_name(method),
                "snapshots": reverse_snapshots,
                "titles": reverse_titles,
                "color": method_color(method),
            }
        )
    return rows


def load_edm_bundle_if_present(run_dir: Path | None, checkpoint_mode: str, *, device: torch.device):
    if run_dir is None or not run_dir.is_dir():
        return None
    checkpoint_path = resolve_checkpoint_path(
        explicit_path=None,
        run_dir=run_dir,
        checkpoint_mode=checkpoint_mode,
        fixed_epoch=None,
    )
    return load_edm_checkpoint_bundle(checkpoint_path, device=device)


def load_markov_bundle(*, run_dir: Path, checkpoint_mode: str, device: torch.device) -> dict:
    checkpoint_path = resolve_checkpoint_path(
        explicit_path=None,
        run_dir=run_dir,
        checkpoint_mode=checkpoint_mode,
        fixed_epoch=None,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    schedule = schedule_from_dict(checkpoint["schedule"], device=device)
    score_model = build_markov_score_model(config=config, device=device)
    score_model.load_state_dict(checkpoint["score_model_state"])
    score_model.eval()

    control_model = None
    if checkpoint.get("control_model_state") is not None:
        control_model = build_markov_control_model(config=config, device=device)
        control_model.load_state_dict(checkpoint["control_model_state"])
        control_model.eval()

    terminal = checkpoint["terminal_stats"]
    terminal_stats = TerminalStats(
        mean=terminal["mean"].to(device=device),
        var=terminal["var"].to(device=device),
        replay=(terminal["replay"] if terminal.get("replay") is not None else None),
    )
    return {
        "config": config,
        "schedule": schedule,
        "score_model": score_model,
        "control_model": control_model,
        "terminal_stats": terminal_stats,
    }


def schedule_from_dict(payload: dict, *, device: torch.device) -> MarkovVPSchedule:
    def to_tensor(key: str):
        value = payload.get(key)
        if value is None:
            return None
        return torch.tensor(value, device=device, dtype=torch.float32)

    return MarkovVPSchedule(
        family=str(payload["family"]),
        total_time=float(payload["total_time"]),
        dt=to_tensor("dt"),
        drift_coeff=to_tensor("drift_coeff"),
        beta=to_tensor("beta"),
        g=to_tensor("g"),
        step_sigma=to_tensor("step_sigma"),
        weights=to_tensor("weights"),
        sigma_levels=to_tensor("sigma_levels"),
    )


def build_markov_score_model(*, config: dict, device: torch.device):
    kwargs = {
        "data_dim": 2,
        "hidden_dim": int(config["score_hidden_dim"]),
        "depth": int(config["score_depth"]),
        "embedding_dim": int(config["embedding_dim"]),
    }
    if config.get("score_arch", "raw") == "precond":
        model = MarkovPrecondScoreMLP(**kwargs, sigma_data=float(config.get("sigma_data", 0.5)))
    else:
        model = MarkovScoreMLP(**kwargs)
    return model.to(device)


def build_markov_control_model(*, config: dict, device: torch.device):
    kwargs = {
        "data_dim": 2,
        "hidden_dim": int(config["control_hidden_dim"]),
        "depth": int(config["control_depth"]),
        "embedding_dim": int(config["embedding_dim"]),
        "control_scale": float(config["control_scale"]),
    }
    if config.get("control_arch", "mlp") == "gru":
        model = MarkovControlGRU(**kwargs)
    else:
        model = MarkovControlMLP(**kwargs)
    return model.to(device)


def pretty_method_name(method: str) -> str:
    mapping = {
        "baseline": "Baseline",
        "wdro": "WDRO-EDM",
        "cdro": "Legacy CDRO",
        "baseline_score": "Baseline Score",
        "wdro_score": "WDRO Score",
        "cdro_markov": "CDRO Markov",
    }
    return mapping.get(method, method.replace("_", " ").title())


def method_color(method: str) -> str:
    mapping = {
        "baseline": "#1f77b4",
        "wdro": "#ff7f0e",
        "cdro": "#d62728",
        "baseline_score": "#17becf",
        "wdro_score": "#bcbd22",
        "cdro_markov": "#2ca02c",
    }
    return mapping.get(method, "#444444")


def discover_datasets(comparison_dir: Path, methods: list[str], seed: int) -> list[str]:
    datasets: list[str] = []
    for path in sorted(comparison_dir.iterdir()):
        if not path.is_dir():
            continue
        found = False
        for fraction_dir in sorted(path.iterdir()):
            if not fraction_dir.is_dir():
                continue
            if any((fraction_dir / method / f"seed{seed}" / "summary.json").is_file() for method in methods):
                found = True
                break
        if found:
            datasets.append(path.name)
    return datasets


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


if __name__ == "__main__":
    main()
