from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from toy_2d.cdro_markov import MarkovVPSchedule, TerminalStats
from toy_2d.datasets import build_dataset
from toy_2d.train_cdro_markov import (
    build_control_model,
    build_score_model,
    evaluate_model,
    resolve_device,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-evaluate a saved Markov CDRO checkpoint without retraining.")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-mode", choices=("best", "last"), default="best")
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=("cpu", "cuda", "auto"))
    parser.add_argument("--seed", type=int, default=None, help="Evaluation seed. Defaults to the training seed.")

    parser.add_argument("--num-eval-samples", type=int, default=None)
    parser.add_argument("--metric-samples", type=int, default=None)
    parser.add_argument("--num-snapshot-steps", type=int, default=None)
    parser.add_argument("--diagnostic-batch-size", type=int, default=None)

    parser.add_argument("--reverse-solver", choices=("euler", "heun"), default=None)
    parser.add_argument("--terminal-sampler", choices=("gaussian", "replay"), default=None)
    parser.add_argument("--terminal-jitter-scale", type=float, default=None)
    parser.add_argument("--reverse-noise-scale", type=float, default=None)
    parser.add_argument("--reverse-control-scale", type=float, default=None)
    parser.add_argument("--reverse-tail-noise-scale", type=float, default=None)
    parser.add_argument("--reverse-deterministic-tail-steps", type=int, default=None)
    parser.add_argument(
        "--log-reverse-ablation",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--fast-tuning",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip plot/sample exports and only compute metrics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = resolve_checkpoint_path(
        checkpoint=args.checkpoint,
        run_dir=args.run_dir,
        checkpoint_mode=args.checkpoint_mode,
    )
    outdir = resolve_outdir(
        outdir=args.outdir,
        run_dir=args.run_dir,
        checkpoint_path=checkpoint_path,
        checkpoint_mode=args.checkpoint_mode,
    )
    outdir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = dict(checkpoint["config"])
    config_ns = argparse.Namespace(**config)
    eval_seed = int(args.seed) if args.seed is not None else int(config["seed"])
    set_seed(eval_seed)

    dataset = build_dataset(
        name=str(config["dataset"]),
        num_samples=int(config["num_samples"]),
        seed=int(config["seed"]),
        noise=float(config.get("noise", 0.08)),
        standardize=bool(config.get("standardize", True)),
    )
    schedule = schedule_from_dict(checkpoint["schedule"], device=device)

    score_model = build_score_model(args=config_ns, device=device)
    score_model.load_state_dict(checkpoint["score_model_state"])
    score_model.eval()

    control_model = None
    if checkpoint.get("control_model_state") is not None:
        control_model = build_control_model(args=config_ns, device=device)
        control_model.load_state_dict(checkpoint["control_model_state"])
        control_model.eval()

    terminal = checkpoint["terminal_stats"]
    terminal_stats = TerminalStats(
        mean=terminal["mean"].to(device=device, dtype=torch.float32),
        var=terminal["var"].to(device=device, dtype=torch.float32),
        replay=(
            terminal["replay"].to(device=device, dtype=torch.float32)
            if terminal.get("replay") is not None
            else None
        ),
    )

    diagnostic_batch_size = min(
        max(resolve_int_override(args.diagnostic_batch_size, config.get("diagnostic_batch_size"), 256), 1),
        int(dataset.train_points.shape[0]),
    )
    diagnostic_points = dataset.train_points[:diagnostic_batch_size].to(device)
    diagnostic_noise = torch.randn(
        diagnostic_batch_size,
        int(schedule.dt.shape[0]),
        dataset.train_points.shape[1],
        generator=torch.Generator().manual_seed(int(config["seed"]) + 12345),
        dtype=dataset.train_points.dtype,
    ).to(device)

    method_name = str(config.get("method", "cdro_markov"))
    zero_control = control_model is None or method_name != "cdro_markov"

    resolved_eval_config = {
        "num_eval_samples": resolve_int_override(args.num_eval_samples, config.get("num_eval_samples"), 2048),
        "metric_samples": resolve_int_override(args.metric_samples, config.get("metric_samples"), 1024),
        "num_snapshot_steps": resolve_int_override(args.num_snapshot_steps, config.get("num_snapshot_steps"), 6),
        "reverse_solver": resolve_choice_override(args.reverse_solver, config.get("reverse_solver"), "heun"),
        "terminal_sampler": resolve_choice_override(args.terminal_sampler, config.get("terminal_sampler"), "gaussian"),
        "terminal_jitter_scale": resolve_float_override(
            args.terminal_jitter_scale,
            config.get("terminal_jitter_scale"),
            0.0,
        ),
        "reverse_noise_scale": resolve_float_override(
            args.reverse_noise_scale,
            config.get("reverse_noise_scale"),
            1.0,
        ),
        "reverse_control_scale": resolve_float_override(
            args.reverse_control_scale,
            config.get("reverse_control_scale"),
            1.0,
        ),
        "reverse_tail_noise_scale": resolve_float_override(
            args.reverse_tail_noise_scale,
            config.get("reverse_tail_noise_scale"),
            1.0,
        ),
        "reverse_deterministic_tail_steps": resolve_int_override(
            args.reverse_deterministic_tail_steps,
            config.get("reverse_deterministic_tail_steps"),
            0,
        ),
        "log_reverse_ablation": resolve_bool_override(
            args.log_reverse_ablation,
            config.get("log_reverse_ablation"),
            True,
        ),
        "fast_tuning": bool(args.fast_tuning),
    }

    save_json(
        outdir / "config.json",
        {
            "source_run_dir": None if args.run_dir is None else str(args.run_dir),
            "source_checkpoint": str(checkpoint_path),
            "checkpoint_mode": args.checkpoint_mode,
            "device": str(device),
            "eval_seed": eval_seed,
            "resolved_eval_config": resolved_eval_config,
        },
    )

    eval_metrics = evaluate_model(
        dataset=dataset,
        score_model=score_model,
        control_model=control_model,
        terminal_stats=terminal_stats,
        schedule=schedule,
        outdir=outdir,
        epoch=int(checkpoint.get("epoch", 0)),
        device=device,
        num_eval_samples=resolved_eval_config["num_eval_samples"],
        metric_samples=resolved_eval_config["metric_samples"],
        seed=eval_seed,
        zero_control=zero_control,
        num_snapshot_steps=resolved_eval_config["num_snapshot_steps"],
        method_name=method_name,
        reverse_solver=resolved_eval_config["reverse_solver"],
        terminal_sampler=resolved_eval_config["terminal_sampler"],
        terminal_jitter_scale=resolved_eval_config["terminal_jitter_scale"],
        reverse_noise_scale=resolved_eval_config["reverse_noise_scale"],
        reverse_control_scale=resolved_eval_config["reverse_control_scale"],
        reverse_tail_noise_scale=resolved_eval_config["reverse_tail_noise_scale"],
        reverse_deterministic_tail_steps=resolved_eval_config["reverse_deterministic_tail_steps"],
        diagnostic_points=diagnostic_points,
        diagnostic_noise=diagnostic_noise,
        log_reverse_ablation=resolved_eval_config["log_reverse_ablation"],
        fast_tuning=resolved_eval_config["fast_tuning"],
    )

    checkpoint_metrics = checkpoint.get("metrics")
    summary = {
        "source_run_dir": None if args.run_dir is None else str(args.run_dir),
        "checkpoint": str(checkpoint_path),
        "checkpoint_mode": args.checkpoint_mode,
        "checkpoint_epoch": int(checkpoint.get("epoch", 0)),
        "method": method_name,
        "device": str(device),
        "eval_seed": eval_seed,
        "checkpoint_metrics": checkpoint_metrics,
        "reeval_metrics": eval_metrics,
        "metric_deltas": compute_metric_deltas(checkpoint_metrics=checkpoint_metrics, eval_metrics=eval_metrics),
    }
    save_json(outdir / "summary.json", summary)
    print(json.dumps(summary, indent=2, default=_json_default))


def resolve_checkpoint_path(
    *,
    checkpoint: Path | None,
    run_dir: Path | None,
    checkpoint_mode: str,
) -> Path:
    if checkpoint is not None:
        return checkpoint
    if run_dir is None:
        raise ValueError("Provide either --checkpoint or --run-dir.")
    if checkpoint_mode == "best":
        return run_dir / "checkpoint_best.pt"
    return run_dir / "checkpoint_last.pt"


def resolve_outdir(
    *,
    outdir: Path | None,
    run_dir: Path | None,
    checkpoint_path: Path,
    checkpoint_mode: str,
) -> Path:
    if outdir is not None:
        return outdir
    if run_dir is not None:
        return run_dir / "reeval" / checkpoint_mode
    return checkpoint_path.parent / f"reeval_{checkpoint_path.stem}"


def schedule_from_dict(payload: dict, *, device: torch.device) -> MarkovVPSchedule:
    def to_tensor(key: str):
        value = payload.get(key)
        if value is None:
            return None
        return torch.tensor(value, device=device, dtype=torch.float32)

    dt = to_tensor("dt")
    beta = to_tensor("beta")
    g = to_tensor("g")
    step_sigma = to_tensor("step_sigma")
    weights = to_tensor("weights")
    sigma_levels = to_tensor("sigma_levels")
    family = str(payload.get("family") or infer_schedule_family(beta=beta, sigma_levels=sigma_levels))
    drift_coeff = to_tensor("drift_coeff")
    if drift_coeff is None:
        drift_coeff = reconstruct_drift_coeff(
            family=family,
            beta=beta,
            dt=dt,
            step_sigma=step_sigma,
        )

    return MarkovVPSchedule(
        family=family,
        total_time=float(payload["total_time"]),
        dt=dt,
        drift_coeff=drift_coeff,
        beta=beta,
        g=g,
        step_sigma=step_sigma,
        weights=weights,
        sigma_levels=sigma_levels,
    )


def infer_schedule_family(*, beta: torch.Tensor | None, sigma_levels: torch.Tensor | None) -> str:
    if sigma_levels is not None:
        return "ve_geometric"
    if beta is not None and torch.allclose(beta, torch.zeros_like(beta)):
        return "ve_geometric"
    return "vp_linear"


def reconstruct_drift_coeff(
    *,
    family: str,
    beta: torch.Tensor | None,
    dt: torch.Tensor | None,
    step_sigma: torch.Tensor | None,
) -> torch.Tensor:
    if dt is None or step_sigma is None:
        raise ValueError("Cannot reconstruct schedule without dt and step_sigma.")
    if family == "ve_geometric":
        return torch.zeros_like(step_sigma)
    if family == "vp_linear":
        if beta is None:
            raise ValueError("Cannot reconstruct vp_linear drift coefficients without beta.")
        mean_scale = torch.exp(-0.5 * beta * dt)
        return (mean_scale - 1.0) / dt
    if family == "vp_cosine":
        mean_scale = (1.0 - step_sigma.square()).clamp(min=1e-8, max=1.0).sqrt()
        return (mean_scale - 1.0) / dt
    raise ValueError(f"Unsupported schedule family: {family}")


def resolve_int_override(cli_value, config_value, default: int) -> int:
    if cli_value is not None:
        return int(cli_value)
    if config_value is not None:
        return int(config_value)
    return int(default)


def resolve_float_override(cli_value, config_value, default: float) -> float:
    if cli_value is not None:
        return float(cli_value)
    if config_value is not None:
        return float(config_value)
    return float(default)


def resolve_choice_override(cli_value, config_value, default: str) -> str:
    if cli_value is not None:
        return str(cli_value)
    if config_value is not None:
        return str(config_value)
    return str(default)


def resolve_bool_override(cli_value, config_value, default: bool) -> bool:
    if cli_value is not None:
        return bool(cli_value)
    if config_value is not None:
        return bool(config_value)
    return bool(default)


def compute_metric_deltas(*, checkpoint_metrics: dict | None, eval_metrics: dict) -> dict[str, float | None]:
    keys = (
        "mmd_rbf",
        "sliced_wasserstein",
        "reverse_no_control_mmd_rbf",
        "reverse_no_control_sliced_wasserstein",
        "reverse_control_gain_mmd",
        "reverse_control_gain_swd",
    )
    deltas: dict[str, float | None] = {}
    for key in keys:
        previous = None if checkpoint_metrics is None else checkpoint_metrics.get(key)
        current = eval_metrics.get(key)
        if previous is None or current is None:
            deltas[key] = None
        else:
            deltas[key] = float(current) - float(previous)
    return deltas


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


if __name__ == "__main__":
    main()
