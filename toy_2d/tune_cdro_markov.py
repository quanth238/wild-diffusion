from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

from toy_2d.robust_defaults import CDRO_MARKOV_DEFAULTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Staged search for Markov CDRO settings.")
    parser.add_argument("--outdir", type=Path, default=Path("toy-runs") / "tuning_cdro_markov")
    parser.add_argument("--datasets", nargs="+", default=["eight_gaussians"])
    parser.add_argument("--fractions", nargs="+", type=float, default=[0.5, 1.0])
    parser.add_argument("--full-samples", type=int, default=2000)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--trials", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--proxy-epochs", type=int, default=60)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--proxy-batch-size", type=int, default=None)
    parser.add_argument("--final-batch-size", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--proxy-eval-every", type=int, default=10)
    parser.add_argument("--num-eval-samples", type=int, default=2048)
    parser.add_argument("--proxy-num-eval-samples", type=int, default=1024)
    parser.add_argument("--metric-samples", type=int, default=1024)
    parser.add_argument("--proxy-metric-samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reference-aggregate", type=Path, default=None)
    parser.add_argument("--reference-method", type=str, default="wdro")
    parser.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--search-strategy", type=str, default="coordinate", choices=("coordinate", "random"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    trial_configs = build_trial_configs(args=args)
    reference_summary = load_reference_summary(
        path=args.reference_aggregate,
        method=args.reference_method,
    )

    proxy_payloads = [
        (
            trial_id,
            config,
            args,
            reference_summary,
            StageConfig(
                name="proxy",
                epochs=args.proxy_epochs,
                batch_size=args.proxy_batch_size if args.proxy_batch_size is not None else args.batch_size,
                eval_every=args.proxy_eval_every,
                num_eval_samples=args.proxy_num_eval_samples,
                metric_samples=args.proxy_metric_samples,
                fast_tuning=True,
            ),
        )
        for trial_id, config in enumerate(trial_configs)
    ]
    proxy_results = run_stage(
        payloads=proxy_payloads,
        workers=args.workers,
        output_path=args.outdir / "proxy_trial_results.jsonl",
    )
    proxy_ranked = rank_results(proxy_results)
    save_json(args.outdir / "proxy_ranking.json", proxy_ranked)

    shortlisted = proxy_ranked[: max(1, min(args.topk, len(proxy_ranked)))]
    final_payloads = [
        (
            item["trial_id"],
            item["config"],
            args,
            reference_summary,
            StageConfig(
                name="final",
                epochs=args.epochs,
                batch_size=args.final_batch_size if args.final_batch_size is not None else args.batch_size,
                eval_every=args.eval_every,
                num_eval_samples=args.num_eval_samples,
                metric_samples=args.metric_samples,
                fast_tuning=True,
            ),
        )
        for item in shortlisted
    ]
    results = run_stage(
        payloads=final_payloads,
        workers=args.workers,
        output_path=args.outdir / "final_trial_results.jsonl",
    )
    ranked = rank_results(results)
    save_json(args.outdir / "ranking.json", ranked)
    if ranked:
        save_json(args.outdir / "best_config.json", ranked[0])
        print(json.dumps(ranked[0], indent=2))
    else:
        print("[]")


def build_trial_configs(*, args: argparse.Namespace) -> list[dict]:
    if args.search_strategy == "coordinate":
        return build_coordinate_trial_configs(args=args)
    return build_random_trial_configs(args=args)


def build_random_trial_configs(*, args: argparse.Namespace) -> list[dict]:
    rng = random.Random(args.seed)
    baseline = dict(CDRO_MARKOV_DEFAULTS)
    search_space = {
        "score_lr": [5e-4, 1e-3],
        "control_lr": [1e-4, 2e-4, 5e-4],
        "lambda_lr": [1e-2, 2e-2, 5e-2],
        "lambda_init": [0.05, 0.1, 0.2],
        "lambda_min": [0.01, 0.02, 0.05, 0.08],
        "control_radius": [0.02, 0.03, 0.05, 0.08],
        "warmup_epochs": [8, 12, 16],
        "adversary_steps": [1],
        "score_steps": [8, 12, 16],
        "terminal_momentum": [0.95, 0.98],
        "num_steps": [12, 16, 20],
        "total_time": [1.0],
        "beta_min": [0.2],
        "beta_max": [6.0, 8.0, 10.0],
        "score_weight_schedule": ["uniform", "inv_sigma_sq", "sigma_sq"],
        "score_hidden_dim": [128, 192, 256],
        "score_depth": [4],
        "control_arch": ["mlp", "gru"],
        "control_hidden_dim": [64, 96, 128],
        "control_depth": [3],
        "control_scale": [0.3, 0.5],
        "ema_decay": [0.995],
        "embedding_dim": [32],
    }

    configs = [baseline]
    seen = {config_key(baseline)}
    max_unique = 1
    for key, values in search_space.items():
        max_unique *= len(values)

    target = min(args.trials, max_unique)
    keys = list(search_space)
    while len(configs) < target:
        candidate = {key: rng.choice(search_space[key]) for key in keys}
        key = config_key(candidate)
        if key in seen:
            continue
        seen.add(key)
        configs.append(candidate)
    return configs


def build_coordinate_trial_configs(*, args: argparse.Namespace) -> list[dict]:
    baseline = dict(CDRO_MARKOV_DEFAULTS)
    ordered_space = [
        ("score_weight_schedule", [baseline["score_weight_schedule"], "inv_sigma_sq", "sigma_sq"]),
        ("num_steps", [baseline["num_steps"], 16, 20]),
        ("beta_max", [baseline["beta_max"], 8.0, 10.0]),
        ("score_hidden_dim", [baseline["score_hidden_dim"], 192, 256]),
        ("score_steps", [baseline["score_steps"], 12, 16]),
        ("warmup_epochs", [baseline["warmup_epochs"], 12, 16]),
        ("lambda_min", [baseline["lambda_min"], 0.05, 0.08, 0.01]),
        ("control_radius", [baseline["control_radius"], 0.03, 0.02, 0.08]),
        ("lambda_lr", [baseline["lambda_lr"], 0.01, 0.05]),
        ("control_arch", [baseline["control_arch"], "gru"]),
        ("control_hidden_dim", [baseline["control_hidden_dim"], 96, 128]),
        ("control_scale", [baseline["control_scale"], 0.3]),
        ("control_lr", [baseline["control_lr"], 1e-4, 5e-4]),
        ("lambda_init", [baseline["lambda_init"], 0.2, 0.05]),
        ("score_lr", [baseline["score_lr"], 5e-4]),
        ("terminal_momentum", [baseline["terminal_momentum"], 0.98]),
    ]

    configs = [baseline]
    seen = {config_key(baseline)}
    target = max(1, args.trials)
    max_levels = max(len(values) - 1 for _, values in ordered_space)

    for level in range(max_levels):
        for key, values in ordered_space:
            alt_index = level + 1
            if alt_index >= len(values):
                continue
            candidate = dict(baseline)
            candidate[key] = values[alt_index]
            candidate_key = config_key(candidate)
            if candidate_key in seen:
                continue
            seen.add(candidate_key)
            configs.append(candidate)
            if len(configs) >= target:
                return configs
    return configs


def config_key(config: dict) -> tuple:
    return tuple(sorted(config.items()))


def load_reference_summary(*, path: Path | None, method: str) -> dict[tuple[str, float], dict] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload["summary"] if isinstance(payload, dict) and "summary" in payload else payload
    rows = {}
    for item in summary:
        rows[(item["dataset"], item["fraction"])] = {
            "best_swd": item[f"{method}_best_swd"],
            "last_swd": item[f"{method}_last_swd"],
            "mmd": item[f"{method}_mmd"],
        }
    return rows


class StageConfig:
    def __init__(
        self,
        *,
        name: str,
        epochs: int,
        batch_size: int,
        eval_every: int,
        num_eval_samples: int,
        metric_samples: int,
        fast_tuning: bool,
    ):
        self.name = name
        self.epochs = epochs
        self.batch_size = batch_size
        self.eval_every = eval_every
        self.num_eval_samples = num_eval_samples
        self.metric_samples = metric_samples
        self.fast_tuning = fast_tuning


def run_stage(*, payloads: list[tuple], workers: int, output_path: Path) -> list[dict]:
    results: list[dict] = []
    if output_path.exists():
        output_path.unlink()
    if workers <= 1:
        for payload in payloads:
            result = run_trial(payload)
            results.append(result)
            append_jsonl(output_path, result)
        return results
    with futures.ProcessPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(run_trial, payload): payload[0] for payload in payloads}
        for future in futures.as_completed(future_map):
            result = future.result()
            results.append(result)
            append_jsonl(output_path, result)
    return results


def rank_results(results: list[dict]) -> list[dict]:
    return sorted(
        results,
        key=lambda item: (
            item.get("mean_best_gap_to_reference", float("inf")),
            item["mean_best_swd"],
            item["mean_last_swd"],
            item["trial_id"],
        ),
    )


def run_trial(payload: tuple[int, dict, argparse.Namespace, dict | None, StageConfig]) -> dict:
    trial_id, config, args, reference_summary, stage = payload
    root_dir = Path(__file__).resolve().parents[1]
    trial_root = args.outdir / stage.name / f"trial_{trial_id:03d}"
    if trial_root.exists():
        shutil.rmtree(trial_root)
    trial_root.mkdir(parents=True, exist_ok=True)

    dataset_results = []
    for dataset in args.datasets:
        for fraction in args.fractions:
            num_samples = max(64, int(round(args.full_samples * fraction)))
            fraction_tag = fraction_to_tag(fraction)
            per_seed = []
            for seed in args.seeds:
                run_dir = trial_root / dataset / fraction_tag / f"seed{seed}"
                command = [
                    sys.executable,
                    "-m",
                    "toy_2d.train_cdro_markov",
                    "--dataset",
                    dataset,
                    "--epochs",
                    str(stage.epochs),
                    "--num-samples",
                    str(num_samples),
                    "--batch-size",
                    str(stage.batch_size),
                    "--eval-every",
                    str(stage.eval_every),
                    "--num-eval-samples",
                    str(stage.num_eval_samples),
                    "--metric-samples",
                    str(stage.metric_samples),
                    "--seed",
                    str(seed),
                    "--outdir",
                    str(run_dir),
                    "--device",
                    str(args.device),
                ]
                if stage.fast_tuning:
                    command.append("--fast-tuning")
                append_optional_arg(command, "--ema-decay", config, "ema_decay")
                append_optional_arg(command, "--embedding-dim", config, "embedding_dim")
                append_optional_arg(command, "--score-lr", config, "score_lr")
                append_optional_arg(command, "--control-lr", config, "control_lr")
                append_optional_arg(command, "--lambda-lr", config, "lambda_lr")
                append_optional_arg(command, "--lambda-init", config, "lambda_init")
                append_optional_arg(command, "--lambda-min", config, "lambda_min")
                append_optional_arg(command, "--control-radius", config, "control_radius")
                append_optional_arg(command, "--warmup-epochs", config, "warmup_epochs")
                append_optional_arg(command, "--adversary-steps", config, "adversary_steps")
                append_optional_arg(command, "--score-steps", config, "score_steps")
                append_optional_arg(command, "--terminal-momentum", config, "terminal_momentum")
                append_optional_arg(command, "--num-steps", config, "num_steps")
                append_optional_arg(command, "--total-time", config, "total_time")
                append_optional_arg(command, "--beta-min", config, "beta_min")
                append_optional_arg(command, "--beta-max", config, "beta_max")
                append_optional_arg(command, "--score-weight-schedule", config, "score_weight_schedule")
                append_optional_arg(command, "--score-hidden-dim", config, "score_hidden_dim")
                append_optional_arg(command, "--score-depth", config, "score_depth")
                append_optional_arg(command, "--control-arch", config, "control_arch")
                append_optional_arg(command, "--control-hidden-dim", config, "control_hidden_dim")
                append_optional_arg(command, "--control-depth", config, "control_depth")
                append_optional_arg(command, "--control-scale", config, "control_scale")
                subprocess.run(command, cwd=root_dir, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
                per_seed.append(summary)

            mean_best_swd = mean(item["best_sliced_wasserstein"] for item in per_seed)
            mean_last_swd = mean(item["last_eval"]["sliced_wasserstein"] for item in per_seed)
            mean_mmd = mean(item["last_eval"]["mmd_rbf"] for item in per_seed)
            mean_control_cost = mean(item["last_eval"].get("control_cost", 0.0) for item in per_seed)
            cell_result = {
                "dataset": dataset,
                "fraction": fraction,
                "fraction_tag": fraction_tag,
                "num_samples": num_samples,
                "mean_best_swd": mean_best_swd,
                "mean_last_swd": mean_last_swd,
                "mean_mmd": mean_mmd,
                "mean_control_cost": mean_control_cost,
            }
            if reference_summary is not None and (dataset, fraction) in reference_summary:
                ref = reference_summary[(dataset, fraction)]
                cell_result["best_gap_to_reference"] = mean_best_swd - ref["best_swd"]
                cell_result["last_gap_to_reference"] = mean_last_swd - ref["last_swd"]
                cell_result["mmd_gap_to_reference"] = mean_mmd - ref["mmd"]
            dataset_results.append(cell_result)

    result = {
        "trial_id": trial_id,
        "stage": stage.name,
        "config": config,
        "cells": dataset_results,
        "mean_best_swd": mean(item["mean_best_swd"] for item in dataset_results),
        "mean_last_swd": mean(item["mean_last_swd"] for item in dataset_results),
        "mean_mmd": mean(item["mean_mmd"] for item in dataset_results),
        "mean_control_cost": mean(item["mean_control_cost"] for item in dataset_results),
    }
    if reference_summary is not None:
        result["mean_best_gap_to_reference"] = mean(item["best_gap_to_reference"] for item in dataset_results)
        result["mean_last_gap_to_reference"] = mean(item["last_gap_to_reference"] for item in dataset_results)
        result["mean_mmd_gap_to_reference"] = mean(item["mmd_gap_to_reference"] for item in dataset_results)
    return result


def append_optional_arg(command: list[str], flag: str, config: dict, key: str) -> None:
    if key in config:
        command.extend([flag, str(config[key])])


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def fraction_to_tag(fraction: float) -> str:
    return f"{int(round(fraction * 100)):d}pct"


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


if __name__ == "__main__":
    main()
