from __future__ import annotations

import argparse
import concurrent.futures as futures
import itertools
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

from toy_2d.robust_defaults import CAUSAL_WDRO_DEFAULTS, CAUSAL_WDRO_TUNING_SPACE, WDRO_CORE_DEFAULTS, WDRO_CORE_TUNING_SPACE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Random search for robust toy_2d WILD settings.")
    parser.add_argument("--outdir", type=Path, default=Path("toy-runs") / "tuning")
    parser.add_argument("--datasets", nargs="+", default=["eight_gaussians", "spiral"])
    parser.add_argument("--methods", nargs="+", default=["wdro"])
    parser.add_argument("--trials", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--num-samples", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--num-eval-samples", type=int, default=1024)
    parser.add_argument("--metric-samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    payloads = []
    trial_id = 0
    for method in args.methods:
        configs = build_trial_configs(args=args, method=method)
        for cfg in configs:
            payloads.append((trial_id, method, cfg, args))
            trial_id += 1

    results: list[dict] = []
    with futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_map = {executor.submit(run_trial, payload): payload[0] for payload in payloads}
        for future in futures.as_completed(future_map):
            result = future.result()
            results.append(result)
            append_jsonl(args.outdir / "trial_results.jsonl", result)

    ranked = sorted(results, key=lambda item: (item["score"], item["mean_mmd"], item["trial_id"]))
    save_json(args.outdir / "ranking.json", ranked)
    if ranked:
        save_json(args.outdir / "best_config.json", ranked[0])
        print(json.dumps(ranked[0], indent=2))
    else:
        print("[]")


def build_trial_configs(*, args: argparse.Namespace, method: str) -> list[dict]:
    rng = random.Random(args.seed)
    shared_baseline = {
        "lr": 1e-3,
        "ema_decay": 0.995,
        "hidden_dim": 128,
        "depth": 4,
        "embedding_dim": 32,
        "sampler_steps": 40,
    }
    shared_space = {
        "lr": [5e-4, 1e-3, 2e-3],
        "ema_decay": [0.99, 0.995, 0.999],
        "hidden_dim": [64, 128, 256],
        "depth": [3, 4, 5],
        "embedding_dim": [16, 32, 64],
        "sampler_steps": [20, 40, 60],
    }

    if method == "baseline":
        baseline = dict(shared_baseline)
        search_space = dict(shared_space)
    elif method == "wdro":
        baseline = {
            **shared_baseline,
            "wdro_k": WDRO_CORE_DEFAULTS["k"],
            "wdro_step_size": WDRO_CORE_DEFAULTS["step_size"],
            "wdro_gamma": WDRO_CORE_DEFAULTS["gamma"],
            "wdro_p_adv": WDRO_CORE_DEFAULTS["p_adv"],
            "wdro_warmup_epochs": 3,
            "wdro_refresh_every": 3,
        }
        search_space = {
            **shared_space,
            "wdro_k": list(WDRO_CORE_TUNING_SPACE["k"]),
            "wdro_step_size": list(WDRO_CORE_TUNING_SPACE["step_size"]),
            "wdro_gamma": list(WDRO_CORE_TUNING_SPACE["gamma"]),
            "wdro_p_adv": [0.5, 1.0],
            "wdro_warmup_epochs": [0, 2, 3, 5],
            "wdro_refresh_every": [1, 2, 3, 5],
        }
    elif method == "causal_wdro":
        baseline = {
            "lr": 1e-3,
            "ema_decay": 0.99,
            "hidden_dim": 256,
            "depth": 4,
            "embedding_dim": 64,
            "sampler_steps": 20,
            "causal_warmup_epochs": CAUSAL_WDRO_DEFAULTS["warmup_epochs"],
            "causal_path_steps": CAUSAL_WDRO_DEFAULTS["path_steps"],
            "causal_inner_steps": CAUSAL_WDRO_DEFAULTS["inner_steps"],
            "causal_step_size": CAUSAL_WDRO_DEFAULTS["step_size"],
            "causal_gamma": CAUSAL_WDRO_DEFAULTS["gamma"],
            "causal_total_budget": CAUSAL_WDRO_DEFAULTS["total_budget"],
            "causal_budget_mode": CAUSAL_WDRO_DEFAULTS["budget_mode"],
            "causal_exact_budget_split": CAUSAL_WDRO_DEFAULTS["exact_budget_split"],
            "causal_sigma_schedule": CAUSAL_WDRO_DEFAULTS["sigma_schedule"],
        }
        search_space = {
            "lr": [5e-4, 1e-3, 2e-3],
            "ema_decay": [0.99, 0.995],
            "hidden_dim": [256],
            "depth": [4],
            "embedding_dim": [64],
            "sampler_steps": [20, 60],
            **CAUSAL_WDRO_TUNING_SPACE,
        }
    else:
        raise ValueError(f"Unsupported method: {method}")

    configs = [baseline]
    seen = {config_key(baseline)}
    keys = list(search_space)
    max_unique = 1
    for key in keys:
        max_unique *= len(search_space[key])

    target = min(args.trials, max_unique)
    while len(configs) < target:
        candidate = {key: rng.choice(search_space[key]) for key in keys}
        key = config_key(candidate)
        if key in seen:
            continue
        seen.add(key)
        configs.append(candidate)
    return configs


def config_key(config: dict) -> tuple:
    return tuple(sorted(config.items()))


def run_trial(payload: tuple[int, str, dict, argparse.Namespace]) -> dict:
    trial_id, method, config, args = payload
    root_dir = Path(__file__).resolve().parents[1]
    trial_root = args.outdir / f"trial_{trial_id:03d}"
    if trial_root.exists():
        shutil.rmtree(trial_root)
    trial_root.mkdir(parents=True, exist_ok=True)

    dataset_results = []
    for dataset in args.datasets:
        run_dir = trial_root / dataset
        command = [
            sys.executable,
            "-m",
            "toy_2d.train_wild",
            "--dataset",
            dataset,
            "--method",
            method,
            "--epochs",
            str(args.epochs),
            "--num-samples",
            str(args.num_samples),
            "--batch-size",
            str(args.batch_size),
            "--eval-every",
            str(args.eval_every),
            "--num-eval-samples",
            str(args.num_eval_samples),
            "--metric-samples",
            str(args.metric_samples),
            "--seed",
            str(args.seed),
            "--outdir",
            str(run_dir),
            "--lr",
            str(config["lr"]),
            "--ema-decay",
            str(config["ema_decay"]),
            "--hidden-dim",
            str(config["hidden_dim"]),
            "--depth",
            str(config["depth"]),
            "--embedding-dim",
            str(config["embedding_dim"]),
            "--sampler-steps",
            str(config["sampler_steps"]),
        ]
        append_optional_arg(command, "--wdro-k", config, "wdro_k")
        append_optional_arg(command, "--wdro-step-size", config, "wdro_step_size")
        append_optional_arg(command, "--wdro-gamma", config, "wdro_gamma")
        append_optional_arg(command, "--wdro-p-adv", config, "wdro_p_adv")
        append_optional_arg(command, "--wdro-warmup-epochs", config, "wdro_warmup_epochs")
        append_optional_arg(command, "--wdro-refresh-every", config, "wdro_refresh_every")
        append_optional_arg(command, "--causal-path-steps", config, "causal_path_steps")
        append_optional_arg(command, "--causal-warmup-epochs", config, "causal_warmup_epochs")
        append_optional_arg(command, "--causal-inner-steps", config, "causal_inner_steps")
        append_optional_arg(command, "--causal-step-size", config, "causal_step_size")
        append_optional_arg(command, "--causal-gamma", config, "causal_gamma")
        append_optional_arg(command, "--causal-total-budget", config, "causal_total_budget")
        append_optional_arg(command, "--causal-budget-mode", config, "causal_budget_mode")
        append_bool_arg(command, "--causal-exact-budget-split", "--no-causal-exact-budget-split", config, "causal_exact_budget_split")
        append_optional_arg(command, "--causal-sigma-schedule", config, "causal_sigma_schedule")
        subprocess.run(command, cwd=root_dir, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        dataset_results.append(
            {
                "dataset": dataset,
                "best_sliced_wasserstein": summary["best_sliced_wasserstein"],
                "mmd_rbf": summary["last_eval"]["mmd_rbf"],
                "last_sliced_wasserstein": summary["last_eval"]["sliced_wasserstein"],
                "mean_transport_cost": summary["last_eval"].get("mean_transport_cost", 0.0),
                "total_transport_cost": summary["last_eval"].get(
                    "total_transport_cost", summary["last_eval"].get("mean_transport_cost", 0.0)
                ),
                "final_loss": summary["final_loss"],
            }
        )

    score = sum(item["best_sliced_wasserstein"] for item in dataset_results) / len(dataset_results)
    mean_mmd = sum(item["mmd_rbf"] for item in dataset_results) / len(dataset_results)
    mean_total_transport_cost = sum(item["total_transport_cost"] for item in dataset_results) / len(dataset_results)
    result = {
        "trial_id": trial_id,
        "method": method,
        "score": score,
        "mean_mmd": mean_mmd,
        "mean_total_transport_cost": mean_total_transport_cost,
        "config": config,
        "datasets": dataset_results,
    }
    return result


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def append_optional_arg(command: list[str], flag: str, config: dict, key: str) -> None:
    if key in config:
        command.extend([flag, str(config[key])])


def append_bool_arg(command: list[str], true_flag: str, false_flag: str, config: dict, key: str) -> None:
    if key in config:
        command.append(true_flag if bool(config[key]) else false_flag)


if __name__ == "__main__":
    main()
