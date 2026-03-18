from __future__ import annotations

import argparse
import concurrent.futures as futures
import csv
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from toy_2d.robust_defaults import CAUSAL_WDRO_DEFAULTS, WDRO_CORE_DEFAULTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare toy_2d methods across data fractions.")
    parser.add_argument("--outdir", type=Path, default=Path("toy-runs") / "method_table")
    parser.add_argument("--method-configs", type=Path, default=None)
    parser.add_argument("--datasets", nargs="+", default=["eight_gaussians", "spiral", "two_moons"])
    parser.add_argument("--methods", nargs="+", default=["baseline", "wdro", "causal_wdro"])
    parser.add_argument("--fractions", nargs="+", type=float, default=[0.2, 0.5, 1.0])
    parser.add_argument("--full-samples", type=int, default=2000)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--figure-seed", type=int, default=0)
    parser.add_argument("--cleanup-runs", action="store_true")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--num-eval-samples", type=int, default=2048)
    parser.add_argument("--metric-samples", type=int, default=1024)

    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--wdro-k", type=int, default=WDRO_CORE_DEFAULTS["k"])
    parser.add_argument("--wdro-step-size", type=float, default=WDRO_CORE_DEFAULTS["step_size"])
    parser.add_argument("--wdro-gamma", type=float, default=WDRO_CORE_DEFAULTS["gamma"])
    parser.add_argument("--wdro-p-adv", type=float, default=WDRO_CORE_DEFAULTS["p_adv"])
    parser.add_argument("--wdro-warmup-epochs", type=int, default=25)
    parser.add_argument("--wdro-refresh-every", type=int, default=10)
    parser.add_argument("--causal-warmup-epochs", type=int, default=CAUSAL_WDRO_DEFAULTS["warmup_epochs"])
    parser.add_argument("--causal-path-steps", type=int, default=CAUSAL_WDRO_DEFAULTS["path_steps"])
    parser.add_argument("--causal-inner-steps", type=int, default=CAUSAL_WDRO_DEFAULTS["inner_steps"])
    parser.add_argument("--causal-step-size", type=float, default=CAUSAL_WDRO_DEFAULTS["step_size"])
    parser.add_argument("--causal-gamma", type=float, default=CAUSAL_WDRO_DEFAULTS["gamma"])
    parser.add_argument("--causal-total-budget", type=float, default=CAUSAL_WDRO_DEFAULTS["total_budget"])
    parser.add_argument(
        "--causal-budget-mode",
        type=str,
        default=CAUSAL_WDRO_DEFAULTS["budget_mode"],
        choices=("fixed", "match_wdro", "match_wdro_run"),
    )
    parser.add_argument(
        "--causal-exact-budget-split",
        action=argparse.BooleanOptionalAction,
        default=CAUSAL_WDRO_DEFAULTS["exact_budget_split"],
    )
    parser.add_argument(
        "--causal-sigma-schedule",
        type=str,
        default=CAUSAL_WDRO_DEFAULTS["sigma_schedule"],
        choices=("edm_random", "edm_quantiles", "karras_grid"),
    )
    parser.add_argument("--sampler-steps", type=int, default=20)
    parser.add_argument("--save-eval-checkpoints", action="store_true")
    parser.add_argument("--figure-epoch-mode", type=str, default="last", choices=("best", "last", "fixed"))
    parser.add_argument("--figure-fixed-epoch", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    args.outdir.mkdir(parents=True, exist_ok=True)
    method_configs = load_method_configs(args.method_configs)

    jobs = build_jobs(args=args, root=root, method_configs=method_configs)
    results = run_jobs(jobs=jobs, workers=args.workers)

    aggregate = aggregate_results(results=results, args=args, method_configs=method_configs)
    write_outputs(outdir=args.outdir, aggregate=aggregate)
    export_figures(
        outdir=args.outdir,
        aggregate=aggregate,
        figure_seed=args.figure_seed,
        epoch_mode=args.figure_epoch_mode,
        fixed_epoch=args.figure_fixed_epoch,
    )
    if args.cleanup_runs:
        cleanup_run_dirs(outdir=args.outdir, datasets=args.datasets)
    print(json.dumps(aggregate["summary"], indent=2))


def build_jobs(*, args: argparse.Namespace, root: Path, method_configs: dict[str, dict]) -> list[dict]:
    jobs = []
    wdro_reference_config = method_configs.get("wdro", {})
    if any(
        str(method_configs.get("causal_wdro", {}).get("causal_budget_mode", args.causal_budget_mode)) == "match_wdro_run"
        for _ in [0]
    ) and "wdro" not in args.methods:
        raise ValueError("causal_budget_mode=match_wdro_run requires wdro to be included in --methods.")
    for dataset in args.datasets:
        for fraction in args.fractions:
            num_samples = max(64, int(round(args.full_samples * fraction)))
            fraction_tag = fraction_to_tag(fraction)
            for method in args.methods:
                method_config = method_configs.get(method, {})
                for seed in args.seeds:
                    requested_causal_budget_mode = str(
                        get_config_value(method_config, "causal_budget_mode", args.causal_budget_mode)
                    )
                    resolved_causal_budget_mode = (
                        "fixed" if requested_causal_budget_mode == "match_wdro_run" else requested_causal_budget_mode
                    )
                    outdir = args.outdir / dataset / fraction_tag / method / f"seed{seed}"
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
                        str(num_samples),
                        "--batch-size",
                        str(args.batch_size),
                        "--eval-every",
                        str(args.eval_every),
                        "--num-eval-samples",
                        str(args.num_eval_samples),
                        "--metric-samples",
                        str(args.metric_samples),
                        "--seed",
                        str(seed),
                        "--outdir",
                        str(outdir),
                        "--lr",
                        str(get_config_value(method_config, "lr", args.lr)),
                        "--ema-decay",
                        str(get_config_value(method_config, "ema_decay", args.ema_decay)),
                        "--hidden-dim",
                        str(get_config_value(method_config, "hidden_dim", args.hidden_dim)),
                        "--depth",
                        str(get_config_value(method_config, "depth", args.depth)),
                        "--embedding-dim",
                        str(get_config_value(method_config, "embedding_dim", args.embedding_dim)),
                        "--wdro-k",
                        str(get_config_value(method_config, "wdro_k", args.wdro_k)),
                        "--wdro-step-size",
                        str(get_config_value(method_config, "wdro_step_size", args.wdro_step_size)),
                        "--wdro-gamma",
                        str(get_config_value(method_config, "wdro_gamma", args.wdro_gamma)),
                        "--wdro-p-adv",
                        str(get_config_value(method_config, "wdro_p_adv", args.wdro_p_adv)),
                        "--wdro-warmup-epochs",
                        str(get_config_value(method_config, "wdro_warmup_epochs", args.wdro_warmup_epochs)),
                        "--wdro-refresh-every",
                        str(get_config_value(method_config, "wdro_refresh_every", args.wdro_refresh_every)),
                        "--causal-path-steps",
                        str(get_config_value(method_config, "causal_path_steps", args.causal_path_steps)),
                        "--causal-warmup-epochs",
                        str(get_config_value(method_config, "causal_warmup_epochs", args.causal_warmup_epochs)),
                        "--causal-inner-steps",
                        str(get_config_value(method_config, "causal_inner_steps", args.causal_inner_steps)),
                        "--causal-step-size",
                        str(get_config_value(method_config, "causal_step_size", args.causal_step_size)),
                        "--causal-gamma",
                        str(get_config_value(method_config, "causal_gamma", args.causal_gamma)),
                        "--causal-total-budget",
                        str(get_config_value(method_config, "causal_total_budget", args.causal_total_budget)),
                        "--causal-budget-mode",
                        resolved_causal_budget_mode,
                        "--causal-exact-budget-split"
                        if bool(get_config_value(method_config, "causal_exact_budget_split", args.causal_exact_budget_split))
                        else "--no-causal-exact-budget-split",
                        "--causal-sigma-schedule",
                        str(get_config_value(method_config, "causal_sigma_schedule", args.causal_sigma_schedule)),
                        "--causal-reference-wdro-k",
                        str(get_config_value(method_config, "causal_reference_wdro_k", get_config_value(wdro_reference_config, "wdro_k", args.wdro_k))),
                        "--causal-reference-wdro-step-size",
                        str(
                            get_config_value(
                                method_config,
                                "causal_reference_wdro_step_size",
                                get_config_value(wdro_reference_config, "wdro_step_size", args.wdro_step_size),
                            )
                        ),
                        "--causal-reference-wdro-gamma",
                        str(
                            get_config_value(
                                method_config,
                                "causal_reference_wdro_gamma",
                                get_config_value(wdro_reference_config, "wdro_gamma", args.wdro_gamma),
                            )
                        ),
                        "--sampler-steps",
                        str(get_config_value(method_config, "sampler_steps", args.sampler_steps)),
                    ]
                    if args.save_eval_checkpoints:
                        command.append("--save-eval-checkpoints")
                    jobs.append(
                        {
                            "root": root,
                            "outdir": outdir,
                            "dataset": dataset,
                            "fraction": fraction,
                            "fraction_tag": fraction_tag,
                            "num_samples": num_samples,
                            "method": method,
                            "seed": seed,
                            "command": command,
                            "requested_causal_budget_mode": requested_causal_budget_mode if method == "causal_wdro" else None,
                            "reference_wdro_outdir": (
                                args.outdir / dataset / fraction_tag / "wdro" / f"seed{seed}" if method == "causal_wdro" else None
                            ),
                        }
                    )
    return jobs


def run_jobs(*, jobs: list[dict], workers: int) -> list[dict]:
    dependent_causal_jobs = [
        job
        for job in jobs
        if job["method"] == "causal_wdro" and job.get("requested_causal_budget_mode") == "match_wdro_run"
    ]
    phase_one_jobs = [job for job in jobs if job not in dependent_causal_jobs]

    results: list[dict] = []
    if phase_one_jobs:
        with futures.ThreadPoolExecutor(max_workers=workers) as executor:
            for item in executor.map(run_job, phase_one_jobs):
                results.append(item)
    if dependent_causal_jobs:
        with futures.ThreadPoolExecutor(max_workers=workers) as executor:
            for item in executor.map(run_job, dependent_causal_jobs):
                results.append(item)
    return results


def run_job(job: dict) -> dict:
    summary_path = job["outdir"] / "summary.json"
    if not summary_path.is_file():
        command = list(job["command"])
        if job["method"] == "causal_wdro" and job.get("requested_causal_budget_mode") == "match_wdro_run":
            ref_summary_path = Path(job["reference_wdro_outdir"]) / "summary.json"
            if not ref_summary_path.is_file():
                raise FileNotFoundError(f"Missing WDRO reference summary for causal budget match: {ref_summary_path}")
            ref_summary = json.loads(ref_summary_path.read_text(encoding="utf-8"))
            epsilon = ref_summary["last_eval"].get(
                "total_transport_cost", ref_summary["last_eval"].get("mean_transport_cost", 0.0)
            )
            replace_command_arg(command, "--causal-total-budget", str(epsilon))
        subprocess.run(
            command,
            cwd=job["root"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "dataset": job["dataset"],
        "fraction": job["fraction"],
        "fraction_tag": job["fraction_tag"],
        "num_samples": job["num_samples"],
        "method": job["method"],
        "seed": job["seed"],
        "best_swd": summary["best_sliced_wasserstein"],
        "best_epoch": summary.get("best_epoch"),
        "last_epoch": summary["last_eval"]["epoch"],
        "last_swd": summary["last_eval"]["sliced_wasserstein"],
        "mmd": summary["last_eval"]["mmd_rbf"],
        "mean_transport_cost": summary["last_eval"].get("mean_transport_cost", 0.0),
        "max_transport_cost": summary["last_eval"].get("max_transport_cost", 0.0),
        "total_transport_cost": summary["last_eval"].get(
            "total_transport_cost", summary["last_eval"].get("mean_transport_cost", 0.0)
        ),
        "target_total_budget": summary["last_eval"].get("causal_target_total_budget"),
        "max_total_transport_cost": summary["last_eval"].get(
            "max_total_transport_cost", summary["last_eval"].get("max_transport_cost", 0.0)
        ),
        "final_loss": summary["final_loss"],
        "runtime_minutes": summary["runtime_minutes"],
        "run_dir": str(job["outdir"]),
    }


def aggregate_results(*, results: list[dict], args: argparse.Namespace, method_configs: dict[str, dict]) -> dict:
    grouped: dict[tuple[str, float, str], list[dict]] = defaultdict(list)
    for row in results:
        grouped[(row["dataset"], row["fraction"], row["method"])].append(row)

    summary_rows = []
    for dataset in args.datasets:
        for fraction in args.fractions:
            row = {
                "dataset": dataset,
                "fraction": fraction,
                "fraction_tag": fraction_to_tag(fraction),
            }
            for method in args.methods:
                method_rows = grouped[(dataset, fraction, method)]
                row[f"{method}_best_swd"] = mean(item["best_swd"] for item in method_rows)
                row[f"{method}_last_swd"] = mean(item["last_swd"] for item in method_rows)
                row[f"{method}_mmd"] = mean(item["mmd"] for item in method_rows)
                row[f"{method}_mean_transport_cost"] = mean(item["mean_transport_cost"] for item in method_rows)
                row[f"{method}_max_transport_cost"] = mean(item["max_transport_cost"] for item in method_rows)
                row[f"{method}_total_transport_cost"] = mean(item["total_transport_cost"] for item in method_rows)
                target_budgets = [item["target_total_budget"] for item in method_rows if item["target_total_budget"] is not None]
                row[f"{method}_target_total_budget"] = mean(target_budgets) if target_budgets else None
                row[f"{method}_max_total_transport_cost"] = mean(
                    item["max_total_transport_cost"] for item in method_rows
                )
                row[f"{method}_runtime_min"] = mean(item["runtime_minutes"] for item in method_rows)

            if "baseline" in args.methods:
                baseline_best = row["baseline_best_swd"]
                baseline_last = row["baseline_last_swd"]
                baseline_mmd = row["baseline_mmd"]
                baseline_runtime = row["baseline_runtime_min"]
                for method in args.methods:
                    if method == "baseline":
                        continue
                    row[f"{method}_best_swd_delta_pct"] = percent_improvement(
                        baseline_best, row[f"{method}_best_swd"]
                    )
                    row[f"{method}_last_swd_delta_pct"] = percent_improvement(
                        baseline_last, row[f"{method}_last_swd"]
                    )
                    row[f"{method}_mmd_delta_pct"] = percent_improvement(
                        baseline_mmd, row[f"{method}_mmd"]
                    )
                    row[f"{method}_runtime_overhead_pct"] = percent_overhead(
                        baseline_runtime, row[f"{method}_runtime_min"]
                    )
                if "wdro" in args.methods and "causal_wdro" in args.methods:
                    wdro_cost = row["wdro_total_transport_cost"]
                    causal_cost = row["causal_wdro_total_transport_cost"]
                    causal_target = row["causal_wdro_target_total_budget"]
                    row["causal_vs_wdro_total_cost_gap_pct"] = percent_overhead(wdro_cost, causal_cost)
                    if causal_target is not None:
                        row["causal_target_vs_wdro_gap_pct"] = percent_overhead(wdro_cost, causal_target)

            summary_rows.append(row)

    return {
        "config": {
            "method_configs_path": str(args.method_configs) if args.method_configs is not None else None,
            "method_configs": method_configs,
            "datasets": args.datasets,
            "methods": args.methods,
            "fractions": args.fractions,
            "full_samples": args.full_samples,
            "seeds": args.seeds,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "eval_every": args.eval_every,
            "num_eval_samples": args.num_eval_samples,
            "metric_samples": args.metric_samples,
            "lr": args.lr,
            "ema_decay": args.ema_decay,
            "hidden_dim": args.hidden_dim,
            "depth": args.depth,
            "embedding_dim": args.embedding_dim,
            "wdro_k": args.wdro_k,
            "wdro_step_size": args.wdro_step_size,
            "wdro_gamma": args.wdro_gamma,
            "wdro_p_adv": args.wdro_p_adv,
            "wdro_warmup_epochs": args.wdro_warmup_epochs,
            "wdro_refresh_every": args.wdro_refresh_every,
            "causal_path_steps": args.causal_path_steps,
            "causal_warmup_epochs": args.causal_warmup_epochs,
            "causal_inner_steps": args.causal_inner_steps,
            "causal_step_size": args.causal_step_size,
            "causal_gamma": args.causal_gamma,
            "causal_total_budget": args.causal_total_budget,
            "causal_budget_mode": args.causal_budget_mode,
            "causal_exact_budget_split": args.causal_exact_budget_split,
            "causal_sigma_schedule": args.causal_sigma_schedule,
            "sampler_steps": args.sampler_steps,
            "save_eval_checkpoints": args.save_eval_checkpoints,
            "figure_epoch_mode": args.figure_epoch_mode,
            "figure_fixed_epoch": args.figure_fixed_epoch,
        },
        "runs": results,
        "summary": summary_rows,
    }


def write_outputs(*, outdir: Path, aggregate: dict) -> None:
    (outdir / "aggregate.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")

    csv_path = outdir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate["summary"][0].keys()))
        writer.writeheader()
        writer.writerows(aggregate["summary"])

    methods = aggregate["config"]["methods"]
    md_path = outdir / "summary.md"
    swd_header = ["Dataset", "Data"] + [f"{format_method_name(method)} SWD" for method in methods]
    swd_header += [f"{format_method_name(method)} Delta" for method in methods if method != "baseline"]
    lines = [
        "## Last SWD",
        "",
        markdown_header(swd_header),
    ]
    for row in aggregate["summary"]:
        cells = [row["dataset"], row["fraction_tag"]]
        cells.extend(f"{row[f'{method}_last_swd']:.4f}" for method in methods)
        cells.extend(f"{row[f'{method}_last_swd_delta_pct']:.2f}%" for method in methods if method != "baseline")
        lines.append(markdown_row(cells))
    lines += [
        "",
        "## Best SWD",
        "",
        markdown_header(swd_header),
    ]
    for row in aggregate["summary"]:
        cells = [row["dataset"], row["fraction_tag"]]
        cells.extend(f"{row[f'{method}_best_swd']:.4f}" for method in methods)
        cells.extend(f"{row[f'{method}_best_swd_delta_pct']:.2f}%" for method in methods if method != "baseline")
        lines.append(markdown_row(cells))
    lines += [
        "",
        "## MMD",
        "",
        markdown_header(
            ["Dataset", "Data"]
            + [f"{format_method_name(method)} MMD" for method in methods]
            + [f"{format_method_name(method)} Delta" for method in methods if method != "baseline"]
        ),
    ]
    for row in aggregate["summary"]:
        cells = [row["dataset"], row["fraction_tag"]]
        cells.extend(f"{row[f'{method}_mmd']:.4f}" for method in methods)
        cells.extend(f"{row[f'{method}_mmd_delta_pct']:.2f}%" for method in methods if method != "baseline")
        lines.append(markdown_row(cells))
    lines += [
        "",
        "## Total Transport Cost",
        "",
        markdown_header(["Dataset", "Data"] + [f"{format_method_name(method)} Cost" for method in methods]),
    ]
    for row in aggregate["summary"]:
        cells = [row["dataset"], row["fraction_tag"]]
        cells.extend(f"{row[f'{method}_total_transport_cost']:.4f}" for method in methods)
        lines.append(markdown_row(cells))
    if "wdro" in methods and "causal_wdro" in methods and any(
        row.get("causal_wdro_target_total_budget") is not None for row in aggregate["summary"]
    ):
        lines += [
            "",
            "## Budget Match",
            "",
            markdown_header(
                [
                    "Dataset",
                    "Data",
                    "WDRO Cost",
                    "Causal Target",
                    "Causal Realized",
                    "Target Gap",
                    "Realized Gap",
                ]
            ),
        ]
        for row in aggregate["summary"]:
            target_budget = row["causal_wdro_target_total_budget"]
            cells = [
                row["dataset"],
                row["fraction_tag"],
                f"{row['wdro_total_transport_cost']:.4f}",
                "n/a" if target_budget is None else f"{target_budget:.4f}",
                f"{row['causal_wdro_total_transport_cost']:.4f}",
                "n/a"
                if target_budget is None
                else f"{row['causal_target_vs_wdro_gap_pct']:.2f}%",
                f"{row['causal_vs_wdro_total_cost_gap_pct']:.2f}%",
            ]
            lines.append(markdown_row(cells))
    lines += [
        "",
        "## Mean Per-Step Transport Cost",
        "",
        markdown_header(["Dataset", "Data"] + [f"{format_method_name(method)} Cost" for method in methods]),
    ]
    for row in aggregate["summary"]:
        cells = [row["dataset"], row["fraction_tag"]]
        cells.extend(f"{row[f'{method}_mean_transport_cost']:.4f}" for method in methods)
        lines.append(markdown_row(cells))
    lines += [
        "",
        "## Runtime",
        "",
        markdown_header(
            ["Dataset", "Data"]
            + [f"{format_method_name(method)} min" for method in methods]
            + [f"{format_method_name(method)} Overhead" for method in methods if method != "baseline"]
        ),
    ]
    for row in aggregate["summary"]:
        cells = [row["dataset"], row["fraction_tag"]]
        cells.extend(f"{row[f'{method}_runtime_min']:.2f}" for method in methods)
        cells.extend(
            f"{row[f'{method}_runtime_overhead_pct']:.2f}%" for method in methods if method != "baseline"
        )
        lines.append(markdown_row(cells))
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    runtime_csv = outdir / "runtime.csv"
    with runtime_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["dataset", "fraction", "fraction_tag"]
        fieldnames.extend(f"{method}_runtime_min" for method in methods)
        fieldnames.extend(f"{method}_runtime_overhead_pct" for method in methods if method != "baseline")
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in aggregate["summary"]:
            writer.writerow({key: row[key] for key in fieldnames})


def export_figures(
    *,
    outdir: Path,
    aggregate: dict,
    figure_seed: int,
    epoch_mode: str,
    fixed_epoch: int | None,
) -> None:
    figures_dir = outdir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Figure Index",
        "",
        figure_caption(epoch_mode=epoch_mode, figure_seed=figure_seed, fixed_epoch=fixed_epoch),
        "",
    ]

    run_lookup = {
        (row["dataset"], row["fraction"], row["method"], row["seed"]): row
        for row in aggregate["runs"]
    }

    for summary in aggregate["summary"]:
        dataset = summary["dataset"]
        fraction = summary["fraction"]
        fraction_tag = summary["fraction_tag"]
        lines.append(f"## {dataset} {fraction_tag}")
        lines.append("")
        for method in aggregate["config"]["methods"]:
            row = run_lookup.get((dataset, fraction, method, figure_seed))
            if row is None:
                continue
            figure_epoch = resolve_figure_epoch(row=row, epoch_mode=epoch_mode, fixed_epoch=fixed_epoch)
            if figure_epoch is None:
                continue
            src = Path(row["run_dir"]) / "plots" / f"samples_epoch_{int(figure_epoch):04d}.png"
            if not src.is_file():
                continue
            dst_name = f"{dataset}_{fraction_tag}_{method}.png"
            dst = figures_dir / dst_name
            shutil.copy2(src, dst)
            lines.append(f"- `{method}`: [{dst_name}]({dst_name})")
        lines.append("")

    (figures_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def cleanup_run_dirs(*, outdir: Path, datasets: list[str]) -> None:
    for dataset in datasets:
        path = outdir / dataset
        if path.exists():
            shutil.rmtree(path)


def fraction_to_tag(fraction: float) -> str:
    return f"{int(round(fraction * 100)):d}pct"


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values)


def percent_improvement(baseline: float, candidate: float) -> float:
    if baseline == 0:
        return 0.0
    return 100.0 * (baseline - candidate) / baseline


def percent_overhead(baseline: float, candidate: float) -> float:
    if baseline == 0:
        return 0.0
    return 100.0 * (candidate - baseline) / baseline


def load_method_configs(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def get_config_value(method_config: dict, key: str, fallback):
    return method_config.get(key, fallback)


def replace_command_arg(command: list[str], flag: str, value: str) -> None:
    try:
        index = command.index(flag)
    except ValueError as exc:
        raise ValueError(f"Flag {flag} not found in command.") from exc
    if index + 1 >= len(command):
        raise ValueError(f"Flag {flag} is missing its value.")
    command[index + 1] = value


def format_method_name(method: str) -> str:
    if method == "baseline":
        return "Baseline"
    if method == "wdro":
        return "WDRO"
    if method == "causal_wdro":
        return "Causal WDRO"
    return method.replace("_", " ").title()


def resolve_figure_epoch(*, row: dict, epoch_mode: str, fixed_epoch: int | None) -> int | None:
    if epoch_mode == "best":
        return row.get("best_epoch")
    if epoch_mode == "last":
        return row.get("last_epoch")
    if epoch_mode == "fixed":
        return fixed_epoch
    raise ValueError(f"Unsupported epoch_mode: {epoch_mode}")


def figure_caption(*, epoch_mode: str, figure_seed: int, fixed_epoch: int | None) -> str:
    if epoch_mode == "best":
        return f"Representative sample plots for `seed={figure_seed}` at each method's best-eval epoch."
    if epoch_mode == "last":
        return f"Representative sample plots for `seed={figure_seed}` at the final eval epoch."
    if epoch_mode == "fixed":
        return f"Representative sample plots for `seed={figure_seed}` at shared eval epoch `{fixed_epoch}`."
    raise ValueError(f"Unsupported epoch_mode: {epoch_mode}")


def markdown_header(columns: list[str]) -> str:
    align = ["---", "---:"] + ["---:"] * (len(columns) - 2)
    return markdown_row(columns) + "\n" + markdown_row(align)


def markdown_row(columns) -> str:
    return "| " + " | ".join(str(value) for value in columns) + " |"


if __name__ == "__main__":
    main()
