from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-evaluate all cdro_markov runs in a comparison directory and summarize the deltas."
    )
    parser.add_argument("--comparison-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--fraction-tags", nargs="*", default=None)
    parser.add_argument("--checkpoint-modes", nargs="+", choices=("best", "last"), default=["best", "last"])
    parser.add_argument("--device", type=str, default="auto", choices=("cpu", "cuda", "auto"))
    parser.add_argument("--seed", type=int, default=None)
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
        default=True,
        help="Skip plot/sample exports during reevaluation.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Re-run checkpoints even if reeval summaries already exist.",
    )
    parser.add_argument("--run-out-subdir", type=str, default="reeval")
    parser.add_argument("--report-subdir", type=str, default="reeval_cdro_markov")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dirs = discover_run_dirs(
        comparison_dir=args.comparison_dir,
        datasets=args.datasets,
        fraction_tags=args.fraction_tags,
    )
    if not run_dirs:
        raise SystemExit("No cdro_markov run directories found.")

    results: list[dict] = []
    total = len(run_dirs) * len(args.checkpoint_modes)
    item_index = 0
    for run_dir in run_dirs:
        for checkpoint_mode in args.checkpoint_modes:
            item_index += 1
            checkpoint_path = checkpoint_path_for_mode(run_dir=run_dir, checkpoint_mode=checkpoint_mode)
            outdir = run_dir / args.run_out_subdir / checkpoint_mode
            summary_path = outdir / "summary.json"
            if not checkpoint_path.is_file():
                print(f"[{item_index}/{total}] skip {run_dir} {checkpoint_mode} (missing checkpoint)")
                results.append(
                    {
                        "run_dir": str(run_dir),
                        "dataset": run_dir.parents[2].name,
                        "fraction_tag": run_dir.parents[1].name,
                        "seed": run_dir.name,
                        "checkpoint_mode": checkpoint_mode,
                        "checkpoint_path": str(checkpoint_path),
                        "status": "missing_checkpoint",
                        "summary_path": None,
                        "summary": None,
                    }
                )
                continue
            if summary_path.is_file() and not args.overwrite:
                print(f"[{item_index}/{total}] skip {run_dir} {checkpoint_mode} (existing summary)")
            else:
                print(f"[{item_index}/{total}] reeval {run_dir} {checkpoint_mode}")
                run_reeval_command(
                    run_dir=run_dir,
                    checkpoint_mode=checkpoint_mode,
                    outdir=outdir,
                    args=args,
                )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            results.append(
                {
                    "run_dir": str(run_dir),
                    "dataset": run_dir.parents[2].name,
                    "fraction_tag": run_dir.parents[1].name,
                    "seed": run_dir.name,
                    "checkpoint_mode": checkpoint_mode,
                    "checkpoint_path": str(checkpoint_path),
                    "status": "evaluated",
                    "summary_path": str(summary_path),
                    "summary": summary,
                }
            )

    report = build_report(results=results, comparison_dir=args.comparison_dir)
    report_dir = args.comparison_dir / args.report_subdir
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "aggregate.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (report_dir / "summary.md").write_text(render_markdown(report), encoding="utf-8")
    print(f"Wrote report to {report_dir / 'summary.md'}")


def discover_run_dirs(
    *,
    comparison_dir: Path,
    datasets: list[str] | None,
    fraction_tags: list[str] | None,
) -> list[Path]:
    dataset_names = datasets or [path.name for path in sorted(comparison_dir.iterdir()) if path.is_dir()]
    run_dirs: list[Path] = []
    for dataset in dataset_names:
        dataset_dir = comparison_dir / dataset
        if not dataset_dir.is_dir():
            continue
        tags = fraction_tags or [path.name for path in sorted(dataset_dir.iterdir()) if path.is_dir()]
        for fraction_tag in tags:
            method_dir = dataset_dir / fraction_tag / "cdro_markov"
            if not method_dir.is_dir():
                continue
            seed_dirs = sorted(path for path in method_dir.iterdir() if path.is_dir() and path.name.startswith("seed"))
            run_dirs.extend(seed_dirs)
    return run_dirs


def run_reeval_command(
    *,
    run_dir: Path,
    checkpoint_mode: str,
    outdir: Path,
    args: argparse.Namespace,
) -> None:
    command = [
        sys.executable,
        "-m",
        "toy_2d.eval_cdro_markov_checkpoint",
        "--run-dir",
        str(run_dir),
        "--checkpoint-mode",
        checkpoint_mode,
        "--outdir",
        str(outdir),
        "--device",
        args.device,
    ]
    if args.seed is not None:
        command += ["--seed", str(args.seed)]
    if args.num_eval_samples is not None:
        command += ["--num-eval-samples", str(args.num_eval_samples)]
    if args.metric_samples is not None:
        command += ["--metric-samples", str(args.metric_samples)]
    if args.num_snapshot_steps is not None:
        command += ["--num-snapshot-steps", str(args.num_snapshot_steps)]
    if args.diagnostic_batch_size is not None:
        command += ["--diagnostic-batch-size", str(args.diagnostic_batch_size)]
    if args.reverse_solver is not None:
        command += ["--reverse-solver", args.reverse_solver]
    if args.terminal_sampler is not None:
        command += ["--terminal-sampler", args.terminal_sampler]
    if args.terminal_jitter_scale is not None:
        command += ["--terminal-jitter-scale", str(args.terminal_jitter_scale)]
    if args.reverse_noise_scale is not None:
        command += ["--reverse-noise-scale", str(args.reverse_noise_scale)]
    if args.reverse_control_scale is not None:
        command += ["--reverse-control-scale", str(args.reverse_control_scale)]
    if args.reverse_tail_noise_scale is not None:
        command += ["--reverse-tail-noise-scale", str(args.reverse_tail_noise_scale)]
    if args.reverse_deterministic_tail_steps is not None:
        command += ["--reverse-deterministic-tail-steps", str(args.reverse_deterministic_tail_steps)]
    if args.log_reverse_ablation is not None:
        command.append("--log-reverse-ablation" if args.log_reverse_ablation else "--no-log-reverse-ablation")
    command.append("--fast-tuning" if args.fast_tuning else "--no-fast-tuning")

    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Re-eval failed for {run_dir} {checkpoint_mode}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )


def checkpoint_path_for_mode(*, run_dir: Path, checkpoint_mode: str) -> Path:
    return run_dir / f"checkpoint_{checkpoint_mode}.pt"


def build_report(*, results: list[dict], comparison_dir: Path) -> dict:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for item in results:
        grouped[(item["checkpoint_mode"], item["dataset"], item["fraction_tag"])].append(item)

    rows: list[dict] = []
    for (checkpoint_mode, dataset, fraction_tag), group in sorted(grouped.items()):
        evaluated_group = [item for item in group if item["status"] == "evaluated"]
        skipped_group = [item for item in group if item["status"] != "evaluated"]
        checkpoint_gain_swd = [
            float(item["summary"]["checkpoint_metrics"]["reverse_control_gain_swd"]) for item in evaluated_group
        ]
        reeval_gain_swd = [float(item["summary"]["reeval_metrics"]["reverse_control_gain_swd"]) for item in evaluated_group]
        checkpoint_gain_mmd = [
            float(item["summary"]["checkpoint_metrics"]["reverse_control_gain_mmd"]) for item in evaluated_group
        ]
        reeval_gain_mmd = [float(item["summary"]["reeval_metrics"]["reverse_control_gain_mmd"]) for item in evaluated_group]
        reverse_control_scales = [
            float(item["summary"]["reeval_metrics"]["reverse_control_scale"])
            for item in evaluated_group
        ]
        rows.append(
            {
                "checkpoint_mode": checkpoint_mode,
                "dataset": dataset,
                "fraction_tag": fraction_tag,
                "num_requested": len(group),
                "num_evaluated": len(evaluated_group),
                "num_skipped": len(skipped_group),
                "skipped_statuses": sorted(set(item["status"] for item in skipped_group)),
                "checkpoint_gain_swd_mean": mean(checkpoint_gain_swd) if checkpoint_gain_swd else None,
                "reeval_gain_swd_mean": mean(reeval_gain_swd) if reeval_gain_swd else None,
                "gain_swd_delta_mean": (
                    mean(reeval - old for reeval, old in zip(reeval_gain_swd, checkpoint_gain_swd))
                    if checkpoint_gain_swd
                    else None
                ),
                "checkpoint_gain_mmd_mean": mean(checkpoint_gain_mmd) if checkpoint_gain_mmd else None,
                "reeval_gain_mmd_mean": mean(reeval_gain_mmd) if reeval_gain_mmd else None,
                "gain_mmd_delta_mean": (
                    mean(reeval - old for reeval, old in zip(reeval_gain_mmd, checkpoint_gain_mmd))
                    if checkpoint_gain_mmd
                    else None
                ),
                "reverse_control_scale_values": sorted(set(reverse_control_scales)),
                "shared_randomness_all": bool(evaluated_group) and all(
                    bool(item["summary"]["reeval_metrics"].get("reverse_ablation_shared_randomness", False))
                    for item in evaluated_group
                ),
            }
        )

    return {
        "comparison_dir": str(comparison_dir),
        "num_runs_requested": len(results),
        "num_runs_evaluated": sum(1 for item in results if item["status"] == "evaluated"),
        "num_runs_skipped": sum(1 for item in results if item["status"] != "evaluated"),
        "rows": rows,
    }


def render_markdown(report: dict) -> str:
    lines = [
        "## CDRO Markov Re-eval",
        "",
        f"Comparison dir: `{report['comparison_dir']}`",
        "",
        f"Runs requested: {report['num_runs_requested']}",
        "",
        f"Runs evaluated: {report['num_runs_evaluated']}",
        "",
        f"Runs skipped: {report['num_runs_skipped']}",
        "",
    ]
    for checkpoint_mode in ("best", "last"):
        mode_rows = [row for row in report["rows"] if row["checkpoint_mode"] == checkpoint_mode]
        if not mode_rows:
            continue
        lines += [
            f"## {checkpoint_mode.title()} Checkpoints",
            "",
            "| Dataset | Data | Eval/Skip | Old Gain SWD | New Gain SWD | Delta | Old Gain MMD | New Gain MMD | Control Scale | Shared RNG |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
        for row in mode_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        row["dataset"],
                        row["fraction_tag"],
                        f"{row['num_evaluated']}/{row['num_skipped']}",
                        format_optional_float(row["checkpoint_gain_swd_mean"]),
                        format_optional_float(row["reeval_gain_swd_mean"]),
                        format_optional_float(row["gain_swd_delta_mean"]),
                        format_optional_float(row["checkpoint_gain_mmd_mean"]),
                        format_optional_float(row["reeval_gain_mmd_mean"]),
                        ", ".join(f"{value:.3f}" for value in row["reverse_control_scale_values"]) or "-",
                        format_shared_randomness(row),
                    ]
                )
                + " |"
            )
            if row["skipped_statuses"]:
                lines.append(
                    f"Skipped {row['dataset']} {row['fraction_tag']} {checkpoint_mode}: "
                    + ", ".join(row["skipped_statuses"])
                )
        lines.append("")
    return "\n".join(lines)


def format_optional_float(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.6f}"


def format_shared_randomness(row: dict) -> str:
    if row["num_evaluated"] == 0:
        return "-"
    return "yes" if row["shared_randomness_all"] else "no"


if __name__ == "__main__":
    main()
