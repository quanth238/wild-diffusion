from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


DEFAULT_METHODS = ["cdro_markov"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize adversary/control activity for comparison runs."
    )
    parser.add_argument("--comparison-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--fraction-tags", nargs="*", default=None)
    parser.add_argument("--out-subdir", type=str, default="control_activity")
    parser.add_argument("--control-threshold", type=float, default=1e-6)
    parser.add_argument("--cost-threshold", type=float, default=1e-10)
    parser.add_argument("--flag-control-norm-below", type=float, default=0.02)
    parser.add_argument("--flag-utilization-below", type=float, default=0.005)
    parser.add_argument("--flag-gap-below", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = collect_rows(
        comparison_dir=args.comparison_dir,
        methods=args.methods,
        datasets=args.datasets,
        fraction_tags=args.fraction_tags,
        control_threshold=float(args.control_threshold),
        cost_threshold=float(args.cost_threshold),
        flag_control_norm_below=float(args.flag_control_norm_below),
        flag_utilization_below=float(args.flag_utilization_below),
        flag_gap_below=float(args.flag_gap_below),
    )
    report = {
        "comparison_dir": str(args.comparison_dir),
        "methods": args.methods,
        "rows": rows,
        "winner_counts": compute_flag_counts(rows),
    }
    outdir = args.comparison_dir / args.out_subdir
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "aggregate.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (outdir / "summary.md").write_text(render_markdown(report), encoding="utf-8")


def collect_rows(
    *,
    comparison_dir: Path,
    methods: list[str],
    datasets: list[str] | None,
    fraction_tags: list[str] | None,
    control_threshold: float,
    cost_threshold: float,
    flag_control_norm_below: float,
    flag_utilization_below: float,
    flag_gap_below: float,
) -> list[dict]:
    dataset_names = datasets or discover_datasets(comparison_dir=comparison_dir, methods=methods)
    rows: list[dict] = []
    for dataset in dataset_names:
        tags = fraction_tags or discover_fraction_tags(
            comparison_dir=comparison_dir,
            dataset=dataset,
            methods=methods,
        )
        for fraction_tag in tags:
            row: dict[str, object] = {"dataset": dataset, "fraction_tag": fraction_tag}
            any_method = False
            for method in methods:
                method_dir = comparison_dir / dataset / fraction_tag / method
                if not method_dir.is_dir():
                    continue
                seed_dirs = sorted(path for path in method_dir.iterdir() if path.is_dir() and path.name.startswith("seed"))
                if not seed_dirs:
                    continue
                any_method = True
                seed_stats = [
                    summarize_seed(
                        seed_dir=seed_dir,
                        control_threshold=control_threshold,
                        cost_threshold=cost_threshold,
                        flag_control_norm_below=flag_control_norm_below,
                        flag_utilization_below=flag_utilization_below,
                        flag_gap_below=flag_gap_below,
                    )
                    for seed_dir in seed_dirs
                ]
                row.update(aggregate_method_stats(method=method, seed_stats=seed_stats))
            if any_method:
                rows.append(row)
    return rows


def discover_datasets(*, comparison_dir: Path, methods: list[str]) -> list[str]:
    datasets: list[str] = []
    for path in sorted(comparison_dir.iterdir()):
        if not path.is_dir():
            continue
        if any((fraction / method).exists() for fraction in path.iterdir() if fraction.is_dir() for method in methods):
            datasets.append(path.name)
    return datasets


def discover_fraction_tags(*, comparison_dir: Path, dataset: str, methods: list[str]) -> list[str]:
    dataset_dir = comparison_dir / dataset
    fraction_tags: list[str] = []
    for path in sorted(dataset_dir.iterdir()):
        if not path.is_dir():
            continue
        if any((path / method).exists() for method in methods):
            fraction_tags.append(path.name)
    return fraction_tags


def summarize_seed(
    *,
    seed_dir: Path,
    control_threshold: float,
    cost_threshold: float,
    flag_control_norm_below: float,
    flag_utilization_below: float,
    flag_gap_below: float,
) -> dict:
    summary_path = seed_dir / "summary.json"
    metrics_path = seed_dir / "metrics.jsonl"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = load_metrics(metrics_path)

    best_swd = float(summary["best_sliced_wasserstein"])
    last_swd = float(summary["last_eval"]["sliced_wasserstein"])

    active_rows = [row for row in metrics if row.get("adversary_update_active", False)]
    nontrivial_rows = [
        row
        for row in active_rows
        if float(row.get("mean_control_norm", 0.0)) > control_threshold
        or float(row.get("control_cost", 0.0)) > cost_threshold
    ]
    active_eval_rows = [row for row in active_rows if "diag_score_loss_gap" in row]

    stats = {
        "best_swd": best_swd,
        "last_swd": last_swd,
        "runtime_minutes": float(summary.get("runtime_minutes", 0.0) or 0.0),
        "adversary_update_epochs": int(summary.get("adversary_update_epochs", len(active_rows))),
        "nontrivial_adversary_epochs": int(
            summary.get("nontrivial_adversary_epochs", len(nontrivial_rows))
        ),
        "adversary_update_epoch_fraction": float(
            summary.get(
                "adversary_update_epoch_fraction",
                len(active_rows) / max(len(metrics), 1),
            )
        ),
        "nontrivial_adversary_epoch_fraction": float(
            summary.get(
                "nontrivial_adversary_epoch_fraction",
                len(nontrivial_rows) / max(len(active_rows), 1),
            )
        ),
        "active_budget_utilization_mean": float(
            summary.get(
                "active_budget_utilization_mean",
                mean([float(row.get("budget_utilization", 0.0)) for row in active_rows]) if active_rows else 0.0,
            )
        ),
        "active_budget_utilization_max": float(
            summary.get(
                "active_budget_utilization_max",
                max((float(row.get("budget_utilization", 0.0)) for row in active_rows), default=0.0),
            )
        ),
        "active_control_cost_mean": float(
            summary.get(
                "active_control_cost_mean",
                mean([float(row.get("control_cost", 0.0)) for row in active_rows]) if active_rows else 0.0,
            )
        ),
        "active_control_norm_mean": float(
            summary.get(
                "active_control_norm_mean",
                mean([float(row.get("mean_control_norm", 0.0)) for row in active_rows]) if active_rows else 0.0,
            )
        ),
        "active_control_norm_max": float(
            summary.get(
                "active_control_norm_max",
                max((float(row.get("max_control_norm", 0.0)) for row in active_rows), default=0.0),
            )
        ),
        "active_diag_score_loss_gap_mean": float(
            summary.get(
                "active_diag_score_loss_gap_mean",
                mean([float(row.get("diag_score_loss_gap", 0.0)) for row in active_eval_rows]) if active_eval_rows else 0.0,
            )
        ),
    }
    stats["collapse_risk"] = is_collapse_risk(
        stats=stats,
        flag_control_norm_below=flag_control_norm_below,
        flag_utilization_below=flag_utilization_below,
        flag_gap_below=flag_gap_below,
    )
    return stats


def load_metrics(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def is_collapse_risk(
    *,
    stats: dict,
    flag_control_norm_below: float,
    flag_utilization_below: float,
    flag_gap_below: float,
) -> bool:
    return (
        float(stats["active_control_norm_mean"]) < flag_control_norm_below
        and float(stats["active_budget_utilization_mean"]) < flag_utilization_below
        and abs(float(stats["active_diag_score_loss_gap_mean"])) < flag_gap_below
    )


def aggregate_method_stats(*, method: str, seed_stats: list[dict]) -> dict:
    prefix = f"{method}_"
    out: dict[str, object] = {
        f"{prefix}num_seeds": len(seed_stats),
        f"{prefix}collapse_risk_seeds": sum(1 for row in seed_stats if row["collapse_risk"]),
        f"{prefix}collapse_risk_fraction": safe_mean([1.0 if row["collapse_risk"] else 0.0 for row in seed_stats]),
    }
    numeric_keys = [
        "best_swd",
        "last_swd",
        "runtime_minutes",
        "adversary_update_epochs",
        "nontrivial_adversary_epochs",
        "adversary_update_epoch_fraction",
        "nontrivial_adversary_epoch_fraction",
        "active_budget_utilization_mean",
        "active_budget_utilization_max",
        "active_control_cost_mean",
        "active_control_norm_mean",
        "active_control_norm_max",
        "active_diag_score_loss_gap_mean",
    ]
    for key in numeric_keys:
        values = [float(row[key]) for row in seed_stats]
        out[f"{prefix}{key}_mean"] = safe_mean(values)
        out[f"{prefix}{key}_min"] = min(values) if values else 0.0
        out[f"{prefix}{key}_max"] = max(values) if values else 0.0
    out[f"{prefix}seed_stats"] = seed_stats
    return out


def safe_mean(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def compute_flag_counts(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        for key, value in row.items():
            if key.endswith("_collapse_risk_seeds"):
                method = key[: -len("_collapse_risk_seeds")]
                if int(value) > 0:
                    counts[method] += 1
    return dict(counts)


def render_markdown(report: dict) -> str:
    methods = report["methods"]
    lines: list[str] = [
        "## Control Activity",
        "",
        "Rows below summarize whether the adversary was materially active during training.",
        "",
    ]
    for method in methods:
        header = [
            "Dataset",
            "Data",
            "Best SWD",
            "Last SWD",
            "Active Epochs",
            "Nontrivial Fraction",
            "Budget Util",
            "Control Norm",
            "Diag Gap",
            "Collapse Risk",
        ]
        lines.extend([f"### {pretty_method_name(method)}", "", markdown_header(header)])
        prefix = f"{method}_"
        for row in report["rows"]:
            if f"{prefix}num_seeds" not in row:
                continue
            lines.append(
                markdown_row(
                    [
                        row["dataset"],
                        row["fraction_tag"],
                        f"{row[f'{prefix}best_swd_mean']:.4f}",
                        f"{row[f'{prefix}last_swd_mean']:.4f}",
                        f"{row[f'{prefix}adversary_update_epochs_mean']:.1f}",
                        f"{row[f'{prefix}nontrivial_adversary_epoch_fraction_mean']:.3f}",
                        f"{row[f'{prefix}active_budget_utilization_mean_mean']:.4f}",
                        f"{row[f'{prefix}active_control_norm_mean_mean']:.4f}",
                        f"{row[f'{prefix}active_diag_score_loss_gap_mean_mean']:.4f}",
                        f"{int(row[f'{prefix}collapse_risk_seeds'])}/{int(row[f'{prefix}num_seeds'])}",
                    ]
                )
            )
        lines.extend([""])
    lines.extend(
        [
            "## Flag Counts",
            "",
            "Counts below show how many dataset/fraction rows had at least one seed flagged as a collapse risk.",
            "",
            json.dumps(report["winner_counts"], indent=2),
            "",
        ]
    )
    return "\n".join(lines)


def pretty_method_name(method: str) -> str:
    mapping = {
        "baseline": "Baseline",
        "wdro": "WDRO",
        "cdro": "Legacy CDRO",
        "baseline_score": "Baseline Score",
        "wdro_score": "WDRO Score",
        "cdro_markov": "CDRO Markov",
    }
    return mapping.get(method, method.replace("_", " ").title())


def markdown_row(values: list[object]) -> str:
    return "| " + " | ".join(str(value) for value in values) + " |"


def markdown_header(values: list[str]) -> str:
    align = ["---", "---:"] + ["---:"] * (len(values) - 2)
    return markdown_row(values) + "\n" + markdown_row(align)


if __name__ == "__main__":
    main()
