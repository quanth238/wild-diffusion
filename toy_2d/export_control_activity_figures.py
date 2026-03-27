from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export budget/control overview figures from control activity reports."
    )
    parser.add_argument("--report", type=Path, required=True, help="Path to control_activity/aggregate.json")
    parser.add_argument("--methods", nargs="+", default=["cdro_markov", "baseline_score", "wdro_score"])
    parser.add_argument("--outdir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    outdir = args.outdir or args.report.parent
    outdir.mkdir(parents=True, exist_ok=True)

    export_overview(report=report, methods=args.methods, outdir=outdir)
    export_cdro_activity(report=report, method="cdro_markov", outdir=outdir)


def export_overview(*, report: dict, methods: list[str], outdir: Path) -> None:
    rows = report["rows"]
    labels = [f"{row['dataset']} {row['fraction_tag']}" for row in rows]
    x = np.arange(len(labels), dtype=float)
    width = 0.8 / max(len(methods), 1)
    colors = {
        "cdro_markov": "#2ca02c",
        "baseline_score": "#1f77b4",
        "wdro_score": "#ff7f0e",
    }
    metric_specs = [
        ("best_swd_mean", "Best SWD", True),
        ("last_swd_mean", "Last SWD", True),
        ("active_budget_utilization_mean_mean", "Active Budget Utilization", False),
        ("active_control_norm_mean_mean", "Active Control Norm", False),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    axes = axes.reshape(2, 2)
    for ax, (suffix, title, lower_is_better) in zip(axes.flat, metric_specs):
        for idx, method in enumerate(methods):
            values = [float(row.get(f"{method}_{suffix}", 0.0) or 0.0) for row in rows]
            offsets = x + (idx - (len(methods) - 1) / 2.0) * width
            ax.bar(
                offsets,
                values,
                width=width,
                label=pretty_method_name(method),
                color=colors.get(method, None),
                alpha=0.9,
            )
        ax.set_title(title + (" (lower is better)" if lower_is_better else ""))
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.grid(alpha=0.2, axis="y")
    axes[0, 0].legend(frameon=False, ncol=len(methods), loc="upper center", bbox_to_anchor=(1.05, 1.25))
    fig.suptitle("Quality vs control-activity overview", fontsize=14)
    fig.tight_layout()
    fig.savefig(outdir / "quality_vs_activity_overview.png", dpi=180)
    plt.close(fig)


def export_cdro_activity(*, report: dict, method: str, outdir: Path) -> None:
    rows = report["rows"]
    labels = [f"{row['dataset']} {row['fraction_tag']}" for row in rows if f"{method}_best_swd_mean" in row]
    if not labels:
        return
    selected = [row for row in rows if f"{method}_best_swd_mean" in row]
    x = np.arange(len(selected), dtype=float)

    metrics = {
        "Active epochs": [float(row[f"{method}_adversary_update_epochs_mean"]) for row in selected],
        "Nontrivial fraction": [float(row[f"{method}_nontrivial_adversary_epoch_fraction_mean"]) for row in selected],
        "Budget utilization": [float(row[f"{method}_active_budget_utilization_mean_mean"]) for row in selected],
        "Control norm": [float(row[f"{method}_active_control_norm_mean_mean"]) for row in selected],
        "Diag score gap": [float(row[f"{method}_active_diag_score_loss_gap_mean_mean"]) for row in selected],
    }

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    axes = axes.reshape(3, 2)
    colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728", "#9467bd"]
    for ax, (title, values), color in zip(axes.flat, metrics.items(), colors):
        ax.bar(x, values, color=color, alpha=0.9)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.grid(alpha=0.2, axis="y")
    # Use last pane for quality summary.
    quality_ax = axes[2, 1]
    quality_ax.plot(x, [float(row[f"{method}_best_swd_mean"]) for row in selected], marker="o", color="#2ca02c", linewidth=1.8, label="best SWD")
    quality_ax.plot(x, [float(row[f"{method}_last_swd_mean"]) for row in selected], marker="o", color="#1f77b4", linewidth=1.8, label="last SWD")
    quality_ax.set_title("Quality summary")
    quality_ax.set_xticks(x)
    quality_ax.set_xticklabels(labels, rotation=35, ha="right")
    quality_ax.grid(alpha=0.2)
    quality_ax.legend(frameon=False)

    fig.suptitle(f"{pretty_method_name(method)} control activity profile", fontsize=14)
    fig.tight_layout()
    fig.savefig(outdir / f"{method}_activity_profile.png", dpi=180)
    plt.close(fig)


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


if __name__ == "__main__":
    main()
