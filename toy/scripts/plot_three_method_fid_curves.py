#!/usr/bin/env python3
import argparse
import csv
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge baseline, WDRO, and CDRO curve artifacts into three-method FID plots."
    )
    parser.add_argument("--wdro-compare-csv", type=str, required=True)
    parser.add_argument("--cdro-compare-csv", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--prefix", type=str, required=True)
    parser.add_argument("--train-percent-label", type=str, default="1%")
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_csv_rows(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _safe_float(value):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_row(row: Dict) -> Dict:
    normalized = dict(row)
    normalized["step"] = int(float(row["step"]))
    for key in (
        "compute_budget_be",
        "weighted_compute_units",
        "train_wall_clock_sec",
        "train_gpu_hours",
        "fid",
    ):
        normalized[key] = _safe_float(row.get(key))
    return normalized


def load_three_method_rows(*, wdro_compare_csv: str, cdro_compare_csv: str) -> List[Dict]:
    wdro_rows = [normalize_row(row) for row in load_csv_rows(wdro_compare_csv)]
    cdro_rows = [normalize_row(row) for row in load_csv_rows(cdro_compare_csv)]

    baseline_by_step: Dict[int, Dict] = {}
    for row in wdro_rows + cdro_rows:
        if row.get("method") != "baseline_edm":
            continue
        step = int(row["step"])
        if step not in baseline_by_step:
            baseline_by_step[step] = row

    merged_rows = list(baseline_by_step.values())
    merged_rows.extend(row for row in wdro_rows if row.get("method") == "wdro")
    merged_rows.extend(row for row in cdro_rows if row.get("method") == "cdro")
    merged_rows.sort(
        key=lambda row: (
            str(row.get("method", "")),
            float(row.get("train_wall_clock_sec")) if row.get("train_wall_clock_sec") is not None else float("inf"),
            float(row.get("weighted_compute_units")) if row.get("weighted_compute_units") is not None else float("inf"),
            float(row.get("compute_budget_be")) if row.get("compute_budget_be") is not None else float("inf"),
        )
    )
    return merged_rows


def write_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_plot(
    *,
    path: str,
    rows: List[Dict],
    x_key: str,
    x_label: str,
    title_suffix: str,
    train_percent_label: str,
) -> None:
    plt.figure(figsize=(8, 5))
    style_by_method = {
        "baseline_edm": {"label": "Baseline EDM", "marker": "o"},
        "wdro": {"label": "WDRO", "marker": "o"},
        "cdro": {"label": "CDRO", "marker": "o"},
    }
    for method in ("baseline_edm", "wdro", "cdro"):
        points = [
            (row[x_key], row["fid"])
            for row in rows
            if row.get("method") == method and row.get(x_key) is not None and row.get("fid") is not None
        ]
        if not points:
            continue
        points.sort(key=lambda point: float(point[0]))
        plt.plot(
            [value for value, _ in points],
            [fid for _, fid in points],
            marker=style_by_method[method]["marker"],
            linewidth=2.0,
            label=style_by_method[method]["label"],
        )
    plt.xlabel(x_label)
    plt.ylabel("FID")
    plt.title(f"Simpsons-MNIST RGB {train_percent_label}: FID vs {title_suffix}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def make_dual_plot(*, path: str, rows: List[Dict], train_percent_label: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_specs = (
        ("train_wall_clock_sec", "Train Wall-Clock (sec)", "Train Wall-Clock"),
        ("weighted_compute_units", "Weighted Compute Units", "Weighted Compute"),
    )
    style_by_method = {
        "baseline_edm": {"label": "Baseline EDM", "marker": "o"},
        "wdro": {"label": "WDRO", "marker": "o"},
        "cdro": {"label": "CDRO", "marker": "o"},
    }
    for ax, (x_key, x_label, title_suffix) in zip(axes, plot_specs):
        for method in ("baseline_edm", "wdro", "cdro"):
            points = [
                (row[x_key], row["fid"])
                for row in rows
                if row.get("method") == method and row.get(x_key) is not None and row.get("fid") is not None
            ]
            if not points:
                continue
            points.sort(key=lambda point: float(point[0]))
            ax.plot(
                [value for value, _ in points],
                [fid for _, fid in points],
                marker=style_by_method[method]["marker"],
                linewidth=2.0,
                label=style_by_method[method]["label"],
            )
        ax.set_xlabel(x_label)
        ax.set_ylabel("FID")
        ax.set_title(f"FID vs {title_suffix}")
        ax.grid(True, alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.965),
            ncol=3,
            frameon=False,
        )
    fig.suptitle(f"Simpsons-MNIST RGB {train_percent_label}: Three-Method Comparison", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ensure_dir(args.outdir)
    rows = load_three_method_rows(
        wdro_compare_csv=args.wdro_compare_csv,
        cdro_compare_csv=args.cdro_compare_csv,
    )
    combined_csv = os.path.join(args.outdir, f"{args.prefix}_three_method_compare.csv")
    summary_json = os.path.join(args.outdir, f"{args.prefix}_three_method_compare_summary.json")
    plot_wall_clock = os.path.join(args.outdir, f"{args.prefix}_fid_vs_train_wall_clock.png")
    plot_weighted = os.path.join(args.outdir, f"{args.prefix}_fid_vs_weighted_compute.png")
    plot_legacy = os.path.join(args.outdir, f"{args.prefix}_fid_vs_batch_equiv.png")
    plot_dual = os.path.join(args.outdir, f"{args.prefix}_fid_vs_wall_clock_and_weighted_compute.png")

    write_csv(combined_csv, rows)
    make_plot(
        path=plot_wall_clock,
        rows=rows,
        x_key="train_wall_clock_sec",
        x_label="Train Wall-Clock (sec)",
        title_suffix="Train Wall-Clock",
        train_percent_label=str(args.train_percent_label),
    )
    make_plot(
        path=plot_weighted,
        rows=rows,
        x_key="weighted_compute_units",
        x_label="Weighted Compute Units",
        title_suffix="Weighted Compute",
        train_percent_label=str(args.train_percent_label),
    )
    make_plot(
        path=plot_legacy,
        rows=rows,
        x_key="compute_budget_be",
        x_label="Legacy Batch-Equivalent Denoiser Evals",
        title_suffix="Legacy Batch-Equivalent Compute",
        train_percent_label=str(args.train_percent_label),
    )
    make_dual_plot(
        path=plot_dual,
        rows=rows,
        train_percent_label=str(args.train_percent_label),
    )
    summary = {
        "rows": rows,
        "sources": {
            "wdro_compare_csv": os.path.abspath(args.wdro_compare_csv),
            "cdro_compare_csv": os.path.abspath(args.cdro_compare_csv),
        },
        "plot_paths": {
            "train_wall_clock_sec": plot_wall_clock,
            "weighted_compute_units": plot_weighted,
            "wall_clock_and_weighted_compute": plot_dual,
            "batch_equiv_denoiser_evals": plot_legacy,
        },
        "combined_csv": combined_csv,
    }
    with open(summary_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"[three-method] wrote {combined_csv}", flush=True)
    print(f"[three-method] wrote {summary_json}", flush=True)
    print(f"[three-method] wrote {plot_wall_clock}", flush=True)
    print(f"[three-method] wrote {plot_weighted}", flush=True)
    print(f"[three-method] wrote {plot_dual}", flush=True)
    print(f"[three-method] wrote {plot_legacy}", flush=True)


if __name__ == "__main__":
    main()
