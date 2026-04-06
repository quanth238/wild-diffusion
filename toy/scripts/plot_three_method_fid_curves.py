#!/usr/bin/env python3
import argparse
import csv
import json
import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt


STYLE_BY_METHOD = {
    "baseline_edm": {"label": "Baseline EDM", "marker": "o", "color": "tab:blue"},
    "wdro": {"label": "WDRO", "marker": "o", "color": "tab:orange"},
    "cdro": {"label": "CDRO", "marker": "o", "color": "tab:green"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge baseline, WDRO, and CDRO curve artifacts into three-method FID plots."
    )
    parser.add_argument("--combined-csv", type=str, default="")
    parser.add_argument("--wdro-compare-csv", type=str, default="")
    parser.add_argument("--cdro-compare-csv", type=str, default="")
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
        "baseline_compute_be",
        "weighted_compute_units",
        "baseline_weighted_compute_units",
        "train_wall_clock_sec",
        "baseline_train_wall_clock_sec_effective",
        "train_gpu_hours",
        "fid",
        "loss_final",
        "loss_mean_last",
        "fixed_warmup_steps",
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


def load_combined_rows(*, combined_csv: str) -> List[Dict]:
    rows = [normalize_row(row) for row in load_csv_rows(combined_csv)]
    rows = [row for row in rows if row.get("method") in ("baseline_edm", "wdro", "cdro")]
    rows.sort(
        key=lambda row: (
            str(row.get("method", "")),
            float(row.get("train_wall_clock_sec")) if row.get("train_wall_clock_sec") is not None else float("inf"),
            float(row.get("weighted_compute_units")) if row.get("weighted_compute_units") is not None else float("inf"),
            float(row.get("compute_budget_be")) if row.get("compute_budget_be") is not None else float("inf"),
        )
    )
    return rows


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


def _phase_boundary_x(*, rows: List[Dict], method: str, x_key: str) -> Optional[float]:
    robust_rows = [
        row
        for row in rows
        if row.get("method") == method and row.get("row_origin") == "trajectory_robust_phase"
    ]
    robust_rows.sort(key=lambda row: int(row.get("step", 0)))
    if robust_rows:
        first = robust_rows[0]
        if x_key == "train_wall_clock_sec":
            boundary = _safe_float(first.get("baseline_train_wall_clock_sec_effective"))
        elif x_key == "weighted_compute_units":
            boundary = _safe_float(first.get("baseline_weighted_compute_units"))
        elif x_key == "compute_budget_be":
            boundary = _safe_float(first.get("baseline_compute_be"))
            if boundary is None:
                boundary = _safe_float(first.get("fixed_warmup_steps"))
        else:
            boundary = None
        if boundary is not None:
            return float(boundary)
    warmup_rows = [
        row
        for row in rows
        if row.get("method") == method
        and row.get("row_origin") == "trajectory_warmup_phase"
        and row.get(x_key) is not None
    ]
    if not warmup_rows:
        return None
    return max(float(row[x_key]) for row in warmup_rows)


def _draw_phase_boundaries(*, ax, rows: List[Dict], x_key: str) -> None:
    if x_key not in ("train_wall_clock_sec", "weighted_compute_units", "compute_budget_be"):
        return
    for method in ("wdro", "cdro"):
        boundary_x = _phase_boundary_x(rows=rows, method=method, x_key=x_key)
        if boundary_x is None:
            continue
        ax.axvline(
            x=boundary_x,
            color=STYLE_BY_METHOD[method]["color"],
            linestyle="--",
            linewidth=1.4,
            alpha=0.75,
            label=f"{STYLE_BY_METHOD[method]['label']} warmup end",
        )


def _series_segments(*, rows: List[Dict], method: str, x_key: str, y_key: str) -> List[List[tuple]]:
    points = [
        (float(row[x_key]), float(row[y_key]), str(row.get("loss_kind", "")))
        for row in rows
        if row.get("method") == method and row.get(x_key) is not None and row.get(y_key) is not None
    ]
    if not points:
        return []
    points.sort(key=lambda point: float(point[0]))
    if not y_key.startswith("loss_") or method == "baseline_edm":
        return [[(x_value, y_value) for x_value, y_value, _ in points]]
    segments: List[List[tuple]] = []
    current_segment: List[tuple] = []
    previous_loss_kind: Optional[str] = None
    for x_value, y_value, loss_kind in points:
        if previous_loss_kind is not None and loss_kind != previous_loss_kind and current_segment:
            segments.append(list(current_segment))
            current_segment = []
        current_segment.append((x_value, y_value))
        previous_loss_kind = loss_kind
    if current_segment:
        segments.append(list(current_segment))
    return segments


def make_plot(
    *,
    path: str,
    rows: List[Dict],
    x_key: str,
    x_label: str,
    y_key: str,
    y_label: str,
    plot_label: str,
    title_suffix: str,
    train_percent_label: str,
) -> None:
    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    for method in ("baseline_edm", "wdro", "cdro"):
        segments = _series_segments(rows=rows, method=method, x_key=x_key, y_key=y_key)
        if not segments:
            continue
        for segment_index, segment in enumerate(segments):
            ax.plot(
                [value for value, _ in segment],
                [value for _, value in segment],
                marker=STYLE_BY_METHOD[method]["marker"],
                linewidth=2.0,
                color=STYLE_BY_METHOD[method]["color"],
                label=STYLE_BY_METHOD[method]["label"] if segment_index == 0 else "_nolegend_",
            )
    _draw_phase_boundaries(ax=ax, rows=rows, x_key=x_key)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"Simpsons-MNIST RGB {train_percent_label}: {plot_label} vs {title_suffix}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def make_dual_plot(
    *,
    path: str,
    rows: List[Dict],
    y_key: str,
    y_label: str,
    plot_label: str,
    train_percent_label: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_specs = (
        ("train_wall_clock_sec", "Train Wall-Clock (sec)", "Train Wall-Clock"),
        ("weighted_compute_units", "Weighted Compute Units", "Weighted Compute"),
    )
    for ax, (x_key, x_label, title_suffix) in zip(axes, plot_specs):
        for method in ("baseline_edm", "wdro", "cdro"):
            segments = _series_segments(rows=rows, method=method, x_key=x_key, y_key=y_key)
            if not segments:
                continue
            for segment_index, segment in enumerate(segments):
                ax.plot(
                    [value for value, _ in segment],
                    [value for _, value in segment],
                    marker=STYLE_BY_METHOD[method]["marker"],
                    linewidth=2.0,
                    color=STYLE_BY_METHOD[method]["color"],
                    label=STYLE_BY_METHOD[method]["label"] if segment_index == 0 else "_nolegend_",
                )
        _draw_phase_boundaries(ax=ax, rows=rows, x_key=x_key)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f"{plot_label} vs {title_suffix}")
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
    fig.suptitle(f"Simpsons-MNIST RGB {train_percent_label}: {plot_label} Comparison", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    combined_csv_arg = str(getattr(args, "combined_csv", "")).strip()
    wdro_compare_csv_arg = str(getattr(args, "wdro_compare_csv", "")).strip()
    cdro_compare_csv_arg = str(getattr(args, "cdro_compare_csv", "")).strip()
    if combined_csv_arg:
        if wdro_compare_csv_arg or cdro_compare_csv_arg:
            raise SystemExit(
                "[ERROR] Use either --combined-csv or the pair --wdro-compare-csv/--cdro-compare-csv, not both."
            )
    elif not (wdro_compare_csv_arg and cdro_compare_csv_arg):
        raise SystemExit(
            "[ERROR] Provide either --combined-csv or both --wdro-compare-csv and --cdro-compare-csv."
        )
    ensure_dir(args.outdir)
    rows = (
        load_combined_rows(combined_csv=combined_csv_arg)
        if combined_csv_arg
        else load_three_method_rows(
            wdro_compare_csv=wdro_compare_csv_arg,
            cdro_compare_csv=cdro_compare_csv_arg,
        )
    )
    combined_csv = os.path.join(args.outdir, f"{args.prefix}_three_method_compare.csv")
    summary_json = os.path.join(args.outdir, f"{args.prefix}_three_method_compare_summary.json")
    plot_wall_clock = os.path.join(args.outdir, f"{args.prefix}_fid_vs_train_wall_clock.png")
    plot_weighted = os.path.join(args.outdir, f"{args.prefix}_fid_vs_weighted_compute.png")
    plot_legacy = os.path.join(args.outdir, f"{args.prefix}_fid_vs_batch_equiv.png")
    plot_dual = os.path.join(args.outdir, f"{args.prefix}_fid_vs_wall_clock_and_weighted_compute.png")
    plot_loss_wall_clock = os.path.join(args.outdir, f"{args.prefix}_loss_mean_last_vs_train_wall_clock.png")
    plot_loss_weighted = os.path.join(args.outdir, f"{args.prefix}_loss_mean_last_vs_weighted_compute.png")
    plot_loss_dual = os.path.join(
        args.outdir,
        f"{args.prefix}_loss_mean_last_vs_wall_clock_and_weighted_compute.png",
    )

    write_csv(combined_csv, rows)
    make_plot(
        path=plot_wall_clock,
        rows=rows,
        x_key="train_wall_clock_sec",
        x_label="Train Wall-Clock (sec)",
        y_key="fid",
        y_label="FID",
        plot_label="FID",
        title_suffix="Train Wall-Clock",
        train_percent_label=str(args.train_percent_label),
    )
    make_plot(
        path=plot_weighted,
        rows=rows,
        x_key="weighted_compute_units",
        x_label="Weighted Compute Units",
        y_key="fid",
        y_label="FID",
        plot_label="FID",
        title_suffix="Weighted Compute",
        train_percent_label=str(args.train_percent_label),
    )
    make_plot(
        path=plot_legacy,
        rows=rows,
        x_key="compute_budget_be",
        x_label="Legacy Batch-Equivalent Denoiser Evals",
        y_key="fid",
        y_label="FID",
        plot_label="FID",
        title_suffix="Legacy Batch-Equivalent Compute",
        train_percent_label=str(args.train_percent_label),
    )
    make_dual_plot(
        path=plot_dual,
        rows=rows,
        y_key="fid",
        y_label="FID",
        plot_label="FID",
        train_percent_label=str(args.train_percent_label),
    )
    make_plot(
        path=plot_loss_wall_clock,
        rows=rows,
        x_key="train_wall_clock_sec",
        x_label="Train Wall-Clock (sec)",
        y_key="loss_mean_last",
        y_label="Loss (mean last)",
        plot_label="Loss (mean last)",
        title_suffix="Train Wall-Clock",
        train_percent_label=str(args.train_percent_label),
    )
    make_plot(
        path=plot_loss_weighted,
        rows=rows,
        x_key="weighted_compute_units",
        x_label="Weighted Compute Units",
        y_key="loss_mean_last",
        y_label="Loss (mean last)",
        plot_label="Loss (mean last)",
        title_suffix="Weighted Compute",
        train_percent_label=str(args.train_percent_label),
    )
    make_dual_plot(
        path=plot_loss_dual,
        rows=rows,
        y_key="loss_mean_last",
        y_label="Loss (mean last)",
        plot_label="Loss (mean last)",
        train_percent_label=str(args.train_percent_label),
    )
    summary = {
        "rows": rows,
        "sources": {
            "combined_csv": None if not combined_csv_arg else os.path.abspath(combined_csv_arg),
            "wdro_compare_csv": None if not wdro_compare_csv_arg else os.path.abspath(wdro_compare_csv_arg),
            "cdro_compare_csv": None if not cdro_compare_csv_arg else os.path.abspath(cdro_compare_csv_arg),
        },
        "notes": {
            "loss_metric_warning": (
                "Baseline uses baseline denoising loss, while WDRO/CDRO use robust outer loss. "
                "Loss plots are optimization diagnostics, not a primary cross-method fairness metric."
            )
        },
        "plot_paths": {
            "train_wall_clock_sec": plot_wall_clock,
            "weighted_compute_units": plot_weighted,
            "wall_clock_and_weighted_compute": plot_dual,
            "batch_equiv_denoiser_evals": plot_legacy,
            "loss_mean_last_train_wall_clock_sec": plot_loss_wall_clock,
            "loss_mean_last_weighted_compute_units": plot_loss_weighted,
            "loss_mean_last_wall_clock_and_weighted_compute": plot_loss_dual,
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
    print(f"[three-method] wrote {plot_loss_wall_clock}", flush=True)
    print(f"[three-method] wrote {plot_loss_weighted}", flush=True)
    print(f"[three-method] wrote {plot_loss_dual}", flush=True)


if __name__ == "__main__":
    main()
