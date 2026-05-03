#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt

from comparison_outputs import mirror_many


STYLE = {
    "baseline": {"label": "Baseline EDM", "color": "tab:blue"},
    "wild_diffusion": {"label": "Wild-Diffusion EDM", "color": "tab:orange"},
    "cdro": {"label": "CDRO EDM", "color": "tab:green"},
}
STYLE_ORDER = ["baseline", "wild_diffusion", "cdro"]
AXIS_SPECS = {
    "train_wall_clock_sec": {
        "label": "Train Wall-Clock (sec)",
        "title": "FID vs Train Wall-Clock (Zoom)",
    },
    "total_train_pflops": {
        "label": "Train Compute (PFLOPs)",
        "title": "FID vs Train FLOPs (Zoom)",
    },
    "images_shown_m": {
        "label": "Clean Images Shown (MIMG)",
        "title": "FID vs Clean Images Shown (Zoom)",
    },
    "weighted_compute_units": {
        "label": "Weighted Compute Units",
        "title": "FID vs Weighted Compute (Zoom)",
    },
}
DEFAULT_X_AXES = "train_wall_clock_sec,total_train_pflops,images_shown_m"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot a zoomed FID comparison for CIFAR baseline and robust "
            "variants using a merged comparison CSV."
        )
    )
    parser.add_argument("--manifest-csv", type=str, required=True)
    parser.add_argument("--out-png", type=str, required=True)
    parser.add_argument("--out-csv", type=str, default="")
    parser.add_argument("--dataset-label", type=str, default="CIFAR-10")
    parser.add_argument("--train-percent-label", type=str, default="20%")
    parser.add_argument("--baseline-zoom-start-kimg", type=float, default=20000.0)
    parser.add_argument("--comparison-graphs-dir", type=str, default="")
    parser.add_argument("--no-comparison-graph-mirror", action="store_true")
    parser.add_argument(
        "--x-axes",
        type=str,
        default=DEFAULT_X_AXES,
        help=(
            "Comma-separated x-axis keys. Supported keys: "
            f"{','.join(AXIS_SPECS.keys())}. Use 'all' to plot every supported axis."
        ),
    )
    return parser.parse_args()


def load_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _safe_float(value):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_axes(text: str) -> List[str]:
    requested = [item.strip() for item in str(text).split(",") if item.strip()]
    if not requested:
        requested = DEFAULT_X_AXES.split(",")
    if len(requested) == 1 and requested[0].lower() == "all":
        return list(AXIS_SPECS.keys())
    unknown = [item for item in requested if item not in AXIS_SPECS]
    if unknown:
        raise ValueError(f"Unsupported x-axis key(s): {unknown}. Supported: {sorted(AXIS_SPECS)}")
    out: List[str] = []
    for item in requested:
        if item not in out:
            out.append(item)
    return out


def _float_or_blank(row: Dict[str, str], key: str):
    value = _safe_float(row.get(key))
    return "" if value is None else float(value)


def build_rows(manifest_rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in manifest_rows:
        robust_method = str(row.get("robust_method", "")).strip()
        if robust_method not in STYLE:
            continue
        fid_value = _safe_float(row.get("fid"))
        if fid_value is None:
            continue
        baseline_train_flops = _safe_float(row.get("baseline_train_flops"))
        out.append(
            {
                "robust_method": robust_method,
                "series_label": STYLE[robust_method]["label"],
                "step": int(float(row["step"])),
                "fid": float(fid_value),
                "images_shown_m": _float_or_blank(row, "images_shown_m"),
                "train_wall_clock_sec": _float_or_blank(row, "train_wall_clock_sec"),
                "weighted_compute_units": _float_or_blank(row, "weighted_compute_units"),
                "total_train_pflops": _float_or_blank(row, "total_train_pflops"),
                "baseline_train_wall_clock_sec_effective": _float_or_blank(
                    row, "baseline_train_wall_clock_sec_effective"
                ),
                "baseline_weighted_compute_units": _float_or_blank(row, "baseline_weighted_compute_units"),
                "baseline_train_flops": "" if baseline_train_flops is None else float(baseline_train_flops),
                "baseline_train_pflops": (
                    "" if baseline_train_flops is None else float(baseline_train_flops) / 1e15
                ),
                "row_origin": row.get("row_origin", ""),
            }
        )
    out.sort(key=lambda item: (item["robust_method"], int(item["step"])))
    return out


def _axis_rows(rows: List[Dict[str, object]], x_key: str) -> List[Dict[str, object]]:
    return [row for row in rows if _safe_float(row.get(x_key)) is not None]


def _warmup_boundary(rows: List[Dict[str, object]], x_key: str):
    robust_rows = [row for row in rows if row["robust_method"] != "baseline"]
    if not robust_rows:
        return None
    if x_key == "train_wall_clock_sec":
        return _safe_float(robust_rows[0].get("baseline_train_wall_clock_sec_effective"))
    if x_key == "weighted_compute_units":
        return _safe_float(robust_rows[0].get("baseline_weighted_compute_units"))
    if x_key == "total_train_pflops":
        return _safe_float(robust_rows[0].get("baseline_train_pflops"))
    if x_key == "images_shown_m":
        values = [_safe_float(row.get("images_shown_m")) for row in robust_rows]
        values = [value for value in values if value is not None]
        return min(values) if values else None
    return None


def plot_zoom(
    rows: List[Dict[str, object]],
    *,
    out_png: Path,
    dataset_label: str,
    train_percent_label: str,
    baseline_zoom_start_kimg: float,
    x_axes: Sequence[str],
) -> None:
    baseline_zoom_rows = [
        row
        for row in rows
        if row["robust_method"] == "baseline" and float(row["step"]) >= float(baseline_zoom_start_kimg)
    ]
    robust_rows = [row for row in rows if row["robust_method"] != "baseline"]
    zoom_rows = baseline_zoom_rows + robust_rows
    if not zoom_rows:
        raise RuntimeError("No rows available for zoomed FID plot.")

    y_vals = [float(row["fid"]) for row in zoom_rows]
    y_min = min(y_vals)
    y_max = max(y_vals)
    y_pad = max((y_max - y_min) * 0.08, 0.05)

    plot_specs = []
    for x_key in x_axes:
        axis_rows = _axis_rows(zoom_rows, x_key)
        if not axis_rows:
            print(f"[fid-zoom] skipping x-axis with no numeric rows: {x_key}")
            continue
        spec = AXIS_SPECS[x_key]
        plot_specs.append((x_key, spec["label"], spec["title"], axis_rows))
    if not plot_specs:
        raise RuntimeError("No requested x-axis has numeric rows for the zoomed FID plot.")

    n_plots = len(plot_specs)
    n_cols = min(n_plots, 3)
    n_rows = int(math.ceil(n_plots / n_cols))
    fig, axes_grid = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows), squeeze=False)
    axes = [ax for axis_row in axes_grid for ax in axis_row]

    for ax, (x_key, x_label, title, axis_rows) in zip(axes, plot_specs):
        x_vals = [float(row[x_key]) for row in axis_rows]
        x_min = min(x_vals)
        x_max = max(x_vals)
        x_pad = max((x_max - x_min) * 0.05, 1.0)
        for robust_method in STYLE_ORDER:
            style = STYLE[robust_method]
            method_rows = [row for row in axis_rows if row["robust_method"] == robust_method]
            method_rows.sort(key=lambda row: float(row[x_key]))
            if not method_rows:
                continue
            ax.plot(
                [float(row[x_key]) for row in method_rows],
                [float(row["fid"]) for row in method_rows],
                marker="o",
                linewidth=2.0,
                color=style["color"],
                label=style["label"],
            )
        boundary = _warmup_boundary(axis_rows, x_key)
        if boundary is not None:
            ax.axvline(float(boundary), color="tab:orange", alpha=0.45, label="Robust warmup end")
        ax.set_xlabel(x_label)
        ax.set_ylabel("FID")
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    for ax in axes[n_plots:]:
        ax.set_axis_off()
    handles = []
    labels = []
    for ax in axes[:n_plots]:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=4, frameon=False)
    fig.suptitle(f"{dataset_label} {train_percent_label}: FID Comparison (Zoomed)", y=1.03)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest_csv).resolve()
    out_png = Path(args.out_png).resolve()
    out_csv = Path(args.out_csv).resolve() if str(args.out_csv).strip() else None
    x_axes = _parse_axes(args.x_axes)

    manifest_rows = load_csv_rows(manifest_path)
    rows = build_rows(manifest_rows)
    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        write_csv(out_csv, rows)
        print(f"[fid-zoom] wrote {out_csv}")
    plot_zoom(
        rows,
        out_png=out_png,
        dataset_label=str(args.dataset_label),
        train_percent_label=str(args.train_percent_label),
        baseline_zoom_start_kimg=float(args.baseline_zoom_start_kimg),
        x_axes=x_axes,
    )
    print(f"[fid-zoom] wrote {out_png}")
    if not args.no_comparison_graph_mirror:
        artifacts = [out_png]
        if out_csv is not None:
            artifacts.append(out_csv)
        comparison_dir = Path(args.comparison_graphs_dir).resolve() if str(args.comparison_graphs_dir).strip() else None
        for source, dest in mirror_many(artifacts, comparison_dir=comparison_dir):
            print(f"[fid-zoom] mirrored {source} -> {dest}")


if __name__ == "__main__":
    main()
