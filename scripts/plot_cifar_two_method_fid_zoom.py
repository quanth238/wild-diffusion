#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


STYLE = {
    "baseline": {"label": "Baseline EDM", "color": "tab:blue"},
    "wild_diffusion": {"label": "Wild-Diffusion EDM", "color": "tab:orange"},
    "cdro": {"label": "CDRO EDM", "color": "tab:green"},
}
STYLE_ORDER = ["baseline", "wild_diffusion", "cdro"]


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


def build_rows(manifest_rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in manifest_rows:
        robust_method = str(row.get("robust_method", "")).strip()
        if robust_method not in STYLE:
            continue
        fid_value = _safe_float(row.get("fid"))
        wall_clock = _safe_float(row.get("train_wall_clock_sec"))
        weighted_compute = _safe_float(row.get("weighted_compute_units"))
        if fid_value is None or wall_clock is None or weighted_compute is None:
            continue
        out.append(
            {
                "robust_method": robust_method,
                "series_label": STYLE[robust_method]["label"],
                "step": int(float(row["step"])),
                "fid": float(fid_value),
                "train_wall_clock_sec": float(wall_clock),
                "weighted_compute_units": float(weighted_compute),
                "baseline_train_wall_clock_sec_effective": (
                    ""
                    if not row.get("baseline_train_wall_clock_sec_effective")
                    else float(row["baseline_train_wall_clock_sec_effective"])
                ),
                "baseline_weighted_compute_units": (
                    ""
                    if not row.get("baseline_weighted_compute_units")
                    else float(row["baseline_weighted_compute_units"])
                ),
                "row_origin": row.get("row_origin", ""),
            }
        )
    out.sort(key=lambda item: (item["robust_method"], int(item["step"])))
    return out


def plot_zoom(
    rows: List[Dict[str, object]],
    *,
    out_png: Path,
    dataset_label: str,
    train_percent_label: str,
    baseline_zoom_start_kimg: float,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

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

    plot_specs = [
        ("train_wall_clock_sec", "Train Wall-Clock (sec)", "FID vs Train Wall-Clock (Zoom)"),
        ("weighted_compute_units", "Weighted Compute Units", "FID vs Weighted Compute (Zoom)"),
    ]
    for ax, (x_key, x_label, title) in zip(axes, plot_specs):
        x_vals = [float(row[x_key]) for row in zoom_rows]
        x_min = min(x_vals)
        x_max = max(x_vals)
        x_pad = max((x_max - x_min) * 0.05, 1.0)
        for robust_method in STYLE_ORDER:
            style = STYLE[robust_method]
            method_rows = [row for row in zoom_rows if row["robust_method"] == robust_method]
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
        if robust_rows:
            boundary = (
                robust_rows[0]["baseline_train_wall_clock_sec_effective"]
                if x_key == "train_wall_clock_sec"
                else robust_rows[0]["baseline_weighted_compute_units"]
            )
            if boundary not in ("", None):
                ax.axvline(float(boundary), color="tab:orange", alpha=0.45, label="Robust warmup end")
        ax.set_xlabel(x_label)
        ax.set_ylabel("FID")
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
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
    )
    print(f"[fid-zoom] wrote {out_png}")


if __name__ == "__main__":
    main()
