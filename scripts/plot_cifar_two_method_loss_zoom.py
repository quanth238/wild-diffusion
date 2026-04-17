#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


STYLE = {
    "baseline": {"label": "Baseline EDM", "color": "tab:blue"},
    "wild_diffusion": {"label": "Wild-Diffusion EDM", "color": "tab:orange"},
    "cdro": {"label": "CDRO EDM (outer)", "color": "tab:green"},
}
STYLE_ORDER = ["baseline", "wild_diffusion", "cdro"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot a zoomed loss comparison for CIFAR baseline and robust "
            "variants using a merged comparison CSV and training stats traces."
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


def _extract_loss_mean(payload: Dict) -> float:
    for key in ("Loss", "Loss/loss"):
        value = payload.get(key, {})
        if isinstance(value, dict) and value.get("mean") is not None:
            return float(value["mean"])
    raise KeyError("Missing loss metric: expected 'Loss' or legacy 'Loss/loss'.")


def load_loss_trace(stats_path: Path) -> List[Tuple[float, float]]:
    trace: List[Tuple[float, float]] = []
    with stats_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            trace.append(
                (
                    float(payload["Progress/kimg"]["mean"]),
                    _extract_loss_mean(payload),
                )
            )
    if not trace:
        raise RuntimeError(f"No stats rows found in {stats_path}")
    trace.sort(key=lambda item: item[0])
    return trace


def nearest_loss(trace: List[Tuple[float, float]], target_kimg: float) -> Tuple[float, float]:
    return min(trace, key=lambda item: abs(item[0] - float(target_kimg)))


def build_rows(manifest_rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    trace_cache: Dict[Path, List[Tuple[float, float]]] = {}
    out: List[Dict[str, object]] = []
    for row in manifest_rows:
        if row.get("robust_method") not in STYLE:
            continue
        run_dir = Path(str(row["run_dir"])).resolve()
        stats_path = run_dir / "stats.jsonl"
        if stats_path not in trace_cache:
            trace_cache[stats_path] = load_loss_trace(stats_path)
        matched_kimg, loss_value = nearest_loss(trace_cache[stats_path], float(row["step"]))
        out.append(
            {
                "robust_method": row["robust_method"],
                "series_label": STYLE[row["robust_method"]]["label"],
                "step": int(float(row["step"])),
                "matched_stats_kimg": float(matched_kimg),
                "loss_mean": float(loss_value),
                "train_wall_clock_sec": float(row["train_wall_clock_sec"]),
                "weighted_compute_units": float(row["weighted_compute_units"]),
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


def plot_zoom(rows: List[Dict[str, object]], *, out_png: Path, dataset_label: str, train_percent_label: str, baseline_zoom_start_kimg: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    baseline_zoom_rows = [
        row
        for row in rows
        if row["robust_method"] == "baseline" and float(row["step"]) >= float(baseline_zoom_start_kimg)
    ]
    robust_rows = [row for row in rows if row["robust_method"] != "baseline"]
    zoom_rows = baseline_zoom_rows + robust_rows
    if not zoom_rows:
        raise RuntimeError("No rows available for zoomed loss plot.")
    y_vals = [float(row["loss_mean"]) for row in zoom_rows]
    y_min = min(y_vals)
    y_max = max(y_vals)
    pad = max((y_max - y_min) * 0.08, 0.001)

    plot_specs = [
        ("train_wall_clock_sec", "Train Wall-Clock (sec)", "Loss vs Train Wall-Clock (Zoom)"),
        ("weighted_compute_units", "Weighted Compute Units", "Loss vs Weighted Compute (Zoom)"),
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
                [float(row["loss_mean"]) for row in method_rows],
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
        ax.set_ylabel("Loss")
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=4, frameon=False)
    fig.suptitle(f"{dataset_label} {train_percent_label}: Loss Comparison (Zoomed)", y=1.03)
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
        print(f"[loss-zoom] wrote {out_csv}")
    plot_zoom(
        rows,
        out_png=out_png,
        dataset_label=str(args.dataset_label),
        train_percent_label=str(args.train_percent_label),
        baseline_zoom_start_kimg=float(args.baseline_zoom_start_kimg),
    )
    print(f"[loss-zoom] wrote {out_png}")


if __name__ == "__main__":
    main()
