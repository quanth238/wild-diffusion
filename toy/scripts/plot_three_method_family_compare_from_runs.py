#!/usr/bin/env python3
"""Discover three-method combined CSVs and render a family-aware comparison plot set."""

from __future__ import annotations

import argparse
import json
import os
import re
from glob import glob
from typing import Dict, Iterable, List

import plot_three_method_fid_curves as base_plot


COMBINED_SUFFIX = "_all_methods_raw_seed_rows.csv"
ALLOWED_METHODS = {"baseline", "wild_diffusion", "cdro"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Discover one or more three-method combined CSVs, keep them source-distinct when "
            "needed, and emit the same family-aware FID plots as plot_three_method_fid_curves.py."
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        action="append",
        default=[],
        help="Input directory or combined CSV path. Repeatable.",
    )
    parser.add_argument(
        "--combined-csv",
        type=str,
        action="append",
        default=[],
        help="Direct combined CSV path. Alias for --input.",
    )
    parser.add_argument(
        "--glob",
        type=str,
        action="append",
        default=[],
        help="Recursive glob to use when an input is a directory. Default: '*_all_methods_raw_seed_rows.csv'.",
    )
    parser.add_argument(
        "--include-substring",
        type=str,
        action="append",
        default=[],
        help="Keep only discovered CSV paths containing one of these substrings. Repeatable.",
    )
    parser.add_argument(
        "--exclude-substring",
        type=str,
        action="append",
        default=[],
        help="Drop discovered CSV paths containing any of these substrings. Repeatable.",
    )
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--prefix", type=str, required=True)
    parser.add_argument("--train-percent-label", type=str, default="100%")
    parser.add_argument("--dataset-label", type=str, default="Family Compare")
    parser.add_argument("--loss-metric-key", type=str, default="objective_clean_probe")
    parser.add_argument("--loss-metric-label", type=str, default="")
    parser.add_argument("--loss-metric-path-stem", type=str, default="")
    return parser.parse_args()


def _slug(value: str) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "source"


def _source_tag(path: str) -> str:
    base = os.path.basename(path)
    if base.endswith(COMBINED_SUFFIX):
        return base[: -len(COMBINED_SUFFIX)]
    stem, _sep, _ext = base.partition(".")
    return stem or base


def _discover_csvs(
    *,
    inputs: Iterable[str],
    globs: Iterable[str],
    include_substrings: Iterable[str],
    exclude_substrings: Iterable[str],
) -> List[str]:
    patterns = [str(pattern).strip() for pattern in globs if str(pattern).strip()]
    if not patterns:
        patterns = [f"**/{COMBINED_SUFFIX}", f"**/*{COMBINED_SUFFIX}"]
    include_tokens = [str(token).strip() for token in include_substrings if str(token).strip()]
    exclude_tokens = [str(token).strip() for token in exclude_substrings if str(token).strip()]
    discovered: List[str] = []
    for raw_item in inputs:
        item = os.path.abspath(str(raw_item))
        if os.path.isfile(item):
            discovered.append(item)
            continue
        if os.path.isdir(item):
            for pattern in patterns:
                discovered.extend(glob(os.path.join(item, pattern), recursive=True))
            continue
        raise FileNotFoundError(f"Input path does not exist: {item}")
    filtered: List[str] = []
    seen = set()
    for path in sorted(os.path.abspath(path) for path in discovered if str(path).endswith(".csv")):
        if include_tokens and not any(token in path for token in include_tokens):
            continue
        if exclude_tokens and any(token in path for token in exclude_tokens):
            continue
        if path in seen:
            continue
        seen.add(path)
        filtered.append(path)
    return filtered


def _load_source_rows(path: str) -> List[Dict]:
    source_rows: List[Dict] = []
    source_tag = _source_tag(path)
    for raw_row in base_plot.load_csv_rows(path):
        row = base_plot.normalize_row(raw_row)
        if str(row.get("robust_method", "")).strip() not in ALLOWED_METHODS:
            continue
        row["source_csv"] = os.path.abspath(path)
        row["source_tag"] = source_tag
        row["base_series_key"] = str(row.get("series_key", ""))
        row["base_series_label"] = str(row.get("series_label", ""))
        source_rows.append(row)
    return source_rows


def _series_source_counts(rows: List[Dict]) -> Dict[str, int]:
    counts: Dict[str, set] = {}
    for row in rows:
        counts.setdefault(str(row.get("base_series_key", "")), set()).add(str(row.get("source_tag", "")))
    return {key: len(value) for key, value in counts.items()}


def _tag_duplicate_series(rows: List[Dict]) -> List[Dict]:
    tagged: List[Dict] = []
    source_counts = _series_source_counts(rows)
    for row in rows:
        out = dict(row)
        base_series_key = str(out.get("base_series_key", ""))
        base_series_label = str(out.get("base_series_label", ""))
        source_tag = str(out.get("source_tag", ""))
        if source_counts.get(base_series_key, 0) > 1:
            out["series_key"] = f"{base_series_key}__{_slug(source_tag)}"
            out["series_label"] = f"{base_series_label} [{source_tag}]"
        tagged.append(out)
    return tagged


def _source_summary_rows(paths: List[str], rows: List[Dict]) -> List[Dict]:
    summary_rows: List[Dict] = []
    for path in paths:
        source_path = os.path.abspath(path)
        source_rows = [row for row in rows if str(row.get("source_csv", "")) == source_path]
        if not source_rows:
            continue
        summary_rows.append(
            {
                "source_tag": str(source_rows[0].get("source_tag", "")),
                "source_csv": source_path,
                "rows_total": int(len(source_rows)),
                "rows_with_fid": int(sum(1 for row in source_rows if base_plot._row_has_metric(row, "fid"))),
                "series_keys": "|".join(sorted({str(row.get("base_series_key", "")) for row in source_rows})),
                "families": "|".join(sorted({str(row.get("backbone_family", "")) for row in source_rows})),
                "methods": "|".join(sorted({str(row.get("robust_method", "")) for row in source_rows})),
                "seeds": "|".join(sorted({str(row.get("seed", "")) for row in source_rows})),
            }
        )
    return summary_rows


def main() -> None:
    args = parse_args()
    inputs = [str(value).strip() for value in list(args.input) + list(args.combined_csv) if str(value).strip()]
    if not inputs:
        raise SystemExit("[ERROR] Provide at least one --input or --combined-csv.")
    combined_csvs = _discover_csvs(
        inputs=inputs,
        globs=args.glob,
        include_substrings=args.include_substring,
        exclude_substrings=args.exclude_substring,
    )
    if not combined_csvs:
        raise SystemExit("[ERROR] No combined CSVs found after discovery/filtering.")

    base_plot.ensure_dir(args.outdir)
    raw_rows: List[Dict] = []
    for combined_csv in combined_csvs:
        raw_rows.extend(_load_source_rows(combined_csv))
    rows = base_plot._dedupe_rows(_tag_duplicate_series(raw_rows))
    if not rows:
        raise SystemExit("[ERROR] No plottable baseline/WDRO/CDRO rows found in the discovered CSVs.")

    loss_metric_key = base_plot._resolve_loss_metric(
        rows,
        str(getattr(args, "loss_metric_key", "objective_clean_probe")).strip() or "objective_clean_probe",
    )
    loss_metric_label = str(getattr(args, "loss_metric_label", "")).strip() or base_plot._default_loss_metric_label(
        loss_metric_key
    )
    loss_metric_path_stem = str(getattr(args, "loss_metric_path_stem", "")).strip() or loss_metric_key

    combined_csv = os.path.join(args.outdir, f"{args.prefix}_family_compare.csv")
    summary_json = os.path.join(args.outdir, f"{args.prefix}_family_compare_summary.json")
    source_summary_csv = os.path.join(args.outdir, f"{args.prefix}_family_compare_sources.csv")
    plot_wall_clock = os.path.join(args.outdir, f"{args.prefix}_fid_vs_train_wall_clock.png")
    plot_weighted = os.path.join(args.outdir, f"{args.prefix}_fid_vs_weighted_compute.png")
    plot_legacy = os.path.join(args.outdir, f"{args.prefix}_fid_vs_batch_equiv.png")
    plot_dual = os.path.join(args.outdir, f"{args.prefix}_fid_vs_wall_clock_and_weighted_compute.png")
    plot_loss_wall_clock = os.path.join(args.outdir, f"{args.prefix}_{loss_metric_path_stem}_vs_train_wall_clock.png")
    plot_loss_weighted = os.path.join(args.outdir, f"{args.prefix}_{loss_metric_path_stem}_vs_weighted_compute.png")
    plot_loss_dual = os.path.join(
        args.outdir,
        f"{args.prefix}_{loss_metric_path_stem}_vs_wall_clock_and_weighted_compute.png",
    )

    base_plot.write_csv(combined_csv, rows)
    base_plot.write_csv(source_summary_csv, _source_summary_rows(combined_csvs, rows))
    base_plot.make_plot(
        path=plot_wall_clock,
        rows=rows,
        x_key="train_wall_clock_sec",
        x_label="Train Wall-Clock (sec)",
        y_key="fid",
        y_label="FID",
        plot_label="FID",
        title_suffix="Train Wall-Clock",
        train_percent_label=str(args.train_percent_label),
        dataset_label=str(args.dataset_label),
    )
    base_plot.make_plot(
        path=plot_weighted,
        rows=rows,
        x_key="weighted_compute_units",
        x_label="Weighted Compute Units",
        y_key="fid",
        y_label="FID",
        plot_label="FID",
        title_suffix="Weighted Compute",
        train_percent_label=str(args.train_percent_label),
        dataset_label=str(args.dataset_label),
    )
    base_plot.make_plot(
        path=plot_legacy,
        rows=rows,
        x_key="compute_budget_be",
        x_label="Legacy Batch-Equivalent Denoiser Evals",
        y_key="fid",
        y_label="FID",
        plot_label="FID",
        title_suffix="Legacy Batch-Equivalent Compute",
        train_percent_label=str(args.train_percent_label),
        dataset_label=str(args.dataset_label),
    )
    base_plot.make_dual_plot(
        path=plot_dual,
        rows=rows,
        y_key="fid",
        y_label="FID",
        plot_label="FID",
        train_percent_label=str(args.train_percent_label),
        dataset_label=str(args.dataset_label),
    )
    base_plot.make_plot(
        path=plot_loss_wall_clock,
        rows=rows,
        x_key="train_wall_clock_sec",
        x_label="Train Wall-Clock (sec)",
        y_key=loss_metric_key,
        y_label=loss_metric_label,
        plot_label=loss_metric_label,
        title_suffix="Train Wall-Clock",
        train_percent_label=str(args.train_percent_label),
        dataset_label=str(args.dataset_label),
    )
    base_plot.make_plot(
        path=plot_loss_weighted,
        rows=rows,
        x_key="weighted_compute_units",
        x_label="Weighted Compute Units",
        y_key=loss_metric_key,
        y_label=loss_metric_label,
        plot_label=loss_metric_label,
        title_suffix="Weighted Compute",
        train_percent_label=str(args.train_percent_label),
        dataset_label=str(args.dataset_label),
    )
    base_plot.make_dual_plot(
        path=plot_loss_dual,
        rows=rows,
        y_key=loss_metric_key,
        y_label=loss_metric_label,
        plot_label=loss_metric_label,
        train_percent_label=str(args.train_percent_label),
        dataset_label=str(args.dataset_label),
    )

    summary = {
        "combined_csvs": [os.path.abspath(path) for path in combined_csvs],
        "source_summary_csv": source_summary_csv,
        "combined_csv": combined_csv,
        "plot_paths": {
            "train_wall_clock_sec": plot_wall_clock,
            "weighted_compute_units": plot_weighted,
            "wall_clock_and_weighted_compute": plot_dual,
            "batch_equiv_denoiser_evals": plot_legacy,
            f"{loss_metric_path_stem}_train_wall_clock_sec": plot_loss_wall_clock,
            f"{loss_metric_path_stem}_weighted_compute_units": plot_loss_weighted,
            f"{loss_metric_path_stem}_wall_clock_and_weighted_compute": plot_loss_dual,
        },
        "loss_metric_key": loss_metric_key,
        "loss_metric_label": loss_metric_label,
        "notes": {
            "source_discovery": (
                "Each input may be a combined CSV or a directory. When multiple inputs contain the same "
                "base series key, source tags are appended so curves stay distinct."
            ),
            "style_encoding": (
                "Color encodes the robustifier family (Baseline / Wild-Diffusion / CDRO); "
                "line style and marker encode the backbone family (EDM / RF / Score VE)."
            ),
        },
    }
    with open(summary_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"[family-compare] discovered {len(combined_csvs)} combined CSV(s)", flush=True)
    print(f"[family-compare] wrote {combined_csv}", flush=True)
    print(f"[family-compare] wrote {source_summary_csv}", flush=True)
    print(f"[family-compare] wrote {summary_json}", flush=True)
    print(f"[family-compare] wrote {plot_weighted}", flush=True)
    print(f"[family-compare] wrote {plot_dual}", flush=True)


if __name__ == "__main__":
    main()
