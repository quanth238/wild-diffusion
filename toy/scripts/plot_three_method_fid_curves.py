#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt


ROBUST_STYLE = {
    "baseline": {"label": "Baseline", "color": "tab:blue", "order": 0},
    "wild_diffusion": {"label": "Wild-Diffusion", "color": "tab:orange", "order": 1},
    "cdro": {"label": "CDRO", "color": "tab:green", "order": 2},
}

FAMILY_STYLE = {
    "edm": {"label": "EDM", "marker": "o", "linestyle": "-", "order": 0},
    "rf": {"label": "RF", "marker": "s", "linestyle": "--", "order": 1},
    "score": {"label": "Score VE", "marker": "^", "linestyle": ":", "order": 2},
}

BOUNDARY_STYLE = {
    "shared_edm_warm_start": {"linestyle": ":", "alpha": 0.4, "linewidth": 1.2},
    "baseline_rf_reflow_start": {"linestyle": "-.", "alpha": 0.5, "linewidth": 1.3},
    "robust_rf_reflow_start": {"linestyle": "-.", "alpha": 0.5, "linewidth": 1.3},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge baseline, Wild-Diffusion, and CDRO curve artifacts into family-aware "
            "FID plots where color tracks the robustifier and line style tracks the backbone family."
        )
    )
    parser.add_argument("--combined-csv", type=str, action="append", default=[])
    parser.add_argument("--wdro-compare-csv", type=str, default="")
    parser.add_argument("--cdro-compare-csv", type=str, default="")
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--prefix", type=str, required=True)
    parser.add_argument("--train-percent-label", type=str, default="1%")
    parser.add_argument("--dataset-label", type=str, default="Simpsons-MNIST RGB")
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_csv_rows(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def _safe_float(value):
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return float(parsed)


def _row_has_metric(row: Dict, key: str) -> bool:
    return _safe_float(row.get(key)) is not None


def _safe_bool(value) -> bool:
    if isinstance(value, bool):
        return bool(value)
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def _canonical_robust_method(method_name: str) -> str:
    method = str(method_name or "").strip().lower()
    if method in {"baseline", "baseline_edm", "baseline_rf", "baseline_score"}:
        return "baseline"
    if method in {"wdro", "wild", "wild_diffusion"}:
        return "wild_diffusion"
    if method == "cdro":
        return "cdro"
    return method


def _backbone_family(method_name: str, training_objective: str) -> str:
    objective = str(training_objective or "").strip().lower()
    if objective == "rf" or str(method_name).strip().lower().endswith("_rf"):
        return "rf"
    if objective == "score" or str(method_name).strip().lower().endswith("_score"):
        return "score"
    return "edm"


def _normalize_series_fields(row: Dict) -> Dict:
    out = dict(row)
    robust_method = _canonical_robust_method(row.get("robust_method", row.get("method", "")))
    backbone_family = _backbone_family(row.get("method", ""), row.get("training_objective", ""))
    robust_label = str(row.get("robust_label", "")).strip() or ROBUST_STYLE.get(
        robust_method,
        {"label": robust_method.replace("_", " ").title()},
    )["label"]
    if robust_method == "wild_diffusion":
        robust_label = "Wild-Diffusion"
    backbone_label = str(row.get("backbone_label", "")).strip() or FAMILY_STYLE.get(
        backbone_family,
        {"label": backbone_family.replace("_", " ").title()},
    )["label"]
    if backbone_family == "rf":
        backbone_label = "RF"
    out["method"] = robust_method
    out["robust_method"] = robust_method
    out["robust_label"] = robust_label
    out["backbone_family"] = backbone_family
    out["backbone_label"] = backbone_label
    out["series_key"] = str(row.get("series_key", "")).strip() or f"{robust_method}_{backbone_family}"
    out["series_label"] = f"{robust_label} {backbone_label}"
    out["method_version_used"] = str(row.get("method_version_used", row.get("method", ""))).strip()
    return out


def _prefer_row_with_metric(current: Optional[Dict], candidate: Dict, *, key: str) -> Dict:
    if current is None:
        return candidate
    current_has_metric = _row_has_metric(current, key)
    candidate_has_metric = _row_has_metric(candidate, key)
    if candidate_has_metric and not current_has_metric:
        return candidate
    if candidate_has_metric == current_has_metric:
        current_selected = _safe_bool(current.get("fid_eval_selected"))
        candidate_selected = _safe_bool(candidate.get("fid_eval_selected"))
        if candidate_selected and not current_selected:
            return candidate
    return current


def normalize_row(row: Dict) -> Dict:
    normalized = _normalize_series_fields(row)
    normalized["step"] = int(float(row["step"]))
    for key in (
        "compute_budget_be",
        "baseline_compute_be",
        "weighted_compute_units",
        "baseline_weighted_compute_units",
        "train_wall_clock_sec",
        "baseline_train_wall_clock_sec_effective",
        "train_gpu_hours",
        "predicted_denoiser_train_wall_clock_sec",
        "baseline_predicted_denoiser_wall_clock_sec",
        "robust_predicted_denoiser_wall_clock_sec",
        "non_denoiser_train_overhead_wall_clock_sec",
        "baseline_non_denoiser_overhead_wall_clock_sec",
        "robust_non_denoiser_overhead_wall_clock_sec",
        "observed_over_predicted_denoiser_wall_clock_ratio",
        "fid",
        "loss_final",
        "loss_mean_last",
        "fixed_warmup_steps",
        "shared_edm_warm_start_compute_be",
        "shared_edm_warm_start_weighted_compute_units",
        "shared_edm_warm_start_train_wall_clock_sec",
        "baseline_rf_reflow_start_compute_be",
        "baseline_rf_reflow_start_weighted_compute_units",
        "baseline_rf_reflow_start_train_wall_clock_sec",
        "robust_rf_reflow_start_compute_be",
        "robust_rf_reflow_start_weighted_compute_units",
        "robust_rf_reflow_start_train_wall_clock_sec",
    ):
        normalized[key] = _safe_float(row.get(key))
    normalized["fid_eval_selected"] = _safe_bool(row.get("fid_eval_selected"))
    normalized["fid_evaluated"] = _safe_bool(row.get("fid_evaluated"))
    normalized["shared_edm_warm_start_available"] = _safe_bool(row.get("shared_edm_warm_start_available"))
    normalized["baseline_rf_reflow_start_available"] = _safe_bool(row.get("baseline_rf_reflow_start_available"))
    normalized["robust_rf_reflow_start_available"] = _safe_bool(row.get("robust_rf_reflow_start_available"))
    normalized["weighted_compute_calibration_required_match"] = _safe_bool(
        row.get("weighted_compute_calibration_required_match")
    )
    normalized["weighted_compute_calibration_exact_hardware_match"] = _safe_bool(
        row.get("weighted_compute_calibration_exact_hardware_match")
    )
    normalized["train_accelerator_count"] = _safe_float(row.get("train_accelerator_count"))
    return normalized


def _warn_if_mixed_plot_provenance(*, rows: List[Dict], x_key: str, plot_path: str) -> None:
    if x_key == "train_wall_clock_sec":
        hardware_signatures = sorted(
            {
                (
                    str(row.get("train_accelerator_kind", "")).strip(),
                    str(row.get("train_accelerator_name", "")).strip(),
                    str(row.get("train_accelerator_count", "")).strip(),
                )
                for row in rows
                if str(row.get("train_accelerator_name", "")).strip()
            }
        )
        if len(hardware_signatures) > 1:
            joined = "; ".join("/".join(signature) for signature in hardware_signatures)
            print(
                f"[plot][WARN] mixed accelerator metadata in wall-clock plot {plot_path}: {joined}",
                flush=True,
            )
    if x_key == "weighted_compute_units":
        calibration_signatures = sorted(
            {
                (
                    str(row.get("weighted_compute_calibration_path", "")).strip(),
                    str(row.get("weighted_compute_calibration_source", "")).strip(),
                )
                for row in rows
                if str(row.get("weighted_compute_calibration_path", "")).strip()
                or str(row.get("weighted_compute_calibration_source", "")).strip()
            }
        )
        if len(calibration_signatures) > 1:
            joined = "; ".join(f"{path or '<none>'} [{source or 'unknown'}]" for path, source in calibration_signatures)
            print(
                f"[plot][WARN] mixed weighted-compute calibrations in plot {plot_path}: {joined}",
                flush=True,
            )


def _series_sort_key(series_key: str, rows: List[Dict]) -> tuple:
    sample = next((row for row in rows if row.get("series_key") == series_key), None)
    if sample is None:
        return (99, 99, series_key)
    robust_method = str(sample.get("robust_method", ""))
    backbone_family = str(sample.get("backbone_family", ""))
    return (
        ROBUST_STYLE.get(robust_method, {"order": 99})["order"],
        FAMILY_STYLE.get(backbone_family, {"order": 99})["order"],
        series_key,
    )


def _dedupe_rows(rows: List[Dict]) -> List[Dict]:
    deduped: Dict[tuple, Dict] = {}
    for row in rows:
        key = (
            str(row.get("series_key", "")),
            int(row.get("seed", 0)),
            int(row.get("step", 0)),
            str(row.get("row_origin", "")),
        )
        deduped[key] = _prefer_row_with_metric(deduped.get(key), row, key="fid")
    out = list(deduped.values())
    out.sort(
        key=lambda row: (
            ROBUST_STYLE.get(str(row.get("robust_method", "")), {"order": 99})["order"],
            FAMILY_STYLE.get(str(row.get("backbone_family", "")), {"order": 99})["order"],
            str(row.get("series_key", "")),
            int(row.get("seed", 0)),
            int(row.get("step", 0)),
        )
    )
    return out


def load_three_method_rows(*, wdro_compare_csv: str, cdro_compare_csv: str) -> List[Dict]:
    wdro_rows = [normalize_row(row) for row in load_csv_rows(wdro_compare_csv)]
    cdro_rows = [normalize_row(row) for row in load_csv_rows(cdro_compare_csv)]

    baseline_by_key: Dict[tuple, Dict] = {}
    for row in wdro_rows + cdro_rows:
        if row.get("robust_method") != "baseline":
            continue
        key = (str(row.get("backbone_family", "")), int(row["step"]))
        baseline_by_key[key] = _prefer_row_with_metric(
            baseline_by_key.get(key),
            row,
            key="fid",
        )

    merged_rows = list(baseline_by_key.values())
    merged_rows.extend(row for row in wdro_rows if row.get("robust_method") == "wild_diffusion")
    merged_rows.extend(row for row in cdro_rows if row.get("robust_method") == "cdro")
    return _dedupe_rows(merged_rows)


def load_combined_rows(*, combined_csvs: List[str]) -> List[Dict]:
    rows: List[Dict] = []
    for combined_csv in combined_csvs:
        rows.extend(normalize_row(row) for row in load_csv_rows(combined_csv))
    rows = [
        row
        for row in rows
        if row.get("robust_method") in ("baseline", "wild_diffusion", "cdro")
    ]
    return _dedupe_rows(rows)


def _phase_boundary_x(*, rows: List[Dict], series_key: str, x_key: str) -> Optional[float]:
    robust_rows = [
        row
        for row in rows
        if row.get("series_key") == series_key and row.get("row_origin") == "trajectory_robust_phase"
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
        if row.get("series_key") == series_key
        and row.get("row_origin") == "trajectory_warmup_phase"
        and row.get(x_key) is not None
    ]
    if not warmup_rows:
        return None
    return max(float(row[x_key]) for row in warmup_rows)


def _draw_phase_boundaries(*, ax, rows: List[Dict], x_key: str, series_keys: List[str]) -> None:
    if x_key not in ("train_wall_clock_sec", "weighted_compute_units", "compute_budget_be"):
        return
    for series_key in series_keys:
        sample = next(row for row in rows if row.get("series_key") == series_key)
        boundary_x = _phase_boundary_x(rows=rows, series_key=series_key, x_key=x_key)
        if boundary_x is None:
            continue
        ax.axvline(
            x=boundary_x,
            color=ROBUST_STYLE[str(sample.get("robust_method"))]["color"],
            linestyle=FAMILY_STYLE[str(sample.get("backbone_family"))]["linestyle"],
            linewidth=1.2,
            alpha=0.45,
            label=f"{sample['series_label']} warmup end",
        )


def _boundary_series_field(x_key: str, boundary_name: str) -> Optional[str]:
    if x_key == "train_wall_clock_sec":
        suffix = "train_wall_clock_sec"
    elif x_key == "weighted_compute_units":
        suffix = "weighted_compute_units"
    elif x_key == "compute_budget_be":
        suffix = "compute_be"
    else:
        return None
    return f"{boundary_name}_{suffix}"


def _stable_series_boundary_x(*, rows: List[Dict], series_key: str, field_name: str, availability_field: str) -> Optional[float]:
    values = [
        float(row[field_name])
        for row in rows
        if row.get("series_key") == series_key
        and _safe_bool(row.get(availability_field))
        and row.get(field_name) is not None
    ]
    if not values:
        return None
    min_value = min(values)
    max_value = max(values)
    tolerance = max(1e-9, 1e-6 * max(1.0, abs(min_value), abs(max_value)))
    if abs(max_value - min_value) > tolerance:
        return None
    return float(sum(values) / float(len(values)))


def _draw_rf_boundaries(*, ax, rows: List[Dict], x_key: str, series_keys: List[str]) -> None:
    boundary_specs = (
        ("shared_edm_warm_start", "shared_edm_warm_start_available", "warm-start"),
        ("baseline_rf_reflow_start", "baseline_rf_reflow_start_available", "reflow start"),
        ("robust_rf_reflow_start", "robust_rf_reflow_start_available", "reflow start"),
    )
    for series_key in series_keys:
        sample = next(row for row in rows if row.get("series_key") == series_key)
        for boundary_name, availability_field, label_suffix in boundary_specs:
            if boundary_name == "shared_edm_warm_start" and sample.get("robust_method") != "baseline":
                continue
            field_name = _boundary_series_field(x_key, boundary_name)
            if field_name is None:
                continue
            boundary_x = _stable_series_boundary_x(
                rows=rows,
                series_key=series_key,
                field_name=field_name,
                availability_field=availability_field,
            )
            if boundary_x is None:
                continue
            style = BOUNDARY_STYLE[boundary_name]
            ax.axvline(
                x=boundary_x,
                color=ROBUST_STYLE[str(sample.get("robust_method"))]["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                alpha=style["alpha"],
                label=f"{sample['series_label']} {label_suffix}",
            )


def _series_segments(*, rows: List[Dict], series_key: str, x_key: str, y_key: str) -> List[List[tuple]]:
    points = [
        (float(row[x_key]), float(row[y_key]), str(row.get("loss_kind", "")))
        for row in rows
        if row.get("series_key") == series_key and row.get(x_key) is not None and row.get(y_key) is not None
    ]
    if not points:
        return []
    points.sort(key=lambda point: float(point[0]))
    sample = next(row for row in rows if row.get("series_key") == series_key)
    if not y_key.startswith("loss_") or sample.get("robust_method") == "baseline":
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


def _series_keys(rows: List[Dict]) -> List[str]:
    keys = sorted({str(row.get("series_key", "")) for row in rows})
    return sorted(keys, key=lambda key: _series_sort_key(key, rows))


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
    dataset_label: str,
) -> None:
    _warn_if_mixed_plot_provenance(rows=rows, x_key=x_key, plot_path=path)
    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    series_keys = _series_keys(rows)
    for series_key in series_keys:
        sample = next(row for row in rows if row.get("series_key") == series_key)
        segments = _series_segments(rows=rows, series_key=series_key, x_key=x_key, y_key=y_key)
        if not segments:
            continue
        for segment_index, segment in enumerate(segments):
            ax.plot(
                [value for value, _ in segment],
                [value for _, value in segment],
                marker=FAMILY_STYLE[str(sample.get("backbone_family"))]["marker"],
                linestyle=FAMILY_STYLE[str(sample.get("backbone_family"))]["linestyle"],
                linewidth=2.0,
                color=ROBUST_STYLE[str(sample.get("robust_method"))]["color"],
                label=str(sample.get("series_label")) if segment_index == 0 else "_nolegend_",
            )
    _draw_phase_boundaries(ax=ax, rows=rows, x_key=x_key, series_keys=series_keys)
    _draw_rf_boundaries(ax=ax, rows=rows, x_key=x_key, series_keys=series_keys)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"{dataset_label} {train_percent_label}: {plot_label} vs {title_suffix}")
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
    dataset_label: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_specs = (
        ("train_wall_clock_sec", "Train Wall-Clock (sec)", "Train Wall-Clock"),
        ("weighted_compute_units", "Weighted Compute Units", "Weighted Compute"),
    )
    series_keys = _series_keys(rows)
    for ax, (x_key, x_label, title_suffix) in zip(axes, plot_specs):
        for series_key in series_keys:
            sample = next(row for row in rows if row.get("series_key") == series_key)
            segments = _series_segments(rows=rows, series_key=series_key, x_key=x_key, y_key=y_key)
            if not segments:
                continue
            for segment_index, segment in enumerate(segments):
                ax.plot(
                    [value for value, _ in segment],
                    [value for _, value in segment],
                    marker=FAMILY_STYLE[str(sample.get("backbone_family"))]["marker"],
                    linestyle=FAMILY_STYLE[str(sample.get("backbone_family"))]["linestyle"],
                    linewidth=2.0,
                    color=ROBUST_STYLE[str(sample.get("robust_method"))]["color"],
                    label=str(sample.get("series_label")) if segment_index == 0 else "_nolegend_",
                )
        _draw_phase_boundaries(ax=ax, rows=rows, x_key=x_key, series_keys=series_keys)
        _draw_rf_boundaries(ax=ax, rows=rows, x_key=x_key, series_keys=series_keys)
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
            ncol=max(1, min(len(handles), 3)),
            frameon=False,
        )
    fig.suptitle(f"{dataset_label} {train_percent_label}: {plot_label} Comparison", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    combined_csv_args = [str(path).strip() for path in getattr(args, "combined_csv", []) if str(path).strip()]
    wdro_compare_csv_arg = str(getattr(args, "wdro_compare_csv", "")).strip()
    cdro_compare_csv_arg = str(getattr(args, "cdro_compare_csv", "")).strip()
    if combined_csv_args:
        if wdro_compare_csv_arg or cdro_compare_csv_arg:
            raise SystemExit(
                "[ERROR] Use either --combined-csv (repeatable) or the pair --wdro-compare-csv/--cdro-compare-csv, not both."
            )
    elif not (wdro_compare_csv_arg and cdro_compare_csv_arg):
        raise SystemExit(
            "[ERROR] Provide either one or more --combined-csv values or both --wdro-compare-csv and --cdro-compare-csv."
        )
    ensure_dir(args.outdir)
    rows = (
        load_combined_rows(combined_csvs=combined_csv_args)
        if combined_csv_args
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
        dataset_label=str(args.dataset_label),
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
        dataset_label=str(args.dataset_label),
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
        dataset_label=str(args.dataset_label),
    )
    make_dual_plot(
        path=plot_dual,
        rows=rows,
        y_key="fid",
        y_label="FID",
        plot_label="FID",
        train_percent_label=str(args.train_percent_label),
        dataset_label=str(args.dataset_label),
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
        dataset_label=str(args.dataset_label),
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
        dataset_label=str(args.dataset_label),
    )
    make_dual_plot(
        path=plot_loss_dual,
        rows=rows,
        y_key="loss_mean_last",
        y_label="Loss (mean last)",
        plot_label="Loss (mean last)",
        train_percent_label=str(args.train_percent_label),
        dataset_label=str(args.dataset_label),
    )
    summary = {
        "rows": rows,
        "fid_coverage_by_series": {
            series_key: {
                "series_label": next(row["series_label"] for row in rows if row.get("series_key") == series_key),
                "rows_total": sum(1 for row in rows if row.get("series_key") == series_key),
                "rows_with_fid": sum(
                    1 for row in rows if row.get("series_key") == series_key and _row_has_metric(row, "fid")
                ),
            }
            for series_key in _series_keys(rows)
        },
        "sources": {
            "combined_csvs": [os.path.abspath(path) for path in combined_csv_args],
            "wdro_compare_csv": None if not wdro_compare_csv_arg else os.path.abspath(wdro_compare_csv_arg),
            "cdro_compare_csv": None if not cdro_compare_csv_arg else os.path.abspath(cdro_compare_csv_arg),
        },
        "notes": {
            "style_encoding": (
                "Color encodes the robustifier family (Baseline / Wild-Diffusion / CDRO); "
                "line style and marker encode the backbone family (EDM / RF / Score VE)."
            ),
            "loss_metric_warning": (
                "Baseline uses the backbone-native baseline loss, while robust methods use robust outer losses. "
                "Loss plots are optimization diagnostics, not a primary cross-method fairness metric."
            ),
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
