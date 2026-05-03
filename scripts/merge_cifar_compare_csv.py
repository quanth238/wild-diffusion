#!/usr/bin/env python3
import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List

from comparison_outputs import mirror_many


DEFAULT_BASE_COMPARE_CSV = (
    "/home/bachlc/GM-CDRO/training-runs/fid-sweeps/cifar10_baseline_vs_wdro_coarse_20260414/"
    "cifar10_baseline_vs_wdro_coarse_20260414_three_method_compare.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge the baseline+WDRO three-method CIFAR comparison CSV with CDRO rows "
            "from a manifest, preserving method ordering for downstream plotting."
        )
    )
    parser.add_argument("--base-compare-csv", type=str, default=DEFAULT_BASE_COMPARE_CSV)
    parser.add_argument("--cdro-manifest-csv", type=str, required=True)
    parser.add_argument("--out-csv", type=str, required=True)
    parser.add_argument("--out-summary-json", type=str, default="")
    parser.add_argument("--comparison-graphs-dir", type=str, default="")
    parser.add_argument("--no-comparison-graph-mirror", action="store_true")
    parser.add_argument(
        "--cdro-row-mode",
        type=str,
        choices=["all", "fid_evaluated"],
        default="all",
    )
    return parser.parse_args()


def _truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _safe_float(value: object):
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def load_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: List[Dict[str, object]]) -> None:
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def method_sort_key(method: str) -> tuple[int, str]:
    order = {"baseline": 0, "wild_diffusion": 1, "cdro": 2}
    return (order.get(str(method), 99), str(method))


def sort_rows(rows: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    return sorted(
        rows,
        key=lambda row: (
            method_sort_key(str(row.get("robust_method", row.get("method", "")))),
            int(float(row.get("step", 0) or 0)),
        ),
    )


def select_cdro_rows(rows: List[Dict[str, str]], *, mode: str) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for row in rows:
        robust_method = str(row.get("robust_method", row.get("method", ""))).strip()
        if robust_method != "cdro":
            continue
        if mode == "fid_evaluated":
            if not _truthy(row.get("fid_evaluated")):
                continue
            if _safe_float(row.get("fid")) is None:
                continue
        out.append(row)
    return out


def summarize_rows(rows: List[Dict[str, object]]) -> Dict[str, object]:
    methods = sorted(
        {
            str(row.get("robust_method", row.get("method", ""))).strip()
            for row in rows
            if str(row.get("robust_method", row.get("method", ""))).strip()
        },
        key=method_sort_key,
    )
    per_method: Dict[str, Dict[str, object]] = {}
    for method in methods:
        method_rows = [
            row for row in rows
            if str(row.get("robust_method", row.get("method", ""))).strip() == method
        ]
        record: Dict[str, object] = {"num_rows": len(method_rows)}
        fid_rows = [row for row in method_rows if _safe_float(row.get("fid")) is not None]
        if fid_rows:
            best_fid_row = min(fid_rows, key=lambda row: float(row["fid"]))
            final_fid_row = max(fid_rows, key=lambda row: float(row["step"]))
            record.update(
                {
                    "best_fid": float(best_fid_row["fid"]),
                    "best_fid_step": int(float(best_fid_row["step"])),
                    "final_fid": float(final_fid_row["fid"]),
                    "final_fid_step": int(float(final_fid_row["step"])),
                }
            )
        loss_rows = [
            row
            for row in method_rows
            if _safe_float(row.get("loss_comparable_final")) is not None
            or _safe_float(row.get("loss_probe_clean")) is not None
        ]
        if loss_rows:
            final_loss_row = max(loss_rows, key=lambda row: float(row["step"]))
            final_loss = _safe_float(final_loss_row.get("loss_comparable_final"))
            if final_loss is None:
                final_loss = _safe_float(final_loss_row.get("loss_probe_clean"))
            record.update(
                {
                    "final_loss": final_loss,
                    "final_loss_step": int(float(final_loss_row["step"])),
                }
            )
        per_method[method] = record
    return {"num_rows": len(rows), "methods": per_method}


def main() -> None:
    args = parse_args()
    base_compare_csv = Path(args.base_compare_csv).resolve()
    cdro_manifest_csv = Path(args.cdro_manifest_csv).resolve()
    out_csv = Path(args.out_csv).resolve()
    out_summary_json = (
        Path(args.out_summary_json).resolve() if str(args.out_summary_json).strip() else None
    )

    base_rows = load_csv_rows(base_compare_csv)
    cdro_rows = select_cdro_rows(load_csv_rows(cdro_manifest_csv), mode=str(args.cdro_row_mode))
    merged_rows = sort_rows([*base_rows, *cdro_rows])
    write_csv_rows(out_csv, merged_rows)
    print(f"[merge-compare] wrote {out_csv}")

    if out_summary_json is not None:
        payload = {
            "created_at_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "base_compare_csv": str(base_compare_csv),
            "cdro_manifest_csv": str(cdro_manifest_csv),
            "cdro_row_mode": str(args.cdro_row_mode),
            "merged_compare_csv": str(out_csv),
            "summary": summarize_rows(merged_rows),
        }
        out_summary_json.parent.mkdir(parents=True, exist_ok=True)
        out_summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[merge-compare] wrote {out_summary_json}")

    if not args.no_comparison_graph_mirror:
        artifacts = [out_csv]
        if out_summary_json is not None:
            artifacts.append(out_summary_json)
        comparison_dir = Path(args.comparison_graphs_dir).resolve() if str(args.comparison_graphs_dir).strip() else None
        for source, dest in mirror_many(artifacts, comparison_dir=comparison_dir):
            print(f"[merge-compare] mirrored {source} -> {dest}")


if __name__ == "__main__":
    main()
