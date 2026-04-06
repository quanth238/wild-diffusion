#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
from typing import Dict, List


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


from toy.process_title import apply_process_title, build_process_title  # noqa: E402
from toy.scripts.queue_three_method_ablation_matrix import _load_rows, _make_family_plot, _summarize_case  # noqa: E402


_APPLIED_PROCESS_TITLE = apply_process_title()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a warmup family comparison plot and summary from existing reevaluated case CSVs."
    )
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--prefix", type=str, required=True)
    parser.add_argument(
        "--case",
        action="append",
        required=True,
        help="Case spec in the form case_id::label::warmup_fraction::combined_csv",
    )
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


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


def _parse_case_spec(text: str) -> Dict:
    parts = text.split("::", 3)
    if len(parts) != 4:
        raise ValueError(f"Invalid --case spec: {text}")
    case_id, label, warmup_fraction, combined_csv = parts
    return {
        "case_id": str(case_id),
        "label": str(label),
        "warmup_fraction": float(warmup_fraction),
        "combined_csv": str(combined_csv),
    }


def main() -> None:
    args = parse_args()
    if _APPLIED_PROCESS_TITLE is None:
        apply_process_title(build_process_title("wdiff", "warmup_plot", args.prefix))
    ensure_dir(args.outdir)
    summary_dir = os.path.join(args.outdir, "summary")
    ensure_dir(summary_dir)

    case_specs = [_parse_case_spec(text) for text in args.case]
    case_ids = [spec["case_id"] for spec in case_specs]
    case_cfgs: Dict[str, Dict] = {}
    case_rows: Dict[str, List[Dict[str, str]]] = {}
    summary_rows: List[Dict] = []

    for spec in case_specs:
        case_id = spec["case_id"]
        case_cfg = {
            "label": spec["label"],
            "wdro_warmup_fraction": float(spec["warmup_fraction"]),
            "cdro_warmup_fraction": float(spec["warmup_fraction"]),
            "outer_attack_weight": 0.5,
            "outer_clean_weight": 1.0,
            "cdro_total_budget_rho": 0.02,
            "cdro_n_steps_path": 24,
        }
        case_cfgs[case_id] = case_cfg
        case_rows[case_id] = _load_rows(spec["combined_csv"])
        summary_rows.extend(_summarize_case(case_id=case_id, case_cfg=case_cfg, combined_csv=spec["combined_csv"]))

    plot_path = _make_family_plot(
        group_name="warmup",
        available_case_ids=case_ids,
        case_rows=case_rows,
        case_cfgs=case_cfgs,
        outdir=summary_dir,
    )
    summary_csv = os.path.join(summary_dir, f"{args.prefix}_case_method_summary.csv")
    write_csv(summary_csv, summary_rows)
    manifest = {
        "case_ids": case_ids,
        "plot_path": plot_path,
        "summary_csv": summary_csv,
        "cases": case_specs,
    }
    with open(os.path.join(summary_dir, f"{args.prefix}_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"[warmup-family] wrote {plot_path}", flush=True)
    print(f"[warmup-family] wrote {summary_csv}", flush=True)


if __name__ == "__main__":
    main()
