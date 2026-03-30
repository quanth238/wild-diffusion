#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a set of MNIST WDRO retune runs.")
    parser.add_argument("--run-root", type=Path, default=Path("training-runs"))
    parser.add_argument("--prefix", required=True, help="Run tag prefix, e.g. mnist-wdro-retune2")
    parser.add_argument("--candidates", nargs="+", required=True, help="Candidate suffixes, e.g. d e f")
    parser.add_argument("--baseline-eval-json", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--summary-md", type=Path, default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    summary_json = args.summary_json or (args.run_root / f"{args.prefix}-summary.json")
    summary_md = args.summary_md or (args.run_root / f"{args.prefix}-summary.md")

    baseline = load_json(args.baseline_eval_json)
    rows: list[dict] = []
    for candidate in args.candidates:
        eval_root = args.run_root / f"{args.prefix}-{candidate}-100pct-v1" / "eval" / "wdro"
        eval_files = sorted(eval_root.glob("*/evaluation_result.json"))
        if not eval_files:
            continue
        payload = load_json(eval_files[-1])
        rows.append(
            {
                "candidate": candidate,
                "fid": float(payload["fid"]),
                "run_dir": payload["run_dir"],
                "eval_result": str(eval_files[-1]),
                "network_pkl": payload["network_pkl"],
            }
        )

    rows.sort(key=lambda item: item["fid"])
    summary_payload = {
        "baseline_reference": {
            "fid": float(baseline["fid"]),
            "run_dir": baseline["run_dir"],
            "eval_result": str(args.baseline_eval_json),
            "network_pkl": baseline["network_pkl"],
        },
        "results": rows,
    }
    summary_json.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    lines = [
        "# MNIST WDRO Retune Summary",
        "",
        f"Baseline reference FID: `{float(baseline['fid']):.4f}`",
        "",
        "| Rank | Candidate | FID |",
        "| --- | --- | ---: |",
    ]
    for idx, row in enumerate(rows, start=1):
        lines.append(f"| {idx} | `{row['candidate']}` | {row['fid']:.4f} |")
    lines.extend(["", "## Artifacts", ""])
    lines.append(f"- `baseline_reference` eval: `{args.baseline_eval_json}`")
    for row in rows:
        lines.append(f"- `candidate {row['candidate']}` run: `{row['run_dir']}`")
        lines.append(f"- `candidate {row['candidate']}` eval: `{row['eval_result']}`")
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary_md)


if __name__ == "__main__":
    main()
