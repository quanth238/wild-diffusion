#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path("/root/wild-diffusion")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export MNIST comparison tables and charts.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "training-runs" / "mnist-final-comparison-v1",
        help="Where to write the comparison pack.",
    )
    return parser.parse_args()


def load_eval(path: Path) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return float(payload.get("fid50k_full", payload.get("fid")))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "model": "baseline_edm_3.0mimg",
            "family": "baseline_edm",
            "stage_kimg": 3000,
            "fairness": "reference",
            "eval_json": ROOT / "training-runs" / "mnist-compare-100pct-v3" / "eval" / "baseline-3mimg" / "eval-20260330-024016" / "evaluation_result.json",
        },
        {
            "model": "wdro_edm_3.0mimg",
            "family": "wdro_edm",
            "stage_kimg": 3000,
            "fairness": "reference",
            "eval_json": ROOT / "training-runs" / "mnist-wdro-retune3-g-100pct-v1" / "eval-3mimg" / "eval-20260330-015225" / "evaluation_result.json",
        },
        {
            "model": "cdro_ft_0.1mimg",
            "family": "cdro_ft",
            "stage_kimg": 100,
            "fairness": "fine_tune_from_baseline3m",
            "eval_json": ROOT / "training-runs" / "mnist-cdro-finetune-a-100pct-v1x" / "eval" / "cdro" / "eval-20260330-071737" / "evaluation_result.json",
        },
        {
            "model": "cdro_ft_0.3mimg",
            "family": "cdro_ft",
            "stage_kimg": 300,
            "fairness": "fine_tune_from_baseline3m",
            "eval_json": ROOT / "training-runs" / "mnist-cdro-finetune-a-100pct-v1x" / "eval-300kimg-10k" / "eval-20260330-115617" / "evaluation_result.json",
        },
        {
            "model": "cdro_ft_0.4mimg",
            "family": "cdro_ft",
            "stage_kimg": 400,
            "fairness": "fine_tune_from_baseline3m",
            "eval_json": ROOT / "training-runs" / "mnist-cdro-finetune-a-100pct-v1x" / "eval-400kimg-10k" / "eval-20260330-121753" / "evaluation_result.json",
        },
        {
            "model": "cdro_ft_0.5mimg",
            "family": "cdro_ft",
            "stage_kimg": 500,
            "fairness": "fine_tune_from_baseline3m",
            "eval_json": ROOT / "training-runs" / "mnist-cdro-finetune-a-100pct-v1x" / "eval-500kimg-10k" / "eval-20260330-133022" / "evaluation_result.json",
        },
    ]

    baseline34 = ROOT / "training-runs" / "mnist-compare-100pct-v3" / "eval" / "baseline-34mimg"
    baseline34_jsons = sorted(baseline34.glob("eval-*/evaluation_result.json"))
    if baseline34_jsons:
        rows.append(
            {
                "model": "baseline_edm_3.4mimg",
                "family": "baseline_edm",
                "stage_kimg": 3400,
                "fairness": "compute_matched_to_cdro_0.4mimg",
                "eval_json": baseline34_jsons[-1],
            }
        )

    for row in rows:
        row["fid"] = load_eval(row["eval_json"])

    rows.sort(key=lambda r: (r["family"], r["stage_kimg"]))

    csv_path = args.output_dir / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "family", "stage_kimg", "fairness", "fid", "eval_json"])
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "eval_json": str(row["eval_json"])})

    # Bar chart for current headline comparison.
    headline = [row for row in rows if row["model"] in {"baseline_edm_3.0mimg", "wdro_edm_3.0mimg", "cdro_ft_0.4mimg", "cdro_ft_0.5mimg", "baseline_edm_3.4mimg"}]
    if not headline:
        headline = [row for row in rows if row["model"] in {"baseline_edm_3.0mimg", "wdro_edm_3.0mimg", "cdro_ft_0.4mimg", "cdro_ft_0.5mimg"}]
    plt.figure(figsize=(8.5, 4.5))
    names = [row["model"] for row in headline]
    values = [row["fid"] for row in headline]
    colors = ["#4C78A8" if "baseline" in n else "#F58518" if "wdro" in n else "#54A24B" for n in names]
    bars = plt.bar(range(len(names)), values, color=colors)
    plt.xticks(range(len(names)), names, rotation=20, ha="right")
    plt.ylabel("FID")
    plt.title("MNIST Matched Model Comparison")
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.03, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(args.output_dir / "fid_bar.png", dpi=180)
    plt.close()

    # CDRO progression chart with baseline / WDRO references.
    cdro_rows = sorted([row for row in rows if row["family"] == "cdro_ft"], key=lambda r: r["stage_kimg"])
    baseline_ref = next(row["fid"] for row in rows if row["model"] == "baseline_edm_3.0mimg")
    wdro_ref = next(row["fid"] for row in rows if row["model"] == "wdro_edm_3.0mimg")
    plt.figure(figsize=(8.5, 4.5))
    plt.plot([row["stage_kimg"] for row in cdro_rows], [row["fid"] for row in cdro_rows], marker="o", linewidth=2, color="#54A24B", label="CDRO fine-tune")
    plt.axhline(baseline_ref, linestyle="--", color="#4C78A8", label=f"Baseline EDM 3.0 MIMG ({baseline_ref:.3f})")
    plt.axhline(wdro_ref, linestyle="--", color="#F58518", label=f"WDRO-EDM 3.0 MIMG ({wdro_ref:.3f})")
    baseline34_row = next((row for row in rows if row["model"] == "baseline_edm_3.4mimg"), None)
    if baseline34_row is not None:
        plt.axhline(baseline34_row["fid"], linestyle=":", color="#4C78A8", label=f"Baseline EDM 3.4 MIMG ({baseline34_row['fid']:.3f})")
    plt.xlabel("CDRO fine-tune stage (kimg)")
    plt.ylabel("FID")
    plt.title("CDRO Fine-Tune Progression")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_dir / "cdro_progression.png", dpi=180)
    plt.close()

    best_cdro = min(cdro_rows, key=lambda r: r["fid"])
    summary_path = args.output_dir / "summary.md"
    lines = [
        "# MNIST Model Comparison",
        "",
        "## Headline",
        f"- Best current CDRO fine-tune: `{best_cdro['fid']:.5f}` at `{best_cdro['stage_kimg']} kimg`",
        f"- Baseline EDM 3.0 MIMG: `{baseline_ref:.5f}`",
        f"- WDRO-EDM 3.0 MIMG: `{wdro_ref:.5f}`",
    ]
    baseline34_row = next((row for row in rows if row["model"] == "baseline_edm_3.4mimg"), None)
    if baseline34_row is not None:
        lines.append(f"- Baseline EDM 3.4 MIMG: `{baseline34_row['fid']:.5f}`")
    lines += [
        "",
        "## Results",
        "",
        "| Model | Stage | FID | Fairness note |",
        "| --- | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(f"| `{row['model']}` | `{row['stage_kimg']}` | `{row['fid']:.5f}` | `{row['fairness']}` |")
    lines += [
        "",
        "## Artifacts",
        "",
        f"- CSV: [{csv_path}]({csv_path})",
        f"- Bar chart: [{args.output_dir / 'fid_bar.png'}]({args.output_dir / 'fid_bar.png'})",
        f"- CDRO progression: [{args.output_dir / 'cdro_progression.png'}]({args.output_dir / 'cdro_progression.png'})",
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary_path)


if __name__ == "__main__":
    main()
