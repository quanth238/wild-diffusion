from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from toy_2d.plotting import save_labeled_metric_tradeoff_curves


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Average metric-vs-x curves across multiple run directories per label.")
    parser.add_argument(
        "--groups",
        nargs="+",
        required=True,
        help="Specs of the form label=run_dir1,run_dir2,... where each run dir contains metrics.jsonl.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", type=str, default="Seed-averaged metric tradeoff comparison")
    parser.add_argument("--metric-key", type=str, default="sliced_wasserstein")
    parser.add_argument("--x-key", type=str, default="elapsed_minutes")
    parser.add_argument("--x-label", type=str, default=None)
    parser.add_argument("--y-label", type=str, default=None)
    parser.add_argument(
        "--best-so-far",
        action="store_true",
        help="Plot the running best (minimum) metric against the chosen x-axis after seed averaging.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def parse_group_specs(group_specs: list[str]) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for spec in group_specs:
        if "=" not in spec:
            raise ValueError(f"Group spec must be label=path1,path2,..., got: {spec}")
        label, raw_paths = spec.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"Group label cannot be empty: {spec}")
        paths = [Path(item.strip()) for item in raw_paths.split(",") if item.strip()]
        if not paths:
            raise ValueError(f"Group must include at least one run dir: {spec}")
        groups[label] = paths
    return groups


def average_histories(
    histories: list[list[dict]],
    *,
    metric_key: str,
    x_key: str,
) -> list[dict]:
    by_epoch: dict[int, list[dict]] = defaultdict(list)
    for history in histories:
        for row in history:
            if row.get("epoch") is None:
                continue
            if row.get(metric_key) is None or row.get(x_key) is None:
                continue
            by_epoch[int(row["epoch"])].append(row)

    averaged: list[dict] = []
    for epoch in sorted(by_epoch):
        rows = by_epoch[epoch]
        averaged.append(
            {
                "epoch": int(epoch),
                metric_key: sum(float(row[metric_key]) for row in rows) / len(rows),
                x_key: sum(float(row[x_key]) for row in rows) / len(rows),
            }
        )
    return averaged


def main() -> None:
    args = parse_args()
    groups = parse_group_specs(args.groups)

    averaged_histories: dict[str, list[dict]] = {}
    for label, run_dirs in groups.items():
        histories: list[list[dict]] = []
        for run_dir in run_dirs:
            metrics_path = run_dir / "metrics.jsonl"
            if not metrics_path.is_file():
                raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
            histories.append(load_jsonl(metrics_path))
        averaged_histories[label] = average_histories(histories, metric_key=args.metric_key, x_key=args.x_key)

    save_labeled_metric_tradeoff_curves(
        path=args.out,
        title=args.title,
        run_histories=averaged_histories,
        metric_key=args.metric_key,
        x_key=args.x_key,
        x_label=args.x_label,
        y_label=args.y_label,
        best_so_far=args.best_so_far,
    )
    print(args.out)


if __name__ == "__main__":
    main()
