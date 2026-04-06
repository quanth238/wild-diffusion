from __future__ import annotations

import argparse
import json
from pathlib import Path

from toy_2d.plotting import save_labeled_metric_tradeoff_curves


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overlay a metric against time or another x-axis across arbitrary runs.")
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Pairs of label=run_dir, where run_dir contains metrics.jsonl.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output image path.")
    parser.add_argument("--title", type=str, default="Metric tradeoff comparison")
    parser.add_argument("--metric-key", type=str, default="sliced_wasserstein")
    parser.add_argument("--x-key", type=str, default="elapsed_minutes")
    parser.add_argument("--x-label", type=str, default=None)
    parser.add_argument("--y-label", type=str, default=None)
    parser.add_argument(
        "--best-so-far",
        action="store_true",
        help="Plot the running best (minimum) metric against the chosen x-axis.",
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


def parse_run_specs(run_specs: list[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for spec in run_specs:
        if "=" not in spec:
            raise ValueError(f"Run spec must be label=path, got: {spec}")
        label, raw_path = spec.split("=", 1)
        label = label.strip()
        run_path = Path(raw_path.strip())
        if not label:
            raise ValueError(f"Run label cannot be empty: {spec}")
        parsed[label] = run_path
    return parsed


def main() -> None:
    args = parse_args()
    run_dirs = parse_run_specs(args.runs)
    histories: dict[str, list[dict]] = {}
    for label, run_dir in run_dirs.items():
        metrics_path = run_dir / "metrics.jsonl"
        if not metrics_path.is_file():
            raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
        histories[label] = load_jsonl(metrics_path)

    save_labeled_metric_tradeoff_curves(
        path=args.out,
        title=args.title,
        run_histories=histories,
        metric_key=args.metric_key,
        x_key=args.x_key,
        x_label=args.x_label,
        y_label=args.y_label,
        best_so_far=args.best_so_far,
    )
    print(args.out)


if __name__ == "__main__":
    main()
