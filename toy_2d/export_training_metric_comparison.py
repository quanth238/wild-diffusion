from __future__ import annotations

import argparse
import json
from pathlib import Path

from toy_2d import normalize_comparison_method_names
from toy_2d.plotting import save_method_training_metric_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export overlaid training/eval metric plots across methods.")
    parser.add_argument("--comparison-dir", type=Path, required=True, help="Root comparison directory.")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Datasets to export. Defaults to all dataset directories under comparison-dir.",
    )
    parser.add_argument("--fraction-tag", type=str, default="100pct", help="Fraction tag subdirectory, e.g. 100pct.")
    parser.add_argument("--seed", type=int, default=0, help="Seed to export.")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["baseline", "wdro", "cdro", "cdro_markov"],
        help="Methods to overlay.",
    )
    parser.add_argument(
        "--out-subdir",
        type=str,
        default="training_metric_comparisons",
        help="Subdirectory inside comparison-dir for exported plots.",
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


def resolve_datasets(
    comparison_dir: Path,
    datasets: list[str] | None,
    *,
    fraction_tag: str,
    methods: list[str],
    seed: int,
) -> list[str]:
    if datasets:
        return datasets

    resolved: list[str] = []
    for path in sorted(comparison_dir.iterdir()):
        if not path.is_dir():
            continue
        has_metrics = any(
            (path / fraction_tag / method / f"seed{seed}" / "metrics.jsonl").exists()
            for method in methods
        )
        if has_metrics:
            resolved.append(path.name)
    return resolved


def main() -> None:
    args = parse_args()
    args.methods = normalize_comparison_method_names(args.methods)
    comparison_dir = args.comparison_dir
    datasets = resolve_datasets(
        comparison_dir,
        args.datasets,
        fraction_tag=args.fraction_tag,
        methods=args.methods,
        seed=args.seed,
    )

    for dataset in datasets:
        method_histories: dict[str, list[dict]] = {}
        for method in args.methods:
            metrics_path = comparison_dir / dataset / args.fraction_tag / method / f"seed{args.seed}" / "metrics.jsonl"
            if not metrics_path.exists():
                raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
            method_histories[method] = load_jsonl(metrics_path)

        out_path = (
            comparison_dir
            / args.out_subdir
            / dataset
            / f"training_eval_comparison_seed{args.seed}.png"
        )
        save_method_training_metric_comparison(
            path=out_path,
            dataset=dataset,
            fraction_tag=args.fraction_tag,
            seed=args.seed,
            method_histories=method_histories,
        )
        print(out_path)


if __name__ == "__main__":
    main()
