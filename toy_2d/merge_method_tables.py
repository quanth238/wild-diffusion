from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


PREFERRED_METHOD_ORDER = [
    "baseline",
    "wdro",
    "cdro",
    "baseline_score",
    "wdro_score",
    "cdro_markov",
    "baseline_score_raw",
    "wdro_score_raw",
    "cdro_markov_raw",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge aggregate benchmark tables into one summary.")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Named source aggregate, e.g. score_family=toy-runs/score_family_best_200ep_v2/aggregate.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.source:
        raise ValueError("At least one --source is required.")

    sources = parse_sources(args.source)
    merged_rows = merge_summary_rows(sources)
    methods = resolve_method_order(merged_rows)
    winner_counts = compute_winner_counts(merged_rows, methods)

    aggregate = {
        "methods": methods,
        "summary": merged_rows,
        "winner_counts": winner_counts,
        "sources": {name: str(path) for name, path in sources.items()},
    }
    write_outputs(outdir=args.outdir, aggregate=aggregate)


def parse_sources(items: list[str]) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --source value: {item}")
        name, raw_path = item.split("=", 1)
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        resolved[name] = path
    return resolved


def merge_summary_rows(sources: dict[str, Path]) -> list[dict]:
    merged: dict[tuple[str, str], dict] = {}
    fraction_values: dict[tuple[str, str], float] = {}
    dataset_order: list[str] = []
    seen_datasets: set[str] = set()
    fraction_order: dict[str, float] = {}

    for path in sources.values():
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload["summary"]:
            dataset = row["dataset"]
            fraction_tag = row["fraction_tag"]
            key = (dataset, fraction_tag)
            if dataset not in seen_datasets:
                seen_datasets.add(dataset)
                dataset_order.append(dataset)
            fraction = row.get("fraction")
            if fraction is not None:
                fraction_values[key] = float(fraction)
                fraction_order[fraction_tag] = float(fraction)
            target = merged.setdefault(key, {"dataset": dataset, "fraction_tag": fraction_tag})
            if fraction is not None:
                target["fraction"] = float(fraction)
            for item_key, value in row.items():
                if item_key in {"dataset", "fraction_tag", "fraction"}:
                    continue
                target[item_key] = value

    def sort_key(row: dict) -> tuple[int, float]:
        dataset_index = dataset_order.index(row["dataset"])
        fraction_value = row.get("fraction", fraction_order.get(row["fraction_tag"], 0.0))
        return dataset_index, fraction_value

    return sorted(merged.values(), key=sort_key)


def resolve_method_order(rows: list[dict]) -> list[str]:
    found: set[str] = set()
    for row in rows:
        for key in row:
            if key.endswith("_best_swd"):
                found.add(key[: -len("_best_swd")])
    ordered = [method for method in PREFERRED_METHOD_ORDER if method in found]
    extras = sorted(found.difference(ordered))
    return ordered + extras


def compute_winner_counts(rows: list[dict], methods: list[str]) -> dict[str, dict[str, int]]:
    best_counts: dict[str, int] = defaultdict(int)
    last_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        best_value = min(row[f"{method}_best_swd"] for method in methods)
        last_value = min(row[f"{method}_last_swd"] for method in methods)
        best_winners = [method for method in methods if row[f"{method}_best_swd"] == best_value]
        last_winners = [method for method in methods if row[f"{method}_last_swd"] == last_value]
        row["best_winner"] = ", ".join(best_winners)
        row["last_winner"] = ", ".join(last_winners)
        for winner in best_winners:
            best_counts[winner] += 1
        for winner in last_winners:
            last_counts[winner] += 1
    return {
        "best_swd": dict(best_counts),
        "last_swd": dict(last_counts),
    }


def format_method_name(method: str) -> str:
    if method == "baseline":
        return "Baseline"
    if method == "wdro":
        return "WDRO-EDM"
    if method == "cdro":
        return "Legacy CDRO"
    if method == "baseline_score":
        return "Baseline Score"
    if method == "wdro_score":
        return "WDRO Score"
    if method == "cdro_markov":
        return "CDRO Markov"
    if method == "baseline_score_raw":
        return "Baseline Score Raw"
    if method == "wdro_score_raw":
        return "WDRO Score Raw"
    if method == "cdro_markov_raw":
        return "CDRO Markov Raw"
    return method.replace("_", " ").title()


def markdown_row(columns: list[str]) -> str:
    return "| " + " | ".join(str(value) for value in columns) + " |"


def markdown_header(columns: list[str]) -> str:
    align = ["---", "---:"] + ["---:"] * (len(columns) - 3) + ["---"]
    return markdown_row(columns) + "\n" + markdown_row(align)


def write_outputs(*, outdir: Path, aggregate: dict) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "aggregate.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")

    methods = aggregate["methods"]
    header = ["Dataset", "Data"] + [format_method_name(method) for method in methods] + ["Winner"]
    lines = [
        "## Best SWD",
        "",
        markdown_header(header),
    ]
    for row in aggregate["summary"]:
        cells = [row["dataset"], row["fraction_tag"]]
        cells.extend(f"{row[f'{method}_best_swd']:.4f}" for method in methods)
        cells.append(str(row["best_winner"]))
        lines.append(markdown_row(cells))

    lines += [
        "",
        "## Last SWD",
        "",
        markdown_header(header),
    ]
    for row in aggregate["summary"]:
        cells = [row["dataset"], row["fraction_tag"]]
        cells.extend(f"{row[f'{method}_last_swd']:.4f}" for method in methods)
        cells.append(str(row["last_winner"]))
        lines.append(markdown_row(cells))

    lines += [
        "",
        "## Winner Counts",
        "",
        "Counts below include tied cells.",
        "",
        f"Best SWD: {aggregate['winner_counts']['best_swd']}",
        "",
        f"Last SWD: {aggregate['winner_counts']['last_swd']}",
        "",
    ]
    (outdir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
