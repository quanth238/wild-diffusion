from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from toy_2d import normalize_comparison_method_names


TITLE_HEIGHT = 64
CARD_HEADER_HEIGHT = 42
CARD_FOOTER_HEIGHT = 34
CARD_GAP = 18
SIDE_PAD = 20
INNER_PAD = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export stitched sample/reverse/training comparison PNGs.")
    parser.add_argument("--comparison-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--fraction-tags", nargs="*", default=None)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epoch-mode", type=str, default="best", choices=("best", "last", "fixed"))
    parser.add_argument("--fixed-epoch", type=int, default=None)
    parser.add_argument("--out-subdir", type=str, default="visual_comparisons")
    parser.add_argument("--target-width", type=int, default=1280)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.methods = normalize_comparison_method_names(args.methods)
    datasets = resolve_datasets(
        comparison_dir=args.comparison_dir,
        datasets=args.datasets,
        methods=args.methods,
        seed=args.seed,
    )

    for dataset in datasets:
        fraction_tags = resolve_fraction_tags(
            comparison_dir=args.comparison_dir,
            dataset=dataset,
            fraction_tags=args.fraction_tags,
            methods=args.methods,
            seed=args.seed,
        )
        for fraction_tag in fraction_tags:
            export_bundle(
                comparison_dir=args.comparison_dir,
                out_subdir=args.out_subdir,
                dataset=dataset,
                fraction_tag=fraction_tag,
                methods=args.methods,
                seed=args.seed,
                epoch_mode=args.epoch_mode,
                fixed_epoch=args.fixed_epoch,
                target_width=int(args.target_width),
            )


def resolve_datasets(
    *,
    comparison_dir: Path,
    datasets: list[str] | None,
    methods: list[str],
    seed: int,
) -> list[str]:
    if datasets:
        return datasets
    resolved: list[str] = []
    for path in sorted(comparison_dir.iterdir()):
        if not path.is_dir():
            continue
        found = False
        for fraction_dir in sorted(path.iterdir()):
            if not fraction_dir.is_dir():
                continue
            if any((fraction_dir / method / f"seed{seed}" / "summary.json").is_file() for method in methods):
                found = True
                break
        if found:
            resolved.append(path.name)
    return resolved


def resolve_fraction_tags(
    *,
    comparison_dir: Path,
    dataset: str,
    fraction_tags: list[str] | None,
    methods: list[str],
    seed: int,
) -> list[str]:
    if fraction_tags:
        return fraction_tags
    dataset_dir = comparison_dir / dataset
    resolved: list[str] = []
    for path in sorted(dataset_dir.iterdir()):
        if not path.is_dir():
            continue
        if any((path / method / f"seed{seed}" / "summary.json").is_file() for method in methods):
            resolved.append(path.name)
    return resolved


def export_bundle(
    *,
    comparison_dir: Path,
    out_subdir: str,
    dataset: str,
    fraction_tag: str,
    methods: list[str],
    seed: int,
    epoch_mode: str,
    fixed_epoch: int | None,
    target_width: int,
) -> None:
    sample_cards: list[dict] = []
    reverse_cards: list[dict] = []
    training_cards: list[dict] = []

    for method in methods:
        run_dir = comparison_dir / dataset / fraction_tag / method / f"seed{seed}"
        summary_path = run_dir / "summary.json"
        if not summary_path.is_file():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        epoch = resolve_epoch(summary=summary, epoch_mode=epoch_mode, fixed_epoch=fixed_epoch)
        caption = build_caption(summary=summary, epoch=epoch, epoch_mode=epoch_mode)
        sample_cards.append(
            {
                "label": pretty_method_name(method),
                "caption": caption,
                "image": run_dir / "plots" / f"samples_epoch_{epoch:04d}.png",
            }
        )
        reverse_cards.append(
            {
                "label": pretty_method_name(method),
                "caption": caption,
                "image": run_dir / "plots" / f"reverse_process_epoch_{epoch:04d}.png",
            }
        )
        training_cards.append(
            {
                "label": pretty_method_name(method),
                "caption": (
                    f"best={summary['best_sliced_wasserstein']:.4f} @ {summary['best_epoch']} | "
                    f"last={summary['last_eval']['sliced_wasserstein']:.4f} | "
                    f"runtime={summary['runtime_minutes']:.2f} min"
                ),
                "image": run_dir / "plots" / "training_curves.png",
            }
        )

    out_dir = comparison_dir / out_subdir / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    title_base = f"{dataset} | {fraction_tag} | seed {seed}"

    save_card_sheet(
        path=out_dir / f"{fraction_tag}_denoising_{epoch_mode}_seed{seed}.png",
        title=f"{title_base} | Denoising",
        cards=sample_cards,
        target_width=target_width,
    )
    save_card_sheet(
        path=out_dir / f"{fraction_tag}_noising_process_{epoch_mode}_seed{seed}.png",
        title=f"{title_base} | Noising Process",
        cards=reverse_cards,
        target_width=target_width,
    )
    save_card_sheet(
        path=out_dir / f"{fraction_tag}_training_seed{seed}.png",
        title=f"{title_base} | training curves",
        cards=training_cards,
        target_width=target_width,
    )


def resolve_epoch(*, summary: dict, epoch_mode: str, fixed_epoch: int | None) -> int:
    if epoch_mode == "best":
        return int(summary["best_epoch"])
    if epoch_mode == "last":
        return int(summary["last_eval"]["epoch"])
    if fixed_epoch is None:
        raise ValueError("--fixed-epoch is required when --epoch-mode fixed.")
    return int(fixed_epoch)


def build_caption(*, summary: dict, epoch: int, epoch_mode: str) -> str:
    best = float(summary["best_sliced_wasserstein"])
    best_epoch = int(summary["best_epoch"])
    last = float(summary["last_eval"]["sliced_wasserstein"])
    mode_label = f"{epoch_mode} epoch={epoch}"
    return f"{mode_label} | best={best:.4f} @ {best_epoch} | last={last:.4f}"


def pretty_method_name(method: str) -> str:
    mapping = {
        "baseline": "Baseline",
        "baseline_edm": "Baseline EDM",
        "wdro": "WDRO",
        "wdro_edm": "WDRO EDM",
        "cdro": "Legacy CDRO",
        "cdro_edm": "CDRO EDM",
        "baseline_score": "Baseline Score",
        "baseline_score_raw": "Baseline Score Raw",
        "wdro_score": "WDRO Score",
        "wdro_score_raw": "WDRO Score Raw",
        "cdro_markov": "CDRO Markov",
        "cdro_markov_raw": "CDRO Markov Raw",
    }
    return mapping.get(method, method.replace("_", " ").title())


def save_card_sheet(*, path: Path, title: str, cards: list[dict], target_width: int) -> None:
    if not cards:
        return
    font = ImageFont.load_default()

    rendered_cards: list[Image.Image] = []
    for card in cards:
        rendered_cards.append(render_card(card=card, font=font, target_width=target_width))

    sheet_width = max(image.width for image in rendered_cards) + 2 * SIDE_PAD
    sheet_height = TITLE_HEIGHT + SIDE_PAD + sum(image.height for image in rendered_cards) + CARD_GAP * (len(rendered_cards) - 1) + SIDE_PAD
    canvas = Image.new("RGB", (sheet_width, sheet_height), color=(248, 249, 251))
    draw = ImageDraw.Draw(canvas)
    draw.text((SIDE_PAD, 18), title, fill=(20, 20, 20), font=font)

    y = TITLE_HEIGHT
    for image in rendered_cards:
        x = (sheet_width - image.width) // 2
        canvas.paste(image, (x, y))
        y += image.height + CARD_GAP

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def render_card(*, card: dict, font, target_width: int) -> Image.Image:
    image_path = Path(card["image"])
    body = load_body_image(path=image_path, target_width=target_width)
    card_width = body.width + 2 * INNER_PAD
    card_height = CARD_HEADER_HEIGHT + body.height + CARD_FOOTER_HEIGHT + 2 * INNER_PAD

    canvas = Image.new("RGB", (card_width, card_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((0, 0, card_width - 1, card_height - 1), radius=12, outline=(210, 214, 220), width=1, fill=(255, 255, 255))

    draw.text((INNER_PAD, 10), str(card["label"]), fill=(16, 16, 16), font=font)
    draw.text((INNER_PAD, card_height - CARD_FOOTER_HEIGHT + 8), str(card["caption"]), fill=(80, 80, 80), font=font)
    canvas.paste(body, (INNER_PAD, CARD_HEADER_HEIGHT))
    return canvas


def load_body_image(*, path: Path, target_width: int) -> Image.Image:
    if path.is_file():
        image = Image.open(path).convert("RGB")
        return ImageOps.contain(image, (target_width, 2000), method=Image.Resampling.LANCZOS)

    placeholder = Image.new("RGB", (target_width, max(target_width // 2, 360)), color=(242, 243, 245))
    draw = ImageDraw.Draw(placeholder)
    font = ImageFont.load_default()
    message = f"Missing image:\n{path.name}"
    draw.multiline_text((24, 24), message, fill=(120, 120, 120), font=font, spacing=6)
    return placeholder


if __name__ == "__main__":
    main()
