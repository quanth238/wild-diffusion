#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import importlib
import pickle
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.dataset import ImageFolderDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect MNIST CDRO perturbations on a fixed batch.")
    parser.add_argument("--cdro-network-pkl", type=Path, required=True)
    parser.add_argument("--baseline-network-pkl", type=Path, default=None)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=1)
    parser.add_argument("--sigma-values", type=str, default="0.05,0.15,0.3,0.6")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def load_snapshot(path: Path):
    with path.open("rb") as f:
        data = pickle.load(f)
    net = data["ema"].eval().requires_grad_(False)
    loss_fn = None
    # Prefer reconstructing from training_options so local code edits are reflected
    # when analyzing an older snapshot.
    if path.parent.joinpath("training_options.json").is_file():
        training_options = json.loads(path.parent.joinpath("training_options.json").read_text(encoding="utf-8"))
        loss_kwargs = training_options.get("loss_kwargs", {})
        class_name = loss_kwargs.pop("class_name", None)
        if class_name is not None:
            module_name, class_attr = class_name.rsplit(".", 1)
            loss_cls = getattr(importlib.import_module(module_name), class_attr)
            loss_fn = loss_cls(**loss_kwargs)
    if loss_fn is None:
        loss_fn = data.get("loss_fn", None)
    if loss_fn is not None and "loss_fn_state" in data and hasattr(loss_fn, "load_state_dict"):
        loss_fn.load_state_dict(data["loss_fn_state"])
    return net, loss_fn


def collect_balanced_batch(dataset_dir: Path, per_class: int):
    dataset = ImageFolderDataset(path=str(dataset_dir), use_labels=True, xflip=False, cache=True)
    images = []
    labels = []
    counts = {i: 0 for i in range(10)}
    for idx in range(len(dataset)):
        image_np, label_np = dataset[idx]
        label = int(label_np.argmax())
        if counts[label] >= per_class:
            continue
        image = torch.from_numpy(image_np).to(torch.float32) / 127.5 - 1.0
        images.append(image)
        labels.append(torch.from_numpy(label_np).to(torch.float32))
        counts[label] += 1
        if all(v >= per_class for v in counts.values()):
            break
    if not images:
        raise RuntimeError("No images collected from dataset.")
    return torch.stack(images), torch.stack(labels)


def to_u8(img: torch.Tensor) -> Image.Image:
    x = img.detach().cpu().clamp(-1, 1)
    if x.ndim == 3:
        x = x.squeeze(0)
    arr = ((x + 1) * 127.5).round().to(torch.uint8).numpy()
    return Image.fromarray(arr, mode="L")


def to_delta_u8(img: torch.Tensor) -> Image.Image:
    x = img.detach().cpu()
    x = x.abs()
    x = x / (x.max().clamp(min=1e-8))
    if x.ndim == 3:
        x = x.squeeze(0)
    arr = (x * 255).round().to(torch.uint8).numpy()
    return Image.fromarray(arr, mode="L")


def tile_row(images, scale=4, pad=2):
    ims = [ImageOps.autocontrast(im).resize((im.width * scale, im.height * scale), Image.Resampling.NEAREST) for im in images]
    width = sum(im.width for im in ims) + pad * (len(ims) + 1)
    height = ims[0].height + pad * 2
    canvas = Image.new("L", (width, height), color=255)
    x = pad
    for im in ims:
        canvas.paste(im, (x, pad))
        x += im.width + pad
    return canvas


def write_panel(path: Path, title: str, row_specs: list[tuple[str, list[Image.Image]]]):
    label_width = 180
    label_height = 20
    margin = 10
    row_tiles = [tile_row(images) for _, images in row_specs]
    width = label_width + max(tile.width for tile in row_tiles) + margin * 3
    height = margin + sum(tile.height + label_height + margin for tile in row_tiles)
    canvas = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 2), title, fill="black")
    y = margin + 18
    for (label, _), tile in zip(row_specs, row_tiles):
        draw.text((margin, y), label, fill="black")
        canvas.paste(Image.merge("RGB", (tile, tile, tile)), (label_width, y))
        y += tile.height + label_height + margin
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    cdro_net, cdro_loss = load_snapshot(args.cdro_network_pkl)
    if cdro_loss is None or not hasattr(cdro_loss, "build_batch"):
        raise RuntimeError("CDRO snapshot does not contain a usable EDMLossCDRO object.")
    cdro_net = cdro_net.to(device)

    baseline_net = None
    if args.baseline_network_pkl is not None:
        baseline_net, _ = load_snapshot(args.baseline_network_pkl)
        baseline_net = baseline_net.to(device)

    images, labels = collect_balanced_batch(args.dataset_dir, args.per_class)
    images = images.to(device)
    labels = labels.to(device)
    batch_size = images.shape[0]

    summary = {"rows": []}
    for sigma_value in [float(x) for x in args.sigma_values.split(",") if x.strip()]:
        sigma = torch.full((batch_size, 1, 1, 1), sigma_value, device=device)
        noise = torch.randn_like(images) * sigma
        with torch.enable_grad():
            batch = cdro_loss.build_batch(cdro_net, images, labels, sigma=sigma, noise=noise)
        if baseline_net is not None:
            baseline_pred = baseline_net(batch["base_noisy"], batch["sigma"], labels)
        else:
            baseline_pred = None

        title = (
            f"sigma={sigma_value:.3f} | gate_mean={batch['gate'].mean().item():.4f} | "
            f"transport_mean={batch['transport_per'].mean().item():.6e} | "
            f"applied_rms={batch['applied_delta'].square().mean(dim=[1,2,3]).sqrt().mean().item():.6f}"
        )
        rows = [
            ("Clean", [to_u8(x) for x in images]),
            ("Noisy", [to_u8(x) for x in batch["base_noisy"]]),
            ("Perturbed", [to_u8(x) for x in (batch["base_noisy"] + batch["applied_delta"])]),
            ("|Applied Delta|", [to_delta_u8(x) for x in batch["applied_delta"]]),
            ("CDRO Denoise(noisy)", [to_u8(x) for x in batch["pred_clean"]]),
            ("CDRO Denoise(perturbed)", [to_u8(x) for x in batch["pred_robust"]]),
        ]
        if baseline_pred is not None:
            rows.insert(4, ("Baseline Denoise(noisy)", [to_u8(x) for x in baseline_pred]))

        out_path = args.output_dir / f"sigma_{sigma_value:.3f}.png"
        write_panel(out_path, title, rows)
        summary["rows"].append(
            {
                "sigma": sigma_value,
                "gate_mean": float(batch["gate"].mean().item()),
                "transport_mean": float(batch["transport_per"].mean().item()),
                "raw_delta_rms": float(batch["delta"].square().mean(dim=[1, 2, 3]).sqrt().mean().item()),
                "applied_delta_rms": float(batch["applied_delta"].square().mean(dim=[1, 2, 3]).sqrt().mean().item()),
                "mse_cdro_noisy": float(((batch["pred_clean"] - batch["clean"]) ** 2).mean().item()),
                "mse_cdro_perturbed": float(((batch["pred_robust"] - batch["clean"]) ** 2).mean().item()),
                "mse_baseline_noisy": float(((baseline_pred - batch["clean"]) ** 2).mean().item()) if baseline_pred is not None else None,
                "panel": str(out_path),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(args.output_dir / "summary.json")


if __name__ == "__main__":
    main()
