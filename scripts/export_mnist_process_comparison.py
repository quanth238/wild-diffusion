#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import torch
import torchvision
from PIL import Image, ImageDraw

ROOT = Path("/root/wild-diffusion")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.loss import EDMLossWdro
from training.training_wdro_loop import wdro_attack


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export MNIST noising and denoising comparison sheets.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "training-runs" / "mnist-final-comparison-v1",
        help="Where to write the PNGs.",
    )
    return parser.parse_args()


def load_bundle(path: Path, device: torch.device):
    with open(path, "rb") as f:
        data = pickle.load(f)
    net = data["ema"].to(device).eval()
    loss_fn = data.get("loss_fn")
    if loss_fn is not None and hasattr(loss_fn, "to"):
        loss_fn = loss_fn.to(device)
    return net, loss_fn


def make_one_hot(labels: list[int], label_dim: int, device: torch.device) -> torch.Tensor:
    out = torch.zeros(len(labels), label_dim, device=device)
    for i, label in enumerate(labels):
        out[i, label] = 1
    return out


def load_real_images(dataset_dir: Path, labels: list[int], device: torch.device) -> torch.Tensor:
    images = []
    for label in labels:
        path = sorted((dataset_dir / str(label)).glob("*.png"))[0]
        img = Image.open(path).convert("L")
        tensor = torchvision.transforms.functional.pil_to_tensor(img).float() / 127.5 - 1.0
        images.append(tensor)
    return torch.stack(images, dim=0).to(device)


def to_uint8(images: torch.Tensor) -> torch.Tensor:
    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)
    images = images.detach().cpu().clamp(-1, 1)
    return ((images + 1) * 127.5).round().to(torch.uint8)


def grid_image(images: torch.Tensor, nrow: int) -> Image.Image:
    grid = torchvision.utils.make_grid(to_uint8(images), nrow=nrow, padding=2)
    return Image.fromarray(grid.permute(1, 2, 0).numpy())


def add_titled_rows(rows: list[tuple[str, list[Image.Image]]], output: Path) -> None:
    label_w = 180
    pad = 12
    label_h = 26
    cell_w = max(img.width for _, imgs in rows for img in imgs)
    cell_h = max(img.height for _, imgs in rows for img in imgs)
    num_cols = max(len(imgs) for _, imgs in rows)
    width = label_w + pad * (num_cols + 2) + cell_w * num_cols
    height = pad + len(rows) * (label_h + cell_h + pad)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    y = pad
    for title, imgs in rows:
        draw.text((pad, y), title, fill="black")
        x = label_w
        for img in imgs:
            canvas.paste(img, (x, y + label_h))
            x += cell_w + pad
        y += label_h + cell_h + pad

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def edm_sampler_trajectory(net, latents, class_labels, num_steps=35, capture_indices=None):
    if capture_indices is None:
        capture_indices = [0, 8, 16, 24, 34]
    sigma_min = max(0.002, float(net.sigma_min))
    sigma_max = min(80.0, float(net.sigma_max))
    rho = 7
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    frames = []
    x_next = latents.to(torch.float64) * t_steps[0]
    if 0 in capture_indices:
        frames.append(x_next.to(torch.float32))

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        denoised = net(x_cur, t_cur, class_labels).to(torch.float64)
        d_cur = (x_cur - denoised) / t_cur
        x_next = x_cur + (t_next - t_cur) * d_cur
        if i < num_steps - 1:
            denoised_next = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised_next) / t_next
            x_next = x_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)
        if i in capture_indices[1:]:
            frames.append(x_next.to(torch.float32))
    return frames


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    baseline_pkl = ROOT / "training-runs" / "mnist-compare-100pct-v3" / "runs" / "baseline" / "00000-mnist-32x32-train-cond-ddpmpp-wdroedm-baseline-gpus1-batch512-fp16" / "network-snapshot-003000.pkl"
    wdro_pkl = ROOT / "training-runs" / "mnist-wdro-retune3-g-100pct-v1" / "runs" / "wdro" / "00000-mnist-32x32-train-cond-ddpmpp-wdroedm-wdro-gpus1-batch512-fp16" / "network-snapshot-003000.pkl"
    cdro_pkl = ROOT / "training-runs" / "mnist-cdro-finetune-a-100pct-v1x" / "runs" / "cdro" / "00000-mnist-32x32-train-cond-ddpmpp-cdroedm-baseline-gpus1-batch96-fp16" / "network-snapshot-000400.pkl"
    wdro_opts = json.loads(
        (ROOT / "training-runs" / "mnist-wdro-retune3-g-100pct-v1" / "runs" / "wdro" / "00000-mnist-32x32-train-cond-ddpmpp-wdroedm-wdro-gpus1-batch512-fp16" / "training_options.json").read_text(encoding="utf-8")
    )

    baseline_net, _ = load_bundle(baseline_pkl, device)
    wdro_net, _ = load_bundle(wdro_pkl, device)
    cdro_net, cdro_loss = load_bundle(cdro_pkl, device)
    if cdro_loss is None:
        raise RuntimeError("CDRO snapshot did not contain loss_fn.")

    # Forward / noising process comparison.
    labels = [0, 1, 2, 3]
    class_labels = make_one_hot(labels, baseline_net.label_dim, device)
    real_images = load_real_images(ROOT / "datasets" / "mnist-32x32-train", labels, device)
    generator = torch.Generator(device=device).manual_seed(1234)
    eps = torch.randn(real_images.shape, generator=generator, device=device)
    sigmas = [0.10, 0.30, 0.70]

    baseline_row = [grid_image(real_images, nrow=len(labels))]
    wdro_row = [grid_image(real_images, nrow=len(labels))]
    cdro_row = [grid_image(real_images, nrow=len(labels))]

    wdro_loss = EDMLossWdro()
    wdro_adv = wdro_attack(
        real_images,
        class_labels,
        model=wdro_net,
        loss_fn=wdro_loss,
        augment_pipe=None,
        gamma=wdro_opts["wdro_gamma"],
        alpha=wdro_opts["wdro_step_size"],
        iters=wdro_opts["wdro_k"],
    ).detach()

    for sigma_value in sigmas:
        sigma = torch.full((real_images.shape[0], 1, 1, 1), sigma_value, device=device)
        noise = eps * sigma
        baseline_row.append(grid_image((real_images + noise).clamp(-1, 1), nrow=len(labels)))
        wdro_row.append(grid_image((wdro_adv + noise).clamp(-1, 1), nrow=len(labels)))
        batch = cdro_loss.build_batch(
            cdro_net,
            real_images,
            class_labels,
            augment_pipe=None,
            sigma=torch.full((real_images.shape[0], 1, 1, 1), sigma_value, device=device),
            noise=noise,
        )
        cdro_row.append(grid_image((batch["base_noisy"] + batch["applied_delta"]).clamp(-1, 1), nrow=len(labels)))

    add_titled_rows(
        [
            ("Baseline noising: clean, sigma=0.10, 0.30, 0.70", baseline_row),
            ("WDRO noising: clean, sigma=0.10, 0.30, 0.70", wdro_row),
            ("CDRO noising: clean, sigma=0.10, 0.30, 0.70", cdro_row),
        ],
        args.output_dir / "noising_process_comparison.png",
    )

    # Denoising trajectory comparison.
    latents = torch.randn((4, baseline_net.img_channels, baseline_net.img_resolution, baseline_net.img_resolution), generator=generator, device=device)
    traj_labels = make_one_hot([0, 1, 2, 3], baseline_net.label_dim, device)

    baseline_frames = edm_sampler_trajectory(baseline_net, latents, traj_labels)
    wdro_frames = edm_sampler_trajectory(wdro_net, latents, traj_labels)
    cdro_frames = edm_sampler_trajectory(cdro_net, latents, traj_labels)

    add_titled_rows(
        [
            ("Baseline denoising trajectory", [grid_image(frame, nrow=4) for frame in baseline_frames]),
            ("WDRO denoising trajectory", [grid_image(frame, nrow=4) for frame in wdro_frames]),
            ("CDRO denoising trajectory", [grid_image(frame, nrow=4) for frame in cdro_frames]),
        ],
        args.output_dir / "denoising_process_comparison.png",
    )

    print(args.output_dir)


if __name__ == "__main__":
    main()
