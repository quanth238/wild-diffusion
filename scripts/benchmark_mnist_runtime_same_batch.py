#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch

ROOT = Path("/root/wild-diffusion")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generate import edm_sampler
from training.dataset import ImageFolderDataset
from training.training_wdro_loop import wdro_attack


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark MNIST runtime with the same batch size.")
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--train-warmup", type=int, default=3)
    parser.add_argument("--train-iters", type=int, default=10)
    parser.add_argument("--attack-warmup", type=int, default=2)
    parser.add_argument("--attack-iters", type=int, default=6)
    parser.add_argument("--sample-count", type=int, default=960)
    parser.add_argument("--sample-batch", type=int, default=96)
    parser.add_argument("--sample-steps", type=int, default=35)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "training-runs" / "mnist-final-comparison-v1",
    )
    return parser.parse_args()


def load_snapshot(path: Path, device: torch.device):
    with path.open("rb") as f:
        data = pickle.load(f)
    net = data["ema"].to(device)
    loss_fn = data.get("loss_fn")
    if loss_fn is not None and "loss_fn_state" in data and hasattr(loss_fn, "load_state_dict"):
        loss_fn.load_state_dict(data["loss_fn_state"])
    return net, loss_fn


def collect_batch(dataset_dir: Path, batch_size: int, device: torch.device):
    dataset = ImageFolderDataset(path=str(dataset_dir), use_labels=True, xflip=False, cache=True)
    images = []
    labels = []
    for idx in range(batch_size):
        image_np, label_np = dataset[idx]
        images.append(torch.from_numpy(image_np).to(torch.float32) / 127.5 - 1.0)
        labels.append(torch.from_numpy(label_np).to(torch.float32))
    return torch.stack(images).to(device), torch.stack(labels).to(device)


def benchmark_train_step(net, loss_fn, images, labels, warmup: int, iters: int) -> dict:
    net = net.train().requires_grad_(True)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8)
    with torch.enable_grad():
        for _ in range(warmup):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(net=net, images=images, labels=labels, augment_pipe=None)
            loss.sum().div(images.shape[0]).backward()
            optimizer.step()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(net=net, images=images, labels=labels, augment_pipe=None)
            loss.sum().div(images.shape[0]).backward()
            optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    return {
        "batch_size": images.shape[0],
        "iters": iters,
        "seconds_total": elapsed,
        "seconds_per_iter": elapsed / iters,
        "images_per_second": (images.shape[0] * iters) / elapsed,
    }


def benchmark_wdro_attack(net, loss_fn, images, labels, gamma: float, alpha: float, iters_attack: int, warmup: int, timed_iters: int) -> dict:
    net = net.eval().requires_grad_(False)
    for _ in range(warmup):
        _ = wdro_attack(images, labels, model=net, loss_fn=loss_fn, augment_pipe=None, gamma=gamma, alpha=alpha, iters=iters_attack)
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(timed_iters):
        _ = wdro_attack(images, labels, model=net, loss_fn=loss_fn, augment_pipe=None, gamma=gamma, alpha=alpha, iters=iters_attack)
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return {
        "batch_size": images.shape[0],
        "iters": timed_iters,
        "seconds_total": elapsed,
        "seconds_per_call": elapsed / timed_iters,
        "images_per_second": (images.shape[0] * timed_iters) / elapsed,
        "wdro_gamma": gamma,
        "wdro_step_size": alpha,
        "wdro_k": iters_attack,
    }


def benchmark_sampling(net, label_dim: int, sample_count: int, sample_batch: int, sample_steps: int, device: torch.device) -> dict:
    net = net.eval().requires_grad_(False)
    batches = (sample_count + sample_batch - 1) // sample_batch
    torch.manual_seed(0)
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        produced = 0
        for _ in range(batches):
            bs = min(sample_batch, sample_count - produced)
            latents = torch.randn((bs, net.img_channels, net.img_resolution, net.img_resolution), device=device)
            class_labels = torch.zeros((bs, label_dim), device=device)
            for i in range(bs):
                class_labels[i, i % label_dim] = 1.0
            _ = edm_sampler(net, latents, class_labels, num_steps=sample_steps)
            produced += bs
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return {
        "sample_count": sample_count,
        "sample_batch": sample_batch,
        "sample_steps": sample_steps,
        "seconds_total": elapsed,
        "seconds_per_sample": elapsed / sample_count,
        "samples_per_second": sample_count / elapsed,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    paths = {
        "baseline_edm": ROOT / "training-runs" / "mnist-compare-100pct-v3" / "runs" / "baseline" / "00000-mnist-32x32-train-cond-ddpmpp-wdroedm-baseline-gpus1-batch512-fp16" / "network-snapshot-003000.pkl",
        "wdro_edm": ROOT / "training-runs" / "mnist-wdro-retune3-g-100pct-v1" / "runs" / "wdro" / "00000-mnist-32x32-train-cond-ddpmpp-wdroedm-wdro-gpus1-batch512-fp16" / "network-snapshot-003000.pkl",
        "cdro_ft": ROOT / "training-runs" / "mnist-cdro-finetune-a-100pct-v1x" / "runs" / "cdro" / "00000-mnist-32x32-train-cond-ddpmpp-cdroedm-baseline-gpus1-batch96-fp16" / "network-snapshot-000400.pkl",
    }
    wdro_training_options = json.loads(
        (ROOT / "training-runs" / "mnist-wdro-retune3-g-100pct-v1" / "runs" / "wdro" / "00000-mnist-32x32-train-cond-ddpmpp-wdroedm-wdro-gpus1-batch512-fp16" / "training_options.json").read_text(encoding="utf-8")
    )

    images, labels = collect_batch(ROOT / "datasets" / "mnist-32x32-train", args.batch_size, device)
    results = {}

    loaded = {name: load_snapshot(path, device) for name, path in paths.items()}
    for name, (net, loss_fn) in loaded.items():
        results[name] = {
            "train_step": benchmark_train_step(net, loss_fn, images, labels, args.train_warmup, args.train_iters),
            "sampling": benchmark_sampling(net, net.label_dim, args.sample_count, args.sample_batch, args.sample_steps, device),
            "peak_gpu_mem_gb": torch.cuda.max_memory_allocated(device) / 2**30,
        }
        torch.cuda.reset_peak_memory_stats(device)

    wdro_net, wdro_loss = loaded["wdro_edm"]
    results["wdro_edm"]["attack_refresh"] = benchmark_wdro_attack(
        wdro_net,
        wdro_loss,
        images,
        labels,
        gamma=float(wdro_training_options["wdro_gamma"]),
        alpha=float(wdro_training_options["wdro_step_size"]),
        iters_attack=int(wdro_training_options["wdro_k"]),
        warmup=args.attack_warmup,
        timed_iters=args.attack_iters,
    )

    json_path = args.output_dir / "runtime_same_batch.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Chart 1: train step sec/iter
    plt.figure(figsize=(7.5, 4.5))
    names = ["baseline_edm", "wdro_edm", "cdro_ft"]
    train_vals = [results[name]["train_step"]["seconds_per_iter"] for name in names]
    plt.bar(names, train_vals, color=["#4C78A8", "#F58518", "#54A24B"])
    plt.ylabel("Seconds per train iteration")
    plt.title(f"Same-batch training step runtime (batch={args.batch_size})")
    plt.tight_layout()
    plt.savefig(args.output_dir / "runtime_same_batch_train.png", dpi=180)
    plt.close()

    # Chart 2: sampling throughput
    plt.figure(figsize=(7.5, 4.5))
    samp_vals = [results[name]["sampling"]["samples_per_second"] for name in names]
    plt.bar(names, samp_vals, color=["#4C78A8", "#F58518", "#54A24B"])
    plt.ylabel("Samples per second")
    plt.title(f"Same-batch sampling runtime (batch={args.sample_batch}, steps={args.sample_steps})")
    plt.tight_layout()
    plt.savefig(args.output_dir / "runtime_same_batch_sampling.png", dpi=180)
    plt.close()

    md = []
    md.append("# Same-Batch Runtime Benchmark")
    md.append("")
    md.append(f"- Training batch: `{args.batch_size}`")
    md.append(f"- Training timing: `{args.train_iters}` timed iterations after `{args.train_warmup}` warmup iterations")
    md.append(f"- Sampling benchmark: `{args.sample_count}` samples, `batch={args.sample_batch}`, `steps={args.sample_steps}`")
    md.append("")
    md.append("## Training step runtime")
    md.append("")
    md.append("| Model | sec/iter | img/s |")
    md.append("| --- | ---: | ---: |")
    for name in names:
        row = results[name]["train_step"]
        md.append(f"| `{name}` | `{row['seconds_per_iter']:.4f}` | `{row['images_per_second']:.2f}` |")
    md.append("")
    md.append("## WDRO refresh overhead")
    md.append("")
    row = results["wdro_edm"]["attack_refresh"]
    md.append(f"- WDRO adversarial refresh call on the same batch: `{row['seconds_per_call']:.4f}` sec/call, `{row['images_per_second']:.2f}` img/s")
    md.append(f"- WDRO refresh config used: `gamma={row['wdro_gamma']}`, `step={row['wdro_step_size']}`, `k={row['wdro_k']}`")
    md.append("")
    md.append("## Sampling runtime")
    md.append("")
    md.append("| Model | sec total | samples/s |")
    md.append("| --- | ---: | ---: |")
    for name in names:
        row = results[name]["sampling"]
        md.append(f"| `{name}` | `{row['seconds_total']:.4f}` | `{row['samples_per_second']:.2f}` |")
    md.append("")
    md.append(f"- JSON: [{json_path}]({json_path})")
    md.append(f"- Train chart: [{args.output_dir / 'runtime_same_batch_train.png'}]({args.output_dir / 'runtime_same_batch_train.png'})")
    md.append(f"- Sampling chart: [{args.output_dir / 'runtime_same_batch_sampling.png'}]({args.output_dir / 'runtime_same_batch_sampling.png'})")

    md_path = args.output_dir / "runtime_same_batch.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(md_path)


if __name__ == "__main__":
    main()
