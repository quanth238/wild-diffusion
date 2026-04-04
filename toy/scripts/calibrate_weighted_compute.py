#!/usr/bin/env python3
import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone

import torch


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from toy.config import ToyConfig
from toy.models import ImageEDMDenoiser, ImageScoreModel
from toy.shared.objective import compute_training_loss
from toy.shared.runtime import autocast_context, configure_runtime, format_amp_dtype, resolve_amp_dtype
from toy.utils import batch_scalar_like, ensure_dir, pick_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate weighted compute ratios for forward-only, input-gradient, and parameter-backward ops."
    )
    parser.add_argument("--out-json", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp-dtype", type=str, default="auto")
    parser.add_argument("--disable-tf32", action="store_true")
    parser.add_argument("--disable-cudnn-benchmark", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--training-objective", type=str, default="edm", choices=["edm", "score"])
    parser.add_argument("--score-matching-weight-power", type=float, default=2.0)
    parser.add_argument("--image-size", type=int, default=28)
    parser.add_argument("--image-channels", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--sigma-data", type=float, default=0.5)
    parser.add_argument("--sigma-min", type=float, default=0.002)
    parser.add_argument("--sigma-max", type=float, default=2.0)
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--measure-iters", type=int, default=40)
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_cfg(args: argparse.Namespace) -> ToyConfig:
    return ToyConfig(
        device=str(args.device),
        amp_dtype=str(args.amp_dtype),
        allow_tf32=not bool(args.disable_tf32),
        cudnn_benchmark=not bool(args.disable_cudnn_benchmark),
        training_objective=str(args.training_objective),
        score_matching_weight_power=float(args.score_matching_weight_power),
        sigma_data=float(args.sigma_data),
        image_size=int(args.image_size),
        image_channels=int(args.image_channels),
        batch_size=int(args.batch_size),
        hidden_dim=int(args.hidden_dim),
        sigma_min=float(args.sigma_min),
        sigma_max=float(args.sigma_max),
    )


def build_denoiser(cfg: ToyConfig, device: torch.device) -> torch.nn.Module:
    denoiser_cls = ImageEDMDenoiser if cfg.training_objective == "edm" else ImageScoreModel
    denoiser = denoiser_cls(
        in_channels=int(cfg.image_channels),
        hidden_dim=int(cfg.hidden_dim),
        sigma_data=float(cfg.sigma_data),
    ).to(device)
    return denoiser


def make_batch(cfg: ToyConfig, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_clean = torch.randn(
        int(cfg.batch_size),
        int(cfg.image_channels),
        int(cfg.image_size),
        int(cfg.image_size),
        device=device,
    )
    sigma = torch.exp(
        torch.empty(int(cfg.batch_size), device=device).uniform_(
            float(torch.log(torch.tensor(cfg.sigma_min)).item()),
            float(torch.log(torch.tensor(cfg.sigma_max)).item()),
        )
    )
    x_noisy = x_clean + batch_scalar_like(sigma, x_clean) * torch.randn_like(x_clean)
    return x_noisy, x_clean, sigma


def benchmark(op_name: str, fn, warmup_iters: int, measure_iters: int, device: torch.device) -> dict:
    for _ in range(int(warmup_iters)):
        fn()
    synchronize(device)

    samples = []
    for _ in range(int(measure_iters)):
        synchronize(device)
        t0 = time.perf_counter()
        fn()
        synchronize(device)
        samples.append(float(time.perf_counter() - t0))

    return {
        "op_name": op_name,
        "num_samples": int(len(samples)),
        "mean_sec": float(statistics.mean(samples)),
        "median_sec": float(statistics.median(samples)),
        "min_sec": float(min(samples)),
        "max_sec": float(max(samples)),
        "samples_sec": [float(v) for v in samples],
    }


def main() -> None:
    args = parse_args()
    cfg = build_cfg(args)
    device = pick_device(cfg.device)
    configure_runtime(
        device=device,
        allow_tf32=bool(cfg.allow_tf32),
        cudnn_benchmark=bool(cfg.cudnn_benchmark),
    )
    amp_dtype = resolve_amp_dtype(device, cfg.amp_dtype)
    set_seed(int(args.seed))

    denoiser = build_denoiser(cfg, device)
    denoiser.train()
    x_noisy_ref, x_clean_ref, sigma_ref = make_batch(cfg, device)

    def forward_only():
        with torch.no_grad():
            with autocast_context(device, amp_dtype):
                loss = compute_training_loss(cfg, denoiser, x_noisy_ref, x_clean_ref, sigma_ref)
            _ = float(loss.detach().item())

    def forward_plus_inputgrad():
        denoiser.zero_grad(set_to_none=True)
        x_noisy = x_noisy_ref.detach().clone().requires_grad_(True)
        with autocast_context(device, amp_dtype):
            loss = compute_training_loss(cfg, denoiser, x_noisy, x_clean_ref, sigma_ref)
        grad_x = torch.autograd.grad(loss, x_noisy, only_inputs=True)[0]
        _ = float(grad_x.detach().square().mean().item())

    def forward_plus_parambackward():
        denoiser.zero_grad(set_to_none=True)
        with autocast_context(device, amp_dtype):
            loss = compute_training_loss(cfg, denoiser, x_noisy_ref, x_clean_ref, sigma_ref)
        loss.backward()
        grad_sum = 0.0
        for parameter in denoiser.parameters():
            if parameter.grad is not None:
                grad_sum += float(parameter.grad.detach().abs().mean().item())
                break
        _ = grad_sum

    forward_stats = benchmark(
        "forward_only",
        forward_only,
        warmup_iters=int(args.warmup_iters),
        measure_iters=int(args.measure_iters),
        device=device,
    )
    inputgrad_stats = benchmark(
        "forward_plus_inputgrad",
        forward_plus_inputgrad,
        warmup_iters=int(args.warmup_iters),
        measure_iters=int(args.measure_iters),
        device=device,
    )
    parambackward_stats = benchmark(
        "forward_plus_parambackward",
        forward_plus_parambackward,
        warmup_iters=int(args.warmup_iters),
        measure_iters=int(args.measure_iters),
        device=device,
    )

    forward_median = float(forward_stats["median_sec"])
    inputgrad_median = float(inputgrad_stats["median_sec"])
    parambackward_median = float(parambackward_stats["median_sec"])
    if forward_median <= 0.0:
        raise RuntimeError(f"Forward-only benchmark returned a non-positive median: {forward_median}")

    device_name = str(device)
    gpu_count = 1 if device.type == "cuda" else 0
    if device.type == "cuda" and torch.cuda.is_available():
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        device_name = str(torch.cuda.get_device_name(device_index))

    payload = {
        "format": "toy_weighted_compute_calibration_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hardware": {
            "device": str(device),
            "device_type": str(device.type),
            "device_name": device_name,
            "gpu_count": int(gpu_count),
            "allow_tf32": bool(cfg.allow_tf32),
            "cudnn_benchmark": bool(cfg.cudnn_benchmark),
            "amp_dtype": format_amp_dtype(amp_dtype),
        },
        "workload": {
            "model_kind": "image_conv",
            "training_objective": str(cfg.training_objective),
            "score_matching_weight_power": float(cfg.score_matching_weight_power),
            "batch_size": int(cfg.batch_size),
            "image_channels": int(cfg.image_channels),
            "image_size": int(cfg.image_size),
            "hidden_dim": int(cfg.hidden_dim),
            "sigma_data": float(cfg.sigma_data),
            "sigma_min": float(cfg.sigma_min),
            "sigma_max": float(cfg.sigma_max),
        },
        "benchmark": {
            "seed": int(args.seed),
            "warmup_iters": int(args.warmup_iters),
            "measure_iters": int(args.measure_iters),
        },
        "timings_sec": {
            "forward_only": forward_stats,
            "forward_plus_inputgrad": inputgrad_stats,
            "forward_plus_parambackward": parambackward_stats,
        },
        "ratios": {
            "inputgrad_alpha": float(inputgrad_median / forward_median),
            "parambackward_beta": float(parambackward_median / forward_median),
        },
    }

    ensure_dir(os.path.dirname(os.path.abspath(args.out_json)) or ".")
    with open(args.out_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(
        "[weighted-calibration] "
        f"device={device_name} amp_dtype={format_amp_dtype(amp_dtype)} "
        f"forward_median={forward_median:.6f}s "
        f"inputgrad_alpha={payload['ratios']['inputgrad_alpha']:.4f} "
        f"parambackward_beta={payload['ratios']['parambackward_beta']:.4f}",
        flush=True,
    )
    print(f"[weighted-calibration] wrote {os.path.abspath(args.out_json)}", flush=True)


if __name__ == "__main__":
    main()
