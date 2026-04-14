#!/usr/bin/env python3
import argparse
import json
import os
import pickle
import statistics
import sys
import time
from datetime import datetime, timezone

import torch


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import dnnlib  # noqa: E402
from torch_utils import misc  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate weighted-compute ratios on the real CIFAR image stack by benchmarking "
            "forward-only, forward+inputgrad, and forward+parameter-backward costs."
        )
    )
    parser.add_argument("--out-json", type=str, required=True)
    parser.add_argument("--network-pkl", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arch", type=str, default="ddpmpp", choices=["ddpmpp", "ncsnpp", "adm"])
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--image-channels", type=int, default=3)
    parser.add_argument("--label-dim", type=int, default=0)
    parser.add_argument("--fp16", type=int, default=1, choices=[0, 1])
    parser.add_argument("--dropout", type=float, default=0.13)
    parser.add_argument("--augment", type=float, default=0.12)
    parser.add_argument("--sigma-data", type=float, default=0.5)
    parser.add_argument("--attack-gamma", type=float, default=1.0)
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--measure-iters", type=int, default=20)
    return parser.parse_args()


def build_network_kwargs(args: argparse.Namespace) -> dnnlib.EasyDict:
    kwargs = dnnlib.EasyDict(class_name="training.networks.EDMPrecond")
    if args.arch == "ddpmpp":
        kwargs.update(
            model_type="SongUNet",
            embedding_type="positional",
            encoder_type="standard",
            decoder_type="standard",
            channel_mult_noise=1,
            resample_filter=[1, 1],
            model_channels=128,
            channel_mult=[2, 2, 2],
        )
    elif args.arch == "ncsnpp":
        kwargs.update(
            model_type="SongUNet",
            embedding_type="fourier",
            encoder_type="residual",
            decoder_type="standard",
            channel_mult_noise=2,
            resample_filter=[1, 3, 3, 1],
            model_channels=128,
            channel_mult=[2, 2, 2],
        )
    else:
        kwargs.update(model_type="DhariwalUNet", model_channels=192, channel_mult=[1, 2, 3, 4])

    kwargs.update(
        use_fp16=bool(args.fp16),
        dropout=float(args.dropout),
        sigma_data=float(args.sigma_data),
    )
    return kwargs


def build_augment_pipe(args: argparse.Namespace, device: torch.device):
    if float(args.augment) <= 0.0:
        return None
    augment_kwargs = dnnlib.EasyDict(class_name="training.augment.AugmentPipe", p=float(args.augment))
    augment_kwargs.update(xflip=1e8, yflip=1, scale=1, rotate_frac=1, aniso=1, translate_frac=1)
    del device
    return dnnlib.util.construct_class_by_name(**augment_kwargs)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


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


def load_snapshot_if_requested(net: torch.nn.Module, network_pkl: str) -> None:
    if not str(network_pkl).strip():
        return
    with dnnlib.util.open_url(str(network_pkl).strip(), verbose=True) as handle:
        payload = pickle.load(handle)
    misc.copy_params_and_buffers(src_module=payload["ema"], dst_module=net, require_all=False)


def main() -> None:
    args = parse_args()
    device = torch.device(str(args.device))
    if device.type != "cuda":
        raise RuntimeError("This calibration is intended for CUDA devices.")
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False.")

    torch.manual_seed(int(args.seed))
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    interface_kwargs = dict(
        img_resolution=int(args.image_size),
        img_channels=int(args.image_channels),
        label_dim=int(args.label_dim),
    )
    net = dnnlib.util.construct_class_by_name(**build_network_kwargs(args), **interface_kwargs)
    net = net.train().requires_grad_(True).to(device)
    load_snapshot_if_requested(net, args.network_pkl)
    net.train()

    loss_fn = dnnlib.util.construct_class_by_name(class_name="training.loss.EDMLossWdro", sigma_data=float(args.sigma_data))
    augment_pipe = build_augment_pipe(args, device)

    images_ref = torch.randn(
        int(args.batch_size),
        int(args.image_channels),
        int(args.image_size),
        int(args.image_size),
        device=device,
        dtype=torch.float32,
    ).clamp_(-1.0, 1.0)
    labels_ref = None
    if int(args.label_dim) > 0:
        class_idx = torch.randint(int(args.label_dim), size=[int(args.batch_size)], device=device)
        labels_ref = torch.eye(int(args.label_dim), device=device)[class_idx]

    def forward_only():
        net.zero_grad(set_to_none=True)
        net.train()
        with torch.no_grad():
            loss = loss_fn(net=net, images=images_ref, labels=labels_ref, augment_pipe=augment_pipe).mean()
            _ = float(loss.detach().item())

    def forward_plus_inputgrad():
        net.zero_grad(set_to_none=True)
        net.train()
        x_adv = images_ref.detach().clone().requires_grad_(True)
        loss_cls = loss_fn(net=net, images=x_adv, labels=labels_ref, augment_pipe=augment_pipe).mean()
        delta = (x_adv - images_ref).view(images_ref.shape[0], -1)
        constraint = 0.5 * delta.pow(2).sum(dim=1).mean()
        loss = loss_cls - float(args.attack_gamma) * constraint
        grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]
        _ = float(grad.detach().square().mean().item())

    def forward_plus_parambackward():
        net.zero_grad(set_to_none=True)
        net.train()
        loss = loss_fn(net=net, images=images_ref, labels=labels_ref, augment_pipe=augment_pipe).mean()
        loss.backward()
        grad_probe = 0.0
        for parameter in net.parameters():
            if parameter.grad is not None:
                grad_probe = float(parameter.grad.detach().abs().mean().item())
                break
        _ = grad_probe

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

    device_index = device.index if device.index is not None else torch.cuda.current_device()
    device_name = str(torch.cuda.get_device_name(device_index))
    payload = {
        "format": "image_weighted_compute_calibration_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hardware": {
            "device": str(device),
            "device_type": str(device.type),
            "device_name": device_name,
            "gpu_count": 1,
            "allow_tf32": False,
            "cudnn_benchmark": True,
        },
        "workload": {
            "model_kind": "image_edmprecond",
            "arch": str(args.arch),
            "precond": "wdroedm",
            "batch_size": int(args.batch_size),
            "image_channels": int(args.image_channels),
            "image_size": int(args.image_size),
            "label_dim": int(args.label_dim),
            "use_fp16": bool(args.fp16),
            "dropout": float(args.dropout),
            "augment_p": float(args.augment),
            "sigma_data": float(args.sigma_data),
            "attack_gamma": float(args.attack_gamma),
            "network_pkl": (os.path.abspath(str(args.network_pkl)) if str(args.network_pkl).strip() else None),
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

    out_json = os.path.abspath(str(args.out_json))
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(
        "[weighted-calibration] "
        f"device={device_name} batch={int(args.batch_size)} fp16={bool(args.fp16)} "
        f"forward_median={forward_median:.6f}s "
        f"inputgrad_alpha={payload['ratios']['inputgrad_alpha']:.4f} "
        f"parambackward_beta={payload['ratios']['parambackward_beta']:.4f}",
        flush=True,
    )
    print(f"[weighted-calibration] wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
