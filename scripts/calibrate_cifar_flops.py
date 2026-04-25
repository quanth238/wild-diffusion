#!/usr/bin/env python3
import argparse
import json
import math
import os
import pickle
import statistics
import sys
import time
from datetime import datetime, timezone

import torch
from torch.profiler import ProfilerActivity, profile


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import dnnlib  # noqa: E402
from torch_utils import misc  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate per-batch denoiser FLOPs on the real CIFAR image stack using "
            "torch.profiler operator FLOP accounting for forward-only, "
            "forward+inputgrad, and forward+parameter-backward primitives."
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
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--measure-iters", type=int, default=5)
    parser.add_argument("--training-inputgrad-forward-multiplier", type=float, default=2.0)
    parser.add_argument("--training-parambackward-forward-multiplier", type=float, default=3.0)
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


def load_snapshot_if_requested(net: torch.nn.Module, network_pkl: str) -> None:
    if not str(network_pkl).strip():
        return
    with dnnlib.util.open_url(str(network_pkl).strip(), verbose=True) as handle:
        payload = pickle.load(handle)
    misc.copy_params_and_buffers(src_module=payload["ema"], dst_module=net, require_all=False)


def _hidden_dim_for_arch(arch: str) -> int:
    return 192 if str(arch) == "adm" else 128


def _amp_dtype_label(fp16: int) -> str:
    return "float16" if bool(fp16) else "float32"


def _profiler_activities(device: torch.device):
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    return activities


def _extract_total_flops(prof) -> float:
    total_flops = 0.0
    for event in prof.key_averages():
        event_flops = getattr(event, "flops", None)
        if event_flops is None:
            continue
        value = float(event_flops)
        if math.isfinite(value) and value > 0.0:
            total_flops += value
    return float(total_flops)


def profile_flops(
    op_name: str,
    fn,
    *,
    warmup_iters: int,
    measure_iters: int,
    device: torch.device,
    batch_size: int,
) -> dict:
    for _ in range(int(warmup_iters)):
        fn()
    synchronize(device)

    samples_flops = []
    samples_wall_clock_sec = []
    activities = _profiler_activities(device)
    for _ in range(int(measure_iters)):
        synchronize(device)
        t0 = time.perf_counter()
        with profile(
            activities=activities,
            record_shapes=True,
            profile_memory=False,
            with_flops=True,
        ) as prof:
            fn()
        synchronize(device)
        total_flops = _extract_total_flops(prof)
        elapsed_sec = float(time.perf_counter() - t0)
        if total_flops <= 0.0:
            raise RuntimeError(
                f"Profiler reported non-positive FLOPs for {op_name}. "
                "This usually means the current torch.profiler build cannot "
                "attribute FLOPs for the requested workload."
            )
        samples_flops.append(float(total_flops))
        samples_wall_clock_sec.append(elapsed_sec)

    median_flops = float(statistics.median(samples_flops))
    mean_flops = float(statistics.mean(samples_flops))
    median_wall_clock_sec = float(statistics.median(samples_wall_clock_sec))
    return {
        "op_name": op_name,
        "num_samples": int(len(samples_flops)),
        "mean_per_batch_flops": mean_flops,
        "median_per_batch_flops": median_flops,
        "min_per_batch_flops": float(min(samples_flops)),
        "max_per_batch_flops": float(max(samples_flops)),
        "samples_per_batch_flops": [float(v) for v in samples_flops],
        "per_batch_flops": median_flops,
        "per_image_flops": float(median_flops / float(batch_size)),
        "mean_wall_clock_sec": float(statistics.mean(samples_wall_clock_sec)),
        "median_wall_clock_sec": median_wall_clock_sec,
        "samples_wall_clock_sec": [float(v) for v in samples_wall_clock_sec],
    }


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
    network_kwargs = build_network_kwargs(args)
    net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs)
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

    forward_stats = profile_flops(
        "forward_only",
        forward_only,
        warmup_iters=int(args.warmup_iters),
        measure_iters=int(args.measure_iters),
        device=device,
        batch_size=int(args.batch_size),
    )
    inputgrad_stats = profile_flops(
        "forward_plus_inputgrad",
        forward_plus_inputgrad,
        warmup_iters=int(args.warmup_iters),
        measure_iters=int(args.measure_iters),
        device=device,
        batch_size=int(args.batch_size),
    )
    parambackward_stats = profile_flops(
        "forward_plus_parambackward",
        forward_plus_parambackward,
        warmup_iters=int(args.warmup_iters),
        measure_iters=int(args.measure_iters),
        device=device,
        batch_size=int(args.batch_size),
    )

    forward_median = float(forward_stats["per_batch_flops"])
    inputgrad_median = float(inputgrad_stats["per_batch_flops"])
    parambackward_median = float(parambackward_stats["per_batch_flops"])
    if forward_median <= 0.0:
        raise RuntimeError(f"Forward-only FLOP benchmark returned a non-positive median: {forward_median}")
    inputgrad_training_multiplier = float(args.training_inputgrad_forward_multiplier)
    parambackward_training_multiplier = float(args.training_parambackward_forward_multiplier)
    if inputgrad_training_multiplier <= 0.0 or parambackward_training_multiplier <= 0.0:
        raise RuntimeError("Training FLOP multipliers must be positive.")
    training_inputgrad_flops = float(forward_median * inputgrad_training_multiplier)
    training_parambackward_flops = float(forward_median * parambackward_training_multiplier)

    device_index = device.index if device.index is not None else torch.cuda.current_device()
    device_name = str(torch.cuda.get_device_name(device_index))
    payload = {
        "format": "image_flop_calibration_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hardware": {
            "device": str(device),
            "device_type": str(device.type),
            "device_name": device_name,
            "gpu_count": 1,
            "allow_tf32": False,
            "cudnn_benchmark": True,
            "amp_dtype": _amp_dtype_label(int(args.fp16)),
        },
        "workload": {
            "model_kind": "image_edmprecond",
            "training_objective": "edm",
            "image_backbone": str(args.arch),
            "hidden_dim": int(_hidden_dim_for_arch(str(args.arch))),
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
            "tool": "torch.profiler",
            "with_flops": True,
            "record_shapes": True,
            "flop_definition": "sum(key_averages().flops) over profiler-supported operators per measured batch",
            "activities": [str(activity).split(".")[-1].lower() for activity in _profiler_activities(device)],
        },
        "flop_accounting": {
            "selected_cost_group": "training_flops",
            "selected_cost_source": "analytical_training_from_profiler_forward",
            "rationale": (
                "torch.profiler with_flops is used to measure forward-only denoiser FLOPs. "
                "CUDA backward FLOP attribution is incomplete for autograd graphs, so training "
                "compute uses the standard analytical convention: forward+input-gradient = 2x "
                "forward and forward+parameter-backward = 3x forward."
            ),
        },
        "flops": {
            "forward_only": forward_stats,
            "forward_plus_inputgrad": inputgrad_stats,
            "forward_plus_parambackward": parambackward_stats,
        },
        "training_flops": {
            "source": "analytical_training_from_profiler_forward",
            "definition": (
                "Measured forward-only torch.profiler FLOPs; input-gradient primitive = "
                f"{inputgrad_training_multiplier:g}x forward; parameter-backward training primitive = "
                f"{parambackward_training_multiplier:g}x forward."
            ),
            "forward_only": {
                "per_batch_flops": float(forward_median),
                "per_image_flops": float(forward_median / float(args.batch_size)),
                "forward_multiplier": 1.0,
            },
            "forward_plus_inputgrad": {
                "per_batch_flops": float(training_inputgrad_flops),
                "per_image_flops": float(training_inputgrad_flops / float(args.batch_size)),
                "forward_multiplier": float(inputgrad_training_multiplier),
            },
            "forward_plus_parambackward": {
                "per_batch_flops": float(training_parambackward_flops),
                "per_image_flops": float(training_parambackward_flops / float(args.batch_size)),
                "forward_multiplier": float(parambackward_training_multiplier),
            },
        },
        "profiler_diagnostics": {
            "forward_plus_inputgrad_profiler_over_forward": float(inputgrad_median / forward_median),
            "forward_plus_parambackward_profiler_over_forward": float(parambackward_median / forward_median),
            "forward_plus_inputgrad_wall_clock_over_forward": float(
                inputgrad_stats["median_wall_clock_sec"] / forward_stats["median_wall_clock_sec"]
            ),
            "forward_plus_parambackward_wall_clock_over_forward": float(
                parambackward_stats["median_wall_clock_sec"] / forward_stats["median_wall_clock_sec"]
            ),
            "note": (
                "Profiler FLOP ratios near 1x for backward workloads indicate unsupported or "
                "incomplete CUDA backward FLOP attribution; use training_flops for reported "
                "training compute."
            ),
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
        "[flop-calibration] "
        f"device={device_name} batch={int(args.batch_size)} fp16={bool(args.fp16)} "
        f"forward_per_batch_flops={forward_median:.3e} "
        f"training_inputgrad={inputgrad_training_multiplier:.3g}x "
        f"training_parambackward={parambackward_training_multiplier:.3g}x "
        f"profiler_inputgrad_alpha={payload['ratios']['inputgrad_alpha']:.4f} "
        f"profiler_parambackward_beta={payload['ratios']['parambackward_beta']:.4f}",
        flush=True,
    )
    print(f"[flop-calibration] wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
