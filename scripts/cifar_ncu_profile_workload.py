#!/usr/bin/env python3
"""NVTX-marked CIFAR denoiser primitives for Nsight Compute diagnostics.

This script intentionally does not define the paper FLOP convention. It only
creates representative CUDA work ranges that `ncu` can profile as a secondary
kernel/instruction diagnostic.
"""

import argparse
import os
import sys
from datetime import datetime, timezone

import torch


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import dnnlib  # noqa: E402
from scripts.calibrate_cifar_flops import (  # noqa: E402
    build_augment_pipe,
    build_network_kwargs,
    load_snapshot_if_requested,
    synchronize,
)


OPERATIONS = ("forward_only", "forward_plus_inputgrad", "forward_plus_parambackward")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one NVTX-marked CIFAR denoiser primitive for Nsight Compute. "
            "Use scripts/profile_cifar_ncu_flops.py to launch this under ncu."
        )
    )
    parser.add_argument("--operation", type=str, required=True, choices=OPERATIONS)
    parser.add_argument("--nvtx-range", type=str, default="")
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
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--measure-iters", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(str(args.device))
    if device.type != "cuda":
        raise RuntimeError("Nsight Compute diagnostics require a CUDA device.")
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

    loss_fn = dnnlib.util.construct_class_by_name(
        class_name="training.loss.EDMLossWdro",
        sigma_data=float(args.sigma_data),
    )
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

    def forward_only() -> None:
        net.zero_grad(set_to_none=True)
        net.train()
        with torch.no_grad():
            loss = loss_fn(net=net, images=images_ref, labels=labels_ref, augment_pipe=augment_pipe).mean()
            _ = float(loss.detach().item())

    def forward_plus_inputgrad() -> None:
        net.zero_grad(set_to_none=True)
        net.train()
        x_adv = images_ref.detach().clone().requires_grad_(True)
        loss_cls = loss_fn(net=net, images=x_adv, labels=labels_ref, augment_pipe=augment_pipe).mean()
        delta = (x_adv - images_ref).view(images_ref.shape[0], -1)
        constraint = 0.5 * delta.pow(2).sum(dim=1).mean()
        loss = loss_cls - float(args.attack_gamma) * constraint
        grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]
        _ = float(grad.detach().square().mean().item())

    def forward_plus_parambackward() -> None:
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

    operation_fn = {
        "forward_only": forward_only,
        "forward_plus_inputgrad": forward_plus_inputgrad,
        "forward_plus_parambackward": forward_plus_parambackward,
    }[str(args.operation)]

    for _ in range(max(int(args.warmup_iters), 0)):
        operation_fn()
    synchronize(device)

    range_name = str(args.nvtx_range).strip() or f"cifar_ncu_{args.operation}"
    torch.cuda.nvtx.range_push(range_name)
    try:
        for _ in range(max(int(args.measure_iters), 1)):
            operation_fn()
    finally:
        torch.cuda.nvtx.range_pop()
    synchronize(device)

    print(
        "[cifar-ncu-workload] "
        f"finished operation={args.operation} batch={int(args.batch_size)} "
        f"measure_iters={max(int(args.measure_iters), 1)} nvtx_range={range_name} "
        f"finished_at={datetime.now(timezone.utc).isoformat()}",
        file=sys.stderr,
        flush=True,
    )


if __name__ == "__main__":
    main()
