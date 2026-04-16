#!/usr/bin/env python3
import argparse
import os
import sys

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import dnnlib
import torch
from torch_utils import distributed as dist
from training import training_loop
from training.dataset import ImageFolderDataset

if __package__ is None or __package__ == "":
    from toy.mainline_baseline import (
        resolve_mainline_image_subset_raw_indices,
        snapshot_path_for_kimg,
        state_dump_path_for_kimg,
        steps_to_kimg_exact,
    )
else:
    from ..mainline_baseline import (
        resolve_mainline_image_subset_raw_indices,
        snapshot_path_for_kimg,
        state_dump_path_for_kimg,
        steps_to_kimg_exact,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train one mainline EDM image-baseline segment for toy eval.")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--dataset-val-path", type=str, default="")
    parser.add_argument("--image-size", type=int, required=True)
    parser.add_argument("--image-train-size", type=int, required=True)
    parser.add_argument("--image-val-size", type=int, required=True)
    parser.add_argument("--image-split-seed", type=int, default=0)
    parser.add_argument("--limited-data-enabled", action="store_true")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--batch-gpu", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--target-steps", type=int, required=True)
    parser.add_argument("--resume-steps", type=int, default=0)
    parser.add_argument("--resume-pkl", type=str, default="")
    parser.add_argument("--resume-state-dump", type=str, default="")
    parser.add_argument("--sigma-data", type=float, required=True)
    parser.add_argument("--lr-theta", type=float, required=True)
    parser.add_argument("--p-mean", type=float, default=-1.2)
    parser.add_argument("--p-std", type=float, default=1.2)
    parser.add_argument("--ema-halflife-kimg", type=float, default=500.0)
    parser.add_argument("--ema-rampup-ratio", type=float, default=0.05)
    parser.add_argument("--disable-ema-rampup", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--cudnn-benchmark", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if int(args.hidden_dim) != 128:
        raise ValueError(f"--hidden-dim must be 128 for mainline ddpmpp, got {args.hidden_dim}")
    if float(args.sigma_data) <= 0.0:
        raise ValueError(f"--sigma-data must be > 0, got {args.sigma_data}")
    if float(args.lr_theta) <= 0.0:
        raise ValueError(f"--lr-theta must be > 0, got {args.lr_theta}")
    if float(args.p_std) <= 0.0:
        raise ValueError(f"--p-std must be > 0, got {args.p_std}")
    if int(args.workers) <= 0:
        raise ValueError(f"--workers must be > 0, got {args.workers}")
    if int(args.batch_size) <= 0:
        raise ValueError(f"--batch-size must be > 0, got {args.batch_size}")
    if int(args.batch_gpu) < 0:
        raise ValueError(f"--batch-gpu must be >= 0, got {args.batch_gpu}")
    if int(args.batch_gpu) > 0:
        if int(args.batch_gpu) > int(args.batch_size):
            raise ValueError(
                f"--batch-gpu must be <= --batch-size, got batch_gpu={args.batch_gpu}, batch_size={args.batch_size}"
            )
        if int(args.batch_size) % int(args.batch_gpu) != 0:
            raise ValueError(
                "--batch-gpu must divide --batch-size exactly for mainline gradient accumulation, "
                f"got batch_gpu={args.batch_gpu}, batch_size={args.batch_size}"
            )
    if not torch.cuda.is_available():
        raise RuntimeError("Mainline EDM baseline training requires CUDA, but CUDA is unavailable.")

    target_kimg = steps_to_kimg_exact(args.target_steps, args.batch_size, label="target_steps")
    resume_kimg = steps_to_kimg_exact(args.resume_steps, args.batch_size, label="resume_steps")
    if resume_kimg > target_kimg:
        raise ValueError(
            f"--resume-steps must map to <= target kimg, got resume={resume_kimg} target={target_kimg}"
        )
    if int(args.resume_steps) > 0:
        if not args.resume_pkl or not args.resume_state_dump:
            raise ValueError("--resume-pkl and --resume-state-dump are required when --resume-steps > 0.")
        if not os.path.isfile(args.resume_pkl):
            raise FileNotFoundError(f"Resume snapshot not found: {args.resume_pkl}")
        if not os.path.isfile(args.resume_state_dump):
            raise FileNotFoundError(f"Resume state dump not found: {args.resume_state_dump}")

    os.makedirs(args.run_dir, exist_ok=True)

    preview_dataset = ImageFolderDataset(
        path=args.dataset_path,
        resolution=int(args.image_size),
        use_labels=False,
        xflip=False,
        cache=False,
    )
    try:
        subset_raw_idx = resolve_mainline_image_subset_raw_indices(
            argparse.Namespace(
                dataset_val_path=str(args.dataset_val_path),
                limited_data_enabled=bool(args.limited_data_enabled),
                image_train_size=int(args.image_train_size),
                image_val_size=int(args.image_val_size),
                image_split_seed=int(args.image_split_seed),
            ),
            total_images=len(preview_dataset),
        )
    finally:
        preview_dataset.close()

    dataset_kwargs = dnnlib.EasyDict(
        class_name="toy.mainline_dataset.ToyImageFolderSubsetDataset",
        path=args.dataset_path,
        resolution=int(args.image_size),
        use_labels=False,
        xflip=False,
        cache=True,
        subset_raw_idx=subset_raw_idx,
    )
    data_loader_kwargs = dnnlib.EasyDict(
        pin_memory=True,
        num_workers=int(args.workers),
        prefetch_factor=2,
    )
    network_kwargs = dnnlib.EasyDict(
        class_name="training.networks.EDMPrecond",
        sigma_data=float(args.sigma_data),
        model_type="SongUNet",
        embedding_type="positional",
        encoder_type="standard",
        decoder_type="standard",
        channel_mult_noise=1,
        resample_filter=[1, 1],
        model_channels=128,
        channel_mult=[2, 2, 2],
        num_blocks=4,
        attn_resolutions=[16],
        dropout=float(args.dropout),
        use_fp16=False,
    )
    loss_kwargs = dnnlib.EasyDict(
        class_name="training.loss.EDMLossWdro",
        P_mean=float(args.p_mean),
        P_std=float(args.p_std),
        sigma_data=float(args.sigma_data),
    )
    optimizer_kwargs = dnnlib.EasyDict(
        class_name="torch.optim.Adam",
        lr=float(args.lr_theta),
        betas=[0.9, 0.999],
        eps=1e-8,
    )

    tick_kimg = max(int(target_kimg - resume_kimg), 1)
    if not torch.distributed.is_initialized():
        dist.init()

    training_loop.training_loop(
        run_dir=str(args.run_dir),
        dataset_kwargs=dataset_kwargs,
        data_loader_kwargs=data_loader_kwargs,
        network_kwargs=network_kwargs,
        loss_kwargs=loss_kwargs,
        optimizer_kwargs=optimizer_kwargs,
        augment_kwargs=None,
        seed=int(args.seed),
        batch_size=int(args.batch_size),
        batch_gpu=(None if int(args.batch_gpu) <= 0 else int(args.batch_gpu)),
        total_kimg=int(target_kimg),
        ema_halflife_kimg=float(args.ema_halflife_kimg),
        ema_rampup_ratio=None if bool(args.disable_ema_rampup) else float(args.ema_rampup_ratio),
        lr_rampup_kimg=0.0,
        loss_scaling=1.0,
        kimg_per_tick=tick_kimg,
        snapshot_ticks=10**9,
        state_dump_ticks=10**9,
        resume_pkl=(str(args.resume_pkl).strip() or None),
        resume_state_dump=(str(args.resume_state_dump).strip() or None),
        resume_kimg=int(resume_kimg),
        cudnn_benchmark=bool(args.cudnn_benchmark),
        wandb_kwargs=dnnlib.EasyDict(enabled=False),
        wandb_run=None,
        device=torch.device("cuda"),
    )

    final_snapshot = snapshot_path_for_kimg(args.run_dir, target_kimg)
    final_state_dump = state_dump_path_for_kimg(args.run_dir, target_kimg)
    print(f"[mainline-baseline] snapshot={final_snapshot}", flush=True)
    print(f"[mainline-baseline] state_dump={final_state_dump}", flush=True)


if __name__ == "__main__":
    main()
