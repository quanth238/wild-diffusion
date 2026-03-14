import json
import os
import pickle
import time

import dnnlib
import numpy as np
import PIL.Image
import torch
from fid import calculate_fid_from_inception_stats
from generate import edm_sampler
from torch_utils import distributed as dist
from training import dataset as dataset_lib


def _images_to_uint8(images):
    return (images * 127.5 + 128).clip(0, 255).to(torch.uint8)


def _safe_cpu_mem_gb():
    # Telemetry must never crash training if process accounting is transiently unavailable.
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / 2**30
    except (ImportError, OSError):
        return float("nan")
    except Exception:
        return float("nan")


def _save_image_grid(images_uint8, path, ncols=8):
    if images_uint8 is None or images_uint8.numel() == 0:
        return
    images = images_uint8.cpu()
    n, c, h, w = images.shape
    ncols = max(1, min(ncols, n))
    nrows = (n + ncols - 1) // ncols
    canvas = np.zeros((nrows * h, ncols * w, 3), dtype=np.uint8)
    for idx in range(n):
        row = idx // ncols
        col = idx % ncols
        img = images[idx].permute(1, 2, 0).numpy()
        if img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        canvas[row * h : (row + 1) * h, col * w : (col + 1) * w, :] = img
    os.makedirs(os.path.dirname(path), exist_ok=True)
    PIL.Image.fromarray(canvas, "RGB").save(path)


def _delta_to_bw_uint8(orig_images, adv_images):
    # Per-pixel absolute difference magnitude, normalized per image to improve contrast.
    delta = (adv_images - orig_images).abs().mean(dim=1, keepdim=True)
    delta_max = delta.amax(dim=(2, 3), keepdim=True).clamp(min=1e-8)
    delta = (delta / delta_max).clamp(0, 1)
    return (delta * 255).to(torch.uint8).repeat(1, 3, 1, 1)


def _save_triplet_rows_grid(orig_uint8, adv_uint8, delta_uint8, path, max_cols=8):
    if orig_uint8 is None or adv_uint8 is None or delta_uint8 is None:
        return
    if orig_uint8.numel() == 0 or adv_uint8.numel() == 0 or delta_uint8.numel() == 0:
        return

    orig = orig_uint8.cpu()
    adv = adv_uint8.cpu()
    delta = delta_uint8.cpu()

    n = min(orig.shape[0], adv.shape[0], delta.shape[0])
    if n < 1:
        return
    ncols = max(1, min(int(max_cols), n))
    nshow = ncols

    h = int(orig.shape[2])
    w = int(orig.shape[3])
    canvas = np.zeros((3 * h, nshow * w, 3), dtype=np.uint8)

    for col in range(nshow):
        orig_img = orig[col].permute(1, 2, 0).numpy()
        adv_img = adv[col].permute(1, 2, 0).numpy()
        delta_img = delta[col].permute(1, 2, 0).numpy()
        if orig_img.shape[2] == 1:
            orig_img = np.repeat(orig_img, 3, axis=2)
        if adv_img.shape[2] == 1:
            adv_img = np.repeat(adv_img, 3, axis=2)
        if delta_img.shape[2] == 1:
            delta_img = np.repeat(delta_img, 3, axis=2)
        x0 = col * w
        x1 = (col + 1) * w
        canvas[0 * h : 1 * h, x0:x1, :] = orig_img
        canvas[1 * h : 2 * h, x0:x1, :] = adv_img
        canvas[2 * h : 3 * h, x0:x1, :] = delta_img

    os.makedirs(os.path.dirname(path), exist_ok=True)
    PIL.Image.fromarray(canvas, "RGB").save(path)


def _save_individual_images(images_uint8, out_dir, prefix):
    if images_uint8 is None or images_uint8.numel() == 0:
        return
    os.makedirs(out_dir, exist_ok=True)
    for idx, image in enumerate(images_uint8.cpu()):
        image_np = image.permute(1, 2, 0).numpy()
        if image_np.shape[2] == 1:
            PIL.Image.fromarray(image_np[:, :, 0], "L").save(
                os.path.join(out_dir, f"{prefix}-{idx:03d}.png")
            )
        else:
            PIL.Image.fromarray(image_np, "RGB").save(
                os.path.join(out_dir, f"{prefix}-{idx:03d}.png")
            )


def _load_inception_detector(device):
    detector_path = os.environ.get("FID_DETECTOR_PATH", "").strip()
    detector_url = "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl"
    if detector_path:
        if not os.path.isfile(detector_path):
            raise FileNotFoundError(f"FID_DETECTOR_PATH does not exist: {detector_path}")
        with open(detector_path, "rb") as f:
            detector_net = pickle.load(f).to(device)
    else:
        with dnnlib.util.open_url(detector_url, verbose=(dist.get_rank() == 0)) as f:
            detector_net = pickle.load(f).to(device)
    detector_net.eval().requires_grad_(False)
    return detector_net


def _compute_feature_stats_from_tensor(detector_net, images_uint8, batch_size, device):
    n = int(images_uint8.shape[0])
    if n < 2:
        raise RuntimeError("Need at least 2 images to compute FID statistics.")
    feature_dim = 2048
    mu = torch.zeros([feature_dim], dtype=torch.float64, device=device)
    sigma = torch.zeros([feature_dim, feature_dim], dtype=torch.float64, device=device)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            images = images_uint8[start:end].to(device)
            if images.shape[1] == 1:
                images = images.repeat([1, 3, 1, 1])
            features = detector_net(images, return_features=True).to(torch.float64)
            mu += features.sum(0)
            sigma += features.T @ features
    mu /= n
    sigma -= mu.ger(mu) * n
    sigma /= max(n - 1, 1)
    return mu.cpu().numpy(), sigma.cpu().numpy()


def _compute_reference_stats_from_dataset(
    detector_net, dataset_path, batch_size, num_workers, prefetch_factor, device
):
    dataset_obj = dataset_lib.ImageFolderDataset(path=dataset_path)
    n = len(dataset_obj)
    if n < 2:
        raise RuntimeError(f"Need at least 2 real images for FID reference, found {n}.")
    feature_dim = 2048
    mu = torch.zeros([feature_dim], dtype=torch.float64, device=device)
    sigma = torch.zeros([feature_dim, feature_dim], dtype=torch.float64, device=device)
    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=num_workers,
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    loader = torch.utils.data.DataLoader(dataset_obj, **loader_kwargs)
    with torch.no_grad():
        for images, _labels in loader:
            images = images.to(device)
            if images.shape[1] == 1:
                images = images.repeat([1, 3, 1, 1])
            features = detector_net(images, return_features=True).to(torch.float64)
            mu += features.sum(0)
            sigma += features.T @ features
    mu /= n
    sigma -= mu.ger(mu) * n
    sigma /= max(n - 1, 1)
    return mu.cpu().numpy(), sigma.cpu().numpy()


def _run_quick_eval(
    *,
    ema,
    run_dir,
    dataset_path,
    cur_nimg,
    device,
    state,
    num_images=512,
    num_steps=18,
    batch_size=64,
    num_visual=32,
    ref_path=None,
):
    if num_images < 2:
        return
    eval_root = os.path.join(run_dir, "quick_eval")
    os.makedirs(eval_root, exist_ok=True)
    kimg = int(cur_nimg // 1000)
    tag = f"{kimg:06d}"
    interval_dir = os.path.join(eval_root, f"kimg-{tag}")
    os.makedirs(interval_dir, exist_ok=True)

    if state.get("detector_net") is None:
        state["detector_net"] = _load_inception_detector(device=device)
    detector_net = state["detector_net"]

    if state.get("mu_ref") is None or state.get("sigma_ref") is None:
        loaded = False
        if ref_path and os.path.isfile(ref_path):
            ref_data = dict(np.load(ref_path))
            state["mu_ref"] = ref_data["mu"]
            state["sigma_ref"] = ref_data["sigma"]
            loaded = True
            dist.print0(f"[QuickEval] Loaded reference stats from {ref_path}")
        if not loaded:
            num_workers = max(1, min(4, int(state.get("num_workers", 2))))
            prefetch_factor = int(state.get("prefetch_factor", 2))
            mu_ref, sigma_ref = _compute_reference_stats_from_dataset(
                detector_net=detector_net,
                dataset_path=dataset_path,
                batch_size=batch_size,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
                device=device,
            )
            state["mu_ref"] = mu_ref
            state["sigma_ref"] = sigma_ref
            np.savez(
                os.path.join(eval_root, "reference_stats.npz"), mu=mu_ref, sigma=sigma_ref
            )
            dist.print0(
                f'[QuickEval] Computed and cached reference stats at {os.path.join(eval_root, "reference_stats.npz")}'
            )

    gen = torch.Generator(device=device)
    gen.manual_seed(int(state.get("seed_base", 0)) + kimg)

    preview_images = []
    generated_uint8 = []
    ema.eval()
    with torch.no_grad():
        remaining = num_images
        while remaining > 0:
            b = min(batch_size, remaining)
            latents = torch.randn(
                [b, ema.img_channels, ema.img_resolution, ema.img_resolution],
                generator=gen,
                device=device,
            )
            class_labels = None
            if ema.label_dim:
                class_idx = torch.randint(ema.label_dim, size=[b], generator=gen, device=device)
                class_labels = torch.eye(ema.label_dim, device=device)[class_idx]
            images = edm_sampler(ema, latents, class_labels=class_labels, num_steps=num_steps)
            images = images.to(torch.float32).clamp(-1, 1)
            images_u8 = _images_to_uint8(images).cpu()
            generated_uint8.append(images_u8)
            if len(preview_images) * batch_size < num_visual:
                preview_images.append(images_u8)
            remaining -= b

    generated_uint8 = torch.cat(generated_uint8, dim=0)[:num_images]
    preview_uint8 = torch.cat(preview_images, dim=0)[:num_visual]

    _save_image_grid(preview_uint8, os.path.join(interval_dir, "preview_grid.png"), ncols=8)
    _save_individual_images(preview_uint8, os.path.join(interval_dir, "preview_images"), prefix="sample")

    mu_gen, sigma_gen = _compute_feature_stats_from_tensor(
        detector_net=detector_net,
        images_uint8=generated_uint8,
        batch_size=batch_size,
        device=device,
    )
    fid_val = calculate_fid_from_inception_stats(
        mu_gen, sigma_gen, state["mu_ref"], state["sigma_ref"]
    )
    metrics = dict(
        timestamp=time.time(),
        cur_nimg=int(cur_nimg),
        kimg=kimg,
        quick_fid=float(fid_val),
        num_images=int(num_images),
        num_steps=int(num_steps),
        batch_size=int(batch_size),
        num_visual=int(num_visual),
        out_dir=interval_dir,
    )
    metrics_path = os.path.join(eval_root, "quick_eval_metrics.jsonl")
    with open(metrics_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(metrics) + "\n")
    dist.print0(f"[QuickEval] kimg={kimg} quick_fid={fid_val:.4f} preview={interval_dir}")
