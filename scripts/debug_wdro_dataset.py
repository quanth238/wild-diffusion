#!/usr/bin/env python3
import argparse
import json
import os
import random
from typing import Any, Dict, List

import PIL.Image
import torch


def _safe_torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # Older PyTorch versions do not support weights_only.
        return torch.load(path, map_location="cpu")


def _float(v: torch.Tensor) -> float:
    return float(v.detach().cpu().item())


def _tensor_info(x: torch.Tensor) -> Dict[str, Any]:
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "requires_grad": bool(x.requires_grad),
        "device": str(x.device),
    }


def _basic_stats(x: torch.Tensor) -> Dict[str, Any]:
    x32 = x.to(torch.float32)
    nan_count = int(torch.isnan(x32).sum().item())
    inf_count = int(torch.isinf(x32).sum().item())
    finite_mask = torch.isfinite(x32)
    finite_count = int(finite_mask.sum().item())

    if finite_count == 0:
        return {
            "nan_count": nan_count,
            "inf_count": inf_count,
            "finite_count": finite_count,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
        }

    finite_vals = x32[finite_mask]
    return {
        "nan_count": nan_count,
        "inf_count": inf_count,
        "finite_count": finite_count,
        "min": _float(finite_vals.min()),
        "max": _float(finite_vals.max()),
        "mean": _float(finite_vals.mean()),
        "std": _float(finite_vals.std(unbiased=False)),
    }


def _infer_image_scale(x: torch.Tensor, x_stats: Dict[str, Any]) -> str:
    if x_stats["min"] is None or x_stats["max"] is None:
        return "unknown (no finite values)"
    x_min = x_stats["min"]
    x_max = x_stats["max"]
    if x_min >= -1.1 and x_max <= 1.1:
        return "likely normalized [-1, 1]"
    if x_min >= -0.01 and x_max <= 255.01:
        return "likely raw [0, 255] range"
    return "unknown / mixed range"


def _channel_stats(x: torch.Tensor) -> Dict[str, Any]:
    if x.ndim != 4:
        return {"available": False}
    x32 = x.to(torch.float32)
    # Reduce over N, H, W and keep C.
    per_ch_mean = x32.mean(dim=(0, 2, 3))
    per_ch_std = x32.std(dim=(0, 2, 3), unbiased=False)
    return {
        "available": True,
        "mean": [float(v) for v in per_ch_mean.cpu().tolist()],
        "std": [float(v) for v in per_ch_std.cpu().tolist()],
    }


def _analyze_labels(labels: torch.Tensor, max_classes_to_show: int = 20) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "tensor": _tensor_info(labels),
        "num_labels": int(labels.shape[0]) if labels.ndim >= 1 else 0,
    }

    if labels.ndim == 0:
        out["format"] = "scalar"
        return out

    if labels.ndim == 1:
        out["format"] = "class_index_1d"
        classes = labels.to(torch.int64)
    elif labels.ndim == 2 and labels.shape[1] > 1:
        out["format"] = "vector_2d"
        row_sums = labels.to(torch.float32).sum(dim=1)
        is_prob_like = bool(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3))
        values = labels.to(torch.float32)
        is_binary_like = bool(
            torch.all((values == 0) | (values == 1)).item()
        )
        out["is_onehot_like"] = bool(is_prob_like and is_binary_like)
        classes = labels.argmax(dim=1).to(torch.int64)
    else:
        out["format"] = f"other_ndim_{labels.ndim}"
        return out

    unique_classes, counts = torch.unique(classes, return_counts=True, sorted=True)
    top = list(zip(unique_classes.tolist(), counts.tolist()))
    top = top[:max_classes_to_show]
    out["num_unique_classes"] = int(unique_classes.numel())
    out["class_counts_head"] = [{"class": int(c), "count": int(n)} for c, n in top]
    out["min_class"] = int(unique_classes.min().item()) if unique_classes.numel() else None
    out["max_class"] = int(unique_classes.max().item()) if unique_classes.numel() else None
    return out


def _build_report(path: str, images: torch.Tensor, labels: torch.Tensor, sample_images: int) -> Dict[str, Any]:
    num_images = int(images.shape[0]) if images.ndim >= 1 else 0
    sample_n = min(num_images, sample_images)
    images_sample = images[:sample_n]
    labels_sample = labels[:sample_n] if labels.ndim >= 1 else labels

    image_info = _tensor_info(images)
    image_stats = _basic_stats(images_sample)
    channel_stats = _channel_stats(images_sample)
    label_info = _analyze_labels(labels)
    label_sample_stats = _basic_stats(labels_sample.to(torch.float32)) if labels.ndim >= 1 else {}

    report: Dict[str, Any] = {
        "input_path": path,
        "file_size_bytes": int(os.path.getsize(path)),
        "images": {
            "tensor": image_info,
            "num_images": num_images,
            "sample_n_for_stats": sample_n,
            "sample_stats": image_stats,
            "sample_channel_stats": channel_stats,
            "scale_inference": _infer_image_scale(images_sample, image_stats),
        },
        "labels": {
            "analysis": label_info,
            "sample_stats": label_sample_stats,
        },
        "consistency_checks": {
            "num_images_equals_num_labels": bool(num_images == (int(labels.shape[0]) if labels.ndim >= 1 else 0)),
        },
    }
    return report


def _report_to_text(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("WDRO Combined Dataset Debug Report")
    lines.append("=" * 36)
    lines.append(f"input_path: {report['input_path']}")
    lines.append(f"file_size_bytes: {report['file_size_bytes']}")
    lines.append("")

    img = report["images"]
    lines.append("[Images]")
    lines.append(f"shape: {img['tensor']['shape']}")
    lines.append(f"dtype: {img['tensor']['dtype']}")
    lines.append(f"num_images: {img['num_images']}")
    lines.append(f"sample_n_for_stats: {img['sample_n_for_stats']}")
    lines.append(f"scale_inference: {img['scale_inference']}")
    lines.append(f"sample_stats: {img['sample_stats']}")
    lines.append(f"sample_channel_stats: {img['sample_channel_stats']}")
    lines.append("")

    lab = report["labels"]
    lines.append("[Labels]")
    lines.append(f"analysis: {lab['analysis']}")
    lines.append(f"sample_stats: {lab['sample_stats']}")
    lines.append("")

    checks = report["consistency_checks"]
    lines.append("[Checks]")
    lines.append(f"num_images_equals_num_labels: {checks['num_images_equals_num_labels']}")
    return "\n".join(lines) + "\n"


def _tensor_to_uint8_chw(image: torch.Tensor) -> torch.Tensor:
    x = image.detach().cpu().to(torch.float32)
    x_min = float(x.min().item())
    x_max = float(x.max().item())

    # Preferred range in this repo for training tensors.
    if x_min >= -1.1 and x_max <= 1.1:
        x = (x + 1.0) * 127.5
    x = x.clamp(0, 255).round().to(torch.uint8)
    return x


def _save_png(chw: torch.Tensor, path: str) -> None:
    if chw.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got shape={tuple(chw.shape)}")
    if chw.shape[0] == 1:
        arr = chw[0].numpy()
        img = PIL.Image.fromarray(arr, mode="L")
    elif chw.shape[0] == 3:
        arr = chw.permute(1, 2, 0).numpy()
        img = PIL.Image.fromarray(arr, mode="RGB")
    else:
        raise ValueError(f"Expected 1 or 3 channels, got C={chw.shape[0]}")
    img.save(path)


def _export_random_samples(
    images: torch.Tensor,
    labels: torch.Tensor,
    out_dir: str,
    k: int,
    seed: int,
) -> Dict[str, Any]:
    num_images = int(images.shape[0])
    if k <= 0 or num_images == 0:
        return {"saved": 0, "out_dir": out_dir}

    os.makedirs(out_dir, exist_ok=True)
    rnd = random.Random(seed)
    indices = list(range(num_images))
    rnd.shuffle(indices)
    indices = indices[: min(k, num_images)]

    manifest: List[Dict[str, Any]] = []
    for rank_idx, idx in enumerate(indices):
        image_u8 = _tensor_to_uint8_chw(images[idx])
        label = labels[idx]
        png_name = f"sample_{rank_idx:03d}_idx{idx:08d}.png"
        png_path = os.path.join(out_dir, png_name)
        _save_png(image_u8, png_path)

        label_value: Any
        if label.ndim == 0:
            label_value = int(label.item())
        elif label.ndim == 1 and label.numel() > 1:
            label_value = int(label.argmax().item())
        else:
            label_value = label.detach().cpu().tolist()

        manifest.append(
            {
                "index": int(idx),
                "file": png_name,
                "label": label_value,
            }
        )

    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"samples": manifest}, f, indent=2)

    return {
        "saved": len(manifest),
        "out_dir": out_dir,
        "manifest": manifest_path,
    }


def _build_pairs_for_padv1(num_images: int, batch_size: int) -> List[tuple]:
    if batch_size <= 0:
        raise ValueError("--pair-batch-size must be > 0 for pair export")
    if num_images % 2 != 0:
        raise ValueError(
            "Combined dataset length must be even for p_adv=1 pair export. "
            f"Got N={num_images}."
        )

    raw_total = num_images // 2
    full_batches = raw_total // batch_size
    remainder = raw_total % batch_size
    pairs: List[tuple] = []

    cursor = 0
    for _ in range(full_batches):
        b = batch_size
        for k in range(b):
            pairs.append((cursor + k, cursor + b + k))
        cursor += 2 * b

    if remainder > 0:
        b = remainder
        for k in range(b):
            pairs.append((cursor + k, cursor + b + k))
        cursor += 2 * b

    if cursor != num_images:
        raise RuntimeError(f"Internal pair parsing mismatch: cursor={cursor}, N={num_images}")
    return pairs


def _save_triptych(adv: torch.Tensor, raw: torch.Tensor, path: str) -> None:
    adv_u8 = _tensor_to_uint8_chw(adv)
    raw_u8 = _tensor_to_uint8_chw(raw)
    if adv_u8.shape != raw_u8.shape:
        raise ValueError(f"Shape mismatch in pair image: adv={adv_u8.shape}, raw={raw_u8.shape}")

    # Visualize perturbation magnitude as grayscale map.
    delta = (adv.to(torch.float32) - raw.to(torch.float32)).abs()
    delta = delta.mean(dim=0, keepdim=True)
    max_delta = float(delta.max().item())
    if max_delta > 0:
        delta = delta / max_delta
    delta_u8 = (delta * 255.0).clamp(0, 255).round().to(torch.uint8).repeat(3, 1, 1)

    if adv_u8.shape[0] == 1:
        adv_u8 = adv_u8.repeat(3, 1, 1)
        raw_u8 = raw_u8.repeat(3, 1, 1)
    elif adv_u8.shape[0] != 3:
        raise ValueError(f"Unsupported channel count for triptych: {adv_u8.shape[0]}")

    panel = torch.cat([adv_u8, raw_u8, delta_u8], dim=2)  # CHW concat by width.
    _save_png(panel, path)


def _export_pair_samples_padv1(
    images: torch.Tensor,
    labels: torch.Tensor,
    out_dir: str,
    k: int,
    seed: int,
    pair_batch_size: int,
) -> Dict[str, Any]:
    num_images = int(images.shape[0])
    if k <= 0 or num_images == 0:
        return {"saved": 0, "out_dir": out_dir}

    pairs = _build_pairs_for_padv1(num_images=num_images, batch_size=pair_batch_size)
    if not pairs:
        return {"saved": 0, "out_dir": out_dir}

    os.makedirs(out_dir, exist_ok=True)
    rnd = random.Random(seed)
    rnd.shuffle(pairs)
    pairs = pairs[: min(k, len(pairs))]

    manifest: List[Dict[str, Any]] = []
    for rank_idx, (adv_idx, raw_idx) in enumerate(pairs):
        adv = images[adv_idx]
        raw = images[raw_idx]
        triptych_name = f"pair_{rank_idx:03d}_adv{adv_idx:08d}_raw{raw_idx:08d}.png"
        triptych_path = os.path.join(out_dir, triptych_name)
        _save_triptych(adv, raw, triptych_path)

        label = labels[raw_idx]
        if label.ndim == 0:
            label_value: Any = int(label.item())
        elif label.ndim == 1 and label.numel() > 1:
            label_value = int(label.argmax().item())
        else:
            label_value = label.detach().cpu().tolist()

        manifest.append(
            {
                "adv_index": int(adv_idx),
                "raw_index": int(raw_idx),
                "file": triptych_name,
                "label": label_value,
            }
        )

    manifest_path = os.path.join(out_dir, "pairs_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"pairs": manifest}, f, indent=2)

    return {
        "saved": len(manifest),
        "out_dir": out_dir,
        "manifest": manifest_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect combined_dataset-*.pt and write debug summary.")
    parser.add_argument("input", help="Path to combined_dataset-*.pt")
    parser.add_argument("--sample-images", type=int, default=4096, help="How many first images to use for stats.")
    parser.add_argument(
        "--out-json",
        default=None,
        help="Output JSON path. Default: <input>.debug.json",
    )
    parser.add_argument(
        "--out-txt",
        default=None,
        help="Output TXT path. Default: <input>.debug.txt",
    )
    parser.add_argument(
        "--save-random-k",
        type=int,
        default=0,
        help="Save K random images from combined dataset as PNG files.",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Directory for random PNG samples. Default: <input>.samples",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for sample selection.",
    )
    parser.add_argument(
        "--save-pairs-k",
        type=int,
        default=0,
        help=(
            "Save K random (adv, raw, diff) triptych images. "
            "Requires p_adv=1 combined dataset and --pair-batch-size."
        ),
    )
    parser.add_argument(
        "--pair-batch-size",
        type=int,
        default=0,
        help="Per-GPU batch size used when generating combined_dataset (needed for pair export).",
    )
    args = parser.parse_args()

    in_path = os.path.abspath(args.input)
    if not os.path.isfile(in_path):
        raise FileNotFoundError(f"Input file not found: {in_path}")

    data = _safe_torch_load(in_path)
    if not (isinstance(data, (tuple, list)) and len(data) == 2):
        raise ValueError("Expected tuple/list with 2 elements: (combined_images, combined_labels)")

    images, labels = data
    if not isinstance(images, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise TypeError("Expected both combined_images and combined_labels to be torch.Tensor")

    report = _build_report(in_path, images, labels, sample_images=max(int(args.sample_images), 1))

    out_json = os.path.abspath(args.out_json or (in_path + ".debug.json"))
    out_txt = os.path.abspath(args.out_txt or (in_path + ".debug.txt"))

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    os.makedirs(os.path.dirname(out_txt), exist_ok=True)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(_report_to_text(report))

    print(f"[OK] Wrote JSON report: {out_json}")
    print(f"[OK] Wrote TXT report:  {out_txt}")

    if args.save_random_k > 0:
        out_samples = os.path.abspath(args.save_dir or (in_path + ".samples"))
        saved = _export_random_samples(
            images=images,
            labels=labels,
            out_dir=out_samples,
            k=int(args.save_random_k),
            seed=int(args.seed),
        )
        print(f"[OK] Saved {saved['saved']} PNG samples to: {saved['out_dir']}")
        if "manifest" in saved:
            print(f"[OK] Sample manifest: {saved['manifest']}")

    if args.save_pairs_k > 0:
        if args.pair_batch_size <= 0:
            raise ValueError("--pair-batch-size must be > 0 when --save-pairs-k is used")
        out_pairs = os.path.abspath(args.save_dir or (in_path + ".samples"))
        saved = _export_pair_samples_padv1(
            images=images,
            labels=labels,
            out_dir=out_pairs,
            k=int(args.save_pairs_k),
            seed=int(args.seed),
            pair_batch_size=int(args.pair_batch_size),
        )
        print(f"[OK] Saved {saved['saved']} pair triptychs to: {saved['out_dir']}")
        if "manifest" in saved:
            print(f"[OK] Pair manifest: {saved['manifest']}")


if __name__ == "__main__":
    main()
