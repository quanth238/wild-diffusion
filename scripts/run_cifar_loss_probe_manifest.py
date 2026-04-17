#!/usr/bin/env python3
import argparse
import csv
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import dnnlib  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a posthoc clean one-step EDM loss probe for rows in a CIFAR manifest CSV. "
            "The probe is evaluated from saved checkpoints, so it does not affect training wall-clock."
        )
    )
    parser.add_argument("--manifest-csv", type=str, required=True)
    parser.add_argument("--methods", type=str, default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--only-pending", action="store_true", default=False)
    parser.add_argument("--force", action="store_true", default=False)
    return parser.parse_args()


def load_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_manifest(path: Path, rows: List[Dict[str, str]]) -> None:
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _safe_float(value: object):
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def should_run_row(row: Dict[str, str], *, allowed_methods: set, only_pending: bool, force: bool) -> bool:
    robust_method = str(row.get("robust_method", row.get("method", ""))).strip()
    if allowed_methods and robust_method not in allowed_methods:
        return False
    if force:
        return True
    result_json = Path(str(row.get("loss_probe_result_json", "")).strip())
    if result_json.is_file():
        return True
    if only_pending and _truthy(row.get("loss_probe_evaluated", "")):
        return False
    return not _truthy(row.get("loss_probe_evaluated", ""))


def resolve_eval_paths(row: Dict[str, str]) -> Tuple[Path, Path]:
    if str(row.get("loss_probe_result_json", "")).strip():
        result_json = Path(str(row["loss_probe_result_json"])).resolve()
        return result_json.parent, result_json
    if str(row.get("eval_dir", "")).strip():
        eval_dir = Path(str(row["eval_dir"])).resolve()
    elif str(row.get("eval_root", "")).strip() and str(row.get("eval_tag", "")).strip():
        eval_dir = Path(str(row["eval_root"])).resolve() / str(row["eval_tag"]).strip()
    else:
        run_dir = Path(str(row["run_dir"])).resolve()
        eval_dir = run_dir / "posthoc_loss_probe" / str(row.get("eval_tag", f"step-{row.get('step', 'unknown')}"))
    return eval_dir, eval_dir / "loss_probe_result.json"


def _dataset_cache_key(dataset_kwargs: Dict, max_images: int, seed: int) -> str:
    return json.dumps(
        {
            "dataset_kwargs": dataset_kwargs,
            "max_images": int(max_images),
            "seed": int(seed),
        },
        sort_keys=True,
    )


def _make_dataset(dataset_kwargs: Dict, *, max_images: int, seed: int):
    dataset = dnnlib.util.construct_class_by_name(**dataset_kwargs)
    if int(max_images) <= 0 or len(dataset) <= int(max_images):
        return dataset
    rng = np.random.RandomState(int(seed) % (1 << 31))
    indices = np.arange(len(dataset), dtype=np.int64)
    rng.shuffle(indices)
    keep = np.sort(indices[: int(max_images)]).tolist()
    return torch.utils.data.Subset(dataset, keep)


def _clean_edm_probe_batch(
    *,
    net,
    images: torch.Tensor,
    labels: torch.Tensor | None,
    p_mean: float,
    p_std: float,
    sigma_data: float,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    batch_size = int(images.shape[0])
    rnd_normal = torch.randn((batch_size, 1, 1, 1), generator=generator, dtype=torch.float32)
    sigma = (rnd_normal * float(p_std) + float(p_mean)).exp().to(device)
    noise = torch.randn(tuple(images.shape), generator=generator, dtype=torch.float32).to(device)
    weight = (sigma.square() + float(sigma_data) ** 2) / (sigma * float(sigma_data)).square().clamp_min(1e-8)
    pred = net(images + noise * sigma, sigma, labels)
    return weight * (pred - images).square()


def evaluate_row(
    row: Dict[str, str],
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    max_images: int,
    seed: int,
    dataset_cache: Dict[str, object],
) -> Dict[str, object]:
    network_pkl = str(row.get("network_pkl", "")).strip()
    if not network_pkl:
        raise RuntimeError(f"Row is missing network_pkl: {row}")

    with dnnlib.util.open_url(network_pkl, verbose=False) as handle:
        payload = pickle.load(handle)

    net = payload["ema"].eval().requires_grad_(False).to(device)
    loss_fn = payload.get("loss_fn", None)
    dataset_kwargs = dict(payload.get("dataset_kwargs", {}))
    if not dataset_kwargs:
        raise RuntimeError(f"Snapshot is missing dataset_kwargs: {network_pkl}")

    p_mean = float(getattr(loss_fn, "P_mean", -1.2))
    p_std = float(getattr(loss_fn, "P_std", 1.2))
    sigma_data = float(getattr(loss_fn, "sigma_data", getattr(net, "sigma_data", 0.5)))

    cache_key = _dataset_cache_key(dataset_kwargs, max_images=max_images, seed=seed)
    if cache_key not in dataset_cache:
        dataset_cache[cache_key] = _make_dataset(dataset_kwargs, max_images=max_images, seed=seed)
    dataset = dataset_cache[cache_key]

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=(device.type == "cuda"),
    )

    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(seed))

    total_sum = 0.0
    total_numel = 0
    total_images = 0
    total_batches = 0
    t0 = time.time()
    with torch.no_grad():
        for images, labels in loader:
            if images.dtype == torch.uint8:
                images = images.to(device, non_blocking=True).to(torch.float32) / 127.5 - 1.0
            else:
                images = images.to(device, non_blocking=True).to(torch.float32)
            if labels is not None:
                labels = labels.to(device, non_blocking=True)
            loss = _clean_edm_probe_batch(
                net=net,
                images=images,
                labels=labels,
                p_mean=p_mean,
                p_std=p_std,
                sigma_data=sigma_data,
                generator=rng,
                device=device,
            )
            total_sum += float(loss.sum().item())
            total_numel += int(loss.numel())
            total_images += int(images.shape[0])
            total_batches += 1

    probe_value = total_sum / max(total_numel, 1)
    return {
        "metric_name": "checkpoint_clean_edm_probe",
        "loss_probe_clean": float(probe_value),
        "loss_reduction": "mean_per_pixel",
        "network_pkl": str(Path(network_pkl).resolve()),
        "run_dir": str(Path(str(row.get("run_dir", Path(network_pkl).parent))).resolve()),
        "dataset_path": str(dataset_kwargs.get("path", "")),
        "dataset_num_images": int(total_images),
        "dataset_num_batches": int(total_batches),
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "max_images": int(max_images),
        "seed": int(seed),
        "p_mean": float(p_mean),
        "p_std": float(p_std),
        "sigma_data": float(sigma_data),
        "elapsed_sec": float(time.time() - t0),
        "created_at_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }


def write_result_files(eval_dir: Path, result: Dict[str, object]) -> Tuple[Path, Path]:
    eval_dir.mkdir(parents=True, exist_ok=True)
    result_json = eval_dir / "loss_probe_result.json"
    result_txt = eval_dir / "loss_probe_result.txt"
    result_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [
        f"metric_name={result['metric_name']}",
        f"loss_probe_clean={result['loss_probe_clean']:.8f}",
        f"dataset_num_images={result['dataset_num_images']}",
        f"dataset_num_batches={result['dataset_num_batches']}",
        f"batch_size={result['batch_size']}",
        f"seed={result['seed']}",
        f"elapsed_sec={result['elapsed_sec']:.3f}",
        f"network_pkl={result['network_pkl']}",
        f"dataset_path={result['dataset_path']}",
    ]
    result_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result_json, result_txt


def update_row_from_result(row: Dict[str, str], result_json: Path) -> None:
    payload = json.loads(result_json.read_text(encoding="utf-8"))
    probe_value = float(payload["loss_probe_clean"])
    row["loss_probe_clean"] = str(probe_value)
    row["loss_probe_metric_key"] = str(payload.get("metric_name", "checkpoint_clean_edm_probe"))
    row["loss_probe_evaluated"] = "True"
    row["loss_probe_source"] = "posthoc_checkpoint_clean_edm_probe"
    row["loss_probe_status"] = "done"
    row["loss_probe_result_json"] = str(result_json)
    row["loss_probe_result_txt"] = str(result_json.with_name("loss_probe_result.txt"))
    row["loss_probe_eval_dir"] = str(result_json.parent)
    row["loss_probe_dataset_images"] = str(int(payload.get("dataset_num_images", 0)))
    row["loss_probe_dataset_batches"] = str(int(payload.get("dataset_num_batches", 0)))
    row["loss_probe_batch_size"] = str(int(payload.get("batch_size", 0)))
    row["loss_probe_seed"] = str(int(payload.get("seed", 0)))
    row["loss_comparable_final"] = str(probe_value)
    row["loss_comparable_key"] = str(payload.get("metric_name", "checkpoint_clean_edm_probe"))


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest_csv).resolve()
    rows = load_manifest(manifest_path)
    if not rows:
        raise SystemExit(f"[ERROR] Manifest is empty: {manifest_path}")

    allowed_methods = {item.strip() for item in str(args.methods).split(",") if item.strip()}
    selected_rows = [
        row
        for row in rows
        if should_run_row(
            row,
            allowed_methods=allowed_methods,
            only_pending=bool(args.only_pending),
            force=bool(args.force),
        )
    ]
    if args.limit > 0:
        selected_rows = selected_rows[: int(args.limit)]
    if not selected_rows:
        print(f"[OK] Nothing to do for manifest: {manifest_path}")
        return

    device = torch.device(str(args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device for loss probe, but torch.cuda.is_available() is False.")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    dataset_cache: Dict[str, object] = {}
    print(f"[INFO] Running {len(selected_rows)} loss probe eval(s) from {manifest_path}")
    for row in selected_rows:
        eval_dir, result_json = resolve_eval_paths(row)
        row["loss_probe_result_json"] = str(result_json)
        row["loss_probe_result_txt"] = str(result_json.with_name("loss_probe_result.txt"))
        row["loss_probe_eval_dir"] = str(eval_dir)
        if result_json.is_file() and not args.force:
            update_row_from_result(row, result_json)
            write_manifest(manifest_path, rows)
            print(
                f"[SKIP] Reused loss probe for {row.get('robust_method', row.get('method', ''))} "
                f"kimg={row.get('snapshot_kimg', row.get('step', ''))} "
                f"loss={row['loss_probe_clean']}"
            )
            continue

        row["loss_probe_status"] = "running"
        write_manifest(manifest_path, rows)
        print(
            f"[RUN ] {row.get('robust_method', row.get('method', ''))} "
            f"kimg={row.get('snapshot_kimg', row.get('step', ''))}",
            flush=True,
        )
        result = evaluate_row(
            row,
            device=device,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            max_images=int(args.max_images),
            seed=int(args.seed),
            dataset_cache=dataset_cache,
        )
        result_json, _ = write_result_files(eval_dir, result)
        update_row_from_result(row, result_json)
        write_manifest(manifest_path, rows)
        print(
            f"[DONE] {row.get('robust_method', row.get('method', ''))} "
            f"kimg={row.get('snapshot_kimg', row.get('step', ''))} "
            f"loss={row['loss_probe_clean']}",
            flush=True,
        )

    print(f"[OK] Manifest updated in place: {manifest_path}")


if __name__ == "__main__":
    main()
