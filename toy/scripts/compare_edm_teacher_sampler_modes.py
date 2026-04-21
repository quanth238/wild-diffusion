#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import torch


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


from toy.app.experiment import _checkpoint_state_dict_for_rf_init  # noqa: E402
from toy.config import ToyConfig  # noqa: E402
from toy.data_backends.provider import build_dataset_bundle  # noqa: E402
from toy.metrics import evaluate_nearest_reference_distance  # noqa: E402
from toy.model_backends.provider import build_model_bundle  # noqa: E402
from toy.scripts.reevaluate_three_method_fids_from_checkpoints import (  # noqa: E402
    _calculate_fid_from_stats_torch,
    _load_detector,
    _load_ref_stats,
)
from toy.shared.reverse import resolve_ve_sampler_mode, reverse_paths_from_terminal  # noqa: E402
from toy.shared.runtime import autocast_context, configure_runtime, resolve_amp_dtype  # noqa: E402
from toy.shared.sigma import build_sigma_levels, resolve_rf_teacher_n_steps_path  # noqa: E402
from toy.utils import pick_device, set_seed, tensor_to_numpy  # noqa: E402


DEFAULT_MODES = ["edm_heun", "ancestral_stochastic", "ancestral_mean_only"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare EDM teacher sampling modes on one checkpoint using shared terminal noise so "
            "EDM Heun, ancestral stochastic, and posterior-mean-only can be measured side by side."
        )
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--metrics-json", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp-dtype", type=str, default="off")
    parser.add_argument("--num-images", type=int, default=2000)
    parser.add_argument("--gen-batch", type=int, default=250)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--n-steps-path", type=int, default=0)
    parser.add_argument("--sigma-min", type=float, default=-1.0)
    parser.add_argument("--sigma-max", type=float, default=-1.0)
    parser.add_argument("--modes", type=str, default=",".join(DEFAULT_MODES))
    parser.add_argument("--skip-fid", action="store_true")
    parser.add_argument("--json-out", type=str, default="")
    return parser.parse_args()


def _resolve_repo_path(path: str) -> str:
    text = str(path or "").strip()
    if not text:
        return ""
    if os.path.isabs(text):
        return text
    return os.path.abspath(os.path.join(ROOT_DIR, text))


def _load_cfg(metrics_json: str) -> ToyConfig:
    with open(metrics_json, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    cfg = ToyConfig()
    source_cfg = payload.get("config", {})
    if isinstance(source_cfg, dict):
        for key, value in source_cfg.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
    cfg.dataset_path = _resolve_repo_path(str(getattr(cfg, "dataset_path", "")))
    cfg.dataset_val_path = _resolve_repo_path(str(getattr(cfg, "dataset_val_path", "")))
    cfg.fid_ref_path = _resolve_repo_path(str(getattr(cfg, "fid_ref_path", "")))
    cfg.training_objective = "edm"
    cfg.compute_fid = False
    cfg.run_checks = False
    cfg.baseline_gate_enabled = False
    return cfg


@torch.no_grad()
def _fid_from_images(
    *,
    images: torch.Tensor,
    detector_net,
    mu_ref: torch.Tensor,
    sigma_ref: torch.Tensor,
    gen_batch: int,
    device: torch.device,
) -> float:
    feature_dim = 2048
    mu = torch.zeros([feature_dim], dtype=torch.float64, device=device)
    sigma = torch.zeros([feature_dim, feature_dim], dtype=torch.float64, device=device)
    n_done = 0
    for start in range(0, images.shape[0], int(gen_batch)):
        batch = images[start : start + int(gen_batch)]
        batch_u8 = ((batch + 1.0) * 127.5).clamp(0.0, 255.0).to(torch.uint8)
        if batch_u8.shape[1] == 1:
            batch_u8 = batch_u8.repeat([1, 3, 1, 1])
        features = detector_net(batch_u8, return_features=True).to(torch.float64)
        mu += features.sum(0)
        sigma += features.T @ features
        n_done += features.shape[0]
    mu /= n_done
    sigma -= mu.ger(mu) * n_done
    sigma /= max(n_done - 1, 1)
    return _calculate_fid_from_stats_torch(mu, sigma, mu_ref, sigma_ref)


@torch.no_grad()
def _sample_x0_for_mode(
    *,
    denoiser,
    sigma_levels: torch.Tensor,
    shared_terminal: torch.Tensor,
    shared_reverse_noise: torch.Tensor,
    sampler_mode: str,
    amp_dtype,
    device: torch.device,
    gen_batch: int,
) -> torch.Tensor:
    outputs: List[torch.Tensor] = []
    canonical_mode = resolve_ve_sampler_mode(stochastic=True, sampler_mode=sampler_mode)
    for start in range(0, shared_terminal.shape[0], int(gen_batch)):
        end = min(start + int(gen_batch), shared_terminal.shape[0])
        x_terminal = shared_terminal[start:end]
        noise_schedule = (
            shared_reverse_noise[:, start:end]
            if canonical_mode == "ancestral_stochastic"
            else None
        )
        with autocast_context(device, amp_dtype):
            paths = reverse_paths_from_terminal(
                denoiser=denoiser,
                x_terminal=x_terminal,
                sigma_levels=sigma_levels,
                stochastic=canonical_mode == "ancestral_stochastic",
                sampler_mode=canonical_mode,
                noise_schedule=noise_schedule,
            )
        outputs.append(paths[:, 0].detach())
    return torch.cat(outputs, dim=0)


def main() -> None:
    args = parse_args()
    checkpoint_path = _resolve_repo_path(args.checkpoint)
    metrics_json = _resolve_repo_path(args.metrics_json)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not os.path.isfile(metrics_json):
        raise FileNotFoundError(f"Metrics JSON not found: {metrics_json}")

    cfg = _load_cfg(metrics_json)
    if float(args.sigma_min) > 0.0:
        cfg.sigma_min = float(args.sigma_min)
    if float(args.sigma_max) > 0.0:
        cfg.sigma_max = float(args.sigma_max)

    device = pick_device(str(args.device))
    configure_runtime(
        device=device,
        allow_tf32=bool(getattr(cfg, "allow_tf32", True)),
        cudnn_benchmark=bool(getattr(cfg, "cudnn_benchmark", True)),
    )
    amp_dtype = resolve_amp_dtype(device, str(args.amp_dtype))
    set_seed(int(args.seed))

    dataset = build_dataset_bundle(cfg, device)
    if float(getattr(cfg, "sigma_data", -1.0)) <= 0.0:
        cfg.sigma_data = float(dataset.estimate_sigma_data())
    model_bundle = build_model_bundle(cfg, dataset, float(cfg.sigma_data), device)
    model = model_bundle.baseline

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict, source_key = _checkpoint_state_dict_for_rf_init(payload)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    n_steps_path = (
        int(args.n_steps_path)
        if int(args.n_steps_path) > 0
        else int(resolve_rf_teacher_n_steps_path(cfg))
    )
    sigma_levels = build_sigma_levels(
        float(cfg.sigma_min),
        float(cfg.sigma_max),
        n_steps_path,
        device=device,
    )

    num_images = max(int(args.num_images), 1)
    gen_batch = max(int(args.gen_batch), 1)
    shared_terminal = dataset.sample_terminal_batch(num_images, float(sigma_levels[-1].item())).to(
        device=device,
        dtype=sigma_levels.dtype,
    )
    shared_reverse_noise = torch.randn(
        (sigma_levels.numel(), num_images, *dataset.data_shape),
        device=device,
        dtype=sigma_levels.dtype,
    )
    val_np = tensor_to_numpy(dataset.val_pool)

    detector_net = None
    mu_ref = None
    sigma_ref = None
    if not bool(args.skip_fid):
        detector_net = _load_detector(device)
        mu_ref, sigma_ref = _load_ref_stats(cfg.fid_ref_path)
        mu_ref = mu_ref.to(device=device)
        sigma_ref = sigma_ref.to(device=device)

    modes = [resolve_ve_sampler_mode(stochastic=True, sampler_mode=text.strip()) for text in args.modes.split(",") if text.strip()]
    if not modes:
        raise ValueError("At least one sampler mode must be requested.")

    results: Dict[str, Dict[str, object]] = {}
    for mode in modes:
        images = _sample_x0_for_mode(
            denoiser=model,
            sigma_levels=sigma_levels,
            shared_terminal=shared_terminal,
            shared_reverse_noise=shared_reverse_noise,
            sampler_mode=mode,
            amp_dtype=amp_dtype,
            device=device,
            gen_batch=gen_batch,
        )
        images_np = tensor_to_numpy(images)
        mode_metrics: Dict[str, object] = {
            "mode": mode,
            "global_metrics": dataset.evaluate_sample_metrics(images_np),
            "nearest_reference_distance": evaluate_nearest_reference_distance(images_np, val_np, chunk_size=128),
        }
        if detector_net is not None and mu_ref is not None and sigma_ref is not None:
            mode_metrics["fid"] = float(
                _fid_from_images(
                    images=images,
                    detector_net=detector_net,
                    mu_ref=mu_ref,
                    sigma_ref=sigma_ref,
                    gen_batch=gen_batch,
                    device=device,
                )
            )
        results[mode] = mode_metrics

    report = {
        "checkpoint": checkpoint_path,
        "metrics_json": metrics_json,
        "checkpoint_source_key": source_key,
        "device": str(device),
        "amp_dtype": str(args.amp_dtype),
        "seed": int(args.seed),
        "num_images": int(num_images),
        "gen_batch": int(gen_batch),
        "sigma_min": float(cfg.sigma_min),
        "sigma_max": float(cfg.sigma_max),
        "n_steps_path": int(n_steps_path),
        "results": results,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if str(args.json_out).strip():
        out_path = _resolve_repo_path(args.json_out)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
