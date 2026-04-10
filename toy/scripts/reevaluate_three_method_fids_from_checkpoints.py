#!/usr/bin/env python3
import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
from typing import Dict, List, Optional, TextIO, Tuple

import numpy as np
import torch


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


from toy.config import ToyConfig  # noqa: E402
from toy.data_backends.provider import build_dataset_bundle  # noqa: E402
from toy.model_backends.provider import build_model_bundle  # noqa: E402
from toy.process_title import apply_process_title, build_process_title, child_process_env  # noqa: E402
from toy.shared.reverse import sample_reverse_paths  # noqa: E402
from toy.shared.runtime import autocast_context, configure_runtime, format_amp_dtype, resolve_amp_dtype  # noqa: E402
from toy.shared.sigma import build_sigma_levels  # noqa: E402
from toy.utils import ensure_dir, pick_device, set_seed  # noqa: E402


_APPLIED_PROCESS_TITLE = apply_process_title()

DEFAULT_TRAIN_ROOT = os.path.join(ROOT_DIR, "toy_data", "simpsons_mnist_rgb", "imagefolder", "train")
DEFAULT_VAL_ROOT = os.path.join(ROOT_DIR, "toy_data", "simpsons_mnist_rgb", "imagefolder", "test")
DEFAULT_FID_REF = os.path.join(
    ROOT_DIR,
    "toy_data",
    "simpsons_mnist_rgb",
    "fid_refs",
    "simpsons_mnist_rgb_test_28x28.npz",
)
DEFAULT_CALIBRATION = os.path.join(
    ROOT_DIR,
    "toy_outputs",
    "compute_calibration",
    "simpsons_mnist_rgb_image_conv_edm_b256_h64_cuda.json",
)
DEFAULT_FID_BATCH_SIZE = 512


@dataclass
class EvalContext:
    key: Tuple[object, ...]
    cfg: ToyConfig
    device: torch.device
    amp_dtype: Optional[torch.dtype]
    dataset: object
    sigma_levels: torch.Tensor
    baseline_model: torch.nn.Module
    robust_model: torch.nn.Module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-evaluate existing three-method trajectory checkpoints with in-memory batched FID "
            "directly from saved checkpoints, then regenerate per-case plots from the refreshed combined CSV."
        )
    )
    parser.add_argument("--combined-csv", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--prefix", type=str, required=True)
    parser.add_argument("--python-bin", type=str, default=os.path.join(ROOT_DIR, ".venv", "bin", "python"))
    parser.add_argument(
        "--plot-script",
        type=str,
        default=os.path.join(ROOT_DIR, "toy", "scripts", "plot_three_method_fid_curves.py"),
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp-dtype", type=str, default="auto")
    parser.add_argument("--dataset-path", type=str, default=DEFAULT_TRAIN_ROOT)
    parser.add_argument("--dataset-val-path", type=str, default=DEFAULT_VAL_ROOT)
    parser.add_argument("--fid-ref-path", type=str, default=DEFAULT_FID_REF)
    parser.add_argument("--weighted-compute-calibration-path", type=str, default=DEFAULT_CALIBRATION)
    parser.add_argument("--weighted-inputgrad-alpha", type=float, default=0.0)
    parser.add_argument("--weighted-parambackward-beta", type=float, default=0.0)
    parser.add_argument("--image-size", type=int, default=28)
    parser.add_argument("--image-channels", type=int, default=3)
    parser.add_argument("--image-train-size", type=int, default=80)
    parser.add_argument("--image-val-size", type=int, default=2000)
    parser.add_argument("--image-split-seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--eval-samples", type=int, default=2000)
    parser.add_argument("--fid-samples", type=int, default=50000)
    parser.add_argument("--fid-batch-size", type=int, default=DEFAULT_FID_BATCH_SIZE)
    parser.add_argument("--debug-eval-batch", type=int, default=64)
    parser.add_argument("--debug-terminal-step", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--n-steps-path-default", type=int, default=24)
    parser.add_argument("--sigma-min", type=float, default=0.002)
    parser.add_argument("--sigma-max", type=float, default=2.0)
    parser.add_argument("--metrics-eval-seed-offset", type=int, default=ToyConfig.eval_seed_offset_metrics)
    parser.add_argument("--train-percent-label", type=str, default="1%")
    parser.add_argument("--methods", type=str, default="")
    parser.add_argument("--only-missing-fid", action="store_true")
    parser.add_argument("--respect-fid-selection", action="store_true")
    parser.add_argument("--skip-plot", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def load_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str, rows: List[Dict[str, str]]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _safe_float(value, default: Optional[float] = None) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(parsed):
        return default
    return float(parsed)


def _safe_bool(value, *, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off", ""}:
        return False
    return bool(default)


def _parse_method_filter(text: str) -> Optional[set]:
    methods = {_canonical_robust_method(token) for token in str(text).split(",") if token.strip()}
    return None if not methods else methods


def _load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_repo_path(path: str) -> str:
    text = str(path or "").strip()
    if not text:
        return ""
    if os.path.isabs(text):
        return text
    return os.path.join(ROOT_DIR, text)


def _row_seed(row: Dict[str, str]) -> int:
    return _safe_int(row.get("seed"), 0)


def _row_step(row: Dict[str, str]) -> int:
    return _safe_int(row.get("step"), 0)


def _canonical_robust_method(method_name: str) -> str:
    method = str(method_name or "").strip().lower()
    if method in {"baseline", "baseline_edm", "baseline_rf", "baseline_score"}:
        return "baseline"
    if method in {"wdro", "wild", "wild_diffusion"}:
        return "wild_diffusion"
    if method == "cdro":
        return "cdro"
    return method or "unknown"


def _is_baseline_style_row(row: Dict[str, str]) -> bool:
    robust_method = _canonical_robust_method(row.get("robust_method", row.get("method", "")))
    row_origin = str(row.get("row_origin", "")).strip()
    return robust_method == "baseline" or row_origin == "trajectory_warmup_phase"


def _checkpoint_branch(row: Dict[str, str]) -> str:
    return "baseline" if _is_baseline_style_row(row) else "robust"


def _metrics_fid_key(row: Dict[str, str]) -> str:
    return "baseline_fid" if _is_baseline_style_row(row) else "robust_fid"


def _exp_name_from_row(prefix: str, row: Dict[str, str], occurrence_index: int) -> str:
    method = _canonical_robust_method(row.get("robust_method", row.get("method", "")))
    row_origin = str(row.get("row_origin", "")).strip() or "row"
    seed = _row_seed(row)
    step = _row_step(row)
    return f"{prefix}_{method}_{row_origin}_st{step}_s{seed}_k{occurrence_index:03d}"


def _reeval_cache_key(row: Dict[str, str]) -> Tuple[str, str]:
    checkpoint_path = _resolve_repo_path(row.get("checkpoint_path", ""))
    if not checkpoint_path:
        raise RuntimeError(f"Missing checkpoint_path in row: method={row.get('method')} step={row.get('step')}")
    return (_checkpoint_branch(row), checkpoint_path)


def _find_detector_path() -> str:
    env_path = os.environ.get("FID_DETECTOR_PATH", "").strip()
    if env_path and os.path.isfile(env_path):
        return env_path
    cache_root = Path.home() / ".cache" / "dnnlib" / "downloads"
    matches = sorted(cache_root.rglob("*inception-2015-12-05.pkl"))
    if matches:
        return str(matches[0])
    return ""


def _load_detector(device: torch.device):
    import dnnlib

    detector_path = _find_detector_path()
    detector_url = (
        "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/"
        "versions/1/files/metrics/inception-2015-12-05.pkl"
    )
    if detector_path:
        with open(detector_path, "rb") as handle:
            detector_net = pickle.load(handle).to(device)
    else:
        with dnnlib.util.open_url(detector_url, verbose=True) as handle:
            detector_net = pickle.load(handle).to(device)
    detector_net.eval()
    return detector_net


def _load_ref_stats(path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    ref = np.load(path)
    mu = torch.from_numpy(ref["mu"]).to(dtype=torch.float64)
    sigma = torch.from_numpy(ref["sigma"]).to(dtype=torch.float64)
    return mu, sigma


def _symmetrize_cov_torch(matrix: torch.Tensor) -> torch.Tensor:
    return 0.5 * (matrix + matrix.transpose(-1, -2))


def _trace_sqrt_product_torch(sigma: torch.Tensor, sigma_ref: torch.Tensor) -> torch.Tensor:
    sigma = _symmetrize_cov_torch(sigma)
    sigma_ref = _symmetrize_cov_torch(sigma_ref)
    evals, evecs = torch.linalg.eigh(sigma)
    evals = evals.clamp_min(0.0)
    sqrt_sigma = (evecs * evals.sqrt().unsqueeze(0)) @ evecs.transpose(-1, -2)
    middle = _symmetrize_cov_torch(sqrt_sigma @ sigma_ref @ sqrt_sigma)
    middle_evals = torch.linalg.eigvalsh(middle).clamp_min(0.0)
    return middle_evals.sqrt().sum()


def _calculate_fid_from_stats_torch(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    mu_ref: torch.Tensor,
    sigma_ref: torch.Tensor,
) -> float:
    diff = mu - mu_ref
    trace_sqrt = _trace_sqrt_product_torch(sigma, sigma_ref)
    fid = diff.dot(diff) + torch.trace(sigma) + torch.trace(sigma_ref) - (2.0 * trace_sqrt)
    return float(torch.real(fid).item())


@torch.no_grad()
def _sample_reverse_x0(
    *,
    denoiser: torch.nn.Module,
    sigma_levels: torch.Tensor,
    n_samples: int,
    device: torch.device,
    amp_dtype: Optional[torch.dtype],
    sample_terminal_batch_fn,
) -> torch.Tensor:
    with autocast_context(device, amp_dtype):
        states = sample_reverse_paths(
            denoiser=denoiser,
            sigma_levels=sigma_levels,
            n_samples=n_samples,
            device=device,
            stochastic=True,
            sample_terminal_batch_fn=sample_terminal_batch_fn,
        )
    return states[:, 0]


@torch.no_grad()
def _compute_fid_for_model(
    *,
    denoiser: torch.nn.Module,
    sigma_levels: torch.Tensor,
    dataset,
    detector_net,
    mu_ref: torch.Tensor,
    sigma_ref: torch.Tensor,
    num_images: int,
    gen_batch: int,
    device: torch.device,
    amp_dtype: Optional[torch.dtype],
    log_handle: Optional[TextIO] = None,
) -> float:
    feature_dim = 2048
    mu = torch.zeros([feature_dim], dtype=torch.float64, device=device)
    sigma = torch.zeros([feature_dim, feature_dim], dtype=torch.float64, device=device)
    n_done = 0
    t_start = time.perf_counter()

    denoiser.eval()
    while n_done < num_images:
        cur = min(gen_batch, num_images - n_done)
        images = _sample_reverse_x0(
            denoiser=denoiser,
            sigma_levels=sigma_levels,
            n_samples=cur,
            device=device,
            amp_dtype=amp_dtype,
            sample_terminal_batch_fn=dataset.sample_terminal_batch,
        )
        images = ((images + 1.0) * 127.5).clamp(0.0, 255.0).to(torch.uint8)
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])
        features = detector_net(images, return_features=True).to(torch.float64)
        mu += features.sum(0)
        sigma += features.T @ features
        n_done += cur
        if log_handle is not None:
            elapsed = time.perf_counter() - t_start
            log_handle.write(
                f"[fid-only] progress images={n_done}/{num_images} batch={cur} elapsed_sec={elapsed:.2f}\n"
            )
            log_handle.flush()

    mu /= num_images
    sigma -= mu.ger(mu) * num_images
    sigma /= max(num_images - 1, 1)
    return _calculate_fid_from_stats_torch(mu, sigma, mu_ref, sigma_ref)


def _load_checkpoint_payload(ckpt_path: str) -> Dict:
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid checkpoint format (expect dict): {ckpt_path}")
    return payload


def _load_baseline_state_dict(ckpt_path: str) -> Dict[str, torch.Tensor]:
    payload = _load_checkpoint_payload(ckpt_path)
    if "baseline_state_dict" in payload:
        state_dict = payload["baseline_state_dict"]
    elif "state_dict" in payload:
        state_dict = payload["state_dict"]
    else:
        raise RuntimeError(
            "Missing keys 'baseline_state_dict' and 'state_dict' in baseline checkpoint: "
            f"{ckpt_path}"
        )
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Invalid baseline state_dict in checkpoint: {ckpt_path}")
    return state_dict


def _load_robust_state_dict(
    ckpt_path: str,
    method_name: str,
    *,
    prefer_ema: bool = False,
) -> Tuple[Dict[str, torch.Tensor], str]:
    payload = _load_checkpoint_payload(ckpt_path)
    saved_method = _canonical_robust_method(payload.get("method_name", ""))
    expected_method = _canonical_robust_method(method_name)
    if saved_method and saved_method != expected_method:
        raise RuntimeError(
            f"Robust resume checkpoint method mismatch: current={expected_method} saved={saved_method} ({ckpt_path})"
        )
    if prefer_ema:
        trainer_state = payload.get("trainer_state", {})
        if isinstance(trainer_state, dict):
            ema_state_dict = trainer_state.get("ema_state_dict")
            if isinstance(ema_state_dict, dict):
                return ema_state_dict, "ema_state_dict"
    state_dict = payload.get("robust_state_dict")
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Missing robust_state_dict in robust checkpoint: {ckpt_path}")
    return state_dict, "robust_state_dict"


def _base_cfg_from_args(args: argparse.Namespace) -> ToyConfig:
    cfg = ToyConfig()
    cfg.outdir = args.outdir
    cfg.exp_name = args.prefix
    cfg.seed = 0
    cfg.device = args.device
    cfg.amp_dtype = args.amp_dtype
    cfg.dataset_kind = "image_folder"
    cfg.model_kind = "image_conv"
    cfg.diagnostics_kind = "image_basic"
    cfg.dataset_path = args.dataset_path
    cfg.dataset_val_path = args.dataset_val_path
    cfg.image_size = int(args.image_size)
    cfg.image_channels = int(args.image_channels)
    cfg.image_train_size = int(args.image_train_size)
    cfg.image_val_size = int(args.image_val_size)
    cfg.image_split_seed = int(args.image_split_seed)
    cfg.batch_size = int(args.batch_size)
    cfg.hidden_dim = int(args.hidden_dim)
    cfg.eval_samples = int(args.eval_samples)
    cfg.fid_samples = int(args.fid_samples)
    cfg.n_steps_path = int(args.n_steps_path_default)
    cfg.sigma_min = float(args.sigma_min)
    cfg.sigma_max = float(args.sigma_max)
    cfg.training_objective = "edm"
    cfg.compute_fid = True
    cfg.run_checks = False
    cfg.baseline_gate_enabled = False
    cfg.eval_seed_offset_metrics = int(args.metrics_eval_seed_offset)
    return cfg


def _apply_cfg_overrides(cfg: ToyConfig, source: Dict[str, object]) -> None:
    field_names = {
        "amp_dtype",
        "allow_tf32",
        "batch_size",
        "cudnn_benchmark",
        "dataset_kind",
        "dataset_path",
        "dataset_val_path",
        "eval_seed_offset_metrics",
        "fid_ref_path",
        "hidden_dim",
        "image_channels",
        "image_size",
        "image_split_seed",
        "image_train_size",
        "image_val_size",
        "limited_data_enabled",
        "model_kind",
        "n_steps_path",
        "sigma_data",
        "sigma_max",
        "sigma_min",
        "training_objective",
        "use_ema_eval",
    }
    for key in field_names:
        if key in source and source.get(key) is not None:
            setattr(cfg, key, source.get(key))


def _config_from_row(args: argparse.Namespace, row: Dict[str, str]) -> ToyConfig:
    cfg = _base_cfg_from_args(args)
    cfg.seed = _row_seed(row)
    cfg.steps = _row_step(row)
    metrics_path = _resolve_repo_path(row.get("metrics_path", ""))
    if metrics_path:
        payload = _load_json(metrics_path)
        source_cfg = payload.get("config", {})
        if isinstance(source_cfg, dict):
            _apply_cfg_overrides(cfg, source_cfg)
    if row.get("training_objective"):
        cfg.training_objective = str(row.get("training_objective"))
    if row.get("n_steps_path"):
        cfg.n_steps_path = _safe_int(row.get("n_steps_path"), cfg.n_steps_path)
    if row.get("sigma_min"):
        sigma_min = _safe_float(row.get("sigma_min"))
        if sigma_min is not None:
            cfg.sigma_min = float(sigma_min)
    if row.get("sigma_max"):
        sigma_max = _safe_float(row.get("sigma_max"))
        if sigma_max is not None:
            cfg.sigma_max = float(sigma_max)
    cfg.amp_dtype = args.amp_dtype
    cfg.outdir = args.outdir
    cfg.exp_name = args.prefix
    cfg.dataset_path = _resolve_repo_path(cfg.dataset_path)
    cfg.dataset_val_path = _resolve_repo_path(cfg.dataset_val_path)
    cfg.fid_ref_path = _resolve_repo_path(getattr(cfg, "fid_ref_path", "") or args.fid_ref_path)
    return cfg


def _context_key_for_cfg(cfg: ToyConfig, device: torch.device) -> Tuple[object, ...]:
    return (
        str(device),
        str(cfg.amp_dtype),
        str(cfg.dataset_kind),
        str(cfg.model_kind),
        str(cfg.training_objective),
        str(cfg.dataset_path),
        str(cfg.dataset_val_path),
        int(cfg.image_size),
        int(cfg.image_channels),
        int(cfg.image_train_size),
        int(cfg.image_val_size),
        int(cfg.image_split_seed),
        bool(cfg.limited_data_enabled),
        int(cfg.hidden_dim),
        int(cfg.n_steps_path),
        float(cfg.sigma_min),
        float(cfg.sigma_max),
        float(getattr(cfg, "sigma_data", -1.0)),
        bool(getattr(cfg, "allow_tf32", True)),
        bool(getattr(cfg, "cudnn_benchmark", True)),
        int(getattr(cfg, "eval_seed_offset_metrics", ToyConfig.eval_seed_offset_metrics)),
    )


def _get_or_build_context(
    *,
    args: argparse.Namespace,
    row: Dict[str, str],
    cache: Dict[Tuple[object, ...], EvalContext],
) -> EvalContext:
    cfg = _config_from_row(args, row)
    device = pick_device(str(cfg.device))
    key = _context_key_for_cfg(cfg, device)
    cached = cache.get(key)
    if cached is not None:
        return cached

    configure_runtime(
        device=device,
        allow_tf32=bool(getattr(cfg, "allow_tf32", True)),
        cudnn_benchmark=bool(getattr(cfg, "cudnn_benchmark", True)),
    )
    amp_dtype = resolve_amp_dtype(device, str(cfg.amp_dtype))
    dataset = build_dataset_bundle(cfg, device)
    if float(getattr(cfg, "sigma_data", -1.0)) <= 0.0:
        cfg.sigma_data = float(dataset.estimate_sigma_data())
    model_bundle = build_model_bundle(cfg, dataset, float(cfg.sigma_data), device)
    sigma_levels = build_sigma_levels(float(cfg.sigma_min), float(cfg.sigma_max), int(cfg.n_steps_path), device=device)
    context = EvalContext(
        key=key,
        cfg=cfg,
        device=device,
        amp_dtype=amp_dtype,
        dataset=dataset,
        sigma_levels=sigma_levels,
        baseline_model=model_bundle.baseline,
        robust_model=model_bundle.robust,
    )
    cache[key] = context
    return context


def _extract_fid(metrics_path: str, row: Dict[str, str]) -> float:
    payload = _load_json(metrics_path)
    sample_quality = payload.get("metrics", {}).get("sample_quality_debug", {})
    key = _metrics_fid_key(row)
    fid_value = _safe_float(sample_quality.get(key))
    if fid_value is None:
        raise RuntimeError(f"Missing {key} in reevaluated metrics: {metrics_path}")
    return float(fid_value)


def _write_metrics_payload(
    *,
    path: str,
    row: Dict[str, str],
    ctx: EvalContext,
    fid_value: float,
    fid_batch_size: int,
    metrics_eval_seed: int,
    runtime_sec: float,
    checkpoint_path: str,
    checkpoint_state_variant: str,
) -> None:
    branch = _checkpoint_branch(row)
    sample_quality = {
        "baseline_fid": None,
        "robust_fid": None,
        "fid_branch": branch,
        "fid_samples": int(ctx.cfg.fid_samples),
        "fid_batch_size": int(fid_batch_size),
        "evaluation_protocol": {
            "metrics_eval_seed": int(metrics_eval_seed),
            "seed_reset_per_branch": True,
            "direct_fid_only": True,
        },
    }
    sample_quality[_metrics_fid_key(row)] = float(fid_value)
    payload = {
        "format": "toy_fid_reeval_direct_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "dataset_kind": str(ctx.cfg.dataset_kind),
            "dataset_path": str(ctx.cfg.dataset_path),
            "dataset_val_path": str(ctx.cfg.dataset_val_path),
            "image_size": int(ctx.cfg.image_size),
            "image_channels": int(ctx.cfg.image_channels),
            "image_train_size": int(ctx.cfg.image_train_size),
            "image_val_size": int(ctx.cfg.image_val_size),
            "image_split_seed": int(ctx.cfg.image_split_seed),
            "hidden_dim": int(ctx.cfg.hidden_dim),
            "n_steps_path": int(ctx.cfg.n_steps_path),
            "sigma_min": float(ctx.cfg.sigma_min),
            "sigma_max": float(ctx.cfg.sigma_max),
            "sigma_data": float(ctx.cfg.sigma_data),
            "training_objective": str(ctx.cfg.training_objective),
            "amp_dtype": str(ctx.cfg.amp_dtype),
            "eval_seed_offset_metrics": int(ctx.cfg.eval_seed_offset_metrics),
            "fid_ref_path": str(ctx.cfg.fid_ref_path),
            "compute_fid": True,
        },
        "row": {
            "method": str(row.get("method", "")),
            "robust_method": str(row.get("robust_method", "")),
            "backbone_family": str(row.get("backbone_family", "")),
            "series_key": str(row.get("series_key", "")),
            "series_label": str(row.get("series_label", "")),
            "row_origin": str(row.get("row_origin", "")),
            "seed": int(_row_seed(row)),
            "step": int(_row_step(row)),
            "checkpoint_path": checkpoint_path,
        },
        "metrics": {
            "sample_quality_debug": sample_quality,
            "flow_debug": {
                "baseline_only": bool(branch == "baseline"),
                "direct_fid_only": True,
                "checkpoint_branch": branch,
                "checkpoint_state_variant": str(checkpoint_state_variant),
            },
        },
        "runtime_sec": {
            "fid_total": float(runtime_sec),
        },
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _evaluate_checkpoint_fid(
    *,
    args: argparse.Namespace,
    row: Dict[str, str],
    ctx: EvalContext,
    detector_net,
    mu_ref: torch.Tensor,
    sigma_ref: torch.Tensor,
    metrics_path: str,
    log_path: str,
) -> float:
    checkpoint_path = _resolve_repo_path(row.get("checkpoint_path", ""))
    if not checkpoint_path:
        raise RuntimeError(f"Missing checkpoint_path in row: method={row.get('method')} step={row.get('step')}")
    ensure_dir(os.path.dirname(metrics_path))
    ensure_dir(os.path.dirname(log_path))

    branch = _checkpoint_branch(row)
    metrics_eval_seed = int(_row_seed(row) + int(ctx.cfg.eval_seed_offset_metrics))
    proc_title = build_process_title("wdiff", "reeval", row.get("method", ""), f"st{_row_step(row)}")
    apply_process_title(proc_title)
    model = ctx.baseline_model if branch == "baseline" else ctx.robust_model

    with open(log_path, "w", encoding="utf-8") as log_handle:
        log_handle.write(
            f"[fid-only] branch={branch} method={row.get('method')} row_origin={row.get('row_origin')} "
            f"step={_row_step(row)} seed={_row_seed(row)}\n"
        )
        log_handle.write(f"[fid-only] checkpoint={checkpoint_path}\n")
        log_handle.write(
            f"[fid-only] device={ctx.device} amp_dtype={format_amp_dtype(ctx.amp_dtype)} "
            f"fid_samples={int(args.fid_samples)} fid_batch_size={int(args.fid_batch_size)}\n"
        )
        log_handle.write(
            f"[fid-only] metrics_seed={metrics_eval_seed} image_shape="
            f"{tuple(int(v) for v in ctx.dataset.data_shape)} sigma_data={float(ctx.cfg.sigma_data):.6f}\n"
        )
        log_handle.write("[fid-only] legacy eval-samples argument is ignored in direct FID mode.\n")
        log_handle.flush()

        checkpoint_state_variant = "baseline_eval_state_dict"
        if branch == "baseline":
            state_dict = _load_baseline_state_dict(checkpoint_path)
        else:
            state_dict, checkpoint_state_variant = _load_robust_state_dict(
                checkpoint_path,
                str(row.get("method", "")),
                prefer_ema=bool(getattr(ctx.cfg, "use_ema_eval", False)),
            )
        log_handle.write(f"[fid-only] checkpoint_state_variant={checkpoint_state_variant}\n")
        log_handle.flush()
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        set_seed(metrics_eval_seed)
        t_start = time.perf_counter()
        fid_value = _compute_fid_for_model(
            denoiser=model,
            sigma_levels=ctx.sigma_levels,
            dataset=ctx.dataset,
            detector_net=detector_net,
            mu_ref=mu_ref.to(device=ctx.device),
            sigma_ref=sigma_ref.to(device=ctx.device),
            num_images=int(args.fid_samples),
            gen_batch=max(int(args.fid_batch_size), 1),
            device=ctx.device,
            amp_dtype=ctx.amp_dtype,
            log_handle=log_handle,
        )
        runtime_sec = float(time.perf_counter() - t_start)
        log_handle.write(f"[fid-only] done fid={fid_value:.6f} runtime_sec={runtime_sec:.2f}\n")
        log_handle.flush()

    _write_metrics_payload(
        path=metrics_path,
        row=row,
        ctx=ctx,
        fid_value=fid_value,
        fid_batch_size=max(int(args.fid_batch_size), 1),
        metrics_eval_seed=metrics_eval_seed,
        runtime_sec=runtime_sec,
        checkpoint_path=checkpoint_path,
        checkpoint_state_variant=checkpoint_state_variant,
    )
    return fid_value


def _run_plot_command(
    *,
    args: argparse.Namespace,
    combined_out: str,
    plots_dir: str,
    plot_log: str,
) -> None:
    plot_cmd = [
        args.python_bin,
        args.plot_script,
        "--combined-csv",
        combined_out,
        "--outdir",
        plots_dir,
        "--prefix",
        args.prefix,
        "--train-percent-label",
        args.train_percent_label,
    ]
    with open(plot_log, "w", encoding="utf-8") as handle:
        subprocess.run(
            plot_cmd,
            cwd=ROOT_DIR,
            check=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=child_process_env(proc_title=build_process_title("wdiff", "plot", args.prefix)),
        )


def _row_matches_filters(args: argparse.Namespace, row: Dict[str, str], methods_filter: Optional[set]) -> bool:
    if methods_filter is not None and _canonical_robust_method(row.get("robust_method", row.get("method", ""))) not in methods_filter:
        return False
    if bool(args.respect_fid_selection) and not _safe_bool(row.get("fid_eval_selected"), default=True):
        return False
    if bool(args.only_missing_fid) and _safe_float(row.get("fid")) is not None:
        return False
    return True


def main() -> None:
    args = parse_args()
    args.combined_csv = _resolve_repo_path(args.combined_csv)
    args.outdir = _resolve_repo_path(args.outdir)
    args.plot_script = _resolve_repo_path(args.plot_script)
    args.dataset_path = _resolve_repo_path(args.dataset_path)
    args.dataset_val_path = _resolve_repo_path(args.dataset_val_path)
    args.fid_ref_path = _resolve_repo_path(args.fid_ref_path)

    if _APPLIED_PROCESS_TITLE is None:
        apply_process_title(build_process_title("wdiff", "reeval", args.prefix))

    ensure_dir(args.outdir)
    eval_runs_dir = os.path.join(args.outdir, "reeval_runs")
    logs_dir = os.path.join(args.outdir, "logs")
    plots_dir = os.path.join(args.outdir, "plots")
    ensure_dir(eval_runs_dir)
    ensure_dir(logs_dir)
    ensure_dir(plots_dir)

    input_rows = load_rows(args.combined_csv)
    if not input_rows:
        raise RuntimeError(f"No rows found in combined CSV: {args.combined_csv}")

    methods_filter = _parse_method_filter(args.methods)
    eligible_row_indices = [
        row_index
        for row_index, row in enumerate(input_rows)
        if _row_matches_filters(args, row, methods_filter)
    ]
    eligible_row_index_set = set(int(row_index) for row_index in eligible_row_indices)
    eligible_rows = [input_rows[row_index] for row_index in eligible_row_indices]

    device = pick_device(args.device)
    detector_net = None
    mu_ref = None
    sigma_ref = None
    if eligible_rows:
        detector_net = _load_detector(device)
        mu_ref, sigma_ref = _load_ref_stats(args.fid_ref_path)
        mu_ref = mu_ref.to(device=device)
        sigma_ref = sigma_ref.to(device=device)

    reevaluated_rows: List[Dict[str, str]] = []
    cache: Dict[Tuple[str, str], Dict[str, str]] = {}
    context_cache: Dict[Tuple[object, ...], EvalContext] = {}

    for row_index, row in enumerate(input_rows):
        if row_index not in eligible_row_index_set:
            reevaluated_rows.append(dict(row))
            continue
        cache_key = _reeval_cache_key(row)
        cached = cache.get(cache_key)
        if cached is None:
            exp_name = _exp_name_from_row(args.prefix, row, row_index)
            metrics_path = os.path.join(eval_runs_dir, exp_name, "metrics.json")
            log_path = os.path.join(logs_dir, f"{exp_name}.log")
            if bool(args.skip_existing) and os.path.isfile(metrics_path):
                fid_value = _extract_fid(metrics_path, row)
            else:
                ctx = _get_or_build_context(args=args, row=row, cache=context_cache)
                fid_value = _evaluate_checkpoint_fid(
                    args=args,
                    row=row,
                    ctx=ctx,
                    detector_net=detector_net,
                    mu_ref=mu_ref,
                    sigma_ref=sigma_ref,
                    metrics_path=metrics_path,
                    log_path=log_path,
                )
            cached = {
                "fid": str(fid_value),
                "reeval_metrics_path": metrics_path,
                "reeval_log_path": log_path,
                "reeval_exp_name": exp_name,
            }
            cache[cache_key] = cached

        updated = dict(row)
        updated["fid_original"] = str(row.get("fid", ""))
        updated["fid"] = str(cached["fid"])
        updated["reeval_metrics_path"] = str(cached["reeval_metrics_path"])
        updated["reeval_log_path"] = str(cached["reeval_log_path"])
        updated["reeval_exp_name"] = str(cached["reeval_exp_name"])
        updated["reeval_fid_samples"] = str(int(args.fid_samples))
        updated["reeval_fid_batch_size"] = str(max(int(args.fid_batch_size), 1))
        updated["reeval_mode"] = "direct_fid_only_in_memory"
        updated["fid_evaluated"] = True
        updated["fid_missing_reason"] = ""
        updated["fid_source"] = "reevaluated_from_checkpoint"
        reevaluated_rows.append(updated)

    combined_out = os.path.join(args.outdir, f"{args.prefix}_all_methods_raw_seed_rows.csv")
    write_csv(combined_out, reevaluated_rows)

    plot_log = None
    if not bool(args.skip_plot):
        plot_log = os.path.join(logs_dir, f"{args.prefix}_plot.log")
        _run_plot_command(
            args=args,
            combined_out=combined_out,
            plots_dir=plots_dir,
            plot_log=plot_log,
        )

    manifest = {
        "combined_csv_in": args.combined_csv,
        "combined_csv_out": combined_out,
        "fid_samples": int(args.fid_samples),
        "fid_batch_size": max(int(args.fid_batch_size), 1),
        "reeval_mode": "direct_fid_only_in_memory",
        "device": str(device),
        "fid_ref_path": args.fid_ref_path,
        "methods_filter": None if methods_filter is None else sorted(methods_filter),
        "respect_fid_selection": bool(args.respect_fid_selection),
        "only_missing_fid": bool(args.only_missing_fid),
        "eligible_rows": len(eligible_rows),
        "plots_dir": None if bool(args.skip_plot) else plots_dir,
        "plot_log": plot_log,
        "unique_reevaluations": len(cache),
        "unique_contexts": len(context_cache),
    }
    with open(os.path.join(args.outdir, f"{args.prefix}_reeval_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"[reeval] wrote {combined_out}", flush=True)
    print(
        f"[reeval] eligible_rows={len(eligible_rows)} unique_reevaluations={len(cache)} "
        f"fid_samples={int(args.fid_samples)} "
        f"fid_batch_size={max(int(args.fid_batch_size), 1)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
