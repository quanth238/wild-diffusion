#!/usr/bin/env python3
"""Checkpointed MNIST baseline-EDM convergence sweep.

This is the efficient version of the MNIST convergence sweep:
- train baseline EDM once per (train_percent, seed) to max(step_list),
- evaluate intermediate checkpoints at the requested step milestones,
- aggregate FID/loss across seeds,
- extract an earliest-threshold step based on best median FID.

Unlike `sweep_mnist_convergence.py`, this does not retrain from scratch for
every checkpoint.
"""

from __future__ import annotations

import copy
import contextlib
import csv
import json
import math
import os
import pickle
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from toy.compute_accounting import load_weighted_compute_calibration, weighted_compute_units
    from toy.config import ToyConfig
    from toy.data_backends.provider import build_dataset_bundle
    from toy.export_mnist_fid_ref import build_mnist_fid_reference, default_mnist_fid_policy_name
    from toy.model_backends.provider import build_model_bundle
    from toy.models import set_requires_grad
    from toy.process_title import apply_process_title, build_process_title
    from toy.shared.ema import ema_config_dict, init_ema_model, update_ema_model
    from toy.shared.objective import build_training_state, compute_training_loss, weighted_denoise_loss
    from toy.shared.reverse import sample_reverse_paths
    from toy.shared.sigma import build_sigma_levels, sample_target_indices, sample_target_indices_log_normal
    from toy.shared.train_utils import sample_train_batch
    from toy.utils import batch_scalar_like, ensure_dir, has_nan_or_inf, pick_device, scalarize, set_seed
else:
    from ..compute_accounting import load_weighted_compute_calibration, weighted_compute_units
    from ..config import ToyConfig
    from ..data_backends.provider import build_dataset_bundle
    from ..export_mnist_fid_ref import build_mnist_fid_reference, default_mnist_fid_policy_name
    from ..model_backends.provider import build_model_bundle
    from ..models import set_requires_grad
    from ..process_title import apply_process_title, build_process_title
    from ..shared.ema import ema_config_dict, init_ema_model, update_ema_model
    from ..shared.objective import build_training_state, compute_training_loss, weighted_denoise_loss
    from ..shared.reverse import sample_reverse_paths
    from ..shared.sigma import build_sigma_levels, sample_target_indices, sample_target_indices_log_normal
    from ..shared.train_utils import sample_train_batch
    from ..utils import batch_scalar_like, ensure_dir, has_nan_or_inf, pick_device, scalarize, set_seed


_APPLIED_PROCESS_TITLE = apply_process_title()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parse_int_list(text: str, *, allow_zero: bool = False) -> List[int]:
    values: List[int] = []
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        value = int(tok)
        if allow_zero:
            if value < 0:
                raise ValueError(f"Expected non-negative integer, got {value}")
        else:
            if value <= 0:
                raise ValueError(f"Expected positive integer, got {value}")
        values.append(value)
    if not values:
        raise ValueError("Empty list.")
    return sorted(set(values))


def _parse_float_list(text: str) -> List[float]:
    values: List[float] = []
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        value = float(tok)
        if value <= 0.0:
            raise ValueError(f"Expected positive float, got {value}")
        values.append(value)
    if not values:
        raise ValueError("Empty list.")
    return sorted(set(values))


def _checkpoint_steps_from_mimg_list(mimg_values: Sequence[float], *, batch_size: int) -> List[int]:
    steps: List[int] = []
    for mimg in mimg_values:
        step = max(1, int(round(float(mimg) * 1_000_000.0 / float(batch_size))))
        steps.append(step)
    return sorted(set(steps))


def _safe_float(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _coerce_bool(value: Any, *, default: bool = False) -> bool:
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


def _fmt(value: float, ndigits: int = 4) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value:.{ndigits}f}"


def _resolve_weighted_calibration(args) -> Dict[str, Any]:
    return load_weighted_compute_calibration(
        calibration_path=str(getattr(args, "weighted_compute_calibration_path", "")).strip(),
        inputgrad_alpha=float(getattr(args, "weighted_inputgrad_alpha", 0.0)),
        parambackward_beta=float(getattr(args, "weighted_parambackward_beta", 0.0)),
    )


def _count_image_files(root: Path) -> int:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}
    return sum(1 for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows_list = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows_list:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows_list[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_list:
            writer.writerow(row)


def _median(values: List[Any]) -> float:
    finite = [fv for fv in (_safe_float(v) for v in values) if math.isfinite(fv)]
    if not finite:
        return float("nan")
    return float(statistics.median(finite))


def _mean(values: List[Any]) -> float:
    finite = [fv for fv in (_safe_float(v) for v in values) if math.isfinite(fv)]
    if not finite:
        return float("nan")
    return float(statistics.fmean(finite))


def _std(values: List[Any]) -> float:
    finite = [fv for fv in (_safe_float(v) for v in values) if math.isfinite(fv)]
    if len(finite) < 2:
        return float("nan")
    return float(statistics.stdev(finite))


def _capture_rng_state() -> Dict[str, Any]:
    state = {
        "python": __import__("random").getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Dict[str, Any]) -> None:
    __import__("random").setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _run_with_scoped_seed(seed: int, fn):
    state = _capture_rng_state()
    set_seed(int(seed))
    try:
        return fn()
    finally:
        _restore_rng_state(state)


def _apply_auto_log_normal_params(cfg: ToyConfig) -> None:
    if cfg.use_log_normal_sigma_sampling and cfg.auto_log_normal_params:
        log_min = math.log(cfg.sigma_min)
        log_max = math.log(cfg.sigma_max)
        cfg.p_mean = 0.5 * (log_min + log_max)
        cfg.p_std = max((log_max - log_min) / 6.0, 1e-3)


def _save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    raw_model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    ema_model: Optional[torch.nn.Module] = None,
    rng_state: Optional[Dict[str, Any]] = None,
    step: int,
    train_percent: float,
    seed: int,
    history: Dict[str, Any],
    train_wall_clock_sec: Optional[float] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "mnist_baseline_curve_ckpt_v1",
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "step": int(step),
        "train_percent": float(train_percent),
        "seed": int(seed),
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "history": history,
        "baseline_runtime": {
            "step": int(step),
            "train_wall_clock_sec": None if train_wall_clock_sec is None else float(train_wall_clock_sec),
        },
    }
    if raw_model is not None:
        payload["model_state_dict"] = _to_cpu_tree(raw_model.state_dict())
    if optimizer is not None:
        payload["optimizer_state_dict"] = _to_cpu_tree(optimizer.state_dict())
    if ema_model is not None:
        payload["ema_state_dict"] = _to_cpu_tree(ema_model.state_dict())
    if rng_state is not None:
        payload["rng_state"] = _to_cpu_tree(rng_state)
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def _to_cpu_tree(obj: Any) -> Any:
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {key: _to_cpu_tree(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu_tree(value) for value in obj]
    if isinstance(obj, tuple):
        return tuple(_to_cpu_tree(value) for value in obj)
    return obj


def _optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device=device)


def _build_run_state_signature(*, cfg: ToyConfig, dataset, train_percent: float, seed: int) -> Dict[str, Any]:
    return {
        "signature_version": 1,
        "dataset_kind": str(cfg.dataset_kind),
        "dataset_path": str(getattr(cfg, "dataset_path", "")),
        "dataset_val_path": str(getattr(cfg, "dataset_val_path", "")),
        "train_percent": float(train_percent),
        "seed": int(seed),
        "image_size": int(cfg.image_size),
        "image_channels": int(cfg.image_channels),
        "train_subset_size": int(dataset.train_pool.shape[0]) if dataset.train_pool is not None else None,
        "val_subset_size": int(dataset.val_pool.shape[0]),
        "batch_size": int(cfg.batch_size),
        "hidden_dim": int(cfg.hidden_dim),
        "lr_theta": float(cfg.lr_theta),
        "training_objective": str(cfg.training_objective),
        "n_steps_path": int(cfg.n_steps_path),
        "sigma_min": float(cfg.sigma_min),
        "sigma_max": float(cfg.sigma_max),
        "use_log_normal_sigma_sampling": bool(cfg.use_log_normal_sigma_sampling),
        "p_mean": float(cfg.p_mean),
        "p_std": float(cfg.p_std),
        **ema_config_dict(cfg),
    }


def _save_run_state(
    path: Path,
    *,
    signature: Dict[str, Any],
    step: int,
    planned_max_step: int,
    checkpoint_steps: List[int],
    fid_eval_steps: List[int],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    grad_scaler,
    ema_model: Optional[torch.nn.Module],
    history: Dict[str, List[float]],
    sigma_counts: torch.Tensor,
    rows: List["RunRow"],
    combo_train_elapsed_sec: float,
    completed: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "baseline_curve_run_state_v1",
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "signature": signature,
        "step": int(step),
        "planned_max_step": int(planned_max_step),
        "checkpoint_steps": [int(v) for v in checkpoint_steps],
        "fid_eval_steps": [int(v) for v in fid_eval_steps],
        "model_state_dict": _to_cpu_tree(model.state_dict()),
        "optimizer_state_dict": _to_cpu_tree(optimizer.state_dict()),
        "grad_scaler_state_dict": None if grad_scaler is None else _to_cpu_tree(grad_scaler.state_dict()),
        "ema_state_dict": None if ema_model is None else _to_cpu_tree(ema_model.state_dict()),
        "history": {
            "loss": [float(v) for v in history.get("loss", [])],
            "proxy_weighted_denoise_loss": [float(v) for v in history.get("proxy_weighted_denoise_loss", [])],
        },
        "sigma_counts": [int(v) for v in sigma_counts.detach().cpu().tolist()],
        "rows": [row.to_dict() for row in rows],
        "combo_train_elapsed_sec": float(combo_train_elapsed_sec),
        "rng_state": _capture_rng_state(),
        "completed": bool(completed),
    }
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def _load_run_state(path: Path) -> Dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid run state payload at {path}")
    if payload.get("format") != "baseline_curve_run_state_v1":
        raise RuntimeError(f"Unsupported run state format at {path}: {payload.get('format')}")
    return payload


def _write_combo_rows_summary(
    summary_path: Path,
    *,
    combo_name: str,
    train_percent: float,
    seed: int,
    rows: List["RunRow"],
    run_state_path: Path,
    completed: bool,
) -> None:
    summary_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "combo_name": combo_name,
                "train_percent": float(train_percent),
                "seed": int(seed),
                "completed": bool(completed),
                "run_state_path": str(run_state_path),
                "rows": [row.to_dict() for row in rows],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


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
        print(f"[fid] using detector: {detector_path}", flush=True)
        with open(detector_path, "rb") as handle:
            detector_net = pickle.load(handle).to(device)
    else:
        print("[fid] detector not cached locally, using remote open_url()", flush=True)
        with dnnlib.util.open_url(detector_url, verbose=True) as handle:
            detector_net = pickle.load(handle).to(device)
    detector_net.eval()
    return detector_net


def _load_ref_stats(path: Path) -> tuple[np.ndarray, np.ndarray]:
    ref = np.load(path)
    return ref["mu"], ref["sigma"]


def _configure_runtime(*, device: torch.device, allow_tf32: bool, cudnn_benchmark: bool) -> None:
    if device.type != "cuda":
        return
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
        torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def _resolve_amp_dtype(device: torch.device, amp_dtype: str) -> Optional[torch.dtype]:
    mode = str(amp_dtype).strip().lower()
    if device.type != "cuda" or mode == "off":
        return None
    if mode == "auto":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if mode in {"bf16", "bfloat16"}:
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("Requested BF16 autocast but this CUDA device does not support BF16.")
        return torch.bfloat16
    if mode in {"fp16", "float16", "half"}:
        return torch.float16
    raise ValueError(f"Unsupported amp dtype '{amp_dtype}'. Expected one of: auto, off, bfloat16, float16.")


def _autocast_context(device: torch.device, amp_dtype: Optional[torch.dtype]):
    if amp_dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=amp_dtype)


def _format_amp_dtype(amp_dtype: Optional[torch.dtype]) -> str:
    if amp_dtype is None:
        return "off"
    if amp_dtype == torch.bfloat16:
        return "bfloat16"
    if amp_dtype == torch.float16:
        return "float16"
    return str(amp_dtype)


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
) -> float:
    feature_dim = 2048
    mu = torch.zeros([feature_dim], dtype=torch.float64, device=device)
    sigma = torch.zeros([feature_dim, feature_dim], dtype=torch.float64, device=device)
    n_done = 0

    denoiser.eval()
    while n_done < num_images:
        cur = min(gen_batch, num_images - n_done)
        with _autocast_context(device, amp_dtype):
            states = sample_reverse_paths(
                denoiser=denoiser,
                sigma_levels=sigma_levels,
                n_samples=cur,
                device=device,
                stochastic=True,
                sample_terminal_batch_fn=dataset.sample_terminal_batch,
            )
        images = states[:, 0]
        images = ((images + 1.0) * 127.5).clamp(0.0, 255.0).to(torch.uint8)
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])
        features = detector_net(images, return_features=True).to(torch.float64)
        mu += features.sum(0)
        sigma += features.T @ features
        n_done += cur

    mu /= num_images
    sigma -= mu.ger(mu) * num_images
    sigma /= max(num_images - 1, 1)
    return _calculate_fid_from_stats_torch(mu, sigma, mu_ref, sigma_ref)


@dataclass
class RunRow:
    train_percent: float
    seed: int
    step: int
    images_shown_m: float
    exp_name: str
    exp_dir: str
    checkpoint_path: str
    train_subset_size: int
    val_subset_size: int
    baseline_fid: Optional[float]
    baseline_loss_final: float
    baseline_loss_mean_last: float
    train_elapsed_sec: float
    train_wall_clock_sec: float
    train_gpu_hours: float
    weighted_compute_units: float
    batch_equiv_denoiser_evals: float
    fid_elapsed_sec: Optional[float]
    fid_eval_selected: bool
    fid_evaluated: bool
    fid_missing_reason: str
    fid_source: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "train_percent": self.train_percent,
            "seed": self.seed,
            "step": self.step,
            "images_shown_m": self.images_shown_m,
            "exp_name": self.exp_name,
            "exp_dir": self.exp_dir,
            "checkpoint_path": self.checkpoint_path,
            "train_subset_size": self.train_subset_size,
            "val_subset_size": self.val_subset_size,
            "baseline_fid": self.baseline_fid,
            "baseline_loss_final": self.baseline_loss_final,
            "baseline_loss_mean_last": self.baseline_loss_mean_last,
            "train_elapsed_sec": self.train_elapsed_sec,
            "train_wall_clock_sec": self.train_wall_clock_sec,
            "train_gpu_hours": self.train_gpu_hours,
            "weighted_compute_units": self.weighted_compute_units,
            "batch_equiv_denoiser_evals": self.batch_equiv_denoiser_evals,
            "fid_elapsed_sec": self.fid_elapsed_sec,
            "fid_eval_selected": bool(self.fid_eval_selected),
            "fid_evaluated": bool(self.fid_evaluated),
            "fid_missing_reason": str(self.fid_missing_reason),
            "fid_source": str(self.fid_source),
        }


@dataclass
class AggregateRow:
    train_percent: float
    step: int
    images_shown_m: float
    train_subset_size_median: float
    n_runs: int
    fid_median: float
    fid_mean: float
    fid_std: float
    loss_final_median: float
    loss_mean_last_median: float
    train_elapsed_median_sec: float
    train_wall_clock_median_sec: float
    train_gpu_hours_median: float
    weighted_compute_units_median: float
    batch_equiv_denoiser_evals_median: float
    fid_elapsed_median_sec: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "train_percent": self.train_percent,
            "step": self.step,
            "row_aggregation": "median",
            "images_shown_m": self.images_shown_m,
            "train_subset_size_median": self.train_subset_size_median,
            "n_runs": self.n_runs,
            "fid_median": self.fid_median,
            "fid_mean": self.fid_mean,
            "fid_std": self.fid_std,
            "loss_final_median": self.loss_final_median,
            "loss_mean_last_median": self.loss_mean_last_median,
            "train_elapsed_median_sec": self.train_elapsed_median_sec,
            "train_wall_clock_median_sec": self.train_wall_clock_median_sec,
            "train_gpu_hours_median": self.train_gpu_hours_median,
            "weighted_compute_units_median": self.weighted_compute_units_median,
            "batch_equiv_denoiser_evals_median": self.batch_equiv_denoiser_evals_median,
            "fid_elapsed_median_sec": self.fid_elapsed_median_sec,
        }


def _aggregate(rows: List[RunRow]) -> List[AggregateRow]:
    buckets: Dict[tuple[float, int], List[RunRow]] = {}
    for row in rows:
        buckets.setdefault((row.train_percent, row.step), []).append(row)

    out: List[AggregateRow] = []
    for (train_percent, step), sub in sorted(buckets.items(), key=lambda item: (item[0][0], item[0][1])):
        out.append(
            AggregateRow(
                train_percent=float(train_percent),
                step=int(step),
                images_shown_m=sub[0].images_shown_m,
                train_subset_size_median=_median([float(row.train_subset_size) for row in sub]),
                n_runs=len(sub),
                fid_median=_median([row.baseline_fid for row in sub]),
                fid_mean=_mean([row.baseline_fid for row in sub]),
                fid_std=_std([row.baseline_fid for row in sub]),
                loss_final_median=_median([row.baseline_loss_final for row in sub]),
                loss_mean_last_median=_median([row.baseline_loss_mean_last for row in sub]),
                train_elapsed_median_sec=_median([row.train_elapsed_sec for row in sub]),
                train_wall_clock_median_sec=_median([row.train_wall_clock_sec for row in sub]),
                train_gpu_hours_median=_median([row.train_gpu_hours for row in sub]),
                weighted_compute_units_median=_median([row.weighted_compute_units for row in sub]),
                batch_equiv_denoiser_evals_median=_median([row.batch_equiv_denoiser_evals for row in sub]),
                fid_elapsed_median_sec=_median([row.fid_elapsed_sec for row in sub]),
            )
        )
    return out


def _run_row_from_dict(
    row: Dict[str, Any],
    *,
    weighted_calibration: Dict[str, Any],
    train_accelerator_count: int,
) -> RunRow:
    payload = dict(row)
    step = int(payload["step"])
    train_elapsed = float(payload.get("train_elapsed_sec", float("nan")))
    payload.setdefault("train_wall_clock_sec", train_elapsed)
    payload.setdefault(
        "train_gpu_hours",
        float(train_elapsed) * float(max(int(train_accelerator_count), 0)) / 3600.0,
    )
    baseline_fid = _safe_float(payload.get("baseline_fid"))
    payload["baseline_fid"] = None if not math.isfinite(baseline_fid) else float(baseline_fid)
    fid_elapsed = _safe_float(payload.get("fid_elapsed_sec"))
    payload["fid_elapsed_sec"] = None if not math.isfinite(fid_elapsed) else float(fid_elapsed)
    if "weighted_compute_units" not in payload:
        weighted_units = weighted_compute_units(
            n_fwd=0.0,
            n_fwd_inputgrad=0.0,
            n_fwd_parambackward=float(step),
            calibration=weighted_calibration,
        )
        payload["weighted_compute_units"] = float(weighted_units) if weighted_units is not None else float("nan")
    payload.setdefault("batch_equiv_denoiser_evals", float(step))
    payload.setdefault("fid_eval_selected", _coerce_bool(payload.get("fid_eval_selected"), default=True))
    payload.setdefault(
        "fid_evaluated",
        _coerce_bool(
            payload.get("fid_evaluated"),
            default=math.isfinite(_safe_float(payload.get("baseline_fid"))),
        ),
    )
    payload.setdefault(
        "fid_missing_reason",
        ""
        if payload["fid_evaluated"]
        else ("not_selected_by_schedule" if not payload["fid_eval_selected"] else "missing_or_failed"),
    )
    payload.setdefault(
        "fid_source",
        "in_run_metrics" if payload["fid_evaluated"] else "not_evaluated",
    )
    return RunRow(**payload)


def _extract_thresholds(
    rows: List[AggregateRow],
    *,
    threshold_pct: float,
    overfit_pct: float,
    overfit_patience: int,
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    grouped: Dict[float, List[AggregateRow]] = {}
    for row in rows:
        grouped.setdefault(row.train_percent, []).append(row)

    for train_percent, sub in grouped.items():
        sub = sorted(sub, key=lambda row: row.step)
        finite = [row for row in sub if math.isfinite(row.fid_median)]
        if not finite:
            out[str(train_percent)] = {
                "train_percent": float(train_percent),
                "threshold_pct": float(threshold_pct),
                "best_fid_median": None,
                "best_step": None,
                "best_images_shown_m": None,
                "threshold_fid_ceiling": None,
                "threshold_step": None,
                "threshold_images_shown_m": None,
                "threshold_loss_mean_last_median": None,
                "overfit_pct": float(overfit_pct),
                "overfit_step": None,
                "overfit_images_shown_m": None,
            }
            continue

        best_row = min(finite, key=lambda row: (row.fid_median, row.step))
        threshold_fid_ceiling = best_row.fid_median * (1.0 + threshold_pct / 100.0)
        threshold_row = next((row for row in finite if row.fid_median <= threshold_fid_ceiling), best_row)

        overfit_fid_floor = best_row.fid_median * (1.0 + overfit_pct / 100.0)
        overfit_row: Optional[AggregateRow] = None
        consecutive = 0
        start_row: Optional[AggregateRow] = None
        for row in finite:
            if row.step <= best_row.step:
                continue
            if row.fid_median > overfit_fid_floor:
                consecutive += 1
                if consecutive == 1:
                    start_row = row
                if consecutive >= overfit_patience:
                    overfit_row = start_row
                    break
            else:
                consecutive = 0
                start_row = None

        out[str(train_percent)] = {
            "train_percent": float(train_percent),
            "threshold_pct": float(threshold_pct),
            "best_fid_median": float(best_row.fid_median),
            "best_step": int(best_row.step),
            "best_images_shown_m": float(best_row.images_shown_m),
            "best_loss_mean_last_median": float(best_row.loss_mean_last_median),
            "threshold_fid_ceiling": float(threshold_fid_ceiling),
            "threshold_step": int(threshold_row.step),
            "threshold_images_shown_m": float(threshold_row.images_shown_m),
            "threshold_loss_mean_last_median": float(threshold_row.loss_mean_last_median),
            "overfit_pct": float(overfit_pct),
            "overfit_fid_floor": float(overfit_fid_floor),
            "overfit_step": None if overfit_row is None else int(overfit_row.step),
            "overfit_images_shown_m": None if overfit_row is None else float(overfit_row.images_shown_m),
        }
    return out


def _plot_curves(
    *,
    rows: List[AggregateRow],
    thresholds: Dict[str, Dict[str, Any]],
    out_path: Path,
    title: str,
    dataset_label: str,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required to render the convergence figure.") from exc

    grouped: Dict[float, List[AggregateRow]] = {}
    for row in rows:
        grouped.setdefault(row.train_percent, []).append(row)

    colors = {5.0: "#8c564b", 10.0: "#4d4d4d", 20.0: "#e84a5f", 50.0: "#2b6ee8", 100.0: "#2ca25f"}
    markers = {5.0: "D", 10.0: "s", 20.0: "o", 50.0: "^", 100.0: "v"}
    fig, (ax_fid, ax_loss) = plt.subplots(1, 2, figsize=(14.5, 5.4), constrained_layout=True)

    for train_percent in sorted(grouped.keys()):
        sub = sorted(grouped[train_percent], key=lambda row: row.step)
        color = colors.get(train_percent, None)
        marker = markers.get(train_percent, "o")
        x = [row.images_shown_m for row in sub]
        y_fid = [row.fid_median for row in sub]
        y_loss = [row.loss_mean_last_median for row in sub]
        label = f"{dataset_label} ({train_percent:g}%)"
        ax_fid.plot(x, y_fid, color=color, marker=marker, linewidth=2.0, markersize=6.0, label=label)
        ax_loss.plot(x, y_loss, color=color, marker=marker, linewidth=2.0, markersize=6.0, label=label)
        summary = thresholds.get(str(train_percent))
        if summary is not None and summary.get("threshold_images_shown_m") is not None:
            ax_fid.axvline(float(summary["threshold_images_shown_m"]), color=color, linestyle="--", alpha=0.18, linewidth=1.2)

    ax_fid.set_title(f"{dataset_label} Convergence (FID)")
    ax_fid.set_xlabel("Number of images shown to model (M)")
    ax_fid.set_ylabel("FID")
    ax_fid.grid(alpha=0.25)
    ax_fid.legend(loc="best", fontsize=9)

    ax_loss.set_title(f"{dataset_label} Training Loss")
    ax_loss.set_xlabel("Number of images shown to model (M)")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(alpha=0.25)
    ax_loss.legend(loc="best", fontsize=9)

    fig.suptitle(title, fontsize=14)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _build_config(args, *, train_percent: float, seed: int) -> ToyConfig:
    cfg = ToyConfig()
    cfg.outdir = str(args.outdir)
    cfg.exp_name = f"{args.prefix}_{train_percent:g}pct_s{seed}"
    cfg.seed = int(seed)
    cfg.device = str(args.device)
    cfg.method_version = "v1.1"
    cfg.dataset_kind = str(args.dataset_kind)
    cfg.model_kind = "image_conv"
    cfg.image_size = int(args.image_size)
    cfg.image_channels = int(args.image_channels)
    cfg.image_split_seed = int(seed + args.image_split_seed_offset)
    cfg.steps = int(max(_parse_int_list(args.steps_list, allow_zero=True)))
    cfg.batch_size = int(args.batch_size)
    cfg.hidden_dim = int(args.hidden_dim)
    cfg.training_objective = str(args.training_objective)
    cfg.n_steps_path = int(args.n_steps_path)
    cfg.sigma_min = float(args.sigma_min)
    cfg.sigma_max = float(args.sigma_max)
    cfg.use_ema_eval = bool(args.use_ema_eval)
    cfg.ema_mode = str(args.ema_mode)
    cfg.ema_decay = float(args.ema_decay)
    cfg.ema_halflife_kimg = float(args.ema_halflife_kimg)
    cfg.ema_rampup_ratio = None if bool(args.disable_ema_rampup) else float(args.ema_rampup_ratio)
    cfg.auto_log_normal_params = bool(args.auto_log_normal_params)
    cfg.log_every = int(args.log_every)
    cfg.limited_data_enabled = True
    cfg.compute_fid = False
    cfg.run_checks = False
    if cfg.dataset_kind == "mnist":
        cfg.mnist_use_percent_split = True
        cfg.mnist_train_percent = float(train_percent)
        cfg.mnist_val_percent = float(args.mnist_val_percent)
    elif cfg.dataset_kind == "image_folder":
        dataset_path = str(getattr(args, "dataset_path", "")).strip()
        if not dataset_path:
            raise ValueError("--dataset-path is required when --dataset-kind=image_folder")
        dataset_val_path = str(getattr(args, "dataset_val_path", "")).strip()
        if not dataset_val_path:
            raise ValueError("--dataset-val-path is required when --dataset-kind=image_folder")
        source_train_size = _count_image_files(Path(dataset_path))
        if source_train_size <= 0:
            raise ValueError(f"No images found under --dataset-path={dataset_path}")
        source_val_size = _count_image_files(Path(dataset_val_path))
        if source_val_size <= 0:
            raise ValueError(f"No images found under --dataset-val-path={dataset_val_path}")
        resolved_train_size = max(1, int(round(float(train_percent) * float(source_train_size) / 100.0)))
        resolved_train_size = min(resolved_train_size, int(source_train_size))
        cfg.dataset_path = dataset_path
        cfg.dataset_val_path = dataset_val_path
        cfg.image_train_size = int(resolved_train_size)
        cfg.image_val_size = int(source_val_size)
        cfg.mnist_use_percent_split = False
    else:
        raise ValueError(f"Unsupported --dataset-kind={cfg.dataset_kind}")
    _apply_auto_log_normal_params(cfg)
    return cfg


def _objective_display_name(training_objective: str) -> str:
    objective = str(training_objective).strip().lower()
    if objective == "rf":
        return "Rectified Flow"
    if objective == "score":
        return "Score VE"
    return "EDM"


def _run_combo(
    *,
    args,
    train_percent: float,
    seed: int,
    checkpoint_steps: List[int],
    fid_eval_steps: List[int],
    detector_net,
    mu_ref: torch.Tensor,
    sigma_ref: torch.Tensor,
    weighted_calibration: Dict[str, Any],
) -> List[RunRow]:
    cfg = _build_config(args, train_percent=train_percent, seed=seed)
    device = pick_device(cfg.device)
    if str(device) == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    _configure_runtime(
        device=device,
        allow_tf32=not bool(args.disable_tf32),
        cudnn_benchmark=not bool(args.disable_cudnn_benchmark),
    )
    amp_dtype = _resolve_amp_dtype(device, args.amp_dtype)

    combo_name = f"{args.prefix}_{train_percent:g}pct_s{seed}"
    combo_dir = Path(args.outdir) / combo_name
    ensure_dir(str(combo_dir))
    summary_path = combo_dir / "combo_rows.json"
    run_state_path = combo_dir / "run_state.pt"

    if args.skip_existing and summary_path.is_file():
        rows_payload = json.loads(summary_path.read_text(encoding="utf-8"))
        if bool(rows_payload.get("completed", True)):
            print(f"[skip-existing] {summary_path}", flush=True)
            return [
                _run_row_from_dict(
                    row,
                    weighted_calibration=weighted_calibration,
                    train_accelerator_count=int(args.train_accelerator_count),
                )
                for row in rows_payload["rows"]
            ]

    print(f"[combo] start pct={train_percent:g} seed={seed} max_step={cfg.steps}", flush=True)
    set_seed(cfg.seed)
    dataset = build_dataset_bundle(cfg, device)
    cfg.sigma_data = dataset.estimate_sigma_data()
    sigma_levels = build_sigma_levels(cfg.sigma_min, cfg.sigma_max, cfg.n_steps_path, device=device)
    model_bundle = build_model_bundle(cfg, dataset, sigma_data=cfg.sigma_data, device=device)
    baseline = model_bundle.baseline

    optimizer = torch.optim.Adam(baseline.parameters(), lr=cfg.lr_theta)
    grad_scaler = None
    if device.type == "cuda":
        grad_scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))
    history: Dict[str, List[float]] = {"loss": [], "proxy_weighted_denoise_loss": []}
    sigma_counts = torch.zeros(sigma_levels.numel() - 1, device=sigma_levels.device, dtype=torch.long)
    ema_model = init_ema_model(baseline, cfg)

    rows: List[RunRow] = []
    checkpoint_set = set(int(step) for step in checkpoint_steps)
    fid_eval_set = set(int(step) for step in fid_eval_steps)
    run_state_signature = _build_run_state_signature(cfg=cfg, dataset=dataset, train_percent=train_percent, seed=seed)
    start_step = 0
    combo_train_elapsed_offset_sec = 0.0
    if not bool(getattr(args, "disable_resume", False)) and run_state_path.is_file():
        payload = _load_run_state(run_state_path)
        saved_signature = payload.get("signature")
        if saved_signature != run_state_signature:
            raise RuntimeError(
                "Run-state signature mismatch.\n"
                f"run_state={run_state_path}\n"
                f"current={json.dumps(run_state_signature, sort_keys=True)}\n"
                f"saved={json.dumps(saved_signature, sort_keys=True)}"
            )
        start_step = int(payload.get("step", 0))
        if start_step < 0 or start_step > int(cfg.steps):
            raise RuntimeError(
                f"Invalid saved step in run state: step={start_step} current_max={int(cfg.steps)} ({run_state_path})"
            )
        saved_checkpoint_steps = [int(v) for v in payload.get("checkpoint_steps", [])]
        if saved_checkpoint_steps != [int(v) for v in checkpoint_steps]:
            raise RuntimeError(
                "Checkpoint-step schedule mismatch.\n"
                f"run_state={run_state_path}\n"
                f"current={checkpoint_steps}\n"
                f"saved={saved_checkpoint_steps}"
            )
        saved_fid_eval_steps = [int(v) for v in payload.get("fid_eval_steps", saved_checkpoint_steps)]
        if saved_fid_eval_steps != [int(v) for v in fid_eval_steps]:
            raise RuntimeError(
                "FID-eval schedule mismatch.\n"
                f"run_state={run_state_path}\n"
                f"current={fid_eval_steps}\n"
                f"saved={saved_fid_eval_steps}"
            )
        baseline.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        _optimizer_state_to_device(optimizer, device)
        scaler_state = payload.get("grad_scaler_state_dict")
        if grad_scaler is not None and scaler_state is not None:
            grad_scaler.load_state_dict(scaler_state)
        ema_state = payload.get("ema_state_dict")
        if ema_model is not None and ema_state is not None:
            ema_model.load_state_dict(ema_state)
        history = {
            "loss": [float(v) for v in payload.get("history", {}).get("loss", [])],
            "proxy_weighted_denoise_loss": [
                float(v) for v in payload.get("history", {}).get("proxy_weighted_denoise_loss", [])
            ],
        }
        sigma_counts = torch.tensor(
            payload.get("sigma_counts", []),
            dtype=torch.long,
            device=sigma_levels.device,
        )
        rows = [
            _run_row_from_dict(
                row,
                weighted_calibration=weighted_calibration,
                train_accelerator_count=int(args.train_accelerator_count),
            )
            for row in payload.get("rows", [])
        ]
        combo_train_elapsed_offset_sec = float(payload.get("combo_train_elapsed_sec", 0.0))
        _restore_rng_state(payload["rng_state"])
        print(
            f"[resume] {run_state_path} step={start_step} rows={len(rows)}"
            f" completed={bool(payload.get('completed', False))}",
            flush=True,
        )
        if start_step >= int(cfg.steps):
            _write_combo_rows_summary(
                summary_path,
                combo_name=combo_name,
                train_percent=train_percent,
                seed=seed,
                rows=rows,
                run_state_path=run_state_path,
                completed=bool(payload.get("completed", False)),
            )
            return rows

    combo_train_t0 = time.perf_counter()
    eval_seed = int(cfg.seed + args.eval_seed_offset_metrics)
    state_save_every = max(int(getattr(args, "state_save_every", 0)), 0)

    def _emit_checkpoint_row(*, step: int, baseline_loss_final: float, baseline_loss_mean_last: float) -> None:
        eval_model = ema_model if ema_model is not None else baseline
        ckpt_path = combo_dir / "checkpoints" / f"baseline_step{step:05d}.pt"
        history_snapshot = {
            "loss": [float(v) for v in history["loss"]],
            "proxy_weighted_denoise_loss": [float(v) for v in history["proxy_weighted_denoise_loss"]],
            "sigma_counts": [int(v) for v in sigma_counts.detach().cpu().tolist()],
        }
        _save_checkpoint(
            ckpt_path,
            model=eval_model,
            raw_model=baseline,
            optimizer=optimizer,
            ema_model=ema_model,
            rng_state=_capture_rng_state(),
            step=step,
            train_percent=train_percent,
            seed=seed,
            history=history_snapshot,
            train_wall_clock_sec=float(combo_train_elapsed_offset_sec + float(time.perf_counter() - combo_train_t0)),
        )
        train_elapsed = combo_train_elapsed_offset_sec + float(time.perf_counter() - combo_train_t0)
        train_gpu_hours = float(train_elapsed) * float(max(int(args.train_accelerator_count), 0)) / 3600.0
        weighted_units = weighted_compute_units(
            n_fwd=0.0,
            n_fwd_inputgrad=0.0,
            n_fwd_parambackward=float(step),
            calibration=weighted_calibration,
        )
        fid_eval_selected = bool(step in fid_eval_set)
        fid_value: Optional[float] = None
        fid_elapsed: Optional[float] = None
        fid_evaluated = False
        fid_missing_reason = ""
        fid_source = "not_evaluated"
        if fid_eval_selected:
            if detector_net is None or mu_ref is None or sigma_ref is None:
                raise RuntimeError("FID evaluation requested but detector/ref stats were not initialized.")
            baseline_was_training = baseline.training
            baseline.eval()
            if ema_model is not None:
                ema_model.eval()
            t_fid = time.perf_counter()
            fid_value = _run_with_scoped_seed(
                eval_seed,
                lambda: _compute_fid_for_model(
                    denoiser=eval_model,
                    sigma_levels=sigma_levels,
                    dataset=dataset,
                    detector_net=detector_net,
                    mu_ref=mu_ref,
                    sigma_ref=sigma_ref,
                    num_images=int(args.fid_samples),
                    gen_batch=int(args.gen_batch),
                    device=device,
                    amp_dtype=amp_dtype,
                ),
            )
            fid_elapsed = float(time.perf_counter() - t_fid)
            fid_evaluated = math.isfinite(_safe_float(fid_value))
            fid_missing_reason = "" if fid_evaluated else "selected_but_missing_or_failed"
            fid_source = "in_run_metrics" if fid_evaluated else "missing_in_run_metrics"
            if baseline_was_training:
                baseline.train()
        else:
            fid_missing_reason = "not_selected_by_schedule"

        row = RunRow(
            train_percent=float(train_percent),
            seed=int(seed),
            step=int(step),
            images_shown_m=float(step * cfg.batch_size) / 1_000_000.0,
            exp_name=combo_name,
            exp_dir=str(combo_dir),
            checkpoint_path=str(ckpt_path),
            train_subset_size=int(dataset.metadata.get("train_subset_size_resolved") or 0),
            val_subset_size=int(dataset.metadata.get("val_subset_size_resolved") or 0),
            baseline_fid=None if fid_value is None else float(fid_value),
            baseline_loss_final=float(baseline_loss_final),
            baseline_loss_mean_last=float(baseline_loss_mean_last),
            train_elapsed_sec=train_elapsed,
            train_wall_clock_sec=float(train_elapsed),
            train_gpu_hours=float(train_gpu_hours),
            weighted_compute_units=float(weighted_units) if weighted_units is not None else float("nan"),
            batch_equiv_denoiser_evals=float(step),
            fid_elapsed_sec=None if fid_elapsed is None else float(fid_elapsed),
            fid_eval_selected=bool(fid_eval_selected),
            fid_evaluated=bool(fid_evaluated),
            fid_missing_reason=str(fid_missing_reason),
            fid_source=str(fid_source),
        )
        rows.append(row)
        print(
            "[row]"
            f" pct={train_percent:g} seed={seed} step={step}"
            f" mimg={_fmt(row.images_shown_m, 3)}"
            f" fid={_fmt(_safe_float(row.baseline_fid), 3)}"
            f" loss={_fmt(row.baseline_loss_final, 5)}"
            f" train_elapsed={_fmt(row.train_elapsed_sec, 1)}s"
            f" weighted={_fmt(row.weighted_compute_units, 1)}"
            f" fid_selected={str(row.fid_eval_selected).lower()}"
            f" fid_elapsed={_fmt(_safe_float(row.fid_elapsed_sec), 1)}s",
            flush=True,
        )
        _write_combo_rows_summary(
            summary_path,
            combo_name=combo_name,
            train_percent=train_percent,
            seed=seed,
            rows=rows,
            run_state_path=run_state_path,
            completed=False,
        )

    if start_step == 0 and 0 in checkpoint_set and not any(int(row.step) == 0 for row in rows):
        _emit_checkpoint_row(step=0, baseline_loss_final=float("nan"), baseline_loss_mean_last=float("nan"))

    for step in range(start_step + 1, cfg.steps + 1):
        x0 = sample_train_batch(
            cfg,
            centers=dataset.centers,
            train_pool=dataset.train_pool,
            sample_train_batch_fn=dataset.sample_train_batch,
            sample_population_batch_fn=dataset.sample_population_batch,
        )
        if cfg.use_log_normal_sigma_sampling:
            indices = sample_target_indices_log_normal(cfg.batch_size, sigma_levels, p_mean=cfg.p_mean, p_std=cfg.p_std)
        else:
            indices = sample_target_indices(cfg.batch_size, sigma_levels)
        sigma_counts += torch.bincount(indices - 1, minlength=sigma_counts.numel())
        sigma = sigma_levels[indices]
        x_noisy = build_training_state(
            cfg=cfg,
            x_clean=x0,
            sigma=sigma,
            sample_terminal_batch_fn=dataset.sample_terminal_batch,
        )

        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, amp_dtype):
            loss = compute_training_loss(cfg, baseline, x_noisy, x0, sigma)
        if has_nan_or_inf(loss):
            raise RuntimeError("NaN/Inf detected in baseline loss.")
        if grad_scaler is not None and grad_scaler.is_enabled():
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            loss.backward()
            optimizer.step()

        update_ema_model(
            ema_model,
            baseline,
            cfg,
            cur_nimg=float((step - 1) * int(cfg.batch_size)),
            batch_size=int(cfg.batch_size),
        )

        history["loss"].append(scalarize(loss))
        if str(cfg.training_objective).lower() == "edm":
            proxy_loss = loss
        else:
            with torch.no_grad():
                proxy_loss = weighted_denoise_loss(baseline, x_noisy, x0, sigma, cfg.sigma_data)
        history["proxy_weighted_denoise_loss"].append(scalarize(proxy_loss))

        if step % cfg.log_every == 0:
            print(f"[baseline] pct={train_percent:g} seed={seed} step={step:05d} loss={loss.item():.6f}", flush=True)

        if step in checkpoint_set:
            _emit_checkpoint_row(
                step=step,
                baseline_loss_final=float(history["loss"][-1]),
                baseline_loss_mean_last=float(
                    sum(history["loss"][-min(200, len(history["loss"])) :]) / min(200, len(history["loss"]))
                ),
            )

        should_save_state = (
            step == int(cfg.steps)
            or (state_save_every > 0 and step % state_save_every == 0)
            or step in checkpoint_set
        )
        if should_save_state:
            combo_train_elapsed_sec = combo_train_elapsed_offset_sec + float(time.perf_counter() - combo_train_t0)
            _save_run_state(
                run_state_path,
                signature=run_state_signature,
                step=step,
                planned_max_step=int(cfg.steps),
                checkpoint_steps=checkpoint_steps,
                fid_eval_steps=fid_eval_steps,
                model=baseline,
                optimizer=optimizer,
                grad_scaler=grad_scaler,
                ema_model=ema_model,
                history=history,
                sigma_counts=sigma_counts,
                rows=rows,
                combo_train_elapsed_sec=combo_train_elapsed_sec,
                completed=(step == int(cfg.steps)),
            )

    _write_combo_rows_summary(
        summary_path,
        combo_name=combo_name,
        train_percent=train_percent,
        seed=seed,
        rows=rows,
        run_state_path=run_state_path,
        completed=True,
    )
    return rows


def build_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Efficient checkpointed baseline-EDM convergence sweep.")
    parser.add_argument("--outdir", type=Path, default=Path("toy_outputs/mnist_convergence_checkpointed"))
    parser.add_argument("--prefix", type=str, default="mnist_baseline_curve_ckpt")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--train-percents", type=str, default="5,10,20,50,100")
    parser.add_argument("--steps-list", type=str, default="1000,2000,4000,8000,12000,16000,20000")
    parser.add_argument("--mimg-list", type=str, default="")
    parser.add_argument("--fid-eval-steps-list", type=str, default="")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--disable-resume", action="store_true")
    parser.add_argument("--state-save-every", type=int, default=500)
    parser.add_argument("--skip-plot", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dataset-kind", type=str, default="mnist", choices=["mnist", "image_folder"])
    parser.add_argument("--dataset-path", type=str, default="")
    parser.add_argument("--dataset-val-path", type=str, default="")
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--image-channels", type=int, default=1)
    parser.add_argument("--mnist-val-percent", type=float, default=100.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--training-objective", type=str, default="edm", choices=["edm", "score", "rf"])
    parser.add_argument("--n-steps-path", type=int, default=24)
    parser.add_argument("--sigma-min", type=float, default=0.01)
    parser.add_argument("--sigma-max", type=float, default=80.0)
    parser.add_argument("--auto-log-normal-params", action="store_true")
    parser.add_argument("--use-ema-eval", action="store_true")
    parser.add_argument("--ema-mode", type=str, default=ToyConfig.ema_mode, choices=["official", "fixed"])
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--ema-halflife-kimg", type=float, default=ToyConfig.ema_halflife_kimg)
    parser.add_argument("--ema-rampup-ratio", type=float, default=ToyConfig.ema_rampup_ratio)
    parser.add_argument("--disable-ema-rampup", action="store_true")
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--image-split-seed-offset", type=int, default=0)
    parser.add_argument("--eval-seed-offset-metrics", type=int, default=20000)
    parser.add_argument("--fid-samples", type=int, default=2000)
    parser.add_argument("--gen-batch", type=int, default=2048)
    parser.add_argument("--fid-ref-path", type=Path, default=None)
    parser.add_argument("--fid-ref-split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--fid-ref-subset-percent", type=float, default=100.0)
    parser.add_argument(
        "--fid-ref-subset-sampling",
        type=str,
        default="stratified",
        choices=["first", "global", "stratified"],
    )
    parser.add_argument("--fid-ref-subset-seed", type=int, default=0)
    parser.add_argument("--fid-ref-max-images", type=int, default=5000)
    parser.add_argument("--force-ref-refresh", action="store_true")
    parser.add_argument("--amp-dtype", type=str, default="auto", choices=["auto", "off", "bfloat16", "float16"])
    parser.add_argument("--disable-tf32", action="store_true")
    parser.add_argument("--disable-cudnn-benchmark", action="store_true")
    parser.add_argument("--weighted-compute-calibration-path", type=str, default="")
    parser.add_argument("--weighted-inputgrad-alpha", type=float, default=0.0)
    parser.add_argument("--weighted-parambackward-beta", type=float, default=0.0)
    parser.add_argument("--train-accelerator-count", type=int, default=1)

    parser.add_argument("--threshold-pct", type=float, default=5.0)
    parser.add_argument("--overfit-pct", type=float, default=10.0)
    parser.add_argument("--overfit-patience", type=int, default=2)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if _APPLIED_PROCESS_TITLE is None:
        apply_process_title(build_process_title("wdiff", "baseline", args.prefix))
    seeds = _parse_int_list(args.seeds, allow_zero=True)
    train_percents = _parse_float_list(args.train_percents)
    if str(args.mimg_list).strip():
        checkpoint_steps = _checkpoint_steps_from_mimg_list(
            _parse_float_list(args.mimg_list),
            batch_size=int(args.batch_size),
        )
        args.steps_list = ",".join(str(step) for step in checkpoint_steps)
    else:
        checkpoint_steps = _parse_int_list(args.steps_list, allow_zero=True)
    fid_eval_steps_arg = str(args.fid_eval_steps_list).strip().lower()
    if fid_eval_steps_arg == "none":
        fid_eval_steps = []
    elif str(args.fid_eval_steps_list).strip():
        fid_eval_steps = _parse_int_list(args.fid_eval_steps_list, allow_zero=True)
    else:
        fid_eval_steps = list(checkpoint_steps)
    invalid_fid_eval_steps = sorted(set(int(step) for step in fid_eval_steps) - set(int(step) for step in checkpoint_steps))
    if invalid_fid_eval_steps:
        raise SystemExit(
            "[ERROR] --fid-eval-steps-list must be a subset of --steps-list / resolved checkpoint steps. "
            f"Invalid steps: {invalid_fid_eval_steps}"
        )
    repo_root = _repo_root()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if args.require_cuda and not torch.cuda.is_available():
        raise SystemExit("[ERROR] --require-cuda is set but CUDA is unavailable.")

    if str(args.dataset_kind) == "mnist":
        fid_ref_policy_name = default_mnist_fid_policy_name(
            split=str(args.fid_ref_split),
            image_size=int(args.image_size),
            subset_percent=float(args.fid_ref_subset_percent),
            subset_sampling=str(args.fid_ref_subset_sampling),
            subset_seed=int(args.fid_ref_subset_seed),
            max_images=int(args.fid_ref_max_images),
        )
        fid_ref_dir = (outdir / "_fid_refs").resolve()
        ref_npz = fid_ref_dir / f"{fid_ref_policy_name}.npz"
        fid_ref_meta = build_mnist_fid_reference(
            split=str(args.fid_ref_split),
            image_size=int(args.image_size),
            subset_percent=float(args.fid_ref_subset_percent),
            subset_sampling=str(args.fid_ref_subset_sampling),
            subset_seed=int(args.fid_ref_subset_seed),
            max_images=int(args.fid_ref_max_images),
            dest=ref_npz,
            images_dir=fid_ref_dir / f"{fid_ref_policy_name}_images",
            policy_name=fid_ref_policy_name,
            force=bool(args.force_ref_refresh),
        )
    else:
        ref_npz = Path(args.fid_ref_path).resolve() if args.fid_ref_path else None
        if ref_npz is None or not ref_npz.is_file():
            raise SystemExit("[ERROR] --fid-ref-path is required and must exist for --dataset-kind=image_folder.")
        fid_ref_meta = {
            "source": "explicit_path",
            "dataset_kind": "image_folder",
            "dataset_path": str(Path(args.dataset_path).resolve()) if args.dataset_path else "",
            "dataset_val_path": str(Path(args.dataset_val_path).resolve()) if args.dataset_val_path else "",
            "fid_ref_path": str(ref_npz),
        }

    device = pick_device(args.device)
    _configure_runtime(
        device=device,
        allow_tf32=not bool(args.disable_tf32),
        cudnn_benchmark=not bool(args.disable_cudnn_benchmark),
    )
    amp_dtype = _resolve_amp_dtype(device, args.amp_dtype)
    weighted_calibration = _resolve_weighted_calibration(args)
    print(
        "[runtime]"
        f" device={device}"
        f" amp={_format_amp_dtype(amp_dtype)}"
        f" tf32={str(not bool(args.disable_tf32)).lower()}"
        f" cudnn_benchmark={str(not bool(args.disable_cudnn_benchmark)).lower()}"
        f" batch_size={int(args.batch_size)}"
        f" gen_batch={int(args.gen_batch)}",
        flush=True,
    )
    print(
        "[runtime]"
        f" weighted_compute_available={weighted_calibration['available']}"
        f" source={weighted_calibration['source']}"
        f" alpha={weighted_calibration['inputgrad_alpha']}"
        f" beta={weighted_calibration['parambackward_beta']}"
        f" train_accelerator_count={int(args.train_accelerator_count)}",
        flush=True,
    )
    if str(args.mimg_list).strip():
        print(f"[runtime] target_mimg={str(args.mimg_list).strip()} -> checkpoint_steps={checkpoint_steps}", flush=True)
    print(f"[runtime] fid_eval_steps={fid_eval_steps}", flush=True)
    detector_net = None
    mu_ref = None
    sigma_ref = None
    if fid_eval_steps:
        detector_net = _load_detector(device)
        mu_ref_np, sigma_ref_np = _load_ref_stats(ref_npz)
        mu_ref = torch.as_tensor(mu_ref_np, dtype=torch.float64, device=device)
        sigma_ref = torch.as_tensor(sigma_ref_np, dtype=torch.float64, device=device)

    all_rows: List[RunRow] = []
    for train_percent in train_percents:
        for seed in seeds:
            rows = _run_combo(
                args=args,
                train_percent=train_percent,
                seed=seed,
                checkpoint_steps=checkpoint_steps,
                fid_eval_steps=fid_eval_steps,
                detector_net=detector_net,
                mu_ref=mu_ref,
                sigma_ref=sigma_ref,
                weighted_calibration=weighted_calibration,
            )
            all_rows.extend(rows)

    all_rows.sort(key=lambda row: (row.train_percent, row.seed, row.step))
    aggregates = _aggregate(all_rows)
    thresholds = _extract_thresholds(
        aggregates,
        threshold_pct=float(args.threshold_pct),
        overfit_pct=float(args.overfit_pct),
        overfit_patience=int(args.overfit_patience),
    )

    runs_csv = outdir / f"{args.prefix}_runs.csv"
    agg_csv = outdir / f"{args.prefix}_aggregate.csv"
    fig_path = outdir / f"{args.prefix}_curves.png"
    summary_path = outdir / f"{args.prefix}_summary.json"

    _write_csv(runs_csv, (row.to_dict() for row in all_rows))
    _write_csv(agg_csv, (row.to_dict() for row in aggregates))

    plot_warning = None
    plot_written = False
    if not args.skip_plot:
        try:
            _plot_curves(
                rows=aggregates,
                thresholds=thresholds,
                out_path=fig_path,
                title=(
                    f"{str(args.dataset_kind).replace('_', ' ').title()} Baseline "
                    f"{_objective_display_name(args.training_objective)} Convergence | "
                    f"threshold={args.threshold_pct:.1f}% of best median FID"
                ),
                dataset_label=str(args.dataset_kind).replace("_", " ").title(),
            )
            plot_written = True
        except RuntimeError as exc:
            plot_warning = str(exc)
            print(f"[warn] {plot_warning}", flush=True)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "name": f"{str(args.dataset_kind)}_checkpointed_baseline_convergence",
            "family": "from_scratch_curve",
            "description": (
                f"Train baseline {_objective_display_name(args.training_objective)} once per subset+seed "
                "and evaluate intermediate checkpoints."
            ),
            "primary_metric": "train_wall_clock_sec",
            "secondary_metric": "weighted_compute_units",
            "legacy_metric": "batch_equiv_denoiser_evals",
            "dataset_kind": str(args.dataset_kind),
            "dataset_path": str(Path(args.dataset_path).resolve()) if args.dataset_path else None,
            "dataset_val_path": str(Path(args.dataset_val_path).resolve()) if args.dataset_val_path else None,
            "seeds": seeds,
            "train_percents": train_percents,
            "steps_list": checkpoint_steps,
            "fid_eval_steps_list": fid_eval_steps,
            "batch_size": int(args.batch_size),
            "image_size": int(args.image_size),
            "training_objective": str(args.training_objective),
            "sigma_min": float(args.sigma_min),
            "sigma_max": float(args.sigma_max),
            "auto_log_normal_params": bool(args.auto_log_normal_params),
            **ema_config_dict(args),
            "fid_samples": int(args.fid_samples),
            "gen_batch": int(args.gen_batch),
            "fid_reference": fid_ref_meta,
            "threshold_pct": float(args.threshold_pct),
            "overfit_pct": float(args.overfit_pct),
            "overfit_patience": int(args.overfit_patience),
            "train_accelerator_count": int(args.train_accelerator_count),
            "weighted_compute_calibration": weighted_calibration,
        },
        "thresholds": thresholds,
        "artifacts": {
            "runs_csv": str(runs_csv),
            "aggregate_csv": str(agg_csv),
            "curves_png": str(fig_path) if plot_written else None,
        },
        "plot": {
            "requested": bool(not args.skip_plot),
            "written": bool(plot_written),
            "warning": plot_warning,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[thresholds]")
    for train_percent in train_percents:
        info = thresholds[str(train_percent)]
        print(
            f"  {train_percent:g}%:"
            f" best_fid={_fmt(_safe_float(info['best_fid_median']), 3)}"
            f" @ step={info['best_step']} ({_fmt(_safe_float(info['best_images_shown_m']), 3)}M)"
            f" | threshold={_fmt(_safe_float(info['threshold_fid_ceiling']), 3)}"
            f" -> step={info['threshold_step']} ({_fmt(_safe_float(info['threshold_images_shown_m']), 3)}M)",
            flush=True,
        )

    print("[done] saved:")
    print(f"  - {runs_csv}")
    print(f"  - {agg_csv}")
    if plot_written:
        print(f"  - {fig_path}")
    print(f"  - {summary_path}")


if __name__ == "__main__":
    main()
