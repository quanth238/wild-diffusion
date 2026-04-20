#!/usr/bin/env python3
import argparse
import copy
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
from toy.shared.objective import build_rectified_flow_state, weighted_denoise_loss, weighted_rectified_flow_loss  # noqa: E402
from toy.shared.reverse import generated_data_path_index_from_denoiser, sample_reverse_paths  # noqa: E402
from toy.shared.runtime import autocast_context, configure_runtime, format_amp_dtype, resolve_amp_dtype  # noqa: E402
from toy.shared.sigma import (  # noqa: E402
    build_rf_stage_time_quantile_levels,
    build_rf_time_quantile_levels,
    build_sigma_levels,
    build_sigma_levels_from_warmup_quantiles,
    resolve_rf_eval_n_steps_path,
    resolve_rf_teacher_n_steps_path,
    sample_log_sigma_stratified_quantile_ladder,
    sample_sigmas_log_normal,
    sample_target_indices,
)
from toy.shared.trainer_common import generate_reflow_pairs  # noqa: E402
from toy.utils import ensure_dir, pick_device, set_seed  # noqa: E402


_APPLIED_PROCESS_TITLE = apply_process_title()
DETERMINISTIC_MIDPOINT_QUANTILE_LADDER = "deterministic_midpoint_quantile"
STOCHASTIC_STRATIFIED_QUANTILE_LADDER = "stochastic_stratified_quantile"

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
DEFAULT_FID_SAMPLES = int(ToyConfig.fid_samples)
DEFAULT_EDM_CLEAN_PROBE_BATCHES = 8


@dataclass
class EvalContext:
    key: Tuple[object, ...]
    cfg: ToyConfig
    device: torch.device
    amp_dtype: Optional[torch.dtype]
    dataset: object
    sigma_levels: torch.Tensor
    probe_sigma_levels: torch.Tensor
    baseline_model: torch.nn.Module
    robust_model: torch.nn.Module


def _canonical_method_name(value: object) -> str:
    return str(value or "").strip().lower()


def _is_stochastic_cdro_eval(cfg: ToyConfig, method_name: str) -> bool:
    return (
        _canonical_method_name(method_name) == "cdro"
        and str(getattr(cfg, "training_objective", "edm")).strip().lower() != "rf"
        and bool(getattr(cfg, "cdro_eval_stochastic_ladders", False))
        and str(
            getattr(cfg, "cdro_edm_ladder_mode", DETERMINISTIC_MIDPOINT_QUANTILE_LADDER)
        ).strip().lower()
        == STOCHASTIC_STRATIFIED_QUANTILE_LADDER
    )


def _build_eval_sigma_levels(cfg: ToyConfig, *, device: torch.device, method_name: str) -> torch.Tensor:
    if str(cfg.training_objective).strip().lower() == "rf":
        return build_rf_time_quantile_levels(
            float(cfg.sigma_max),
            int(resolve_rf_eval_n_steps_path(cfg)),
            device=device,
            distribution=str(getattr(cfg, "rf_reflow_t_distribution", "u_shaped")),
        )
    return build_sigma_levels(float(cfg.sigma_min), float(cfg.sigma_max), int(cfg.n_steps_path), device=device)


def _sample_eval_sigma_levels(cfg: ToyConfig, method_name: str, sigma_levels: torch.Tensor) -> torch.Tensor:
    if not _is_stochastic_cdro_eval(cfg, method_name):
        return sigma_levels
    return sample_log_sigma_stratified_quantile_ladder(
        float(cfg.sigma_min),
        float(cfg.sigma_max),
        int(cfg.n_steps_path),
        p_mean=float(getattr(cfg, "p_mean", -1.2)),
        p_std=float(getattr(cfg, "p_std", 1.2)),
        device=sigma_levels.device,
        dtype=sigma_levels.dtype,
    ).sigma_levels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-evaluate existing three-method trajectory checkpoints with in-memory batched FID "
            "and an objective-aware clean probe directly from saved checkpoints, then regenerate per-case "
            "plots from the refreshed combined CSV."
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
    parser.add_argument("--image-train-size", type=int, default=400)
    parser.add_argument("--image-val-size", type=int, default=2000)
    parser.add_argument("--image-split-seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--baseline-train-batch-gpu", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument(
        "--image-backbone",
        type=str,
        default=ToyConfig.image_backbone,
        choices=["conv", "songunet", "ddpmpp"],
    )
    parser.add_argument("--eval-samples", type=int, default=2000)
    parser.add_argument("--fid-samples", type=int, default=DEFAULT_FID_SAMPLES)
    parser.add_argument("--fid-batch-size", type=int, default=DEFAULT_FID_BATCH_SIZE)
    parser.add_argument("--debug-eval-batch", type=int, default=64)
    parser.add_argument("--debug-terminal-step", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--disable-edm-clean-probe", action="store_true")
    parser.add_argument("--edm-clean-probe-split", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--edm-clean-probe-batches", type=int, default=DEFAULT_EDM_CLEAN_PROBE_BATCHES)
    parser.add_argument("--edm-clean-probe-batch-size", type=int, default=256)
    parser.add_argument("--n-steps-path-default", type=int, default=32)
    parser.add_argument("--rf-teacher-n-steps-path-default", type=int, default=ToyConfig.rf_teacher_n_steps_path)
    parser.add_argument("--rf-eval-n-steps-path-default", type=int, default=ToyConfig.rf_eval_n_steps_path)
    parser.add_argument("--cdro-eval-stochastic-ladders", action="store_true")
    parser.add_argument("--respect-row-n-steps-path", action="store_true")
    parser.add_argument("--sigma-min", type=float, default=0.002)
    parser.add_argument("--sigma-max", type=float, default=2.0)
    parser.add_argument("--metrics-eval-seed-offset", type=int, default=ToyConfig.eval_seed_offset_metrics)
    parser.add_argument("--train-percent-label", type=str, default="5%")
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
    if row_origin == "trajectory_warmup_phase":
        return True
    if robust_method != "baseline":
        return False
    return row_origin != "trajectory_robust_phase"


def _checkpoint_branch(row: Dict[str, str]) -> str:
    return "baseline" if _is_baseline_style_row(row) else "robust"


def _metrics_fid_key(row: Dict[str, str]) -> str:
    return "baseline_fid" if _is_baseline_style_row(row) else "robust_fid"


def _objective_clean_probe_key_from_objective(training_objective: object) -> str:
    objective = str(training_objective or "").strip().lower()
    return "rf_clean_probe" if objective == "rf" else "edm_clean_probe"


def _is_shared_edm_warmup_row_for_rf(row: Dict[str, str]) -> bool:
    training_objective = str(row.get("training_objective", "")).strip().lower()
    row_origin = str(row.get("row_origin", "")).strip().lower()
    if training_objective != "rf" or row_origin != "trajectory_warmup_phase":
        return False
    explicit_source_objective = str(row.get("shared_warmup_checkpoint_source_training_objective", "")).strip().lower()
    explicit_eval_objective = str(row.get("shared_warmup_checkpoint_eval_training_objective", "")).strip().lower()
    if explicit_source_objective == "edm" or explicit_eval_objective == "edm":
        return True
    return _safe_bool(row.get("warmup_only"), default=False)


def _effective_eval_training_objective_from_row(row: Dict[str, str]) -> str:
    if _is_shared_edm_warmup_row_for_rf(row):
        return "edm"
    return str(row.get("training_objective", "edm"))


def _objective_clean_probe_key(row: Dict[str, str]) -> str:
    return _objective_clean_probe_key_from_objective(_effective_eval_training_objective_from_row(row))


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
    return states[:, generated_data_path_index_from_denoiser(denoiser)]


@torch.no_grad()
def _compute_fid_for_model(
    *,
    denoiser: torch.nn.Module,
    cfg: ToyConfig,
    method_name: str,
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
        sigma_levels_batch = _sample_eval_sigma_levels(cfg, method_name, sigma_levels)
        images = _sample_reverse_x0(
            denoiser=denoiser,
            sigma_levels=sigma_levels_batch,
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
                f"[fid-only] progress images={n_done}/{num_images} batch={cur} "
                f"sigma_max={float(sigma_levels_batch[-1].item()):.6f} elapsed_sec={elapsed:.2f}\n"
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


def _baseline_state_dict_from_payload(payload: Dict, ckpt_path: str) -> Dict[str, torch.Tensor]:
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


def _load_baseline_state_dict(ckpt_path: str) -> Dict[str, torch.Tensor]:
    payload = _load_checkpoint_payload(ckpt_path)
    return _baseline_state_dict_from_payload(payload, ckpt_path)


def _load_robust_state_dict(
    ckpt_path: str,
    method_name: str,
    *,
    prefer_ema: bool = False,
) -> Tuple[Dict[str, torch.Tensor], str]:
    payload = _load_checkpoint_payload(ckpt_path)
    return _robust_state_dict_from_payload(payload, ckpt_path, method_name, prefer_ema=prefer_ema)


def _robust_state_dict_from_payload(
    payload: Dict,
    ckpt_path: str,
    method_name: str,
    *,
    prefer_ema: bool = False,
) -> Tuple[Dict[str, torch.Tensor], str]:
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
    cfg.image_backbone = str(args.image_backbone)
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
    cfg.rf_teacher_n_steps_path = int(args.rf_teacher_n_steps_path_default)
    cfg.rf_eval_n_steps_path = int(args.rf_eval_n_steps_path_default)
    cfg.cdro_eval_stochastic_ladders = bool(args.cdro_eval_stochastic_ladders)
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
        "auto_log_normal_params",
        "batch_size",
        "cdro_edm_ladder_mode",
        "cdro_eval_stochastic_ladders",
        "cudnn_benchmark",
        "dataset_kind",
        "dataset_path",
        "dataset_val_path",
        "eval_seed_offset_metrics",
        "fid_ref_path",
        "hidden_dim",
        "image_backbone",
        "image_channels",
        "image_size",
        "image_split_seed",
        "image_train_size",
        "image_val_size",
        "limited_data_enabled",
        "model_kind",
        "n_steps_path",
        "p_mean",
        "p_std",
        "rf_baseline_mode",
        "rf_cdro_pair_source",
        "rf_eval_n_steps_path",
        "rf_loss",
        "rf_pseudo_huber_delta",
        "rf_reflow_t_distribution",
        "rf_stage1_fraction",
        "rf_teacher_n_steps_path",
        "sigma_data",
        "sigma_max",
        "sigma_min",
        "training_objective",
        "use_log_normal_sigma_sampling",
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
    if not bool(args.respect_row_n_steps_path):
        cfg.n_steps_path = int(args.n_steps_path_default)
    effective_training_objective = _effective_eval_training_objective_from_row(row)
    if effective_training_objective:
        cfg.training_objective = str(effective_training_objective)
    if bool(args.respect_row_n_steps_path) and row.get("n_steps_path"):
        cfg.n_steps_path = _safe_int(row.get("n_steps_path"), cfg.n_steps_path)
    if row.get("rf_teacher_n_steps_path"):
        cfg.rf_teacher_n_steps_path = _safe_int(row.get("rf_teacher_n_steps_path"), cfg.rf_teacher_n_steps_path)
    if row.get("rf_eval_n_steps_path"):
        cfg.rf_eval_n_steps_path = _safe_int(row.get("rf_eval_n_steps_path"), cfg.rf_eval_n_steps_path)
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


def _context_key_for_cfg(cfg: ToyConfig, device: torch.device, method_name: str) -> Tuple[object, ...]:
    return (
        str(device),
        _canonical_method_name(method_name),
        str(cfg.amp_dtype),
        str(cfg.dataset_kind),
        str(cfg.model_kind),
        str(getattr(cfg, "image_backbone", ToyConfig.image_backbone)),
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
        int(resolve_rf_teacher_n_steps_path(cfg)),
        int(resolve_rf_eval_n_steps_path(cfg)),
        float(cfg.sigma_min),
        float(cfg.sigma_max),
        float(getattr(cfg, "p_mean", -1.2)),
        float(getattr(cfg, "p_std", 1.2)),
        float(getattr(cfg, "sigma_data", -1.0)),
        bool(getattr(cfg, "use_log_normal_sigma_sampling", True)),
        str(getattr(cfg, "cdro_edm_ladder_mode", DETERMINISTIC_MIDPOINT_QUANTILE_LADDER)),
        bool(getattr(cfg, "cdro_eval_stochastic_ladders", False)),
        str(getattr(cfg, "rf_reflow_t_distribution", "u_shaped")),
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
    method_name = _canonical_method_name(row.get("method", row.get("robust_method", "")))
    key = _context_key_for_cfg(cfg, device, method_name)
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
    sigma_levels = _build_eval_sigma_levels(cfg, device=device, method_name=method_name)
    probe_sigma_levels = build_sigma_levels(
        float(cfg.sigma_min),
        float(cfg.sigma_max),
        int(cfg.n_steps_path),
        device=device,
    )
    context = EvalContext(
        key=key,
        cfg=cfg,
        device=device,
        amp_dtype=amp_dtype,
        dataset=dataset,
        sigma_levels=sigma_levels,
        probe_sigma_levels=probe_sigma_levels,
        baseline_model=model_bundle.baseline,
        robust_model=model_bundle.robust,
    )
    cache[key] = context
    return context


def _extract_probe_metrics_from_summary(
    objective_debug: Dict[str, object],
    *,
    probe_key: str,
) -> Dict[str, object]:
    probe_summary = objective_debug.get(probe_key)
    probe_value = None
    probe_images = 0
    probe_seed = None
    probe_split = ""
    probe_batches = 0
    probe_batch_size = 0
    probe_evaluated = False
    probe_missing_reason = "missing_from_metrics"
    probe_source = "missing_in_reeval_metrics"
    probe_stage = ""
    probe_t_distribution = ""
    probe_pair_source = ""
    if isinstance(probe_summary, dict):
        probe_value = _safe_float(probe_summary.get("mean"))
        probe_images = _safe_int(probe_summary.get("num_images"), 0)
        probe_seed = _safe_int(probe_summary.get("seed"), 0)
        probe_split = str(probe_summary.get("split", ""))
        probe_batches = _safe_int(probe_summary.get("num_batches"), 0)
        probe_batch_size = _safe_int(probe_summary.get("batch_size"), 0)
        probe_missing_reason = str(probe_summary.get("missing_reason", "")).strip()
        probe_evaluated = probe_value is not None
        probe_source = "reevaluated_from_checkpoint" if probe_evaluated else "not_evaluated"
        probe_stage = str(probe_summary.get("stage_name", ""))
        probe_t_distribution = str(probe_summary.get("t_distribution", ""))
        probe_pair_source = str(probe_summary.get("pair_source", ""))
        if not probe_missing_reason and not probe_evaluated:
            probe_missing_reason = "disabled_or_unsupported"
    return {
        probe_key: probe_value,
        f"{probe_key}_images": int(probe_images),
        f"{probe_key}_seed": probe_seed,
        f"{probe_key}_split": probe_split,
        f"{probe_key}_batches": int(probe_batches),
        f"{probe_key}_batch_size": int(probe_batch_size),
        f"{probe_key}_evaluated": bool(probe_evaluated),
        f"{probe_key}_missing_reason": str(probe_missing_reason),
        f"{probe_key}_source": str(probe_source),
        f"{probe_key}_stage": str(probe_stage),
        f"{probe_key}_t_distribution": str(probe_t_distribution),
        f"{probe_key}_pair_source": str(probe_pair_source),
    }


def _extract_cached_reeval(metrics_path: str, row: Dict[str, str]) -> Dict[str, object]:
    payload = _load_json(metrics_path)
    sample_quality = payload.get("metrics", {}).get("sample_quality_debug", {})
    objective_debug = payload.get("metrics", {}).get("objective_debug", {})
    key = _metrics_fid_key(row)
    fid_value = _safe_float(sample_quality.get(key))
    if fid_value is None:
        raise RuntimeError(f"Missing {key} in reevaluated metrics: {metrics_path}")
    edm_probe = _extract_probe_metrics_from_summary(objective_debug, probe_key="edm_clean_probe")
    rf_probe = _extract_probe_metrics_from_summary(objective_debug, probe_key="rf_clean_probe")
    objective_probe_key = _objective_clean_probe_key(row)
    objective_probe_value = edm_probe if objective_probe_key == "edm_clean_probe" else rf_probe
    return {
        "fid": float(fid_value),
        **edm_probe,
        **rf_probe,
        "objective_clean_probe": objective_probe_value.get(objective_probe_key),
        "objective_clean_probe_kind": objective_probe_key,
        "objective_clean_probe_images": int(objective_probe_value.get(f"{objective_probe_key}_images", 0)),
        "objective_clean_probe_seed": objective_probe_value.get(f"{objective_probe_key}_seed"),
        "objective_clean_probe_split": str(objective_probe_value.get(f"{objective_probe_key}_split", "")),
        "objective_clean_probe_batches": int(objective_probe_value.get(f"{objective_probe_key}_batches", 0)),
        "objective_clean_probe_batch_size": int(objective_probe_value.get(f"{objective_probe_key}_batch_size", 0)),
        "objective_clean_probe_evaluated": bool(
            objective_probe_value.get(f"{objective_probe_key}_evaluated", False)
        ),
        "objective_clean_probe_missing_reason": str(
            objective_probe_value.get(f"{objective_probe_key}_missing_reason", "")
        ),
        "objective_clean_probe_source": str(objective_probe_value.get(f"{objective_probe_key}_source", "")),
        "objective_clean_probe_stage": str(objective_probe_value.get(f"{objective_probe_key}_stage", "")),
        "objective_clean_probe_t_distribution": str(
            objective_probe_value.get(f"{objective_probe_key}_t_distribution", "")
        ),
        "objective_clean_probe_pair_source": str(
            objective_probe_value.get(f"{objective_probe_key}_pair_source", "")
        ),
    }


def _maybe_extract_cached_reeval(metrics_path: str, row: Dict[str, str]) -> Optional[Dict[str, object]]:
    if not os.path.isfile(metrics_path):
        return None
    try:
        return _extract_cached_reeval(metrics_path, row)
    except Exception:
        return None


def _cached_reeval_matches_request(
    metrics_path: str,
    *,
    args: argparse.Namespace,
    row: Dict[str, str],
    cfg: ToyConfig,
) -> bool:
    if not os.path.isfile(metrics_path):
        return False
    try:
        payload = _load_json(metrics_path)
    except Exception:
        return False

    config = payload.get("config", {})
    metrics = payload.get("metrics", {})
    sample_quality = metrics.get("sample_quality_debug", {})
    flow_debug = metrics.get("flow_debug", {})
    if not isinstance(config, dict) or not isinstance(sample_quality, dict) or not isinstance(flow_debug, dict):
        return False

    expected_stochastic_cdro_eval = _is_stochastic_cdro_eval(cfg, str(row.get("method", "")))
    expected_probe_enabled = not bool(args.disable_edm_clean_probe)
    expected_rf_probe_enabled = expected_probe_enabled and not _is_shared_edm_warmup_row_for_rf(row)
    expected_n_steps_path = int(cfg.n_steps_path)
    expected_rf_eval_n_steps_path = (
        int(resolve_rf_eval_n_steps_path(cfg))
        if str(getattr(cfg, "training_objective", "edm")).strip().lower() == "rf"
        else int(cfg.n_steps_path)
    )
    expected_fid_samples = int(args.fid_samples)
    expected_fid_batch_size = max(int(args.fid_batch_size), 1)
    expected_probe_split = str(args.edm_clean_probe_split)
    expected_probe_batches = int(max(args.edm_clean_probe_batches, 0))
    expected_probe_batch_size = int(max(args.edm_clean_probe_batch_size, 0))

    actual_rf_eval_n_steps_path = _safe_int(config.get("rf_eval_n_steps_path"), _safe_int(config.get("n_steps_path"), -1))
    if actual_rf_eval_n_steps_path <= 0:
        actual_rf_eval_n_steps_path = _safe_int(config.get("n_steps_path"), -1)

    return (
        _safe_int(config.get("n_steps_path"), -1) == expected_n_steps_path
        and actual_rf_eval_n_steps_path == expected_rf_eval_n_steps_path
        and _safe_bool(config.get("edm_clean_probe_enabled"), default=False) == expected_probe_enabled
        and str(config.get("edm_clean_probe_split", "")) == expected_probe_split
        and _safe_int(config.get("edm_clean_probe_batches"), -1) == expected_probe_batches
        and _safe_int(config.get("edm_clean_probe_batch_size"), -1) == expected_probe_batch_size
        and _safe_bool(config.get("rf_clean_probe_enabled"), default=False) == expected_rf_probe_enabled
        and str(config.get("rf_clean_probe_split", "")) == expected_probe_split
        and _safe_int(config.get("rf_clean_probe_batches"), -1) == expected_probe_batches
        and _safe_int(config.get("rf_clean_probe_batch_size"), -1) == expected_probe_batch_size
        and _safe_int(sample_quality.get("fid_samples"), -1) == expected_fid_samples
        and _safe_int(sample_quality.get("fid_batch_size"), -1) == expected_fid_batch_size
        and _safe_bool(flow_debug.get("stochastic_cdro_eval"), default=False) == expected_stochastic_cdro_eval
    )


def _summarize_probe_values(values: List[float]) -> Dict[str, object]:
    finite = [float(v) for v in values if np.isfinite(v)]
    if not finite:
        return {
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "final": None,
            "mean_last": None,
            "num_values": 0,
        }
    arr = np.asarray(finite, dtype=np.float64)
    tail = arr[-min(int(arr.size), 10) :]
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "final": float(arr[-1]),
        "mean_last": float(tail.mean()),
        "num_values": int(arr.size),
    }


def _edm_probe_sigma_batch(
    *,
    cfg: ToyConfig,
    sigma_levels: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if bool(getattr(cfg, "use_log_normal_sigma_sampling", True)):
        return sample_sigmas_log_normal(
            int(batch_size),
            sigma_min=float(cfg.sigma_min),
            sigma_max=float(cfg.sigma_max),
            device=device,
            p_mean=float(getattr(cfg, "p_mean", -1.2)),
            p_std=float(getattr(cfg, "p_std", 1.2)),
            dtype=sigma_levels.dtype,
        )
    indices = sample_target_indices(int(batch_size), sigma_levels)
    if sigma_levels.ndim != 1:
        raise ValueError(f"Expected rank-1 probe sigma levels, got shape={tuple(sigma_levels.shape)}")
    return sigma_levels[indices]


def _sample_clean_probe_batch(dataset, split: str, batch_size: int) -> torch.Tensor:
    if str(split).strip().lower() == "val":
        return dataset.sample_val_batch(int(batch_size))
    return dataset.sample_train_batch(int(batch_size))


def _sample_edm_probe_batch(dataset, split: str, batch_size: int) -> torch.Tensor:
    return _sample_clean_probe_batch(dataset, split, batch_size)


def _rf_probe_spec_from_payload(payload: Dict, *, cfg: ToyConfig) -> Dict[str, object]:
    history = payload.get("history_robust", {})
    if not isinstance(history, dict):
        history = {}
    trainer_state = payload.get("trainer_state", {})
    if not isinstance(trainer_state, dict):
        trainer_state = {}
    completed_steps = _safe_int(payload.get("completed_steps", trainer_state.get("completed_steps", 0)), 0)
    stage1_steps = _safe_int(history.get("rf_stage1_steps"), 0)
    reflow_steps = _safe_int(history.get("rf_reflow_steps"), 0)
    stage_name = "rf_reflow" if reflow_steps > 0 and completed_steps > stage1_steps else "rf_stage1"
    distribution_key = f"{stage_name}_t_distribution_resolved"
    t_distribution = str(history.get(distribution_key, "")).strip()
    if not t_distribution:
        t_distribution = (
            "uniform"
            if stage_name == "rf_stage1"
            else str(getattr(cfg, "rf_reflow_t_distribution", "u_shaped"))
        )
    teacher_state_dict = trainer_state.get("rf_teacher_state_dict")
    if not isinstance(teacher_state_dict, dict):
        teacher_state_dict = trainer_state.get("rf_reflow_teacher_state_dict")
    if not isinstance(teacher_state_dict, dict):
        teacher_state_dict = None
    teacher_training_objective = str(trainer_state.get("rf_teacher_training_objective", "")).strip().lower()
    return {
        "stage_name": stage_name,
        "completed_steps": int(completed_steps),
        "stage1_steps": int(stage1_steps),
        "reflow_steps": int(reflow_steps),
        "t_distribution": str(t_distribution),
        "teacher_state_dict": teacher_state_dict,
        "teacher_training_objective": teacher_training_objective,
    }


def _infer_rf_probe_teacher_training_objective(
    *,
    explicit_training_objective: object,
    teacher_state_dict: Optional[Dict[str, object]],
) -> str:
    explicit = str(explicit_training_objective or "").strip().lower()
    if explicit in {"edm", "rf", "score"}:
        return explicit
    if isinstance(teacher_state_dict, dict):
        keys = [str(key) for key in teacher_state_dict.keys()]
        if any("noise_mlp" in key for key in keys):
            return "edm"
        if any("time_mlp" in key for key in keys):
            return "rf"
    return "rf"


def _build_rf_probe_teacher(
    *,
    ctx: EvalContext,
    teacher_state_dict: Dict[str, object],
    teacher_training_objective: str,
) -> Tuple[torch.nn.Module, torch.Tensor]:
    teacher_cfg = copy.deepcopy(ctx.cfg)
    teacher_cfg.training_objective = str(teacher_training_objective).strip().lower() or "rf"
    teacher_bundle = build_model_bundle(teacher_cfg, ctx.dataset, float(ctx.cfg.sigma_data), ctx.device)
    teacher_model = teacher_bundle.baseline.to(device=ctx.device)
    teacher_model.load_state_dict(teacher_state_dict, strict=True)
    teacher_model.eval()
    if str(teacher_cfg.training_objective).strip().lower() == "rf":
        teacher_sigma_levels = build_rf_time_quantile_levels(
            float(teacher_cfg.sigma_max),
            int(resolve_rf_teacher_n_steps_path(teacher_cfg)),
            device=ctx.device,
            distribution=str(getattr(teacher_cfg, "rf_reflow_t_distribution", "u_shaped")),
        )
    else:
        teacher_sigma_levels = build_sigma_levels(
            float(teacher_cfg.sigma_min),
            float(teacher_cfg.sigma_max),
            int(resolve_rf_teacher_n_steps_path(teacher_cfg)),
            device=ctx.device,
        )
    return teacher_model, teacher_sigma_levels.to(dtype=ctx.sigma_levels.dtype)


def _rf_probe_sigma_batch(
    *,
    sigma_levels: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    if sigma_levels.ndim != 1:
        raise ValueError(f"Expected rank-1 RF probe sigma levels, got shape={tuple(sigma_levels.shape)}")
    indices = sample_target_indices(int(batch_size), sigma_levels)
    return sigma_levels[indices]


@torch.no_grad()
def _compute_edm_clean_probe_for_model(
    *,
    denoiser: torch.nn.Module,
    ctx: EvalContext,
    split: str,
    num_batches: int,
    batch_size: int,
    probe_seed: int,
    log_handle: Optional[TextIO] = None,
) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "enabled": True,
        "supported": True,
        "split": str(split),
        "num_batches": max(int(num_batches), 0),
        "batch_size": max(int(batch_size), 0),
        "num_images": 0,
        "seed": int(probe_seed),
        "missing_reason": "",
        "mean": None,
        "std": None,
        "min": None,
        "max": None,
        "final": None,
        "mean_last": None,
        "num_values": 0,
    }
    if str(getattr(ctx.cfg, "training_objective", "edm")).strip().lower() == "rf":
        summary["supported"] = False
        summary["missing_reason"] = "unsupported_training_objective_rf"
        return summary
    if int(num_batches) <= 0 or int(batch_size) <= 0:
        summary["supported"] = False
        summary["missing_reason"] = "non_positive_probe_budget"
        return summary

    denoiser.eval()
    set_seed(int(probe_seed))
    losses: List[float] = []
    total_images = 0
    t_start = time.perf_counter()
    for batch_idx in range(int(num_batches)):
        x0 = _sample_edm_probe_batch(ctx.dataset, split, int(batch_size))
        if x0.device != ctx.device:
            x0 = x0.to(device=ctx.device)
        x0 = x0.to(dtype=ctx.probe_sigma_levels.dtype)
        sigma = _edm_probe_sigma_batch(
            cfg=ctx.cfg,
            sigma_levels=ctx.probe_sigma_levels,
            batch_size=int(x0.shape[0]),
            device=ctx.device,
        )
        sigma_batch = sigma.reshape((int(x0.shape[0]),) + (1,) * (x0.ndim - 1))
        x_noisy = x0 + sigma_batch * torch.randn_like(x0)
        with autocast_context(ctx.device, ctx.amp_dtype):
            loss = weighted_denoise_loss(
                denoiser,
                x_noisy,
                x0,
                sigma,
                float(ctx.cfg.sigma_data),
            )
        loss_value = float(loss.item())
        losses.append(loss_value)
        total_images += int(x0.shape[0])
        if log_handle is not None:
            elapsed = time.perf_counter() - t_start
            log_handle.write(
                f"[edm-clean-probe] progress batch={batch_idx + 1}/{int(num_batches)} "
                f"images={total_images} loss={loss_value:.6f} elapsed_sec={elapsed:.2f}\n"
            )
            log_handle.flush()

    summary.update(_summarize_probe_values(losses))
    summary["num_images"] = int(total_images)
    return summary


@torch.no_grad()
def _compute_rf_clean_probe_for_model(
    *,
    denoiser: torch.nn.Module,
    ctx: EvalContext,
    split: str,
    num_batches: int,
    batch_size: int,
    probe_seed: int,
    probe_spec: Dict[str, object],
    log_handle: Optional[TextIO] = None,
) -> Dict[str, object]:
    stage_name = str(probe_spec.get("stage_name", "rf_stage1")).strip() or "rf_stage1"
    t_distribution = str(probe_spec.get("t_distribution", "")).strip() or (
        "uniform" if stage_name == "rf_stage1" else str(getattr(ctx.cfg, "rf_reflow_t_distribution", "u_shaped"))
    )
    teacher_state_dict = probe_spec.get("teacher_state_dict")
    teacher_training_objective = _infer_rf_probe_teacher_training_objective(
        explicit_training_objective=probe_spec.get("teacher_training_objective"),
        teacher_state_dict=teacher_state_dict if isinstance(teacher_state_dict, dict) else None,
    )
    summary: Dict[str, object] = {
        "enabled": True,
        "supported": True,
        "split": str(split),
        "num_batches": max(int(num_batches), 0),
        "batch_size": max(int(batch_size), 0),
        "num_images": 0,
        "seed": int(probe_seed),
        "missing_reason": "",
        "mean": None,
        "std": None,
        "min": None,
        "max": None,
        "final": None,
        "mean_last": None,
        "num_values": 0,
        "stage_name": stage_name,
        "t_distribution": t_distribution,
        "pair_source": "",
        "teacher_training_objective": teacher_training_objective,
    }
    if str(getattr(ctx.cfg, "training_objective", "edm")).strip().lower() != "rf":
        summary["supported"] = False
        summary["missing_reason"] = "unsupported_training_objective_non_rf"
        return summary
    if int(num_batches) <= 0 or int(batch_size) <= 0:
        summary["supported"] = False
        summary["missing_reason"] = "non_positive_probe_budget"
        return summary

    denoiser.eval()
    set_seed(int(probe_seed))
    probe_sigma_levels = build_rf_stage_time_quantile_levels(
        float(ctx.cfg.sigma_max),
        int(ctx.cfg.n_steps_path),
        ctx.device,
        stage_name=stage_name,
        reflow_distribution=str(getattr(ctx.cfg, "rf_reflow_t_distribution", "u_shaped")),
    ).to(dtype=ctx.sigma_levels.dtype)
    sigma_max_value = max(float(ctx.cfg.sigma_max), 1e-8)
    teacher_model = None
    teacher_sigma_levels = ctx.sigma_levels
    if stage_name == "rf_reflow" and isinstance(teacher_state_dict, dict):
        teacher_model, teacher_sigma_levels = _build_rf_probe_teacher(
            ctx=ctx,
            teacher_state_dict=teacher_state_dict,
            teacher_training_objective=teacher_training_objective,
        )
        summary["pair_source"] = "teacher_reflow_pairs"
    elif stage_name == "rf_reflow":
        summary["pair_source"] = "straight_clean_pairs_fallback"
    else:
        summary["pair_source"] = "straight_clean_pairs"

    losses: List[float] = []
    total_images = 0
    t_start = time.perf_counter()
    for batch_idx in range(int(num_batches)):
        x0 = _sample_clean_probe_batch(ctx.dataset, split, int(batch_size))
        if x0.device != ctx.device:
            x0 = x0.to(device=ctx.device)
        x0 = x0.to(dtype=probe_sigma_levels.dtype)
        if teacher_model is not None:
            x_left, x_right = generate_reflow_pairs(
                teacher_model,
                teacher_sigma_levels,
                x0,
                sample_terminal_batch_fn=ctx.dataset.sample_terminal_batch,
            )
        else:
            x_left = ctx.dataset.sample_terminal_batch(int(x0.shape[0]), 1.0)
            if x_left.device != ctx.device:
                x_left = x_left.to(device=ctx.device)
            x_left = x_left.to(dtype=x0.dtype)
            x_right = x0
        sigma = _rf_probe_sigma_batch(
            sigma_levels=probe_sigma_levels,
            batch_size=int(x0.shape[0]),
        )
        t = (sigma / sigma_max_value).clamp(0.0, 1.0)
        x_t = build_rectified_flow_state(x_left, x_right, t)
        with autocast_context(ctx.device, ctx.amp_dtype):
            loss = weighted_rectified_flow_loss(
                denoiser,
                x_t,
                x_right,
                sigma,
                sigma_max_value,
                x_left=x_left,
                x_right=x_right,
                loss_kind=str(getattr(ctx.cfg, "rf_loss", "mse")),
                pseudo_huber_delta=float(getattr(ctx.cfg, "rf_pseudo_huber_delta", 0.1)),
            )
        loss_value = float(loss.item())
        losses.append(loss_value)
        total_images += int(x0.shape[0])
        if log_handle is not None:
            elapsed = time.perf_counter() - t_start
            log_handle.write(
                f"[rf-clean-probe] progress batch={batch_idx + 1}/{int(num_batches)} "
                f"images={total_images} stage={stage_name} pair_source={summary['pair_source']} "
                f"loss={loss_value:.6f} elapsed_sec={elapsed:.2f}\n"
            )
            log_handle.flush()

    summary.update(_summarize_probe_values(losses))
    summary["num_images"] = int(total_images)
    return summary


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
    fid_reused: bool,
    edm_clean_probe: Dict[str, object],
    edm_clean_probe_runtime_sec: float,
    rf_clean_probe: Dict[str, object],
    rf_clean_probe_runtime_sec: float,
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
            "direct_checkpoint_eval": True,
            "fid_reused": bool(fid_reused),
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
            "image_backbone": str(getattr(ctx.cfg, "image_backbone", ToyConfig.image_backbone)),
            "n_steps_path": int(ctx.cfg.n_steps_path),
            "rf_teacher_n_steps_path": int(resolve_rf_teacher_n_steps_path(ctx.cfg)),
            "rf_eval_n_steps_path": int(resolve_rf_eval_n_steps_path(ctx.cfg)),
            "sigma_min": float(ctx.cfg.sigma_min),
            "sigma_max": float(ctx.cfg.sigma_max),
            "sigma_data": float(ctx.cfg.sigma_data),
            "use_log_normal_sigma_sampling": bool(getattr(ctx.cfg, "use_log_normal_sigma_sampling", True)),
            "training_objective": str(ctx.cfg.training_objective),
            "cdro_edm_ladder_mode": str(
                getattr(ctx.cfg, "cdro_edm_ladder_mode", DETERMINISTIC_MIDPOINT_QUANTILE_LADDER)
            ),
            "rf_reflow_t_distribution": str(getattr(ctx.cfg, "rf_reflow_t_distribution", "u_shaped")),
            "amp_dtype": str(ctx.cfg.amp_dtype),
            "eval_seed_offset_metrics": int(ctx.cfg.eval_seed_offset_metrics),
            "fid_ref_path": str(ctx.cfg.fid_ref_path),
            "compute_fid": True,
            "edm_clean_probe_enabled": bool(edm_clean_probe.get("enabled", False)),
            "edm_clean_probe_split": str(edm_clean_probe.get("split", "")),
            "edm_clean_probe_batches": int(edm_clean_probe.get("num_batches", 0) or 0),
            "edm_clean_probe_batch_size": int(edm_clean_probe.get("batch_size", 0) or 0),
            "rf_clean_probe_enabled": bool(rf_clean_probe.get("enabled", False)),
            "rf_clean_probe_split": str(rf_clean_probe.get("split", "")),
            "rf_clean_probe_batches": int(rf_clean_probe.get("num_batches", 0) or 0),
            "rf_clean_probe_batch_size": int(rf_clean_probe.get("batch_size", 0) or 0),
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
                "direct_checkpoint_eval": True,
                "fid_reused": bool(fid_reused),
                "edm_clean_probe_enabled": bool(edm_clean_probe.get("enabled", False)),
                "rf_clean_probe_enabled": bool(rf_clean_probe.get("enabled", False)),
                "checkpoint_branch": branch,
                "checkpoint_state_variant": str(checkpoint_state_variant),
                "stochastic_cdro_eval": bool(_is_stochastic_cdro_eval(ctx.cfg, str(row.get("method", "")))),
            },
            "objective_debug": {
                "edm_clean_probe": edm_clean_probe,
                "rf_clean_probe": rf_clean_probe,
            },
        },
        "runtime_sec": {
            "fid_total": float(runtime_sec),
            "edm_clean_probe_total": float(edm_clean_probe_runtime_sec),
            "rf_clean_probe_total": float(rf_clean_probe_runtime_sec),
        },
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _evaluate_checkpoint_metrics(
    *,
    args: argparse.Namespace,
    row: Dict[str, str],
    ctx: EvalContext,
    detector_net,
    mu_ref: torch.Tensor,
    sigma_ref: torch.Tensor,
    metrics_path: str,
    log_path: str,
    existing_fid_value: Optional[float] = None,
) -> Dict[str, object]:
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
        log_handle.write(
            "[fid-only] legacy eval-samples argument is ignored in direct FID mode; "
            "checkpoint clean probes use their own fixed evaluation budget.\n"
        )
        log_handle.flush()

        payload = _load_checkpoint_payload(checkpoint_path)
        checkpoint_state_variant = "baseline_eval_state_dict"
        if branch == "baseline":
            state_dict = _baseline_state_dict_from_payload(payload, checkpoint_path)
        else:
            state_dict, checkpoint_state_variant = _robust_state_dict_from_payload(
                payload,
                checkpoint_path,
                str(row.get("method_version_used", row.get("method", ""))),
                prefer_ema=bool(getattr(ctx.cfg, "use_ema_eval", False)),
            )
        log_handle.write(f"[fid-only] checkpoint_state_variant={checkpoint_state_variant}\n")
        log_handle.flush()
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        probe_seed = int(metrics_eval_seed + 1)
        edm_clean_probe_runtime_sec = 0.0
        rf_clean_probe_runtime_sec = 0.0
        if bool(args.disable_edm_clean_probe):
            edm_clean_probe = {
                "enabled": False,
                "supported": False,
                "split": str(args.edm_clean_probe_split),
                "num_batches": int(max(args.edm_clean_probe_batches, 0)),
                "batch_size": int(max(args.edm_clean_probe_batch_size, 0)),
                "num_images": 0,
                "seed": int(probe_seed),
                "missing_reason": "disabled_by_flag",
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "final": None,
                "mean_last": None,
                "num_values": 0,
            }
        else:
            log_handle.write(
                f"[edm-clean-probe] split={args.edm_clean_probe_split} "
                f"batches={int(max(args.edm_clean_probe_batches, 0))} "
                f"batch_size={int(max(args.edm_clean_probe_batch_size, 0))} "
                f"seed={probe_seed}\n"
            )
            log_handle.flush()
            t_probe_start = time.perf_counter()
            edm_clean_probe = _compute_edm_clean_probe_for_model(
                denoiser=model,
                ctx=ctx,
                split=str(args.edm_clean_probe_split),
                num_batches=max(int(args.edm_clean_probe_batches), 0),
                batch_size=max(int(args.edm_clean_probe_batch_size), 0),
                probe_seed=probe_seed,
                log_handle=log_handle,
            )
            edm_clean_probe_runtime_sec = float(time.perf_counter() - t_probe_start)
            log_handle.write(
                "[edm-clean-probe] done "
                f"mean={edm_clean_probe.get('mean')} "
                f"runtime_sec={edm_clean_probe_runtime_sec:.2f} "
                f"missing_reason={edm_clean_probe.get('missing_reason', '')}\n"
            )
            log_handle.flush()

        if _is_shared_edm_warmup_row_for_rf(row):
            rf_clean_probe = {
                "enabled": False,
                "supported": False,
                "split": str(args.edm_clean_probe_split),
                "num_batches": int(max(args.edm_clean_probe_batches, 0)),
                "batch_size": int(max(args.edm_clean_probe_batch_size, 0)),
                "num_images": 0,
                "seed": int(probe_seed),
                "missing_reason": "shared_edm_warmup_checkpoint",
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "final": None,
                "mean_last": None,
                "num_values": 0,
                "stage_name": "",
                "t_distribution": "",
                "pair_source": "",
            }
        elif bool(args.disable_edm_clean_probe):
            rf_clean_probe = {
                "enabled": False,
                "supported": False,
                "split": str(args.edm_clean_probe_split),
                "num_batches": int(max(args.edm_clean_probe_batches, 0)),
                "batch_size": int(max(args.edm_clean_probe_batch_size, 0)),
                "num_images": 0,
                "seed": int(probe_seed),
                "missing_reason": "disabled_by_flag",
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "final": None,
                "mean_last": None,
                "num_values": 0,
                "stage_name": "",
                "t_distribution": "",
                "pair_source": "",
            }
        else:
            rf_probe_spec = _rf_probe_spec_from_payload(payload, cfg=ctx.cfg)
            log_handle.write(
                f"[rf-clean-probe] split={args.edm_clean_probe_split} "
                f"batches={int(max(args.edm_clean_probe_batches, 0))} "
                f"batch_size={int(max(args.edm_clean_probe_batch_size, 0))} "
                f"seed={probe_seed} stage={rf_probe_spec.get('stage_name', '')}\n"
            )
            log_handle.flush()
            t_probe_start = time.perf_counter()
            rf_clean_probe = _compute_rf_clean_probe_for_model(
                denoiser=model,
                ctx=ctx,
                split=str(args.edm_clean_probe_split),
                num_batches=max(int(args.edm_clean_probe_batches), 0),
                batch_size=max(int(args.edm_clean_probe_batch_size), 0),
                probe_seed=probe_seed,
                probe_spec=rf_probe_spec,
                log_handle=log_handle,
            )
            rf_clean_probe_runtime_sec = float(time.perf_counter() - t_probe_start)
            log_handle.write(
                "[rf-clean-probe] done "
                f"mean={rf_clean_probe.get('mean')} "
                f"runtime_sec={rf_clean_probe_runtime_sec:.2f} "
                f"missing_reason={rf_clean_probe.get('missing_reason', '')}\n"
            )
            log_handle.flush()

        if existing_fid_value is None:
            set_seed(metrics_eval_seed)
            t_start = time.perf_counter()
            fid_value = _compute_fid_for_model(
                denoiser=model,
                cfg=ctx.cfg,
                method_name=str(row.get("method", "")),
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
            fid_reused = False
            log_handle.write(f"[fid-only] done fid={fid_value:.6f} runtime_sec={runtime_sec:.2f}\n")
        else:
            fid_value = float(existing_fid_value)
            runtime_sec = 0.0
            fid_reused = True
            log_handle.write(f"[fid-only] reused_existing_fid={fid_value:.6f}\n")
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
        fid_reused=fid_reused,
        edm_clean_probe=edm_clean_probe,
        edm_clean_probe_runtime_sec=edm_clean_probe_runtime_sec,
        rf_clean_probe=rf_clean_probe,
        rf_clean_probe_runtime_sec=rf_clean_probe_runtime_sec,
    )
    return _extract_cached_reeval(metrics_path, row)


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
        objective_probe_key = _objective_clean_probe_key(row)
        objective_probe_present = (
            _safe_float(row.get(objective_probe_key)) is not None
            or _safe_bool(row.get(f"{objective_probe_key}_evaluated"), default=False)
            or _safe_float(row.get("objective_clean_probe")) is not None
            or _safe_bool(row.get("objective_clean_probe_evaluated"), default=False)
        )
        if objective_probe_present:
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
            request_cfg = _config_from_row(args, row)
            existing_cached = _maybe_extract_cached_reeval(metrics_path, row)
            if existing_cached is not None and not _cached_reeval_matches_request(
                metrics_path,
                args=args,
                row=row,
                cfg=request_cfg,
            ):
                existing_cached = None
            objective_probe_key = _objective_clean_probe_key(row)
            probe_needed = bool(
                not args.disable_edm_clean_probe
                and (
                    existing_cached is None
                    or not bool(existing_cached.get(f"{objective_probe_key}_evaluated", False))
                )
            )
            if bool(args.skip_existing) and existing_cached is not None:
                cached_metrics = existing_cached
            elif existing_cached is not None and not probe_needed:
                cached_metrics = existing_cached
            else:
                ctx = _get_or_build_context(args=args, row=row, cache=context_cache)
                existing_fid_value = None
                if existing_cached is not None:
                    existing_fid_value = _safe_float(existing_cached.get("fid"))
                cached_metrics = _evaluate_checkpoint_metrics(
                    args=args,
                    row=row,
                    ctx=ctx,
                    detector_net=detector_net,
                    mu_ref=mu_ref,
                    sigma_ref=sigma_ref,
                    metrics_path=metrics_path,
                    log_path=log_path,
                    existing_fid_value=existing_fid_value,
                )
            cached = {
                "fid": str(cached_metrics["fid"]),
                "reeval_metrics_path": metrics_path,
                "reeval_log_path": log_path,
                "reeval_exp_name": exp_name,
                "edm_clean_probe": cached_metrics["edm_clean_probe"],
                "edm_clean_probe_images": int(cached_metrics["edm_clean_probe_images"]),
                "edm_clean_probe_seed": cached_metrics["edm_clean_probe_seed"],
                "edm_clean_probe_split": str(cached_metrics["edm_clean_probe_split"]),
                "edm_clean_probe_batches": int(cached_metrics["edm_clean_probe_batches"]),
                "edm_clean_probe_batch_size": int(cached_metrics["edm_clean_probe_batch_size"]),
                "edm_clean_probe_evaluated": bool(cached_metrics["edm_clean_probe_evaluated"]),
                "edm_clean_probe_missing_reason": str(cached_metrics["edm_clean_probe_missing_reason"]),
                "edm_clean_probe_source": str(cached_metrics["edm_clean_probe_source"]),
                "rf_clean_probe": cached_metrics["rf_clean_probe"],
                "rf_clean_probe_images": int(cached_metrics["rf_clean_probe_images"]),
                "rf_clean_probe_seed": cached_metrics["rf_clean_probe_seed"],
                "rf_clean_probe_split": str(cached_metrics["rf_clean_probe_split"]),
                "rf_clean_probe_batches": int(cached_metrics["rf_clean_probe_batches"]),
                "rf_clean_probe_batch_size": int(cached_metrics["rf_clean_probe_batch_size"]),
                "rf_clean_probe_stage": str(cached_metrics["rf_clean_probe_stage"]),
                "rf_clean_probe_t_distribution": str(cached_metrics["rf_clean_probe_t_distribution"]),
                "rf_clean_probe_pair_source": str(cached_metrics["rf_clean_probe_pair_source"]),
                "rf_clean_probe_evaluated": bool(cached_metrics["rf_clean_probe_evaluated"]),
                "rf_clean_probe_missing_reason": str(cached_metrics["rf_clean_probe_missing_reason"]),
                "rf_clean_probe_source": str(cached_metrics["rf_clean_probe_source"]),
                "objective_clean_probe": cached_metrics["objective_clean_probe"],
                "objective_clean_probe_kind": str(cached_metrics["objective_clean_probe_kind"]),
                "objective_clean_probe_images": int(cached_metrics["objective_clean_probe_images"]),
                "objective_clean_probe_seed": cached_metrics["objective_clean_probe_seed"],
                "objective_clean_probe_split": str(cached_metrics["objective_clean_probe_split"]),
                "objective_clean_probe_batches": int(cached_metrics["objective_clean_probe_batches"]),
                "objective_clean_probe_batch_size": int(cached_metrics["objective_clean_probe_batch_size"]),
                "objective_clean_probe_stage": str(cached_metrics["objective_clean_probe_stage"]),
                "objective_clean_probe_t_distribution": str(
                    cached_metrics["objective_clean_probe_t_distribution"]
                ),
                "objective_clean_probe_pair_source": str(cached_metrics["objective_clean_probe_pair_source"]),
                "objective_clean_probe_evaluated": bool(cached_metrics["objective_clean_probe_evaluated"]),
                "objective_clean_probe_missing_reason": str(
                    cached_metrics["objective_clean_probe_missing_reason"]
                ),
                "objective_clean_probe_source": str(cached_metrics["objective_clean_probe_source"]),
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
        updated["reeval_mode"] = "direct_fid_and_objective_clean_probe_in_memory"
        updated["fid_evaluated"] = True
        updated["fid_missing_reason"] = ""
        updated["fid_source"] = "reevaluated_from_checkpoint"
        updated["edm_clean_probe"] = cached["edm_clean_probe"]
        updated["edm_clean_probe_evaluated"] = bool(cached["edm_clean_probe_evaluated"])
        updated["edm_clean_probe_missing_reason"] = str(cached["edm_clean_probe_missing_reason"])
        updated["edm_clean_probe_source"] = str(cached["edm_clean_probe_source"])
        updated["edm_clean_probe_split"] = str(cached["edm_clean_probe_split"])
        updated["edm_clean_probe_batches"] = int(cached["edm_clean_probe_batches"])
        updated["edm_clean_probe_batch_size"] = int(cached["edm_clean_probe_batch_size"])
        updated["edm_clean_probe_images"] = int(cached["edm_clean_probe_images"])
        updated["edm_clean_probe_seed"] = cached["edm_clean_probe_seed"]
        updated["rf_clean_probe"] = cached["rf_clean_probe"]
        updated["rf_clean_probe_evaluated"] = bool(cached["rf_clean_probe_evaluated"])
        updated["rf_clean_probe_missing_reason"] = str(cached["rf_clean_probe_missing_reason"])
        updated["rf_clean_probe_source"] = str(cached["rf_clean_probe_source"])
        updated["rf_clean_probe_split"] = str(cached["rf_clean_probe_split"])
        updated["rf_clean_probe_batches"] = int(cached["rf_clean_probe_batches"])
        updated["rf_clean_probe_batch_size"] = int(cached["rf_clean_probe_batch_size"])
        updated["rf_clean_probe_images"] = int(cached["rf_clean_probe_images"])
        updated["rf_clean_probe_seed"] = cached["rf_clean_probe_seed"]
        updated["rf_clean_probe_stage"] = str(cached["rf_clean_probe_stage"])
        updated["rf_clean_probe_t_distribution"] = str(cached["rf_clean_probe_t_distribution"])
        updated["rf_clean_probe_pair_source"] = str(cached["rf_clean_probe_pair_source"])
        updated["objective_clean_probe"] = cached["objective_clean_probe"]
        updated["objective_clean_probe_kind"] = str(cached["objective_clean_probe_kind"])
        updated["objective_clean_probe_evaluated"] = bool(cached["objective_clean_probe_evaluated"])
        updated["objective_clean_probe_missing_reason"] = str(cached["objective_clean_probe_missing_reason"])
        updated["objective_clean_probe_source"] = str(cached["objective_clean_probe_source"])
        updated["objective_clean_probe_split"] = str(cached["objective_clean_probe_split"])
        updated["objective_clean_probe_batches"] = int(cached["objective_clean_probe_batches"])
        updated["objective_clean_probe_batch_size"] = int(cached["objective_clean_probe_batch_size"])
        updated["objective_clean_probe_images"] = int(cached["objective_clean_probe_images"])
        updated["objective_clean_probe_seed"] = cached["objective_clean_probe_seed"]
        updated["objective_clean_probe_stage"] = str(cached["objective_clean_probe_stage"])
        updated["objective_clean_probe_t_distribution"] = str(cached["objective_clean_probe_t_distribution"])
        updated["objective_clean_probe_pair_source"] = str(cached["objective_clean_probe_pair_source"])
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
        "reeval_mode": "direct_fid_and_objective_clean_probe_in_memory",
        "device": str(device),
        "fid_ref_path": args.fid_ref_path,
        "edm_clean_probe_enabled": bool(not args.disable_edm_clean_probe),
        "edm_clean_probe_split": str(args.edm_clean_probe_split),
        "edm_clean_probe_batches": int(max(args.edm_clean_probe_batches, 0)),
        "edm_clean_probe_batch_size": int(max(args.edm_clean_probe_batch_size, 0)),
        "rf_clean_probe_enabled": bool(not args.disable_edm_clean_probe),
        "rf_clean_probe_split": str(args.edm_clean_probe_split),
        "rf_clean_probe_batches": int(max(args.edm_clean_probe_batches, 0)),
        "rf_clean_probe_batch_size": int(max(args.edm_clean_probe_batch_size, 0)),
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
        f"fid_batch_size={max(int(args.fid_batch_size), 1)} "
        f"objective_clean_probe={'off' if bool(args.disable_edm_clean_probe) else 'on'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
