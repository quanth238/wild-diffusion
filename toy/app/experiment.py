import json
import math
import os
import random
import hashlib
import uuid
import time
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np
import torch

from ..checks import run_preflight_checks
from ..compute_accounting import (
    baseline_weighted_compute_units_for_steps,
    cdro_robust_step_weighted_compute_units,
    ensure_denoiser_op_count_history,
    load_weighted_compute_calibration,
    read_denoiser_op_count_totals,
    solve_warmup_steps_for_target_compute_fraction,
    wdro_robust_step_weighted_compute_units,
    weighted_compute_units,
)
from ..data_backends.provider import DatasetBundle, build_dataset_bundle
from ..diagnostics_backends.provider import build_diagnostics_bundle
from ..shared.ema import ema_config_dict
from ..shared.objective import terminal_prior_scale_from_objective
from ..shared.sigma import (
    build_rf_stage_time_quantile_levels,
    build_sigma_levels,
    build_sigma_levels_from_warmup_quantiles,
    resolve_rf_stage_t_distribution,
    sample_target_indices,
)
from ..shared.runtime import autocast_context, configure_runtime, format_amp_dtype, resolve_amp_dtype
from ..model_backends.provider import build_model_bundle
from .utils import (
    compute_terminal_match_stats,
    empty_robust_history,
    summarize_series,
    summarize_train_vs_val_curve_gaps,
)
from ..metrics import (
    compute_control_constraint_stats_by_step,
    compute_denoise_error_curves,
    compute_paired_reverse_delta_by_step,
    compute_path_mse_by_step,
    compute_x0_recovery_vs_terminal_step,
    evaluate_nearest_reference_distance,
    summarize_attack_gap_windows,
)
from ..shared.reverse import generated_data_path_index_from_denoiser, sample_rectified_flow_paths_from_source
from ..trainer import reverse_paths_from_terminal, sample_reverse_paths, train_baseline
from ..utils import as_jsonable_metrics, ensure_dir, pick_device, set_seed, tensor_to_numpy
from ..versions.cdro.trainer import _resolve_rf_cdro_pair_source, _resolve_rf_cdro_stage_steps
from ..versions.registry import resolve_method_module


def _capture_rng_state() -> Dict:
    """Capture Python/NumPy/Torch RNG states for temporary reseeding blocks."""

    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Dict) -> None:
    """Restore Python/NumPy/Torch RNG states captured by `_capture_rng_state`."""

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _run_with_scoped_seed(seed: int, fn):
    """Run `fn` with a temporary global seed, then restore prior RNG states."""

    state = _capture_rng_state()
    set_seed(int(seed))
    try:
        return fn()
    finally:
        _restore_rng_state(state)


def _sample_terminal_batch_for_objective(
    *,
    cfg,
    dataset: DatasetBundle,
    batch_size: int,
    sigma_levels: torch.Tensor,
) -> torch.Tensor:
    """Sample terminal states using the correct objective-family terminal law."""

    terminal_scale = terminal_prior_scale_from_objective(
        getattr(cfg, "training_objective", "edm"),
        float(sigma_levels[-1].item()),
    )
    return dataset.sample_terminal_batch(batch_size, terminal_scale)


def _build_family_sigma_levels(cfg, device: torch.device) -> torch.Tensor:
    """Build the solver/continuation grid for the active generative family."""

    objective = str(getattr(cfg, "training_objective", "edm")).strip().lower()
    if objective == "rf":
        return build_rf_stage_time_quantile_levels(
            float(getattr(cfg, "sigma_max", 1.0)),
            int(getattr(cfg, "n_steps_path", 1)),
            device=device,
            stage_name="rf_reflow",
            reflow_distribution=str(getattr(cfg, "rf_reflow_t_distribution", "u_shaped")),
        )
    return build_sigma_levels(cfg.sigma_min, cfg.sigma_max, cfg.n_steps_path, device=device)


def _is_cdro_rf_port(cfg, method_name: str) -> bool:
    """Return whether the active run is the paper-port CDRO-RF configuration."""

    return (
        str(method_name).strip().lower() == "cdro"
        and str(getattr(cfg, "training_objective", "edm")).strip().lower() == "rf"
    )


def _resolve_cdro_attack_num_steps(cfg) -> tuple[int, str]:
    """Resolve CDRO attack steps with legacy `inner_steps` fallback."""

    explicit = getattr(cfg, "attack_num_steps", None)
    if explicit is not None:
        resolved = int(explicit)
        if resolved not in (1, 2):
            raise ValueError(f"attack_num_steps must be one of (1, 2), got {explicit}")
        return resolved, "attack_num_steps"
    legacy = int(getattr(cfg, "inner_steps", 1))
    if legacy <= 0:
        return 0, "inner_steps_legacy"
    return legacy, "inner_steps_legacy"


def _resolve_rf_cdro_eval_stage(cfg, total_steps: int) -> str:
    """Resolve which RF stage law should drive final CDRO-RF evaluation grids."""

    pair_source = _resolve_rf_cdro_pair_source(cfg)
    stage1_steps, reflow_steps = _resolve_rf_cdro_stage_steps(
        int(total_steps),
        float(getattr(cfg, "rf_stage1_fraction", 0.5)),
        pair_source,
    )
    del stage1_steps
    return "rf_reflow" if int(reflow_steps) > 0 else "rf_stage1"


def _checkpoint_state_dict_for_rf_init(payload: Any) -> tuple[Dict[str, torch.Tensor], str]:
    """Extract a likely model state dict from common repo checkpoint payloads."""

    if not isinstance(payload, dict):
        raise RuntimeError("RF EDM init checkpoint must contain a state-dict-like payload.")
    for key in ("baseline_state_dict", "state_dict", "model_state_dict", "robust_state_dict", "ema_state_dict"):
        candidate = payload.get(key)
        if isinstance(candidate, dict):
            return candidate, key
    if payload and all(torch.is_tensor(value) for value in payload.values()):
        return payload, "raw_state_dict"
    raise RuntimeError(
        "Could not find a model state dict in RF EDM init checkpoint. "
        "Expected one of baseline_state_dict, state_dict, model_state_dict, robust_state_dict, ema_state_dict, "
        "or a raw tensor state dict."
    )


def _normalize_source_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            continue
        text_key = str(key)
        out[text_key] = value
        if text_key.startswith("module."):
            out[text_key[len("module.") :]] = value
    return out


def _candidate_edm_source_keys_for_rf(target_key: str) -> list[str]:
    candidates = [target_key]
    if "time_mlp" in target_key:
        candidates.append(target_key.replace("time_mlp", "noise_mlp"))
    return candidates


def _warm_start_rf_model_from_state_dict(model, source_state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """Best-effort EDM -> RF weight transfer for compatible trunks/backbones."""

    source = _normalize_source_state_dict(source_state_dict)
    target = model.state_dict()
    updated = dict(target)
    transferred = []
    shape_mismatches = []
    missing = []
    for target_key, target_tensor in target.items():
        copied = False
        saw_candidate = False
        for source_key in _candidate_edm_source_keys_for_rf(target_key):
            source_tensor = source.get(source_key)
            if source_tensor is None:
                continue
            saw_candidate = True
            if tuple(source_tensor.shape) != tuple(target_tensor.shape):
                shape_mismatches.append(
                    {
                        "target_key": target_key,
                        "source_key": source_key,
                        "target_shape": list(target_tensor.shape),
                        "source_shape": list(source_tensor.shape),
                    }
                )
                continue
            updated[target_key] = source_tensor.to(device=target_tensor.device, dtype=target_tensor.dtype)
            transferred.append({"target_key": target_key, "source_key": source_key})
            copied = True
            break
        if not copied and not saw_candidate:
            missing.append(target_key)
    model.load_state_dict(updated, strict=True)
    return {
        "transferred_count": int(len(transferred)),
        "target_key_count": int(len(target)),
        "shape_mismatch_count": int(len(shape_mismatches)),
        "missing_count": int(len(missing)),
        "transferred_preview": transferred[:12],
        "shape_mismatch_preview": shape_mismatches[:12],
        "missing_preview": missing[:12],
    }


def _maybe_initialize_rf_from_edm_checkpoint(cfg, model_bundle) -> Dict[str, Any]:
    """Initialize RF models from an EDM checkpoint when requested."""

    objective = str(getattr(cfg, "training_objective", "edm")).strip().lower()
    ckpt_path = str(getattr(cfg, "rf_edm_init_ckpt_path", "")).strip()
    report: Dict[str, Any] = {
        "enabled": bool(objective == "rf" and ckpt_path),
        "path": ckpt_path,
        "source_state_key": None,
        "baseline": None,
        "robust": None,
    }
    if objective != "rf" or not ckpt_path:
        return report
    abs_path = os.path.abspath(ckpt_path)
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(f"RF EDM init checkpoint not found: {abs_path}")
    payload = torch.load(abs_path, map_location="cpu", weights_only=False)
    source_state, source_key = _checkpoint_state_dict_for_rf_init(payload)
    report["path"] = abs_path
    report["source_state_key"] = source_key
    report["baseline"] = _warm_start_rf_model_from_state_dict(model_bundle.baseline, source_state)
    report["robust"] = _warm_start_rf_model_from_state_dict(model_bundle.robust, source_state)
    if int(report["baseline"]["transferred_count"]) <= 0:
        raise RuntimeError(
            f"RF EDM init checkpoint did not transfer any compatible baseline weights: {abs_path}"
        )
    return report


def _load_json_if_exists(path: str) -> Optional[Dict[str, Any]]:
    """Load small JSON sidecar metadata when present."""

    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _optional_float(value: Any) -> Optional[float]:
    """Best-effort float conversion used for external artifact metadata."""

    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sum_float_series(values) -> float:
    """Stable float sum over metric histories that may contain mixed scalar types."""

    return float(sum(float(v) for v in values))


def _count_positive_weight(value: float) -> int:
    """Return 1 when a loss branch is active and 0 otherwise."""

    return 1 if float(value) > 0.0 else 0


def _device_accounting_metadata(device: torch.device) -> Dict[str, Any]:
    """Capture the fixed-hardware metadata needed for wall-clock and GPU-hour reporting."""

    device_type = str(device.type)
    accelerator_kind = "gpu" if device_type == "cuda" else device_type
    accelerator_count = 1 if device_type in ("cuda", "mps", "cpu") else 0
    accelerator_name = str(device)
    if device_type == "cuda" and torch.cuda.is_available():
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        accelerator_name = str(torch.cuda.get_device_name(device_index))
    return {
        "train_accelerator_kind": accelerator_kind,
        "train_accelerator_name": accelerator_name,
        "train_accelerator_count": int(accelerator_count),
        "train_gpu_count": int(1 if device_type == "cuda" else 0),
    }


def _resolve_baseline_reference_train_wall_clock_sec(
    *,
    ckpt_path: str,
    planned_steps: int,
) -> Dict[str, Any]:
    """Recover baseline warmup train time from a reused checkpoint when possible."""

    if not ckpt_path or not os.path.isfile(ckpt_path):
        return {"train_wall_clock_sec": None, "source": "unavailable"}

    try:
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception:
        payload = None
    if isinstance(payload, dict):
        runtime_block = payload.get("baseline_runtime")
        if isinstance(runtime_block, dict):
            runtime_value = _optional_float(
                runtime_block.get("train_wall_clock_sec", runtime_block.get("baseline_train_wall_clock_sec"))
            )
            runtime_step = runtime_block.get("step")
            if runtime_value is not None and (runtime_step is None or int(runtime_step) == int(planned_steps)):
                return {
                    "train_wall_clock_sec": float(runtime_value),
                    "source": "baseline_checkpoint_payload",
                }
        direct_runtime = _optional_float(payload.get("train_elapsed_sec"))
        direct_step = payload.get("step")
        if direct_runtime is not None and (direct_step is None or int(direct_step) == int(planned_steps)):
            return {
                "train_wall_clock_sec": float(direct_runtime),
                "source": "baseline_checkpoint_direct_field",
            }

    checkpoint_dir = os.path.dirname(os.path.abspath(ckpt_path))
    exp_dir = os.path.dirname(checkpoint_dir)
    run_state_path = os.path.join(exp_dir, "run_state.pt")
    if os.path.isfile(run_state_path):
        try:
            run_state = torch.load(run_state_path, map_location="cpu", weights_only=False)
        except Exception:
            run_state = None
        if isinstance(run_state, dict):
            for row in run_state.get("rows", []):
                if not isinstance(row, dict):
                    continue
                row_step = row.get("step")
                if row_step is None or int(row_step) != int(planned_steps):
                    continue
                runtime_value = _optional_float(row.get("train_elapsed_sec"))
                if runtime_value is not None:
                    return {
                        "train_wall_clock_sec": float(runtime_value),
                        "source": "baseline_run_state_rows",
                    }

    return {"train_wall_clock_sec": None, "source": "unavailable"}


def _compute_weighted_accounting(
    *,
    cfg,
    method_name: str,
    sigma_levels: torch.Tensor,
    baseline_steps_total: int,
    history_baseline: Optional[Dict[str, Any]],
    robust_steps_total: int,
    history_robust: Dict[str, Any],
    attack_training_executed: bool,
    calibration: Dict[str, Any],
    preserved_baseline_counts: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build method-aware weighted-op counts and totals from training histories."""

    path_steps = max(int(sigma_levels.numel()) - 1, 0)
    robust_method_supported = True
    baseline_n_fwd = 0.0
    baseline_n_fwd_inputgrad = 0.0
    baseline_n_fwd_parambackward = float(max(int(baseline_steps_total), 0))
    baseline_count_source = "inferred_from_baseline_steps"
    if preserved_baseline_counts is not None:
        baseline_n_fwd = float(preserved_baseline_counts.get("n_fwd", baseline_n_fwd))
        baseline_n_fwd_inputgrad = float(
            preserved_baseline_counts.get("n_fwd_inputgrad", baseline_n_fwd_inputgrad)
        )
        baseline_n_fwd_parambackward = float(
            preserved_baseline_counts.get("n_fwd_parambackward", baseline_n_fwd_parambackward)
        )
        baseline_count_source = "preserved_resume_counts"
    else:
        direct_baseline_counts = read_denoiser_op_count_totals(history_baseline)
        if direct_baseline_counts is not None:
            baseline_n_fwd = float(direct_baseline_counts["n_fwd"])
            baseline_n_fwd_inputgrad = float(direct_baseline_counts["n_fwd_inputgrad"])
            baseline_n_fwd_parambackward = float(direct_baseline_counts["n_fwd_parambackward"])
            baseline_count_source = "history_direct"

    robust_n_fwd = 0.0
    robust_n_fwd_inputgrad = 0.0
    robust_n_fwd_parambackward = 0.0
    robust_count_source = "no_robust_phase"
    if attack_training_executed and robust_steps_total > 0:
        direct_robust_counts = read_denoiser_op_count_totals(history_robust)
        if direct_robust_counts is not None:
            robust_n_fwd = float(direct_robust_counts["n_fwd"])
            robust_n_fwd_inputgrad = float(direct_robust_counts["n_fwd_inputgrad"])
            robust_n_fwd_parambackward = float(direct_robust_counts["n_fwd_parambackward"])
            robust_count_source = "history_direct"
        elif method_name == "wdro":
            robust_n_fwd_inputgrad = _sum_float_series(
                history_robust.get("batch_equiv_denoiser_evals_attack_construction", [])
            )
            robust_n_fwd_parambackward = float(max(int(robust_steps_total), 0))
            robust_count_source = "history_inferred_wdro"
        elif method_name in ("cdro", "v1.1", "1.1"):
            robust_n_fwd_inputgrad = _sum_float_series(
                history_robust.get("batch_equiv_denoiser_evals_attack_construction", [])
            )
            if method_name == "cdro":
                cdro_attack_num_steps, _ = _resolve_cdro_attack_num_steps(cfg)
                attack_enabled = bool(float(cfg.outer_attack_weight) > 0.0 and int(cdro_attack_num_steps) > 0)
            else:
                attack_enabled = bool(float(cfg.outer_attack_weight) > 0.0 and int(cfg.inner_steps) > 0)
            if method_name == "cdro":
                robust_n_fwd = float(path_steps * robust_steps_total) if attack_enabled else 0.0
            else:
                robust_n_fwd = float(path_steps * robust_steps_total)
            active_outer_branches = _count_positive_weight(float(cfg.outer_attack_weight)) + _count_positive_weight(
                float(cfg.outer_clean_weight)
            )
            robust_n_fwd_parambackward = float(path_steps * robust_steps_total * active_outer_branches)
            robust_count_source = "history_inferred_pathwise"
        elif method_name == "clean":
            robust_n_fwd_parambackward = float(max(int(robust_steps_total), 0))
            robust_count_source = "history_inferred_clean"
        else:
            robust_method_supported = False
            robust_count_source = "unsupported_method"

    baseline_weighted_total = weighted_compute_units(
        n_fwd=baseline_n_fwd,
        n_fwd_inputgrad=baseline_n_fwd_inputgrad,
        n_fwd_parambackward=baseline_n_fwd_parambackward,
        calibration=calibration,
    )
    robust_weighted_total = None
    if robust_method_supported:
        robust_weighted_total = weighted_compute_units(
            n_fwd=robust_n_fwd,
            n_fwd_inputgrad=robust_n_fwd_inputgrad,
            n_fwd_parambackward=robust_n_fwd_parambackward,
            calibration=calibration,
        )
    effective_weighted_total = (
        None
        if baseline_weighted_total is None or robust_weighted_total is None
        else float(baseline_weighted_total + robust_weighted_total)
    )
    return {
        "baseline": {
            "n_fwd": float(baseline_n_fwd),
            "n_fwd_inputgrad": float(baseline_n_fwd_inputgrad),
            "n_fwd_parambackward": float(baseline_n_fwd_parambackward),
            "weighted_compute_units": None if baseline_weighted_total is None else float(baseline_weighted_total),
            "supported": True,
            "count_source": baseline_count_source,
        },
        "robust": {
            "n_fwd": float(robust_n_fwd),
            "n_fwd_inputgrad": float(robust_n_fwd_inputgrad),
            "n_fwd_parambackward": float(robust_n_fwd_parambackward),
            "weighted_compute_units": None if robust_weighted_total is None else float(robust_weighted_total),
            "supported": bool(robust_method_supported),
            "count_source": robust_count_source,
        },
        "effective": {
            "n_fwd": float(baseline_n_fwd + robust_n_fwd),
            "n_fwd_inputgrad": float(baseline_n_fwd_inputgrad + robust_n_fwd_inputgrad),
            "n_fwd_parambackward": float(baseline_n_fwd_parambackward + robust_n_fwd_parambackward),
            "weighted_compute_units": None if effective_weighted_total is None else float(effective_weighted_total),
            "supported": bool(robust_method_supported),
            "count_source": (
                f"baseline={baseline_count_source};robust={robust_count_source}"
            ),
        },
    }


def _print_dataset_info(cfg, dataset: DatasetBundle) -> None:
    """Print dataset regime summary (limited-data vs population sampling)."""

    print(f"[info] dataset_backend={dataset.name} data_shape={dataset.data_shape}", flush=True)
    if cfg.limited_data_enabled:
        if dataset.name == "toy_gmm":
            print(
                "[info] limited_data_enabled=True "
                f"train_pool_size={dataset.train_pool.shape[0]} "
                f"({cfg.train_points_per_mode} points/mode x {cfg.n_modes} modes)",
                flush=True,
            )
        else:
            print(
                "[info] limited_data_enabled=True "
                f"train_pool_size={dataset.train_pool.shape[0]}",
                flush=True,
            )
    else:
        print("[info] limited_data_enabled=False (population sampling mode)", flush=True)
    print(f"[info] val_pool_size={dataset.val_pool.shape[0]}", flush=True)
    if dataset.name == "image_folder":
        disjoint = dataset.metadata.get("train_val_disjoint_guarantee")
        if disjoint is False:
            print(
                "[warn] image_folder backend is using dataset_val_path mode; train/val disjointness is not guaranteed "
                "unless the two roots are independently curated.",
                flush=True,
            )


def _method_rollout_kwargs(cfg, method) -> Dict:
    """Method-specific rollout kwargs to keep train/eval rollout behavior aligned."""

    method_name = str(getattr(method, "NAME", cfg.method_version)).lower()
    if method_name in ("v1.1", "1.1"):
        return {
            "total_budget": float(cfg.v11_total_budget_rho),
            "projection_mode": str(cfg.v11_projection_mode).lower(),
        }
    if method_name == "cdro":
        return {
            "total_budget": float(cfg.cdro_total_budget_rho),
            "time_horizon": float(cfg.cdro_time_horizon),
        }
    return {}


def _rollout_for_eval(
    *,
    cfg,
    method,
    x0: torch.Tensor,
    target_indices: torch.Tensor,
    control,
    sigma_levels: torch.Tensor,
    kappa_by_step: torch.Tensor,
    rollout_kwargs: Dict,
    attack_net,
):
    """Method-aware eval rollout. v1.1/v1.2 may override with denoiser-dependent attack."""

    rollout_eval_fn = getattr(method, "rollout_eval", None)
    if callable(rollout_eval_fn):
        return rollout_eval_fn(
            cfg=cfg,
            x0=x0,
            target_indices=target_indices,
            attack_net=attack_net,
            sigma_levels=sigma_levels,
            control_radius_kappa=cfg.control_radius_kappa,
            kappa_by_step=kappa_by_step,
        )
    try:
        return method.rollout_controlled_ve(
            x0=x0,
            target_indices=target_indices,
            control_net=control,
            sigma_levels=sigma_levels,
            grad_through_control=False,
            control_radius_kappa=cfg.control_radius_kappa,
            kappa_by_step=kappa_by_step,
            cfg=cfg,
            **rollout_kwargs,
        )
    except TypeError:
        return method.rollout_controlled_ve(
            x0=x0,
            target_indices=target_indices,
            control_net=control,
            sigma_levels=sigma_levels,
            grad_through_control=False,
            control_radius_kappa=cfg.control_radius_kappa,
            kappa_by_step=kappa_by_step,
            **rollout_kwargs,
        )


def _empty_baseline_history() -> Dict:
    """Baseline history schema used when loading checkpoints without curves."""

    history = {"loss": [], "proxy_weighted_denoise_loss": [], "sigma_counts": []}
    ensure_denoiser_op_count_history(history)
    return history


def _build_baseline_signature(cfg, dataset: DatasetBundle, model_bundle, sigma_levels: torch.Tensor) -> Dict:
    """Build a strict signature for fair baseline checkpoint reuse."""

    train_selection_policy = dataset.metadata.get("train_selection_policy")
    sigma_list = [float(v.item()) for v in sigma_levels.detach().cpu()]
    objective = str(cfg.training_objective).strip().lower()
    # NOTE: `method_version` is intentionally excluded so v2/v2.1/v1.1 can share
    # one identical baseline checkpoint under the same baseline data+train settings.
    signature = {
        "signature_version": 2 if objective == "rf" else 1,
        "seed": int(cfg.seed),
        "dataset_kind": str(dataset.name),
        "dataset_path": str(getattr(cfg, "dataset_path", "")),
        "dataset_val_path": str(getattr(cfg, "dataset_val_path", "")),
        "data_shape": [int(v) for v in dataset.data_shape],
        "limited_data_enabled": bool(cfg.limited_data_enabled),
        "train_pool_size": int(dataset.train_pool.shape[0]) if dataset.train_pool is not None else None,
        "val_pool_size": int(dataset.val_pool.shape[0]),
        "train_selection_policy": train_selection_policy,
        "train_subset_size_resolved": dataset.metadata.get("train_subset_size_resolved"),
        "val_subset_size_resolved": dataset.metadata.get("val_subset_size_resolved"),
        "train_subset_fraction_resolved": dataset.metadata.get("train_subset_fraction_resolved"),
        "model_backend": str(model_bundle.name),
        "hidden_dim": int(cfg.hidden_dim),
        "training_objective": str(cfg.training_objective),
        "sigma_data": float(cfg.sigma_data),
        "n_steps_path": int(cfg.n_steps_path),
        "sigma_min": float(cfg.sigma_min),
        "sigma_max": float(cfg.sigma_max),
        "sigma_levels": sigma_list,
        "baseline_steps": int(cfg.steps),
        "batch_size": int(cfg.batch_size),
        "lr_theta": float(cfg.lr_theta),
        "use_log_normal_sigma_sampling": bool(cfg.use_log_normal_sigma_sampling),
        "p_mean": float(cfg.p_mean),
        "p_std": float(cfg.p_std),
        **ema_config_dict(cfg),
    }
    if objective == "rf":
        signature.update(
            {
                "rf_baseline_mode": str(getattr(cfg, "rf_baseline_mode", "strong")),
                "rf_stage1_fraction": float(getattr(cfg, "rf_stage1_fraction", 0.5)),
                "rf_reflow_t_distribution": str(getattr(cfg, "rf_reflow_t_distribution", "u_shaped")),
                "rf_loss": str(getattr(cfg, "rf_loss", "pseudo_huber")),
                "rf_pseudo_huber_delta": float(getattr(cfg, "rf_pseudo_huber_delta", 0.1)),
                "rf_edm_init_ckpt_path": str(getattr(cfg, "rf_edm_init_ckpt_path", "")),
                "rf_ema_forced_for_strong_baseline": bool(
                    str(getattr(cfg, "rf_baseline_mode", "strong")).strip().lower() == "strong"
                ),
            }
        )
    return signature


def _estimate_wdro_robust_step_batch_equiv(cfg, train_pool_size: Optional[int]) -> float:
    """Estimate WDRO robust-step compute in batch-equivalent denoiser evals.

    Uses the expected refresh/attack workload from the current config. When `train_pool_size`
    is unavailable, fall back to the large-dataset approximation
    `1 + adv_prob * attack_steps / refresh_epochs`.
    """

    attack_steps = max(int(getattr(cfg, "wdro_attack_steps", 0)), 0)
    adv_prob = max(0.0, min(float(getattr(cfg, "wdro_adv_prob", 0.0)), 1.0))
    refresh_epochs = max(float(getattr(cfg, "wdro_refresh_epochs", 1.0)), 1e-8)
    if train_pool_size is None or int(train_pool_size) <= 0:
        return 1.0 + adv_prob * float(attack_steps) / refresh_epochs

    batch_size = max(int(getattr(cfg, "batch_size", 1)), 1)
    num_batches = max(int(math.ceil(float(train_pool_size) / float(batch_size))), 1)
    refresh_interval_steps = max(
        int(math.ceil(refresh_epochs * float(train_pool_size) / float(batch_size))),
        1,
    )
    expected_attack_batches = adv_prob * float(num_batches)
    expected_attack_units_per_refresh = expected_attack_batches * float(attack_steps)
    return 1.0 + expected_attack_units_per_refresh / float(refresh_interval_steps)


def _estimate_wdro_warmup_compute_fraction(
    cfg,
    train_pool_size: Optional[int],
    warmup_fraction: Optional[float] = None,
) -> tuple[float, float]:
    """Return WDRO-style warmup compute share and estimated robust-step compute."""

    total_steps = int(cfg.steps)
    if total_steps <= 0:
        return 0.0, 1.0
    if warmup_fraction is None:
        warmup_fraction = float(getattr(cfg, "wdro_warmup_fraction", 0.0))
    if int(getattr(cfg, "baseline_steps_override", 0)) > 0:
        baseline_steps = int(cfg.baseline_steps_override)
    else:
        baseline_steps = int(total_steps * float(warmup_fraction))
    baseline_steps = max(0, min(baseline_steps, total_steps))
    robust_steps = max(total_steps - baseline_steps, 0)
    wdro_robust_step_units = _estimate_wdro_robust_step_batch_equiv(cfg, train_pool_size)
    warmup_units = float(baseline_steps)
    robust_units = float(robust_steps) * float(wdro_robust_step_units)
    total_units = warmup_units + robust_units
    if total_units <= 0.0:
        return 0.0, float(wdro_robust_step_units)
    return warmup_units / total_units, float(wdro_robust_step_units)


def _estimate_wdro_warmup_weighted_compute_fraction(
    cfg,
    train_pool_size: Optional[int],
    calibration: Dict[str, Any],
    warmup_fraction: Optional[float] = None,
) -> tuple[Optional[float], Optional[float]]:
    """Return WDRO-style warmup weighted-compute share and robust-step weighted cost."""

    total_steps = int(cfg.steps)
    if total_steps <= 0:
        return 0.0, 0.0
    if warmup_fraction is None:
        warmup_fraction = float(getattr(cfg, "wdro_warmup_fraction", 0.0))
    if int(getattr(cfg, "baseline_steps_override", 0)) > 0:
        baseline_steps = int(cfg.baseline_steps_override)
    else:
        baseline_steps = int(total_steps * float(warmup_fraction))
    baseline_steps = max(0, min(baseline_steps, total_steps))
    robust_steps = max(total_steps - baseline_steps, 0)
    baseline_weighted_units = baseline_weighted_compute_units_for_steps(
        steps=int(baseline_steps),
        calibration=calibration,
    )
    wdro_robust_step_weighted_units = wdro_robust_step_weighted_compute_units(
        batch_size=int(getattr(cfg, "batch_size", 1)),
        train_pool_size=train_pool_size,
        refresh_epochs=float(getattr(cfg, "wdro_refresh_epochs", 1.0)),
        adv_prob=float(getattr(cfg, "wdro_adv_prob", 0.0)),
        attack_steps=int(getattr(cfg, "wdro_attack_steps", 0)),
        calibration=calibration,
    )
    if baseline_weighted_units is None or wdro_robust_step_weighted_units is None:
        return None, None
    total_weighted_units = float(baseline_weighted_units) + float(robust_steps) * float(wdro_robust_step_weighted_units)
    if total_weighted_units <= 0.0:
        return 0.0, float(wdro_robust_step_weighted_units)
    return float(baseline_weighted_units) / float(total_weighted_units), float(wdro_robust_step_weighted_units)


def _estimate_cdro_robust_step_batch_equiv(cfg) -> float:
    """Estimate CDRO robust-step compute in batch-equivalent denoiser evals."""

    path_steps = max(int(getattr(cfg, "n_steps_path", 0)), 0)
    if path_steps <= 0:
        return 0.0
    attack_num_steps, _ = _resolve_cdro_attack_num_steps(cfg)
    attack_enabled = float(getattr(cfg, "outer_attack_weight", 0.0)) > 0.0 and int(attack_num_steps) > 0
    clean_enabled = float(getattr(cfg, "outer_clean_weight", 0.0)) > 0.0
    rollout_multiplier = 2.0 if bool(getattr(cfg, "cdro_antithetic_rollouts", False)) else 1.0
    if not attack_enabled:
        return float(path_steps if clean_enabled else 0.0) * rollout_multiplier
    attack_construction_units = float(path_steps * max(int(attack_num_steps), 0)) * rollout_multiplier
    attack_eval_units = float(path_steps * 2) * rollout_multiplier
    clean_eval_units = float(path_steps if clean_enabled else 0) * rollout_multiplier
    return attack_construction_units + attack_eval_units + clean_eval_units


def _resolve_phase_steps(
    cfg,
    method_name: str,
    train_pool_size: Optional[int] = None,
    weighted_calibration: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Resolve baseline/robust step budgets while preserving legacy behavior."""

    total_steps = int(cfg.steps)
    baseline_steps_override = int(getattr(cfg, "baseline_steps_override", 0))
    baseline_steps = int(cfg.baseline_steps_override) if baseline_steps_override > 0 else total_steps
    robust_steps = total_steps
    split_mode = "baseline_steps_override" if baseline_steps_override > 0 else "legacy_equal_steps"

    wdro_warmup_compute_fraction = None
    wdro_robust_step_batch_equiv = None
    cdro_robust_step_batch_equiv = None
    wdro_warmup_weighted_compute_fraction = None
    wdro_robust_step_weighted_units = None
    cdro_robust_step_weighted_units = None

    if str(method_name).lower() == "wdro":
        split_mode = "wdro_baseline_steps_override" if baseline_steps_override > 0 else "wdro_paper_style"
        if baseline_steps_override > 0:
            baseline_steps = int(cfg.baseline_steps_override)
        else:
            baseline_steps = int(total_steps * float(cfg.wdro_warmup_fraction))
        baseline_steps = max(0, min(baseline_steps, total_steps))
        robust_steps = max(total_steps - baseline_steps, 0)
        wdro_warmup_compute_fraction, wdro_robust_step_batch_equiv = _estimate_wdro_warmup_compute_fraction(
            cfg,
            train_pool_size=train_pool_size,
        )
        if weighted_calibration is not None:
            wdro_warmup_weighted_compute_fraction, wdro_robust_step_weighted_units = (
                _estimate_wdro_warmup_weighted_compute_fraction(
                    cfg,
                    train_pool_size=train_pool_size,
                    calibration=weighted_calibration,
                )
            )
    elif str(method_name).lower() == "cdro":
        cdro_robust_step_batch_equiv = _estimate_cdro_robust_step_batch_equiv(cfg)
        cdro_attack_num_steps, _ = _resolve_cdro_attack_num_steps(cfg)
        if str(getattr(cfg, "training_objective", "edm")).strip().lower() == "rf":
            baseline_steps = 0
            robust_steps = total_steps
            split_mode = "cdro_rf_shared_edm_warm_start"
        else:
            split_mode = "cdro_baseline_steps_override" if baseline_steps_override > 0 else "cdro_fixed_fraction_warmup"
            if baseline_steps_override > 0:
                baseline_steps = int(cfg.baseline_steps_override)
            else:
                cdro_reference_warmup_fraction = float(getattr(cfg, "cdro_warmup_fraction", 0.0))
                wdro_warmup_compute_fraction, wdro_robust_step_batch_equiv = _estimate_wdro_warmup_compute_fraction(
                    cfg,
                    train_pool_size=train_pool_size,
                    warmup_fraction=cdro_reference_warmup_fraction,
                )
                if weighted_calibration is not None:
                    wdro_warmup_weighted_compute_fraction, wdro_robust_step_weighted_units = (
                        _estimate_wdro_warmup_weighted_compute_fraction(
                            cfg,
                            train_pool_size=train_pool_size,
                            calibration=weighted_calibration,
                            warmup_fraction=cdro_reference_warmup_fraction,
                        )
                    )
                    cdro_robust_step_weighted_units = cdro_robust_step_weighted_compute_units(
                        n_steps_path=int(getattr(cfg, "n_steps_path", 0)),
                        inner_steps=int(cdro_attack_num_steps),
                        outer_attack_weight=float(getattr(cfg, "outer_attack_weight", 0.0)),
                        outer_clean_weight=float(getattr(cfg, "outer_clean_weight", 0.0)),
                        antithetic_rollouts=bool(getattr(cfg, "cdro_antithetic_rollouts", False)),
                        calibration=weighted_calibration,
                    )
                    baseline_step_weighted_units = baseline_weighted_compute_units_for_steps(
                        steps=1,
                        calibration=weighted_calibration,
                    )
                    if (
                        wdro_warmup_weighted_compute_fraction is not None
                        and cdro_robust_step_weighted_units is not None
                        and baseline_step_weighted_units is not None
                    ):
                        baseline_steps = solve_warmup_steps_for_target_compute_fraction(
                            total_steps=int(total_steps),
                            target_warmup_compute_fraction=float(wdro_warmup_weighted_compute_fraction),
                            baseline_step_compute_units=float(baseline_step_weighted_units),
                            robust_step_compute_units=float(cdro_robust_step_weighted_units),
                        )
                        split_mode = "cdro_weighted_compute_matched_warmup"
                    else:
                        baseline_steps = int(total_steps * cdro_reference_warmup_fraction)
                        split_mode = "cdro_fixed_fraction_warmup_unweighted_fallback"
                else:
                    baseline_steps = int(total_steps * cdro_reference_warmup_fraction)
                    split_mode = "cdro_fixed_fraction_warmup_unweighted_fallback"
            baseline_steps = max(0, min(baseline_steps, total_steps))
            robust_steps = max(total_steps - baseline_steps, 0)

    return {
        "total_steps": int(total_steps),
        "baseline_steps": int(baseline_steps),
        "robust_steps": int(robust_steps),
        "split_mode": split_mode,
        "wdro_warmup_compute_fraction": (
            None if wdro_warmup_compute_fraction is None else float(wdro_warmup_compute_fraction)
        ),
        "wdro_warmup_weighted_compute_fraction": (
            None
            if wdro_warmup_weighted_compute_fraction is None
            else float(wdro_warmup_weighted_compute_fraction)
        ),
        "wdro_robust_step_batch_equiv": (
            None if wdro_robust_step_batch_equiv is None else float(wdro_robust_step_batch_equiv)
        ),
        "wdro_robust_step_weighted_compute_units": (
            None if wdro_robust_step_weighted_units is None else float(wdro_robust_step_weighted_units)
        ),
        "cdro_robust_step_batch_equiv": (
            None if cdro_robust_step_batch_equiv is None else float(cdro_robust_step_batch_equiv)
        ),
        "cdro_robust_step_weighted_compute_units": (
            None if cdro_robust_step_weighted_units is None else float(cdro_robust_step_weighted_units)
        ),
    }


def _resolve_baseline_ckpt_path(cfg, signature: Dict) -> str:
    """Resolve checkpoint path: explicit path or deterministic auto cache path."""

    explicit = str(getattr(cfg, "baseline_ckpt_path", "")).strip()
    if explicit:
        return os.path.abspath(explicit)
    sig_blob = json.dumps(signature, sort_keys=True, separators=(",", ":")).encode("utf-8")
    sig_hash = hashlib.sha1(sig_blob).hexdigest()[:16]
    cache_dir = os.path.join(cfg.outdir, "_baseline_cache")
    ensure_dir(cache_dir)
    return os.path.join(cache_dir, f"baseline_{sig_hash}.pt")


def _signature_mismatch_lines(current_signature: Dict, loaded_signature: Dict):
    """Return human-readable mismatch lines for strict checkpoint validation."""

    mismatches = []
    all_keys = sorted(set(current_signature.keys()) | set(loaded_signature.keys()))
    for key in all_keys:
        cur = current_signature.get(key)
        old = loaded_signature.get(key)
        if cur != old:
            mismatches.append(f"{key}: current={cur} saved={old}")
    return mismatches


def _load_baseline_checkpoint(
    *,
    baseline_model,
    ckpt_path: str,
    signature: Dict,
    strict_meta: bool,
):
    """Load baseline checkpoint if available and signature is compatible."""

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid baseline checkpoint format (expect dict): {ckpt_path}")

    # Support both native baseline checkpoints and the lighter checkpoint format
    # emitted by the MNIST sweep scripts (`state_dict` / `history` only).
    if "baseline_state_dict" in payload:
        state_dict = payload["baseline_state_dict"]
        history = payload.get("baseline_history", _empty_baseline_history())
    elif "state_dict" in payload:
        state_dict = payload["state_dict"]
        history = payload.get("history", _empty_baseline_history())
    else:
        raise RuntimeError(
            "Missing keys 'baseline_state_dict' and 'state_dict' in baseline checkpoint: "
            f"{ckpt_path}"
        )

    loaded_signature = payload.get("baseline_signature", {})
    if not isinstance(loaded_signature, dict):
        loaded_signature = {}

    mismatch_lines = _signature_mismatch_lines(signature, loaded_signature)
    if mismatch_lines:
        preview = "\n  - ".join(mismatch_lines[:12])
        if len(mismatch_lines) > 12:
            preview += f"\n  - ... ({len(mismatch_lines) - 12} more)"
        msg = (
            "Baseline checkpoint signature mismatch.\n"
            f"checkpoint={ckpt_path}\n"
            f"  - {preview}\n"
            "Use --baseline-ckpt-force-retrain or a different --baseline-ckpt-path."
        )
        if strict_meta:
            raise RuntimeError(msg)
        print(f"[warn] {msg}", flush=True)

    baseline_model.load_state_dict(state_dict, strict=True)
    if not isinstance(history, dict):
        history = _empty_baseline_history()
    history.setdefault("loss", [])
    history.setdefault("proxy_weighted_denoise_loss", [])
    history.setdefault("sigma_counts", [])
    ensure_denoiser_op_count_history(history)
    continuation_state = None
    continuation_source = "unavailable"
    expected_step = signature.get("baseline_steps")
    payload_step = payload.get("step")
    step_matches = expected_step is None or payload_step is None or int(payload_step) == int(expected_step)
    raw_state_dict = payload.get("model_state_dict")
    if not step_matches:
        continuation_source = "step_mismatch"
    elif isinstance(raw_state_dict, dict):
        continuation_state = {
            "model_state_dict": raw_state_dict,
            "optimizer_theta_state": payload.get("optimizer_state_dict"),
            "ema_state_dict": payload.get("ema_state_dict"),
            "rng_state": payload.get("rng_state"),
            "step": None if payload_step is None else int(payload_step),
        }
        continuation_source = "baseline_checkpoint_full_state"
    return {
        "history": history,
        "saved_at": payload.get("saved_at"),
        "signature": loaded_signature,
        "runtime": payload.get("baseline_runtime"),
        "continuation_state": continuation_state,
        "continuation_source": continuation_source,
    }


def _save_baseline_checkpoint(
    *,
    ckpt_path: str,
    baseline_eval,
    baseline_history: Dict,
    signature: Dict,
    baseline_runtime: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist baseline eval model + history + signature for later fair reuse."""

    ensure_dir(os.path.dirname(ckpt_path) or ".")
    payload = {
        "format": "toy_baseline_ckpt_v1",
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "baseline_signature": signature,
        "baseline_history": baseline_history,
        "baseline_state_dict": baseline_eval.state_dict(),
        "baseline_runtime": baseline_runtime,
    }
    tmp_path = f"{ckpt_path}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, ckpt_path)


def _load_robust_resume_checkpoint(*, ckpt_path: str, method_name: str) -> Dict[str, Any]:
    """Load a robust continuation checkpoint for chained step sweeps."""

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid robust resume checkpoint format (expect dict): {ckpt_path}")
    if "robust_state_dict" not in payload or "history_robust" not in payload or "trainer_state" not in payload:
        raise RuntimeError(
            "Missing one of ('robust_state_dict', 'history_robust', 'trainer_state') "
            f"in robust resume checkpoint: {ckpt_path}"
        )
    saved_method = str(payload.get("method_name", "")).lower()
    if saved_method and saved_method != str(method_name).lower():
        raise RuntimeError(
            f"Robust resume checkpoint method mismatch: current={method_name} saved={saved_method} ({ckpt_path})"
        )
    return payload


def _save_robust_resume_checkpoint(
    *,
    ckpt_path: str,
    cfg,
    method,
    baseline_eval,
    robust,
    control,
    history_robust: Dict[str, Any],
    trainer_state: Dict[str, Any],
    rng_state: Dict[str, Any],
    runtime_sec: Dict[str, Any],
    run_wall_start: str,
) -> None:
    """Persist robust training state so later runs can continue from this point."""

    ensure_dir(os.path.dirname(ckpt_path) or ".")
    resume_robust_state_dict = trainer_state.get("resume_robust_state_dict")
    if not isinstance(resume_robust_state_dict, dict):
        resume_robust_state_dict = robust.state_dict()

    payload = {
        "format": "toy_robust_resume_ckpt_v1",
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "method_name": str(getattr(method, "NAME", cfg.method_version)).lower(),
        "method_version": str(cfg.method_version),
        "completed_steps": int(trainer_state.get("completed_steps", cfg.steps)),
        "history_robust": history_robust,
        "baseline_state_dict": baseline_eval.state_dict(),
        "robust_state_dict": resume_robust_state_dict,
        "control_state_dict": control.state_dict(),
        "trainer_state": trainer_state,
        "rng_state": rng_state,
        "cumulative_runtime": {
            "baseline_phase": float(runtime_sec.get("baseline_phase", 0.0)),
            "baseline_train": float(runtime_sec.get("baseline_train", 0.0)),
            "total": float(runtime_sec.get("total", 0.0)),
            "total_without_fid": float(runtime_sec.get("total_without_fid", 0.0)),
            "robust_phase": float(runtime_sec.get("robust_phase", 0.0)),
            "fid_baseline": float(runtime_sec.get("fid_baseline", 0.0)),
            "fid_robust": float(runtime_sec.get("fid_robust", 0.0)),
            "run_started_utc": str(runtime_sec.get("run_started_utc", run_wall_start)),
        },
        "cumulative_accounting": {
            "baseline_images_seen_total": int(runtime_sec.get("baseline_images_seen_total", 0)),
            "baseline_batch_equiv_denoiser_evals_total": float(
                runtime_sec.get("baseline_batch_equiv_denoiser_evals_total", 0.0)
            ),
            "baseline_train_wall_clock_sec_effective": runtime_sec.get("baseline_train_wall_clock_sec_effective"),
            "baseline_weighted_counts": {
                "n_fwd": float(runtime_sec.get("baseline_weighted_n_fwd", 0.0)),
                "n_fwd_inputgrad": float(runtime_sec.get("baseline_weighted_n_fwd_inputgrad", 0.0)),
                "n_fwd_parambackward": float(runtime_sec.get("baseline_weighted_n_fwd_parambackward", 0.0)),
            },
            "baseline_weighted_compute_units": runtime_sec.get("baseline_weighted_compute_units"),
        },
    }
    tmp_path = f"{ckpt_path}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, ckpt_path)


def _build_baseline_gate(
    cfg,
    diagnostics,
    baseline_eval,
    control,
    sigma_levels: torch.Tensor,
    kappa_by_step: torch.Tensor,
    dataset: DatasetBundle,
    method,
    rollout_kwargs: Dict,
    amp_dtype,
) -> Dict:
    """Compute baseline acceptance gate metrics before robust phase.

    Gate is evaluated from:
    - endpoint quality after forward(ref)->reverse,
    - pure generative quality from Gaussian terminal noise.
    """

    n_steps = int(sigma_levels.numel() - 1)
    gate_terminal_step = cfg.debug_terminal_step
    if gate_terminal_step < 1 or gate_terminal_step > n_steps:
        gate_terminal_step = n_steps
    sigma_levels_gate = sigma_levels[: gate_terminal_step + 1]

    x_gate = dataset.sample_val_batch(cfg.debug_eval_batch)
    idx_gate = sample_target_indices(cfg.debug_eval_batch, sigma_levels)
    with autocast_context(x_gate.device, amp_dtype):
        gate_roll = _rollout_for_eval(
            cfg=cfg,
            method=method,
            x0=x_gate,
            target_indices=idx_gate,
            control=control,
            sigma_levels=sigma_levels,
            kappa_by_step=kappa_by_step,
            rollout_kwargs=rollout_kwargs,
            attack_net=baseline_eval,
        )
        gate_ref_paths = gate_roll.states_ref[:, : gate_terminal_step + 1]
        if str(getattr(cfg, "training_objective", "edm")).strip().lower() == "rf":
            sigma_gate = torch.full(
                (gate_ref_paths.shape[0],),
                float(sigma_levels_gate[-1].item()),
                device=x_gate.device,
                dtype=x_gate.dtype,
            )
            gate_endpoint = baseline_eval(gate_ref_paths[:, -1], sigma_gate)
        else:
            gate_rev_det = reverse_paths_from_terminal(
                denoiser=baseline_eval,
                x_terminal=gate_ref_paths[:, -1],
                sigma_levels=sigma_levels_gate,
                stochastic=False,
            )
            gate_endpoint = gate_rev_det[:, 0]
    gate_endpoint_mode_metrics = dataset.evaluate_sample_metrics(tensor_to_numpy(gate_endpoint))
    gate_endpoint_recovery_mse = float((gate_endpoint - x_gate).reshape(x_gate.shape[0], -1).pow(2).mean().item())
    with autocast_context(x_gate.device, amp_dtype):
        gate_gen_paths = sample_reverse_paths(
            denoiser=baseline_eval,
            sigma_levels=sigma_levels,
            n_samples=int(cfg.eval_samples),
            device=x_gate.device,
            stochastic=True,
            sample_terminal_batch_fn=dataset.sample_terminal_batch,
        )
    gate_generated = gate_gen_paths[:, generated_data_path_index_from_denoiser(baseline_eval)]
    gate_generated_mode_metrics = dataset.evaluate_sample_metrics(tensor_to_numpy(gate_generated))
    baseline_gate = diagnostics.build_baseline_gate(
        cfg,
        {
            "generated_metrics": gate_generated_mode_metrics,
            "endpoint_metrics": gate_endpoint_mode_metrics,
            "endpoint_recovery_mse_mean": gate_endpoint_recovery_mse,
            "sigma_terminal": float(sigma_levels_gate[-1].item()),
        },
    )
    return baseline_gate


def _run_robust_phase(
    cfg,
    robust,
    control,
    centers,
    sigma_levels,
    dataset: DatasetBundle,
    baseline_gate: Dict,
    method,
    robust_resume_payload: Optional[Dict[str, Any]] = None,
    baseline_handoff_state: Optional[Dict[str, Any]] = None,
    return_trainer_state: bool = False,
):
    """Execute or skip robust training depending on baseline-only mode and gate status."""

    attack_training_executed = False
    if cfg.baseline_only:
        with torch.no_grad():
            for p in control.parameters():
                p.zero_()
        return False, empty_robust_history(), robust, None

    if cfg.baseline_gate_enabled and not baseline_gate["passed"]:
        reason_lines = [
            f"{c['name']}: value={c['value']:.4f} {c['op']} {c['threshold']:.4f} (FAIL)"
            for c in baseline_gate["failed_checks"]
        ]
        msg = "Baseline acceptance gate failed. Attack training skipped.\n  - " + "\n  - ".join(reason_lines)
        if cfg.baseline_gate_error_on_fail:
            raise RuntimeError(msg)
        print(f"[gate] {msg}", flush=True)
        with torch.no_grad():
            for p in control.parameters():
                p.zero_()
        return False, empty_robust_history(), robust, None

    if int(cfg.steps) <= 0:
        with torch.no_grad():
            for p in control.parameters():
                p.zero_()
        return False, empty_robust_history(), robust, None

    trainer_kwargs: Dict[str, Any] = {}
    method_name = str(getattr(method, "NAME", cfg.method_version)).lower()
    if robust_resume_payload is not None:
        trainer_state_in = robust_resume_payload.get("trainer_state", {})
        completed_steps = int(
            robust_resume_payload.get("completed_steps", trainer_state_in.get("completed_steps", 0))
        )
        history_init = robust_resume_payload.get("history_robust", empty_robust_history())
        if completed_steps >= int(cfg.steps):
            return True, history_init, robust, trainer_state_in
        if method_name in ("clean", "v2", "wild", "wdro", "v1.1", "1.1", "cdro"):
            trainer_kwargs["start_step"] = int(completed_steps)
            trainer_kwargs["history_state"] = history_init
            trainer_kwargs["return_state"] = bool(return_trainer_state)
            if method_name == "v2":
                trainer_kwargs["optimizer_theta_state"] = trainer_state_in.get("optimizer_theta_state")
                trainer_kwargs["optimizer_phi_state"] = trainer_state_in.get("optimizer_phi_state")
            elif method_name in ("clean", "wdro", "cdro"):
                trainer_kwargs["optimizer_theta_state"] = trainer_state_in.get("optimizer_theta_state")
                trainer_kwargs["ema_state_dict"] = trainer_state_in.get("ema_state_dict")
                if method_name == "cdro":
                    trainer_kwargs["rf_reflow_teacher_state_dict"] = trainer_state_in.get("rf_reflow_teacher_state_dict")
            else:
                trainer_kwargs["optimizer_theta_state"] = trainer_state_in.get("optimizer_theta_state")
        else:
            raise RuntimeError(f"Robust resume is not implemented for method_version='{cfg.method_version}'.")
    else:
        if return_trainer_state and method_name in ("clean", "v2", "wild", "wdro", "v1.1", "1.1", "cdro"):
            trainer_kwargs["return_state"] = True
        if baseline_handoff_state is not None and method_name in ("clean", "wdro", "cdro"):
            trainer_kwargs["optimizer_theta_state"] = baseline_handoff_state.get("optimizer_theta_state")
            trainer_kwargs["ema_state_dict"] = baseline_handoff_state.get("ema_state_dict")

    attack_training_executed = True
    train_result = method.train_trajectory_robust(
        robust,
        control,
        centers,
        sigma_levels,
        cfg,
        train_pool=dataset.train_pool,
        sample_train_batch_fn=dataset.sample_train_batch,
        sample_population_batch_fn=dataset.sample_population_batch,
        **trainer_kwargs,
    )
    if return_trainer_state and isinstance(train_result, tuple):
        history_robust, trainer_state = train_result
    else:
        history_robust = train_result
        trainer_state = None
    return attack_training_executed, history_robust, robust, trainer_state


def run_experiment(cfg) -> dict:
    """End-to-end toy experiment pipeline.

    Steps:
    1) build data + sigma ladder,
    2) train baseline and evaluate gate,
    3) run robust phase if allowed,
    4) compute diagnostics/plots/metrics and persist outputs.
    """

    device = pick_device(cfg.device)
    configure_runtime(
        device=device,
        allow_tf32=bool(getattr(cfg, "allow_tf32", True)),
        cudnn_benchmark=bool(getattr(cfg, "cudnn_benchmark", True)),
    )
    amp_dtype = resolve_amp_dtype(device, getattr(cfg, "amp_dtype", "auto"))
    set_seed(cfg.seed)
    run_t0 = time.perf_counter()
    run_wall_start = datetime.now(timezone.utc).isoformat()
    runtime_sec = {
        "preflight": 0.0,
        "baseline_phase": 0.0,
        "baseline_ckpt_load": 0.0,
        "baseline_train": 0.0,
        "baseline_ckpt_save": 0.0,
        "baseline_gate_eval": 0.0,
        "robust_phase": 0.0,
        "post_train_eval": 0.0,
        "fid_baseline": 0.0,
        "fid_robust": 0.0,
        "plotting_and_persist": 0.0,
    }
    accelerator_meta = _device_accounting_metadata(device)
    weighted_calibration = load_weighted_compute_calibration(
        calibration_path=str(getattr(cfg, "weighted_compute_calibration_path", "")).strip(),
        inputgrad_alpha=float(getattr(cfg, "weighted_inputgrad_alpha", 0.0)),
        parambackward_beta=float(getattr(cfg, "weighted_parambackward_beta", 0.0)),
    )

    ensure_dir(cfg.outdir)
    exp_dir = os.path.join(cfg.outdir, cfg.exp_name)
    ensure_dir(exp_dir)
    flow_mode = (
        "baseline_only"
        if cfg.baseline_only
        else ("robust_forced_no_gate" if not cfg.baseline_gate_enabled else "robust_with_gate")
    )
    method = resolve_method_module(cfg.method_version)
    method_name = str(getattr(method, "NAME", cfg.method_version)).lower()
    is_cdro_rf = _is_cdro_rf_port(cfg, method_name)
    rollout_kwargs = _method_rollout_kwargs(cfg, method)
    if not getattr(method, "IMPLEMENTED", True):
        raise NotImplementedError(
            f"method_version='{cfg.method_version}' is marked IMPLEMENTED=False. "
            "Please implement its rollout/train API under toy/versions/<version>/."
        )
    print(f"[info] device={device}", flush=True)
    print(
        "[info] train_hardware "
        f"accelerator_kind={accelerator_meta['train_accelerator_kind']} "
        f"accelerator_name={accelerator_meta['train_accelerator_name']} "
        f"accelerator_count={accelerator_meta['train_accelerator_count']}",
        flush=True,
    )
    print(
        "[info] runtime "
        f"allow_tf32={bool(getattr(cfg, 'allow_tf32', True))} "
        f"cudnn_benchmark={bool(getattr(cfg, 'cudnn_benchmark', True))} "
        f"amp_dtype={format_amp_dtype(amp_dtype)}",
        flush=True,
    )
    print(
        "[info] weighted_compute "
        f"available={weighted_calibration['available']} "
        f"source={weighted_calibration['source']} "
        f"alpha={weighted_calibration['inputgrad_alpha']} "
        f"beta={weighted_calibration['parambackward_beta']}",
        flush=True,
    )
    print(f"[info] exp_dir={exp_dir}", flush=True)
    print(
        "[info] flow_mode="
        f"{flow_mode} baseline_gate_enabled={cfg.baseline_gate_enabled} "
        f"outer_attack_weight={cfg.outer_attack_weight} outer_clean_weight={cfg.outer_clean_weight} "
        f"method_version={cfg.method_version}",
        flush=True,
    )
    gate_eval_seed = int(cfg.seed + cfg.eval_seed_offset_gate)
    metrics_eval_seed = int(cfg.seed + cfg.eval_seed_offset_metrics)
    print(
        "[info] fair_eval "
        f"gate_seed={gate_eval_seed} metrics_seed={metrics_eval_seed} "
        f"shared_terminal_noise={cfg.eval_use_shared_terminal_noise} "
        f"shared_reverse_noise={cfg.eval_use_shared_reverse_noise}",
        flush=True,
    )

    dataset = build_dataset_bundle(cfg, device)
    diagnostics = build_diagnostics_bundle(cfg, dataset)
    centers = dataset.centers
    train_pool_size = int(dataset.train_pool.shape[0]) if dataset.train_pool is not None else None
    phase_steps = _resolve_phase_steps(
        cfg,
        method_name,
        train_pool_size=train_pool_size,
        weighted_calibration=weighted_calibration,
    )
    baseline_steps_for_phase = int(phase_steps["baseline_steps"])
    robust_steps_for_phase = int(phase_steps["robust_steps"])
    cfg_baseline = replace(cfg, steps=baseline_steps_for_phase)
    cfg_robust = replace(cfg, steps=robust_steps_for_phase)
    rf_cdro_eval_stage = None
    rf_cdro_stage1_t_distribution = None
    rf_cdro_reflow_t_distribution = None
    rf_cdro_eval_t_distribution = None
    if _is_cdro_rf_port(cfg, method_name):
        rf_cdro_eval_stage = _resolve_rf_cdro_eval_stage(cfg_robust, robust_steps_for_phase)
        rf_cdro_stage1_t_distribution = resolve_rf_stage_t_distribution(
            "rf_stage1",
            reflow_distribution=str(getattr(cfg, "rf_reflow_t_distribution", "u_shaped")),
        )
        rf_cdro_reflow_t_distribution = resolve_rf_stage_t_distribution(
            "rf_reflow",
            reflow_distribution=str(getattr(cfg, "rf_reflow_t_distribution", "u_shaped")),
        )
        rf_cdro_eval_t_distribution = resolve_rf_stage_t_distribution(
            rf_cdro_eval_stage,
            reflow_distribution=str(getattr(cfg, "rf_reflow_t_distribution", "u_shaped")),
        )
    if cfg.sigma_data <= 0:
        cfg.sigma_data = dataset.estimate_sigma_data()
    print(f"[info] sigma_data={cfg.sigma_data:.6f}", flush=True)
    _print_dataset_info(cfg, dataset)

    baseline_sigma_levels = _build_family_sigma_levels(cfg, device)
    sigma_levels = baseline_sigma_levels
    if method_name == "cdro":
        if str(getattr(cfg, "training_objective", "edm")).strip().lower() == "rf":
            sigma_levels = build_rf_stage_time_quantile_levels(
                float(cfg.sigma_max),
                int(cfg.n_steps_path),
                device=device,
                stage_name=str(rf_cdro_eval_stage),
                reflow_distribution=str(getattr(cfg, "rf_reflow_t_distribution", "u_shaped")),
                quantile_rule=str(getattr(cfg, "rf_cdro_quantile_rule", "right_endpoint")),
            )
        else:
            sigma_levels = build_sigma_levels_from_warmup_quantiles(
                cfg.sigma_min,
                cfg.sigma_max,
                cfg.n_steps_path,
                device=device,
                p_mean=cfg.p_mean,
                p_std=cfg.p_std,
            )
    kappa_by_step = method.build_kappa_schedule(
        sigma_levels=sigma_levels,
        base_kappa=cfg.control_radius_kappa,
        use_time_dependent=cfg.use_time_dependent_kappa,
        low_multiplier=cfg.kappa_low_multiplier,
        mid_multiplier=cfg.kappa_mid_multiplier,
        high_multiplier=cfg.kappa_high_multiplier,
        preserve_l2_budget=cfg.kappa_preserve_l2_budget,
    ).to(device=sigma_levels.device, dtype=sigma_levels.dtype)
    constraint_radius_builder = getattr(method, "build_constraint_radii", None)
    constraint_radius_by_step = None
    constraint_radius_source = "kappa_delta_sigma"
    if callable(constraint_radius_builder):
        constraint_radius_by_step = constraint_radius_builder(cfg=cfg, sigma_levels=sigma_levels).to(
            device=sigma_levels.device,
            dtype=sigma_levels.dtype,
        )
        constraint_radius_source = "method_override"

    model_bundle = build_model_bundle(cfg, dataset, sigma_data=cfg.sigma_data, device=device)
    baseline = model_bundle.baseline
    robust = model_bundle.robust
    control = model_bundle.control
    rf_edm_init_report = _maybe_initialize_rf_from_edm_checkpoint(cfg, model_bundle)
    if is_cdro_rf and not rf_edm_init_report.get("enabled"):
        raise RuntimeError(
            "CDRO-RF requires a shared EDM warm-start checkpoint. "
            "Set --rf-edm-init-ckpt-path to the shared 5% EDM warm-start artifact."
        )
    if rf_edm_init_report.get("enabled"):
        print(
            "[rf-init] loaded EDM warm start "
            f"path={rf_edm_init_report['path']} "
            f"source={rf_edm_init_report['source_state_key']} "
            f"baseline_transferred={rf_edm_init_report['baseline']['transferred_count']} "
            f"robust_transferred={rf_edm_init_report['robust']['transferred_count']}",
            flush=True,
        )

    check_report = {}
    if cfg.run_checks:
        t_phase = time.perf_counter()
        check_report = run_preflight_checks(
            robust,
            control,
            centers,
            sigma_levels,
            cfg,
            sample_batch_fn=dataset.sample_population_batch,
            method=method,
        )
        runtime_sec["preflight"] = float(time.perf_counter() - t_phase)
        print("[check] preflight passed", flush=True)
        print(
            f"[check] grad_rel_err denoiser={check_report['gradcheck_denoiser']['relative_error']:.3e} "
            f"control={check_report['gradcheck_control']['relative_error']:.3e}",
            flush=True,
        )

    baseline_signature = _build_baseline_signature(cfg_baseline, dataset, model_bundle, baseline_sigma_levels)
    baseline_signature_hash = hashlib.sha1(
        json.dumps(baseline_signature, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    baseline_ckpt_path = _resolve_baseline_ckpt_path(cfg, baseline_signature)
    baseline_ckpt_enabled = bool(getattr(cfg, "baseline_ckpt_enabled", True)) and not is_cdro_rf
    baseline_ckpt_force_retrain = bool(getattr(cfg, "baseline_ckpt_force_retrain", False))
    baseline_ckpt_strict_meta = bool(getattr(cfg, "baseline_ckpt_strict_meta", True))
    baseline_ckpt_loaded = False
    baseline_ckpt_saved = False
    baseline_ckpt_saved_at = None
    baseline_runtime_from_ckpt = None
    baseline_handoff_state = None
    baseline_handoff_source = "unavailable"
    baseline_handoff_rng_restored = False
    baseline_reference_runtime = {"train_wall_clock_sec": None, "source": "unavailable"}
    robust_resume_payload = None
    robust_resume_loaded = False
    robust_resume_completed_steps = 0
    robust_resume_runtime = {}
    robust_resume_accounting = {}
    robust_resume_path = str(getattr(cfg, "robust_resume_ckpt_path", "")).strip()
    robust_save_path = str(getattr(cfg, "robust_save_ckpt_path", "")).strip()
    t_baseline_phase = time.perf_counter()
    baseline_restored_from_robust_resume = False

    if robust_resume_path:
        robust_resume_payload = _load_robust_resume_checkpoint(
            ckpt_path=robust_resume_path,
            method_name=str(getattr(method, "NAME", cfg.method_version)).lower(),
        )
        baseline_state_dict = robust_resume_payload.get("baseline_state_dict")
        if isinstance(baseline_state_dict, dict):
            baseline.load_state_dict(baseline_state_dict, strict=True)
            history_baseline = _empty_baseline_history()
            baseline_eval = baseline
            baseline_restored_from_robust_resume = True
            print(f"[baseline] restored from robust resume: {robust_resume_path}", flush=True)

    if (
        not baseline_restored_from_robust_resume
        and baseline_ckpt_enabled
        and os.path.isfile(baseline_ckpt_path)
        and not baseline_ckpt_force_retrain
    ):
        t_phase = time.perf_counter()
        load_info = _load_baseline_checkpoint(
            baseline_model=baseline,
            ckpt_path=baseline_ckpt_path,
            signature=baseline_signature,
            strict_meta=baseline_ckpt_strict_meta,
        )
        runtime_sec["baseline_ckpt_load"] += float(time.perf_counter() - t_phase)
        if load_info is not None:
            history_baseline = load_info["history"]
            baseline_eval = baseline
            baseline_ckpt_loaded = True
            baseline_ckpt_saved_at = load_info.get("saved_at")
            baseline_runtime_from_ckpt = load_info.get("runtime")
            baseline_handoff_state = load_info.get("continuation_state")
            baseline_handoff_source = str(load_info.get("continuation_source", "unavailable"))
            print(f"[baseline] loaded checkpoint: {baseline_ckpt_path}", flush=True)
    if not baseline_restored_from_robust_resume and not baseline_ckpt_loaded:
        t_phase = time.perf_counter()
        if baseline_steps_for_phase > 0:
            history_baseline, baseline_eval = train_baseline(
                baseline,
                centers,
                baseline_sigma_levels,
                cfg_baseline,
                train_pool=dataset.train_pool,
                sample_train_batch_fn=dataset.sample_train_batch,
                sample_population_batch_fn=dataset.sample_population_batch,
                sample_terminal_batch_fn=dataset.sample_terminal_batch,
            )
            runtime_sec["baseline_train"] += float(time.perf_counter() - t_phase)
            if baseline_ckpt_enabled:
                t_phase = time.perf_counter()
                _save_baseline_checkpoint(
                    ckpt_path=baseline_ckpt_path,
                    baseline_eval=baseline_eval,
                    baseline_history=history_baseline,
                    signature=baseline_signature,
                    baseline_runtime={
                        "step": int(baseline_steps_for_phase),
                        "train_wall_clock_sec": float(runtime_sec["baseline_train"]),
                    },
                )
                runtime_sec["baseline_ckpt_save"] += float(time.perf_counter() - t_phase)
                baseline_ckpt_saved = True
                print(f"[baseline] saved checkpoint: {baseline_ckpt_path}", flush=True)
        else:
            history_baseline = _empty_baseline_history()
            baseline_eval = baseline
    runtime_sec["baseline_phase"] = float(time.perf_counter() - t_baseline_phase)
    baseline_eval.eval()
    if isinstance(baseline_handoff_state, dict) and isinstance(baseline_handoff_state.get("model_state_dict"), dict):
        robust.load_state_dict(baseline_handoff_state["model_state_dict"], strict=True)
        print(
            "[baseline->robust] restored raw handoff state "
            f"source={baseline_handoff_source} step={baseline_handoff_state.get('step')}",
            flush=True,
        )
    else:
        robust.load_state_dict(baseline_eval.state_dict())
    if robust_resume_path:
        robust.load_state_dict(robust_resume_payload["robust_state_dict"], strict=True)
        if "control_state_dict" in robust_resume_payload:
            control.load_state_dict(robust_resume_payload["control_state_dict"], strict=True)
        robust_resume_loaded = True
        robust_resume_completed_steps = int(
            robust_resume_payload.get(
                "completed_steps",
                robust_resume_payload.get("trainer_state", {}).get("completed_steps", 0),
            )
        )
        robust_resume_runtime = robust_resume_payload.get("cumulative_runtime", {})
        robust_resume_accounting = robust_resume_payload.get("cumulative_accounting", {})
        if robust_resume_payload.get("rng_state") is not None:
            _restore_rng_state(robust_resume_payload["rng_state"])
        baseline_gate = {
            "passed": True,
            "resumed_from_ckpt": True,
            "resume_completed_steps": int(robust_resume_completed_steps),
        }
        baseline_gate["eval_seed"] = int(gate_eval_seed)
        baseline_gate["eval_seed_scoped"] = False
    elif is_cdro_rf and baseline_steps_for_phase <= 0:
        baseline_gate = {
            "passed": True,
            "skipped_for_cdro_rf_warm_start": True,
            "checks": [],
            "failed_checks": [],
        }
        baseline_gate["eval_seed"] = int(gate_eval_seed)
        baseline_gate["eval_seed_scoped"] = False
    else:
        t_phase = time.perf_counter()

        def _baseline_gate_eval():
            return _build_baseline_gate(
                cfg=cfg,
                diagnostics=diagnostics,
                baseline_eval=baseline_eval,
                control=control,
                sigma_levels=sigma_levels,
                kappa_by_step=kappa_by_step,
                dataset=dataset,
                method=method,
                rollout_kwargs=rollout_kwargs,
                amp_dtype=amp_dtype,
            )

        baseline_gate = _run_with_scoped_seed(
            gate_eval_seed,
            _baseline_gate_eval,
        )
        runtime_sec["baseline_gate_eval"] = float(time.perf_counter() - t_phase)
        baseline_gate["eval_seed"] = int(gate_eval_seed)
        baseline_gate["eval_seed_scoped"] = True
        if isinstance(baseline_handoff_state, dict) and baseline_handoff_state.get("rng_state") is not None:
            _restore_rng_state(baseline_handoff_state["rng_state"])
            baseline_handoff_rng_restored = True

    t_phase = time.perf_counter()
    attack_training_executed, history_robust, robust, robust_trainer_state = _run_robust_phase(
        cfg=cfg_robust,
        robust=robust,
        control=control,
        centers=centers,
        sigma_levels=sigma_levels,
        dataset=dataset,
        baseline_gate=baseline_gate,
        method=method,
        robust_resume_payload=robust_resume_payload,
        baseline_handoff_state=baseline_handoff_state,
        return_trainer_state=bool(robust_save_path),
    )
    runtime_sec["robust_phase"] = float(time.perf_counter() - t_phase)
    if not attack_training_executed:
        robust = baseline_eval

    resume_rng_state = _capture_rng_state()

    # Re-seed post-training diagnostics to keep v2/v2.1 comparisons reproducible.
    set_seed(metrics_eval_seed)
    t_post_eval = time.perf_counter()

    # Evaluate denoise curves on train-pool and held-out pools to expose overfitting under limited data.
    x_train_eval = dataset.sample_train_batch(cfg.debug_eval_batch)
    idx_train_eval = sample_target_indices(cfg.debug_eval_batch, sigma_levels)
    with autocast_context(device, amp_dtype):
        train_roll = _rollout_for_eval(
            cfg=cfg,
            method=method,
            x0=x_train_eval,
            target_indices=idx_train_eval,
            control=control,
            sigma_levels=sigma_levels,
            kappa_by_step=kappa_by_step,
            rollout_kwargs=rollout_kwargs,
            attack_net=robust,
        )
        denoise_error_curves_train = compute_denoise_error_curves(
            baseline_model=baseline_eval,
            robust_model=robust,
            x0=x_train_eval,
            states_ref=train_roll.states_ref,
            states_ctrl=train_roll.states_ctrl,
            sigma_levels=sigma_levels,
        )

    x_val_eval = dataset.sample_val_batch(cfg.debug_eval_batch)
    idx_val_eval = sample_target_indices(cfg.debug_eval_batch, sigma_levels)
    with autocast_context(device, amp_dtype):
        val_roll = _rollout_for_eval(
            cfg=cfg,
            method=method,
            x0=x_val_eval,
            target_indices=idx_val_eval,
            control=control,
            sigma_levels=sigma_levels,
            kappa_by_step=kappa_by_step,
            rollout_kwargs=rollout_kwargs,
            attack_net=robust,
        )
        denoise_error_curves_val = compute_denoise_error_curves(
            baseline_model=baseline_eval,
            robust_model=robust,
            x0=x_val_eval,
            states_ref=val_roll.states_ref,
            states_ctrl=val_roll.states_ctrl,
            sigma_levels=sigma_levels,
        )
    denoise_error_curves = denoise_error_curves_val
    x_demo = x_val_eval
    demo_roll = val_roll
    n_steps = int(sigma_levels.numel() - 1)
    terminal_step = cfg.debug_terminal_step
    if terminal_step < 1 or terminal_step > n_steps:
        terminal_step = n_steps
    sigma_levels_plot = sigma_levels[: terminal_step + 1]
    ref_paths_plot = demo_roll.states_ref[:, : terminal_step + 1]
    ctrl_paths_plot = demo_roll.states_ctrl[:, : terminal_step + 1]
    shared_reverse_noise = None
    if cfg.plot_stochastic_backward:
        # Fair comparison: baseline/attack reverse use the same stochastic increments.
        shared_reverse_noise = torch.randn(
            (terminal_step + 1, ref_paths_plot.shape[0], *ref_paths_plot.shape[2:]),
            device=device,
            dtype=ref_paths_plot.dtype,
        )
    if str(getattr(cfg, "training_objective", "edm")).strip().lower() == "rf":
        rev_baseline_from_ref = ref_paths_plot.clone()
        rev_baseline_from_attack = ctrl_paths_plot.clone()
        rev_baseline_from_ref_plot = rev_baseline_from_ref
        rev_baseline_from_attack_plot = rev_baseline_from_attack
    else:
        # Deterministic reverse for pairwise-recovery diagnostics.
        with autocast_context(device, amp_dtype):
            rev_baseline_from_ref = reverse_paths_from_terminal(
                denoiser=baseline_eval,
                x_terminal=ref_paths_plot[:, -1],
                sigma_levels=sigma_levels_plot,
                stochastic=False,
            )
            rev_baseline_from_attack = reverse_paths_from_terminal(
                denoiser=baseline_eval,
                x_terminal=ctrl_paths_plot[:, -1],
                sigma_levels=sigma_levels_plot,
                stochastic=False,
            )
            # Optional stochastic reverse for visualization of clustered generative behavior.
            rev_baseline_from_ref_plot = reverse_paths_from_terminal(
                denoiser=baseline_eval,
                x_terminal=ref_paths_plot[:, -1],
                sigma_levels=sigma_levels_plot,
                stochastic=cfg.plot_stochastic_backward,
                noise_schedule=shared_reverse_noise,
            )
            rev_baseline_from_attack_plot = reverse_paths_from_terminal(
                denoiser=baseline_eval,
                x_terminal=ctrl_paths_plot[:, -1],
                sigma_levels=sigma_levels_plot,
                stochastic=cfg.plot_stochastic_backward,
                noise_schedule=shared_reverse_noise,
            )
    paired_reverse_delta_plot = compute_paired_reverse_delta_by_step(
        reverse_paths_ref=rev_baseline_from_ref_plot,
        reverse_paths_attack=rev_baseline_from_attack_plot,
    )
    paired_reverse_delta_det = compute_paired_reverse_delta_by_step(
        reverse_paths_ref=rev_baseline_from_ref,
        reverse_paths_attack=rev_baseline_from_attack,
    )
    rev_baseline_from_ref_np = tensor_to_numpy(rev_baseline_from_ref_plot)
    rev_baseline_from_attack_np = tensor_to_numpy(rev_baseline_from_attack_plot)

    terminal_consistency = {
        "tolerance": float(cfg.reverse_terminal_assert_tol),
        "plot_ref": compute_terminal_match_stats(rev_baseline_from_ref_plot, ref_paths_plot),
        "plot_attack": compute_terminal_match_stats(rev_baseline_from_attack_plot, ctrl_paths_plot),
        "det_ref": compute_terminal_match_stats(rev_baseline_from_ref, ref_paths_plot),
        "det_attack": compute_terminal_match_stats(rev_baseline_from_attack, ctrl_paths_plot),
    }
    terminal_consistency["passed"] = bool(
        terminal_consistency["plot_ref"]["max_l2"] <= cfg.reverse_terminal_assert_tol
        and terminal_consistency["plot_attack"]["max_l2"] <= cfg.reverse_terminal_assert_tol
        and terminal_consistency["det_ref"]["max_l2"] <= cfg.reverse_terminal_assert_tol
        and terminal_consistency["det_attack"]["max_l2"] <= cfg.reverse_terminal_assert_tol
    )
    if not terminal_consistency["passed"]:
        raise RuntimeError(
            "Reverse terminal consistency check failed: "
            f"plot_ref.max_l2={terminal_consistency['plot_ref']['max_l2']:.3e}, "
            f"plot_attack.max_l2={terminal_consistency['plot_attack']['max_l2']:.3e}, "
            f"det_ref.max_l2={terminal_consistency['det_ref']['max_l2']:.3e}, "
            f"det_attack.max_l2={terminal_consistency['det_attack']['max_l2']:.3e}, "
            f"tol={cfg.reverse_terminal_assert_tol:.3e}"
        )

    with autocast_context(device, amp_dtype):
        recovery_ref_curve = compute_x0_recovery_vs_terminal_step(
            denoiser=baseline_eval,
            forward_paths=demo_roll.states_ref,
            sigma_levels=sigma_levels,
            reverse_fn=reverse_paths_from_terminal,
            target_x=x_demo,
        )
        recovery_attack_curve = compute_x0_recovery_vs_terminal_step(
            denoiser=baseline_eval,
            forward_paths=demo_roll.states_ctrl,
            sigma_levels=sigma_levels,
            reverse_fn=reverse_paths_from_terminal,
            target_x=x_demo,
        )
    bayes_terminal_mse = diagnostics.estimate_bayes_terminal_mse(
        dataset,
        cfg,
        sigma=float(sigma_levels[terminal_step].item()),
    )
    shared_gen_terminal = None
    if cfg.eval_use_shared_terminal_noise:
        shared_gen_terminal = _sample_terminal_batch_for_objective(
            cfg=cfg,
            dataset=dataset,
            batch_size=int(cfg.eval_samples),
            sigma_levels=sigma_levels,
        )
    shared_gen_reverse_noise = None
    if cfg.eval_use_shared_reverse_noise:
        noise_ref = shared_gen_terminal
        if noise_ref is None:
            noise_ref = _sample_terminal_batch_for_objective(
                cfg=cfg,
                dataset=dataset,
                batch_size=int(cfg.eval_samples),
                sigma_levels=sigma_levels,
            )
        shared_gen_reverse_noise = torch.randn(
            (sigma_levels.numel(), noise_ref.shape[0], *noise_ref.shape[1:]),
            device=noise_ref.device,
            dtype=noise_ref.dtype,
        )
    eval_objective = str(getattr(cfg, "training_objective", "edm")).strip().lower()
    if shared_gen_terminal is not None and eval_objective == "rf":
        with autocast_context(device, amp_dtype):
            baseline_gen_paths = sample_rectified_flow_paths_from_source(
                denoiser=baseline_eval,
                x_source=shared_gen_terminal,
                sigma_levels=sigma_levels,
            )
    elif shared_gen_terminal is not None:
        with autocast_context(device, amp_dtype):
            baseline_gen_paths = reverse_paths_from_terminal(
                denoiser=baseline_eval,
                x_terminal=shared_gen_terminal,
                sigma_levels=sigma_levels,
                stochastic=True,
                noise_schedule=shared_gen_reverse_noise,
            )
    else:
        with autocast_context(device, amp_dtype):
            baseline_gen_paths = sample_reverse_paths(
                denoiser=baseline_eval,
                sigma_levels=sigma_levels,
                n_samples=int(cfg.eval_samples),
                device=device,
                stochastic=True,
                sample_terminal_batch_fn=dataset.sample_terminal_batch,
            )
    baseline_gen_np = tensor_to_numpy(
        baseline_gen_paths[:, generated_data_path_index_from_denoiser(baseline_eval)]
    )
    if shared_gen_terminal is not None and eval_objective == "rf":
        with autocast_context(device, amp_dtype):
            robust_gen_paths = sample_rectified_flow_paths_from_source(
                denoiser=robust,
                x_source=shared_gen_terminal,
                sigma_levels=sigma_levels,
            )
    elif shared_gen_terminal is not None:
        with autocast_context(device, amp_dtype):
            robust_gen_paths = reverse_paths_from_terminal(
                denoiser=robust,
                x_terminal=shared_gen_terminal,
                sigma_levels=sigma_levels,
                stochastic=True,
                noise_schedule=shared_gen_reverse_noise,
            )
    else:
        with autocast_context(device, amp_dtype):
            robust_gen_paths = sample_reverse_paths(
                denoiser=robust,
                sigma_levels=sigma_levels,
                n_samples=int(cfg.eval_samples),
                device=device,
                stochastic=True,
                sample_terminal_batch_fn=dataset.sample_terminal_batch,
            )
    robust_gen_np = tensor_to_numpy(
        robust_gen_paths[:, generated_data_path_index_from_denoiser(robust)]
    )
    reverse_endpoint_idx = generated_data_path_index_from_denoiser(baseline_eval)
    runtime_sec["post_train_eval"] = float(time.perf_counter() - t_post_eval)
    val_pool_np = tensor_to_numpy(dataset.val_pool)
    train_pool_np = tensor_to_numpy(dataset.train_pool) if dataset.train_pool is not None else None
    enable_ref_dist = bool(dataset.metadata.get("enable_nearest_reference_distance", True))
    denoise_gap = summarize_train_vs_val_curve_gaps(denoise_error_curves_train, denoise_error_curves_val)
    attack_gap_windows_train = summarize_attack_gap_windows(denoise_error_curves_train)
    attack_gap_windows_val = summarize_attack_gap_windows(denoise_error_curves_val)
    constraint_stats_train = compute_control_constraint_stats_by_step(
        delta_path=train_roll.delta_path,
        sigma_levels=sigma_levels,
        control_radius_kappa=cfg.control_radius_kappa,
        kappa_by_step=kappa_by_step,
        radius_override=constraint_radius_by_step,
        states_ref=train_roll.states_ref,
        states_ctrl=train_roll.states_ctrl,
        saturation_threshold=cfg.constraint_saturation_threshold,
    )
    constraint_stats_val = compute_control_constraint_stats_by_step(
        delta_path=val_roll.delta_path,
        sigma_levels=sigma_levels,
        control_radius_kappa=cfg.control_radius_kappa,
        kappa_by_step=kappa_by_step,
        radius_override=constraint_radius_by_step,
        states_ref=val_roll.states_ref,
        states_ctrl=val_roll.states_ctrl,
        saturation_threshold=cfg.constraint_saturation_threshold,
    )
    diag_gap_ratio_summary = summarize_series(history_robust.get("diag_inner_obj_gap_ratio", []))
    diag_delta_ratio_summary = summarize_series(history_robust.get("diag_delta_norm_ratio_mean", []))
    robust_batch_equiv_step_summary = summarize_series(history_robust.get("batch_equiv_denoiser_evals_step", []))
    robust_batch_equiv_cumulative_summary = summarize_series(
        history_robust.get("batch_equiv_denoiser_evals_cumulative", [])
    )
    collapse_suspected = None
    if diag_gap_ratio_summary["mean_last"] is not None and diag_delta_ratio_summary["mean_last"] is not None:
        collapse_suspected = bool(
            abs(diag_gap_ratio_summary["mean_last"]) <= float(cfg.collapse_gap_ratio_tol)
            and diag_delta_ratio_summary["mean_last"] <= float(cfg.collapse_delta_ratio_tol)
        )
    if baseline_ckpt_loaded and baseline_steps_for_phase > 0:
        baseline_runtime_value = None
        if isinstance(baseline_runtime_from_ckpt, dict):
            baseline_runtime_value = _optional_float(
                baseline_runtime_from_ckpt.get(
                    "train_wall_clock_sec",
                    baseline_runtime_from_ckpt.get("baseline_train_wall_clock_sec"),
                )
            )
        if baseline_runtime_value is not None:
            baseline_reference_runtime = {
                "train_wall_clock_sec": float(baseline_runtime_value),
                "source": "baseline_checkpoint_payload",
            }
        else:
            baseline_reference_runtime = _resolve_baseline_reference_train_wall_clock_sec(
                ckpt_path=baseline_ckpt_path,
                planned_steps=baseline_steps_for_phase,
            )
    actual_robust_steps_completed = (
        int(len(history_robust.get("outer_loss", []))) if attack_training_executed else 0
    )
    preserve_baseline_from_resume = bool(
        robust_resume_loaded
        and isinstance(robust_resume_accounting, dict)
        and (
            "baseline_images_seen_total" in robust_resume_accounting
            or "baseline_batch_equiv_denoiser_evals_total" in robust_resume_accounting
            or "baseline_train_wall_clock_sec_effective" in robust_resume_accounting
            or "baseline_weighted_counts" in robust_resume_accounting
        )
    )
    baseline_phase_images_seen_total = int(baseline_steps_for_phase * int(cfg.batch_size))
    baseline_phase_batch_equiv_total = float(baseline_steps_for_phase)
    direct_baseline_batch_equiv_counts = read_denoiser_op_count_totals(history_baseline)
    if direct_baseline_batch_equiv_counts is not None:
        baseline_phase_batch_equiv_total = float(
            direct_baseline_batch_equiv_counts["n_fwd"]
            + direct_baseline_batch_equiv_counts["n_fwd_inputgrad"]
            + direct_baseline_batch_equiv_counts["n_fwd_parambackward"]
        )
    baseline_images_seen_total = int(
        robust_resume_accounting.get("baseline_images_seen_total", 0)
        if preserve_baseline_from_resume
        else baseline_phase_images_seen_total
    )
    robust_images_seen_total = int(actual_robust_steps_completed * int(cfg.batch_size) if attack_training_executed else 0)
    effective_train_images_seen_total = int(baseline_images_seen_total + robust_images_seen_total)
    baseline_batch_equiv_total = float(
        robust_resume_accounting.get("baseline_batch_equiv_denoiser_evals_total", 0.0)
        if preserve_baseline_from_resume
        else baseline_phase_batch_equiv_total
    )
    robust_batch_equiv_total = float(robust_batch_equiv_cumulative_summary["final"] or 0.0)
    effective_train_batch_equiv_total = float(baseline_batch_equiv_total + robust_batch_equiv_total)
    baseline_train_wall_clock_sec_effective = (
        _optional_float(robust_resume_accounting.get("baseline_train_wall_clock_sec_effective"))
        if preserve_baseline_from_resume
        else None
    )
    baseline_train_wall_clock_source = "robust_resume_accounting"
    if baseline_train_wall_clock_sec_effective is None:
        if baseline_steps_for_phase <= 0:
            baseline_train_wall_clock_sec_effective = 0.0
            baseline_train_wall_clock_source = "zero_steps"
        elif runtime_sec["baseline_train"] > 0.0:
            baseline_train_wall_clock_sec_effective = float(runtime_sec["baseline_train"])
            baseline_train_wall_clock_source = "executed_in_run"
        elif baseline_reference_runtime["train_wall_clock_sec"] is not None:
            baseline_train_wall_clock_sec_effective = float(baseline_reference_runtime["train_wall_clock_sec"])
            baseline_train_wall_clock_source = str(baseline_reference_runtime["source"])
        else:
            baseline_train_wall_clock_source = "missing_reused_baseline_reference_runtime"
    preserved_baseline_weighted_counts = None
    if preserve_baseline_from_resume:
        candidate_counts = robust_resume_accounting.get("baseline_weighted_counts")
        if isinstance(candidate_counts, dict):
            preserved_baseline_weighted_counts = candidate_counts
    weighted_accounting = _compute_weighted_accounting(
        cfg=cfg,
        method_name=method_name,
        sigma_levels=sigma_levels,
        baseline_steps_total=baseline_steps_for_phase,
        history_baseline=history_baseline,
        robust_steps_total=actual_robust_steps_completed,
        history_robust=history_robust,
        attack_training_executed=attack_training_executed,
        calibration=weighted_calibration,
        preserved_baseline_counts=preserved_baseline_weighted_counts,
    )
    runtime_sec["baseline_weighted_n_fwd"] = float(weighted_accounting["baseline"]["n_fwd"])
    runtime_sec["baseline_weighted_n_fwd_inputgrad"] = float(weighted_accounting["baseline"]["n_fwd_inputgrad"])
    runtime_sec["baseline_weighted_n_fwd_parambackward"] = float(
        weighted_accounting["baseline"]["n_fwd_parambackward"]
    )
    runtime_sec["baseline_weighted_compute_units"] = weighted_accounting["baseline"]["weighted_compute_units"]
    runtime_sec["baseline_train_wall_clock_sec_effective"] = baseline_train_wall_clock_sec_effective

    cdro_attack_num_steps, cdro_attack_num_steps_source = _resolve_cdro_attack_num_steps(cfg)
    metrics = {
        "flow_debug": {
            "flow_mode": flow_mode,
            "baseline_only": bool(cfg.baseline_only),
            "baseline_gate_enabled": bool(cfg.baseline_gate_enabled),
            "baseline_gate_error_on_fail": bool(cfg.baseline_gate_error_on_fail),
            "outer_attack_weight": float(cfg.outer_attack_weight),
            "outer_clean_weight": float(cfg.outer_clean_weight),
            "inner_steps": int(cfg.inner_steps),
            "attack_num_steps": None if getattr(cfg, "attack_num_steps", None) is None else int(cfg.attack_num_steps),
            "cdro_attack_num_steps_resolved": int(cdro_attack_num_steps),
            "cdro_attack_num_steps_source": str(cdro_attack_num_steps_source),
            "v1_dual_lambda_enabled": bool(cfg.v1_dual_lambda_enabled),
            "v1_energy_budget_rho": float(cfg.v1_energy_budget_rho),
            "v1_lambda_init": float(cfg.v1_lambda_init),
            "v1_lambda_lr": float(cfg.v1_lambda_lr),
            "v1_lambda_max": float(cfg.v1_lambda_max),
            "lambda_energy_fixed": float(cfg.lambda_energy),
            "cdro_step_size": float(getattr(cfg, "cdro_step_size", 0.0)),
            "cdro_total_budget_rho": float(getattr(cfg, "cdro_total_budget_rho", 0.0)),
            "cdro_time_horizon": float(getattr(cfg, "cdro_time_horizon", 0.0)),
            "cdro_edm_ladder_mode": str(getattr(cfg, "cdro_edm_ladder_mode", "deterministic_midpoint_quantile")),
            "cdro_warmup_fraction": float(getattr(cfg, "cdro_warmup_fraction", 0.0)),
            "cdro_antithetic_rollouts": bool(getattr(cfg, "cdro_antithetic_rollouts", False)),
            "method_version": cfg.method_version,
            "method_description": getattr(method, "DESCRIPTION", ""),
            "phase_step_split_mode": str(phase_steps["split_mode"]),
            "total_steps_requested": int(phase_steps["total_steps"]),
            "baseline_phase_steps": int(baseline_steps_for_phase),
            "robust_phase_steps": int(robust_steps_for_phase),
            "robust_phase_steps_completed": int(actual_robust_steps_completed),
            "wdro_reference_warmup_compute_fraction": phase_steps.get("wdro_warmup_compute_fraction"),
            "wdro_reference_warmup_weighted_compute_fraction": phase_steps.get(
                "wdro_warmup_weighted_compute_fraction"
            ),
            "wdro_reference_robust_step_batch_equiv": phase_steps.get("wdro_robust_step_batch_equiv"),
            "wdro_reference_robust_step_weighted_compute_units": phase_steps.get(
                "wdro_robust_step_weighted_compute_units"
            ),
            "cdro_estimated_robust_step_batch_equiv": phase_steps.get("cdro_robust_step_batch_equiv"),
            "cdro_estimated_robust_step_weighted_compute_units": phase_steps.get(
                "cdro_robust_step_weighted_compute_units"
            ),
            "model_backend": model_bundle.name,
            "diagnostics_backend": diagnostics.name,
            "training_objective": str(getattr(cfg, "training_objective", "edm")),
            "score_matching_weight_power": float(getattr(cfg, "score_matching_weight_power", 2.0)),
            "rf_baseline_mode": str(getattr(cfg, "rf_baseline_mode", "strong")),
            "rf_stage1_fraction": float(getattr(cfg, "rf_stage1_fraction", 0.5)),
            "rf_reflow_t_distribution": str(getattr(cfg, "rf_reflow_t_distribution", "u_shaped")),
            "rf_cdro_quantile_rule": str(getattr(cfg, "rf_cdro_quantile_rule", "right_endpoint")),
            "rf_loss": str(getattr(cfg, "rf_loss", "pseudo_huber")),
            "rf_pseudo_huber_delta": float(getattr(cfg, "rf_pseudo_huber_delta", 0.1)),
            "rf_edm_init": rf_edm_init_report,
            "rf_cdro_pair_source": str(getattr(cfg, "rf_cdro_pair_source", "auto")),
            "rf_cdro_stage1_t_distribution": rf_cdro_stage1_t_distribution,
            "rf_cdro_reflow_t_distribution": rf_cdro_reflow_t_distribution,
            "rf_cdro_eval_stage": rf_cdro_eval_stage,
            "rf_cdro_eval_t_distribution": rf_cdro_eval_t_distribution,
            "rf_cdro_time_grid_distribution": rf_cdro_eval_t_distribution,
            "wild_update_interval": int(getattr(cfg, "wild_update_interval", 0)),
            "wild_cache_batches": int(getattr(cfg, "wild_cache_batches", 0)),
            "wild_inner_steps": int(getattr(cfg, "wild_inner_steps", 0)),
            "wild_step_size": float(getattr(cfg, "wild_step_size", 0.0)),
            "wild_gamma": float(getattr(cfg, "wild_gamma", 0.0)),
            "wild_fixed_noise_inner": bool(getattr(cfg, "wild_fixed_noise_inner", False)),
            "wild_clamp_samples": bool(getattr(cfg, "wild_clamp_samples", False)),
            "wild_sample_min": float(getattr(cfg, "wild_sample_min", 0.0)),
            "wild_sample_max": float(getattr(cfg, "wild_sample_max", 0.0)),
            "wdro_warmup_fraction": float(getattr(cfg, "wdro_warmup_fraction", 0.0)),
            "wdro_refresh_epochs": float(getattr(cfg, "wdro_refresh_epochs", 0.0)),
            "wdro_adv_prob": float(getattr(cfg, "wdro_adv_prob", 0.0)),
            "wdro_attack_steps": int(getattr(cfg, "wdro_attack_steps", 0)),
            "wdro_attack_step_size": float(getattr(cfg, "wdro_attack_step_size", 0.0)),
            "wdro_gamma": float(getattr(cfg, "wdro_gamma", 0.0)),
            "wdro_clamp_samples": bool(getattr(cfg, "wdro_clamp_samples", False)),
            "wdro_sample_min": float(getattr(cfg, "wdro_sample_min", 0.0)),
            "wdro_sample_max": float(getattr(cfg, "wdro_sample_max", 0.0)),
            "eval_seed_gate": int(gate_eval_seed),
            "eval_seed_metrics": int(metrics_eval_seed),
            "eval_shared_terminal_noise": bool(cfg.eval_use_shared_terminal_noise),
            "eval_shared_reverse_noise": bool(cfg.eval_use_shared_reverse_noise),
            "baseline_ckpt_enabled": bool(baseline_ckpt_enabled),
            "baseline_ckpt_path": str(baseline_ckpt_path) if baseline_ckpt_enabled else "",
            "baseline_ckpt_force_retrain": bool(baseline_ckpt_force_retrain),
            "baseline_ckpt_strict_meta": bool(baseline_ckpt_strict_meta),
            "baseline_ckpt_loaded": bool(baseline_ckpt_loaded),
            "baseline_ckpt_saved": bool(baseline_ckpt_saved),
            "baseline_ckpt_saved_at": baseline_ckpt_saved_at,
            "baseline_ckpt_signature_hash": baseline_signature_hash,
            "baseline_handoff_source": str(baseline_handoff_source),
            "baseline_handoff_loaded": bool(
                isinstance(baseline_handoff_state, dict)
                and isinstance(baseline_handoff_state.get("model_state_dict"), dict)
            ),
            "baseline_handoff_has_optimizer_state": bool(
                isinstance(baseline_handoff_state, dict)
                and baseline_handoff_state.get("optimizer_theta_state") is not None
            ),
            "baseline_handoff_has_ema_state": bool(
                isinstance(baseline_handoff_state, dict)
                and isinstance(baseline_handoff_state.get("ema_state_dict"), dict)
            ),
            "baseline_handoff_rng_restored": bool(baseline_handoff_rng_restored),
            "compute_accounting": {
                "primary_metric_name": "train_wall_clock_sec",
                "primary_metric_definition": (
                    "Effective training-only wall-clock for the planned training budget on fixed hardware, "
                    "including reused baseline warmup time when a reference runtime is available."
                ),
                "secondary_metric_name": "weighted_compute_units",
                "secondary_metric_definition": (
                    "Weighted-op budget defined as "
                    "1*N_fwd + alpha*N_fwd_plus_inputgrad + beta*N_fwd_plus_parambackward, "
                    "with alpha/beta supplied by a profiler calibration on the same denoiser workload."
                ),
                "train_wall_clock_sec": None,
                "train_gpu_hours": None,
                "train_wall_clock_complete": None,
                "train_wall_clock_baseline_source": baseline_train_wall_clock_source,
                "train_wall_clock_baseline_reference": baseline_reference_runtime,
                "train_accelerator_kind": accelerator_meta["train_accelerator_kind"],
                "train_accelerator_name": accelerator_meta["train_accelerator_name"],
                "train_accelerator_count": int(accelerator_meta["train_accelerator_count"]),
                "train_gpu_count": int(accelerator_meta["train_gpu_count"]),
                "weighted_compute_units": weighted_accounting["effective"]["weighted_compute_units"],
                "baseline_weighted_compute_units": weighted_accounting["baseline"]["weighted_compute_units"],
                "robust_weighted_compute_units": weighted_accounting["robust"]["weighted_compute_units"],
                "weighted_compute_calibration": weighted_calibration,
                "weighted_counts": weighted_accounting,
                "unit_name": "batch_equiv_denoiser_evals",
                "unit_definition": (
                    "One denoiser forward over one training batch counts as 1 unit; "
                    "a forward over B*T path states counts as T units."
                ),
                "legacy_unit_name": "batch_equiv_denoiser_evals",
                "legacy_unit_definition": (
                    "One denoiser forward over one training batch counts as 1 unit; "
                    "a forward over B*T path states counts as T units."
                ),
                "diagnostics_included": False,
                "baseline_batch_equiv_denoiser_evals_total": baseline_batch_equiv_total,
                "robust_batch_equiv_denoiser_evals_total": robust_batch_equiv_total,
                "effective_train_batch_equiv_denoiser_evals_total": effective_train_batch_equiv_total,
            },
            "budget_accounting": {
                "image_unit_name": "images_seen",
                "image_unit_definition": (
                    "One training sample drawn for one optimizer step counts as 1 image seen."
                ),
                "batch_size": int(cfg.batch_size),
                "baseline_images_seen_total": baseline_images_seen_total,
                "robust_images_seen_total": robust_images_seen_total,
                "effective_train_images_seen_total": effective_train_images_seen_total,
            },
        },
        "dataset_debug": {
            "dataset_backend": dataset.name,
            "data_shape": list(dataset.data_shape),
            "limited_data_enabled": bool(cfg.limited_data_enabled),
            "image_split_seed": int(getattr(cfg, "image_split_seed", 0)),
            "train_points_per_mode": (
                int(cfg.train_points_per_mode) if dataset.name == "toy_gmm" else None
            ),
            "train_pool_size": int(dataset.train_pool.shape[0]) if dataset.train_pool is not None else None,
            "val_pool_size": int(dataset.val_pool.shape[0]),
            "population_size": (
                int(dataset.metadata.get("population_size"))
                if dataset.metadata.get("population_size") is not None
                else None
            ),
            "train_selection_policy": dataset.metadata.get("train_selection_policy"),
            "train_subset_size_resolved": dataset.metadata.get("train_subset_size_resolved"),
            "val_subset_size_resolved": dataset.metadata.get("val_subset_size_resolved"),
            "train_subset_fraction_resolved": dataset.metadata.get("train_subset_fraction_resolved"),
            "train_mode_counts": (
                [
                    int(v)
                    for v in torch.bincount(
                        dataset.train_pool_labels,
                        minlength=int(dataset.metadata.get("num_classes", int(dataset.train_pool_labels.max().item()) + 1)),
                    )
                    .detach()
                    .cpu()
                    .tolist()
                ]
                if dataset.train_pool_labels is not None
                else None
            ),
        },
        "baseline_gate": {
            "enabled": bool(cfg.baseline_gate_enabled),
            "error_on_fail": bool(cfg.baseline_gate_error_on_fail),
            "attack_training_executed": bool(attack_training_executed),
            **baseline_gate,
        },
        "objective_debug": {
            "baseline_loss_kind": str(getattr(cfg, "training_objective", "edm")),
            "baseline_loss": summarize_series(history_baseline["loss"]),
            "baseline_proxy_weighted_denoise_loss": summarize_series(
                history_baseline.get("proxy_weighted_denoise_loss", [])
            ),
            "baseline_loss_curve": [float(v) for v in history_baseline.get("loss", [])],
            "baseline_proxy_weighted_denoise_curve": [
                float(v) for v in history_baseline.get("proxy_weighted_denoise_loss", [])
            ],
            "baseline_sigma_counts": history_baseline.get("sigma_counts", []),
            "baseline_rf_stage_curve": [str(v) for v in history_baseline.get("rf_stage", [])],
            "baseline_rf_t_mean_curve": [float(v) for v in history_baseline.get("rf_t_mean", [])],
            "baseline_rf_stage1_steps": int(history_baseline.get("rf_stage1_steps", 0) or 0),
            "baseline_rf_reflow_steps": int(history_baseline.get("rf_reflow_steps", 0) or 0),
            "robust_rf_stage_curve": [str(v) for v in history_robust.get("rf_stage", [])],
            "robust_rf_t_distribution_curve": [str(v) for v in history_robust.get("rf_t_distribution", [])],
            "robust_rf_stage1_t_distribution_resolved": str(
                history_robust.get("rf_stage1_t_distribution_resolved", "")
            ),
            "robust_rf_reflow_t_distribution_resolved": str(
                history_robust.get("rf_reflow_t_distribution_resolved", "")
            ),
            "robust_rf_eval_t_distribution_resolved": str(
                history_robust.get("rf_eval_t_distribution_resolved", "")
            ),
            "robust_rf_cdro_quantile_rule_resolved": str(
                history_robust.get("rf_cdro_quantile_rule_resolved", "")
            ),
            "robust_cdro_edm_ladder_mode_resolved": str(
                history_robust.get("cdro_edm_ladder_mode_resolved", "")
            ),
            "robust_rf_stage1_steps": int(history_robust.get("rf_stage1_steps", 0) or 0),
            "robust_rf_reflow_steps": int(history_robust.get("rf_reflow_steps", 0) or 0),
            "robust_outer_loss": summarize_series(history_robust["outer_loss"]),
            "robust_outer_loss_attack": summarize_series(history_robust.get("outer_loss_attack", [])),
            "robust_outer_loss_clean": summarize_series(history_robust.get("outer_loss_clean", [])),
            "robust_outer_loss_curve": [float(v) for v in history_robust.get("outer_loss", [])],
            "robust_outer_loss_attack_curve": [float(v) for v in history_robust.get("outer_loss_attack", [])],
            "robust_outer_loss_clean_curve": [float(v) for v in history_robust.get("outer_loss_clean", [])],
            "robust_inner_obj": summarize_series(history_robust["inner_obj"]),
            "robust_inner_obj_curve": [float(v) for v in history_robust.get("inner_obj", [])],
            "robust_energy": summarize_series(history_robust.get("energy", [])),
            "robust_rf_pair_source_resolved": str(history_robust.get("rf_pair_source_resolved", "")),
            "v11_path_transport_cost": summarize_series(
                history_robust.get("energy", []) if str(cfg.method_version).lower() in ("v1.1", "1.1") else []
            ),
            "cdro_control_cost": summarize_series(
                history_robust.get("energy", []) if str(cfg.method_version).lower() == "cdro" else []
            ),
            "wild_sample_transport_cost": summarize_series(history_robust.get("wild_inner_transport_cost", [])),
            "robust_lambda_value": summarize_series(history_robust.get("lambda_value", [])),
            "robust_lambda_value_next": summarize_series(history_robust.get("lambda_value_next", [])),
            "robust_lambda_subgrad": summarize_series(history_robust.get("lambda_subgrad", [])),
            "robust_dual_surrogate": summarize_series(history_robust.get("dual_surrogate", [])),
            "robust_delta_norm_mean": summarize_series(history_robust.get("delta_norm_mean", [])),
            "robust_delta_norm_max": summarize_series(history_robust.get("delta_norm_max", [])),
            "robust_delta_norm_ratio_mean": summarize_series(history_robust.get("delta_norm_ratio_mean", [])),
            "robust_delta_norm_ratio_max": summarize_series(history_robust.get("delta_norm_ratio_max", [])),
            "robust_sched_attack_weight": summarize_series(history_robust.get("sched_attack_weight", [])),
            "robust_sched_clean_weight": summarize_series(history_robust.get("sched_clean_weight", [])),
            "robust_sched_phi_lr_scale": summarize_series(history_robust.get("sched_phi_lr_scale", [])),
            "diag_steps_count": len(history_robust.get("diag_step", [])),
            "diag_inner_obj_current": summarize_series(history_robust.get("diag_inner_obj_current", [])),
            "diag_inner_obj_zero": summarize_series(history_robust.get("diag_inner_obj_zero", [])),
            "diag_inner_obj_gap": summarize_series(history_robust.get("diag_inner_obj_gap", [])),
            "diag_inner_obj_gap_ratio": diag_gap_ratio_summary,
            "diag_delta_norm_mean": summarize_series(history_robust.get("diag_delta_norm_mean", [])),
            "diag_delta_norm_max": summarize_series(history_robust.get("diag_delta_norm_max", [])),
            "diag_delta_norm_ratio_mean": diag_delta_ratio_summary,
            "diag_delta_norm_ratio_max": summarize_series(history_robust.get("diag_delta_norm_ratio_max", [])),
            "diag_path_delta_mean": summarize_series(history_robust.get("diag_path_delta_mean", [])),
            "diag_terminal_delta_mean": summarize_series(history_robust.get("diag_terminal_delta_mean", [])),
            "wild_inner_attack_loss": summarize_series(history_robust.get("wild_inner_attack_loss", [])),
            "wild_inner_transport_cost": summarize_series(history_robust.get("wild_inner_transport_cost", [])),
            "wild_inner_sigma_mean": summarize_series(history_robust.get("wild_inner_sigma_mean", [])),
            "wild_refresh_steps": [int(v) for v in history_robust.get("wild_refresh_step", [])],
            "wild_cache_sizes": [int(v) for v in history_robust.get("wild_cache_size", [])],
            "wdro_refresh_steps": [int(v) for v in history_robust.get("wdro_refresh_step", [])],
            "wdro_dataset_sizes": [int(v) for v in history_robust.get("wdro_dataset_size", [])],
            "wdro_adv_examples": [int(v) for v in history_robust.get("wdro_adv_examples", [])],
            "wdro_attack_batches": [int(v) for v in history_robust.get("wdro_attack_batches", [])],
            "wdro_attack_loss": summarize_series(history_robust.get("wdro_attack_loss", [])),
            "wdro_transport_cost": summarize_series(history_robust.get("wdro_transport_cost", [])),
            "robust_batch_equiv_denoiser_evals_step": robust_batch_equiv_step_summary,
            "robust_batch_equiv_denoiser_evals_attack_construction": summarize_series(
                history_robust.get("batch_equiv_denoiser_evals_attack_construction", [])
            ),
            "robust_batch_equiv_denoiser_evals_attack_eval": summarize_series(
                history_robust.get("batch_equiv_denoiser_evals_attack_eval", [])
            ),
            "robust_batch_equiv_denoiser_evals_clean_eval": summarize_series(
                history_robust.get("batch_equiv_denoiser_evals_clean_eval", [])
            ),
            "robust_batch_equiv_denoiser_evals_cumulative": robust_batch_equiv_cumulative_summary,
            "robust_batch_equiv_denoiser_evals_step_curve": [
                float(v) for v in history_robust.get("batch_equiv_denoiser_evals_step", [])
            ],
            "robust_batch_equiv_denoiser_evals_cumulative_curve": [
                float(v) for v in history_robust.get("batch_equiv_denoiser_evals_cumulative", [])
            ],
            "collapse_gap_ratio_tol": float(cfg.collapse_gap_ratio_tol),
            "collapse_delta_ratio_tol": float(cfg.collapse_delta_ratio_tol),
            "collapse_suspected": collapse_suspected,
        },
        "denoise_debug": denoise_error_curves,
        "denoise_debug_train_pool": denoise_error_curves_train,
        "denoise_debug_heldout_pool": denoise_error_curves_val,
        "generalization_debug": denoise_gap,
        "constraint_debug": {
            "constraint_radius_source": constraint_radius_source,
            "cdro_control_u_radius": history_robust.get("control_u_radius"),
            "cdro_transition_deltas": [float(v) for v in history_robust.get("transition_deltas", [])],
            "cdro_transition_deltas_reference": [
                float(v) for v in history_robust.get("transition_deltas_reference", [])
            ],
            "cdro_stochastic_ladder_first_sigma_levels": [
                float(v) for v in history_robust.get("stochastic_ladder_first_sigma_levels", [])
            ],
            "cdro_stochastic_ladder_first_transition_deltas": [
                float(v) for v in history_robust.get("stochastic_ladder_first_transition_deltas", [])
            ],
            "cdro_stochastic_ladder_last_sigma_levels": [
                float(v) for v in history_robust.get("stochastic_ladder_last_sigma_levels", [])
            ],
            "cdro_stochastic_ladder_last_transition_deltas": [
                float(v) for v in history_robust.get("stochastic_ladder_last_transition_deltas", [])
            ],
            "cdro_rf_stage_transition_deltas": history_robust.get("rf_stage_transition_deltas"),
            "use_time_dependent_kappa": bool(cfg.use_time_dependent_kappa),
            "kappa_base": float(cfg.control_radius_kappa),
            "kappa_low_multiplier": float(cfg.kappa_low_multiplier),
            "kappa_mid_multiplier": float(cfg.kappa_mid_multiplier),
            "kappa_high_multiplier": float(cfg.kappa_high_multiplier),
            "kappa_preserve_l2_budget": bool(cfg.kappa_preserve_l2_budget),
            "kappa_by_step": [float(v.item()) for v in kappa_by_step.detach().cpu()],
            "constraint_radius_by_step": (
                None
                if constraint_radius_by_step is None
                else [float(v.item()) for v in constraint_radius_by_step.detach().cpu()]
            ),
            "train_rollout_by_step": constraint_stats_train,
            "heldout_rollout_by_step": constraint_stats_val,
            "attack_gap_windows_train": attack_gap_windows_train,
            "attack_gap_windows_heldout": attack_gap_windows_val,
        },
        "recovery_debug": {
            "terminal_step_for_plot": int(terminal_step),
            "terminal_sigma_for_plot": float(sigma_levels[terminal_step].item()),
            "plot_stochastic_backward": bool(cfg.plot_stochastic_backward),
            "baseline_x0_mse_from_ref_terminal": float(
                (rev_baseline_from_ref[:, reverse_endpoint_idx] - x_demo)
                .reshape(x_demo.shape[0], -1)
                .pow(2)
                .sum(dim=1)
                .mean()
                .item()
            ),
            "baseline_x0_mse_from_attack_terminal": float(
                (rev_baseline_from_attack[:, reverse_endpoint_idx] - x_demo)
                .reshape(x_demo.shape[0], -1)
                .pow(2)
                .sum(dim=1)
                .mean()
                .item()
            ),
            "bayes_posterior_mean_mse_at_terminal_sigma": (
                None if bayes_terminal_mse is None else float(bayes_terminal_mse)
            ),
            "baseline_to_bayes_mse_ratio_ref_terminal": (
                None
                if bayes_terminal_mse is None
                else float(
                    (
                        (rev_baseline_from_ref[:, reverse_endpoint_idx] - x_demo)
                        .reshape(x_demo.shape[0], -1)
                        .pow(2)
                        .sum(dim=1)
                        .mean()
                        .item()
                    )
                    / max(bayes_terminal_mse, 1e-12)
                )
            ),
            "baseline_plot_endpoint_mode_metrics_ref_terminal": dataset.evaluate_sample_metrics(
                tensor_to_numpy(rev_baseline_from_ref_plot[:, reverse_endpoint_idx])
            ),
            "baseline_plot_endpoint_mode_metrics_attack_terminal": dataset.evaluate_sample_metrics(
                tensor_to_numpy(rev_baseline_from_attack_plot[:, reverse_endpoint_idx])
            ),
            "baseline_path_mse_by_step_from_ref_terminal": compute_path_mse_by_step(rev_baseline_from_ref, ref_paths_plot),
            "baseline_path_mse_by_step_from_attack_terminal": compute_path_mse_by_step(
                rev_baseline_from_attack, ctrl_paths_plot
            ),
            "baseline_x0_mse_vs_terminal_step_forward_baseline": recovery_ref_curve,
            "baseline_x0_mse_vs_terminal_step_forward_attack": recovery_attack_curve,
            "paired_reverse_delta_under_plot_setting": paired_reverse_delta_plot,
            "paired_reverse_delta_deterministic": paired_reverse_delta_det,
            "reverse_terminal_consistency": terminal_consistency,
        },
        "sample_quality_debug": {
            "evaluation_protocol": {
                "metrics_eval_seed": int(metrics_eval_seed),
                "shared_terminal_noise": bool(cfg.eval_use_shared_terminal_noise),
                "shared_reverse_noise": bool(cfg.eval_use_shared_reverse_noise),
            },
            "fid_reference": {
                "policy_name": str(getattr(cfg, "fid_ref_policy", "auto")),
                "requested_path": (
                    os.path.abspath(str(getattr(cfg, "fid_ref_path", "")).strip())
                    if str(getattr(cfg, "fid_ref_path", "")).strip()
                    else None
                ),
                "resolved_path": None,
                "resolved_via": None,
                "checked_candidates": [],
                "metadata_path": None,
                "metadata": None,
            },
            "baseline_generated_metrics": dataset.evaluate_sample_metrics(baseline_gen_np),
            "robust_generated_metrics": dataset.evaluate_sample_metrics(robust_gen_np),
            "baseline_generated_to_train_min_dist": (
                evaluate_nearest_reference_distance(baseline_gen_np, train_pool_np)
                if train_pool_np is not None and enable_ref_dist
                else None
            ),
            "robust_generated_to_train_min_dist": (
                evaluate_nearest_reference_distance(robust_gen_np, train_pool_np)
                if train_pool_np is not None and enable_ref_dist
                else None
            ),
            "heldout_to_train_min_dist": (
                evaluate_nearest_reference_distance(val_pool_np, train_pool_np)
                if train_pool_np is not None and enable_ref_dist
                else None
            ),
        },
    }
    if check_report:
        metrics["checks"] = check_report
    metrics["sample_quality_debug"]["baseline_generated_mode_metrics"] = metrics["sample_quality_debug"][
        "baseline_generated_metrics"
    ]
    metrics["sample_quality_debug"]["robust_generated_mode_metrics"] = metrics["sample_quality_debug"][
        "robust_generated_metrics"
    ]

    if getattr(cfg, "compute_fid", False) and dataset.name in ("mnist", "image_basic", "image_folder"):
        print("  [info] Computing FID via subprocess...", flush=True)
        import re
        import shutil
        import subprocess
        import sys
        from PIL import Image

        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        fid_script = os.path.join(repo_root, "fid.py")
        fid_ref_info = metrics["sample_quality_debug"]["fid_reference"]
        explicit_ref_path = str(getattr(cfg, "fid_ref_path", "")).strip()
        if explicit_ref_path:
            ref_candidates = [os.path.abspath(explicit_ref_path)]
        else:
            ref_candidates = [
                os.path.join(repo_root, cfg.outdir, "mnist_ref.npz"),
                os.path.join(repo_root, "toy_outputs", "mnist_ref.npz"),
            ]
        fid_ref_info["checked_candidates"] = [str(p) for p in ref_candidates]

        # Training/eval models are no longer used after this point; release VRAM
        # before spawning FID subprocess (helps avoid CUDA OOM in child process).
        if torch.cuda.is_available():
            baseline_eval = baseline_eval.to("cpu")
            robust = robust.to("cpu")
            control = control.to("cpu")
            torch.cuda.empty_cache()

        def _extract_float(text: str):
            match = re.search(
                r"^\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*$",
                text,
                re.MULTILINE,
            )
            if match is None:
                return None
            return float(match.group(1))

        def calc_fid_for_batch(batch_np, prefix):
            t_fid = time.perf_counter()

            fid_dir = os.path.join(exp_dir, f"fid_{prefix}")
            if os.path.isdir(fid_dir):
                shutil.rmtree(fid_dir)
            os.makedirs(fid_dir, exist_ok=True)

            n_available = int(batch_np.shape[0])
            n_requested = int(getattr(cfg, "fid_samples", n_available))
            if n_available < 2:
                print(
                    f"[WARN] Need at least 2 generated images for FID, got {n_available} ({prefix}).",
                    flush=True,
                )
                return None, float(time.perf_counter() - t_fid)
            n_images = min(n_available, max(2, n_requested))
            batch_np = batch_np[:n_images]

            for i in range(n_images):
                # Model output is float in [-1, 1]. Remap to uint8 [0, 255].
                img_np = ((batch_np[i] + 1.0) * 0.5).clip(0.0, 1.0)
                img_np = (img_np * 255).astype("uint8")
                if img_np.shape[0] == 1:
                    img = Image.fromarray(img_np[0], mode="L")
                else:
                    img = Image.fromarray(img_np.transpose(1, 2, 0), mode="RGB")
                img.save(os.path.join(fid_dir, f"{i:05d}.png"))

            ref_path = next((p for p in ref_candidates if os.path.exists(p)), "")
            if not ref_path:
                print(
                    f"[WARN] FID reference not found. Checked: {ref_candidates}",
                    flush=True,
                )
                return None, float(time.perf_counter() - t_fid)
            fid_ref_info["resolved_path"] = str(ref_path)
            fid_ref_info["resolved_via"] = "explicit" if explicit_ref_path else "fallback_search"
            ref_meta_path = f"{ref_path}.json"
            ref_meta = _load_json_if_exists(ref_meta_path)
            fid_ref_info["metadata_path"] = ref_meta_path if ref_meta is not None else None
            fid_ref_info["metadata"] = ref_meta

            fid_env = os.environ.copy()
            detector_path = fid_env.get("FID_DETECTOR_PATH", "").strip()
            if detector_path:
                print(f"  [info] FID_DETECTOR_PATH={detector_path}", flush=True)
                if not os.path.isfile(detector_path):
                    print(
                        f"[WARN] FID_DETECTOR_PATH does not exist: {detector_path}. "
                        "fid.py will fail unless detector can be downloaded.",
                        flush=True,
                    )
            else:
                print(
                    "[WARN] FID_DETECTOR_PATH is empty; fid.py will attempt online download.",
                    flush=True,
                )
            fid_env.setdefault("MASTER_ADDR", "127.0.0.1")
            fid_env.setdefault("MASTER_PORT", str(29600 + (os.getpid() % 1000)))
            fid_env.setdefault("RANK", "0")
            fid_env.setdefault("LOCAL_RANK", "0")
            fid_env.setdefault("WORLD_SIZE", "1")

            cmd = [
                sys.executable,
                fid_script,
                "calc",
                "--images",
                fid_dir,
                "--ref",
                ref_path,
                "--num",
                str(n_images),
            ]
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                env=fid_env,
                cwd=repo_root,
            )
            log_path = os.path.join(exp_dir, f"fid_{prefix}_subprocess.log")
            with open(log_path, "w", encoding="utf-8") as fid_log:
                fid_log.write(f"command: {' '.join(cmd)}\n")
                fid_log.write(f"returncode: {res.returncode}\n")
                fid_log.write("----- stdout -----\n")
                fid_log.write(res.stdout or "")
                fid_log.write("\n----- stderr -----\n")
                fid_log.write(res.stderr or "")

            if res.returncode != 0:
                print(
                    f"[WARN] FID subprocess failed for {prefix} (rc={res.returncode}). "
                    f"See {log_path}",
                    flush=True,
                )
                if res.stderr:
                    print(f"[WARN] FID stderr (tail):\n{res.stderr[-2000:]}", flush=True)
                if res.stdout:
                    print(f"[WARN] FID stdout (tail):\n{res.stdout[-1000:]}", flush=True)
                return None, float(time.perf_counter() - t_fid)

            fid_value = _extract_float(res.stdout)
            if fid_value is None:
                fid_value = _extract_float(res.stderr)
            if fid_value is not None:
                return fid_value, float(time.perf_counter() - t_fid)

            print(
                f"[WARN] FID output did not contain a parseable float for {prefix}. "
                f"See {log_path}",
                flush=True,
            )
            return None, float(time.perf_counter() - t_fid)

        baseline_fid, runtime_fid_baseline = calc_fid_for_batch(baseline_gen_np, "baseline")
        robust_fid, runtime_fid_robust = calc_fid_for_batch(robust_gen_np, "robust")
        metrics["sample_quality_debug"]["baseline_fid"] = baseline_fid
        metrics["sample_quality_debug"]["robust_fid"] = robust_fid
        runtime_sec["fid_baseline"] = float(runtime_fid_baseline)
        runtime_sec["fid_robust"] = float(runtime_fid_robust)

    t_plot = time.perf_counter()
    diagnostics.plot_forward_backward_debug(
        fwd_baseline_paths=tensor_to_numpy(ref_paths_plot),
        bwd_baseline_paths=rev_baseline_from_ref_np,
        fwd_attack_paths=tensor_to_numpy(ctrl_paths_plot),
        bwd_attack_paths=rev_baseline_from_attack_np,
        centers=dataset.metadata.get("centers_np"),
        out_path=os.path.join(exp_dir, "forward_backward_baseline_attack.png"),
    )
    diagnostics.plot_loss_debug(
        history_baseline=history_baseline,
        history_robust=history_robust,
        denoise_curves=denoise_error_curves,
        recovery_ref_curve=recovery_ref_curve,
        recovery_attack_curve=recovery_attack_curve,
        out_path=os.path.join(exp_dir, "debug_losses_and_recovery.png"),
    )
    runtime_sec["plotting_and_persist"] = float(time.perf_counter() - t_plot)

    run_total_sec = float(time.perf_counter() - run_t0)
    run_wall_end = datetime.now(timezone.utc).isoformat()
    runtime_sec["total"] = run_total_sec
    runtime_sec["robust_train_only"] = float(runtime_sec["robust_phase"])
    runtime_sec["baseline_train_only"] = float(runtime_sec["baseline_train"])
    runtime_sec["total_without_fid"] = float(
        run_total_sec - runtime_sec["fid_baseline"] - runtime_sec["fid_robust"]
    )
    runtime_sec["total_without_fid"] = max(runtime_sec["total_without_fid"], 0.0)
    runtime_sec["effective_train_total"] = float(runtime_sec["baseline_train"] + runtime_sec["robust_phase"])
    runtime_sec["baseline_steps_per_sec"] = (
        float(baseline_steps_for_phase) / float(runtime_sec["baseline_train"])
        if runtime_sec["baseline_train"] > 0 and baseline_steps_for_phase > 0
        else None
    )
    runtime_sec["robust_steps_per_sec"] = (
        float(actual_robust_steps_completed) / float(runtime_sec["robust_phase"])
        if attack_training_executed and runtime_sec["robust_phase"] > 0 and actual_robust_steps_completed > 0
        else None
    )
    runtime_sec["baseline_images_seen_total"] = baseline_images_seen_total
    runtime_sec["robust_images_seen_total"] = robust_images_seen_total
    runtime_sec["effective_train_images_seen_total"] = effective_train_images_seen_total
    runtime_sec["baseline_batch_equiv_denoiser_evals_total"] = baseline_batch_equiv_total
    runtime_sec["robust_batch_equiv_denoiser_evals_total"] = robust_batch_equiv_total
    runtime_sec["effective_train_batch_equiv_denoiser_evals_total"] = effective_train_batch_equiv_total
    runtime_sec["robust_batch_equiv_denoiser_evals_per_sec"] = (
        robust_batch_equiv_total / float(runtime_sec["robust_phase"])
        if attack_training_executed and runtime_sec["robust_phase"] > 0 and robust_batch_equiv_total > 0
        else None
    )
    runtime_sec["run_started_utc"] = str(robust_resume_runtime.get("run_started_utc", run_wall_start))
    runtime_sec["run_finished_utc"] = run_wall_end
    runtime_sec["attack_training_executed"] = bool(attack_training_executed)
    runtime_sec["weighted_compute_units"] = weighted_accounting["effective"]["weighted_compute_units"]
    runtime_sec["baseline_weighted_compute_units"] = weighted_accounting["baseline"]["weighted_compute_units"]
    runtime_sec["robust_weighted_compute_units"] = weighted_accounting["robust"]["weighted_compute_units"]
    if robust_resume_loaded:
        runtime_sec["baseline_phase"] = float(runtime_sec["baseline_phase"]) + float(
            robust_resume_runtime.get("baseline_phase", 0.0)
        )
        runtime_sec["baseline_train"] = float(runtime_sec["baseline_train"]) + float(
            robust_resume_runtime.get("baseline_train", 0.0)
        )
        runtime_sec["total"] = float(runtime_sec["total"]) + float(robust_resume_runtime.get("total", 0.0))
        runtime_sec["total_without_fid"] = float(runtime_sec["total_without_fid"]) + float(
            robust_resume_runtime.get("total_without_fid", 0.0)
        )
        runtime_sec["robust_phase"] = float(runtime_sec["robust_phase"]) + float(
            robust_resume_runtime.get("robust_phase", 0.0)
        )
        runtime_sec["fid_baseline"] = float(runtime_sec["fid_baseline"]) + float(
            robust_resume_runtime.get("fid_baseline", 0.0)
        )
        runtime_sec["fid_robust"] = float(runtime_sec["fid_robust"]) + float(
            robust_resume_runtime.get("fid_robust", 0.0)
        )
        runtime_sec["robust_train_only"] = float(runtime_sec["robust_phase"])
        runtime_sec["effective_train_total"] = float(runtime_sec["baseline_train"] + runtime_sec["robust_phase"])
        runtime_sec["baseline_steps_per_sec"] = (
            float(baseline_steps_for_phase) / float(runtime_sec["baseline_train"])
            if runtime_sec["baseline_train"] > 0 and baseline_steps_for_phase > 0
            else None
        )
        runtime_sec["robust_steps_per_sec"] = (
            float(actual_robust_steps_completed) / float(runtime_sec["robust_phase"])
            if attack_training_executed and runtime_sec["robust_phase"] > 0 and actual_robust_steps_completed > 0
            else None
        )
        runtime_sec["robust_batch_equiv_denoiser_evals_per_sec"] = (
            robust_batch_equiv_total / float(runtime_sec["robust_phase"])
            if attack_training_executed and runtime_sec["robust_phase"] > 0 and robust_batch_equiv_total > 0
            else None
        )
    train_wall_clock_complete = baseline_train_wall_clock_sec_effective is not None
    train_wall_clock_sec = (
        None
        if not train_wall_clock_complete
        else float(baseline_train_wall_clock_sec_effective + float(runtime_sec["robust_phase"]))
    )
    train_gpu_hours = (
        None
        if train_wall_clock_sec is None
        else float(train_wall_clock_sec) * float(accelerator_meta["train_gpu_count"]) / 3600.0
    )
    runtime_sec["train_wall_clock_sec"] = train_wall_clock_sec
    runtime_sec["train_gpu_hours"] = train_gpu_hours
    runtime_sec["train_wall_clock_complete"] = bool(train_wall_clock_complete)
    runtime_sec["baseline_train_wall_clock_sec_effective_source"] = baseline_train_wall_clock_source
    runtime_sec["baseline_reference_train_wall_clock_sec"] = baseline_reference_runtime["train_wall_clock_sec"]
    runtime_sec["baseline_reference_train_wall_clock_source"] = baseline_reference_runtime["source"]

    metrics["flow_debug"]["compute_accounting"]["train_wall_clock_sec"] = train_wall_clock_sec
    metrics["flow_debug"]["compute_accounting"]["train_gpu_hours"] = train_gpu_hours
    metrics["flow_debug"]["compute_accounting"]["train_wall_clock_complete"] = bool(train_wall_clock_complete)
    metrics["flow_debug"]["compute_accounting"]["baseline_weighted_compute_units"] = (
        weighted_accounting["baseline"]["weighted_compute_units"]
    )
    metrics["flow_debug"]["compute_accounting"]["robust_weighted_compute_units"] = (
        weighted_accounting["robust"]["weighted_compute_units"]
    )
    metrics["flow_debug"]["compute_accounting"]["weighted_compute_units"] = (
        weighted_accounting["effective"]["weighted_compute_units"]
    )

    metrics["flow_debug"]["runtime"] = runtime_sec
    metrics["flow_debug"]["runtime_total_sec"] = float(runtime_sec["total"])
    metrics["flow_debug"]["train_wall_clock_sec"] = train_wall_clock_sec
    metrics["flow_debug"]["train_gpu_hours"] = train_gpu_hours
    metrics["flow_debug"]["weighted_compute_units"] = weighted_accounting["effective"]["weighted_compute_units"]
    metrics["flow_debug"]["robust_resume_loaded"] = bool(robust_resume_loaded)
    metrics["flow_debug"]["robust_resume_ckpt_path"] = robust_resume_path or None
    metrics["flow_debug"]["robust_resume_completed_steps"] = int(robust_resume_completed_steps)

    payload = {"config": vars(cfg), "metrics": as_jsonable_metrics(metrics)}
    with open(os.path.join(exp_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    if robust_save_path and robust_trainer_state is not None:
        _save_robust_resume_checkpoint(
            ckpt_path=robust_save_path,
            cfg=cfg_robust,
            method=method,
            baseline_eval=baseline_eval,
            robust=robust,
            control=control,
            history_robust=history_robust,
            trainer_state=robust_trainer_state,
            rng_state=resume_rng_state,
            runtime_sec=runtime_sec,
            run_wall_start=run_wall_start,
        )

    print("[result] metrics summary", flush=True)
    print(f"  dataset_debug: {metrics['dataset_debug']}", flush=True)
    print(f"  objective_debug: {metrics['objective_debug']}", flush=True)
    print(f"  generalization_debug: {metrics['generalization_debug']}", flush=True)
    heldout_constraint = metrics["constraint_debug"]["heldout_rollout_by_step"]
    heldout_windows = metrics["constraint_debug"]["attack_gap_windows_heldout"]
    frac_sat = heldout_constraint["frac_near_boundary"]
    tail_start = max(len(frac_sat) * 2 // 3, 0)
    frac_sat_tail = float(sum(frac_sat[tail_start:]) / max(len(frac_sat[tail_start:]), 1))
    print(
        "  constraint_debug:"
        f" heldout_overall_win_ratio={heldout_windows['overall']['win_ratio']:.3f},"
        f" high_noise_win_ratio={heldout_windows['high_noise']['win_ratio']:.3f},"
        f" high_noise_mean_gap={heldout_windows['high_noise']['mean_attack_gap']:.6f},"
        f" frac_near_boundary_tail={frac_sat_tail:.3f}",
        flush=True,
    )
    print(
        "  recovery_debug:"
        f" baseline_x0_mse_from_ref_terminal={metrics['recovery_debug']['baseline_x0_mse_from_ref_terminal']:.6f},"
        f" baseline_x0_mse_from_attack_terminal={metrics['recovery_debug']['baseline_x0_mse_from_attack_terminal']:.6f}",
        flush=True,
    )
    paired_plot = metrics["recovery_debug"]["paired_reverse_delta_under_plot_setting"]
    print(
        "  paired_reverse_delta(plot_setting):"
        f" x0_mean_l2={paired_plot['x0_step_mean_l2']:.6f},"
        f" terminal_mean_l2={paired_plot['terminal_step_mean_l2']:.6f}",
        flush=True,
    )
    term_chk = metrics["recovery_debug"]["reverse_terminal_consistency"]
    print(
        "  reverse_terminal_consistency:"
        f" passed={term_chk['passed']}"
        f" max_l2(plot_ref)={term_chk['plot_ref']['max_l2']:.3e}"
        f" max_l2(plot_attack)={term_chk['plot_attack']['max_l2']:.3e}",
        flush=True,
    )
    print(
        f"  baseline_gate: passed={metrics['baseline_gate']['passed']} "
        f"attack_training_executed={metrics['baseline_gate']['attack_training_executed']}",
        flush=True,
    )
    if not metrics["baseline_gate"]["passed"]:
        for failed in metrics["baseline_gate"]["failed_checks"]:
            print(
                "    - "
                f"{failed['name']}: value={failed['value']:.4f} {failed['op']} {failed['threshold']:.4f} (FAIL)",
                flush=True,
            )
    for warning in diagnostics.summarize_warnings(cfg, metrics):
        print(f"  [warn] {warning}", flush=True)
    if check_report:
        print(f"  checks: {check_report}", flush=True)
    print(
        "  runtime_debug:"
        f" total={runtime_sec['total']:.2f}s,"
        f" baseline_train={runtime_sec['baseline_train']:.2f}s,"
        f" robust_phase={runtime_sec['robust_phase']:.2f}s,"
        f" post_eval={runtime_sec['post_train_eval']:.2f}s,"
        f" fid_baseline={runtime_sec['fid_baseline']:.2f}s,"
        f" fid_robust={runtime_sec['fid_robust']:.2f}s,"
        f" train_wall_clock_sec={runtime_sec['train_wall_clock_sec']}"
        f" weighted_compute_units={runtime_sec['weighted_compute_units']}",
        flush=True,
    )
    print(f"[result] artifacts saved to: {exp_dir}", flush=True)
    return payload
