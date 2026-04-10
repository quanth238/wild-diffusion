import json
import math
import os
from typing import Any, Dict, Optional


DENOISER_OP_COUNTS_RECORDED_KEY = "denoiser_op_counts_recorded"
DENOISER_OP_COUNT_STEP_KEYS = (
    "denoiser_op_n_fwd_step",
    "denoiser_op_n_fwd_inputgrad_step",
    "denoiser_op_n_fwd_parambackward_step",
)
DENOISER_OP_COUNT_CUMULATIVE_KEYS = (
    "denoiser_op_n_fwd_cumulative",
    "denoiser_op_n_fwd_inputgrad_cumulative",
    "denoiser_op_n_fwd_parambackward_cumulative",
)


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_weighted_compute_calibration(
    *,
    calibration_path: str = "",
    inputgrad_alpha: float = 0.0,
    parambackward_beta: float = 0.0,
) -> Dict[str, Any]:
    """Resolve weighted-compute calibration from explicit ratios or a JSON payload."""

    explicit_alpha = _safe_float(inputgrad_alpha)
    explicit_beta = _safe_float(parambackward_beta)
    if explicit_alpha is not None and explicit_alpha > 0.0 and explicit_beta is not None and explicit_beta > 0.0:
        return {
            "available": True,
            "forward_weight": 1.0,
            "inputgrad_alpha": float(explicit_alpha),
            "parambackward_beta": float(explicit_beta),
            "source": "explicit_cli",
            "calibration_path": None,
            "payload": None,
        }

    path = str(calibration_path).strip()
    if not path:
        return {
            "available": False,
            "forward_weight": 1.0,
            "inputgrad_alpha": None,
            "parambackward_beta": None,
            "source": "unavailable",
            "calibration_path": None,
            "payload": None,
        }
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Weighted-compute calibration file not found: {path}")

    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    ratios = payload.get("ratios", payload)
    alpha = _safe_float(ratios.get("inputgrad_alpha"))
    beta = _safe_float(ratios.get("parambackward_beta"))
    if alpha is None or alpha <= 0.0 or beta is None or beta <= 0.0:
        raise RuntimeError(
            f"Invalid weighted-compute calibration payload at {path}: "
            "expected positive 'inputgrad_alpha' and 'parambackward_beta'."
        )
    return {
        "available": True,
        "forward_weight": 1.0,
        "inputgrad_alpha": float(alpha),
        "parambackward_beta": float(beta),
        "source": "calibration_json",
        "calibration_path": os.path.abspath(path),
        "payload": payload,
    }


def weighted_compute_units(
    *,
    n_fwd: float,
    n_fwd_inputgrad: float,
    n_fwd_parambackward: float,
    calibration: Dict[str, Any],
) -> Optional[float]:
    """Compute weighted units when a calibration is available."""

    if not calibration.get("available", False):
        return None
    return (
        float(n_fwd)
        + float(calibration["inputgrad_alpha"]) * float(n_fwd_inputgrad)
        + float(calibration["parambackward_beta"]) * float(n_fwd_parambackward)
    )


def estimate_wall_clock_sec_from_batch_equiv(
    *,
    batch_equiv_denoiser_evals: float,
    batch_size: int,
    sec_per_kimg: float,
) -> float:
    """Convert batch-equivalent denoiser evals into wall-clock using a fixed sec/kimg calibration."""

    return float(batch_equiv_denoiser_evals) * float(batch_size) * float(sec_per_kimg) / 1000.0


def ensure_denoiser_op_count_history(history: Dict[str, Any]) -> None:
    """Ensure a history dict has the direct denoiser-op accounting schema."""

    history[DENOISER_OP_COUNTS_RECORDED_KEY] = bool(history.get(DENOISER_OP_COUNTS_RECORDED_KEY, False))
    for key in DENOISER_OP_COUNT_STEP_KEYS + DENOISER_OP_COUNT_CUMULATIVE_KEYS:
        history.setdefault(key, [])


def append_denoiser_op_count_step(
    history: Dict[str, Any],
    *,
    n_fwd: float,
    n_fwd_inputgrad: float,
    n_fwd_parambackward: float,
) -> None:
    """Append one step of direct denoiser-op counts and maintain cumulatives."""

    ensure_denoiser_op_count_history(history)
    last_n_fwd = float(history["denoiser_op_n_fwd_cumulative"][-1]) if history["denoiser_op_n_fwd_cumulative"] else 0.0
    last_n_fwd_inputgrad = (
        float(history["denoiser_op_n_fwd_inputgrad_cumulative"][-1])
        if history["denoiser_op_n_fwd_inputgrad_cumulative"]
        else 0.0
    )
    last_n_fwd_parambackward = (
        float(history["denoiser_op_n_fwd_parambackward_cumulative"][-1])
        if history["denoiser_op_n_fwd_parambackward_cumulative"]
        else 0.0
    )
    step_n_fwd = float(n_fwd)
    step_n_fwd_inputgrad = float(n_fwd_inputgrad)
    step_n_fwd_parambackward = float(n_fwd_parambackward)
    history["denoiser_op_n_fwd_step"].append(step_n_fwd)
    history["denoiser_op_n_fwd_inputgrad_step"].append(step_n_fwd_inputgrad)
    history["denoiser_op_n_fwd_parambackward_step"].append(step_n_fwd_parambackward)
    history["denoiser_op_n_fwd_cumulative"].append(last_n_fwd + step_n_fwd)
    history["denoiser_op_n_fwd_inputgrad_cumulative"].append(last_n_fwd_inputgrad + step_n_fwd_inputgrad)
    history["denoiser_op_n_fwd_parambackward_cumulative"].append(
        last_n_fwd_parambackward + step_n_fwd_parambackward
    )
    history[DENOISER_OP_COUNTS_RECORDED_KEY] = True


def read_denoiser_op_count_totals(history: Optional[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    """Read cumulative direct denoiser-op counts from history when they were recorded."""

    if not isinstance(history, dict):
        return None
    if not bool(history.get(DENOISER_OP_COUNTS_RECORDED_KEY, False)):
        return None
    ensure_denoiser_op_count_history(history)

    def _total(step_key: str, cumulative_key: str) -> float:
        cumulative = history.get(cumulative_key, [])
        if cumulative:
            return float(cumulative[-1])
        steps = history.get(step_key, [])
        return float(sum(float(v) for v in steps))

    return {
        "n_fwd": _total("denoiser_op_n_fwd_step", "denoiser_op_n_fwd_cumulative"),
        "n_fwd_inputgrad": _total("denoiser_op_n_fwd_inputgrad_step", "denoiser_op_n_fwd_inputgrad_cumulative"),
        "n_fwd_parambackward": _total(
            "denoiser_op_n_fwd_parambackward_step",
            "denoiser_op_n_fwd_parambackward_cumulative",
        ),
    }


def baseline_weighted_compute_units_for_steps(*, steps: int, calibration: Dict[str, Any]) -> Optional[float]:
    """Weighted compute for plain EDM training over a number of optimizer steps."""

    return weighted_compute_units(
        n_fwd=0.0,
        n_fwd_inputgrad=0.0,
        n_fwd_parambackward=float(max(int(steps), 0)),
        calibration=calibration,
    )


def cdro_robust_step_weighted_compute_units(
    *,
    n_steps_path: int,
    inner_steps: int,
    outer_attack_weight: float,
    outer_clean_weight: float,
    antithetic_rollouts: bool,
    calibration: Dict[str, Any],
) -> Optional[float]:
    """Weighted compute for one CDRO robust optimizer step under the current objective."""

    path_steps = max(int(n_steps_path), 0)
    if path_steps <= 0:
        return weighted_compute_units(
            n_fwd=0.0,
            n_fwd_inputgrad=0.0,
            n_fwd_parambackward=0.0,
            calibration=calibration,
        )
    attack_enabled = bool(float(outer_attack_weight) > 0.0 and int(inner_steps) > 0)
    clean_enabled = bool(float(outer_clean_weight) > 0.0)
    active_outer_branches = int(float(outer_attack_weight) > 0.0) + int(clean_enabled)
    rollout_multiplier = 2.0 if bool(antithetic_rollouts) else 1.0
    return weighted_compute_units(
        n_fwd=float(path_steps) * rollout_multiplier if attack_enabled else 0.0,
        n_fwd_inputgrad=float(path_steps * max(int(inner_steps), 0)) * rollout_multiplier if attack_enabled else 0.0,
        n_fwd_parambackward=float(path_steps * active_outer_branches) * rollout_multiplier,
        calibration=calibration,
    )


def wdro_expected_attack_construction_units_per_step(
    *,
    batch_size: int,
    train_pool_size: Optional[int],
    refresh_epochs: float,
    adv_prob: float,
    attack_steps: int,
) -> float:
    """Expected WDRO input-gradient attack construction cost per optimizer step."""

    attack_steps_value = max(int(attack_steps), 0)
    adv_prob_value = max(0.0, min(float(adv_prob), 1.0))
    refresh_epochs_value = max(float(refresh_epochs), 1e-8)
    if attack_steps_value <= 0 or adv_prob_value <= 0.0:
        return 0.0
    if train_pool_size is None or int(train_pool_size) <= 0:
        return adv_prob_value * float(attack_steps_value) / refresh_epochs_value

    batch_size_value = max(int(batch_size), 1)
    train_pool_size_value = max(int(train_pool_size), 1)
    num_batches = max(int(math.ceil(float(train_pool_size_value) / float(batch_size_value))), 1)
    refresh_interval_steps = max(
        int(math.ceil(refresh_epochs_value * float(train_pool_size_value) / float(batch_size_value))),
        1,
    )
    expected_attack_batches = adv_prob_value * float(num_batches)
    expected_attack_units_per_refresh = expected_attack_batches * float(attack_steps_value)
    return float(expected_attack_units_per_refresh / float(refresh_interval_steps))


def wdro_robust_step_weighted_compute_units(
    *,
    batch_size: int,
    train_pool_size: Optional[int],
    refresh_epochs: float,
    adv_prob: float,
    attack_steps: int,
    calibration: Dict[str, Any],
) -> Optional[float]:
    """Weighted compute for one WDRO robust optimizer step under expected refresh load."""

    return weighted_compute_units(
        n_fwd=0.0,
        n_fwd_inputgrad=wdro_expected_attack_construction_units_per_step(
            batch_size=int(batch_size),
            train_pool_size=train_pool_size,
            refresh_epochs=float(refresh_epochs),
            adv_prob=float(adv_prob),
            attack_steps=int(attack_steps),
        ),
        n_fwd_parambackward=1.0,
        calibration=calibration,
    )


def solve_warmup_steps_for_target_compute_fraction(
    *,
    total_steps: int,
    target_warmup_compute_fraction: float,
    baseline_step_compute_units: float,
    robust_step_compute_units: float,
) -> int:
    """Solve for warmup steps so baseline compute share matches a target fraction."""

    total_steps_value = max(int(total_steps), 0)
    if total_steps_value <= 0:
        return 0

    target_fraction = max(0.0, min(float(target_warmup_compute_fraction), 1.0))
    if target_fraction <= 0.0:
        return 0
    if target_fraction >= 1.0:
        return total_steps_value

    baseline_units = max(float(baseline_step_compute_units), 0.0)
    robust_units = max(float(robust_step_compute_units), 0.0)
    if baseline_units <= 0.0:
        return 0
    if robust_units <= 0.0:
        return total_steps_value

    denom = baseline_units * (1.0 - target_fraction) + target_fraction * robust_units
    if denom <= 0.0:
        return 0
    warmup_steps_float = target_fraction * float(total_steps_value) * robust_units / denom

    def _fraction_for_steps(warmup_steps: int) -> float:
        warmup_steps_value = max(0, min(int(warmup_steps), total_steps_value))
        warmup_total = float(warmup_steps_value) * baseline_units
        robust_total = float(total_steps_value - warmup_steps_value) * robust_units
        total = warmup_total + robust_total
        if total <= 0.0:
            return 0.0
        return float(warmup_total / total)

    candidate_steps = {
        0,
        total_steps_value,
        max(0, min(total_steps_value, int(math.floor(warmup_steps_float)))),
        max(0, min(total_steps_value, int(math.ceil(warmup_steps_float)))),
        max(0, min(total_steps_value, int(round(warmup_steps_float)))),
    }
    return min(
        candidate_steps,
        key=lambda warmup_steps: (
            abs(_fraction_for_steps(warmup_steps) - target_fraction),
            abs(float(warmup_steps) - warmup_steps_float),
            int(warmup_steps),
        ),
    )
