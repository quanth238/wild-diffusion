import json
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
    return weighted_compute_units(
        n_fwd=float(path_steps) if attack_enabled else 0.0,
        n_fwd_inputgrad=float(path_steps * max(int(inner_steps), 0)) if attack_enabled else 0.0,
        n_fwd_parambackward=float(path_steps * active_outer_branches),
        calibration=calibration,
    )
