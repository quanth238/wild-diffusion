import json
import math
import os
from typing import Any, Dict, List, Optional


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


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return None


def _device_type_from_label(value: Any) -> Optional[str]:
    label = str(value or "").strip().lower()
    if not label:
        return None
    if label == "auto":
        return None
    if label.startswith("cuda"):
        return "cuda"
    if label.startswith("mps"):
        return "mps"
    if label.startswith("cpu"):
        return "cpu"
    return label


def _amp_dtype_label(value: Any) -> Optional[str]:
    label = str(value or "").strip().lower()
    if not label or label == "auto":
        return None
    if label == "bf16":
        return "bfloat16"
    if label in {"fp16", "half"}:
        return "float16"
    return label


def _compare_match_field(
    *,
    name: str,
    expected: Any,
    actual: Any,
    mismatches: List[Dict[str, Any]],
) -> None:
    if expected is None or actual is None:
        return
    if expected == actual:
        return
    mismatches.append(
        {
            "field": str(name),
            "expected": expected,
            "actual": actual,
        }
    )


def calibration_compatibility_report(
    *,
    payload: Optional[Dict[str, Any]],
    training_objective: Optional[str] = None,
    image_backbone: Optional[str] = None,
    batch_size: Optional[int] = None,
    hidden_dim: Optional[int] = None,
    image_size: Optional[int] = None,
    image_channels: Optional[int] = None,
    device: Optional[str] = None,
    device_name: Optional[str] = None,
    amp_dtype: Optional[str] = None,
    allow_tf32: Optional[bool] = None,
    cudnn_benchmark: Optional[bool] = None,
    score_matching_weight_power: Optional[float] = None,
) -> Dict[str, Any]:
    """Compare a calibration payload against the active workload/hardware."""

    report = {
        "available": False,
        "required_match": True,
        "exact_hardware_match": True,
        "required_mismatches": [],
        "advisory_mismatches": [],
    }
    if not isinstance(payload, dict):
        return report

    workload = payload.get("workload")
    hardware = payload.get("hardware")
    if not isinstance(workload, dict):
        workload = {}
    if not isinstance(hardware, dict):
        hardware = {}

    required_mismatches: List[Dict[str, Any]] = []
    advisory_mismatches: List[Dict[str, Any]] = []
    _compare_match_field(
        name="workload.training_objective",
        expected=str(training_objective).strip().lower() if training_objective is not None else None,
        actual=(
            str(workload.get("training_objective")).strip().lower()
            if workload.get("training_objective") is not None
            else None
        ),
        mismatches=required_mismatches,
    )
    _compare_match_field(
        name="workload.image_backbone",
        expected=str(image_backbone).strip().lower() if image_backbone is not None else None,
        actual=str(workload.get("image_backbone")).strip().lower() if workload.get("image_backbone") is not None else None,
        mismatches=required_mismatches,
    )
    _compare_match_field(
        name="workload.batch_size",
        expected=_safe_int(batch_size),
        actual=_safe_int(workload.get("batch_size")),
        mismatches=required_mismatches,
    )
    _compare_match_field(
        name="workload.hidden_dim",
        expected=_safe_int(hidden_dim),
        actual=_safe_int(workload.get("hidden_dim")),
        mismatches=required_mismatches,
    )
    _compare_match_field(
        name="workload.image_size",
        expected=_safe_int(image_size),
        actual=_safe_int(workload.get("image_size")),
        mismatches=required_mismatches,
    )
    _compare_match_field(
        name="workload.image_channels",
        expected=_safe_int(image_channels),
        actual=_safe_int(workload.get("image_channels")),
        mismatches=required_mismatches,
    )
    if str(training_objective or "").strip().lower() == "score":
        _compare_match_field(
            name="workload.score_matching_weight_power",
            expected=_safe_float(score_matching_weight_power),
            actual=_safe_float(workload.get("score_matching_weight_power")),
            mismatches=required_mismatches,
        )

    _compare_match_field(
        name="hardware.device_type",
        expected=_device_type_from_label(device),
        actual=_device_type_from_label(hardware.get("device_type", hardware.get("device"))),
        mismatches=required_mismatches,
    )
    _compare_match_field(
        name="hardware.amp_dtype",
        expected=_amp_dtype_label(amp_dtype),
        actual=_amp_dtype_label(hardware.get("amp_dtype")),
        mismatches=required_mismatches,
    )
    _compare_match_field(
        name="hardware.allow_tf32",
        expected=_safe_bool(allow_tf32),
        actual=_safe_bool(hardware.get("allow_tf32")),
        mismatches=required_mismatches,
    )
    _compare_match_field(
        name="hardware.cudnn_benchmark",
        expected=_safe_bool(cudnn_benchmark),
        actual=_safe_bool(hardware.get("cudnn_benchmark")),
        mismatches=required_mismatches,
    )
    _compare_match_field(
        name="hardware.device_name",
        expected=str(device_name).strip() if device_name is not None else None,
        actual=str(hardware.get("device_name")).strip() if hardware.get("device_name") is not None else None,
        mismatches=advisory_mismatches,
    )

    report.update(
        {
            "available": True,
            "required_match": bool(not required_mismatches),
            "exact_hardware_match": bool(not advisory_mismatches),
            "required_mismatches": required_mismatches,
            "advisory_mismatches": advisory_mismatches,
        }
    )
    return report


def _timing_stat_sec(payload: Optional[Dict[str, Any]], op_name: str, stat_key: str = "median_sec") -> Optional[float]:
    if not isinstance(payload, dict):
        return None
    timings = payload.get("timings_sec")
    if not isinstance(timings, dict):
        return None
    op_stats = timings.get(op_name)
    if not isinstance(op_stats, dict):
        return None
    return _safe_float(op_stats.get(stat_key))


def _flop_stat_value(payload: Optional[Dict[str, Any]], op_name: str) -> Optional[float]:
    if not isinstance(payload, dict):
        return None
    candidate_groups = []
    flops = payload.get("flops")
    if isinstance(flops, dict):
        candidate_groups.append(flops)
    candidate_groups.append(payload)
    for group in candidate_groups:
        if not isinstance(group, dict):
            continue
        op_stats = group.get(op_name)
        if isinstance(op_stats, dict):
            for key in ("per_batch_flops", "flops", "total_flops"):
                value = _safe_float(op_stats.get(key))
                if value is not None and value > 0.0:
                    return float(value)
        elif op_stats is not None:
            value = _safe_float(op_stats)
            if value is not None and value > 0.0:
                return float(value)
    return None


DEFAULT_INPUTGRAD_FORWARD_FLOP_MULTIPLIER = 2.0
DEFAULT_PARAMBACKWARD_FORWARD_FLOP_MULTIPLIER = 3.0


def _flop_values_from_group(payload: Dict[str, Any], group_name: str) -> Optional[Dict[str, Any]]:
    group = payload.get(group_name)
    if not isinstance(group, dict):
        return None
    forward_value = _flop_stat_value(group, "forward_only")
    inputgrad_value = _flop_stat_value(group, "forward_plus_inputgrad")
    parambackward_value = _flop_stat_value(group, "forward_plus_parambackward")
    if (
        forward_value is None
        or forward_value <= 0.0
        or inputgrad_value is None
        or inputgrad_value <= 0.0
        or parambackward_value is None
        or parambackward_value <= 0.0
    ):
        return None
    return {
        "forward_flops": float(forward_value),
        "inputgrad_flops": float(inputgrad_value),
        "parambackward_flops": float(parambackward_value),
        "flop_cost_group": str(group_name),
        "flop_cost_source": str(group.get("source", group_name)),
        "flop_definition": str(group.get("definition", "")),
    }


def _flop_values_from_payload(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for group_name in ("training_flops", "analytical_training_flops", "flop_costs"):
        values = _flop_values_from_group(payload, group_name)
        if values is not None:
            return values

    forward_value = _flop_stat_value(payload, "forward_only")
    if forward_value is None or forward_value <= 0.0:
        return None

    # torch.profiler's CUDA FLOP attribution is useful for forward convolutions,
    # but it commonly undercounts autograd backward kernels. For image
    # calibration payloads generated before `training_flops` existed, synthesize
    # the conventional training costs from the measured forward pass: input-grad
    # is forward+backward-to-input (~2x), and parameter-backward training is
    # forward+activation/weight backward (~3x).
    if str(payload.get("format", "")).strip() == "image_flop_calibration_v1":
        return {
            "forward_flops": float(forward_value),
            "inputgrad_flops": float(forward_value) * DEFAULT_INPUTGRAD_FORWARD_FLOP_MULTIPLIER,
            "parambackward_flops": float(forward_value) * DEFAULT_PARAMBACKWARD_FORWARD_FLOP_MULTIPLIER,
            "flop_cost_group": "training_flops_synthesized",
            "flop_cost_source": "analytical_training_from_profiler_forward_legacy_payload",
            "flop_definition": (
                "Measured forward-only torch.profiler FLOPs; input-gradient "
                "primitive = 2x forward; parameter-backward training primitive = 3x forward."
            ),
        }

    inputgrad_value = _flop_stat_value(payload, "forward_plus_inputgrad")
    parambackward_value = _flop_stat_value(payload, "forward_plus_parambackward")
    if inputgrad_value is None or inputgrad_value <= 0.0 or parambackward_value is None or parambackward_value <= 0.0:
        return None
    return {
        "forward_flops": float(forward_value),
        "inputgrad_flops": float(inputgrad_value),
        "parambackward_flops": float(parambackward_value),
        "flop_cost_group": "flops",
        "flop_cost_source": "profiler_supported_operator_flops",
        "flop_definition": "Raw profiler-supported operator FLOPs from the calibration payload.",
    }


def resolve_default_weighted_compute_calibration_path(
    *,
    calibration_path: str = "",
    training_objective: str = "edm",
    image_backbone: str = "conv",
    batch_size: int = 256,
    hidden_dim: int = 64,
    device: str = "cuda",
) -> str:
    """Resolve the default Simpsons weighted-compute calibration path.

    This keeps the historical EDM calibration as the default for EDM-family
    workloads, while allowing RF-family runs on the locked conv/b256/h64/cuda
    profile to prefer the RF-specific calibration when it exists.
    """

    explicit_path = str(calibration_path).strip()
    if explicit_path:
        return explicit_path

    backbone = str(image_backbone).strip().lower()
    objective = str(training_objective).strip().lower()
    device_label = str(device).strip().lower()
    if backbone != "conv" or int(batch_size) != 256 or int(hidden_dim) != 64:
        return ""
    if device_label != "auto" and not device_label.startswith("cuda"):
        return ""

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    calibration_dir = os.path.join(root_dir, "toy_outputs", "compute_calibration")
    candidate_filenames = []
    if objective == "rf":
        candidate_filenames.append("simpsons_mnist_rgb_image_conv_rf_b256_h64_cuda.json")
    candidate_filenames.append("simpsons_mnist_rgb_image_conv_edm_b256_h64_cuda.json")
    for filename in candidate_filenames:
        candidate_path = os.path.join(calibration_dir, filename)
        if os.path.isfile(candidate_path):
            return candidate_path
    return ""


def load_weighted_compute_calibration(
    *,
    calibration_path: str = "",
    inputgrad_alpha: float = 0.0,
    parambackward_beta: float = 0.0,
    training_objective: Optional[str] = None,
    image_backbone: Optional[str] = None,
    batch_size: Optional[int] = None,
    hidden_dim: Optional[int] = None,
    image_size: Optional[int] = None,
    image_channels: Optional[int] = None,
    device: Optional[str] = None,
    device_name: Optional[str] = None,
    amp_dtype: Optional[str] = None,
    allow_tf32: Optional[bool] = None,
    cudnn_benchmark: Optional[bool] = None,
    score_matching_weight_power: Optional[float] = None,
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
            "compatibility": {
                "available": False,
                "required_match": True,
                "exact_hardware_match": True,
                "required_mismatches": [],
                "advisory_mismatches": [],
            },
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
            "compatibility": {
                "available": False,
                "required_match": True,
                "exact_hardware_match": True,
                "required_mismatches": [],
                "advisory_mismatches": [],
            },
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
    compatibility = calibration_compatibility_report(
        payload=payload,
        training_objective=training_objective,
        image_backbone=image_backbone,
        batch_size=batch_size,
        hidden_dim=hidden_dim,
        image_size=image_size,
        image_channels=image_channels,
        device=device,
        device_name=device_name,
        amp_dtype=amp_dtype,
        allow_tf32=allow_tf32,
        cudnn_benchmark=cudnn_benchmark,
        score_matching_weight_power=score_matching_weight_power,
    )
    if compatibility["available"] and not compatibility["required_match"]:
        mismatch_parts = [
            f"{item['field']}: expected={item['expected']} actual={item['actual']}"
            for item in compatibility["required_mismatches"]
        ]
        mismatch_text = "; ".join(mismatch_parts) if mismatch_parts else "unknown mismatch"
        raise RuntimeError(
            "Weighted-compute calibration is incompatible with the active workload/runtime: "
            f"{mismatch_text}. Calibration path: {os.path.abspath(path)}"
        )
    return {
        "available": True,
        "forward_weight": 1.0,
        "inputgrad_alpha": float(alpha),
        "parambackward_beta": float(beta),
        "source": "calibration_json",
        "calibration_path": os.path.abspath(path),
        "payload": payload,
        "compatibility": compatibility,
    }


def load_flop_calibration(
    *,
    calibration_path: str = "",
    forward_flops: float = 0.0,
    inputgrad_flops: float = 0.0,
    parambackward_flops: float = 0.0,
    training_objective: Optional[str] = None,
    image_backbone: Optional[str] = None,
    batch_size: Optional[int] = None,
    hidden_dim: Optional[int] = None,
    image_size: Optional[int] = None,
    image_channels: Optional[int] = None,
    device: Optional[str] = None,
    device_name: Optional[str] = None,
    amp_dtype: Optional[str] = None,
    allow_tf32: Optional[bool] = None,
    cudnn_benchmark: Optional[bool] = None,
    score_matching_weight_power: Optional[float] = None,
) -> Dict[str, Any]:
    """Resolve FLOP calibration from explicit values or a JSON payload."""

    explicit_forward = _safe_float(forward_flops)
    explicit_inputgrad = _safe_float(inputgrad_flops)
    explicit_parambackward = _safe_float(parambackward_flops)
    if (
        explicit_forward is not None
        and explicit_forward > 0.0
        and explicit_inputgrad is not None
        and explicit_inputgrad > 0.0
        and explicit_parambackward is not None
        and explicit_parambackward > 0.0
    ):
        return {
            "available": True,
            "forward_flops": float(explicit_forward),
            "inputgrad_flops": float(explicit_inputgrad),
            "parambackward_flops": float(explicit_parambackward),
            "inputgrad_forward_multiplier": float(explicit_inputgrad) / float(explicit_forward),
            "parambackward_forward_multiplier": float(explicit_parambackward) / float(explicit_forward),
            "source": "explicit_cli",
            "flop_cost_group": "explicit_cli",
            "flop_cost_source": "explicit_cli",
            "flop_definition": "Explicit per-batch denoiser FLOP costs supplied by the caller.",
            "calibration_path": None,
            "payload": None,
            "compatibility": {
                "available": False,
                "required_match": True,
                "exact_hardware_match": True,
                "required_mismatches": [],
                "advisory_mismatches": [],
            },
        }

    path = str(calibration_path).strip()
    if not path:
        return {
            "available": False,
            "forward_flops": None,
            "inputgrad_flops": None,
            "parambackward_flops": None,
            "source": "unavailable",
            "calibration_path": None,
            "payload": None,
            "compatibility": {
                "available": False,
                "required_match": True,
                "exact_hardware_match": True,
                "required_mismatches": [],
                "advisory_mismatches": [],
            },
        }
    if not os.path.isfile(path):
        raise FileNotFoundError(f"FLOP calibration file not found: {path}")

    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    flop_values = _flop_values_from_payload(payload)
    if flop_values is None:
        raise RuntimeError(
            f"Invalid FLOP calibration payload at {path}: "
            "expected positive per-batch FLOP values for training_flops "
            "or profiler forward_only/forward_plus_* entries."
        )
    compatibility = calibration_compatibility_report(
        payload=payload,
        training_objective=training_objective,
        image_backbone=image_backbone,
        batch_size=batch_size,
        hidden_dim=hidden_dim,
        image_size=image_size,
        image_channels=image_channels,
        device=device,
        device_name=device_name,
        amp_dtype=amp_dtype,
        allow_tf32=allow_tf32,
        cudnn_benchmark=cudnn_benchmark,
        score_matching_weight_power=score_matching_weight_power,
    )
    if compatibility["available"] and not compatibility["required_match"]:
        mismatch_parts = [
            f"{item['field']}: expected={item['expected']} actual={item['actual']}"
            for item in compatibility["required_mismatches"]
        ]
        mismatch_text = "; ".join(mismatch_parts) if mismatch_parts else "unknown mismatch"
        raise RuntimeError(
            "FLOP calibration is incompatible with the active workload/runtime: "
            f"{mismatch_text}. Calibration path: {os.path.abspath(path)}"
        )
    return {
        "available": True,
        "forward_flops": float(flop_values["forward_flops"]),
        "inputgrad_flops": float(flop_values["inputgrad_flops"]),
        "parambackward_flops": float(flop_values["parambackward_flops"]),
        "inputgrad_forward_multiplier": float(flop_values["inputgrad_flops"]) / float(flop_values["forward_flops"]),
        "parambackward_forward_multiplier": float(flop_values["parambackward_flops"]) / float(flop_values["forward_flops"]),
        "source": "calibration_json",
        "flop_cost_group": str(flop_values["flop_cost_group"]),
        "flop_cost_source": str(flop_values["flop_cost_source"]),
        "flop_definition": str(flop_values["flop_definition"]),
        "calibration_path": os.path.abspath(path),
        "payload": payload,
        "compatibility": compatibility,
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


def denoiser_flops(
    *,
    n_fwd: float,
    n_fwd_inputgrad: float,
    n_fwd_parambackward: float,
    calibration: Dict[str, Any],
) -> Optional[float]:
    """Compute total denoiser FLOPs from primitive op counts."""

    if not calibration.get("available", False):
        return None
    return (
        float(n_fwd) * float(calibration["forward_flops"])
        + float(n_fwd_inputgrad) * float(calibration["inputgrad_flops"])
        + float(n_fwd_parambackward) * float(calibration["parambackward_flops"])
    )


def denoiser_flops_from_count_record(
    *,
    count_record: Dict[str, Any],
    calibration: Dict[str, Any],
    override_n_fwd: Optional[float] = None,
) -> Optional[float]:
    """Compute total denoiser FLOPs from a recorded count dict."""

    if not isinstance(count_record, dict):
        return None
    n_fwd = _safe_float(count_record.get("n_fwd"))
    n_fwd_inputgrad = _safe_float(count_record.get("n_fwd_inputgrad"))
    n_fwd_parambackward = _safe_float(count_record.get("n_fwd_parambackward"))
    if n_fwd is None or n_fwd_inputgrad is None or n_fwd_parambackward is None:
        return None
    return denoiser_flops(
        n_fwd=float(override_n_fwd) if override_n_fwd is not None else float(n_fwd),
        n_fwd_inputgrad=float(n_fwd_inputgrad),
        n_fwd_parambackward=float(n_fwd_parambackward),
        calibration=calibration,
    )


def flops_to_gflops(flops: Optional[float]) -> Optional[float]:
    if flops is None:
        return None
    return float(flops) / 1e9


def flops_to_tflops(flops: Optional[float]) -> Optional[float]:
    if flops is None:
        return None
    return float(flops) / 1e12


def flops_to_pflops(flops: Optional[float]) -> Optional[float]:
    if flops is None:
        return None
    return float(flops) / 1e15


def weighted_compute_units_from_count_record(
    *,
    count_record: Dict[str, Any],
    calibration: Dict[str, Any],
    override_n_fwd: Optional[float] = None,
) -> Optional[float]:
    """Compute weighted units from a recorded count dict.

    The record is expected to carry `n_fwd`, `n_fwd_inputgrad`, and
    `n_fwd_parambackward` totals like the payloads under
    `metrics.flow_debug.compute_accounting.weighted_counts.*`.
    """

    if not isinstance(count_record, dict):
        return None
    n_fwd = _safe_float(count_record.get("n_fwd"))
    n_fwd_inputgrad = _safe_float(count_record.get("n_fwd_inputgrad"))
    n_fwd_parambackward = _safe_float(count_record.get("n_fwd_parambackward"))
    if n_fwd is None or n_fwd_inputgrad is None or n_fwd_parambackward is None:
        return None
    return weighted_compute_units(
        n_fwd=float(override_n_fwd) if override_n_fwd is not None else float(n_fwd),
        n_fwd_inputgrad=float(n_fwd_inputgrad),
        n_fwd_parambackward=float(n_fwd_parambackward),
        calibration=calibration,
    )


def predicted_denoiser_wall_clock_sec(
    *,
    n_fwd: float,
    n_fwd_inputgrad: float,
    n_fwd_parambackward: float,
    calibration: Dict[str, Any],
) -> Optional[float]:
    """Predict denoiser-only wall-clock from calibration timing medians."""

    payload = calibration.get("payload") if isinstance(calibration, dict) else None
    forward_only_sec = _timing_stat_sec(payload, "forward_only", "median_sec")
    inputgrad_sec = _timing_stat_sec(payload, "forward_plus_inputgrad", "median_sec")
    parambackward_sec = _timing_stat_sec(payload, "forward_plus_parambackward", "median_sec")
    if forward_only_sec is None or inputgrad_sec is None or parambackward_sec is None:
        return None
    return (
        float(n_fwd) * float(forward_only_sec)
        + float(n_fwd_inputgrad) * float(inputgrad_sec)
        + float(n_fwd_parambackward) * float(parambackward_sec)
    )


def predicted_denoiser_wall_clock_sec_from_count_record(
    *,
    count_record: Dict[str, Any],
    calibration: Dict[str, Any],
    override_n_fwd: Optional[float] = None,
) -> Optional[float]:
    """Predict denoiser-only wall-clock from a recorded denoiser-op count dict."""

    if not isinstance(count_record, dict):
        return None
    n_fwd = _safe_float(count_record.get("n_fwd"))
    n_fwd_inputgrad = _safe_float(count_record.get("n_fwd_inputgrad"))
    n_fwd_parambackward = _safe_float(count_record.get("n_fwd_parambackward"))
    if n_fwd is None or n_fwd_inputgrad is None or n_fwd_parambackward is None:
        return None
    return predicted_denoiser_wall_clock_sec(
        n_fwd=float(override_n_fwd) if override_n_fwd is not None else float(n_fwd),
        n_fwd_inputgrad=float(n_fwd_inputgrad),
        n_fwd_parambackward=float(n_fwd_parambackward),
        calibration=calibration,
    )


def cdro_adjusted_weighted_compute_from_compute_accounting(
    *,
    compute_accounting: Dict[str, Any],
    calibration: Dict[str, Any],
) -> Optional[Dict[str, float]]:
    """Recompute legacy CDRO weighted units excluding the logging-only forward reevaluation.

    Historical CDRO runs recorded one extra attacked-path forward sweep used for
    `inner_obj` logging / NaN guarding. For comparison plots and budget matching,
    this helper removes that monitoring-only pass while keeping the
    attack-construction input-grad and outer param-backward counts intact.
    """

    if not isinstance(compute_accounting, dict):
        return None
    effective_calibration = compute_accounting.get("weighted_compute_calibration")
    if not isinstance(effective_calibration, dict) or not effective_calibration.get("available", False):
        effective_calibration = calibration
    if not isinstance(effective_calibration, dict) or not effective_calibration.get("available", False):
        return None
    weighted_counts = compute_accounting.get("weighted_counts")
    if not isinstance(weighted_counts, dict):
        return None
    baseline_counts = weighted_counts.get("baseline")
    robust_counts = weighted_counts.get("robust")
    if not isinstance(baseline_counts, dict) or not isinstance(robust_counts, dict):
        return None
    baseline_weighted = weighted_compute_units_from_count_record(
        count_record=baseline_counts,
        calibration=effective_calibration,
    )
    robust_count_source = str(robust_counts.get("count_source", "")).strip().lower()
    # Preserve forward counts for modern CDRO records, where `n_fwd` already
    # reflects only legitimate training work (for example RF teacher-pair
    # forwards) and excludes removed logging-only sweeps. Legacy unadjusted
    # records still fall back to zeroing `n_fwd`.
    override_n_fwd = 0.0
    if robust_count_source in {
        "history_direct_cdro_adjusted",
        "history_inferred_cdro_adjusted",
        "history_direct_cdro",
        "history_inferred_cdro",
    }:
        override_n_fwd = _safe_float(robust_counts.get("n_fwd"))
    robust_weighted = weighted_compute_units_from_count_record(
        count_record=robust_counts,
        calibration=effective_calibration,
        override_n_fwd=override_n_fwd,
    )
    robust_logging_only_forward_count = _safe_float(robust_counts.get("n_fwd"))
    if baseline_weighted is None or robust_weighted is None:
        return None
    return {
        "baseline_weighted_compute_units": float(baseline_weighted),
        "robust_weighted_compute_units": float(robust_weighted),
        "weighted_compute_units": float(baseline_weighted + robust_weighted),
        "robust_logging_only_forward_count_excluded": (
            (
                0.0
                if robust_logging_only_forward_count is None
                else float(max(float(robust_logging_only_forward_count) - float(override_n_fwd or 0.0), 0.0))
            )
        ),
    }


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
    total_budget_rho: float,
    outer_attack_weight: float,
    outer_clean_weight: float,
    calibration: Dict[str, Any],
) -> Optional[float]:
    """Weighted compute for one CDRO robust optimizer step."""

    path_steps = max(int(n_steps_path), 0)
    if path_steps <= 0:
        return weighted_compute_units(
            n_fwd=0.0,
            n_fwd_inputgrad=0.0,
            n_fwd_parambackward=0.0,
            calibration=calibration,
        )
    attack_enabled = bool(float(total_budget_rho) > 0.0 and float(outer_attack_weight) > 0.0 and int(inner_steps) > 0)
    reference_path_enabled = bool(float(outer_clean_weight) > 0.0 or (float(outer_attack_weight) > 0.0 and not attack_enabled))
    active_outer_branches = (
        int(float(outer_attack_weight) > 0.0) + int(float(outer_clean_weight) > 0.0)
        if attack_enabled
        else int(reference_path_enabled)
    )
    return weighted_compute_units(
        n_fwd=0.0,
        n_fwd_inputgrad=float(path_steps * max(int(inner_steps), 0)) if attack_enabled else 0.0,
        n_fwd_parambackward=float(path_steps * active_outer_branches),
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
