import contextlib
from typing import Optional

import torch


def configure_runtime(*, device: torch.device, allow_tf32: bool, cudnn_benchmark: bool) -> None:
    """Apply CUDA runtime knobs that improve throughput on Ampere+/Ada cards."""

    if device.type != "cuda":
        return
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
        torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def resolve_amp_dtype(device: torch.device, amp_dtype: str) -> Optional[torch.dtype]:
    """Resolve configured autocast mode to a concrete torch dtype or None."""

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


def autocast_context(device: torch.device, amp_dtype: Optional[torch.dtype]):
    """Return CUDA autocast context when enabled, else a no-op context."""

    if amp_dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=amp_dtype)


def format_amp_dtype(amp_dtype: Optional[torch.dtype]) -> str:
    """Human-readable runtime summary string for resolved autocast mode."""

    if amp_dtype is None:
        return "off"
    if amp_dtype == torch.bfloat16:
        return "bfloat16"
    if amp_dtype == torch.float16:
        return "float16"
    return str(amp_dtype)
