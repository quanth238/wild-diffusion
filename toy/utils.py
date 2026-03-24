import os
import random
from typing import Dict

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set Python/NumPy/PyTorch random seeds for reproducible experiments."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(requested: str) -> torch.device:
    """Resolve `auto|cpu|cuda` into a concrete torch device."""

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor to NumPy without depending on direct shared-memory bridge."""

    # Avoid direct torch->numpy bridge in environments with NumPy ABI mismatch.
    return np.asarray(tensor.detach().cpu().tolist(), dtype=np.float32)


def ensure_dir(path: str) -> None:
    """Create directory if missing (mkdir -p semantics)."""

    os.makedirs(path, exist_ok=True)


def has_nan_or_inf(tensor: torch.Tensor) -> bool:
    """Return True if tensor contains NaN or +/-Inf."""

    return bool(torch.isnan(tensor).any() or torch.isinf(tensor).any())


def scalarize(value: torch.Tensor) -> float:
    """Detach and convert scalar tensor to Python float."""

    return float(value.detach().item())


def as_jsonable_metrics(metrics: Dict) -> Dict:
    """Recursively coerce tensors/NumPy scalars into JSON-serializable values."""

    out = {}
    for k, v in metrics.items():
        if isinstance(v, dict):
            out[k] = as_jsonable_metrics(v)
        elif isinstance(v, list):
            out[k] = [float(x) if isinstance(x, (np.floating, torch.Tensor)) else x for x in v]
        elif isinstance(v, (float, int, str, bool)) or v is None:
            out[k] = v
        elif isinstance(v, np.generic):
            out[k] = float(v.item())
        elif isinstance(v, torch.Tensor):
            out[k] = float(v.detach().item())
        else:
            out[k] = float(v)
    return out
