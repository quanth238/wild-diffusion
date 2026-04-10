import copy
from typing import Any, MutableSequence, Optional

import torch

from ..models import set_requires_grad


def ema_mode(cfg_like: Any) -> str:
    mode = str(getattr(cfg_like, "ema_mode", "official")).strip().lower()
    if mode not in {"official", "fixed"}:
        raise ValueError(f"Unsupported EMA mode: {mode!r}")
    return mode


def ema_rampup_ratio(cfg_like: Any) -> Optional[float]:
    if bool(getattr(cfg_like, "disable_ema_rampup", False)):
        return None
    value = getattr(cfg_like, "ema_rampup_ratio", 0.05)
    if value is None:
        return None
    return float(value)


def ema_config_dict(cfg_like: Any) -> dict:
    return {
        "use_ema_eval": bool(getattr(cfg_like, "use_ema_eval", False)),
        "ema_mode": ema_mode(cfg_like),
        "ema_decay": float(getattr(cfg_like, "ema_decay", 0.999)),
        "ema_halflife_kimg": float(getattr(cfg_like, "ema_halflife_kimg", 500.0)),
        "ema_rampup_ratio": ema_rampup_ratio(cfg_like),
    }


def append_ema_cli_args(cmd: MutableSequence[str], cfg_like: Any) -> None:
    if not bool(getattr(cfg_like, "use_ema_eval", False)):
        return
    mode = ema_mode(cfg_like)
    cmd.extend(["--use-ema-eval", "--ema-mode", mode])
    if mode == "fixed":
        cmd.extend(["--ema-decay", str(float(getattr(cfg_like, "ema_decay", 0.999)))])
        return
    cmd.extend(["--ema-halflife-kimg", str(float(getattr(cfg_like, "ema_halflife_kimg", 500.0)))])
    rampup = ema_rampup_ratio(cfg_like)
    if rampup is None:
        cmd.append("--disable-ema-rampup")
    else:
        cmd.extend(["--ema-rampup-ratio", str(float(rampup))])


def init_ema_model(denoiser, cfg_like: Any, *, ema_state_dict: Optional[dict] = None):
    if not bool(getattr(cfg_like, "use_ema_eval", False)):
        return None
    ema_model = copy.deepcopy(denoiser).eval()
    set_requires_grad(ema_model, False)
    if ema_state_dict is not None:
        ema_model.load_state_dict(ema_state_dict, strict=True)
    return ema_model


def compute_ema_beta(
    cfg_like: Any,
    *,
    cur_nimg: float,
    batch_size: Optional[int] = None,
) -> float:
    mode = ema_mode(cfg_like)
    if mode == "fixed":
        return float(getattr(cfg_like, "ema_decay", 0.999))

    ema_halflife_nimg = float(getattr(cfg_like, "ema_halflife_kimg", 500.0)) * 1000.0
    rampup = ema_rampup_ratio(cfg_like)
    if rampup is not None:
        ema_halflife_nimg = min(ema_halflife_nimg, max(float(cur_nimg), 0.0) * float(rampup))
    step_batch = max(int(batch_size if batch_size is not None else getattr(cfg_like, "batch_size", 1)), 1)
    return float(0.5 ** (float(step_batch) / max(ema_halflife_nimg, 1e-8)))


def update_ema_model(
    ema_model,
    denoiser,
    cfg_like: Any,
    *,
    cur_nimg: float,
    batch_size: Optional[int] = None,
) -> float:
    if ema_model is None:
        return 0.0
    ema_beta = compute_ema_beta(cfg_like, cur_nimg=cur_nimg, batch_size=batch_size)
    with torch.no_grad():
        for p_ema, p_net in zip(ema_model.parameters(), denoiser.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))
    return float(ema_beta)
