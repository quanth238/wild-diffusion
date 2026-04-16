import json
import os
import pickle
from typing import Any, Optional

import torch

from .data_backends.provider import _split_indices, _subsample_indices


SUPPORTED_BASELINE_TRAIN_BACKENDS = ("toy", "mainline")
_MAINLINE_DDPMPP_DROPOUT = 0.10


def resolve_baseline_train_backend(cfg_like: Any) -> str:
    return str(getattr(cfg_like, "baseline_train_backend", "toy")).strip().lower()


def resolve_baseline_train_batch_gpu(cfg_like: Any) -> Optional[int]:
    value = int(getattr(cfg_like, "baseline_train_batch_gpu", 0))
    return None if value <= 0 else int(value)


def validate_mainline_baseline_request(cfg_like: Any) -> None:
    """Fail fast on unsupported uses of the mainline image baseline bridge."""

    if resolve_baseline_train_backend(cfg_like) != "mainline":
        return

    dataset_kind = str(getattr(cfg_like, "dataset_kind", "")).strip().lower()
    objective = str(getattr(cfg_like, "training_objective", "edm")).strip().lower()
    image_backbone = str(getattr(cfg_like, "image_backbone", "conv")).strip().lower()
    device = str(getattr(cfg_like, "device", "auto")).strip().lower()
    batch_size = int(getattr(cfg_like, "batch_size", 0))
    batch_gpu = resolve_baseline_train_batch_gpu(cfg_like)

    if dataset_kind != "image_folder":
        raise ValueError(
            "baseline_train_backend='mainline' currently supports only "
            "--dataset-kind=image_folder."
        )
    if objective != "edm":
        raise ValueError(
            "baseline_train_backend='mainline' currently supports only "
            "--training-objective=edm."
        )
    if image_backbone != "ddpmpp":
        raise ValueError(
            "baseline_train_backend='mainline' currently supports only "
            "--image-backbone=ddpmpp."
        )
    if int(getattr(cfg_like, "hidden_dim", 0)) != 128:
        raise ValueError(
            "baseline_train_backend='mainline' requires hidden_dim=128 for "
            "--image-backbone=ddpmpp."
        )
    if batch_size <= 0:
        raise ValueError(f"baseline_train_backend='mainline' requires batch_size > 0, got {batch_size}.")
    if batch_gpu is not None:
        if batch_gpu > batch_size:
            raise ValueError(
                "baseline_train_backend='mainline' requires --baseline-train-batch-gpu "
                f"<= --batch-size, got batch_gpu={batch_gpu}, batch_size={batch_size}."
            )
        if batch_size % batch_gpu != 0:
            raise ValueError(
                "baseline_train_backend='mainline' requires --baseline-train-batch-gpu "
                f"to divide --batch-size exactly, got batch_gpu={batch_gpu}, batch_size={batch_size}."
            )
    if device == "cpu":
        raise ValueError(
            "baseline_train_backend='mainline' requires CUDA because the repo's "
            "training loop initializes distributed CUDA training."
        )


def mainline_ddpmpp_dropout() -> float:
    return float(_MAINLINE_DDPMPP_DROPOUT)


def steps_to_kimg_exact(steps: int, batch_size: int, *, label: str = "steps") -> int:
    """Map toy step counts to the main trainer's integer-kimg schedule exactly."""

    steps_value = int(steps)
    batch_size_value = int(batch_size)
    if steps_value < 0:
        raise ValueError(f"{label} must be >= 0, got {steps}")
    if batch_size_value <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")

    nimg = steps_value * batch_size_value
    if nimg % 1000 != 0:
        raise ValueError(
            "baseline_train_backend='mainline' requires exact integer-kimg alignment "
            "because training.training_loop measures progress in whole kimg. "
            f"Got {label}={steps_value}, batch_size={batch_size_value}, "
            f"which implies {nimg} images = {nimg / 1000.0:.3f} kimg."
        )
    return int(nimg // 1000)


def resolve_mainline_image_subset_raw_indices(
    cfg_like: Any,
    *,
    total_images: int,
) -> Optional[list[int]]:
    """Recreate the toy image train subset as raw indices for ImageFolderDataset."""

    total = int(total_images)
    if total <= 0:
        raise ValueError(f"total_images must be > 0, got {total_images}")

    dataset_val_path = str(getattr(cfg_like, "dataset_val_path", "")).strip()
    limited_data_enabled = bool(getattr(cfg_like, "limited_data_enabled", True))
    image_train_size = int(getattr(cfg_like, "image_train_size", 0))
    image_val_size = int(getattr(cfg_like, "image_val_size", 0))
    image_split_seed = int(getattr(cfg_like, "image_split_seed", 0))

    if dataset_val_path:
        if not limited_data_enabled:
            return None
        subset = _subsample_indices(total, image_train_size, image_split_seed)
        return [int(v) for v in subset.tolist()]

    train_idx, full_train_idx, _ = _split_indices(
        total=total,
        train_size=image_train_size,
        val_size=image_val_size,
        seed=image_split_seed,
    )
    selected = train_idx if limited_data_enabled else full_train_idx
    return [int(v) for v in selected.tolist()]


def snapshot_path_for_kimg(run_dir: str, kimg: int) -> str:
    return os.path.join(run_dir, f"network-snapshot-{int(kimg):06d}.pkl")


def state_dump_path_for_kimg(run_dir: str, kimg: int) -> str:
    return os.path.join(run_dir, f"training-state-{int(kimg):06d}.pt")


def build_mainline_baseline_segment_cmd(
    *,
    python_bin: str,
    run_dir: str,
    cfg,
    target_steps: int,
    resume_steps: int = 0,
) -> list[str]:
    """Build a subprocess command for one mainline EDM segment."""

    validate_mainline_baseline_request(cfg)
    target_kimg = steps_to_kimg_exact(target_steps, cfg.batch_size, label="target_steps")
    resume_kimg = steps_to_kimg_exact(resume_steps, cfg.batch_size, label="resume_steps")
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script_path = os.path.join(repo_root, "toy", "scripts", "train_mainline_edm_baseline.py")

    cmd = [
        python_bin,
        script_path,
        "--run-dir",
        str(run_dir),
        "--dataset-path",
        str(getattr(cfg, "dataset_path", "")),
        "--image-size",
        str(int(getattr(cfg, "image_size", 0))),
        "--image-train-size",
        str(int(getattr(cfg, "image_train_size", 0))),
        "--image-val-size",
        str(int(getattr(cfg, "image_val_size", 0))),
        "--image-split-seed",
        str(int(getattr(cfg, "image_split_seed", 0))),
        "--batch-size",
        str(int(getattr(cfg, "batch_size", 0))),
        "--batch-gpu",
        str(int(resolve_baseline_train_batch_gpu(cfg) or 0)),
        "--hidden-dim",
        str(int(getattr(cfg, "hidden_dim", 0))),
        "--seed",
        str(int(getattr(cfg, "seed", 0))),
        "--target-steps",
        str(int(target_steps)),
        "--resume-steps",
        str(int(resume_steps)),
        "--sigma-data",
        str(float(getattr(cfg, "sigma_data", 0.5))),
        "--lr-theta",
        str(float(getattr(cfg, "lr_theta", 0.0))),
        "--p-mean",
        str(float(getattr(cfg, "p_mean", -1.2))),
        "--p-std",
        str(float(getattr(cfg, "p_std", 1.2))),
        "--ema-halflife-kimg",
        str(float(getattr(cfg, "ema_halflife_kimg", 500.0))),
        "--dropout",
        str(mainline_ddpmpp_dropout()),
    ]
    if getattr(cfg, "dataset_val_path", ""):
        cmd.extend(["--dataset-val-path", str(cfg.dataset_val_path)])
    if bool(getattr(cfg, "limited_data_enabled", True)):
        cmd.append("--limited-data-enabled")
    if bool(getattr(cfg, "cudnn_benchmark", True)):
        cmd.append("--cudnn-benchmark")
    if getattr(cfg, "ema_rampup_ratio", None) is None:
        cmd.append("--disable-ema-rampup")
    else:
        cmd.extend(["--ema-rampup-ratio", str(float(cfg.ema_rampup_ratio))])
    if int(resume_steps) > 0:
        cmd.extend(
            [
                "--resume-pkl",
                snapshot_path_for_kimg(run_dir, resume_kimg),
                "--resume-state-dump",
                state_dump_path_for_kimg(run_dir, resume_kimg),
            ]
        )
    return cmd


def load_mainline_stats_loss_history(run_dir: str) -> list[float]:
    """Parse the main trainer's `stats.jsonl` loss means into a toy history curve."""

    stats_path = os.path.join(run_dir, "stats.jsonl")
    if not os.path.isfile(stats_path):
        return []

    losses: list[float] = []
    with open(stats_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            block = payload.get("Loss/loss")
            if not isinstance(block, dict):
                continue
            value = block.get("mean")
            if value is None:
                continue
            try:
                losses.append(float(value))
            except (TypeError, ValueError):
                continue
    return losses


def load_mainline_snapshot_ema_state_dict(snapshot_path: str) -> dict[str, torch.Tensor]:
    """Load EMA weights from a `training.training_loop` snapshot pickle."""

    with open(snapshot_path, "rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or "ema" not in payload:
        raise RuntimeError(f"Invalid mainline snapshot payload: {snapshot_path}")
    ema_model = payload["ema"]
    if not isinstance(ema_model, torch.nn.Module):
        raise RuntimeError(f"Snapshot does not contain an EMA module: {snapshot_path}")
    return {key: value.detach().cpu() for key, value in ema_model.state_dict().items()}


def load_mainline_ema_into_toy_model(model: torch.nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    """Load mainline EDMPrecond EMA weights into the toy ddpmpp wrapper."""

    target_keys = set(model.state_dict().keys())
    source_keys = set(state_dict.keys())
    if source_keys == target_keys:
        model.load_state_dict(state_dict, strict=True)
        return

    prefixed_state = {f"model.{key}": value for key, value in state_dict.items()}
    if set(prefixed_state.keys()) == target_keys:
        model.load_state_dict(prefixed_state, strict=True)
        return

    raise RuntimeError(
        "Could not map mainline EMA state dict onto toy model.\n"
        f"target_example={sorted(target_keys)[:3]}\n"
        f"source_example={sorted(source_keys)[:3]}"
    )
