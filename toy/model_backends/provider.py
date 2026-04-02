from dataclasses import dataclass

import torch

from ..models import (
    ControlNet,
    ImageControlNet,
    ImageEDMDenoiser,
    ImageScoreModel,
    ToyEDMDenoiser,
    ToyScoreModel,
)


@dataclass
class ModelBundle:
    """Model contract consumed by the experiment orchestrator."""

    name: str
    baseline: torch.nn.Module
    robust: torch.nn.Module
    control: torch.nn.Module


def build_model_bundle(cfg, dataset, sigma_data: float, device: torch.device) -> ModelBundle:
    """Factory for denoiser/control backends used by the toy protocol."""

    objective = str(getattr(cfg, "training_objective", "edm")).lower()
    if objective not in ("edm", "score"):
        raise ValueError(f"Unsupported training_objective='{objective}'. Expected one of: edm, score.")

    model_kind = cfg.model_kind
    if model_kind == "auto":
        if dataset.name == "toy_gmm":
            model_kind = "toy_mlp"
        elif dataset.name in ("image_folder", "mnist"):
            model_kind = "image_conv"
        else:
            raise NotImplementedError(
                f"Cannot auto-resolve model backend for dataset backend '{dataset.name}'. "
                "Set --model-kind explicitly."
            )

    if model_kind != "toy_mlp":
        if model_kind != "image_conv":
            raise NotImplementedError(
                f"Unsupported model_kind='{cfg.model_kind}'. "
                "Add a new branch in toy/model_backends/provider.py for your model family."
            )
        if len(dataset.data_shape) != 3:
            raise ValueError(
                "model_kind='image_conv' expects image-shaped samples with data_shape=(C,H,W), "
                f"got data_shape={dataset.data_shape}"
            )
        channels = int(dataset.data_shape[0])
        denoiser_cls = ImageEDMDenoiser if objective == "edm" else ImageScoreModel
        return ModelBundle(
            name=f"image_conv_{objective}",
            baseline=denoiser_cls(in_channels=channels, hidden_dim=cfg.hidden_dim, sigma_data=sigma_data).to(device),
            robust=denoiser_cls(in_channels=channels, hidden_dim=cfg.hidden_dim, sigma_data=sigma_data).to(device),
            control=ImageControlNet(in_channels=channels, hidden_dim=cfg.hidden_dim).to(device),
        )

    if tuple(dataset.data_shape) != (2,):
        raise ValueError(
            "model_kind='toy_mlp' expects rank-1 2D samples with data_shape=(2,), "
            f"got data_shape={dataset.data_shape}"
        )

    denoiser_cls = ToyEDMDenoiser if objective == "edm" else ToyScoreModel
    return ModelBundle(
        name=f"toy_mlp_{objective}",
        baseline=denoiser_cls(cfg.hidden_dim, sigma_data=sigma_data).to(device),
        robust=denoiser_cls(cfg.hidden_dim, sigma_data=sigma_data).to(device),
        control=ControlNet(cfg.hidden_dim).to(device),
    )
