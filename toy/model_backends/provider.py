from dataclasses import dataclass

import torch

from ..models import ControlNet, ImageControlNet, ImageEDMDenoiser, ToyEDMDenoiser


@dataclass
class ModelBundle:
    """Model contract consumed by the experiment orchestrator."""

    name: str
    baseline: torch.nn.Module
    robust: torch.nn.Module
    control: torch.nn.Module


def build_model_bundle(cfg, dataset, sigma_data: float, device: torch.device) -> ModelBundle:
    """Factory for denoiser/control backends used by the toy protocol."""

    model_kind = cfg.model_kind
    if model_kind == "auto":
        if dataset.name == "toy_gmm":
            model_kind = "toy_mlp"
        elif dataset.name == "image_folder":
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
        return ModelBundle(
            name="image_conv",
            baseline=ImageEDMDenoiser(in_channels=channels, hidden_dim=cfg.hidden_dim, sigma_data=sigma_data).to(device),
            robust=ImageEDMDenoiser(in_channels=channels, hidden_dim=cfg.hidden_dim, sigma_data=sigma_data).to(device),
            control=ImageControlNet(in_channels=channels, hidden_dim=cfg.hidden_dim).to(device),
        )

    if tuple(dataset.data_shape) != (2,):
        raise ValueError(
            "model_kind='toy_mlp' expects rank-1 2D samples with data_shape=(2,), "
            f"got data_shape={dataset.data_shape}"
        )

    return ModelBundle(
        name="toy_mlp",
        baseline=ToyEDMDenoiser(cfg.hidden_dim, sigma_data=sigma_data).to(device),
        robust=ToyEDMDenoiser(cfg.hidden_dim, sigma_data=sigma_data).to(device),
        control=ControlNet(cfg.hidden_dim).to(device),
    )
