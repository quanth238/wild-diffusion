from dataclasses import dataclass

import torch

from ..models import (
    ControlNet,
    ImageControlNet,
    ImageDDPMPPDenoiser,
    ImageDDPMPPRectifiedFlowModel,
    ImageDDPMPPScoreModel,
    ImageEDMDenoiser,
    ImageRectifiedFlowModel,
    ImageScoreModel,
    ImageSongUNetDenoiser,
    ImageSongUNetRectifiedFlowModel,
    ImageSongUNetScoreModel,
    ToyEDMDenoiser,
    ToyRectifiedFlowModel,
    ToyScoreModel,
)


@dataclass
class ModelBundle:
    """Model contract consumed by the experiment orchestrator."""

    name: str
    baseline: torch.nn.Module
    robust: torch.nn.Module
    control: torch.nn.Module


def resolve_image_denoiser_spec(
    cfg,
    *,
    in_channels: int,
    image_resolution: int,
    sigma_data: float,
) -> tuple[str, type[torch.nn.Module], dict]:
    """Resolve the active image denoiser constructor from config."""

    objective = str(getattr(cfg, "training_objective", "edm")).lower()
    image_backbone = str(getattr(cfg, "image_backbone", "conv")).strip().lower()

    if image_backbone == "conv":
        if objective == "edm":
            return "image_conv", ImageEDMDenoiser, dict(
                in_channels=in_channels,
                hidden_dim=cfg.hidden_dim,
                sigma_data=sigma_data,
            )
        if objective == "score":
            return "image_conv", ImageScoreModel, dict(
                in_channels=in_channels,
                hidden_dim=cfg.hidden_dim,
                sigma_data=sigma_data,
            )
        return "image_conv", ImageRectifiedFlowModel, dict(
            in_channels=in_channels,
            hidden_dim=cfg.hidden_dim,
            sigma_max=cfg.sigma_max,
        )

    if image_backbone == "songunet":
        if objective == "edm":
            return "image_songunet", ImageSongUNetDenoiser, dict(
                img_resolution=image_resolution,
                in_channels=in_channels,
                hidden_dim=cfg.hidden_dim,
                sigma_data=sigma_data,
            )
        if objective == "score":
            return "image_songunet", ImageSongUNetScoreModel, dict(
                img_resolution=image_resolution,
                in_channels=in_channels,
                hidden_dim=cfg.hidden_dim,
                sigma_data=sigma_data,
            )
        return "image_songunet", ImageSongUNetRectifiedFlowModel, dict(
            img_resolution=image_resolution,
            in_channels=in_channels,
            hidden_dim=cfg.hidden_dim,
            sigma_max=cfg.sigma_max,
        )

    if image_backbone == "ddpmpp":
        if objective == "edm":
            return "image_ddpmpp", ImageDDPMPPDenoiser, dict(
                img_resolution=image_resolution,
                in_channels=in_channels,
                hidden_dim=cfg.hidden_dim,
                sigma_data=sigma_data,
            )
        if objective == "score":
            return "image_ddpmpp", ImageDDPMPPScoreModel, dict(
                img_resolution=image_resolution,
                in_channels=in_channels,
                hidden_dim=cfg.hidden_dim,
                sigma_data=sigma_data,
            )
        return "image_ddpmpp", ImageDDPMPPRectifiedFlowModel, dict(
            img_resolution=image_resolution,
            in_channels=in_channels,
            hidden_dim=cfg.hidden_dim,
            sigma_max=cfg.sigma_max,
        )

    raise ValueError(
        f"Unsupported image_backbone='{getattr(cfg, 'image_backbone', image_backbone)}'. "
        "Expected one of: conv, songunet, ddpmpp."
    )


def build_model_bundle(cfg, dataset, sigma_data: float, device: torch.device) -> ModelBundle:
    """Factory for denoiser/control backends used by the toy protocol."""

    objective = str(getattr(cfg, "training_objective", "edm")).lower()
    if objective not in ("edm", "score", "rf"):
        raise ValueError(f"Unsupported training_objective='{objective}'. Expected one of: edm, score, rf.")

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
        height = int(dataset.data_shape[1])
        width = int(dataset.data_shape[2])
        image_backbone = str(getattr(cfg, "image_backbone", "conv")).strip().lower()
        if image_backbone in ("songunet", "ddpmpp") and height != width:
            raise ValueError(
                f"image_backbone='{image_backbone}' expects square image samples with data_shape=(C,H,W), "
                f"got data_shape={dataset.data_shape}"
            )
        backend_name, denoiser_cls, denoiser_kwargs = resolve_image_denoiser_spec(
            cfg,
            in_channels=channels,
            image_resolution=height,
            sigma_data=sigma_data,
        )
        return ModelBundle(
            name=f"{backend_name}_{objective}",
            baseline=denoiser_cls(**denoiser_kwargs).to(device),
            robust=denoiser_cls(**denoiser_kwargs).to(device),
            control=ImageControlNet(in_channels=channels, hidden_dim=cfg.hidden_dim).to(device),
        )

    if tuple(dataset.data_shape) != (2,):
        raise ValueError(
            "model_kind='toy_mlp' expects rank-1 2D samples with data_shape=(2,), "
            f"got data_shape={dataset.data_shape}"
        )

    if objective == "edm":
        denoiser_cls = ToyEDMDenoiser
        denoiser_kwargs = dict(hidden_dim=cfg.hidden_dim, sigma_data=sigma_data)
    elif objective == "score":
        denoiser_cls = ToyScoreModel
        denoiser_kwargs = dict(hidden_dim=cfg.hidden_dim, sigma_data=sigma_data)
    else:
        denoiser_cls = ToyRectifiedFlowModel
        denoiser_kwargs = dict(hidden_dim=cfg.hidden_dim, sigma_max=cfg.sigma_max)
    return ModelBundle(
        name=f"toy_mlp_{objective}",
        baseline=denoiser_cls(**denoiser_kwargs).to(device),
        robust=denoiser_cls(**denoiser_kwargs).to(device),
        control=ControlNet(cfg.hidden_dim).to(device),
    )
