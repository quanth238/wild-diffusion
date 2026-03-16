from __future__ import annotations

import math

import torch
from torch import nn


class GaussianFourierEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, scale: float = 16.0):
        super().__init__()
        if embedding_dim % 2 != 0:
            raise ValueError("embedding_dim must be even.")
        weight = torch.randn(embedding_dim // 2) * scale
        self.register_buffer("weight", weight)

    def forward(self, noise_level: torch.Tensor) -> torch.Tensor:
        angles = noise_level[:, None] * self.weight[None, :] * 2.0 * math.pi
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)


class EDMPrecondMLP(nn.Module):
    def __init__(
        self,
        *,
        data_dim: int = 2,
        hidden_dim: int = 128,
        depth: int = 4,
        embedding_dim: int = 32,
        sigma_data: float = 0.5,
    ):
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be at least 2.")
        self.data_dim = data_dim
        self.sigma_data = float(sigma_data)
        self.embedding = GaussianFourierEmbedding(embedding_dim=embedding_dim)

        layers: list[nn.Module] = []
        in_dim = data_dim + embedding_dim
        for layer_idx in range(depth - 1):
            layers.append(nn.Linear(in_dim if layer_idx == 0 else hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden_dim, data_dim))
        self.backbone = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        sigma = _reshape_sigma_like_input(sigma=sigma, x=x)
        sigma_sq = sigma.square()
        sigma_data_sq = self.sigma_data ** 2

        c_skip = sigma_data_sq / (sigma_sq + sigma_data_sq)
        c_out = sigma * self.sigma_data / torch.sqrt(sigma_sq + sigma_data_sq)
        c_in = 1.0 / torch.sqrt(sigma_sq + sigma_data_sq)
        c_noise = sigma.log().squeeze(1) / 4.0

        embedding = self.embedding(c_noise)
        network_in = torch.cat([c_in * x, embedding], dim=1)
        model_out = self.backbone(network_in)
        return c_skip * x + c_out * model_out


def _reshape_sigma_like_input(sigma: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    sigma = sigma.to(device=x.device, dtype=x.dtype)
    if sigma.ndim == 0:
        sigma = sigma.repeat(x.shape[0])
    if sigma.ndim == 1:
        sigma = sigma.unsqueeze(1)
    if sigma.ndim != 2 or sigma.shape[0] != x.shape[0] or sigma.shape[1] != 1:
        raise ValueError(f"sigma must broadcast to [batch, 1], got shape {tuple(sigma.shape)}")
    return sigma
