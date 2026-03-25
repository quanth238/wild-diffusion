from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        if embedding_dim < 2 or embedding_dim % 2 != 0:
            raise ValueError("embedding_dim must be an even integer >= 2.")
        self.embedding_dim = int(embedding_dim)

    def forward(self, time_values: torch.Tensor) -> torch.Tensor:
        if time_values.ndim == 0:
            time_values = time_values.view(1)
        if time_values.ndim != 1:
            raise ValueError(f"time_values must have shape [steps] or [batch], got {tuple(time_values.shape)}")
        half_dim = self.embedding_dim // 2
        device = time_values.device
        dtype = time_values.dtype
        exponents = torch.arange(half_dim, device=device, dtype=dtype)
        exponents = exponents / max(half_dim - 1, 1)
        frequencies = torch.exp(-math.log(10000.0) * exponents)
        angles = time_values[:, None] * frequencies[None, :]
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)


class CausalPredictorGRU(nn.Module):
    def __init__(
        self,
        *,
        data_dim: int,
        hidden_dim: int,
        time_embedding_dim: int,
        num_layers: int = 1,
    ):
        super().__init__()
        self.data_dim = int(data_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.time_embedding = SinusoidalTimeEmbedding(time_embedding_dim)
        self.gru = nn.GRU(
            input_size=self.data_dim + time_embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.data_dim),
        )

    def forward(self, *, path_states: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        batch_size, num_steps, data_dim = path_states.shape
        if data_dim != self.data_dim:
            raise ValueError(f"Expected path_states data_dim={self.data_dim}, got {data_dim}")
        if times.shape != (num_steps,):
            raise ValueError(f"Expected times shape {(num_steps,)}, got {tuple(times.shape)}")
        time_embed = self.time_embedding(times.to(device=path_states.device, dtype=path_states.dtype))
        time_embed = time_embed.unsqueeze(0).expand(batch_size, num_steps, -1)
        gru_in = torch.cat([path_states, time_embed], dim=2)
        features, _ = self.gru(gru_in)
        return self.head(features)


class CausalAdversaryGRU(nn.Module):
    def __init__(
        self,
        *,
        data_dim: int,
        hidden_dim: int,
        time_embedding_dim: int,
        control_scale: float,
    ):
        super().__init__()
        if control_scale <= 0.0:
            raise ValueError("control_scale must be positive.")
        self.data_dim = int(data_dim)
        self.hidden_dim = int(hidden_dim)
        self.control_scale = float(control_scale)
        self.time_embedding = SinusoidalTimeEmbedding(time_embedding_dim)
        self.cell = nn.GRUCell(input_size=self.data_dim + time_embedding_dim, hidden_size=self.hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.data_dim),
        )

    def init_hidden(self, *, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)

    def step(
        self,
        *,
        current_state: torch.Tensor,
        time_value: torch.Tensor,
        hidden_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if time_value.ndim == 0:
            time_value = time_value.view(1)
        if time_value.shape != (1,):
            raise ValueError(f"time_value must have shape [1], got {tuple(time_value.shape)}")
        embedded_time = self.time_embedding(time_value.to(device=current_state.device, dtype=current_state.dtype))
        embedded_time = embedded_time.expand(current_state.shape[0], -1)
        cell_input = torch.cat([current_state, embedded_time], dim=1)
        hidden_next = self.cell(cell_input, hidden_state)
        control = torch.tanh(self.head(hidden_next)) * self.control_scale
        return control, hidden_next


class ReverseGaussianModel(nn.Module):
    def __init__(
        self,
        *,
        data_dim: int,
        hidden_dim: int,
        time_embedding_dim: int,
    ):
        super().__init__()
        self.data_dim = int(data_dim)
        self.hidden_dim = int(hidden_dim)
        self.time_embedding = SinusoidalTimeEmbedding(time_embedding_dim)

        self.terminal_mean = nn.Parameter(torch.zeros(self.data_dim))
        self.terminal_logvar = nn.Parameter(torch.zeros(self.data_dim))

        self.terminal_encoder = nn.Sequential(
            nn.Linear(self.data_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.reverse_cell = nn.GRUCell(
            input_size=self.data_dim + time_embedding_dim,
            hidden_size=self.hidden_dim,
        )
        self.mean_head = nn.Sequential(
            nn.Linear(self.hidden_dim + self.data_dim + time_embedding_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.data_dim),
        )

    def sample_terminal(
        self,
        *,
        num_samples: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        mean = self.terminal_mean.to(device=device, dtype=dtype)
        std = torch.exp(0.5 * self.terminal_logvar).to(device=device, dtype=dtype)
        eps = torch.randn(num_samples, self.data_dim, device=device, dtype=dtype)
        return mean.unsqueeze(0) + std.unsqueeze(0) * eps

    def terminal_nll(self, terminal_states: torch.Tensor) -> torch.Tensor:
        mean = self.terminal_mean.to(device=terminal_states.device, dtype=terminal_states.dtype)
        logvar = self.terminal_logvar.to(device=terminal_states.device, dtype=terminal_states.dtype)
        inv_var = torch.exp(-logvar)
        sq_mahalanobis = (terminal_states - mean.unsqueeze(0)).square() * inv_var.unsqueeze(0)
        return 0.5 * (logvar.unsqueeze(0) + sq_mahalanobis).sum(dim=1)

    def init_reverse_hidden(self, terminal_states: torch.Tensor) -> torch.Tensor:
        return self.terminal_encoder(terminal_states)

    def reverse_mean(
        self,
        *,
        next_states: torch.Tensor,
        time_value: torch.Tensor,
        hidden_state: torch.Tensor,
    ) -> torch.Tensor:
        if time_value.ndim == 0:
            time_value = time_value.view(1)
        time_embed = self.time_embedding(time_value.to(device=next_states.device, dtype=next_states.dtype))
        time_embed = time_embed.expand(next_states.shape[0], -1)
        mean_input = torch.cat([hidden_state, next_states, time_embed], dim=1)
        return self.mean_head(mean_input)

    def update_hidden(
        self,
        *,
        current_states: torch.Tensor,
        time_value: torch.Tensor,
        hidden_state: torch.Tensor,
    ) -> torch.Tensor:
        if time_value.ndim == 0:
            time_value = time_value.view(1)
        time_embed = self.time_embedding(time_value.to(device=current_states.device, dtype=current_states.dtype))
        time_embed = time_embed.expand(current_states.shape[0], -1)
        cell_input = torch.cat([current_states, time_embed], dim=1)
        return self.reverse_cell(cell_input, hidden_state)
