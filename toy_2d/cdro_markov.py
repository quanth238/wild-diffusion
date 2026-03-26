from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from toy_2d.model import GaussianFourierEmbedding


@dataclass(frozen=True)
class MarkovVPSchedule:
    total_time: float
    dt: torch.Tensor
    beta: torch.Tensor
    g: torch.Tensor
    step_sigma: torch.Tensor
    weights: torch.Tensor

    def to_dict(self) -> dict[str, float | list[float]]:
        return {
            "total_time": float(self.total_time),
            "dt": self.dt.detach().cpu().tolist(),
            "beta": self.beta.detach().cpu().tolist(),
            "g": self.g.detach().cpu().tolist(),
            "step_sigma": self.step_sigma.detach().cpu().tolist(),
            "weights": self.weights.detach().cpu().tolist(),
        }


@dataclass
class ForwardRollout:
    states: torch.Tensor
    noises: torch.Tensor
    controls: torch.Tensor
    means: torch.Tensor


@dataclass
class TerminalStats:
    mean: torch.Tensor
    var: torch.Tensor


class MarkovScoreMLP(nn.Module):
    def __init__(
        self,
        *,
        data_dim: int = 2,
        hidden_dim: int = 128,
        depth: int = 4,
        embedding_dim: int = 32,
    ):
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be at least 2.")
        self.data_dim = data_dim
        self.embedding = GaussianFourierEmbedding(embedding_dim=embedding_dim)

        layers: list[nn.Module] = []
        in_dim = data_dim + embedding_dim
        for layer_idx in range(depth - 1):
            layers.append(nn.Linear(in_dim if layer_idx == 0 else hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden_dim, data_dim))
        self.backbone = nn.Sequential(*layers)

    def forward(self, y: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        sigma = _reshape_sigma_like_input(sigma=sigma, x=y)
        embedding = self.embedding(torch.log(sigma.clamp(min=1e-8)).squeeze(1) / 4.0)
        return self.backbone(torch.cat([y, embedding], dim=1))


class MarkovControlMLP(nn.Module):
    def __init__(
        self,
        *,
        data_dim: int = 2,
        hidden_dim: int = 64,
        depth: int = 3,
        embedding_dim: int = 32,
        control_scale: float = 0.5,
    ):
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be at least 2.")
        self.data_dim = data_dim
        self.control_scale = float(control_scale)
        self.embedding = GaussianFourierEmbedding(embedding_dim=embedding_dim)

        layers: list[nn.Module] = []
        in_dim = data_dim + embedding_dim
        for layer_idx in range(depth - 1):
            layers.append(nn.Linear(in_dim if layer_idx == 0 else hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden_dim, data_dim))
        self.backbone = nn.Sequential(*layers)

    def forward(self, y: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        sigma = _reshape_sigma_like_input(sigma=sigma, x=y)
        embedding = self.embedding(torch.log(sigma.clamp(min=1e-8)).squeeze(1) / 4.0)
        raw = self.backbone(torch.cat([y, embedding], dim=1))
        return self.control_scale * torch.tanh(raw)

    def initial_state(self, *, batch_size: int, device: torch.device, dtype: torch.dtype):
        del batch_size, device, dtype
        return None

    def step(self, y: torch.Tensor, sigma: torch.Tensor, state):
        del state
        return self.forward(y, sigma), None


class MarkovControlGRU(nn.Module):
    def __init__(
        self,
        *,
        data_dim: int = 2,
        hidden_dim: int = 64,
        depth: int = 3,
        embedding_dim: int = 32,
        control_scale: float = 0.5,
    ):
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be at least 2.")
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.control_scale = float(control_scale)
        self.embedding = GaussianFourierEmbedding(embedding_dim=embedding_dim)
        self.gru_cell = nn.GRUCell(data_dim + embedding_dim, hidden_dim)

        head_layers: list[nn.Module] = []
        for _ in range(depth - 2):
            head_layers.append(nn.Linear(hidden_dim, hidden_dim))
            head_layers.append(nn.SiLU())
        head_layers.append(nn.Linear(hidden_dim, data_dim))
        self.head = nn.Sequential(*head_layers)

    def initial_state(self, *, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)

    def step(self, y: torch.Tensor, sigma: torch.Tensor, state: torch.Tensor | None):
        sigma = _reshape_sigma_like_input(sigma=sigma, x=y)
        if state is None:
            state = self.initial_state(batch_size=y.shape[0], device=y.device, dtype=y.dtype)
        embedding = self.embedding(torch.log(sigma.clamp(min=1e-8)).squeeze(1) / 4.0)
        hidden = self.gru_cell(torch.cat([y, embedding], dim=1), state)
        raw = self.head(hidden)
        return self.control_scale * torch.tanh(raw), hidden

    def forward(self, y: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        control, _ = self.step(y, sigma, state=None)
        return control


def build_markov_vp_schedule(
    *,
    num_steps: int,
    total_time: float,
    beta_min: float,
    beta_max: float,
    device: torch.device,
    dtype: torch.dtype,
    weight_schedule: str = "uniform",
) -> MarkovVPSchedule:
    if num_steps < 1:
        raise ValueError("num_steps must be at least 1.")
    if total_time <= 0.0:
        raise ValueError("total_time must be positive.")
    dt = torch.full((num_steps,), float(total_time) / float(num_steps), device=device, dtype=dtype)
    beta = torch.linspace(beta_min, beta_max, steps=num_steps, device=device, dtype=dtype)
    g = beta.sqrt()
    step_sigma = (beta * dt).sqrt()
    weights = build_step_weights(step_sigma=step_sigma, schedule=weight_schedule)
    return MarkovVPSchedule(
        total_time=float(total_time),
        dt=dt,
        beta=beta,
        g=g,
        step_sigma=step_sigma,
        weights=weights,
    )


def build_step_weights(*, step_sigma: torch.Tensor, schedule: str) -> torch.Tensor:
    if schedule == "uniform":
        return torch.ones_like(step_sigma)
    if schedule == "sigma_sq":
        return step_sigma.square()
    if schedule == "inv_sigma_sq":
        return step_sigma.square().clamp(min=1e-8).reciprocal()
    raise ValueError(f"Unsupported score weight schedule: {schedule}")


def rollout_markov_forward(
    *,
    clean_points: torch.Tensor,
    control_net,
    schedule: MarkovVPSchedule,
    zero_control: bool = False,
) -> ForwardRollout:
    current = clean_points
    states = [current]
    noises: list[torch.Tensor] = []
    controls: list[torch.Tensor] = []
    means: list[torch.Tensor] = []
    control_state = initialize_control_state(
        control_net=control_net,
        batch_size=clean_points.shape[0],
        device=clean_points.device,
        dtype=clean_points.dtype,
    )

    for step_idx in range(schedule.dt.shape[0]):
        sigma_step = schedule.step_sigma[step_idx].expand(clean_points.shape[0])
        if zero_control or control_net is None:
            control = torch.zeros_like(current)
        else:
            control, control_state = eval_control_step(
                control_net=control_net,
                current=current,
                sigma_step=sigma_step,
                control_state=control_state,
            )
        drift = forward_drift(current, beta=schedule.beta[step_idx], control=control)
        mean = current + drift * schedule.dt[step_idx]
        noise = torch.randn_like(current)
        current = mean + schedule.step_sigma[step_idx] * noise
        states.append(current)
        noises.append(noise)
        controls.append(control)
        means.append(mean)

    return ForwardRollout(
        states=torch.stack(states, dim=1),
        noises=torch.stack(noises, dim=1),
        controls=torch.stack(controls, dim=1),
        means=torch.stack(means, dim=1),
    )


def forward_drift(points: torch.Tensor, *, beta: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
    return -0.5 * beta * points + control


def compute_score_matching_loss(
    *,
    score_net,
    rollout: ForwardRollout,
    schedule: MarkovVPSchedule,
) -> torch.Tensor:
    batch_size, num_steps, data_dim = rollout.noises.shape
    observations = rollout.states[:, 1:, :].reshape(batch_size * num_steps, data_dim)
    sigma = schedule.step_sigma.view(1, num_steps, 1).expand(batch_size, num_steps, 1)
    sigma_flat = sigma.reshape(batch_size * num_steps)
    score_pred = score_net(observations, sigma_flat).view(batch_size, num_steps, data_dim)
    score_target = -rollout.noises / sigma
    sq_error = (score_pred - score_target).square().mean(dim=2)
    return (sq_error * schedule.weights.view(1, num_steps)).mean()


def compute_control_cost(*, rollout: ForwardRollout, schedule: MarkovVPSchedule) -> torch.Tensor:
    return (rollout.controls.square().sum(dim=2) * schedule.dt.view(1, -1)).mean()


def update_terminal_stats(
    *,
    terminal_states: torch.Tensor,
    current: TerminalStats | None,
    momentum: float,
    min_var: float = 1e-4,
) -> TerminalStats:
    batch_mean = terminal_states.mean(dim=0).detach()
    batch_var = terminal_states.var(dim=0, unbiased=False).clamp(min=min_var).detach()
    if current is None:
        return TerminalStats(mean=batch_mean, var=batch_var)
    mean = current.mean * momentum + batch_mean * (1.0 - momentum)
    var = current.var * momentum + batch_var * (1.0 - momentum)
    return TerminalStats(mean=mean, var=var.clamp(min=min_var))


@torch.no_grad()
def sample_reverse_chain(
    score_net,
    *,
    control_net,
    schedule: MarkovVPSchedule,
    terminal_stats: TerminalStats,
    num_samples: int,
    data_dim: int,
    device: torch.device,
    zero_control: bool = False,
    collect_states: bool = True,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    mean = terminal_stats.mean.to(device=device)
    var = terminal_stats.var.to(device=device).clamp(min=1e-6)
    current = mean.view(1, data_dim) + torch.randn(num_samples, data_dim, device=device) * var.sqrt().view(1, data_dim)
    control_state = initialize_control_state(
        control_net=control_net,
        batch_size=num_samples,
        device=device,
        dtype=current.dtype,
    )

    reverse_states = [current.detach().cpu()] if collect_states else []
    for step_idx in range(schedule.dt.shape[0] - 1, -1, -1):
        sigma_step = schedule.step_sigma[step_idx].expand(num_samples)
        score = score_net(current, sigma_step)
        if zero_control or control_net is None:
            control = torch.zeros_like(current)
        else:
            control, control_state = eval_control_step(
                control_net=control_net,
                current=current,
                sigma_step=sigma_step,
                control_state=control_state,
            )
        drift = forward_drift(current, beta=schedule.beta[step_idx], control=control)
        noise = torch.randn_like(current)
        current = current + (-drift + schedule.g[step_idx].square() * score) * schedule.dt[step_idx]
        current = current + schedule.step_sigma[step_idx] * noise
        if collect_states:
            reverse_states.append(current.detach().cpu())

    if collect_states:
        reverse_states.reverse()
    return current, reverse_states


def set_module_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


@torch.no_grad()
def update_ema(*, ema, model, decay: float) -> None:
    for ema_param, model_param in zip(ema.parameters(), model.parameters()):
        ema_param.data.mul_(decay).add_(model_param.data, alpha=1.0 - decay)
    for ema_buffer, model_buffer in zip(ema.buffers(), model.buffers()):
        ema_buffer.copy_(model_buffer)


def _reshape_sigma_like_input(sigma: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    sigma = sigma.to(device=x.device, dtype=x.dtype)
    if sigma.ndim == 0:
        sigma = sigma.repeat(x.shape[0])
    if sigma.ndim == 1:
        sigma = sigma.unsqueeze(1)
    if sigma.ndim != 2 or sigma.shape[0] != x.shape[0] or sigma.shape[1] != 1:
        raise ValueError(f"sigma must broadcast to [batch, 1], got shape {tuple(sigma.shape)}")
    return sigma


def initialize_control_state(*, control_net, batch_size: int, device: torch.device, dtype: torch.dtype):
    if control_net is None or not hasattr(control_net, "initial_state"):
        return None
    return control_net.initial_state(batch_size=batch_size, device=device, dtype=dtype)


def eval_control_step(*, control_net, current: torch.Tensor, sigma_step: torch.Tensor, control_state):
    if hasattr(control_net, "step"):
        return control_net.step(current, sigma_step, control_state)
    return control_net(current, sigma_step), control_state
