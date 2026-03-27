from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from toy_2d.model import GaussianFourierEmbedding


@dataclass(frozen=True)
class MarkovVPSchedule:
    family: str
    total_time: float
    dt: torch.Tensor
    drift_coeff: torch.Tensor
    beta: torch.Tensor
    g: torch.Tensor
    step_sigma: torch.Tensor
    weights: torch.Tensor
    sigma_levels: torch.Tensor | None = None

    def to_dict(self) -> dict[str, float | list[float]]:
        return {
            "family": self.family,
            "total_time": float(self.total_time),
            "dt": self.dt.detach().cpu().tolist(),
            "drift_coeff": self.drift_coeff.detach().cpu().tolist(),
            "beta": self.beta.detach().cpu().tolist(),
            "g": self.g.detach().cpu().tolist(),
            "step_sigma": self.step_sigma.detach().cpu().tolist(),
            "weights": self.weights.detach().cpu().tolist(),
            "sigma_levels": None if self.sigma_levels is None else self.sigma_levels.detach().cpu().tolist(),
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
    replay: torch.Tensor | None = None


@dataclass
class ScoreWdroRefreshResult:
    combined_points: torch.Tensor
    adv_original: torch.Tensor | None
    adv_generated: torch.Tensor | None
    mean_l2_shift: float
    max_l2_shift: float
    mean_transport_cost: float
    max_transport_cost: float


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


class MarkovPrecondScoreMLP(nn.Module):
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

    def forward(self, y: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        sigma = _reshape_sigma_like_input(sigma=sigma, x=y)
        sigma_sq = sigma.square().clamp(min=1e-8)
        sigma_data_sq = self.sigma_data ** 2

        c_skip = sigma_data_sq / (sigma_sq + sigma_data_sq)
        c_out = sigma * self.sigma_data / torch.sqrt(sigma_sq + sigma_data_sq)
        c_in = 1.0 / torch.sqrt(sigma_sq + sigma_data_sq)
        c_noise = sigma.log().squeeze(1) / 4.0

        embedding = self.embedding(c_noise)
        model_out = self.backbone(torch.cat([c_in * y, embedding], dim=1))
        denoised = c_skip * y + c_out * model_out
        return (denoised - y) / sigma_sq


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


def build_markov_schedule(
    *,
    num_steps: int,
    total_time: float,
    beta_min: float,
    beta_max: float,
    sde_family: str,
    device: torch.device,
    dtype: torch.dtype,
    weight_schedule: str = "uniform",
    ve_sigma_min: float = 0.01,
    ve_sigma_max: float = 3.0,
    cosine_s: float = 0.008,
) -> MarkovVPSchedule:
    if num_steps < 1:
        raise ValueError("num_steps must be at least 1.")
    if total_time <= 0.0:
        raise ValueError("total_time must be positive.")
    dt = torch.full((num_steps,), float(total_time) / float(num_steps), device=device, dtype=dtype)
    sigma_levels: torch.Tensor | None = None

    if sde_family == "vp_linear":
        beta = torch.linspace(beta_min, beta_max, steps=num_steps, device=device, dtype=dtype)
        mean_scale = torch.exp(-0.5 * beta * dt)
        step_sigma = (1.0 - torch.exp(-beta * dt)).clamp(min=1e-8).sqrt()
        drift_coeff = (mean_scale - 1.0) / dt
        g = beta.clamp(min=1e-8).sqrt()
    elif sde_family == "vp_cosine":
        time_edges = torch.linspace(0.0, 1.0, steps=num_steps + 1, device=device, dtype=dtype)
        alpha_bar = cosine_alpha_bar(time_edges, s=cosine_s).clamp(min=1e-8)
        alpha_ratio = (alpha_bar[1:] / alpha_bar[:-1]).clamp(min=1e-8, max=0.999999)
        mean_scale = alpha_ratio.sqrt()
        step_sigma = (1.0 - alpha_ratio).clamp(min=1e-8).sqrt()
        drift_coeff = (mean_scale - 1.0) / dt
        beta = (-alpha_ratio.log()) / dt
        g = beta.clamp(min=1e-8).sqrt()
    elif sde_family == "ve_geometric":
        if ve_sigma_min <= 0.0 or ve_sigma_max <= 0.0:
            raise ValueError("VE sigma levels must be positive.")
        log_sigma = torch.linspace(math.log(ve_sigma_min), math.log(ve_sigma_max), steps=num_steps + 1, device=device, dtype=dtype)
        sigma_levels = log_sigma.exp()
        step_var = sigma_levels[1:].square() - sigma_levels[:-1].square()
        step_sigma = step_var.clamp(min=1e-8).sqrt()
        drift_coeff = torch.zeros_like(step_sigma)
        beta = torch.zeros_like(step_sigma)
        g = step_sigma / dt.sqrt()
    else:
        raise ValueError(f"Unsupported sde_family: {sde_family}")

    weights = build_step_weights(step_sigma=step_sigma, schedule=weight_schedule)
    return MarkovVPSchedule(
        family=sde_family,
        total_time=float(total_time),
        dt=dt,
        drift_coeff=drift_coeff,
        beta=beta,
        g=g,
        step_sigma=step_sigma,
        weights=weights,
        sigma_levels=sigma_levels,
    )


def cosine_alpha_bar(t: torch.Tensor, *, s: float) -> torch.Tensor:
    scale = math.pi / 2.0
    numerator = torch.cos(((t + s) / (1.0 + s)) * scale).square()
    denominator = math.cos((s / (1.0 + s)) * scale) ** 2
    return numerator / denominator


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
    noise_override: torch.Tensor | None = None,
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
    if noise_override is not None:
        expected_shape = (clean_points.shape[0], schedule.dt.shape[0], clean_points.shape[1])
        if tuple(noise_override.shape) != expected_shape:
            raise ValueError(f"noise_override must have shape {expected_shape}, got {tuple(noise_override.shape)}")
        noise_override = noise_override.to(device=clean_points.device, dtype=clean_points.dtype)

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
        drift = forward_drift(current, drift_coeff=schedule.drift_coeff[step_idx], control=control)
        mean = current + drift * schedule.dt[step_idx]
        noise = noise_override[:, step_idx, :] if noise_override is not None else torch.randn_like(current)
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


def forward_drift(points: torch.Tensor, *, drift_coeff: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
    return drift_coeff * points + control


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


def estimate_markov_wdro_proxy_budget(
    points: torch.Tensor,
    score_net,
    *,
    schedule: MarkovVPSchedule,
    gamma: float,
    step_size: float,
    iters: int,
    clamp_min: torch.Tensor,
    clamp_max: torch.Tensor,
) -> float:
    adv_points = markov_wdro_proxy_attack(
        points=points,
        score_net=score_net,
        schedule=schedule,
        gamma=gamma,
        step_size=step_size,
        iters=iters,
        clamp_min=clamp_min,
        clamp_max=clamp_max,
    )
    delta = adv_points - points
    return float((0.5 * delta.square().sum(dim=1)).mean().item())


def markov_wdro_proxy_attack(
    *,
    points: torch.Tensor,
    score_net,
    schedule: MarkovVPSchedule,
    gamma: float,
    step_size: float,
    iters: int,
    clamp_min: torch.Tensor,
    clamp_max: torch.Tensor,
) -> torch.Tensor:
    clamp_min = clamp_min.to(device=points.device, dtype=points.dtype).view(1, -1)
    clamp_max = clamp_max.to(device=points.device, dtype=points.dtype).view(1, -1)
    fixed_noises = torch.randn(
        points.shape[0],
        schedule.dt.shape[0],
        points.shape[1],
        device=points.device,
        dtype=points.dtype,
    )
    x_adv = points.detach().clone().requires_grad_(True)
    was_training = score_net.training
    score_net.eval()

    for _ in range(iters):
        with torch.enable_grad():
            rollout = rollout_markov_forward(
                clean_points=x_adv,
                control_net=None,
                schedule=schedule,
                zero_control=True,
                noise_override=fixed_noises,
            )
            score_loss = compute_score_matching_loss(score_net=score_net, rollout=rollout, schedule=schedule)
            delta = (x_adv - points).view(points.shape[0], -1)
            transport_cost = 0.5 * delta.square().sum(dim=1).mean()
            objective = score_loss - gamma * transport_cost
        grad = torch.autograd.grad(objective, x_adv)[0]
        x_adv = (x_adv + step_size * grad).detach()
        x_adv = torch.maximum(torch.minimum(x_adv, clamp_max), clamp_min)
        x_adv.requires_grad_(True)

    if was_training:
        score_net.train()
    return x_adv.detach()


def build_score_wdro_dataset(
    *,
    base_points: torch.Tensor,
    score_net,
    schedule: MarkovVPSchedule,
    batch_size: int,
    gamma: float,
    step_size: float,
    iters: int,
    p_adv: float,
    clamp_min: torch.Tensor,
    clamp_max: torch.Tensor,
    device: torch.device,
    rng,
    debug_adv_points: int = 256,
) -> ScoreWdroRefreshResult:
    combined_batches: list[torch.Tensor] = []
    adv_original: list[torch.Tensor] = []
    adv_generated: list[torch.Tensor] = []
    debug_left = max(int(debug_adv_points), 0)
    l2_shifts: list[torch.Tensor] = []
    transport_costs: list[torch.Tensor] = []

    effective_batch_size = base_points.shape[0] if batch_size <= 0 else min(batch_size, base_points.shape[0])
    for start in range(0, base_points.shape[0], effective_batch_size):
        batch = base_points[start : start + effective_batch_size].to(device)
        if p_adv >= 1.0 or rng.random() < p_adv:
            adv_batch = markov_wdro_proxy_attack(
                points=batch,
                score_net=score_net,
                schedule=schedule,
                gamma=gamma,
                step_size=step_size,
                iters=iters,
                clamp_min=clamp_min,
                clamp_max=clamp_max,
            )
            combined_batches.append(batch.cpu())
            combined_batches.append(adv_batch.cpu())
            delta = adv_batch - batch
            l2_shifts.append(delta.norm(dim=1).cpu())
            transport_costs.append((0.5 * delta.square().sum(dim=1)).cpu())

            if debug_left > 0:
                take = min(debug_left, batch.shape[0])
                adv_original.append(batch[:take].cpu())
                adv_generated.append(adv_batch[:take].cpu())
                debug_left -= take
        else:
            combined_batches.append(batch.cpu())

    return ScoreWdroRefreshResult(
        combined_points=torch.cat(combined_batches, dim=0),
        adv_original=torch.cat(adv_original, dim=0) if adv_original else None,
        adv_generated=torch.cat(adv_generated, dim=0) if adv_generated else None,
        mean_l2_shift=float(torch.cat(l2_shifts).mean().item()) if l2_shifts else 0.0,
        max_l2_shift=float(torch.cat(l2_shifts).max().item()) if l2_shifts else 0.0,
        mean_transport_cost=float(torch.cat(transport_costs).mean().item()) if transport_costs else 0.0,
        max_transport_cost=float(torch.cat(transport_costs).max().item()) if transport_costs else 0.0,
    )


def update_terminal_stats(
    *,
    terminal_states: torch.Tensor,
    current: TerminalStats | None,
    momentum: float,
    replay_max_size: int,
    min_var: float = 1e-4,
) -> TerminalStats:
    batch_mean = terminal_states.mean(dim=0).detach()
    batch_var = terminal_states.var(dim=0, unbiased=False).clamp(min=min_var).detach()
    batch_replay = terminal_states.detach().cpu()
    if current is None:
        replay = truncate_terminal_replay(batch_replay, max_size=replay_max_size)
        return TerminalStats(mean=batch_mean, var=batch_var, replay=replay)
    mean = current.mean * momentum + batch_mean * (1.0 - momentum)
    var = current.var * momentum + batch_var * (1.0 - momentum)
    merged_replay = merge_terminal_replay(current.replay, batch_replay, max_size=replay_max_size)
    return TerminalStats(mean=mean, var=var.clamp(min=min_var), replay=merged_replay)


def truncate_terminal_replay(replay: torch.Tensor, *, max_size: int) -> torch.Tensor | None:
    if max_size <= 0:
        return None
    if replay.shape[0] <= max_size:
        return replay
    return replay[-max_size:].clone()


def merge_terminal_replay(
    current_replay: torch.Tensor | None,
    batch_replay: torch.Tensor,
    *,
    max_size: int,
) -> torch.Tensor | None:
    if max_size <= 0:
        return None
    if current_replay is None:
        return truncate_terminal_replay(batch_replay, max_size=max_size)
    merged = torch.cat([current_replay, batch_replay], dim=0)
    return truncate_terminal_replay(merged, max_size=max_size)


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
    solver: str = "euler",
    terminal_sampler: str = "gaussian",
    terminal_jitter_scale: float = 0.0,
    reverse_noise_scale: float = 1.0,
    reverse_control_scale: float = 1.0,
    reverse_tail_noise_scale: float = 1.0,
    reverse_deterministic_tail_steps: int = 0,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    mean = terminal_stats.mean.to(device=device)
    var = terminal_stats.var.to(device=device).clamp(min=1e-6)
    if terminal_sampler == "replay" and terminal_stats.replay is not None and terminal_stats.replay.shape[0] > 0:
        replay = terminal_stats.replay.to(device=device, dtype=mean.dtype)
        indices = torch.randint(0, replay.shape[0], (num_samples,), device=device)
        current = replay[indices]
        if terminal_jitter_scale > 0.0:
            current = current + torch.randn_like(current) * var.sqrt().view(1, data_dim) * float(terminal_jitter_scale)
    else:
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
            control = control * float(reverse_control_scale)
        noise = torch.randn_like(current)
        dt_step = schedule.dt[step_idx]
        score_scale = schedule.step_sigma[step_idx].square() / dt_step
        reverse_drift = -forward_drift(current, drift_coeff=schedule.drift_coeff[step_idx], control=control) + score_scale * score
        stochastic_scale = float(reverse_noise_scale)
        if reverse_deterministic_tail_steps > 0 and step_idx < reverse_deterministic_tail_steps:
            stochastic_scale = float(reverse_tail_noise_scale)
        stochastic = schedule.step_sigma[step_idx] * stochastic_scale * noise
        if solver == "heun":
            proposal = current + reverse_drift * dt_step + stochastic
            score_next = score_net(proposal, sigma_step)
            if zero_control or control_net is None:
                proposal_control = torch.zeros_like(proposal)
            elif control_state is None:
                proposal_control = control_net(proposal, sigma_step) * float(reverse_control_scale)
            else:
                proposal_control = control
            proposal_drift = -forward_drift(
                proposal,
                drift_coeff=schedule.drift_coeff[step_idx],
                control=proposal_control,
            ) + score_scale * score_next
            current = current + 0.5 * (reverse_drift + proposal_drift) * dt_step + stochastic
        elif solver == "euler":
            current = current + reverse_drift * dt_step + stochastic
        else:
            raise ValueError(f"Unsupported reverse solver: {solver}")
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
