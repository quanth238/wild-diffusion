import torch
from torch_utils import persistence, training_stats


def _sample_edm_corruption(images, *, P_mean, P_std, sigma_data, augment_pipe):
    rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
    sigma = (rnd_normal * P_std + P_mean).exp()
    weight = (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2
    y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
    noise = torch.randn_like(y) * sigma
    return sigma, weight, y, augment_labels, noise


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = float(scale)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.scale * grad_output, None


def _unwrap_module(net):
    return net.module if hasattr(net, "module") else net


@persistence.persistent_class
class EDMLossWdro:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, labels=None, augment_pipe=None):
        sigma, weight, y, augment_labels, noise = _sample_edm_corruption(
            images,
            P_mean=self.P_mean,
            P_std=self.P_std,
            sigma_data=self.sigma_data,
            augment_pipe=augment_pipe,
        )
        D_yn = net(y + noise, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss


#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
@persistence.persistent_class
class EDMLossAdv:
    def __init__(self,
                 P_mean=-1.2,
                 P_std=1.2,
                 sigma_data=0.5,
                 adv_steps=2,
                 adv_step_size=0.1,
                 adv_eps=None,
                 adv_mix=0.5,
                 ):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.adv_steps = adv_steps
        self.adv_step_size = adv_step_size
        self.adv_eps = adv_eps
        self.adv_mix = adv_mix

    def __call__(self, net, images, labels=None, augment_pipe=None):
        sigma, weight, y, augment_labels, noise = _sample_edm_corruption(
            images,
            P_mean=self.P_mean,
            P_std=self.P_std,
            sigma_data=self.sigma_data,
            augment_pipe=augment_pipe,
        )
        base_noisy = y + noise


        batch_size = images.shape[0]
        delta = torch.zeros_like(base_noisy, requires_grad=True)

        for _ in range(self.adv_steps):
            delta.requires_grad_(True)
            noisy_adv = base_noisy + delta     # y + n + δ
            D_yn = net(noisy_adv, sigma, labels, augment_labels=augment_labels)

            inner_loss = weight * ((D_yn - y) ** 2)
            inner_loss = inner_loss.mean()
            grad_delta, = torch.autograd.grad(
                inner_loss, delta, only_inputs=True
            )

            grad_view = grad_delta.view(batch_size, -1)
            grad_norm = grad_view.norm(p=2, dim=1).view(batch_size, 1, 1, 1)
            grad_norm = grad_norm + 1e-12
            step = self.adv_step_size * grad_delta / grad_norm
            delta = delta + step

            if self.adv_eps is not None:
                delta = delta.clamp(-self.adv_eps, self.adv_eps)

            delta = delta.detach()

        D_clean = net(base_noisy, sigma, labels, augment_labels=augment_labels)
        loss_clean = weight * ((D_clean - y) ** 2)

        noisy_adv = base_noisy + delta
        D_adv = net(noisy_adv, sigma, labels, augment_labels=augment_labels)
        loss_adv = weight * ((D_adv - y) ** 2)

        lam = self.adv_mix
        loss = (1.0 - lam) * loss_clean + lam * loss_adv

        return loss


@persistence.persistent_class
class EDMLossCDRO:
    def __init__(
        self,
        P_mean=-1.2,
        P_std=1.2,
        sigma_data=0.5,
        robust_mix=0.3,
        adv_steps=2,
        adv_step_size=0.02,
        max_delta=0.05,
        rho_target=1e-4,
        lambda_init=0.1,
        lambda_lr=1e-3,
        start_kimg=0.0,
        ramp_kimg=0.0,
        sigma_floor=0.0,
        sigma_cut=0.5,
        gate_power=2.0,
        delta_space="image",
    ):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.robust_mix = robust_mix
        self.adv_steps = adv_steps
        self.adv_step_size = adv_step_size
        self.max_delta = max_delta
        self.rho_target = rho_target
        self.lambda_dual = float(lambda_init)
        self.lambda_lr = lambda_lr
        self.start_kimg = float(start_kimg)
        self.ramp_kimg = float(ramp_kimg)
        self.current_kimg = 0.0
        self.sigma_floor = sigma_floor
        self.sigma_cut = sigma_cut
        self.gate_power = gate_power
        if delta_space not in {"image", "noise"}:
            raise ValueError(f"Unsupported delta_space={delta_space!r}")
        self.delta_space = delta_space

    def state_dict(self):
        return {"lambda_dual": float(self.lambda_dual)}

    def load_state_dict(self, state_dict):
        if state_dict is None:
            return
        self.lambda_dual = float(state_dict.get("lambda_dual", self.lambda_dual))

    def set_training_progress(self, *, cur_nimg=None, cur_kimg=None, total_kimg=None):
        if cur_kimg is not None:
            self.current_kimg = float(cur_kimg)
        elif cur_nimg is not None:
            self.current_kimg = float(cur_nimg) / 1000.0

    def activation_scale(self):
        start_kimg = max(float(getattr(self, "start_kimg", 0.0)), 0.0)
        ramp_kimg = max(float(getattr(self, "ramp_kimg", 0.0)), 0.0)
        current_kimg = float(getattr(self, "current_kimg", 0.0))
        if current_kimg < start_kimg:
            return 0.0
        if ramp_kimg <= 0.0:
            return 1.0
        return min(max((current_kimg - start_kimg) / ramp_kimg, 0.0), 1.0)

    def sigma_gate(self, sigma):
        sigma_cut = max(self.sigma_cut, 1e-8)
        gate = (1.0 - sigma / sigma_cut).clamp(min=0.0, max=1.0)
        sigma_floor = max(float(getattr(self, "sigma_floor", 0.0)), 0.0)
        if sigma_floor > 0.0:
            if sigma_floor >= sigma_cut:
                raise ValueError(f"sigma_floor ({sigma_floor}) must be smaller than sigma_cut ({sigma_cut})")
            ramp = ((sigma - sigma_floor) / (sigma_cut - sigma_floor)).clamp(min=0.0, max=1.0)
            gate = gate * ramp
        gate = gate * self.activation_scale()
        return gate ** self.gate_power

    def delta_scale(self, sigma, gate):
        if getattr(self, "delta_space", "image") == "noise":
            return gate * sigma
        return gate

    def applied_delta(self, delta, sigma, gate):
        return self.delta_scale(sigma, gate) * delta

    def transport_cost(self, delta, sigma, gate):
        applied = self.applied_delta(delta, sigma, gate)
        if getattr(self, "delta_space", "image") == "noise":
            return applied.square().mean(dim=[1, 2, 3])
        denom = sigma.square() + self.sigma_data ** 2
        return (applied.square() / denom).mean(dim=[1, 2, 3])

    @staticmethod
    def project_l2(delta, radius):
        flat = delta.view(delta.shape[0], -1)
        radius_flat = radius.view(delta.shape[0], 1)
        rms = (flat.square().mean(dim=1, keepdim=True) + 1e-12).sqrt()
        scale = (radius_flat / rms).clamp(max=1.0)
        return (flat * scale).view_as(delta)

    def delta_radius(self, sigma, gate):
        if getattr(self, "delta_space", "image") == "noise":
            scale = self.delta_scale(sigma, gate).clamp(min=1e-8)
            return self.max_delta / scale
        return torch.full_like(gate, self.max_delta)

    @staticmethod
    def _distributed_mean(value):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            value = value.clone()
            torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
            value = value / torch.distributed.get_world_size()
        return value

    def build_batch(
        self,
        net,
        images,
        labels=None,
        augment_pipe=None,
        *,
        sigma=None,
        noise=None,
    ):
        if sigma is None or noise is None:
            sigma_s, weight, y, augment_labels, noise_s = _sample_edm_corruption(
                images,
                P_mean=self.P_mean,
                P_std=self.P_std,
                sigma_data=self.sigma_data,
                augment_pipe=augment_pipe,
            )
            sigma = sigma_s if sigma is None else sigma
            noise = noise_s if noise is None else noise
        else:
            y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
            weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

        base_noisy = y + noise
        gate = self.sigma_gate(sigma)
        radius = self.delta_radius(sigma, gate)
        delta = torch.zeros_like(base_noisy)

        for _ in range(self.adv_steps):
            delta.requires_grad_(True)
            perturbed = base_noisy + self.applied_delta(delta, sigma, gate)
            pred = net(perturbed, sigma, labels, augment_labels=augment_labels)
            robust_loss = (weight * (pred - y) ** 2).mean()
            transport = self.transport_cost(delta, sigma, gate).mean()
            inner_objective = robust_loss - self.lambda_dual * transport
            grad_delta, = torch.autograd.grad(inner_objective, delta, only_inputs=True)

            grad_view = grad_delta.view(grad_delta.shape[0], -1)
            grad_norm = grad_view.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
            step = self.adv_step_size * (grad_view / grad_norm).view_as(delta)
            delta = (delta + step).detach()
            delta = self.project_l2(delta, radius).detach()

        applied_delta = self.applied_delta(delta, sigma, gate)
        transport_per = self.transport_cost(delta.detach(), sigma.detach(), gate.detach())
        pred_clean = net(base_noisy, sigma, labels, augment_labels=augment_labels)
        pred_robust = net(base_noisy + applied_delta, sigma, labels, augment_labels=augment_labels)
        return {
            "sigma": sigma,
            "weight": weight,
            "clean": y,
            "augment_labels": augment_labels,
            "noise": noise,
            "base_noisy": base_noisy,
            "gate": gate,
            "delta": delta,
            "applied_delta": applied_delta,
            "transport_per": transport_per,
            "pred_clean": pred_clean,
            "pred_robust": pred_robust,
        }

    def __call__(self, net, images, labels=None, augment_pipe=None):
        batch = self.build_batch(net, images, labels, augment_pipe=augment_pipe)
        sigma = batch["sigma"]
        weight = batch["weight"]
        y = batch["clean"]
        gate = batch["gate"]
        delta = batch["delta"]
        applied_delta = batch["applied_delta"]
        pred_clean = batch["pred_clean"]
        pred_robust = batch["pred_robust"]
        loss_clean = weight * ((pred_clean - y) ** 2)
        loss_robust = weight * ((pred_robust - y) ** 2)
        loss = (1.0 - self.robust_mix) * loss_clean + self.robust_mix * loss_robust

        mean_transport = batch["transport_per"].mean()
        mean_transport = self._distributed_mean(mean_transport)
        self.lambda_dual = max(
            0.0,
            self.lambda_dual + self.lambda_lr * (mean_transport.item() - self.rho_target),
        )

        training_stats.report("CDRO/lambda_dual", self.lambda_dual)
        training_stats.report("CDRO/mean_transport_cost", mean_transport)
        training_stats.report("CDRO/activation_scale", self.activation_scale())
        training_stats.report("CDRO/gate_mean", gate.mean())
        training_stats.report("CDRO/raw_delta_rms", delta.detach().square().mean(dim=[1, 2, 3]).sqrt().mean())
        training_stats.report("CDRO/applied_delta_rms", applied_delta.detach().square().mean(dim=[1, 2, 3]).sqrt().mean())

        return loss


@persistence.persistent_class
class EDMLossCDROMarkov:
    def __init__(
        self,
        P_mean=-1.2,
        P_std=1.2,
        sigma_data=0.5,
        robust_mix=0.05,
        max_delta=0.03,
        rho_target=2e-5,
        lambda_init=1e-3,
        lambda_lr=5e-4,
        start_kimg=0.0,
        ramp_kimg=0.0,
        sigma_floor=0.12,
        sigma_cut=0.70,
        gate_power=1.0,
        delta_space="noise",
    ):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.robust_mix = robust_mix
        self.max_delta = max_delta
        self.rho_target = rho_target
        self.lambda_dual = float(lambda_init)
        self.lambda_lr = lambda_lr
        self.start_kimg = float(start_kimg)
        self.ramp_kimg = float(ramp_kimg)
        self.current_kimg = 0.0
        self.sigma_floor = sigma_floor
        self.sigma_cut = sigma_cut
        self.gate_power = gate_power
        if delta_space not in {"image", "noise"}:
            raise ValueError(f"Unsupported delta_space={delta_space!r}")
        self.delta_space = delta_space

    def state_dict(self):
        return {"lambda_dual": float(self.lambda_dual)}

    def load_state_dict(self, state_dict):
        if state_dict is None:
            return
        self.lambda_dual = float(state_dict.get("lambda_dual", self.lambda_dual))

    def set_training_progress(self, *, cur_nimg=None, cur_kimg=None, total_kimg=None):
        if cur_kimg is not None:
            self.current_kimg = float(cur_kimg)
        elif cur_nimg is not None:
            self.current_kimg = float(cur_nimg) / 1000.0

    def activation_scale(self):
        start_kimg = max(float(getattr(self, "start_kimg", 0.0)), 0.0)
        ramp_kimg = max(float(getattr(self, "ramp_kimg", 0.0)), 0.0)
        current_kimg = float(getattr(self, "current_kimg", 0.0))
        if current_kimg < start_kimg:
            return 0.0
        if ramp_kimg <= 0.0:
            return 1.0
        return min(max((current_kimg - start_kimg) / ramp_kimg, 0.0), 1.0)

    def sigma_gate(self, sigma):
        sigma_cut = max(self.sigma_cut, 1e-8)
        gate = (1.0 - sigma / sigma_cut).clamp(min=0.0, max=1.0)
        sigma_floor = max(float(getattr(self, "sigma_floor", 0.0)), 0.0)
        if sigma_floor > 0.0:
            if sigma_floor >= sigma_cut:
                raise ValueError(f"sigma_floor ({sigma_floor}) must be smaller than sigma_cut ({sigma_cut})")
            ramp = ((sigma - sigma_floor) / (sigma_cut - sigma_floor)).clamp(min=0.0, max=1.0)
            gate = gate * ramp
        gate = gate * self.activation_scale()
        return gate ** self.gate_power

    def delta_scale(self, sigma, gate):
        if getattr(self, "delta_space", "image") == "noise":
            return gate * sigma
        return gate

    def project_l2(self, delta, radius):
        flat = delta.view(delta.shape[0], -1)
        radius_flat = radius.view(delta.shape[0], 1)
        rms = (flat.square().mean(dim=1, keepdim=True) + 1e-12).sqrt()
        scale = (radius_flat / rms).clamp(max=1.0)
        return (flat * scale).view_as(delta)

    def radius(self, sigma, gate):
        if getattr(self, "delta_space", "image") == "noise":
            scale = self.delta_scale(sigma, gate).clamp(min=1e-8)
            return self.max_delta / scale
        return torch.full_like(gate, self.max_delta)

    def applied_delta(self, delta, sigma, gate):
        return self.delta_scale(sigma, gate) * delta

    def transport_cost(self, delta, sigma, gate):
        applied = self.applied_delta(delta, sigma, gate)
        if getattr(self, "delta_space", "image") == "noise":
            return applied.square().mean(dim=[1, 2, 3])
        denom = sigma.square() + self.sigma_data ** 2
        return (applied.square() / denom).mean(dim=[1, 2, 3])

    @staticmethod
    def _distributed_mean(value):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            value = value.clone()
            torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
            value = value / torch.distributed.get_world_size()
        return value

    def __call__(self, net, images, labels=None, augment_pipe=None):
        sigma, weight, y, augment_labels, noise = _sample_edm_corruption(
            images,
            P_mean=self.P_mean,
            P_std=self.P_std,
            sigma_data=self.sigma_data,
            augment_pipe=augment_pipe,
        )
        base_noisy = y + noise
        gate = self.sigma_gate(sigma)

        net_module = _unwrap_module(net)
        control_raw = net_module.control(base_noisy, sigma, labels, augment_labels=augment_labels)
        control = self.project_l2(control_raw, self.radius(sigma, gate)).to(base_noisy.dtype)
        applied_delta = self.applied_delta(control, sigma, gate)
        controlled_noisy = base_noisy + _GradReverse.apply(applied_delta, 1.0)

        pred_clean = net(base_noisy, sigma, labels, augment_labels=augment_labels)
        pred_robust = net(controlled_noisy, sigma, labels, augment_labels=augment_labels)

        loss_clean = weight * ((pred_clean - y) ** 2)
        loss_robust = weight * ((pred_robust - y) ** 2)
        transport_per = self.transport_cost(control.detach(), sigma.detach(), gate.detach())
        transport_loss = self.transport_cost(control, sigma, gate).view(-1, 1, 1, 1)

        loss = (1.0 - self.robust_mix) * loss_clean + self.robust_mix * loss_robust + self.lambda_dual * transport_loss

        mean_transport = self._distributed_mean(transport_per.mean())
        self.lambda_dual = max(
            0.0,
            self.lambda_dual + self.lambda_lr * (mean_transport.item() - self.rho_target),
        )

        training_stats.report("CDROMarkov/lambda_dual", self.lambda_dual)
        training_stats.report("CDROMarkov/mean_transport_cost", mean_transport)
        training_stats.report("CDROMarkov/activation_scale", self.activation_scale())
        training_stats.report("CDROMarkov/gate_mean", gate.mean())
        training_stats.report("CDROMarkov/control_rms", control.detach().square().mean(dim=[1, 2, 3]).sqrt().mean())
        training_stats.report("CDROMarkov/applied_delta_rms", applied_delta.detach().square().mean(dim=[1, 2, 3]).sqrt().mean())

        return loss
