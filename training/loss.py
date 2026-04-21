import torch
from torch_utils import persistence
from torch_utils import training_stats

from training.cdro_utils import (
    build_cdro_ladder,
    control_transport_cost,
    pathwise_l2,
    reduce_per_sample,
    rollout_path_heuristic_attack,
    weighted_edm_loss_per_pixel,
)


@persistence.persistent_class
class EDMLossWdro:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
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
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        base_noisy = y + n


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
        cdro_n_steps_path=32,
        cdro_step_size=0.02,
        cdro_total_budget_rho=32.0,
        cdro_time_horizon=1.0,
        cdro_sigma_min=0.002,
        cdro_sigma_max=80.0,
        cdro_edm_ladder_mode="stochastic_stratified_quantile",
        cdro_per_example_sigma_ladders=True,
        attack_num_steps=1,
        outer_attack_weight=0.3,
        outer_clean_weight=0.0,
    ):
        if int(cdro_n_steps_path) <= 0:
            raise ValueError(f"cdro_n_steps_path must be > 0, got {cdro_n_steps_path}")
        if float(cdro_step_size) <= 0.0:
            raise ValueError(f"cdro_step_size must be > 0, got {cdro_step_size}")
        if float(cdro_time_horizon) <= 0.0:
            raise ValueError(f"cdro_time_horizon must be > 0, got {cdro_time_horizon}")
        if float(cdro_sigma_min) <= 0.0 or float(cdro_sigma_max) <= float(cdro_sigma_min):
            raise ValueError(
                f"Expected 0 < cdro_sigma_min < cdro_sigma_max, got {cdro_sigma_min}, {cdro_sigma_max}"
            )
        self.P_mean = float(P_mean)
        self.P_std = float(P_std)
        self.sigma_data = float(sigma_data)
        self.cdro_n_steps_path = int(cdro_n_steps_path)
        self.cdro_step_size = float(cdro_step_size)
        self.cdro_total_budget_rho = float(cdro_total_budget_rho)
        self.cdro_time_horizon = float(cdro_time_horizon)
        self.cdro_sigma_min = float(cdro_sigma_min)
        self.cdro_sigma_max = float(cdro_sigma_max)
        self.cdro_edm_ladder_mode = str(cdro_edm_ladder_mode)
        self.cdro_per_example_sigma_ladders = bool(cdro_per_example_sigma_ladders)
        self.attack_num_steps = int(attack_num_steps)
        self.outer_attack_weight = float(outer_attack_weight)
        self.outer_clean_weight = float(outer_clean_weight)

    @staticmethod
    def _path_n_steps(sigma_levels: torch.Tensor) -> int:
        return int(sigma_levels.shape[-1] - 1)

    @staticmethod
    def _sigma_batch_for_step(sigma_levels: torch.Tensor, *, step_idx: int, batch_size: int, device: torch.device) -> torch.Tensor:
        if sigma_levels.ndim == 1:
            return torch.full(
                (batch_size,),
                float(sigma_levels[step_idx + 1].item()),
                device=device,
                dtype=torch.float32,
            )
        if sigma_levels.ndim == 2:
            return sigma_levels[:, step_idx + 1].to(device=device, dtype=torch.float32)
        raise ValueError(f"Expected rank-1 or rank-2 sigma ladder, got shape={tuple(sigma_levels.shape)}")

    @staticmethod
    def _delta_ratio(delta_l2: torch.Tensor, radius_by_step: torch.Tensor) -> torch.Tensor:
        if radius_by_step.ndim == 1:
            return delta_l2 / radius_by_step.view(1, -1).clamp_min(1e-8)
        if radius_by_step.ndim == 2:
            return delta_l2 / radius_by_step.clamp_min(1e-8)
        raise ValueError(f"Expected rank-1 or rank-2 path radii, got shape={tuple(radius_by_step.shape)}")

    def _prepare_batch(self, images, augment_pipe=None):
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        sigma_levels, transition_deltas, radius_by_step = build_cdro_ladder(
            n_steps_path=self.cdro_n_steps_path,
            sigma_min=self.cdro_sigma_min,
            sigma_max=self.cdro_sigma_max,
            p_mean=self.P_mean,
            p_std=self.P_std,
            total_budget_rho=self.cdro_total_budget_rho,
            time_horizon=self.cdro_time_horizon,
            ladder_mode=self.cdro_edm_ladder_mode,
            batch_size=int(images.shape[0]),
            per_example_sigma_ladders=self.cdro_per_example_sigma_ladders,
            device=y.device,
            dtype=y.dtype,
        )
        return y, augment_labels, sigma_levels, transition_deltas, radius_by_step

    def _build_rollout(self, *, net, y, labels, augment_labels, sigma_levels, transition_deltas, radius_by_step):
        base_net = net.module if hasattr(net, "module") else net
        grad_params = [param for param in base_net.parameters() if param.requires_grad]
        for param in grad_params:
            param.requires_grad_(False)
        try:
            rollout = rollout_path_heuristic_attack(
                attack_net=base_net,
                x0=y,
                sigma_levels=sigma_levels,
                transition_deltas=transition_deltas,
                radius_by_step=radius_by_step,
                inner_steps=self.attack_num_steps if self.outer_attack_weight > 0.0 else 0,
                step_size=self.cdro_step_size,
                sigma_data=self.sigma_data,
                labels=labels,
                augment_labels=augment_labels,
            )
        finally:
            for param in grad_params:
                param.requires_grad_(True)
        return rollout

    def _resolve_outer_weights(self):
        raw_attack_weight = max(self.outer_attack_weight, 0.0)
        raw_clean_weight = max(self.outer_clean_weight, 0.0)
        mixture_mass = raw_attack_weight + raw_clean_weight
        if mixture_mass <= 0.0:
            raise RuntimeError("CDRO outer_attack_weight and outer_clean_weight cannot both be zero.")
        return raw_attack_weight / mixture_mass, raw_clean_weight / mixture_mass

    def _report_stats(
        self,
        *,
        device,
        lambda_ctrl: float,
        lambda_ref: float,
        outer_mean_per_pixel: float,
        outer_attack_scalar: float,
        outer_clean_scalar: float,
        control_cost: torch.Tensor,
        delta_l2: torch.Tensor,
        delta_ratio: torch.Tensor,
    ) -> None:
        attack_enabled = bool(
            lambda_ctrl > 0.0 and self.attack_num_steps > 0 and self.cdro_total_budget_rho > 0.0
        )
        # The image-port rollout does not spend a standalone forward-only denoiser call per path step.
        n_fwd = 0.0
        n_fwd_inputgrad = float(self.cdro_n_steps_path * max(self.attack_num_steps, 0)) if attack_enabled else 0.0
        n_fwd_parambackward = float(self.cdro_n_steps_path * (int(lambda_ctrl > 0.0) + int(lambda_ref > 0.0)))
        batch_equiv_evals = float(
            self.cdro_n_steps_path
            * (
                (max(self.attack_num_steps, 0) if attack_enabled else 0)
                + int(lambda_ctrl > 0.0)
                + int(lambda_ref > 0.0)
            )
        )
        training_stats.report("CDRO/outer_loss", torch.as_tensor(outer_mean_per_pixel, device=device))
        training_stats.report("CDRO/outer_loss_attack", torch.as_tensor(outer_attack_scalar, device=device))
        training_stats.report("CDRO/outer_loss_clean", torch.as_tensor(outer_clean_scalar, device=device))
        training_stats.report("CDRO/control_cost", control_cost)
        training_stats.report("CDRO/delta_norm_mean", delta_l2.mean())
        training_stats.report("CDRO/delta_norm_max", delta_l2.max())
        training_stats.report("CDRO/delta_ratio_mean", delta_ratio.mean())
        training_stats.report("CDRO/delta_ratio_max", delta_ratio.max())
        training_stats.report("CDRO/lambda_attack", torch.as_tensor(lambda_ctrl, device=device))
        training_stats.report("CDRO/lambda_clean", torch.as_tensor(lambda_ref, device=device))
        training_stats.report("CDRO/path_steps", torch.as_tensor(float(self.cdro_n_steps_path), device=device))
        training_stats.report("CDRO/attack_num_steps", torch.as_tensor(float(self.attack_num_steps), device=device))
        training_stats.report(
            "CDRO/per_example_sigma_ladders",
            torch.as_tensor(float(self.cdro_per_example_sigma_ladders), device=device),
        )
        training_stats.report("CDRO/rho", torch.as_tensor(float(self.cdro_total_budget_rho), device=device))
        training_stats.report("CDRO/n_fwd_step", torch.as_tensor(n_fwd, device=device))
        training_stats.report("CDRO/n_fwd_inputgrad_step", torch.as_tensor(n_fwd_inputgrad, device=device))
        training_stats.report("CDRO/n_fwd_parambackward_step", torch.as_tensor(n_fwd_parambackward, device=device))
        training_stats.report("CDRO/batch_equiv_denoiser_evals_step", torch.as_tensor(batch_equiv_evals, device=device))

    def probe_clean_loss(self, net, images, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        pred = net(y + n, sigma, labels, augment_labels=augment_labels)
        return weight * ((pred - y) ** 2)

    def __call__(self, net, images, labels=None, augment_pipe=None):
        y, augment_labels, sigma_levels, transition_deltas, radius_by_step = self._prepare_batch(
            images,
            augment_pipe=augment_pipe,
        )
        rollout = self._build_rollout(
            net=net,
            y=y,
            labels=labels,
            augment_labels=augment_labels,
            sigma_levels=sigma_levels,
            transition_deltas=transition_deltas,
            radius_by_step=radius_by_step,
        )
        lambda_ctrl, lambda_ref = self._resolve_outer_weights()
        n_steps = self._path_n_steps(sigma_levels)
        attack_total = None
        clean_total = None
        for step_idx in range(n_steps):
            sigma_batch = self._sigma_batch_for_step(
                sigma_levels,
                step_idx=step_idx,
                batch_size=int(y.shape[0]),
                device=y.device,
            )
            attack_step = weighted_edm_loss_per_pixel(
                net=net,
                x_noisy=rollout.states_ctrl[:, step_idx + 1],
                x_clean=y,
                sigma=sigma_batch,
                sigma_data=self.sigma_data,
                labels=labels,
                augment_labels=augment_labels,
            )
            attack_total = attack_step if attack_total is None else (attack_total + attack_step)
            if lambda_ref > 0.0:
                clean_step = weighted_edm_loss_per_pixel(
                    net=net,
                    x_noisy=rollout.states_ref[:, step_idx + 1],
                    x_clean=y,
                    sigma=sigma_batch,
                    sigma_data=self.sigma_data,
                    labels=labels,
                    augment_labels=augment_labels,
                )
                clean_total = clean_step if clean_total is None else (clean_total + clean_step)
        attack_loss = (
            torch.zeros_like(y)
            if attack_total is None
            else attack_total / float(n_steps)
        )
        clean_loss = (
            torch.zeros_like(attack_loss)
            if clean_total is None
            else clean_total / float(n_steps)
        )
        loss = lambda_ctrl * attack_loss + lambda_ref * clean_loss
        attack_scalar = float(reduce_per_sample(attack_loss.detach()).mean().item())
        clean_scalar = 0.0 if clean_total is None else float(reduce_per_sample(clean_loss.detach()).mean().item())
        outer_scalar = float(loss.detach().mean().item())
        delta_l2 = pathwise_l2(rollout.delta_path)
        delta_ratio = self._delta_ratio(delta_l2, rollout.radius_by_step)
        control_cost = control_transport_cost(rollout.control_path, rollout.transition_deltas).mean()
        self._report_stats(
            device=y.device,
            lambda_ctrl=lambda_ctrl,
            lambda_ref=lambda_ref,
            outer_mean_per_pixel=outer_scalar,
            outer_attack_scalar=attack_scalar,
            outer_clean_scalar=clean_scalar,
            control_cost=control_cost,
            delta_l2=delta_l2,
            delta_ratio=delta_ratio,
        )
        return loss

    def accumulate_gradients(self, net, images, labels=None, augment_pipe=None, gain=1.0):
        y, augment_labels, sigma_levels, transition_deltas, radius_by_step = self._prepare_batch(
            images,
            augment_pipe=augment_pipe,
        )
        rollout = self._build_rollout(
            net=net,
            y=y,
            labels=labels,
            augment_labels=augment_labels,
            sigma_levels=sigma_levels,
            transition_deltas=transition_deltas,
            radius_by_step=radius_by_step,
        )
        lambda_ctrl, lambda_ref = self._resolve_outer_weights()
        n_steps = self._path_n_steps(sigma_levels)
        outer_mean_per_pixel = 0.0
        outer_attack_scalar = 0.0
        outer_clean_scalar = 0.0

        for step_idx in range(n_steps):
            sigma_batch = self._sigma_batch_for_step(
                sigma_levels,
                step_idx=step_idx,
                batch_size=int(y.shape[0]),
                device=y.device,
            )
            if lambda_ctrl > 0.0:
                attack_step = weighted_edm_loss_per_pixel(
                    net=net,
                    x_noisy=rollout.states_ctrl[:, step_idx + 1],
                    x_clean=y,
                    sigma=sigma_batch,
                    sigma_data=self.sigma_data,
                    labels=labels,
                    augment_labels=augment_labels,
                )
                attack_chunk = attack_step.sum() * (float(gain) * float(lambda_ctrl) / float(n_steps))
                attack_chunk.backward()
                outer_mean_per_pixel += float(lambda_ctrl) * float(attack_step.detach().mean().item()) / float(n_steps)
                outer_attack_scalar += float(reduce_per_sample(attack_step.detach()).mean().item()) / float(n_steps)
            if lambda_ref > 0.0:
                clean_step = weighted_edm_loss_per_pixel(
                    net=net,
                    x_noisy=rollout.states_ref[:, step_idx + 1],
                    x_clean=y,
                    sigma=sigma_batch,
                    sigma_data=self.sigma_data,
                    labels=labels,
                    augment_labels=augment_labels,
                )
                clean_chunk = clean_step.sum() * (float(gain) * float(lambda_ref) / float(n_steps))
                clean_chunk.backward()
                outer_mean_per_pixel += float(lambda_ref) * float(clean_step.detach().mean().item()) / float(n_steps)
                outer_clean_scalar += float(reduce_per_sample(clean_step.detach()).mean().item()) / float(n_steps)

        delta_l2 = pathwise_l2(rollout.delta_path)
        delta_ratio = self._delta_ratio(delta_l2, rollout.radius_by_step)
        control_cost = control_transport_cost(rollout.control_path, rollout.transition_deltas).mean()
        self._report_stats(
            device=y.device,
            lambda_ctrl=lambda_ctrl,
            lambda_ref=lambda_ref,
            outer_mean_per_pixel=outer_mean_per_pixel,
            outer_attack_scalar=outer_attack_scalar,
            outer_clean_scalar=outer_clean_scalar,
            control_cost=control_cost,
            delta_l2=delta_l2,
            delta_ratio=delta_ratio,
        )
        return torch.as_tensor(outer_mean_per_pixel, device=y.device)
