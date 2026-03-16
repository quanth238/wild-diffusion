from __future__ import annotations

import torch


class EDMLoss2D:
    def __init__(self, *, p_mean: float = -1.2, p_std: float = 1.2, sigma_data: float = 0.5):
        self.p_mean = float(p_mean)
        self.p_std = float(p_std)
        self.sigma_data = float(sigma_data)

    def sample_sigma(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        rnd_normal = torch.randn(batch_size, 1, device=device, dtype=dtype)
        return (rnd_normal * self.p_std + self.p_mean).exp()

    def __call__(
        self,
        net,
        points: torch.Tensor,
        *,
        sigma: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if sigma is None:
            sigma = self.sample_sigma(batch_size=points.shape[0], device=points.device, dtype=points.dtype)
        if noise is None:
            noise = torch.randn_like(points) * sigma

        sigma_data_sq = self.sigma_data ** 2
        weight = (sigma.square() + sigma_data_sq) / (sigma * self.sigma_data).square()
        denoised = net(points + noise, sigma)
        sq_error = (denoised - points).square().mean(dim=1)
        return weight.squeeze(1) * sq_error
