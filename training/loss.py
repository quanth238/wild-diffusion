import torch
from torch_utils import persistence


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
