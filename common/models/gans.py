import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torch.optim import Adam

from .mlps import MLP


class MLPDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dims, lr=1e-4):
        super().__init__()
        self.net = MLP(input_dim, hidden_dims, 1, act="LeakyReLU")
        self.apply(lambda x: nn.utils.spectral_norm(x) if isinstance(x, nn.Linear) else x)
        self.optimizer = Adam(self.parameters(), lr=lr)

    def forward(self, x):
        return self.net(x)

    def train(self, x_real, x_fake, tau=None):
        # Requires gradient for computing gradient penalty
        x_real.requires_grad_()

        # Real data
        d_real = self.forward(x_real)
        if tau is None:
            d_loss_real = self._bce_with_logits(d_real, 1)  # JS div
        else:
            d_loss_real = -(tau * d_real).mean()  # chi-squared div

        # Fake data
        d_fake = self.forward(x_fake)
        if tau is None:
            d_loss_fake = self._bce_with_logits(d_fake, 0)  # JS div
        else:
            d_loss_fake = (d_fake + 0.25 * d_fake**2).mean()  # chi-squared div

        # Update discriminator
        loss = d_loss_real + d_loss_fake
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_info = dict(
            real_loss=d_loss_real.item(),
            fake_loss=d_loss_fake.item(),
        )
        return loss_info

    def _bce_with_logits(self, d_out, target, reduction="mean"):
        targets = d_out.new_full(size=d_out.size(), fill_value=target)
        loss = F.binary_cross_entropy_with_logits(d_out, targets, reduction=reduction)
        return loss


class VDBDiscriminator(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dims,
        latent_dim,
        lr=1e-4,
        init_beta=0.1,
        beta_lr=5e-3,
        target_kl=0.1,
        gp_weight=1.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = MLP(input_dim, hidden_dims, 2 * self.latent_dim, act="LeakyReLU")
        self.act = nn.LeakyReLU()
        self.fc = nn.Linear(self.latent_dim, 1)
        self.optimizer = Adam(self.parameters(), lr=lr)

        self.beta = init_beta
        self.beta_lr = beta_lr
        self.target_kl = target_kl
        self.gp_weight = gp_weight

    def forward(self, x, deterministic=False):
        mean, logstd = self.encoder(x).split(self.latent_dim, -1)
        if deterministic:
            out = self.fc(self.act(mean))
        else:
            eps = torch.randn_like(mean)
            lat = mean + eps * torch.exp(logstd)
            out = self.fc(self.act(lat))
        return out, mean, logstd

    def train(self, x_real, x_fake, tau=None):
        # Requires gradient for computing gradient penalty
        x_real.requires_grad_()

        # Real data
        d_real, mean_real, logstd_real = self.forward(x_real)
        if tau is None:
            d_loss_real = self._bce_with_logits(d_real, 1)  # JS div
        else:
            d_loss_real = -(tau * d_real).mean()  # chi-squared div

        # Fake data
        d_fake, mean_fake, logstd_fake = self.forward(x_fake)
        if tau is None:
            d_loss_fake = self._bce_with_logits(d_fake, 0)  # JS div
        else:
            d_loss_fake = (d_fake + 0.25 * d_fake**2).mean()  # chi-squared div

        # KL divergence to prior
        kl_real = self._compute_kl_prior(mean_real, logstd_real).mean()
        kl_fake = self._compute_kl_prior(mean_fake, logstd_fake).mean()
        kl_prior = 0.5 * (kl_real + kl_fake)
        kl_viol = kl_prior - self.target_kl
        kl_loss = self.beta * kl_viol

        # Apply gradient penalty to real samples
        grad_penalty = self.gp_weight * self._compute_grad_penalty(d_real, x_real)

        # Update discriminator
        loss = d_loss_real + d_loss_fake + kl_loss + grad_penalty
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update discriminator beta
        new_beta = self.beta + self.beta_lr * kl_viol.detach()
        new_beta = new_beta.clamp(min=0)
        self.beta = new_beta

        loss_info = dict(
            real_loss=d_loss_real.item(),
            fake_loss=d_loss_fake.item(),
            kl=kl_prior.item(),
            gp=grad_penalty.item(),
            beta=self.beta.item(),
        )
        return loss_info

    def _bce_with_logits(self, d_out, target, reduction="mean"):
        targets = d_out.new_full(size=d_out.size(), fill_value=target)
        loss = F.binary_cross_entropy_with_logits(d_out, targets, reduction=reduction)
        return loss

    def _compute_grad_penalty(self, d_out, x_in):
        # Zero-centered gradient penalty (Mescheder 2018)
        batch_size = x_in.shape[0]
        grad = autograd.grad(
            outputs=d_out.sum(), inputs=x_in, create_graph=True, retain_graph=True
        )[0].view(batch_size, -1)
        penalty = grad.pow(2).sum(dim=1).mean()
        return penalty

    def _compute_kl_prior(self, mean, logstd):
        dim = mean.shape[-1]
        std = torch.exp(logstd)
        kld = (-logstd + 0.5 * (std**2 + mean**2)).sum(dim=-1) - (0.5 * dim)
        return kld
