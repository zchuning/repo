# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import PixelEncoder


def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability."""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def gaussian_entropy(log_std):
    """Compute entropy of Gaussian distribution."""
    return 0.5 * log_std.size(-1) * (1.0 + np.log(2 * np.pi)) + log_std.sum(-1)


def squash(action, log_prob):
    """Apply squashing function.
    See appendix C from https://arxiv.org/pdf/1812.05905.pdf.
    """
    action = torch.tanh(action)
    if log_prob is not None:
        log_prob -= torch.log(F.relu(1 - action.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return action, log_prob


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class Actor(nn.Module):
    """MLP actor network."""

    def __init__(
        self,
        obs_shape,
        action_shape,
        hidden_dim,
        feature_dim,
        max_log_std=2,
        min_log_std=-10,
    ):
        super().__init__()
        self.encoder = PixelEncoder(obs_shape, feature_dim)
        self.max_log_std = max_log_std
        self.min_log_std = min_log_std
        self.trunk = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_shape[0]),
        )
        self.apply(weight_init)

    def forward(self, obs, deterministic=False, detach_encoder=False):
        obs = self.encoder(obs, detach=detach_encoder)
        mean, log_std = self.trunk(obs).chunk(2, dim=-1)
        log_std = self.min_log_std + 0.5 * (self.max_log_std - self.min_log_std) * (
            torch.tanh(log_std) + 1
        )

        if deterministic:
            action = mean
            log_prob = None
            entropy = None
        else:
            eps = torch.randn_like(mean)
            action = mean + eps * log_std.exp()
            log_prob = gaussian_logprob(eps, log_std)
            entropy = gaussian_entropy(log_std)

        # Apply squashing
        action, log_prob = squash(action, log_prob)
        return action, log_prob, entropy


class QFunction(nn.Module):
    """MLP for q-function."""

    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs, action):
        obs_action = torch.cat([obs, action], dim=1)
        return self.trunk(obs_action)


class Critic(nn.Module):
    """Critic network, employes two q-functions."""

    def __init__(
        self,
        obs_shape,
        action_shape,
        hidden_dim,
        feature_dim,
    ):
        super().__init__()
        self.encoder = PixelEncoder(obs_shape, feature_dim)
        self.Q1 = QFunction(self.encoder.feature_dim, action_shape[0], hidden_dim)
        self.Q2 = QFunction(self.encoder.feature_dim, action_shape[0], hidden_dim)
        self.apply(weight_init)

    def forward(self, obs, action, detach_encoder=False):
        # detach_encoder allows to stop gradient propogation to encoder
        obs = self.encoder(obs, detach=detach_encoder)
        q1 = self.Q1(obs, action)
        q2 = self.Q2(obs, action)
        return q1, q2
