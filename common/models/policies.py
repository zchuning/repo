import numpy as np

import torch
import torch.nn as nn

from .cnns import CNN
from .mlps import GaussianMLP
from .utils import TanhDistribution


class GaussianPolicy(nn.Module):
    def __init__(self, obs_shape, act_shape, hidden_dim):
        super().__init__()
        self.pixel_obs = len(obs_shape) == 3
        if self.pixel_obs:
            self.encoder = CNN(
                input_chn=obs_shape[0],
                output_dim=hidden_dim,
                output_act="ReLU",
            )
            input_dim = hidden_dim
        else:
            input_dim = np.prod(obs_shape)
        hidden_dims = [hidden_dim for _ in range(2)]
        self.head = GaussianMLP(input_dim, hidden_dims, np.prod(act_shape))

    def forward_dist(self, obs):
        if self.pixel_obs:
            obs = self.encoder(obs)
        return self.head.forward_dist(obs)

    def forward(self, obs, deterministic=False):
        # Return action and log prob
        dist = self.forward_dist(obs)
        if deterministic:
            act = dist.mean
            log_prob = None
        else:
            act = dist.rsample()
            log_prob = dist.log_prob(act).sum(-1, True)
        return act, log_prob

    def evaluate(self, obs, act):
        # Return log prob
        dist = self.forward_dist(obs)
        log_prob = dist.log_prob(act).sum(-1, True)
        return log_prob


class TanhGaussianPolicy(GaussianPolicy):
    def __init__(self, obs_shape, act_shape, hidden_dim, act_space=None):
        super().__init__(obs_shape, act_shape, hidden_dim)
        if act_space is None:
            self.loc = torch.tensor(0.0)
            self.scale = torch.tensor(1.0)
        else:
            self.loc = torch.tensor((act_space.high + act_space.low) / 2.0)
            self.scale = torch.tensor((act_space.high - act_space.low) / 2.0)

    def forward_dist(self, obs):
        dist = super().forward_dist(obs)
        return TanhDistribution(dist, self.loc, self.scale)


class EntropyGaussianPolicy(GaussianPolicy):
    def forward(self, obs, deterministic=False):
        # Return action, log prob, and entropy
        dist = self.forward_dist(obs)
        if deterministic:
            act = dist.mean
            log_prob, entropy = None, None
        else:
            act = dist.rsample()
            log_prob = dist.log_prob(act).sum(-1, True)
            entropy = dist.entropy().sum(-1, True)
        return act, log_prob, entropy

    def evaluate(self, obs, act):
        # Return log prob and entropy
        dist = self.forward_dist(obs)
        log_prob = dist.log_prob(act).sum(-1, True)
        entropy = dist.entropy().sum(-1, True)
        return log_prob, entropy
