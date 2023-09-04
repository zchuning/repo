import numpy as np

import torch
import torch.nn as nn

from .cnns import CNN
from .mlps import MLP


class ValueNetwork(nn.Module):
    def __init__(self, obs_shape, hidden_dim):
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
        self.head = MLP(input_dim, hidden_dims, 1)

    def forward(self, obs):
        if self.pixel_obs:
            obs = self.encoder(obs)
        return self.head(obs)


class QNetwork(nn.Module):
    def __init__(self, obs_shape, act_shape, hidden_dim):
        super().__init__()
        self.pixel_obs = len(obs_shape) == 3
        if self.pixel_obs:
            self.encoder1 = CNN(input_chn=obs_shape[0], output_dim=hidden_dim)
            self.encoder2 = CNN(input_chn=obs_shape[0], output_dim=hidden_dim)
            input_dim = hidden_dim + np.prod(act_shape)
        else:
            input_dim = np.prod(obs_shape) + np.prod(act_shape)
        hidden_dims = [hidden_dim for _ in range(2)]
        self.q1 = MLP(input_dim, hidden_dims, 1)
        self.q2 = MLP(input_dim, hidden_dims, 1)

    def forward(self, obs, act):
        if self.pixel_obs:
            q1 = self.q1(torch.cat((self.encoder1(obs), act), -1))
            q2 = self.q2(torch.cat((self.encoder2(obs), act), -1))
        else:
            x = torch.cat((obs, act), -1)
            q1 = self.q1(x)
            q2 = self.q2(x)
        return q1, q2
