from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .utils import EnsembleLinearLayer, init_weights


class MLP(nn.Module):
    def __init__(
        self, input_dim, hidden_dims, output_dim, act="ReLU", output_act="Identity"
    ):
        super().__init__()
        act_fn = getattr(nn, act)
        output_act_fn = getattr(nn, output_act)

        layers = []
        curr_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, dim))
            layers.append(act_fn())
            curr_dim = dim
        layers.append(nn.Linear(curr_dim, output_dim))
        layers.append(output_act_fn())

        self.layers = nn.Sequential(*layers)
        self.apply(partial(init_weights, act=act))

    def forward(self, x):
        return self.layers(x)


class GaussianMLP(MLP):
    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim,
        act="ReLU",
        max_log_std=2.0,
        min_log_std=-20.0,
    ):
        super().__init__(input_dim, hidden_dims, 2 * output_dim, act=act)
        self.output_dim = output_dim
        self.max_log_std = max_log_std
        self.min_log_std = min_log_std

    def forward(self, x):
        out = super().forward(x)
        mean, log_std = out.split(int(self.output_dim), -1)
        log_std = self.max_log_std - F.softplus(self.max_log_std - log_std)
        log_std = self.min_log_std + F.softplus(log_std - self.min_log_std)
        return mean, log_std

    def forward_dist(self, x):
        mean, log_std = self.forward(x)
        dist = Normal(mean, torch.exp(log_std))
        return dist

    def sample(self, x, deterministic=False):
        if deterministic:
            mean, _ = self.forward(x)
            return mean
        else:
            dist = self.forward_dist(x)
            return dist.rsample()


class EnsembleMLP(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim,
        ensemble_size,
        act="ReLU",
        output_act="Identity",
    ):
        super().__init__()
        act_fn = getattr(nn, act)
        output_act_fn = getattr(nn, output_act)

        layers = []
        curr_dim = input_dim
        for dim in hidden_dims:
            layers.append(EnsembleLinearLayer(curr_dim, dim, ensemble_size))
            layers.append(act_fn())
            curr_dim = dim
        layers.append(EnsembleLinearLayer(curr_dim, output_dim, ensemble_size))
        layers.append(output_act_fn())

        self.layers = nn.Sequential(*layers)
        self.apply(partial(init_weights, act=act))

    def forward(self, x):
        return self.layers(x)


class EnsembleGaussianMLP(EnsembleMLP):
    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim,
        ensemble_size,
        act="ReLU",
        max_log_std=2.0,
        min_log_std=-20.0,
    ):
        super().__init__(
            input_dim,
            hidden_dims,
            2 * output_dim,
            ensemble_size,
            act=act,
            output_act="Identity",
        )
        self.output_dim = output_dim
        self.max_log_std = max_log_std
        self.min_log_std = min_log_std

    def forward(self, x):
        out = self.layers(x)
        mean, log_std = out.split(int(self.output_dim), -1)
        log_std = self.max_log_std - F.softplus(self.max_log_std - log_std)
        log_std = self.min_log_std + F.softplus(log_std - self.min_log_std)
        return mean, log_std

    def forward_dist(self, x):
        mean, log_std = self.forward(x)
        dist = Normal(mean, torch.exp(log_std))
        return dist

    def sample(self, x, deterministic=False):
        if deterministic:
            mean, _ = self.forward(x)
            return mean
        else:
            dist = self.forward_dist(x)
            return dist.rsample()
