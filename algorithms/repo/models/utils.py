import numpy as np
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


def bottle(f, xs):
    # Wraps the input for a function to process a (time, batch, feature) sequence in (time * batch, feature)
    horizon, batch_size = xs[0].shape[:2]
    ys = f(*(x.reshape(horizon * batch_size, *x.shape[2:]) for x in xs))
    if isinstance(ys, tuple):
        return tuple(y.reshape(horizon, batch_size, *y.shape[1:]) for y in ys)
    else:
        return ys.reshape(horizon, batch_size, *ys.shape[1:])


class EnsembleLinearLayer(nn.Module):
    """
    Efficient linear layer for ensemble models.
    Adapted from https://github.com/facebookresearch/mbrl-lib/blob/main/mbrl/models/util.py
    """

    def __init__(self, in_dim, out_dim, ensemble_size, bias=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.ensemble_size = ensemble_size
        self.weight = nn.Parameter(
            torch.rand(self.ensemble_size, self.in_dim, self.out_dim)
        )
        if bias:
            self.bias = nn.Parameter(torch.rand(self.ensemble_size, 1, self.out_dim))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        xw = torch.matmul(x, self.weight)
        if self.bias is not None:
            return xw + self.bias
        else:
            return xw

    def extra_repr(self):
        return (
            f"in_dim={self.in_dim}, out_dim={self.out_dim}, "
            f"ensemble_size={self.ensemble_size}, bias={self.bias is not None}"
        )


class EnsembleDynamicsModel(nn.Module):
    def __init__(
        self,
        belief_size,
        state_size,
        action_size,
        hidden_size,
        ensemble_size,
        activation_function="relu",
        min_std_dev=0.1,
    ):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.min_std_dev = min_std_dev
        self.fc1 = EnsembleLinearLayer(
            belief_size + state_size + action_size,
            hidden_size,
            ensemble_size,
        )
        self.fc2 = EnsembleLinearLayer(hidden_size, hidden_size, ensemble_size)
        self.fc3 = EnsembleLinearLayer(hidden_size, hidden_size, ensemble_size)
        self.fc4 = EnsembleLinearLayer(hidden_size, belief_size, ensemble_size)

    def forward(self, belief, state, action):
        x = torch.cat((belief, state, action), dim=1)
        hidden = self.act_fn(self.fc1(x))
        hidden = self.act_fn(self.fc2(hidden))
        hidden = self.act_fn(self.fc3(hidden))
        belief = self.fc4(hidden)
        return belief


class InverseDynamicsModel(nn.Module):
    def __init__(
        self,
        belief_size,
        state_size,
        action_size,
        hidden_size,
        activation_function="relu",
        min_std_dev=0.1,
    ):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.min_std_dev = min_std_dev
        self.fc1 = nn.Linear(belief_size + state_size + belief_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 2 * action_size)

    def forward(self, belief, state, next_belief):
        x = torch.cat((belief, state, next_belief), dim=1)
        hidden = self.act_fn(self.fc1(x))
        hidden = self.act_fn(self.fc2(hidden))
        hidden = self.act_fn(self.fc3(hidden))
        mean, std_dev = torch.chunk(self.fc4(hidden), chunks=2, dim=1)
        std_dev = F.softplus(std_dev) + self.min_std_dev
        return mean, std_dev


class TanhBijector(torch.distributions.Transform):
    def __init__(self):
        super().__init__()
        self.bijective = True
        self.domain = torch.distributions.constraints.real
        self.codomain = torch.distributions.constraints.interval(-1.0, 1.0)

    @property
    def sign(self):
        return 1.0

    def _call(self, x):
        return torch.tanh(x)

    def _inverse(self, y: torch.Tensor):
        y = torch.where(
            (torch.abs(y) <= 1.0), torch.clamp(y, -0.99999997, 0.99999997), y
        )
        y = torch.atanh(y)
        return y

    def log_abs_det_jacobian(self, x, y):
        return 2.0 * (np.log(2) - x - F.softplus(-2.0 * x))


class SampleDist:
    def __init__(self, dist, samples=100):
        self._dist = dist
        self._samples = samples

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def mean(self):
        samples = self._dist.rsample((self._samples,))
        return samples.mean(0)

    def mode(self):
        samples = self._dist.rsample((self._samples,))
        log_probs = self._dist.log_prob(samples)
        batch_size, feature_size = samples.shape[1:]
        indices = (
            torch.argmax(log_probs, 0)
            .reshape(1, batch_size, 1)
            .repeat(1, 1, feature_size)
        )
        return torch.gather(samples, 0, indices).squeeze(0)

    def entropy(self):
        samples = self._dist.rsample((self._samples,))
        log_probs = self._dist.log_prob(samples)
        return -log_probs.mean(0)

    def sample(self):
        return self._dist.sample()
