import torch
import torch.nn as nn
from torch.distributions import TransformedDistribution
from torch.distributions.transforms import TanhTransform, AffineTransform


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


class MomentumScaler(object):
    """
    Adapted from https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/vec_env/vec_normalize.py
    """

    def __init__(self, init_mean=0.0, init_var=1.0, epsilon=1e-8):
        self.epsilon = epsilon
        self.mean = init_mean
        self.var = init_var
        self.count = 1

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        new_var = m_2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def fit(self, data, batch_dim=0):
        data = data.detach()
        batch_mean = torch.mean(data, dim=batch_dim, keepdim=True)
        batch_var = torch.var(data, dim=batch_dim, keepdim=True)
        batch_count = data.shape[batch_dim]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def transform(self, data):
        return (data - self.mean) / torch.sqrt(self.var + self.epsilon)

    def inverse_transform(self, data):
        return data * torch.sqrt(self.var + self.epsilon) + self.mean


class TanhDistribution(TransformedDistribution):
    def __init__(self, base_dist, loc=0.0, scale=1.0):
        transforms = [
            TanhTransform(cache_size=1),
            AffineTransform(loc=loc, scale=scale, cache_size=1),
        ]
        super().__init__(base_dist, transforms)

    @property
    def mean(self):
        mean = self.base_dist.mean
        for transform in self.transforms:
            mean = transform(mean)
        return mean

    def entropy(self):
        return self.base_dist.entropy()


def map_activation(act):
    return "leaky_relu" if act == "LeakyReLU" else act.lower()


def init_weights(m, act):
    if isinstance(m, nn.Linear) or isinstance(m, EnsembleLinearLayer):
        nn.init.orthogonal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        gain = nn.init.calculate_gain(map_activation(act))
        nn.init.orthogonal_(m.weight, gain)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
