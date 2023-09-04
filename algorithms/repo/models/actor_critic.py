import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent, TransformedDistribution

from .utils import TanhBijector, SampleDist


class ValueModel(nn.Module):
    def __init__(
        self, belief_size, state_size, hidden_size, activation_function="relu"
    ):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(belief_size + state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 1)

    def forward(self, belief, state):
        x = torch.cat([belief, state], dim=1)
        hidden = self.act_fn(self.fc1(x))
        hidden = self.act_fn(self.fc2(hidden))
        hidden = self.act_fn(self.fc3(hidden))
        reward = self.fc4(hidden).squeeze(dim=1)
        return reward


class ConditionalValueModel(ValueModel):
    def __init__(
        self,
        belief_size,
        state_size,
        hidden_size,
        condition_size,
        activation_function="relu",
    ):
        super().__init__(
            belief_size,
            state_size + condition_size,
            hidden_size,
            activation_function,
        )

    def forward(self, belief, state, condition):
        pseudo_state = torch.cat((state, condition), dim=1)
        return super().forward(belief, pseudo_state)


class ActorModel(nn.Module):
    def __init__(
        self,
        belief_size,
        state_size,
        hidden_size,
        action_size,
        dist="tanh_normal",
        activation_function="elu",
        min_std=0.1,
        init_std=0.0,
        mean_scale=5,
    ):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(belief_size + state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, 2 * action_size)

        self._dist = dist
        self._min_std = min_std
        self._init_std = init_std
        self._mean_scale = mean_scale

    def forward(self, belief, state):
        x = torch.cat([belief, state], dim=1)
        hidden = self.act_fn(self.fc1(x))
        hidden = self.act_fn(self.fc2(hidden))
        hidden = self.act_fn(self.fc3(hidden))
        hidden = self.act_fn(self.fc4(hidden))
        action = self.fc5(hidden).squeeze(dim=1)

        action_mean, action_std_dev = torch.chunk(action, 2, dim=1)
        action_mean = self._mean_scale * torch.tanh(action_mean / self._mean_scale)
        action_std = F.softplus(action_std_dev + self._init_std) + self._min_std
        return action_mean, action_std

    def get_action_dist(self, belief, state):
        action_mean, action_std = self.forward(belief, state)
        dist = Normal(action_mean, action_std)
        dist = TransformedDistribution(dist, TanhBijector())
        dist = Independent(dist, 1)
        dist = SampleDist(dist)
        return dist

    def get_action(self, belief, state, det=False):
        dist = self.get_action_dist(belief, state)
        if det:
            return dist.mode()
        else:
            return dist.rsample()


class ConditionalActorModel(ActorModel):
    def __init__(
        self,
        belief_size,
        state_size,
        hidden_size,
        action_size,
        condition_size,
        dist="tanh_normal",
        activation_function="elu",
        min_std=0.1,
        init_std=0.0,
        mean_scale=5,
    ):
        super().__init__(
            belief_size,
            state_size + condition_size,
            hidden_size,
            action_size,
            dist,
            activation_function,
            min_std,
            init_std,
            mean_scale,
        )

    def forward(self, belief, state, condition):
        pseudo_state = torch.cat((state, condition), dim=1)
        return super().forward(belief, pseudo_state)

    def get_action_dist(self, belief, state, condition):
        action_mean, action_std = self.forward(belief, state, condition)
        dist = Normal(action_mean, action_std)
        dist = TransformedDistribution(dist, TanhBijector())
        dist = Independent(dist, 1)
        dist = SampleDist(dist)
        return dist

    def get_action(self, belief, state, condition, det=False):
        dist = self.get_action_dist(belief, state, condition)
        if det:
            return dist.mode()
        else:
            return dist.rsample()
