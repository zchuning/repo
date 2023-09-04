# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeterministicTransitionModel(nn.Module):
    def __init__(self, feature_dim, action_shape, hidden_dim):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.fc_mean = nn.Linear(hidden_dim, feature_dim)
        print("Deterministic transition model chosen.")

    def forward(self, x):
        x = self.trunk(x)
        mean = self.fc_mean(x)
        std = torch.ones_like(mean)
        return mean, std

    def sample_prediction(self, x):
        mean, std = self(x)
        return mean


class ProbabilisticTransitionModel(nn.Module):
    def __init__(
        self,
        feature_dim,
        action_shape,
        hidden_dim,
        max_std=1e1,
        min_std=1e-4,
    ):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.fc_mean = nn.Linear(hidden_dim, feature_dim)
        self.fc_std = nn.Linear(hidden_dim, feature_dim)
        self.max_std = max_std
        self.min_std = min_std
        print("Probabilistic transition model chosen.")

    def forward(self, x):
        x = self.trunk(x)
        mean = self.fc_mean(x)
        std = torch.sigmoid(self.fc_std(x))
        std = self.min_std + (self.max_std - self.min_std) * std
        return mean, std

    def sample_prediction(self, x):
        mean, std = self(x)
        eps = torch.randn_like(std)
        return mean + std * eps


class EnsembleOfProbabilisticTransitionModels(nn.Module):
    def __init__(self, feature_dim, action_shape, hidden_dim, ensemble_size=5):
        super().__init__()
        self.models = [
            ProbabilisticTransitionModel(feature_dim, action_shape, hidden_dim)
            for _ in range(ensemble_size)
        ]
        print("Ensemble of probabilistic transition models chosen.")

    def forward(self, x):
        mean_std_list = [model(x) for model in self.models]
        means, stds = zip(*mean_std_list)
        means, stds = torch.stack(means), torch.stack(stds)
        return means, stds

    def sample_prediction(self, x):
        model = random.choice(self.models)
        return model.sample_prediction(x)


_AVAILABLE_TRANSITION_MODELS = {
    "deterministic": DeterministicTransitionModel,
    "probabilistic": ProbabilisticTransitionModel,
    "ensemble": EnsembleOfProbabilisticTransitionModels,
}


def make_transition_model(
    transition_model_type, feature_dim, action_shape, hidden_dim=512
):
    assert transition_model_type in _AVAILABLE_TRANSITION_MODELS
    return _AVAILABLE_TRANSITION_MODELS[transition_model_type](
        feature_dim, action_shape, hidden_dim
    )


class StateRewardModel(nn.Module):
    def __init__(self, feature_dim, hidden_dim=512):
        super().__init__()
        self.chunk = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.chunk(x)


class StateActionRewardModel(nn.Module):
    def __init__(self, feature_dim, action_shape, hidden_dim=512):
        super().__init__()
        self.chunk = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.chunk(x)
