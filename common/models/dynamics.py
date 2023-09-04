import numpy as np

import torch
import torch.nn as nn

from .mlps import GaussianMLP, EnsembleGaussianMLP
from .utils import MomentumScaler
from common.utils import to_torch, to_np


class TransitionModel(nn.Module):
    def __init__(self, obs_shape, act_shape, hidden_dims):
        super().__init__()
        assert len(obs_shape) == 1
        obs_dim = np.prod(obs_shape)
        act_dim = np.prod(act_shape)
        self.model = GaussianMLP(obs_dim + act_dim, hidden_dims, obs_dim)

    def forward(self, obs, act):
        x = torch.cat((obs, act), -1)
        mean, log_std = self.model(x)
        return mean, log_std

    def compute_loss(self, obs, act, next_obs):
        x = torch.cat((obs, act), -1)
        dist = self.model.forward_dist(x)
        loss = -dist.log_prob(next_obs).mean()
        return loss


class EnsembleTransitionRewardModel(nn.Module):
    def __init__(
        self,
        obs_shape,
        act_shape,
        hidden_dims,
        ensemble_size,
        normalize=False,
    ):
        super().__init__()
        assert len(obs_shape) == 1
        obs_dim = np.prod(obs_shape)
        act_dim = np.prod(act_shape)
        self.model = EnsembleGaussianMLP(
            obs_dim + act_dim, hidden_dims, obs_dim + 1, ensemble_size, act="SiLU"
        )
        self.scaler = MomentumScaler() if normalize else None

    def forward(self, obs, act, deterministic=False, reduction="rand"):
        x = torch.cat((obs, act), -1).unsqueeze(0)
        if self.scaler:
            x = self.scaler.transform(x)
        ens_preds = self.model.sample(x, deterministic)

        # Reduce ensemble predictions
        if reduction == "mean":
            # Average over ensemble
            preds = torch.mean(ens_preds, 0)
        elif reduction == "rand":
            # Choose a model uniformly at random
            ens_size, batch_size = ens_preds.shape[:2]
            model_inds = torch.randint(0, ens_size, size=(batch_size,))
            batch_inds = torch.arange(0, batch_size)
            preds = ens_preds[model_inds, batch_inds]
        else:
            raise ValueError("Unsupported reduction")

        rew, next_obs = preds[:, :1], preds[:, 1:]
        next_obs += obs
        return next_obs, rew

    def predict(self, obs, act, deterministic=False, reduction="rand"):
        obs, act = map(to_torch, [obs, act])
        next_obs, rew = self.forward(obs, act, deterministic, reduction)
        next_obs, rew = map(to_np, [next_obs, rew])
        return next_obs, rew

    def compute_loss(self, obs, act, rew, next_obs):
        x = torch.cat((obs, act), -1).unsqueeze(0)
        if self.scaler:
            # Update scaler statistics
            self.scaler.fit(x, batch_dim=1)
            x = self.scaler.transform(x)
        # Train model to predict residue
        delta = next_obs - obs
        y = torch.cat((rew, delta), -1).unsqueeze(0).repeat(self.ensemble_size, 1, 1)
        dist = self.model.forward_dist(x)
        # Average loss over batch and output dimension, sum over ensemble dimension
        loss = -dist.log_prob(y).mean((1, 2)).sum()
        return loss
