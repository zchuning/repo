import numpy as np
import os

import torch
import torch.nn.functional as F
from torch.optim import Adam

from .bisim import Bisim
from .models.dynamics import StateActionRewardModel


class DeepMDP(Bisim):
    def build_models(self):
        super().build_models()

        # Replace reward model
        act_shape = self.env.action_space.shape
        self.reward_model = StateActionRewardModel(
            self.c.feature_size,
            act_shape,
        ).to(self.device)
        self.decoder_optim = Adam(
            list(self.transition_model.parameters())
            + list(self.reward_model.parameters()),
            lr=self.c.decoder_lr,
            weight_decay=self.c.decoder_wd,
        )

    def compute_dynamics_reward_loss(self, obs, act, rew, next_obs):
        feat = self.critic.encoder(obs)
        next_feat = self.critic.encoder(next_obs)
        pred_next_mean, pred_next_std = self.transition_model(torch.cat([feat, act], 1))

        # Dynamics loss
        diff = (pred_next_mean - next_feat.detach()) / pred_next_std
        dyn_loss = (0.5 * diff.pow(2) + pred_next_std.log()).mean()

        # Reward loss
        pred_next_rew = self.reward_model(torch.cat([feat, act], 1))
        rew_loss = F.mse_loss(pred_next_rew, rew)
        return dyn_loss, rew_loss

    def update_encoder_decoder(self, obs, act, rew, next_obs):
        dyn_loss, rew_loss = self.compute_dynamics_reward_loss(obs, act, rew, next_obs)

        # Update models
        self.encoder_optim.zero_grad()
        self.decoder_optim.zero_grad()
        total_loss = dyn_loss + rew_loss
        total_loss.backward()
        self.encoder_optim.step()
        self.decoder_optim.step()

        # Logging
        self.logger.record("train/dyn_loss", dyn_loss.item())
        self.logger.record("train/rew_loss", rew_loss.item())
