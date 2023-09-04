import numpy as np
import os

import torch
import torch.nn.functional as F
from torch.optim import Adam

from common.buffers import ReplayBuffer
from common.logger import Video
from common.utils import get_device, soft_update, to_torch, to_np

from .models.actor_critic import Actor, Critic
from .models.dynamics import make_transition_model, StateRewardModel


def preprocess(obs):
    # Preprocess a batch of observations
    obs = obs.astype(np.float32) / 255.0
    return obs


class Bisim:
    def __init__(self, config, env, eval_env, logger):
        assert config.pixel_obs, "Bisim requires pixel observations"
        self.c = config
        self.env = env
        self.eval_env = eval_env
        self.logger = logger
        self.device = get_device()

        self.build_models()
        self.buffer = ReplayBuffer(
            self.c.replay_size,
            self.env.observation_space.shape,
            self.env.action_space.shape,
            obs_type=np.uint8,
        )
        self.step = 0

    def build_models(self):
        obs_shape = self.env.observation_space.shape
        act_shape = self.env.action_space.shape

        # Policy
        self.actor = Actor(
            obs_shape,
            act_shape,
            self.c.hidden_size,
            self.c.feature_size,
        ).to(self.device)
        self.actor_optim = Adam(self.actor.parameters(), lr=self.c.actor_lr)

        # Critic
        self.critic = Critic(
            obs_shape,
            act_shape,
            self.c.hidden_size,
            self.c.feature_size,
        ).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.c.critic_lr)
        self.critic_target = Critic(
            obs_shape,
            act_shape,
            self.c.hidden_size,
            self.c.feature_size,
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Tie encoders between actor and critic
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)
        self.encoder_optim = torch.optim.Adam(
            self.critic.encoder.parameters(), lr=self.c.encoder_lr
        )

        # Automatic entropy tuning
        self.target_entropy = -np.prod(act_shape)
        self.log_alpha = torch.tensor(np.log(self.c.init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optim = torch.optim.Adam(
            [self.log_alpha], lr=self.c.alpha_lr, betas=(self.c.alpha_beta, 0.999)
        )

        # Transition and reward models
        self.transition_model = make_transition_model(
            self.c.transition_model_type,
            self.c.feature_size,
            act_shape,
        ).to(self.device)
        self.reward_model = StateRewardModel(self.c.feature_size).to(self.device)
        self.decoder_optim = Adam(
            list(self.transition_model.parameters())
            + list(self.reward_model.parameters()),
            lr=self.c.decoder_lr,
            weight_decay=self.c.decoder_wd,
        )

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs, evaluate=False):
        obs = to_torch(preprocess(obs[None]))
        action, _, _ = self.actor(obs, deterministic=evaluate)
        action = to_np(action)[0]
        return action

    def update_critic(self, obs, act, rew, next_obs, done):
        # Compute Q target
        with torch.no_grad():
            next_act, next_logp, _ = self.actor(next_obs)
            next_q1_target, next_q2_target = self.critic_target(next_obs, next_act)
            min_next_q_target = torch.min(next_q1_target, next_q2_target)
            q_target = rew + (1 - done) * self.c.gamma * (
                min_next_q_target - self.alpha * next_logp
            )

        # Compute Q loss
        q1, q2 = self.critic(obs, act, detach_encoder=False)
        q1_loss = F.mse_loss(q1, q_target)
        q2_loss = F.mse_loss(q2, q_target)
        q_loss = q1_loss + q2_loss

        # Update critic
        self.critic_optim.zero_grad()
        q_loss.backward()
        self.critic_optim.step()

        # Logging
        self.logger.record("train/critic_1_loss", q1_loss.item())
        self.logger.record("train/critic_2_loss", q2_loss.item())

    def update_actor(self, obs):
        # Compute actor loss
        new_act, new_logp, entropy = self.actor(obs, detach_encoder=True)
        new_q1, new_q2 = self.critic(obs, new_act, detach_encoder=True)
        min_new_q = torch.min(new_q1, new_q2)
        actor_loss = (self.alpha.detach() * new_logp - min_new_q).mean()

        # Update actor
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # Update alpha with dual descent
        alpha_loss = -(self.alpha * (new_logp + self.target_entropy).detach()).mean()
        self.log_alpha_optim.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optim.step()

        # Logging
        self.logger.record("train/actor_loss", actor_loss.item())
        self.logger.record("train/alpha_loss", alpha_loss.item())
        self.logger.record("train/alpha", self.alpha.item())
        self.logger.record("train/entropy", entropy.mean().item())

    def compute_bisim_loss(self, obs, act, rew):
        # Encode observation
        feat = self.critic.encoder(obs)
        with torch.no_grad():
            pred_next_mean, pred_next_std = self.transition_model(
                torch.cat([feat, act], 1)
            )

        # Sample random states across episodes
        perm = torch.randperm(len(obs))
        feat2 = feat[perm]
        rew2 = rew[perm]
        if pred_next_mean.ndim == 2:
            pred_next_mean2 = pred_next_mean[perm]
            pred_next_std2 = pred_next_std[perm]
        elif pred_next_mean.ndim == 3:
            pred_next_mean2 = pred_next_mean[:, perm]
            pred_next_std2 = pred_next_std[:, perm]
        else:
            raise NotImplementedError

        # Compute latent distances
        z_dist = F.smooth_l1_loss(feat, feat2, reduction="none")
        r_dist = F.smooth_l1_loss(rew, rew2, reduction="none")
        if self.c.transition_model_type == "deterministic":
            p_dist = F.smooth_l1_loss(pred_next_mean, pred_next_mean2, reduction="none")
        else:
            p_dist = torch.sqrt(
                (pred_next_mean - pred_next_mean2).pow(2)
                + (pred_next_std - pred_next_std2).pow(2)
            )

        # Bisim loss
        bisimilarity = r_dist + self.c.gamma * p_dist
        bisim_loss = (z_dist - bisimilarity).pow(2).mean()
        return bisim_loss

    def compute_dynamics_reward_loss(self, obs, act, rew, next_obs):
        feat = self.critic.encoder(obs)
        next_feat = self.critic.encoder(next_obs)
        pred_next_mean, pred_next_std = self.transition_model(torch.cat([feat, act], 1))

        # Dynamics loss
        diff = (pred_next_mean - next_feat.detach()) / pred_next_std
        dyn_loss = (0.5 * diff.pow(2) + pred_next_std.log()).mean()

        # Reward loss
        pred_next_feat = self.transition_model.sample_prediction(
            torch.cat([feat, act], 1)
        )
        pred_next_rew = self.reward_model(pred_next_feat)
        rew_loss = F.mse_loss(pred_next_rew, rew)
        return dyn_loss, rew_loss

    def update_encoder_decoder(self, obs, act, rew, next_obs):
        bisim_loss = self.compute_bisim_loss(obs, act, rew)
        dyn_loss, rew_loss = self.compute_dynamics_reward_loss(obs, act, rew, next_obs)

        # Update models
        self.encoder_optim.zero_grad()
        self.decoder_optim.zero_grad()
        total_loss = self.c.bisim_coef * bisim_loss + dyn_loss + rew_loss
        total_loss.backward()
        self.encoder_optim.step()
        self.decoder_optim.step()

        # Logging
        self.logger.record("train/bisim_loss", bisim_loss.item())
        self.logger.record("train/dyn_loss", dyn_loss.item())
        self.logger.record("train/rew_loss", rew_loss.item())

    def update_parameters(self):
        # Sample batch from replay buffer
        obs, act, rew, next_obs, done = self.buffer.sample(self.c.batch_size)
        obs = to_torch(preprocess(obs))
        next_obs = to_torch(preprocess(next_obs))
        act, rew, done = map(to_torch, [act, rew, done])

        # Train critic
        self.update_critic(obs, act, rew, next_obs, done)

        # Train encoder and dynamics model
        self.update_encoder_decoder(obs, act, rew, next_obs)

        # Train actor
        if self.step % self.c.actor_update_freq == 0:
            self.update_actor(obs)

        # Update critic target
        if self.step % self.c.critic_target_update_freq == 0:
            soft_update(self.critic_target.Q1, self.critic.Q1, self.c.critic_tau)
            soft_update(self.critic_target.Q2, self.critic.Q2, self.c.critic_tau)
            soft_update(
                self.critic_target.encoder, self.critic.encoder, self.c.encoder_tau
            )

    def train(self):
        obs = self.env.reset()
        episode_reward = 0
        episode_success = 0
        while self.step < self.c.num_steps:
            # Collect environment step
            if self.step < self.c.init_steps:
                action = self.env.action_space.sample()
            else:
                action = self.select_action(obs)
            next_obs, reward, done, info = self.env.step(action)

            real_done = 0 if info.get("TimeLimit.truncated", False) else float(done)
            self.buffer.push(obs, action, reward, next_obs, real_done)
            obs = next_obs
            episode_reward += reward
            episode_success += info.get("success", 0)

            if done:
                self.logger.record("train/return", episode_reward)
                self.logger.record("train/success", episode_success)
                obs = self.env.reset()
                episode_reward = 0
                episode_success = 0

            # Train agent
            if self.step >= self.c.init_steps:
                train_steps = self.c.init_steps if self.step == self.c.init_steps else 1
                for _ in range(train_steps):
                    self.update_parameters()

            # Evaluate agent
            if self.step % self.c.eval_every == 0:
                self.evaluate()

            # Save checkpoint
            if self.step % self.c.save_every == 0:
                self.save_checkpoint()

            # Logging
            if self.step % self.c.log_every == 0:
                self.logger.record("train/step", self.step)
                self.logger.dump(step=self.step)

            self.step += 1

    def evaluate(self):
        obs = self.eval_env.reset()
        done = False
        episode_reward = 0
        episode_success = 0
        frames = []
        while not done:
            with torch.no_grad():
                action = self.select_action(obs, evaluate=True)
            next_obs, reward, done, info = self.eval_env.step(action)
            episode_reward += reward
            episode_success += info.get("success", 0)
            frames.append(obs[-3:, :, :])
            obs = next_obs
        self.logger.record("test/return", episode_reward)
        self.logger.record("test/success", float(episode_success > 0))

        # Record video
        video = Video(np.stack(frames)[None], fps=30)
        self.logger.record("test/video", video, exclude="stdout")

    def save_checkpoint(self):
        ckpt_path = os.path.join(self.logger.dir, f"models.pt")
        ckpt = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }
        torch.save(ckpt, ckpt_path)

    def load_checkpoint(self):
        ckpt_path = os.path.join(self.logger.dir, f"models.pt")
        ckpt = torch.load(ckpt_path)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
