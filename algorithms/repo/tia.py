import numpy as np

import torch
import torch.nn as nn

from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.optim import Adam

from common.logger import Video
from common.utils import to_torch, to_np, preprocess, postprocess, FreezeParameters
from .dreamer import Dreamer
from .models.decoder import TIAObservationModel, ObservationModel, RewardModel
from .models.rssm import TransitionModel
from .models.utils import bottle


class TIA(Dreamer):
    def build_models(self, config, env):
        super().build_models(config, env)
        if config.pixel_obs:
            obs_size = env.observation_space.shape
        else:
            obs_size = np.prod(env.observation_space.shape).item()
        action_size = np.prod(env.action_space.shape).item()

        # Replace original observation model
        self.obs_model = TIAObservationModel(
            config.belief_size,
            config.state_size,
            config.embedding_size,
            config.cnn_activation_function,
        ).to(self.device)

        # Distractor transition model
        self.distractor_transition_model = TransitionModel(
            config.belief_size,
            config.state_size,
            action_size,
            config.hidden_size,
            config.embedding_size,
            config.dense_activation_function,
        ).to(self.device)

        # Distractor observation model
        self.distractor_obs_model = TIAObservationModel(
            config.belief_size,
            config.state_size,
            config.embedding_size,
            config.cnn_activation_function,
        ).to(self.device)

        # Separate distractor observation model to avoid degeneracy
        self.distractor_only_obs_model = ObservationModel(
            not config.pixel_obs,
            obs_size,
            config.belief_size,
            config.state_size,
            config.embedding_size,
            config.cnn_activation_function,
        ).to(self.device)

        # Distractor reward model
        self.distractor_reward_model = RewardModel(
            config.belief_size,
            config.state_size,
            config.hidden_size,
            config.dense_activation_function,
        ).to(self.device)

        # Mask head for generating observations
        self.mask_head = nn.Sequential(nn.Conv2d(6, 1, 1), nn.Sigmoid()).to(self.device)

        self.model_params = (
            list(self.encoder.parameters())
            + list(self.transition_model.parameters())
            + list(self.reward_model.parameters())
            + list(self.obs_model.parameters())
            + list(self.distractor_transition_model.parameters())
            + list(self.distractor_reward_model.parameters())
            + list(self.distractor_obs_model.parameters())
            + list(self.distractor_only_obs_model.parameters())
            + list(self.mask_head.parameters())
        )
        self.model_optimizer = Adam(self.model_params, lr=config.model_lr)

    def train_dynamics(self, obs, actions, rewards, nonterms):
        # Forward through both task and distractor models
        init_belief = torch.zeros(self.c.batch_size, self.c.belief_size).to(self.device)
        init_state = torch.zeros(self.c.batch_size, self.c.state_size).to(self.device)
        embeds = bottle(self.encoder, (obs,))
        (
            t_beliefs,
            t_prior_states,
            t_prior_means,
            t_prior_std_devs,
            t_post_states,
            t_post_means,
            t_post_std_devs,
        ) = self.transition_model.observe(
            init_belief,
            init_state,
            actions[:-1],
            embeds[1:],
            nonterms[:-1],
        )
        (
            d_beliefs,
            d_prior_states,
            d_prior_means,
            d_prior_std_devs,
            d_post_states,
            d_post_means,
            d_post_std_devs,
        ) = self.distractor_transition_model.observe(
            init_belief,
            init_state,
            actions[:-1],
            embeds[1:],
            nonterms[:-1],
        )

        # Reconstruction loss
        t_recon, t_mask = bottle(self.obs_model, (t_beliefs, t_post_states))
        d_recon, d_mask = bottle(self.distractor_obs_model, (d_beliefs, d_post_states))
        recon_mask = bottle(self.mask_head, (torch.cat((t_mask, d_mask), dim=2),))
        recon = t_recon * recon_mask + d_recon * (1 - recon_mask)
        obs_loss = (
            -Normal(recon, 1)
            .log_prob(obs[1:])
            .sum((2, 3, 4) if self.c.pixel_obs else 2)
            .mean((0, 1))
        )

        # Distractor reconstruction
        d_only_recon = bottle(
            self.distractor_only_obs_model, (d_beliefs, d_post_states)
        )
        d_obs_loss = (
            -Normal(d_only_recon, 1)
            .log_prob(obs[1:])
            .sum((2, 3, 4) if self.c.pixel_obs else 2)
            .mean((0, 1))
        )

        # Reward loss
        # Since we predict rewards from next states, we need to shift reward
        # by one and account for terminal states
        rewards_tgt = rewards[:-1].squeeze(-1)
        mask = nonterms[:-1].squeeze(-1)
        t_reward = bottle(self.reward_model, (t_beliefs, t_post_states))
        # Freeze distractor reward model to only train the latents
        with FreezeParameters(list(self.distractor_reward_model.parameters())):
            d_reward = bottle(self.distractor_reward_model, (d_beliefs, d_post_states))
        t_reward_loss = (-Normal(t_reward, 1).log_prob(rewards_tgt) * mask).mean((0, 1))
        d_reward_loss = (Normal(d_reward, 1).log_prob(rewards_tgt) * mask).mean((0, 1))
        reward_loss = t_reward_loss + self.c.tia_adv_coef * d_reward_loss

        # KL loss
        t_kl_div = kl_divergence(
            Normal(t_post_means, t_post_std_devs),
            Normal(t_prior_means, t_prior_std_devs),
        ).sum(2)
        d_kl_div = kl_divergence(
            Normal(d_post_means, d_post_std_devs),
            Normal(d_prior_means, d_prior_std_devs),
        ).sum(2)
        t_kl_loss = torch.max(t_kl_div, self.free_nats).mean((0, 1))
        d_kl_loss = torch.max(d_kl_div, self.free_nats).mean((0, 1))
        kl_loss = t_kl_loss + d_kl_loss

        # Update model
        model_loss = obs_loss + self.c.tia_obs_coef * d_obs_loss + reward_loss + kl_loss
        self.model_optimizer.zero_grad()
        model_loss.backward()
        nn.utils.clip_grad_norm_(self.model_params, self.c.grad_clip_norm)
        self.model_optimizer.step()

        # Update distractor reward model
        for _ in range(self.c.tia_reward_train_steps):
            d_reward = bottle(
                self.distractor_reward_model,
                (d_beliefs.detach(), d_post_states.detach()),
            )
            d_reward_loss = -(Normal(d_reward, 1).log_prob(rewards_tgt) * mask).mean(
                (0, 1)
            )
            self.model_optimizer.zero_grad()
            d_reward_loss.backward()
            nn.utils.clip_grad_norm_(self.model_params, self.c.grad_clip_norm)
            self.model_optimizer.step()

        # Logging
        self.logger.record("train/obs_loss", obs_loss.item())
        self.logger.record("train/d_obs_loss", d_obs_loss.item())
        self.logger.record("train/reward_loss", reward_loss.item())
        self.logger.record("train/t_reward_loss", t_reward_loss.item())
        self.logger.record("train/d_reward_loss", d_reward_loss.item())
        self.logger.record("train/kl_loss", kl_loss.item())
        self.logger.record("train/t_kl_div", t_kl_div.mean().item())
        self.logger.record("train/d_kl_div", d_kl_div.mean().item())
        self.logger.record("train/model_loss", model_loss.item())
        return t_beliefs.detach(), t_post_states.detach()

    def eval_agent(self):
        self.toggle_train(False)
        belief, posterior_state, action_tensor = self.init_latent_and_action()
        obs = self.eval_env.reset()
        done = False
        episode_reward = 0
        episode_success = 0
        frames = []
        with torch.no_grad():
            while not done:
                obs_tensor = to_torch(preprocess(obs[None]))
                (
                    belief,
                    posterior_state,
                    action_tensor,
                ) = self.update_latent_and_select_action(
                    belief, posterior_state, action_tensor, obs_tensor, False
                )
                action = to_np(action_tensor)[0]
                next_obs, reward, done, info = self.eval_env.step(action)
                if self.c.pixel_obs:
                    obs_hat = to_np(self.obs_model(belief, posterior_state)[0])
                    obs_hat = postprocess(obs_hat)[0]
                    frames.append([obs, obs_hat])
                obs = next_obs
                episode_reward += reward
                episode_success += info.get("success", 0)
        self.logger.record("test/return", episode_reward)
        self.logger.record("test/success", float(episode_success > 0))
        if self.c.pixel_obs:
            # video shape: (T, N, C, H, W) -> (N, T, C, H, W)
            video = Video(np.stack(frames).transpose(1, 0, 2, 3, 4), fps=30)
            self.logger.record("test/video", video, exclude="stdout")
        self.toggle_train(True)
