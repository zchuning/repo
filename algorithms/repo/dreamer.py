import glob
import numpy as np
import os
import torch
import torch.nn as nn

from torch.distributions import Normal, Independent
from torch.distributions.kl import kl_divergence
from torch.optim import Adam

from common.buffers import SequenceReplayBuffer
from common.logger import Video
from common.utils import (
    get_device,
    to_torch,
    to_np,
    FreezeParameters,
    lambda_return,
    preprocess,
    postprocess,
)

from .models.actor_critic import ActorModel, ValueModel
from .models.decoder import ObservationModel, RewardModel
from .models.encoder import Encoder
from .models.rssm import TransitionModel
from .models.utils import bottle, EnsembleDynamicsModel, InverseDynamicsModel

# https://github.com/yusukeurakami/dreamer-pytorch/blob/dreamer-torch1.8.2/main.py


class Dreamer:
    def __init__(self, config, env, eval_env, logger):
        self.c = config
        self.env = env
        self.eval_env = eval_env
        self.logger = logger
        self.device = get_device()

        self.step = 0
        self.build_models(config, env)
        self.buffer = SequenceReplayBuffer(
            config.replay_size,
            env.observation_space.shape,
            env.action_space.shape,
            obs_type=np.uint8 if config.pixel_obs else np.float32,
        )
        self.free_nats = torch.full((1,), config.free_nats).to(self.device)

    def build_models(self, config, env):
        if config.pixel_obs:
            obs_size = env.observation_space.shape
        else:
            obs_size = np.prod(env.observation_space.shape).item()
        action_size = np.prod(env.action_space.shape).item()

        # RSSM
        self.encoder = Encoder(
            not config.pixel_obs,
            obs_size,
            config.embedding_size,
            config.cnn_activation_function,
        ).to(self.device)

        self.transition_model = TransitionModel(
            config.belief_size,
            config.state_size,
            action_size,
            config.hidden_size,
            config.embedding_size,
            config.dense_activation_function,
        ).to(self.device)

        self.obs_model = ObservationModel(
            not config.pixel_obs,
            obs_size,
            config.belief_size,
            config.state_size,
            config.embedding_size,
            config.cnn_activation_function,
        ).to(self.device)

        self.reward_model = RewardModel(
            config.belief_size,
            config.state_size,
            config.hidden_size,
            config.dense_activation_function,
        ).to(self.device)

        self.model_params = (
            list(self.encoder.parameters())
            + list(self.transition_model.parameters())
            + list(self.obs_model.parameters())
            + list(self.reward_model.parameters())
        )
        self.model_optimizer = Adam(self.model_params, lr=config.model_lr)

        # Actor-critic
        self.actor_model = ActorModel(
            config.belief_size,
            config.state_size,
            config.hidden_size,
            action_size,
            config.dense_activation_function,
        ).to(self.device)
        self.actor_optimizer = Adam(self.actor_model.parameters(), lr=config.actor_lr)

        self.value_model = ValueModel(
            config.belief_size,
            config.state_size,
            config.hidden_size,
            config.dense_activation_function,
        ).to(self.device)
        self.value_optimizer = Adam(self.value_model.parameters(), lr=config.value_lr)

        # Disagreement model
        if self.c.disag_model:
            self.disag_model = EnsembleDynamicsModel(
                config.belief_size,
                config.state_size,
                action_size,
                config.hidden_size,
                config.ensemble_size,
                config.dense_activation_function,
            )
            self.disag_optimizer = Adam(
                self.disag_model.parameters(), lr=config.disag_lr
            )

        # Inverse dynamics
        if self.c.inv_dynamics:
            self.inv_dynamics = InverseDynamicsModel(
                config.belief_size,
                config.state_size,
                action_size,
                config.inv_dynamics_hidden_size,
                config.dense_activation_function,
            )
            self.inv_dynamics_optimizer = Adam(
                self.inv_dynamics.parameters(), lr=config.inv_dynamics_lr
            )

    def toggle_train(self, train=True):
        if train:
            self.encoder.train()
            self.transition_model.train()
            self.obs_model.train()
            self.reward_model.train()
            self.actor_model.train()
            self.value_model.train()
        else:
            self.encoder.eval()
            self.transition_model.eval()
            self.obs_model.eval()
            self.reward_model.eval()
            self.actor_model.eval()
            self.value_model.eval()

    def collect_seed_data(self):
        obs = self.env.reset()
        done = False
        # Make sure we collect entire episodes
        while len(self.buffer) < self.c.prefill or not done:
            action = self.env.action_space.sample()
            next_obs, reward, done, _ = self.env.step(action)
            self.buffer.push(obs, action, reward, done)
            obs = next_obs if not done else self.env.reset()

    def init_latent_and_action(self):
        belief = torch.zeros(1, self.c.belief_size).to(self.device)
        posterior_state = torch.zeros(1, self.c.state_size).to(self.device)
        action = torch.zeros(1, np.prod(self.env.action_space.shape)).to(self.device)
        return belief, posterior_state, action

    def update_latent_and_select_action(
        self,
        belief,
        posterior_state,
        action,
        obs,
        explore=False,
    ):
        # Action and observation need extra time dimension
        belief, _, _, _, posterior_state, _, _ = self.transition_model.observe(
            belief,
            posterior_state,
            action.unsqueeze(0),
            self.encoder(obs).unsqueeze(0),
        )
        # Remove time dimension from belief and state
        belief, posterior_state = belief.squeeze(0), posterior_state.squeeze(0)
        action = self.actor_model.get_action(belief, posterior_state, det=not explore)
        if explore:
            action += torch.randn_like(action) * self.c.action_noise
            action = torch.clamp(action, -1, 1)
        return belief, posterior_state, action

    def train_disag(self, beliefs, states, actions, nonterms):
        nonterm_inds = nonterms[1:-1].flatten() == 1
        actions_in, beliefs_in, states_in, beliefs_out = map(
            lambda x: x.detach().flatten(0, 1)[nonterm_inds],
            [
                actions[1:-1],
                beliefs[:-1],
                states[:-1],
                beliefs[1:],
            ],
        )
        ens_preds = self.disag_model(beliefs_in, states_in, actions_in)
        ens_dists = Independent(Normal(ens_preds, 1), 1)
        ens_targets = beliefs_out.unsqueeze(0).repeat(ens_preds.shape[0], 1, 1)
        disag_loss = -ens_dists.log_prob(ens_targets).sum(0).mean()

        self.disag_optimizer.zero_grad()
        disag_loss.backward()
        nn.utils.clip_grad_norm_(self.disag_model.parameters(), self.c.grad_clip_norm)
        self.disag_optimizer.step()
        self.logger.record("train/disag_loss", disag_loss.item())

    def train_inv_dynamics(self, beliefs, states, actions, nonterms):
        nonterm_inds = nonterms[1:-1].flatten() == 1
        actions_in, beliefs_in, states_in, beliefs_out = map(
            lambda x: x.detach().flatten(0, 1)[nonterm_inds],
            [
                actions[1:-1],
                beliefs[:-1],
                states[:-1],
                beliefs[1:],
            ],
        )
        act_mean, act_std = self.inv_dynamics(beliefs_in, states_in, beliefs_out)
        act_dist = Independent(Normal(act_mean, act_std), 1)
        inv_dyn_loss = -act_dist.log_prob(actions_in).mean()

        self.inv_dynamics_optimizer.zero_grad()
        inv_dyn_loss.backward()
        nn.utils.clip_grad_norm_(self.inv_dynamics.parameters(), self.c.grad_clip_norm)
        self.inv_dynamics_optimizer.step()
        self.logger.record("train/inv_dyn_loss", inv_dyn_loss.item())

    def train_dynamics(self, obs, actions, rewards, nonterms):
        init_belief = torch.zeros(self.c.batch_size, self.c.belief_size).to(self.device)
        init_state = torch.zeros(self.c.batch_size, self.c.state_size).to(self.device)
        embeds = bottle(self.encoder, (obs,))
        (
            beliefs,
            prior_states,
            prior_means,
            prior_std_devs,
            posterior_states,
            posterior_means,
            posterior_std_devs,
        ) = self.transition_model.observe(
            init_belief,
            init_state,
            actions[:-1],
            embeds[1:],
            nonterms[:-1],
        )

        # Reconstruction loss
        obs_dist = Normal(bottle(self.obs_model, (beliefs, posterior_states)), 1)
        obs_loss = (
            -obs_dist.log_prob(obs[1:])
            .sum((2, 3, 4) if self.c.pixel_obs else 2)
            .mean((0, 1))
        )

        # Reward loss
        # Since we predict rewards from next states, we need to shift reward
        # by one and account for terminal states
        rewards_tgt = rewards[:-1].squeeze(-1)
        mask = nonterms[:-1].squeeze(-1)
        reward_dist = Normal(bottle(self.reward_model, (beliefs, posterior_states)), 1)
        reward_loss = (-reward_dist.log_prob(rewards_tgt) * mask).mean((0, 1))

        # KL loss
        kl_div = kl_divergence(
            Normal(posterior_means, posterior_std_devs),
            Normal(prior_means, prior_std_devs),
        ).sum(2)
        kl_loss = torch.max(kl_div, self.free_nats).mean((0, 1))

        # Update model
        model_loss = obs_loss + reward_loss + kl_loss
        self.model_optimizer.zero_grad()
        model_loss.backward()
        nn.utils.clip_grad_norm_(self.model_params, self.c.grad_clip_norm)
        self.model_optimizer.step()

        # Logging
        self.logger.record("train/obs_loss", obs_loss.item())
        self.logger.record("train/reward_loss", reward_loss.item())
        self.logger.record("train/kl_loss", kl_loss.item())
        self.logger.record("train/model_loss", model_loss.item())

        # Update disagreement model and inverse dynamics
        if self.c.disag_model:
            self.train_disag(beliefs, posterior_states, actions, nonterms)
        if self.c.inv_dynamics:
            self.train_inv_dynamics(beliefs, posterior_states, actions, nonterms)
        return beliefs.detach(), posterior_states.detach()

    def train_actor_critic(self, beliefs, posterior_states):
        # Train actor
        with FreezeParameters(self.model_params):
            (
                imag_beliefs,
                imag_prior_states,
                imag_prior_means,
                imag_prior_std_devs,
            ) = self.transition_model.imagine(
                beliefs, posterior_states, self.actor_model, self.c.horizon
            )
        with FreezeParameters(self.model_params + list(self.value_model.parameters())):
            reward_preds = bottle(self.reward_model, (imag_beliefs, imag_prior_states))
            value_preds = bottle(self.value_model, (imag_beliefs, imag_prior_states))

        # Action entropy regularization
        action_dists = self.actor_model.get_action_dist(
            imag_beliefs.flatten(0, 1),
            imag_prior_states.flatten(0, 1),
        )
        action_entropy = action_dists.entropy().mean()

        # Latent entropy regularization
        latent_dists = Independent(Normal(imag_prior_means, imag_prior_std_devs), 1)
        latent_entropy = latent_dists.entropy().mean()

        # Disagreement bonus
        if self.c.disag_model and self.c.disag_coef > 0:
            with FreezeParameters(list(self.disag_model.parameters())):
                ens_preds = self.disag_model(
                    imag_beliefs.flatten(0, 1),
                    imag_prior_states.flatten(0, 1),
                    action_dists.rsample(),
                )
            disag = ens_preds.std(0).mean(-1).reshape(reward_preds.shape)
            reward_preds = reward_preds + self.c.disag_coef * disag

        # Generalized value estimation
        discounts = self.c.gamma * torch.ones_like(reward_preds)
        returns = lambda_return(
            reward_preds[:-1],
            value_preds[:-1],
            discounts[:-1],
            value_preds[-1],
            self.c.gae_lambda,
        )
        actor_loss = (
            -returns.mean()
            - self.c.action_ent_coef * action_entropy
            - self.c.latent_ent_coef * latent_entropy
        )

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor_model.parameters(), self.c.grad_clip_norm)
        self.actor_optimizer.step()

        # Train critic
        imag_beliefs = imag_beliefs[:-1].detach()
        imag_prior_states = imag_prior_states[:-1].detach()
        returns = returns.detach()
        value_dist = Normal(
            bottle(self.value_model, (imag_beliefs, imag_prior_states)), 1
        )
        value_loss = -value_dist.log_prob(returns).mean()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.value_model.parameters(), self.c.grad_clip_norm)
        self.value_optimizer.step()

        # Logging
        self.logger.record("train/actor_loss", actor_loss.item())
        self.logger.record("train/value_loss", value_loss.item())
        self.logger.record("train/action_entropy", action_entropy.item())
        self.logger.record("train/latent_entropy", latent_entropy.item())
        if self.c.disag_model and self.c.disag_coef > 0:
            self.logger.record("train/disagreement", disag.mean().item())

    def train_agent(self):
        for _ in range(self.c.train_steps):
            obs, actions, rewards, dones = self.buffer.sample(
                self.c.batch_size, self.c.chunk_size
            )
            obs = to_torch(bottle(preprocess, (obs,)))
            actions = to_torch(actions)
            rewards = to_torch(rewards)
            nonterms = to_torch(1 - dones)

            # Train dynamics model
            beliefs, posterior_states = self.train_dynamics(
                obs, actions, rewards, nonterms
            )

            # Train policy and value function
            beliefs = beliefs.flatten(0, 1)
            posterior_states = posterior_states.flatten(0, 1)
            self.train_actor_critic(beliefs, posterior_states)

    def train(self):
        if self.c.load_checkpoint:
            self.load_checkpoint()
        if len(self.buffer) == 0:
            self.collect_seed_data()

        belief, posterior_state, action_tensor = self.init_latent_and_action()
        obs = self.env.reset()
        episode_reward = 0
        episode_success = 0
        while self.step < self.c.num_steps:
            # Collect environment step
            obs_tensor = to_torch(preprocess(obs[None]))
            with torch.no_grad():
                (
                    belief,
                    posterior_state,
                    action_tensor,
                ) = self.update_latent_and_select_action(
                    belief, posterior_state, action_tensor, obs_tensor, True
                )
            action = to_np(action_tensor)[0]
            next_obs, reward, done, info = self.env.step(action)
            self.buffer.push(obs, action, reward, done)
            obs = next_obs
            episode_reward += reward
            episode_success += info.get("success", 0)
            if done:
                self.logger.record("train/return", episode_reward)
                self.logger.record("train/success", float(episode_success > 0))
                belief, posterior_state, action_tensor = self.init_latent_and_action()
                obs = self.env.reset()
                episode_reward = 0
                episode_success = 0

            # Train agent
            if self.step % self.c.train_every == 0:
                self.train_agent()

            # Evaluate agent
            if self.step % self.c.eval_every == 0:
                self.eval_agent()

            # Save checkpoint
            if self.step % self.c.checkpoint_every == 0:
                self.save_checkpoint()

            # Log metrics
            if self.step % self.c.log_every == 0:
                self.logger.record("train/step", self.step)
                self.logger.dump(step=self.step)

            self.step += 1

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
                    obs_hat = to_np(self.obs_model(belief, posterior_state))
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

    def save_checkpoint(self):
        # Save checkpoint
        params = self.get_param_dict()
        torch.save(params, os.path.join(self.logger.dir, f"models.pt"))

        # Save buffer
        if self.c.save_buffer:
            self.buffer.save(os.path.join(self.logger.dir, "buffer.npz"))

    def get_param_dict(self):
        params = {
            "step": self.step,
            "encoder": self.encoder.state_dict(),
            "transition_model": self.transition_model.state_dict(),
            "obs_model": self.obs_model.state_dict(),
            "reward_model": self.reward_model.state_dict(),
            "actor_model": self.actor_model.state_dict(),
            "value_model": self.value_model.state_dict(),
            "model_optimizer": self.model_optimizer.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "value_optimizer": self.value_optimizer.state_dict(),
        }
        if self.c.disag_model:
            params["disag_model"] = self.disag_model.state_dict()
            params["disag_optimizer"] = self.disag_optimizer.state_dict()
        if self.c.inv_dynamics:
            params["inv_dynamics"] = self.inv_dynamics.state_dict()
            params["inv_dynamics_optimizer"] = self.inv_dynamics_optimizer.state_dict()
        return params

    def load_checkpoint(self, ckpt_dir=None):
        if ckpt_dir == None:
            ckpt_dir = self.logger.dir

        # Load buffer
        buffer_path = os.path.join(ckpt_dir, f"buffer.npz")
        if os.path.exists(buffer_path):
            self.buffer.load(buffer_path)
            print(f"Loaded buffer from {buffer_path}")
        elif self.c.load_offline:
            # Only load offline if no buffer found
            # This is to prevent loading offline data twice
            # when resuming training from checkpoint
            self.load_offline_data()

        # Load models from the latest checkpoint
        params_path = os.path.join(ckpt_dir, f"models.pt")
        if os.path.exists(params_path):
            params = torch.load(params_path)
            self.load_param_dict(params)
            print(f"Loaded parameters from {params_path}")

    def load_param_dict(self, params):
        self.step = params["step"]
        self.encoder.load_state_dict(params["encoder"])
        self.transition_model.load_state_dict(params["transition_model"])
        self.obs_model.load_state_dict(params["obs_model"])
        self.reward_model.load_state_dict(params["reward_model"])
        self.actor_model.load_state_dict(params["actor_model"])
        self.value_model.load_state_dict(params["value_model"])
        self.model_optimizer.load_state_dict(params["model_optimizer"])
        self.actor_optimizer.load_state_dict(params["actor_optimizer"])
        self.value_optimizer.load_state_dict(params["value_optimizer"])

        if self.c.disag_model and "disag_model" in params:
            self.disag_model.load_state_dict(params["disag_model"])
            self.disag_optimizer.load_state_dict(params["disag_optimizer"])

        if self.c.inv_dynamics and "inv_dynamics" in params:
            self.inv_dynamics.load_state_dict(params["inv_dynamics"])
            self.inv_dynamics_optimizer.load_state_dict(
                params["inv_dynamics_optimizer"]
            )

    def load_offline_data(self):
        # Load offline sequence buffers
        paths = list(glob.glob(os.path.join(self.c.offline_dir, "buffer*.npz")))
        data_keys = ["observations", "actions", "rewards", "dones"]
        buffers = {k: [] for k in data_keys}
        for path in paths:
            buffer = np.load(path)
            data = {k: buffer[k] for k in data_keys}
            pos, full = buffer["pos"], buffer["full"]
            if full:
                # Unroll data
                data = {k: np.concatenate((v[pos:], v[:pos])) for k, v in data.items()}
            else:
                # Remove empty space
                data = {k: v[:pos] for k, v in data.items()}
            # Truncate buffer
            size = min(len(data["observations"]), self.c.offline_truncate_size)
            data = {k: v[:size] for k, v in data.items()}
            # Terminate at the end of each buffer
            data["dones"][-1, :] = 1
            for k in data.keys():
                buffers[k].append(data[k])
            print(f"Loaded buffer from {path}")
            buffer.close()
        # Combine data from all buffers
        buffer = {k: np.concatenate(v) for k, v in buffers.items()}
        buffer["capacity"] = len(buffer["observations"])
        buffer["pos"] = 0
        buffer["full"] = True
        for k, v in buffer.items():
            setattr(self.buffer, k, v)
