import numpy as np
import torch
import torch.nn as nn

from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.optim import Adam

from common.buffers import MultitaskSequenceReplayBuffer
from common.logger import Video
from common.utils import (
    to_torch,
    to_np,
    FreezeParameters,
    lambda_return,
    preprocess,
    postprocess,
)

from .dreamer import Dreamer
from .models.actor_critic import ConditionalActorModel, ConditionalValueModel
from .models.decoder import ConditionalObservationModel, ConditionalRewardModel
from .models.encoder import ConditionalEncoder, DummyConditionalEncoder
from .models.rssm import ConditionalTransitionModel, DummyConditionalTransitionModel
from .models.utils import bottle


class MultitaskDreamer(Dreamer):
    def __init__(self, config, env, eval_env, logger):
        super().__init__(config, env, eval_env, logger)
        self.buffer = MultitaskSequenceReplayBuffer(
            config.replay_size,
            env.num_tasks,
            env.observation_space.shape,
            env.action_space.shape,
            obs_type=np.uint8 if config.pixel_obs else np.float32,
        )

    def build_models(self, config, env):
        if config.pixel_obs:
            obs_size = env.observation_space.shape
        else:
            obs_size = np.prod(env.observation_space.shape).item()
        action_size = np.prod(env.action_space.shape).item()
        condition_size = self.env.num_tasks

        # RSSM
        if self.c.share_repr:
            self.encoder = DummyConditionalEncoder(
                not config.pixel_obs,
                obs_size,
                config.embedding_size,
                condition_size,
                config.cnn_activation_function,
            ).to(self.device)

            self.transition_model = DummyConditionalTransitionModel(
                config.belief_size,
                config.state_size,
                action_size,
                config.hidden_size,
                config.embedding_size,
                config.dense_activation_function,
            ).to(self.device)
        else:
            self.encoder = ConditionalEncoder(
                not config.pixel_obs,
                obs_size,
                config.embedding_size,
                condition_size,
                config.cnn_activation_function,
            ).to(self.device)

            self.transition_model = ConditionalTransitionModel(
                config.belief_size,
                config.state_size,
                action_size,
                config.hidden_size,
                config.embedding_size,
                condition_size,
                config.dense_activation_function,
            ).to(self.device)

        self.obs_model = ConditionalObservationModel(
            not config.pixel_obs,
            obs_size,
            config.belief_size,
            config.state_size,
            config.embedding_size,
            condition_size,
            config.cnn_activation_function,
        ).to(self.device)

        self.reward_model = ConditionalRewardModel(
            config.belief_size,
            config.state_size,
            config.hidden_size,
            condition_size,
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
        self.actor_model = ConditionalActorModel(
            config.belief_size,
            config.state_size,
            config.hidden_size,
            action_size,
            condition_size,
            config.dense_activation_function,
        ).to(self.device)
        self.actor_optimizer = Adam(self.actor_model.parameters(), lr=config.actor_lr)

        self.value_model = ConditionalValueModel(
            config.belief_size,
            config.state_size,
            config.hidden_size,
            condition_size,
            config.dense_activation_function,
        ).to(self.device)
        self.value_optimizer = Adam(self.value_model.parameters(), lr=config.value_lr)

    def collect_seed_data(self):
        obs = self.env.reset()
        done = False
        # Make sure we collect entire episodes
        while len(self.buffer) < self.c.prefill or not done:
            action = self.env.action_space.sample()
            next_obs, reward, done, _ = self.env.step(action)
            self.buffer.push(self.env.task_one_hot, obs, action, reward, done)
            obs = next_obs if not done else self.env.reset()

    def update_latent_and_select_action(
        self,
        belief,
        posterior_state,
        action,
        obs,
        task,
        explore=False,
    ):
        # Action and observation need extra time dimension
        belief, _, _, _, posterior_state, _, _ = self.transition_model.observe(
            belief,
            posterior_state,
            action.unsqueeze(0),
            task.unsqueeze(0),
            self.encoder(obs, task).unsqueeze(0),
        )
        # Remove time dimension from belief and state
        belief, posterior_state = belief.squeeze(0), posterior_state.squeeze(0)
        action = self.actor_model.get_action(
            belief, posterior_state, task, det=not explore
        )
        if explore:
            action += torch.randn_like(action) * self.c.action_noise
            action = torch.clamp(action, -1, 1)
        return belief, posterior_state, action

    def train_dynamics(self, tasks, obs, actions, rewards, nonterms):
        init_belief = torch.zeros(self.c.batch_size, self.c.belief_size).to(self.device)
        init_state = torch.zeros(self.c.batch_size, self.c.state_size).to(self.device)
        embeds = bottle(self.encoder, (obs, tasks))
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
            tasks[:-1],
            embeds[1:],
            nonterms[:-1],
        )
        # Match task timestep
        tasks = tasks[1:]

        # Reconstruction loss
        obs_dist = Normal(bottle(self.obs_model, (beliefs, posterior_states, tasks)), 1)
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
        reward_dist = Normal(
            bottle(self.reward_model, (beliefs, posterior_states, tasks)), 1
        )
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
        return beliefs.detach(), posterior_states.detach()

    def train_actor_critic(self, tasks, beliefs, posterior_states):
        # Train actor
        with FreezeParameters(self.model_params):
            (
                imag_beliefs,
                imag_prior_states,
                imag_prior_means,
                imag_prior_std_devs,
            ) = self.transition_model.imagine(
                beliefs, posterior_states, tasks, self.actor_model, self.c.horizon
            )
        # Expand tasks to match timestep
        tasks = tasks[None].repeat(self.c.horizon - 1, 1, 1)
        with FreezeParameters(self.model_params + list(self.value_model.parameters())):
            reward_preds = bottle(
                self.reward_model, (imag_beliefs, imag_prior_states, tasks)
            )
            value_preds = bottle(
                self.value_model, (imag_beliefs, imag_prior_states, tasks)
            )

        # Entropy regularization
        action_dists = self.actor_model.get_action_dist(
            imag_beliefs.flatten(0, 1),
            imag_prior_states.flatten(0, 1),
            tasks.flatten(0, 1),
        )
        action_entropy = action_dists.entropy().mean()
        latent_entropy = imag_prior_std_devs.log().sum(-1).mean()

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
        tasks = tasks[:-1].detach()
        returns = returns.detach()
        value_dist = Normal(
            bottle(self.value_model, (imag_beliefs, imag_prior_states, tasks)), 1
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

    def train_agent(self):
        for _ in range(self.c.train_steps):
            tasks, obs, actions, rewards, dones = self.buffer.sample(
                self.c.batch_size, self.c.chunk_size
            )
            tasks = to_torch(tasks)
            obs = to_torch(bottle(preprocess, (obs,)))
            actions = to_torch(actions)
            rewards = to_torch(rewards)
            nonterms = to_torch(1 - dones)

            # Train dynamics model
            beliefs, posterior_states = self.train_dynamics(
                tasks, obs, actions, rewards, nonterms
            )

            # Train policy and value function
            tasks = tasks[1:].flatten(0, 1)
            beliefs = beliefs.flatten(0, 1)
            posterior_states = posterior_states.flatten(0, 1)
            self.train_actor_critic(tasks, beliefs, posterior_states)

    def train(self):
        if self.c.load_checkpoint:
            self.load_checkpoint()
        if len(self.buffer) == 0:
            self.collect_seed_data()

        belief, posterior_state, action_tensor = self.init_latent_and_action()
        obs = self.env.reset()
        task = self.env.task
        episode_reward = 0
        episode_success = 0
        while self.step < self.c.num_steps:
            # Collect environment step
            obs_tensor = to_torch(preprocess(obs[None]))
            task_tensor = to_torch(self.env.task_one_hot[None])
            with torch.no_grad():
                (
                    belief,
                    posterior_state,
                    action_tensor,
                ) = self.update_latent_and_select_action(
                    belief,
                    posterior_state,
                    action_tensor,
                    obs_tensor,
                    task_tensor,
                    True,
                )
            action = to_np(action_tensor)[0]
            next_obs, reward, done, info = self.env.step(action)
            self.buffer.push(self.env.task_one_hot, obs, action, reward, done)
            obs = next_obs
            episode_reward += reward
            episode_success += info.get("success", 0)
            if done:
                self.logger.record(f"train/return_{task}", episode_reward)
                self.logger.record(f"train/success_{task}", float(episode_success > 0))
                belief, posterior_state, action_tensor = self.init_latent_and_action()
                obs = self.env.reset()
                task = self.env.task
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
        # Iterate over all tasks starting from the current task
        # This ensures we come back to the same task for training
        for _ in range(self.env.num_tasks):
            belief, posterior_state, action_tensor = self.init_latent_and_action()
            task = self.eval_env.sample_task(round_robin=True)
            obs = self.eval_env.reset(task=task)
            done = False
            total_reward = 0
            total_success = 0
            frames = []
            with torch.no_grad():
                while not done:
                    obs_tensor = to_torch(preprocess(obs[None]))
                    task_tensor = to_torch(self.eval_env.task_one_hot[None])
                    (
                        belief,
                        posterior_state,
                        action_tensor,
                    ) = self.update_latent_and_select_action(
                        belief,
                        posterior_state,
                        action_tensor,
                        obs_tensor,
                        task_tensor,
                        False,
                    )
                    action = to_np(action_tensor)[0]
                    next_obs, reward, done, info = self.eval_env.step(action)
                    if self.c.pixel_obs:
                        obs_hat = to_np(
                            self.obs_model(belief, posterior_state, task_tensor)
                        )
                        obs_hat = postprocess(obs_hat)[0]
                        frames.append([obs, obs_hat])
                    obs = next_obs
                    total_reward += reward
                    total_success += info.get("success", 0)
            self.logger.record(f"test/return_{task}", total_reward)
            self.logger.record(f"test/success_{task}", float(total_success > 0))
            if self.c.pixel_obs:
                # video shape: (T, N, C, H, W) -> (N, T, C, H, W)
                video = Video(np.stack(frames).transpose(1, 0, 2, 3, 4), fps=30)
                self.logger.record(f"test/video_{task}", video, exclude="stdout")
        self.toggle_train(True)
