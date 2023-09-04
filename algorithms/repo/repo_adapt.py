import copy
import glob
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal, Independent
from torch.distributions.kl import kl_divergence
from torch.optim import Adam

from common.buffers import SequenceReplayBuffer
from common.utils import (
    to_torch,
    to_np,
    FreezeParameters,
    preprocess,
)
from common.models.gans import VDBDiscriminator
from common.models.mlps import MLP
from .models.utils import bottle
from .repo import RePo


class FinetunedRePo(RePo):
    def build_models(self, config, env):
        super().build_models(config, env)
        self.encoder_optimizer = Adam(self.encoder.parameters(), lr=config.model_lr)

    def train_encoder(self, obs, actions, rewards, nonterms):
        embeds = bottle(self.encoder, (obs,))
        init_belief = torch.zeros(self.c.batch_size, self.c.belief_size).to(self.device)
        init_state = torch.zeros(self.c.batch_size, self.c.state_size).to(self.device)
        with FreezeParameters(list(self.transition_model.parameters())):
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

        # Reward loss
        # Since we predict rewards from next states, we need to shift reward
        # by one and account for terminal states
        rewards_tgt = rewards[:-1].squeeze(-1)
        mask = nonterms[:-1].squeeze(-1)
        with FreezeParameters(list(self.reward_model.parameters())):
            reward_dist = Normal(
                bottle(self.reward_model, (beliefs, posterior_states)), 1
            )
            reward_loss = (-reward_dist.log_prob(rewards_tgt) * mask).mean((0, 1))

        # KL loss
        kl_div = (
            kl_divergence(
                Normal(posterior_means, posterior_std_devs),
                Normal(prior_means, prior_std_devs),
            )
            .sum(2)
            .mean((0, 1))
        )
        kl_viol = kl_div - self.c.target_kl
        kl_loss = self.log_beta.exp().detach() * kl_viol

        # Update encoder
        self.encoder_optimizer.zero_grad()
        encoder_loss = reward_loss + kl_loss
        encoder_loss.backward()
        nn.utils.clip_grad_norm_(self.encoder.parameters(), self.c.grad_clip_norm)
        self.encoder_optimizer.step()

        # Update dual variable
        beta_loss = -self.log_beta * kl_viol.detach()
        self.beta_optimizer.zero_grad()
        beta_loss.backward()
        self.beta_optimizer.step()

        # Logging
        self.logger.record("train/reward_loss", reward_loss.item())
        self.logger.record("train/kl_loss", kl_loss.item())
        self.logger.record("train/kl_div", kl_div.item())
        self.logger.record("train/encoder_loss", encoder_loss.item())
        self.logger.record("train/beta", self.log_beta.exp().item())
        self.logger.record("train/beta_loss", beta_loss.item())

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
            self.train_encoder(obs, actions, rewards, nonterms)

    def train(self):
        self.load_source_models()
        super().train()

    def load_source_models(self):
        # Load models from the latest checkpoint
        ckpt_path = os.path.join(self.c.source_dir, f"models.pt")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path)
            print(f"Loaded checkpoint from {ckpt_path}")

            # Load both src and tgt encoders
            self.encoder.load_state_dict(ckpt["encoder"])
            self.transition_model.load_state_dict(ckpt["transition_model"])
            self.obs_model.load_state_dict(ckpt["obs_model"])
            self.reward_model.load_state_dict(ckpt["reward_model"])
            self.actor_model.load_state_dict(ckpt["actor_model"])
            self.value_model.load_state_dict(ckpt["value_model"])


class CalibrationBuffer(SequenceReplayBuffer):
    def _get_samples(self, batch_inds):
        obs, act, rew, done = super()._get_samples(batch_inds)
        src_obs, tgt_obs = np.split(obs, 2, axis=1)
        return src_obs, tgt_obs, act, rew, done


class CalibratedRePo(RePo):
    def __init__(self, config, env, eval_env, calib_env, logger):
        assert config.disag_model or config.inv_dynamics
        super().__init__(config, env, eval_env, logger)
        self.calib_env = calib_env
        self.calib_buffer = CalibrationBuffer(
            config.calibration_buffer_size,
            calib_env.observation_space.shape,
            calib_env.action_space.shape,
            obs_type=np.uint8,
        )
        self.src_buffer = SequenceReplayBuffer(
            config.replay_size,
            env.observation_space.shape,
            env.action_space.shape,
            obs_type=np.uint8 if config.pixel_obs else np.float32,
        )

    def build_models(self, config, env):
        super().build_models(config, env)
        self.src_encoder = copy.deepcopy(self.encoder)
        self.encoder_optimizer = Adam(self.encoder.parameters(), lr=config.model_lr)

        hidden_dims = [config.f_hidden_size] * 4
        self.disc = VDBDiscriminator(
            input_dim=config.embedding_size,
            hidden_dims=hidden_dims,
            latent_dim=config.f_latent_size,
            lr=config.f_lr,
            target_kl=config.f_target_kl,
        ).to(self.device)

        self.log_tau = MLP(config.embedding_size, hidden_dims, 1).to(self.device)
        self.tau_optimizer = Adam(self.log_tau.parameters(), lr=config.tau_lr)

        self.u = torch.tensor(config.init_u, device=self.device, requires_grad=True)
        self.u_optimizer = Adam([self.u], lr=config.u_lr)

    def expert_update_latent_and_select_action(
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
            self.src_encoder(obs).unsqueeze(0),
        )
        # Remove time dimension from belief and state
        belief, posterior_state = belief.squeeze(0), posterior_state.squeeze(0)
        action = self.actor_model.get_action(belief, posterior_state, det=not explore)
        if explore:
            action += torch.randn_like(action) * self.c.action_noise
            action = torch.clamp(action, -1, 1)
        return belief, posterior_state, action

    def collect_calibration_data(self, expert):
        # Collect random paired trajectories
        print("Collecting calibration trajectories")
        if expert:
            obs = self.calib_env.reset()
            belief, posterior_state, action_tensor = self.init_latent_and_action()
            timestep = 0
            for _ in range(self.c.calibration_buffer_size):
                obs_tensor = to_torch(preprocess(obs[:3][None]))
                with torch.no_grad():
                    (
                        belief,
                        posterior_state,
                        action_tensor,
                    ) = self.expert_update_latent_and_select_action(
                        belief, posterior_state, action_tensor, obs_tensor, False
                    )
                action = to_np(action_tensor)[0]
                next_obs, reward, done, info = self.calib_env.step(action)
                timestep += 1
                if timestep == self.c.calib_time_limit:
                    done = True
                    timestep = 0
                # Push to calibration buffer
                self.calib_buffer.push(obs, action, reward, done)
                # Push to target buffer
                self.buffer.push(obs[3:], action, reward, done)
                obs = next_obs
                if done:
                    obs = self.calib_env.reset()
                    (
                        belief,
                        posterior_state,
                        action_tensor,
                    ) = self.init_latent_and_action()
        else:
            obs = self.calib_env.reset()
            for _ in range(self.c.calibration_buffer_size):
                action = self.calib_env.action_space.sample()
                next_obs, reward, done, info = self.calib_env.step(action)
                # Push to calibration buffer
                self.calib_buffer.push(obs, action, reward, done)
                # Push to target buffer
                self.buffer.push(obs[3:], action, reward, done)
                obs = next_obs
                if done:
                    obs = self.calib_env.reset()

    def pair_calibration(self):
        bz, cz = self.c.batch_size, self.c.chunk_size

        # Sample alignment source trajectories
        aln_src_obs = self.src_buffer.sample(bz, cz)[0]
        aln_src_obs = to_torch(bottle(preprocess, (aln_src_obs,)))
        aln_src_embeds = bottle(self.src_encoder, (aln_src_obs,))

        # Sample alignment target trajectories
        aln_tgt_obs, aln_tgt_actions, _, aln_tgt_dones = self.buffer.sample(bz, cz)
        aln_tgt_obs = to_torch(bottle(preprocess, (aln_tgt_obs,)))
        aln_tgt_actions = to_torch(aln_tgt_actions)
        aln_tgt_nonterms = to_torch(1 - aln_tgt_dones)
        aln_tgt_embeds = bottle(self.encoder, (aln_tgt_obs,))

        # Sample calibration trajectories
        cal_src_obs, cal_tgt_obs, cal_actions, _, cal_dones = self.calib_buffer.sample(
            bz, cz
        )
        cal_src_obs = to_torch(bottle(preprocess, (cal_src_obs,)))
        cal_tgt_obs = to_torch(bottle(preprocess, (cal_tgt_obs,)))
        cal_actions = to_torch(cal_actions)
        cal_nonterms = to_torch(1 - cal_dones)
        cal_src_embeds = bottle(self.src_encoder, (cal_src_obs,))
        cal_tgt_embeds = bottle(self.encoder, (cal_tgt_obs,))

        # Batch forward through the dynamics model
        init_belief = torch.zeros(bz * 3, self.c.belief_size).to(self.device)
        init_state = torch.zeros(bz * 3, self.c.state_size).to(self.device)
        embeds = torch.cat((cal_src_embeds, cal_tgt_embeds, aln_tgt_embeds), 1)
        actions = torch.cat((cal_actions, cal_actions, aln_tgt_actions), 1)
        nonterms = torch.cat((cal_nonterms, cal_nonterms, aln_tgt_nonterms), 1)
        with FreezeParameters(list(self.transition_model.parameters())):
            (
                beliefs,
                prior_states,
                prior_means,
                prior_stds,
                posterior_states,
                posterior_means,
                posterior_stds,
            ) = self.transition_model.observe(
                init_belief,
                init_state,
                actions[:-1],
                embeds[1:],
                nonterms[:-1],
            )
        cal_src_beliefs, cal_tgt_beliefs, aln_tgt_beliefs = beliefs.chunk(3, 1)
        cal_src_posts, cal_tgt_posts, aln_tgt_posts = posterior_states.chunk(3, 1)

        # Alignment
        aln_src_embeds = aln_src_embeds.flatten(0, 1)
        aln_tgt_embeds = aln_tgt_embeds.flatten(0, 1)
        if self.c.alignment_mode == "support":
            tau = self.log_tau(aln_src_embeds).exp()
            disc_losses = self.disc.train(
                aln_src_embeds.detach(), aln_tgt_embeds.detach(), tau.detach()
            )
        else:
            disc_losses = self.disc.train(
                aln_src_embeds.detach(), aln_tgt_embeds.detach()
            )

        d_tgt = self.disc(aln_tgt_embeds)[0]
        if self.c.alignment_mode == "support":
            aln_loss = -(d_tgt + 0.25 * d_tgt**2).mean()
        else:
            aln_loss = self.disc._bce_with_logits(d_tgt, 1)

        # Dynamics consistency
        nonterm_inds = aln_tgt_nonterms[1:-1].flatten() == 1
        actions_in, beliefs_in, states_in, beliefs_out = map(
            lambda x: x.flatten(0, 1)[nonterm_inds],
            [
                aln_tgt_actions[1:-1],
                aln_tgt_beliefs[:-1],
                aln_tgt_posts[:-1],
                aln_tgt_beliefs[1:],
            ],
        )
        if self.c.disag_model:
            ens_preds = self.disag_model(beliefs_in, states_in, actions_in)
            ens_dists = Independent(Normal(ens_preds, 1), 1)
            ens_targets = beliefs_out.unsqueeze(0).repeat(ens_preds.shape[0], 1, 1)
            dyn_loss = -ens_dists.log_prob(ens_targets).mean()
        elif self.c.inv_dynamics:
            act_mean, act_std = self.inv_dynamics(beliefs_in, states_in, beliefs_out)
            act_dist = Independent(Normal(act_mean, act_std), 1)
            dyn_loss = -act_dist.log_prob(actions_in).mean()
        else:
            raise ValueError()

        # Calibration
        nonterm_inds = cal_nonterms[1:-1].flatten() == 1
        actions_in, beliefs_in, states_in, beliefs_out = map(
            lambda x: x.flatten(0, 1)[nonterm_inds],
            [
                cal_actions[1:-1],
                cal_src_beliefs[:-1],
                cal_src_posts[:-1],
                cal_tgt_beliefs[1:],
            ],
        )
        if self.c.disag_model:
            ens_preds = self.disag_model(beliefs_in, states_in, actions_in)
            ens_dists = Independent(Normal(ens_preds, 1), 1)
            ens_targets = beliefs_out.unsqueeze(0).repeat(ens_preds.shape[0], 1, 1)
            calib_loss = -ens_dists.log_prob(ens_targets).mean()
        elif self.c.inv_dynamics:
            act_mean, act_std = self.inv_dynamics(beliefs_in, states_in, beliefs_out)
            act_dist = Independent(Normal(act_mean, act_std), 1)
            calib_loss = -act_dist.log_prob(actions_in).mean()
        else:
            raise ValueError()

        # Update encoder
        self.encoder_optimizer.zero_grad()
        encoder_loss = (
            self.c.aln_coef * aln_loss
            + self.c.dyn_coef * dyn_loss
            + self.c.calib_coef * calib_loss
        )
        encoder_loss.backward()
        self.encoder_optimizer.step()

        self.logger.record("train/f_loss_src", disc_losses["real_loss"])
        self.logger.record("train/f_loss_tgt", disc_losses["fake_loss"])
        self.logger.record("train/f_kl", disc_losses["kl"])
        self.logger.record("train/aln_loss", aln_loss.item())
        self.logger.record("train/dyn_loss", dyn_loss.item())
        self.logger.record("train/calib_loss", calib_loss.item())
        self.logger.record("train/encoder_loss", encoder_loss.item())

        if self.c.alignment_mode == "support":
            # Train density ratio
            d_src = self.disc(aln_src_embeds)[0]
            tau_obj = (tau * d_src).mean()
            tau_constr = self.u.detach() * (tau - 1).mean()

            self.tau_optimizer.zero_grad()
            tau_loss = tau_obj + tau_constr
            tau_loss.backward()
            self.tau_optimizer.step()

            # Update dual variable for density constraint
            self.u_optimizer.zero_grad()
            u_loss = -self.u * (tau - 1).mean().detach()
            u_loss.backward()
            self.u_optimizer.step()

            self.logger.record("train/tau_loss", tau_loss.item())
            self.logger.record("train/tau_mean", tau.mean().item())
            self.logger.record("train/u_value", self.u.item())

    def simple_pair_calibration(self):
        bz, cz = self.c.batch_size, self.c.chunk_size

        # Sample alignment source trajectories
        aln_src_obs = self.src_buffer.sample(bz, cz)[0]
        aln_src_obs = to_torch(bottle(preprocess, (aln_src_obs,)))
        aln_src_embeds = bottle(self.src_encoder, (aln_src_obs,))

        # Sample alignment target trajectories
        aln_tgt_obs, aln_tgt_actions, _, aln_tgt_dones = self.buffer.sample(bz, cz)
        aln_tgt_obs = to_torch(bottle(preprocess, (aln_tgt_obs,)))
        aln_tgt_actions = to_torch(aln_tgt_actions)
        aln_tgt_nonterms = to_torch(1 - aln_tgt_dones)
        aln_tgt_embeds = bottle(self.encoder, (aln_tgt_obs,))

        # Sample calibration trajectories
        cal_src_obs, cal_tgt_obs, cal_actions, _, cal_dones = self.calib_buffer.sample(
            bz, cz
        )
        cal_src_obs = to_torch(bottle(preprocess, (cal_src_obs,)))
        cal_tgt_obs = to_torch(bottle(preprocess, (cal_tgt_obs,)))
        cal_actions = to_torch(cal_actions)
        cal_nonterms = to_torch(1 - cal_dones)
        cal_src_embeds = bottle(self.src_encoder, (cal_src_obs,))
        cal_tgt_embeds = bottle(self.encoder, (cal_tgt_obs,))

        # Alignment
        aln_src_embeds = aln_src_embeds.flatten(0, 1)
        aln_tgt_embeds = aln_tgt_embeds.flatten(0, 1)
        if self.c.alignment_mode == "support":
            tau = self.log_tau(aln_src_embeds).exp()
            disc_losses = self.disc.train(
                aln_src_embeds.detach(), aln_tgt_embeds.detach(), tau.detach()
            )
        else:
            disc_losses = self.disc.train(
                aln_src_embeds.detach(), aln_tgt_embeds.detach()
            )

        d_tgt = self.disc(aln_tgt_embeds)[0]
        if self.c.alignment_mode == "support":
            aln_loss = -(d_tgt + 0.25 * d_tgt**2).mean()
        else:
            aln_loss = self.disc._bce_with_logits(d_tgt, 1)

        # Calibration
        embeds_dist = Normal(cal_tgt_embeds, 1)
        calib_loss = -embeds_dist.log_prob(cal_src_embeds).mean()
        # calib_loss = F.mse_loss(cal_src_embeds, cal_tgt_embeds)

        # Update encoder
        self.encoder_optimizer.zero_grad()
        encoder_loss = self.c.aln_coef * aln_loss + self.c.calib_coef * calib_loss
        encoder_loss.backward()
        self.encoder_optimizer.step()

        self.logger.record("train/f_loss_src", disc_losses["real_loss"])
        self.logger.record("train/f_loss_tgt", disc_losses["fake_loss"])
        self.logger.record("train/f_kl", disc_losses["kl"])
        self.logger.record("train/aln_loss", aln_loss.item())
        self.logger.record("train/calib_loss", calib_loss.item())
        self.logger.record("train/encoder_loss", encoder_loss.item())

        if self.c.alignment_mode == "support":
            # Train density ratio
            d_src = self.disc(aln_src_embeds)[0]
            tau_obj = (tau * d_src).mean()
            tau_constr = self.u.detach() * (tau - 1).mean()

            self.tau_optimizer.zero_grad()
            tau_loss = tau_obj + tau_constr
            tau_loss.backward()
            self.tau_optimizer.step()

            # Update dual variable for density constraint
            self.u_optimizer.zero_grad()
            u_loss = -self.u * (tau - 1).mean().detach()
            u_loss.backward()
            self.u_optimizer.step()

            self.logger.record("train/tau_loss", tau_loss.item())
            self.logger.record("train/tau_mean", tau.mean().item())
            self.logger.record("train/u_value", self.u.item())

    def train_agent(self):
        for _ in range(self.c.train_steps):
            if self.c.calibration_mode == "pair":
                self.pair_calibration()
            elif self.c.calibration_mode == "simple_pair":
                self.simple_pair_calibration()
            else:
                raise ValueError("Unsupported calibration mode")

    def train(self):
        self.load_source_models()
        self.load_source_data()
        if self.c.calibration_mode in ["pair", "simple_pair"]:
            self.collect_calibration_data(expert=self.c.expert_calib_data)

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

            self.step += 1  # increment step first

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

    def load_source_models(self):
        # Load models from the latest checkpoint
        ckpt_path = os.path.join(self.c.source_dir, f"models.pt")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path)
            print(f"Loaded model from {ckpt_path}")

            # Load both src and tgt encoders
            self.src_encoder.load_state_dict(ckpt["encoder"])
            self.encoder.load_state_dict(ckpt["encoder"])
            self.transition_model.load_state_dict(ckpt["transition_model"])
            self.obs_model.load_state_dict(ckpt["obs_model"])
            self.actor_model.load_state_dict(ckpt["actor_model"])
            self.value_model.load_state_dict(ckpt["value_model"])
            if self.c.disag_model and "disag_model" in ckpt:
                self.disag_model.load_state_dict(ckpt["disag_model"])
            if self.c.inv_dynamics and "inv_dynamics" in ckpt:
                self.inv_dynamics.load_state_dict(ckpt["inv_dynamics"])

    def load_source_data(self):
        # Load offline sequence buffers
        paths = list(glob.glob(os.path.join(self.c.source_dir, "buffer*.npz")))
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
            setattr(self.src_buffer, k, v)
