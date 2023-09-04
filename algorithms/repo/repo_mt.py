import numpy as np
import torch
import torch.nn as nn

from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.optim import Adam

from .dreamer_mt import MultitaskDreamer
from .models.utils import bottle


class MultitaskRePo(MultitaskDreamer):
    def build_models(self, config, env):
        super().build_models(config, env)
        # Constrained optimization
        if self.c.share_repr:
            self.log_beta = torch.tensor(
                np.log(config.init_beta),
                dtype=torch.float,
                device=self.device,
                requires_grad=True,
            )
        else:
            # Disentangle log betas between tasks
            self.log_beta = torch.tensor(
                [np.log(config.init_beta) for _ in range(self.env.num_tasks)],
                dtype=torch.float,
                device=self.device,
                requires_grad=True,
            )
        self.beta_optimizer = Adam([self.log_beta], lr=self.c.beta_lr)

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

        # Reconstruction loss for probing
        obs_inputs = (beliefs.detach(), posterior_states.detach(), tasks)
        obs_dist = Normal(bottle(self.obs_model, obs_inputs), 1)
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
        kl_prior = kl_divergence(
            Normal(posterior_means.detach(), posterior_std_devs.detach()),
            Normal(prior_means, prior_std_devs),
        ).sum(2)
        kl_post = kl_divergence(
            Normal(posterior_means, posterior_std_devs),
            Normal(prior_means.detach(), prior_std_devs.detach()),
        ).sum(2)
        kl_alpha = self.c.prior_train_steps / (1 + self.c.prior_train_steps)
        kl_div = kl_alpha * kl_prior + (1 - kl_alpha) * kl_post
        kl_viol = kl_div - self.c.target_kl
        if self.c.share_repr:
            kl_loss = self.log_beta.exp().detach() * kl_viol.mean()
        else:
            log_beta = tasks @ self.log_beta
            kl_loss = (log_beta.exp().detach() * kl_viol).mean()

        # Update model
        model_loss = obs_loss + reward_loss + kl_loss
        self.model_optimizer.zero_grad()
        model_loss.backward()
        nn.utils.clip_grad_norm_(self.model_params, self.c.grad_clip_norm)
        self.model_optimizer.step()

        # Update dual variable
        if self.c.share_repr:
            beta_loss = -self.log_beta * kl_viol.detach().mean()
        else:
            beta_loss = -(log_beta * kl_viol.detach()).mean()
        self.beta_optimizer.zero_grad()
        beta_loss.backward()
        self.beta_optimizer.step()

        # Logging
        self.logger.record("train/obs_loss", obs_loss.item())
        self.logger.record("train/reward_loss", reward_loss.item())
        self.logger.record("train/kl_loss", kl_loss.item())
        self.logger.record("train/kl_div", kl_div.mean().item())
        self.logger.record("train/model_loss", model_loss.item())
        self.logger.record("train/beta_loss", beta_loss.item())
        if self.c.share_repr:
            self.logger.record("train/beta", self.log_beta.exp().item())
        else:
            for i in range(self.env.num_tasks):
                self.logger.record(f"train/beta_{i}", self.log_beta[i].exp().item())
        return beliefs.detach(), posterior_states.detach()

    def get_param_dict(self):
        params = super().get_param_dict()
        params["log_beta"] = self.log_beta
        params["beta_optimizer"] = self.beta_optimizer.state_dict()
        return params

    def load_param_dict(self, params):
        super().load_param_dict(params)
        self.log_beta = params["log_beta"]
        self.beta_optimizer = Adam([self.log_beta], lr=self.c.beta_lr)
        self.beta_optimizer.load_state_dict(params["beta_optimizer"])
