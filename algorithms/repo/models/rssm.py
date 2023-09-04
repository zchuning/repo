from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransitionModel(nn.Module):
    def __init__(
        self,
        belief_size,
        state_size,
        action_size,
        hidden_size,
        embedding_size,
        activation_function="relu",
        min_std_dev=0.1,
    ):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.min_std_dev = min_std_dev
        # Transition dynamics
        self.fc_embed_state_action = nn.Linear(state_size + action_size, belief_size)
        self.rnn = nn.GRUCell(belief_size, belief_size)
        # Prior state
        self.fc_embed_belief_prior = nn.Linear(belief_size, hidden_size)
        self.fc_state_prior = nn.Linear(hidden_size, 2 * state_size)
        # Posterior state
        self.fc_embed_belief_posterior = nn.Linear(
            belief_size + embedding_size, hidden_size
        )
        self.fc_state_posterior = nn.Linear(hidden_size, 2 * state_size)

    def compute_belief(self, prev_belief, state, action):
        # Compute deterministic belief
        hidden = self.act_fn(
            self.fc_embed_state_action(torch.cat([state, action], dim=1))
        )
        belief = self.rnn(hidden, prev_belief)
        return belief

    def compute_prior_state(self, belief):
        # Compute prior stochastic state by applying transition dynamics
        hidden = self.act_fn(self.fc_embed_belief_prior(belief))
        prior_mean, prior_std_dev = torch.chunk(
            self.fc_state_prior(hidden), chunks=2, dim=1
        )
        prior_std_dev = F.softplus(prior_std_dev) + self.min_std_dev
        prior_state = prior_mean + prior_std_dev * torch.randn_like(prior_mean)
        return prior_state, prior_mean, prior_std_dev

    def compute_posterior_state(self, belief, observation):
        # Compute posterior stochastic state by applying transition dynamics and using current observation
        hidden = self.act_fn(
            self.fc_embed_belief_posterior(torch.cat([belief, observation], dim=1))
        )
        posterior_mean, posterior_std_dev = torch.chunk(
            self.fc_state_posterior(hidden), chunks=2, dim=1
        )
        posterior_std_dev = F.softplus(posterior_std_dev) + self.min_std_dev
        posterior_state = posterior_mean + (
            posterior_std_dev * torch.randn_like(posterior_mean)
        )
        return posterior_state, posterior_mean, posterior_std_dev

    # Operates over (previous) belief (previous) state, (previous) actions, (previous) nonterminals (mask), and (current) observations
    # Diagram of expected inputs and outputs for T = 5 (-x- signifying beginning of output belief/state that gets sliced off):
    # t :  0  1  2  3  4  5
    # o :    -X--X--X--X--X-
    # a : -X--X--X--X--X-
    # n : -X--X--X--X--X-
    # pb: -X-
    # ps: -X-
    # b : -X--X--X--X--X--X-
    # s : -X--X--X--X--X--X-
    def observe(
        self,
        prev_belief: torch.Tensor,
        prev_state: torch.Tensor,
        actions: torch.Tensor,
        observations: Optional[torch.Tensor] = None,
        nonterminals: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """
        Observe trajectory to get prior and posterior states.
        :param prev_belief:         previous deterministic belief           (batch_size, belief_size)
        :param prev_state:          previous stochastic state               (batch_size, state_size)
        :param actions:             sequence of actions                     (0:T-1, batch_size, action_shape)
        :param observations:        optional sequence of observations       (1:T, batch_size, obs_shape)
        :param nonterminals:        optional sequence of nonterm indicators (0:T-1, batch_size, 1)
        :return beliefs:            sequence of deterministic beliefs       (1:T, batch_size, belief_size)
        :return prior_states:       sequence of prior stochastic states     (1:T, batch_size, state_size)
        :return prior_means:        means of prior states                   (1:T, batch_size, state_size)
        :return prior_std_devs:     std devs of prior states                (1:T, batch_size, state_size)
        :return posterior_states:   sequence of posterior stochastic states (1:T, batch_size, state_size)
        :return posterior_means:    means of posterior states               (1:T, batch_size, state_size)
        :return posterior_std_devs: std devs of posterior states            (1:T, batch_size, state_size)
        """
        # Create lists for hidden states
        T = actions.size(0) + 1
        (
            beliefs,
            prior_states,
            prior_means,
            prior_std_devs,
            posterior_states,
            posterior_means,
            posterior_std_devs,
        ) = [[torch.empty(0)] * T for _ in range(7)]
        beliefs[0], prior_states[0], posterior_states[0] = (
            prev_belief,
            prev_state,
            prev_state,
        )
        # Loop over time sequence
        for t in range(T - 1):
            # Select appropriate previous state, mask if previous transition was terminal
            state = prior_states[t] if observations is None else posterior_states[t]
            state = state if nonterminals is None else state * nonterminals[t]
            beliefs[t + 1] = self.compute_belief(beliefs[t], state, actions[t])
            (
                prior_states[t + 1],
                prior_means[t + 1],
                prior_std_devs[t + 1],
            ) = self.compute_prior_state(beliefs[t + 1])
            if observations is not None:
                # Observation timestep is shifted by one
                (
                    posterior_states[t + 1],
                    posterior_means[t + 1],
                    posterior_std_devs[t + 1],
                ) = self.compute_posterior_state(beliefs[t + 1], observations[t])
        # Return new hidden states
        hidden = [
            torch.stack(beliefs[1:], dim=0),
            torch.stack(prior_states[1:], dim=0),
            torch.stack(prior_means[1:], dim=0),
            torch.stack(prior_std_devs[1:], dim=0),
        ]
        if observations is not None:
            hidden += [
                torch.stack(posterior_states[1:], dim=0),
                torch.stack(posterior_means[1:], dim=0),
                torch.stack(posterior_std_devs[1:], dim=0),
            ]
        return hidden

    def imagine(self, prev_belief, prev_state, policy, horizon):
        """
        Draw imaginary trajectory using dynamics and policy.
        :param prev_belief:     previous deterministic belief       (batch_size, belief_size)
        :param prev_state:      previous stochastic state           (batch_size, state_size)
        :param policy:          policy network
        :param horizon:         imagination horizon T
        :param task:            optional one-hot task vector
        :return beliefs:        sequence of deterministic beliefs   (1:T, batch_size, belief_size)
        :return prior_states:   sequence of prior stochastic states (1:T, batch_size, state_size)
        :return prior_means:    means of prior states               (1:T, batch_size, state_size)
        :return prior_std_devs: std devs of prior states            (1:T, batch_size, state_size)
        """
        # Create lists for hidden states
        beliefs, prior_states, prior_means, prior_std_devs = [
            [torch.empty(0)] * horizon for _ in range(4)
        ]
        beliefs[0], prior_states[0] = prev_belief, prev_state
        # Loop over time sequence
        for t in range(horizon - 1):
            belief = beliefs[t]
            state = prior_states[t]
            action = policy.get_action(belief.detach(), state.detach())
            beliefs[t + 1] = self.compute_belief(belief, state, action)
            (
                prior_states[t + 1],
                prior_means[t + 1],
                prior_std_devs[t + 1],
            ) = self.compute_prior_state(beliefs[t + 1])
        # Return new hidden states
        imagined_traj = [
            torch.stack(beliefs[1:], dim=0),
            torch.stack(prior_states[1:], dim=0),
            torch.stack(prior_means[1:], dim=0),
            torch.stack(prior_std_devs[1:], dim=0),
        ]
        return imagined_traj


class ConditionalTransitionModel(TransitionModel):
    def __init__(
        self,
        belief_size,
        state_size,
        action_size,
        hidden_size,
        embedding_size,
        condition_size,
        activation_function="relu",
        min_std_dev=0.1,
    ):
        super().__init__(
            belief_size,
            state_size,
            action_size + condition_size,
            hidden_size,
            embedding_size,
            activation_function,
            min_std_dev,
        )

    def observe(
        self,
        prev_belief: torch.Tensor,
        prev_state: torch.Tensor,
        actions: torch.Tensor,
        conditions: torch.Tensor,
        observations: Optional[torch.Tensor] = None,
        nonterminals: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        pseudo_actions = torch.cat((actions, conditions), dim=2)
        return super().observe(
            prev_belief, prev_state, pseudo_actions, observations, nonterminals
        )

    def imagine(self, prev_belief, prev_state, condition, policy, horizon):
        # Create lists for hidden states
        beliefs, prior_states, prior_means, prior_std_devs = [
            [torch.empty(0)] * horizon for _ in range(4)
        ]
        beliefs[0], prior_states[0] = prev_belief, prev_state
        # Loop over time sequence
        for t in range(horizon - 1):
            belief = beliefs[t]
            state = prior_states[t]
            action = policy.get_action(belief.detach(), state.detach(), condition)
            pseudo_action = torch.cat((action, condition), dim=1)
            beliefs[t + 1] = self.compute_belief(belief, state, pseudo_action)
            (
                prior_states[t + 1],
                prior_means[t + 1],
                prior_std_devs[t + 1],
            ) = self.compute_prior_state(beliefs[t + 1])
        # Return new hidden states
        imagined_traj = [
            torch.stack(beliefs[1:], dim=0),
            torch.stack(prior_states[1:], dim=0),
            torch.stack(prior_means[1:], dim=0),
            torch.stack(prior_std_devs[1:], dim=0),
        ]
        return imagined_traj


class DummyConditionalTransitionModel(TransitionModel):
    def __init__(
        self,
        belief_size,
        state_size,
        action_size,
        hidden_size,
        embedding_size,
        condition_size,
        activation_function="relu",
        min_std_dev=0.1,
    ):
        super().__init__(
            belief_size,
            state_size,
            action_size,
            hidden_size,
            embedding_size,
            activation_function,
            min_std_dev,
        )

    def observe(
        self,
        prev_belief: torch.Tensor,
        prev_state: torch.Tensor,
        actions: torch.Tensor,
        conditions: torch.Tensor,
        observations: Optional[torch.Tensor] = None,
        nonterminals: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        return super().observe(
            prev_belief, prev_state, actions, observations, nonterminals
        )

    def imagine(self, prev_belief, prev_state, condition, policy, horizon):
        # Create lists for hidden states
        beliefs, prior_states, prior_means, prior_std_devs = [
            [torch.empty(0)] * horizon for _ in range(4)
        ]
        beliefs[0], prior_states[0] = prev_belief, prev_state
        # Loop over time sequence
        for t in range(horizon - 1):
            belief = beliefs[t]
            state = prior_states[t]
            action = policy.get_action(belief.detach(), state.detach(), condition)
            beliefs[t + 1] = self.compute_belief(belief, state, action)
            (
                prior_states[t + 1],
                prior_means[t + 1],
                prior_std_devs[t + 1],
            ) = self.compute_prior_state(beliefs[t + 1])
        # Return new hidden states
        imagined_traj = [
            torch.stack(beliefs[1:], dim=0),
            torch.stack(prior_states[1:], dim=0),
            torch.stack(prior_means[1:], dim=0),
            torch.stack(prior_std_devs[1:], dim=0),
        ]
        return imagined_traj
