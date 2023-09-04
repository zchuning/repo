import torch
import torch.nn as nn
import torch.nn.functional as F


class SymbolicObservationModel(nn.Module):
    def __init__(
        self,
        observation_size,
        belief_size,
        state_size,
        embedding_size,
        activation_function="relu",
    ):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(belief_size + state_size, embedding_size)
        self.fc2 = nn.Linear(embedding_size, embedding_size)
        self.fc3 = nn.Linear(embedding_size, observation_size)

    def forward(self, belief, state):
        hidden = self.act_fn(self.fc1(torch.cat([belief, state], dim=1)))
        hidden = self.act_fn(self.fc2(hidden))
        observation = self.fc3(hidden)
        return observation


class VisualObservationModel(nn.Module):
    def __init__(
        self, belief_size, state_size, embedding_size, activation_function="relu"
    ):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.embedding_size = embedding_size
        self.fc1 = nn.Linear(belief_size + state_size, embedding_size)
        self.conv1 = nn.ConvTranspose2d(embedding_size, 128, 5, stride=2)
        self.conv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.conv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.conv4 = nn.ConvTranspose2d(32, 3, 6, stride=2)

    def forward(self, belief, state):
        hidden = self.fc1(torch.cat([belief, state], dim=1))  # No nonlinearity here
        hidden = hidden.view(-1, self.embedding_size, 1, 1)
        hidden = self.act_fn(self.conv1(hidden))
        hidden = self.act_fn(self.conv2(hidden))
        hidden = self.act_fn(self.conv3(hidden))
        observation = self.conv4(hidden)
        return observation


def ObservationModel(
    symbolic,
    observation_size,
    belief_size,
    state_size,
    embedding_size,
    activation_function="relu",
):
    if symbolic:
        return SymbolicObservationModel(
            observation_size,
            belief_size,
            state_size,
            embedding_size,
            activation_function,
        )
    else:
        return VisualObservationModel(
            belief_size, state_size, embedding_size, activation_function
        )


class ConditionalSymbolicObservationModel(SymbolicObservationModel):
    def __init__(
        self,
        observation_size,
        belief_size,
        state_size,
        embedding_size,
        condition_size,
        activation_function="relu",
    ):
        super().__init__(
            observation_size,
            belief_size,
            state_size + condition_size,
            embedding_size,
            activation_function,
        )

    def forward(self, belief, state, condition):
        pseudo_state = torch.cat((state, condition), dim=1)
        return super().forward(belief, pseudo_state)


class ConditionalVisualObservationModel(VisualObservationModel):
    def __init__(
        self,
        belief_size,
        state_size,
        embedding_size,
        condition_size,
        activation_function="relu",
    ):
        super().__init__(belief_size, state_size, embedding_size, activation_function)
        self.condition_size = condition_size
        self.film = nn.Linear(condition_size, 2 * (128 + 64 + 32))

    def mod(self, x, gamma, beta):
        return (1 + gamma[..., None, None]) * x + beta[..., None, None]

    def forward(self, belief, state, condition):
        gammas, betas = self.film(condition).chunk(2, dim=1)
        gammas = gammas.split([128, 64, 32], dim=1)
        betas = betas.split([128, 64, 32], dim=1)

        hidden = self.fc1(torch.cat([belief, state], dim=1))  # No nonlinearity here
        hidden = hidden.view(-1, self.embedding_size, 1, 1)
        hidden = self.act_fn(self.mod(self.conv1(hidden), gammas[0], betas[0]))
        hidden = self.act_fn(self.mod(self.conv2(hidden), gammas[1], betas[1]))
        hidden = self.act_fn(self.mod(self.conv3(hidden), gammas[2], betas[2]))
        observation = self.conv4(hidden)  # No modulation here
        return observation


def ConditionalObservationModel(
    symbolic,
    observation_size,
    belief_size,
    state_size,
    embedding_size,
    condition_size,
    activation_function="relu",
):
    if symbolic:
        return ConditionalSymbolicObservationModel(
            observation_size,
            belief_size,
            state_size,
            embedding_size,
            condition_size,
            activation_function,
        )
    else:
        return ConditionalVisualObservationModel(
            belief_size,
            state_size,
            embedding_size,
            condition_size,
            activation_function,
        )


class TIAObservationModel(nn.Module):
    def __init__(
        self, belief_size, state_size, embedding_size, activation_function="relu"
    ):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.embedding_size = embedding_size
        self.fc1 = nn.Linear(belief_size + state_size, embedding_size)
        self.conv1 = nn.ConvTranspose2d(embedding_size, 128, 5, stride=2)
        self.conv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.conv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.conv4 = nn.ConvTranspose2d(32, 6, 6, stride=2)

    def forward(self, belief, state):
        hidden = self.fc1(torch.cat([belief, state], dim=1))  # No nonlinearity here
        hidden = hidden.view(-1, self.embedding_size, 1, 1)
        hidden = self.act_fn(self.conv1(hidden))
        hidden = self.act_fn(self.conv2(hidden))
        hidden = self.act_fn(self.conv3(hidden))
        out = self.conv4(hidden)
        recon, mask = out.chunk(2, 1)
        return recon, mask


class RewardModel(nn.Module):
    def __init__(
        self, belief_size, state_size, hidden_size, activation_function="relu"
    ):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(belief_size + state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 1)

    def forward(self, belief, state):
        x = torch.cat([belief, state], dim=1)
        hidden = self.act_fn(self.fc1(x))
        hidden = self.act_fn(self.fc2(hidden))
        hidden = self.act_fn(self.fc3(hidden))
        reward = self.fc4(hidden).squeeze(dim=1)
        return reward


class ConditionalRewardModel(RewardModel):
    def __init__(
        self,
        belief_size,
        state_size,
        hidden_size,
        condition_size,
        activation_function="relu",
    ):
        super().__init__(
            belief_size, state_size + condition_size, hidden_size, activation_function
        )

    def forward(self, belief, state, condition):
        pseudo_state = torch.cat((state, condition), dim=1)
        return super().forward(belief, pseudo_state)
