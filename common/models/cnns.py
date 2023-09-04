from functools import partial

import torch
import torch.nn as nn

from .utils import init_weights


class CNN(nn.Module):
    def __init__(self, input_chn, output_dim, act="ReLU", output_act="Identity"):
        super().__init__()
        act_fn = getattr(nn, act)
        output_act_fn = getattr(nn, output_act)

        self.layers = nn.Sequential(
            nn.Conv2d(input_chn, 32, 8, stride=4),
            act_fn(),
            nn.Conv2d(32, 64, 4, stride=2),
            act_fn(),
            nn.Conv2d(64, 64, 3, stride=1),
            act_fn(),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, output_dim),
            output_act_fn(),
        )
        self.apply(partial(init_weights, act=act))

    def forward(self, x):
        return self.layers(x)


class MultiheadedCNN(CNN):
    def __init__(
        self, input_chn, num_heads, output_dim, act="ReLU", output_act="Identity"
    ):
        super().__init__(input_chn, output_dim * num_heads, act, output_act)
        self.output_dim = output_dim
        self.num_heads = num_heads

    def forward(self, x, select_inds=None):
        out = super().forward(x)
        out = out.view(-1, self.num_heads, self.output_dim)
        if select_inds is not None:
            out = out[torch.arange(out.shape[0]), select_inds]
        return out


class TransposeCNN(nn.Module):
    def __init__(self, input_dim, output_chn, act="ReLU", output_act="Identity"):
        super().__init__()
        act_fn = getattr(nn, act)
        output_act_fn = getattr(nn, output_act)

        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64 * 4 * 4),
            act_fn(),
            nn.Unflatten(1, (64, 4, 4)),
            nn.ConvTranspose2d(64, 64, 3, stride=1),
            act_fn(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, output_padding=1),
            act_fn(),
            nn.ConvTranspose2d(32, output_chn, 8, stride=4),
            output_act_fn(),
        )
        self.apply(partial(init_weights, act=act))

    def forward(self, x):
        return self.layers(x)
