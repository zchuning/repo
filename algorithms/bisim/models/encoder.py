import torch
import torch.nn as nn


def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


class PixelEncoder(nn.Module):
    """Convolutional encoder of pixels observations."""

    def __init__(self, obs_shape, feature_dim, num_filters=32):
        super().__init__()
        assert len(obs_shape) == 3
        self.feature_dim = feature_dim
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(obs_shape[0], num_filters, 3, stride=2),
                nn.Conv2d(num_filters, num_filters, 3, stride=1),
                nn.Conv2d(num_filters, num_filters, 3, stride=1),
                nn.Conv2d(num_filters, num_filters, 3, stride=1),
            ]
        )
        self.fc = nn.Linear(num_filters * 25 * 25, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

    def forward_conv(self, x):
        for i in range(len(self.convs)):
            x = torch.relu(self.convs[i](x))
        return x.view(x.size(0), -1)

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)
        if detach:
            h = h.detach()
        out = self.ln(self.fc(h))
        return out

    def copy_conv_weights_from(self, source):
        for i in range(len(self.convs)):
            tie_weights(src=source.convs[i], trg=self.convs[i])
