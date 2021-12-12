import torch
import torch.nn as nn
import torch.nn.functional as F

from .util import get_same_padding, init_weights


class ResBlockInner(nn.Module):

    def __init__(self, hidden_channels, kernel_size, dilation_rates, relu_slope=0.2):
        super(ResBlockInner, self).__init__()

        self.relu_slope = relu_slope

        self.convs = nn.ModuleList()
        for dilation_rate in dilation_rates:
            padding = get_same_padding(kernel_size, dilation_rate)
            layer = nn.Conv1d(hidden_channels, hidden_channels, kernel_size,
                              dilation=dilation_rate, padding=padding)
            self.convs.append(nn.utils.weight_norm(layer))

        self.convs.apply(init_weights)

    def forward(self, x):
        out = x.clone()
        for layer in self.convs:
            out = F.leaky_relu(layer(out), negative_slope=self.relu_slope)
        return out + x

    def remove_weight_norm(self):
        for layer in self.convs:
            nn.utils.remove_weight_norm(layer)


class ResBlock(nn.Module):

    def __init__(self, hidden_channels, kernel_sizes, dilation_rates_2d, relu_slope=0.2):
        super(ResBlock, self).__init__()

        layers = []
        for kernel_size, dilation_rates in zip(kernel_sizes, dilation_rates_2d):
            self.layers.append(ResBlockInner(hidden_channels, kernel_size, dilation_rates, relu_slope))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def remove_weight_norm(self):
        for i in range(len(self.net)):
            self.net[i].remove_weight_norm()


class MRFFusion(nn.Module):

    def __init__(self, n_blocks, hidden_channels, kernel_sizes, dilation_rates_2d, relu_slope=0.2):
        super(MRFFusion, self).__init__()

        self.n_blocks = n_blocks
        self.layers = nn.ModuleList()
        for _ in range(n_blocks):
            self.layers.append(
                ResBlockInner(hidden_channels, kernel_sizes, dilation_rates_2d, relu_slope)
            )

    def forward(self, x):
        out = torch.zeros_like(x)
        for layer in self.layers:
            out += layer(x)

        return out / self.n_blocks

    def remove_weight_norm(self):
        for layer in self.layers:
            layer.remove_weight_norm()
