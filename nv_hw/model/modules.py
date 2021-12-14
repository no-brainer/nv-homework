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
            layers.append(ResBlockInner(hidden_channels, kernel_size, dilation_rates, relu_slope))

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
                ResBlock(hidden_channels, kernel_sizes, dilation_rates_2d, relu_slope)
            )

    def forward(self, x):
        out = torch.zeros_like(x)
        for layer in self.layers:
            out += layer(x)

        return out / self.n_blocks

    def remove_weight_norm(self):
        for layer in self.layers:
            layer.remove_weight_norm()


class SubMPD(nn.Module):

    def __init__(self, period, filter_counts, kernel_size=5, stride=3, use_spectral_norm=False, relu_slope=0.2):
        super(SubMPD, self).__init__()

        self.period = period
        self.relu_slope = relu_slope

        norm = nn.utils.spectral_norm if use_spectral_norm else nn.utils.weight_norm

        padding = get_same_padding(kernel_size)

        self.convs = nn.ModuleList()
        for i in range(len(filter_counts)):
            in_filters = 1 if i == 0 else filter_counts[i - 1]
            self.convs.append(norm(nn.Conv2d(in_filters, filter_counts[i],
                                             kernel_size=(kernel_size, 1), stride=(stride, 1), padding=(padding, 0))))

        self.convs.append(
            norm(nn.Conv2d(filter_counts[-1], 1, kernel_size=(3, 1), padding=(1, 0)))
        )

    def forward(self, x):
        fmap = []

        batch_size, channels, t = x.shape
        if t % self.period:
            pad_size = self.period - (t % self.period)
            x = F.pad(x, (0, pad_size), "reflect")
            t += pad_size
        x = x.view(batch_size, channels, t // self.period, self.period)

        for conv in self.convs:
            x = F.leaky_relu(conv(x), negative_slope=self.relu_slope)
            fmap.append(x.clone())

        return x.squeeze(1), fmap


class SubMSD(nn.Module):

    def __init__(self, filter_counts, kernel_sizes, strides, groups, use_spectral_norm=False, relu_slope=0.2):
        super(SubMSD, self).__init__()

        self.relu_slope = relu_slope

        norm = nn.utils.spectral_norm if use_spectral_norm else nn.utils.weight_norm

        self.convs = nn.ModuleList()
        for i in range(len(filter_counts)):
            in_filters = 1 if i == 0 else filter_counts[i - 1]
            padding = get_same_padding(kernel_sizes[i])

            self.convs.append(
                norm(nn.Conv1d(in_filters, filter_counts[i], kernel_sizes[i],
                               stride=strides[i], groups=groups[i], padding=padding))
            )

        self.convs.append(
            norm(nn.Conv1d(filter_counts[-1], 1, 3, padding=1))
        )

    def forward(self, x):
        fmap = []

        for conv in self.convs:
            x = F.leaky_relu(conv(x), negative_slope=self.relu_slope)
            fmap.append(x.clone())

        return x.squeeze(1), fmap
