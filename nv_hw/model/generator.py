import torch
import torch.nn as nn
import torch.nn.functional as F

from nv_hw.base import BaseModel
from nv_hw.model.modules import MRFFusion
from nv_hw.model.util import get_same_padding


class Generator(BaseModel):

    def __init__(
            self,
            n_mels,
            preconv_kernel_size,
            postconv_kernel_size,
            conv_t_kernel_sizes,
            n_mrf_blocks,
            hidden_channels,
            kernel_sizes,
            dilation_rates_2d,
            relu_slope=0.2
    ):
        super(Generator, self).__init__()

        self.pre_conv = nn.Conv1d(n_mels, hidden_channels,
                                  kernel_size=preconv_kernel_size,
                                  padding=get_same_padding(preconv_kernel_size))

        self.num_upsample_blocks = len(conv_t_kernel_sizes)
        self.relu_slope = relu_slope

        upsample_blocks = []
        channels = hidden_channels
        for conv_t_kernel_size in conv_t_kernel_sizes:
            stride = conv_t_kernel_size // 2
            padding = (conv_t_kernel_size - stride) // 2
            channels //= 2

            upsample_blocks.extend([
                nn.LeakyReLU(negative_slope=relu_slope),
                nn.utils.weight_norm(nn.ConvTranspose1d(channels * 2, channels, conv_t_kernel_size,
                                                        stride=stride, padding=padding)),
                MRFFusion(n_mrf_blocks, channels, kernel_sizes, dilation_rates_2d, relu_slope=relu_slope),
            ])

        self.net = nn.Sequential(*upsample_blocks)

        self.post_conv = nn.utils.weight_norm(nn.Conv1d(channels, 1, postconv_kernel_size,
                                                        padding=get_same_padding(postconv_kernel_size)))

    def forward(self, melspecs, *args, **kwargs):
        out = self.net(self.pre_conv(melspecs))
        out = self.post_conv(F.leaky_relu(out, negative_slope=self.relu_slope))
        return torch.tanh(out)

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.pre_conv)
        for i in range(self.num_upsample_blocks):
            nn.utils.remove_weight_norm(self.net[3 * i + 1])
            self.net[3 * i + 2].remove_weight_norm()

        nn.utils.remove_weight_norm(self.post_conv)
