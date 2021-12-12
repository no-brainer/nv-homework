import torch
import torch.nn as nn
import torch.nn.functional as F

from nv_hw.base import BaseModel
from nv_hw.model.modules import SubMSD
from nv_hw.model.util import get_same_padding


class MSDiscriminator(BaseModel):

    def __init__(
            self,
            filter_counts,
            kernel_sizes,
            strides,
            groups,
            use_spectral_norms,
            pool_kernel,
            pool_stride,
            relu_slope=0.2,
    ):
        super(MSDiscriminator, self).__init__()

        self.n_discs = len(use_spectral_norms)

        self.discs = nn.ModuleList()
        for i, use_spectral_norm in enumerate(use_spectral_norms):
            self.discs.append(
                SubMSD(filter_counts, kernel_sizes, strides, groups, use_spectral_norm, relu_slope)
            )

        pool_padding = get_same_padding(pool_kernel)
        self.pools = nn.ModuleList([
            nn.AvgPool1d(pool_kernel, pool_stride, pool_padding)
            for _ in range(self.n_discs - 1)
        ])

    def forward(self, waveform_real, waveform_fake, *args, **kwargs):
        outs_real, outs_fake = [], []
        fmaps_real, fmaps_fake = [], []

        for i, disc in enumerate(self.discs):
            if i > 0:
                waveform_real = self.pools[i - 1](waveform_real)
                waveform_fake = self.pools[i - 1](waveform_fake)

            out_real, fmap_real = disc(waveform_real)
            outs_real.append(out_real)
            fmaps_real.append(fmap_real)

            out_fake, fmap_fake = disc(waveform_fake)
            outs_fake.append(out_fake)
            fmaps_fake.append(fmap_fake)

        return outs_real, outs_fake, fmaps_real, fmaps_fake
