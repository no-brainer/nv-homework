import torch
import torch.nn as nn
import torch.nn.functional as F

from nv_hw.base import BaseModel
from nv_hw.model.modules import SubMPD


class MPDiscriminator(BaseModel):

    def __init__(self, periods, filter_counts):
        super(MPDiscriminator, self).__init__()

        self.modules = nn.ModuleList()
        for period in periods:
            self.modules.append(SubMPD(period, filter_counts))

    def forward(self, waveform_real, waveform_fake, *args, **kwargs):
        outs_real, outs_fake = [], []
        fmaps_real, fmaps_fake = [], []

        for disc in self.modules:
            out_real, fmap_real = disc(waveform_real)
            outs_real.append(out_real)
            fmaps_real.append(fmap_real)

            out_fake, fmap_fake = disc(waveform_fake)
            outs_fake.append(out_fake)
            fmaps_fake.append(fmap_fake)

        return outs_real, outs_fake, fmaps_real, fmaps_fake
