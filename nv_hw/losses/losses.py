import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneratorLoss(nn.Module):

    def __init__(self):
        super(GeneratorLoss, self).__init__()

    def forward(self, disc_outputs, *args, **kwargs):
        loss = 0.
        gen_losses = []
        for disc_output in disc_outputs:
            loss_part = F.mse_loss(disc_output, torch.ones_like(disc_output))
            gen_losses.append(loss_part)
            loss += loss_part

        return loss, gen_losses


class DiscriminatorLoss(nn.Module):

    def __init__(self):
        super(DiscriminatorLoss, self).__init__()

    def forward(self, disc_outs_real, disc_outs_fake, *args, **kwargs):
        loss = 0.
        real_losses = []
        fake_losses = []
        for out_real, out_fake in zip(disc_outs_real, disc_outs_fake):
            real_loss = F.mse_loss(out_real, torch.ones_like(out_real))
            fake_loss = F.mse_loss(out_fake, torch.zeros_like(out_fake))

            loss += real_loss + fake_loss

            real_losses.append(real_loss.item())
            fake_losses.append(fake_loss.item())

        return loss, real_losses, fake_losses


class FeatureLoss(nn.Module):

    def __init__(self):
        super(FeatureLoss, self).__init__()

    def forward(self, disc_fmaps_real, disc_fmaps_fake, *args, **kwargs):
        loss = 0.
        for fmap_real, fmap_fake in zip(disc_fmaps_real, disc_fmaps_fake):
            for fmap_real_i, fmap_fake_i in zip(fmap_real, fmap_fake):
                loss += F.l1_loss(fmap_real_i, fmap_fake_i)

        return 2 * loss


class MelLoss(nn.Module):

    def __init__(self):
        super(MelLoss, self).__init__()

    def forward(self, melspec_real, melspec_fake):
        return F.l1_loss(melspec_fake, melspec_real)
