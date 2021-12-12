import torch
import torch.nn as nn


def get_same_padding(kernel_size: int, dilation: int = 1) -> int:
    return (kernel_size - 1) * dilation // 2


def init_weights(module, mu=0., sigma=0.01):
    if "Conv" in module.__class__.__name__:
        module.weight.data.normal_(mu, sigma)


def apply_weight_norm(module):
    if "Conv" in module.__class__.__name__:
        nn.utils.weight_norm(module)
