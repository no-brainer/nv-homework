from abc import abstractmethod
from typing import Union

import numpy as np
from torch import Tensor
import torch.nn as nn


class BaseModel(nn.Module):
    """
    Base class for all models
    """

    def __init__(self, *args, **kwargs):
        super(BaseModel, self).__init__()

    @abstractmethod
    def forward(self, *args, **kwargs) -> Union[Tensor, dict]:
        """
        Forward pass logic.
        Can return a torch.Tensor (it will be interpreted as logits) or a dict.

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super(BaseModel, self).__str__() + "\nTrainable parameters: {}".format(params)
