import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn.functional import log_softmax, nll_loss
from typing import Union, Optional, Sequence

from torchvision.ops.focal_loss import sigmoid_focal_loss
from collections.abc import Sequence

class FocalLoss(nn.Module):

    def __init__(
            self,
            alpha: list[float],
            pos_weight: list[float],
            gamma: list[float],
            reduction: str = 'mean',
    ):
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(
                'Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.pos_weight = pos_weight

    def __repr__(self):
        arg_keys = ['alpha', 'gamma', 'ignore_index', 'reduction']
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f'{k}={v!r}' for k, v in zip(arg_keys, arg_vals)]
        arg_str = ', '.join(arg_strs)
        return f'{type(self).__name__}({arg_str})'

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        :param x: shape(N, L, C)
        :param y: x: shape(N, L, C)
        :return: tensor of shape (N)
        """
        loss = torch.tensor(0, device=x.device, dtype=torch.float32)
        pos_weight = torch.tensor(self.pos_weight, dtype=torch.float32, device=x.device)

        for class_i, class_alpha in enumerate(self.alpha):
            bce_loss = F.binary_cross_entropy_with_logits(
                input=x[:, :, class_i],
                target=y[:, :, class_i],
                pos_weight=pos_weight[class_i]
            )
            p_t = torch.exp(-bce_loss)
            gamma = self.gamma[class_i] if isinstance(self.gamma, Sequence) else self.gamma
            loss += class_alpha * ((1 - p_t) ** gamma * bce_loss).mean()

        return loss
