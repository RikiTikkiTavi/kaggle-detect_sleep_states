import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn.functional import log_softmax, nll_loss
from typing import Union, Optional, Sequence


class FocalMSELoss(nn.Module):

    def __init__(
            self,
            alpha: list[float],
            pos_weight: list[float],
            gamma: float = 0.,
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
            x_cls_i = x[:, :, class_i]
            y_cls_i = y[:, :, class_i]
            bce_loss = F.binary_cross_entropy_with_logits(
                input=x_cls_i,
                target=y_cls_i,
                pos_weight=pos_weight[class_i]
            )
            p_t = torch.exp(-bce_loss)
            loss += class_alpha * ((1 - p_t) ** self.gamma * bce_loss).mean()
            if class_i > 0:
                mse_loss = nn.MSELoss()(x_cls_i, y_cls_i)
                mse_norm_factor = (x_cls_i.max() - y_cls_i.min()) ** 2 + 1e-5
                loss += (mse_loss / mse_norm_factor)

        return loss
