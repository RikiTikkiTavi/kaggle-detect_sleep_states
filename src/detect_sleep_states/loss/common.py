from typing import Callable, Optional

import torch

from detect_sleep_states.config import TrainConfig, LossConfig
from detect_sleep_states.loss.focal import FocalLoss


class SignalSparsityLoss(torch.nn.Module):
    def __init__(self, a: float, b: float):
        super().__init__()
        self.a = torch.tensor(a)
        self.b = torch.tensor(b)

    def _calculate_sparsity(self, input: torch.Tensor):
        diffs = input[1:] - input[:1]
        abs_diff = torch.abs(diffs)
        pos_diff = torch.maximum(diffs, torch.tensor(0))
        return self.a * torch.sum(abs_diff - self.b * pos_diff)

    def forward(self, input: torch.Tensor, target: torch.Tensor):

        if self.a.device != input.device:
            self.a = self.a.to(input.device)

        if self.b.device != input.device:
            self.b = self.b.to(input.device)

        target_sparsity = self._calculate_sparsity(target)

        return torch.abs(self._calculate_sparsity(input) - target_sparsity) / (target_sparsity + 1)


def proximity_measure(signal1, signal2):
    # Ensure that the signals have the same length
    assert signal1.size() == signal2.size(), "Signals must have the same length"

    # Compute the cross-correlation
    cross_corr = torch.nn.functional.conv1d(signal1.view(1, 1, -1), signal2.view(1, 1, -1), padding=signal1.size(0) - 1).view(-1)

    # Normalize by the standard deviations of the signals
    norm_factor = torch.sqrt((signal1 ** 2).sum() * (signal2 ** 2).sum())
    proximity = cross_corr / norm_factor

    # Add a penalty term for the absence of peaks
    penalty_term = 1.0 - torch.sqrt((signal1 ** 2).sum() + (signal2 ** 2).sum()) / norm_factor

    # Combine the cross-correlation and penalty terms
    proximity_with_penalty = proximity + penalty_term

    return proximity_with_penalty.mean()  # Return a scalar value


class BceLogitsLossWeighted(torch.nn.Module):
    def __init__(
            self,
            class_weights: torch.Tensor,
            pos_weight: torch.Tensor,
            sparsity_loss: Optional[SignalSparsityLoss] = None
    ):
        super().__init__()
        self.class_weights = class_weights
        self.pos_weight = pos_weight
        self.sparsity_loss = sparsity_loss

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """

        :param x1: shape (N, L, C)
        :param x2: shape (N, L, C)
        :return:
        """

        loss = torch.tensor(0.0, dtype=input.dtype, device=input.device)

        if self.class_weights.device != input.device:
            self.class_weights = self.class_weights.to(input.device)

        if self.pos_weight.device != input.device:
            self.pos_weight = self.pos_weight.to(input.device)

        for i in range(input.size(2)):
            loss += 10 * self.class_weights[i] * torch.nn.functional.binary_cross_entropy_with_logits(
                input[:, :, i],
                target[:, :, i],
                pos_weight=self.pos_weight[i]
            )

        for i_batch in range(input.size(0)):
            pm = proximity_measure(input[i_batch, :, 1], input[i_batch, :, 2])
            loss += proximity_measure(input[i_batch, :, 1], input[i_batch, :, 2])
            for i_cls in range(1, input.size(2)):
                loss += self.sparsity_loss(input[i_batch, :, i_cls], target[i_batch, :, i_cls])

        return loss


def get_loss(cfg: LossConfig) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    if cfg.name == "BCE":
        return BceLogitsLossWeighted(
            class_weights=torch.tensor(cfg.params["class_weight"], dtype=torch.float32),
            pos_weight=torch.tensor(cfg.params["pos_weight"], dtype=torch.float32),
            sparsity_loss=SignalSparsityLoss(a=0.0001, b=0.5)
        )
    elif cfg.name == "focal":
        return FocalLoss(**cfg.params)
    elif cfg.name == "CrossEntropyLoss":
        return torch.nn.CrossEntropyLoss()
    else:
        raise Exception(f"Loss '{cfg.name}' is not supported yet.")
