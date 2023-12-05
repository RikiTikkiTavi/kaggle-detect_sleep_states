import torch
import torch.nn as nn
import detect_sleep_states.loss.bce_weighted
import torch.nn.functional as F


def proximity_loss(signal1, signal2):
    # Ensure that the signals have the same length
    assert signal1.size() == signal2.size(), "Signals must have the same length"

    # Compute the cross-correlation
    cross_corr = F.conv1d(signal1.view(1, 1, -1), signal2.view(1, 1, -1), padding=signal1.size(0) - 1).view(-1)

    # Normalize by the standard deviations of the signals
    norm_factor = torch.sqrt((signal1 ** 2).sum() * (signal2 ** 2).sum())
    proximity = cross_corr**2 / norm_factor

    return proximity.sum()


class BCESleepLoss(nn.Module):
    def __init__(
            self,
            lambda1,
            lambda2,
            bce_loss: detect_sleep_states.loss.bce_weighted.BceLogitsLossWeighted
    ):
        super(BCESleepLoss, self).__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.binary_cross_entropy = bce_loss

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        :param predictions: shape (N, L, C)
        :param targets: shape (N, L, C)
        :return: loss value
        """

        if self.lambda1.device != predictions.device:
            self.lambda1.to(predictions.device)

        if self.lambda2.device != predictions.device:
            self.lambda2.to(predictions.device)

        # Binary cross-entropy loss for each label
        bce_loss: torch.Tensor = self.binary_cross_entropy(predictions, targets)

        proximity_penalty = torch.tensor(0, device=predictions.device, dtype=torch.float32)

        for batch_i in range(predictions.size(0)):
            proximity_penalty += proximity_loss(predictions[batch_i, :, 1], predictions[batch_i, :, 2])

        return self.lambda1 * bce_loss + self.lambda2 * proximity_penalty
