# PyTorch
import torch
from torch import nn
import torch.nn.functional as F


class FocalTverskyLoss(nn.Module):
    def __init__(
            self,
            alpha: list[float],
            beta: list[float],
            gamma=1.0,
            smooth=1.0,
    ):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = F.sigmoid(inputs)
        loss = torch.tensor(0, device=inputs.device, dtype=torch.float32)

        for class_i, class_alpha in enumerate(self.alpha):
            inputs_cls_i = inputs[:, :, class_i]
            targets_cls_i = targets[:, :, class_i]

            # flatten label and prediction tensors
            inputs_cls_i = torch.flatten(inputs_cls_i)
            targets_cls_i = torch.flatten(targets_cls_i)

            # True Positives, False Positives & False Negatives
            TP = (inputs_cls_i * targets_cls_i).sum()
            FP = ((1 - targets_cls_i) * inputs_cls_i).sum()
            FN = (targets_cls_i * (1 - inputs_cls_i)).sum()
            alpha = self.alpha[class_i]
            beta = self.beta[class_i]
            tversky_loss = (TP + self.smooth) / (TP + alpha * FP + beta * FN + self.smooth)
            focal_tversky_loss = (1 - tversky_loss) ** self.gamma
            loss += focal_tversky_loss

        return loss
