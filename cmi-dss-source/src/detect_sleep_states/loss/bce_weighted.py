import torch


class BceLogitsLossWeighted(torch.nn.Module):
    def __init__(
            self,
            class_weights: torch.Tensor,
            pos_weight: torch.Tensor
    ):
        super().__init__()
        self.class_weights = class_weights
        self.pos_weight = pos_weight

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """

        :param input: shape (N, L, C)
        :param target: shape (N, L, C)
        :return:
        """

        loss = torch.tensor(0.0, dtype=input.dtype, device=input.device)

        if self.class_weights.device != input.device:
            self.class_weights = self.class_weights.to(input.device)

        if self.pos_weight.device != input.device:
            self.pos_weight = self.pos_weight.to(input.device)

        for i in range(input.size(2)):
            loss += self.class_weights[i] * torch.nn.functional.binary_cross_entropy_with_logits(
                input[:, :, i],
                target[:, :, i],
                pos_weight=self.pos_weight[i]
            )

        return loss
