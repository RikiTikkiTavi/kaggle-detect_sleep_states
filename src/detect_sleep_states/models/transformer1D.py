from typing import Optional

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torchvision.transforms.functional import resize

from detect_sleep_states.augmentation.cutmix import Cutmix
from detect_sleep_states.augmentation.mixup import Mixup
from detect_sleep_states.models.base import BaseModel
from detect_sleep_states.models.decoder.transformerdecoder import TransformerDecoder


class Transformer1D(BaseModel):
    def __init__(
            self,
            decoder: TransformerDecoder,
            mixup_alpha: float = 0.5,
            cutmix_alpha: float = 0.5,
    ):
        super().__init__()
        self.decoder = decoder
        self.mixup = Mixup(mixup_alpha)
        self.cutmix = Cutmix(cutmix_alpha)

    def _forward(
            self,
            x: torch.Tensor,
            labels: Optional[torch.Tensor] = None,
            do_mixup: bool = False,
            do_cutmix: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        :param x: (N, L, E)
        :param labels: (N, L, C)
        :param do_mixup:
        :param do_cutmix:
        :return:
        """

        if do_mixup and labels is not None:
            x, labels = self.mixup(x, labels)
        if do_cutmix and labels is not None:
            x, labels = self.cutmix(x, labels)

        logits = self.decoder(x)

        if labels is not None:
            return logits, labels
        else:
            return logits

    def logits_to_proba_per_step(self, logits: torch.Tensor, org_duration: int) -> torch.Tensor:
        preds = logits.sigmoid()
        return resize(preds, size=[org_duration, preds.shape[-1]], antialias=False)

    def correct_labels(self, labels: torch.Tensor, org_duration: int) -> torch.Tensor:
        return resize(labels, size=[org_duration, labels.shape[-1]], antialias=False)
