from typing import Callable

import torch

from detect_sleep_states.config import TrainConfig, LossConfig
from detect_sleep_states.loss.focal import FocalLoss


def get_loss(cfg: LossConfig) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    if cfg.name == "BCE":
        return torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(cfg.params["pos_weight"], dtype=torch.float32)
        )
    elif cfg.name == "focal":
        return FocalLoss(**cfg.params)
    else:
        raise Exception(f"Loss '{cfg.name}' is not supported yet.")
