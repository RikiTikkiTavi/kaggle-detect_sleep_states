from typing import Callable, Optional

import torch

from detect_sleep_states.config import TrainConfig, LossConfig
import detect_sleep_states.loss.bce_weighted
import detect_sleep_states.loss.bce_sleep
import detect_sleep_states.loss.focal


def get_loss(cfg: LossConfig) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    if cfg.name == "BCE":
        return detect_sleep_states.loss.bce_weighted.BceLogitsLossWeighted(
            class_weights=torch.tensor(cfg.params["class_weight"], dtype=torch.float32),
            pos_weight=torch.tensor(cfg.params["pos_weight"], dtype=torch.float32),
        )
    elif cfg.name == "BCESleep":
        return detect_sleep_states.loss.bce_sleep.BCESleepLoss(
            lambda1=torch.tensor(cfg.params["lambda1"], dtype=torch.float32),
            lambda2=torch.tensor(cfg.params["lambda2"], dtype=torch.float32),
            bce_loss=detect_sleep_states.loss.bce_weighted.BceLogitsLossWeighted(
                class_weights=torch.tensor(cfg.params["class_weight"], dtype=torch.float32),
                pos_weight=torch.tensor(cfg.params["pos_weight"], dtype=torch.float32),
            )
        )
    elif cfg.name == "CrossEntropyLoss":
        return torch.nn.CrossEntropyLoss()
    elif cfg.name == "focal":
        return detect_sleep_states.loss.focal.FocalLoss(
            gamma=cfg.params["gamma"],
            alpha=cfg.params["alpha"],
        )
    else:
        raise Exception(f"Loss '{cfg.name}' is not supported yet.")
