import logging
from pathlib import Path
from typing import Optional, Any, Callable

import mlflow
import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.optim as optim
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import MLFlowLogger
from transformers import get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup

from detect_sleep_states.config import TrainConfig
from detect_sleep_states.models.base import ModelOutput
from detect_sleep_states.models.common import get_model, BaseModel
from detect_sleep_states.loss.common import get_loss
from detect_sleep_states.utils.common import nearest_valid_size
from detect_sleep_states.utils.metrics import event_detection_ap
from detect_sleep_states.utils.post_process import post_process_for_seg
from detect_sleep_states.plot.plot_predictions import plot_predictions_chunk

import detect_sleep_states.data_module

_logger = logging.getLogger(__name__)


class PLSleepModel(LightningModule):
    val_event_df: pd.DataFrame
    model: BaseModel
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    duration: int
    validation_step_outputs: list

    def __init__(
            self,
            cfg: TrainConfig,
            val_event_df: pl.DataFrame,
            model: BaseModel,
            duration: int,
            loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ):
        super().__init__()
        self.cfg = cfg
        self.val_event_df = val_event_df.to_pandas()

        self.model = model
        self.loss_fn = loss_fn
        self.duration = duration
        self.validation_step_outputs: list = []

    @staticmethod
    def from_config(
            cfg: TrainConfig,
            val_event_df: pl.DataFrame,
            n_classes: int,
            duration: int,
            feature_dim: int
    ) -> "PLSleepModel":
        n_timesteps = nearest_valid_size(int(duration * cfg.upsample_rate), cfg.downsample_rate)
        model = get_model(
            cfg,
            feature_dim=feature_dim,
            n_classes=n_classes,
            num_timesteps=n_timesteps // cfg.downsample_rate,
        )
        return PLSleepModel(
            cfg=cfg,
            model=model,
            loss_fn=get_loss(cfg.loss),
            val_event_df=val_event_df,
            duration=duration
        )

    def forward(
            self,
            x: torch.Tensor,
            labels: Optional[torch.Tensor] = None,
            do_mixup: bool = False,
            do_cutmix: bool = False,
    ) -> ModelOutput:
        return self.model(x, labels, do_mixup, do_cutmix)

    def training_step(self, batch, batch_idx):
        do_mixup = np.random.rand() < self.cfg.aug.mixup_prob
        do_cutmix = np.random.rand() < self.cfg.aug.cutmix_prob
        output: ModelOutput = self.model(batch["feature"], batch["label"], do_mixup, do_cutmix)
        loss = self.loss_fn(output.logits, output.labels)

        self.log(
            "train_loss",
            loss.detach().item(),
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        output = self.model.predict(batch["feature"], self.duration, batch["label"])
        loss = self.loss_fn(output.logits, output.labels)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )

        output.labels = self.model.correct_labels(output.labels, self.duration)

        self.validation_step_outputs.append(
            (
                batch["key"],
                output.labels.cpu().numpy(),
                output.preds.cpu().numpy(),
                float(loss.detach().cpu().numpy()),
            )
        )

        return loss

    def on_validation_epoch_end(self):
        keys = []
        for x in self.validation_step_outputs:
            keys.extend(x[0])
        preds = np.concatenate([x[2] for x in self.validation_step_outputs])

        val_pred_df: pd.DataFrame = post_process_for_seg(
            keys=keys,
            preds=preds[:, :, self.cfg.target_labels_idx],
            score_th=self.cfg.pp.score_th,
            distance=self.cfg.pp.distance,
        )
        score = event_detection_ap(self.val_event_df, val_pred_df)
        self.log("val_score", score, on_step=False, on_epoch=True, logger=True, prog_bar=True)

        self.validation_step_outputs.clear()

    def predict_step(self, batch, batch_idx) -> torch.Tensor:
        output = self.model.predict(batch["feature"], self.duration)
        return output.preds

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.cfg.optimizer.lr)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_training_steps=self.trainer.max_steps, **self.cfg.scheduler
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
