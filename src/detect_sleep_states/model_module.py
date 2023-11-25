import logging
from pathlib import Path
from typing import Optional

import mlflow
import numpy as np
import polars as pl
import torch
import torch.optim as optim
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import MLFlowLogger
from transformers import get_cosine_schedule_with_warmup

from detect_sleep_states.config import TrainConfig
from detect_sleep_states.models.base import ModelOutput
from detect_sleep_states.models.common import get_model
from detect_sleep_states.loss.common import get_loss
from detect_sleep_states.utils.common import nearest_valid_size
from detect_sleep_states.utils.metrics import event_detection_ap
from detect_sleep_states.utils.post_process import post_process_for_seg
from detect_sleep_states.plot.plot_predictions import plot_predictions_chunk

import detect_sleep_states.data_module

_logger = logging.getLogger(__name__)


class PLSleepModel(LightningModule):
    def __init__(
            self,
            cfg: TrainConfig,
            val_event_df: pl.DataFrame,
            feature_dim: int,
            num_classes: int,
            duration: int
    ):
        super().__init__()
        self.cfg = cfg
        self.val_event_df = val_event_df
        num_timesteps = nearest_valid_size(int(duration * cfg.upsample_rate), cfg.downsample_rate)
        self.model = get_model(
            cfg,
            feature_dim=feature_dim,
            n_classes=num_classes,
            num_timesteps=num_timesteps // cfg.downsample_rate,
        )
        self.loss_fn = get_loss(cfg.loss)
        self.duration = duration
        self.validation_step_outputs: list = []
        self.__best_loss = np.inf


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

        output.labels = self.model.correct_labels(output.labels, self.duration)

        self.validation_step_outputs.append(
            (
                batch["key"],
                output.labels.detach().cpu().numpy(),
                output.preds.detach().cpu().numpy(),
                loss.detach().item(),
            )
        )
        self.log(
            "val_loss",
            loss.detach().item(),
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )

        return loss

    def on_validation_epoch_end(self):
        keys = []
        for x in self.validation_step_outputs:
            keys.extend(x[0])
        labels = np.concatenate([x[1] for x in self.validation_step_outputs])
        preds = np.concatenate([x[2] for x in self.validation_step_outputs])
        losses = np.array([x[3] for x in self.validation_step_outputs])
        loss = losses.mean()

        val_pred_df = post_process_for_seg(
            keys=keys,
            preds=preds[:, :, self.cfg.target_labels_idx],
            score_th=self.cfg.pp.score_th,
            distance=self.cfg.pp.distance,
        )
        score = event_detection_ap(self.val_event_df.to_pandas(), val_pred_df.to_pandas())
        self.log("val_score", score, on_step=False, on_epoch=True, logger=True, prog_bar=True)

        if loss < self.__best_loss:
            np.save("keys.npy", np.array(keys))
            np.save("labels.npy", labels)
            np.save("preds.npy", preds)
            val_pred_df.write_csv("val_pred_df.csv")
            torch.save(self.model.state_dict(), "best_model.pth")

            logger: MLFlowLogger = self.logger
            logger.experiment.log_artifact(
                run_id=logger.run_id,
                local_path="keys.npy",
                artifact_path="best/keys.npy"
            )
            logger.experiment.log_artifact(
                run_id=logger.run_id,
                local_path="labels.npy",
                artifact_path="best/labels.npy"
            )
            logger.experiment.log_artifact(
                run_id=logger.run_id,
                local_path="preds.npy",
                artifact_path="best/preds.npy"
            )
            logger.experiment.log_artifact(
                run_id=logger.run_id,
                local_path="val_pred_df.csv",
                artifact_path="best/val_pred_df.csv"
            )
            self.__best_loss = loss

        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.cfg.optimizer.lr)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_training_steps=self.trainer.max_steps, **self.cfg.scheduler
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
