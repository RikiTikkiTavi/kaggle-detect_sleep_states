import logging
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
from hydra.types import RunMode
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    TQDMProgressBar,
    Callback
)
from pytorch_lightning.loggers import MLFlowLogger
from lightning.pytorch.accelerators import find_usable_cuda_devices

from detect_sleep_states.config import TrainConfig
from detect_sleep_states.data_module import SleepDataModule
from detect_sleep_states.model_module import PLSleepModel

import detect_sleep_states.plot.plot_predictions
import detect_sleep_states.introspect_model

from hydra.core.hydra_config import HydraConfig

_logger = logging.getLogger(__file__)


@hydra.main(config_path="../../config", config_name="train", version_base="1.2")
def main(cfg: TrainConfig):
    seed_everything(cfg.seed)

    # init data module
    datamodule = SleepDataModule(cfg)
    _logger.info(datamodule.train_event_df)

    _logger.info("Setting up DataModule ...")
    model = PLSleepModel(
        cfg=cfg,
        val_event_df=datamodule.valid_event_df,
        feature_dim=len(cfg.features),
        num_classes=len(cfg.labels),
        duration=cfg.duration
    )

    # set callbacks
    checkpoint_cb = ModelCheckpoint(
        verbose=True,
        monitor=cfg.trainer.monitor,
        mode=cfg.trainer.monitor_mode,
        save_top_k=1,
        save_last=False,
    )
    lr_monitor = LearningRateMonitor("epoch")
    progress_bar = TQDMProgressBar()
    model_summary = RichModelSummary(max_depth=2)

    # init experiment logger
    pl_logger = MLFlowLogger(
        experiment_name=cfg.exp_name,
        save_dir=cfg.dir.mlflow_store_dir,
        run_name=cfg.run_name,
        log_model=True
    )

    pl_logger.log_hyperparams(cfg)

    _logger.info("Setting up Trainer ...")

    hydra_config = HydraConfig.get()
    usable_gpu_devices = find_usable_cuda_devices(-1)
    if hydra_config.mode == RunMode.MULTIRUN:
        gpus = [
            usable_gpu_devices[hydra_config.job.num % len(usable_gpu_devices) + i]
            for i in range(cfg.trainer.gpus)
        ]
        _logger.info(f"Selected gpus {gpus} for job number {hydra_config.job.num}.")
    else:
        gpus = usable_gpu_devices[:cfg.trainer.gpus]
        _logger.info(f"Selected gpus {gpus}")

    trainer = Trainer(
        # env
        default_root_dir=Path.cwd(),
        # num_nodes=cfg.training.num_gpus,
        accelerator=cfg.trainer.accelerator,
        precision="16-mixed" if cfg.trainer.use_amp else "32-true",
        # training
        fast_dev_run=cfg.trainer.debug,  # run only 1 train batch and 1 val batch
        max_epochs=cfg.trainer.epochs,
        max_steps=cfg.trainer.epochs * len(datamodule.train_dataloader()),
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        callbacks=[checkpoint_cb, lr_monitor, progress_bar, model_summary],
        logger=pl_logger,
        # resume_from_checkpoint=resume_from,
        num_sanity_val_steps=0,
        log_every_n_steps=int(len(datamodule.train_dataloader()) * 0.1),
        sync_batchnorm=True,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        devices=gpus,
    )

    trainer.fit(model, datamodule=datamodule)

    if cfg.trainer.debug:
        return

    # load best weights
    _logger.info(f"Loading best weights: {checkpoint_cb.best_model_path}")
    best_model: PLSleepModel = PLSleepModel.load_from_checkpoint(
        checkpoint_cb.best_model_path,
        cfg=cfg,
        val_event_df=datamodule.valid_event_df,
        feature_dim=len(cfg.features),
        num_classes=len(cfg.labels),
        duration=cfg.duration,
    )
    pl_logger.log_metrics({"best_val_score": checkpoint_cb.best_model_score.item()})
    with mlflow.start_run(run_id=pl_logger.run_id) as _:
        mlflow.pytorch.log_state_dict(best_model.model.state_dict(), artifact_path="model")

    _logger.info("Introspecting model ...")
    detect_sleep_states.introspect_model.introspect_model(
        model_module=best_model,
        data_module=datamodule,
        cfg=cfg,
        logger=pl_logger,
        device=torch.device(type="cuda", index=gpus[0])
    )

    return


if __name__ == "__main__":
    main()
