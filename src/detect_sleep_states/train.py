import logging
from pathlib import Path

import hydra
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import MLFlowLogger

from detect_sleep_states.config import TrainConfig
from detect_sleep_states.data_module import SleepDataModule
from detect_sleep_states.model_module import PLSleepModel

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s:%(name)s - %(message)s"
)
_logger = logging.getLogger(__file__)


@hydra.main(config_path="../../config", config_name="train", version_base="1.2")
def main(cfg: TrainConfig):
    seed_everything(cfg.seed)

    # init lightning model
    datamodule = SleepDataModule(cfg)
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
        log_model=True,
        artifact_location=str(Path.cwd())
    )

    pl_logger.log_hyperparams(cfg)

    _logger.info("Setting up Trainer ...")
    trainer = Trainer(
        # env
        default_root_dir=Path.cwd(),
        # num_nodes=cfg.training.num_gpus,
        accelerator=cfg.trainer.accelerator,
        precision=16 if cfg.trainer.use_amp else 32,
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
    )

    trainer.fit(model, datamodule=datamodule)

    # load best weights
    _logger.info(f"Loading best weights: {checkpoint_cb.best_model_path}")
    model = PLSleepModel.load_from_checkpoint(
        checkpoint_cb.best_model_path,
        cfg=cfg,
        val_event_df=datamodule.valid_event_df,
        feature_dim=len(cfg.features),
        num_classes=len(cfg.labels),
        duration=cfg.duration,
    )
    weights_path = Path.cwd() / "model_weights.pth"
    _logger.info(f"Saving best weights: {weights_path} ...")
    torch.save(model.model.state_dict(), weights_path)

    return


if __name__ == "__main__":
    main()
