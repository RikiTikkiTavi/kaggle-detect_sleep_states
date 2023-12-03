import logging
from typing import Any

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

import detect_sleep_states.model_module
import detect_sleep_states.data_module
import detect_sleep_states.config
import detect_sleep_states.plot.plot_predictions
import detect_sleep_states.plot.plot_extractor_outputs
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger

from detect_sleep_states.utils.metrics import event_detection_ap
from detect_sleep_states.utils.post_process import post_process_for_seg

_logger = logging.getLogger(__name__)


def visualize_predictions(
        val_pred_df: pd.DataFrame,
        cfg: detect_sleep_states.config.TrainConfig,
        valid_chunk_features: dict[str, Any],
        keys: np.array,
        labels: np.ndarray,
        predictions: np.ndarray,
        logger: MLFlowLogger
):
    df_keys = pd.DataFrame(
        np.char.split(keys, "_").tolist(),
        columns=["series_id", "chunk_id"]
    ).reset_index(names=["key_i"])

    for n_series, series_id in enumerate(val_pred_df.groupby("series_id")["score"].sum().sort_values().index):

        if n_series >= cfg.n_chunks_visualize:
            break

        for i in df_keys.loc[df_keys["series_id"] == series_id]["key_i"]:
            fig, ax = detect_sleep_states.plot.plot_predictions.plot_predictions_chunk(
                predictions=predictions[i],
                features=valid_chunk_features[keys[i]],
                labels=labels[i],
                cfg=cfg
            )
            if np.any(labels[i][:, 1] > 1 / 2):
                artifact_folder = "plots/val/predictions/onset"
            elif np.any(labels[i][:, 2] > 1 / 2):
                artifact_folder = "plots/val/predictions/wakeup"
            else:
                artifact_folder = "plots/val/predictions/bg"

            logger.experiment.log_figure(
                run_id=logger.run_id,
                figure=fig,
                artifact_file=f"{artifact_folder}/{n_series}-{keys[i]}.png"
            )
            plt.close(fig)


def visualize_extractor(
        val_pred_df: pd.DataFrame,
        cfg: detect_sleep_states.config.TrainConfig,
        valid_chunk_features: dict[str, Any],
        keys: np.array,
        labels: np.ndarray,
        predictions: np.ndarray,
        extractor_outputs: np.ndarray,
        logger: MLFlowLogger
):
    df_keys = pd.DataFrame(
        np.char.split(keys, "_").tolist(),
        columns=["series_id", "chunk_id"]
    ).reset_index(names=["key_i"])

    for n_series, series_id in enumerate(val_pred_df.groupby("series_id")["score"].sum().sort_values().index):

        if n_series >= cfg.n_chunks_visualize:
            break

        for i in df_keys.loc[df_keys["series_id"] == series_id]["key_i"]:

            fig = detect_sleep_states.plot.plot_extractor_outputs.plot_cnn_extractor_outputs_chunk(
                predictions=predictions[i],
                features=valid_chunk_features[keys[i]],
                labels=labels[i],
                extractor_outputs=extractor_outputs[i],
                cfg=cfg
            )

            if np.any(labels[i][:, 1] > 1 / 2):
                artifact_folder = "plots/val/extractor/onset"
            elif np.any(labels[i][:, 2] > 1 / 2):
                artifact_folder = "plots/val/extractor/wakeup"
            else:
                artifact_folder = "plots/val/extractor/bg"

            logger.experiment.log_figure(
                run_id=logger.run_id,
                figure=fig,
                artifact_file=f"{artifact_folder}/{n_series}-{keys[i]}.png"
            )
            plt.close("all")


def introspect_model(
        model_module: detect_sleep_states.model_module.PLSleepModel,
        data_module: detect_sleep_states.data_module.SleepDataModule,
        cfg: detect_sleep_states.config.TrainConfig,
        logger: MLFlowLogger,
        device: torch.device,
):
    keys = []
    labels = []
    predictions = []
    extractor_outputs = []

    model_module.eval()

    _logger.info("Predicting for introspection ...")
    for batch in data_module.val_dataloader():

        batch_x = batch["feature"].to(device)
        batch_y = batch["label"].to(device)

        batch_output = model_module.model.predict(
            x=batch_x,
            org_duration=cfg.duration,
            labels=batch_y
        )

        if hasattr(model_module.model, "feature_extractor"):
            with torch.no_grad():
                extractor_output = model_module.model.feature_extractor(batch_x)
                extractor_outputs.append(extractor_output.numpy(force=True))

        batch_output.labels = model_module.model.correct_labels(batch_output.labels, cfg.duration)

        keys.extend(batch["key"])
        labels.extend(batch_output.labels.tolist())
        predictions.extend(batch_output.preds.tolist())

    predictions = np.array(predictions)
    labels = np.array(labels)
    extractor_outputs = np.concatenate(extractor_outputs, axis=0)

    _logger.info("Calculating val_pred_df ...")
    val_pred_df: pd.DataFrame = post_process_for_seg(
        keys=keys,
        preds=predictions[:, :, cfg.target_labels_idx],
        score_th=cfg.pp.score_th,
        distance=cfg.pp.distance,
    ).to_pandas()

    keys = np.array(keys)

    _logger.info("Plotting predictions ...")
    visualize_predictions(
        val_pred_df=val_pred_df,
        cfg=cfg,
        valid_chunk_features=data_module.valid_chunk_features,
        keys=keys,
        labels=labels,
        predictions=predictions,
        logger=logger
    )

    if hasattr(model_module.model, "feature_extractor"):
        _logger.info("Plotting extractor results ...")
        visualize_extractor(
            val_pred_df=val_pred_df,
            cfg=cfg,
            keys=keys,
            labels=labels,
            predictions=predictions,
            extractor_outputs=extractor_outputs,
            valid_chunk_features=data_module.valid_chunk_features,
            logger=logger,
        )

    _logger.info("Finalizing ...")
    plt.close('all')
