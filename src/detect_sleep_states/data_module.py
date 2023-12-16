import logging
from pathlib import Path
from typing import Optional

import numpy as np
import polars
import polars as pl
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from detect_sleep_states.config import TrainConfig
from detect_sleep_states.dataset.common import get_train_ds, get_valid_ds
from detect_sleep_states.utils.common import pad_if_needed

_logger = logging.getLogger(__name__)


def load_features(
        feature_names: list[str],
        series_ids: Optional[list[str]],
        processed_dir: Path,
        phase: str,
) -> dict[str, np.ndarray]:
    features = {}

    if series_ids is None:
        series_ids = [series_dir.name for series_dir in (processed_dir / phase).glob("*")]

    for series_id in series_ids:
        series_dir = processed_dir / phase / series_id
        this_feature = []
        for feature_name in feature_names:
            this_feature.append(np.load(series_dir / f"{feature_name}.npy"))
        features[series_dir.name] = np.stack(this_feature, axis=1)

    return features


def load_chunk_features(
        duration: int,
        feature_names: list[str],
        series_ids: Optional[list[str]],
        processed_dir: Path,
        phase: str,
) -> dict[str, np.ndarray]:
    features = {}

    if series_ids is None:
        series_ids = [series_dir.name for series_dir in (processed_dir / phase).glob("*")]

    for series_id in series_ids:
        series_dir = processed_dir / phase / series_id
        this_feature = []
        for feature_name in feature_names:
            this_feature.append(np.load(series_dir / f"{feature_name}.npy"))
        this_feature = np.stack(this_feature, axis=1)
        num_chunks = (len(this_feature) // duration) + 1
        for i in range(num_chunks):
            chunk_feature = this_feature[i * duration: (i + 1) * duration]
            chunk_feature = pad_if_needed(chunk_feature, duration, pad_value=0)  # type: ignore
            features[f"{series_id}_{i:07}"] = chunk_feature

    return features  # type: ignore


class SleepDataModule(LightningDataModule):
    cfg: TrainConfig
    train_event_df: polars.DataFrame
    valid_event_df: polars.DataFrame
    train_features: dict[str, np.ndarray]
    valid_chunk_features: dict[str, np.ndarray]

    def __init__(
            self,
            cfg: TrainConfig,
            train_event_df: polars.DataFrame,
            valid_event_df: polars.DataFrame,
            train_features: dict[str, np.ndarray],
            valid_chunk_features: dict[str, np.ndarray]
    ):
        super().__init__()
        self.cfg = cfg
        self.data_dir = Path(cfg.dir.data_dir)
        self.processed_dir = Path(cfg.dir.processed_dir)
        self.train_event_df = train_event_df
        self.valid_event_df = valid_event_df
        self.train_features = train_features
        self.valid_chunk_features = valid_chunk_features

    @staticmethod
    def from_config(cfg: TrainConfig) -> "SleepDataModule":
        processed_dir = Path(cfg.dir.processed_dir)
        event_df = pl.read_csv(processed_dir / "train_events.csv").drop_nulls()
        train_event_df = event_df.filter(
            pl.col("series_id").is_in(cfg.split.train_series_ids)
        )
        valid_event_df = event_df.filter(
            pl.col("series_id").is_in(cfg.split.valid_series_ids)
        )
        train_features = load_features(
            feature_names=cfg.features,
            series_ids=cfg.split.train_series_ids,
            processed_dir=processed_dir,
            phase="train",
        )
        valid_chunk_features = load_chunk_features(
            duration=cfg.duration,
            feature_names=cfg.features,
            series_ids=cfg.split.valid_series_ids,
            processed_dir=processed_dir,
            phase="train",
        )
        return SleepDataModule(
            cfg=cfg,
            train_event_df=train_event_df,
            valid_event_df=valid_event_df,
            train_features=train_features,
            valid_chunk_features=valid_chunk_features
        )

    def train_dataloader(self):
        train_dataset = get_train_ds(
            cfg=self.cfg,
            event_df=self.train_event_df,
            features=self.train_features,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.dataset.batch_size,
            shuffle=True,
            num_workers=self.cfg.dataset.num_workers,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
            multiprocessing_context=self.cfg.dataset.multiprocessing_context,
        )
        return train_loader

    def val_dataloader(self):
        valid_dataset = get_valid_ds(
            cfg=self.cfg,
            event_df=self.valid_event_df,
            chunk_features=self.valid_chunk_features,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=self.cfg.dataset.batch_size,
            shuffle=False,
            num_workers=self.cfg.dataset.num_workers,
            pin_memory=True,
            persistent_workers=True,
            multiprocessing_context=self.cfg.dataset.multiprocessing_context,
        )
        return valid_loader
