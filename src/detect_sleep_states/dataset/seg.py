import random

import numpy as np
import pandas as pd
import polars as pl
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import resize

from detect_sleep_states.config import InferenceConfig, TrainConfig
from detect_sleep_states.utils.common import gaussian_label, nearest_valid_size, negative_sampling, random_crop
from detect_sleep_states.utils.common import pad_if_needed

###################
# Label
###################
def get_seg_label(
        this_event_df: pd.DataFrame, num_frames: int, duration: int, start: int, end: int
) -> np.ndarray:

    this_event_df = this_event_df.query("@start <= wakeup & onset <= @end")

    label = np.zeros((num_frames, 3))

    # onset, wakeup, sleep
    for onset, wakeup in this_event_df[["onset", "wakeup"]].to_numpy():
        onset = int((onset - start) / duration * num_frames)
        wakeup = int((wakeup - start) / duration * num_frames)
        if onset >= 0 and onset < num_frames:
            label[onset, 1] = 1
        if wakeup < num_frames and wakeup >= 0:
            label[wakeup, 2] = 1

        onset = max(0, onset)
        wakeup = min(num_frames, wakeup)
        label[onset:wakeup, 0] = 1  # sleep

    return label


class SegTrainDataset(Dataset):
    def __init__(
            self,
            cfg: TrainConfig,
            features: dict[str, np.ndarray],
            event_df: pl.DataFrame,
    ):
        self.cfg = cfg
        # id | night | onset | wakeup
        self.event_df: pd.DataFrame = (
            event_df.pivot(index=["series_id", "night"], columns="event", values="step", aggregate_function="first")
            .drop_nulls()
            .to_pandas()
        )
        self.features = features
        self.num_features = len(cfg.features)
        self.upsampled_num_frames = nearest_valid_size(
            int(self.cfg.duration * self.cfg.upsample_rate), self.cfg.downsample_rate
        )

    def __len__(self):
        return len(self.event_df)

    def __getitem__(self, idx):
        # Choose event
        event = np.random.choice(["onset", "wakeup"], p=[0.5, 0.5])
        # Get step of the selected event
        pos = self.event_df.at[idx, event]
        # Get series id of the selected event
        series_id = self.event_df.at[idx, "series_id"]
        # Get events corresponding to chosen series_id
        this_event_df = self.event_df.query("series_id == @series_id").reset_index(drop=True)
        # Extract data matching series_id
        this_feature = self.features[series_id]  # (n_steps, num_features)
        # Length of series
        n_steps = this_feature.shape[0]

        # sample background
        # Sample step which is not an event
        if random.random() < self.cfg.dataset.bg_sampling_rate:
            pos = negative_sampling(this_event_df, n_steps)

        # If series is longer then defined duration, crop in a way
        # that pos is contained in the interval
        if n_steps > self.cfg.duration:
            start, end = random_crop(pos, self.cfg.duration, n_steps)
            feature = this_feature[start:end]
        # Otherwsie pad the sequence
        else:
            start, end = 0, self.cfg.duration
            feature = pad_if_needed(this_feature, self.cfg.duration)

        # upsample
        feature = torch.FloatTensor(feature.T).unsqueeze(0)  # (1, num_features, duration)
        feature = resize(
            feature,
            size=[self.num_features, self.upsampled_num_frames],
            antialias=False,
        ).squeeze(0)

        # from hard label to gaussian label
        num_frames = self.upsampled_num_frames // self.cfg.downsample_rate
        label = get_seg_label(this_event_df, num_frames, self.cfg.duration, start, end)
        label[:, [1, 2]] = gaussian_label(
            label[:, [1, 2]],
            sigma=self.cfg.dataset.sigma,
            radius=self.cfg.dataset.radius
        )

        return {
            "series_id": series_id,
            "feature": feature,  # (num_features, upsampled_num_frames)
            "label": torch.FloatTensor(label),  # (pred_length, num_classes)
        }


class SegValidDataset(Dataset):
    def __init__(
            self,
            cfg: TrainConfig,
            chunk_features: dict[str, np.ndarray],
            event_df: pl.DataFrame,
    ):
        self.cfg = cfg
        self.chunk_features = chunk_features
        self.keys = list(chunk_features.keys())
        self.event_df = (
            event_df.pivot(index=["series_id", "night"], columns="event", values="step")
            .drop_nulls()
            .to_pandas()
        )
        self.num_features = len(cfg.features)
        self.upsampled_num_frames = nearest_valid_size(
            int(self.cfg.duration * self.cfg.upsample_rate), self.cfg.downsample_rate
        )

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        feature = self.chunk_features[key]
        feature = torch.FloatTensor(feature.T).unsqueeze(0)  # (1, num_features, duration)
        feature = resize(
            feature,
            size=[self.num_features, self.upsampled_num_frames],
            antialias=False,
        ).squeeze(0)

        series_id, chunk_id = key.split("_")
        chunk_id = int(chunk_id)
        start = chunk_id * self.cfg.duration
        end = start + self.cfg.duration
        num_frames = self.upsampled_num_frames // self.cfg.downsample_rate
        label = get_seg_label(
            self.event_df.query("series_id == @series_id").reset_index(drop=True),
            num_frames,
            self.cfg.duration,
            start,
            end,
        )
        return {
            "key": key,
            "feature": feature,  # (num_features, duration)
            "label": torch.FloatTensor(label),  # (duration, num_classes)
        }


class SegTestDataset(Dataset):
    def __init__(
            self,
            cfg: InferenceConfig,
            chunk_features: dict[str, np.ndarray],
    ):
        self.cfg = cfg
        self.chunk_features = chunk_features
        self.keys = list(chunk_features.keys())
        self.num_features = len(cfg.features)
        self.upsampled_num_frames = nearest_valid_size(
            int(self.cfg.duration * self.cfg.upsample_rate), self.cfg.downsample_rate
        )

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        feature = self.chunk_features[key]
        feature = torch.FloatTensor(feature.T).unsqueeze(0)  # (1, num_features, duration)
        feature = resize(
            feature,
            size=[self.num_features, self.upsampled_num_frames],
            antialias=False,
        ).squeeze(0)

        return {
            "key": key,
            "feature": feature,  # (num_features, duration)
        }
