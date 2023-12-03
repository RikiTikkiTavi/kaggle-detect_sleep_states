import logging
import shutil
from datetime import timedelta
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import polars
import polars as pl
from numba import jit
from pandas.core.indexers.objects import FixedForwardWindowIndexer
from tqdm import tqdm

from detect_sleep_states.config import PrepareDataConfig
from detect_sleep_states.utils.common import trace

SERIES_SCHEMA = {
    "series_id": pl.Utf8,
    "step": pl.UInt32,
    "anglez": pl.Float32,
    "enmo": pl.Float32,
}

FEATURE_NAMES = [
    "anglez",
    "enmo",
    "step",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "minute_sin",
    "minute_cos",
    "anglez_sin",
    "anglez_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "anglez_total_variation",
    "enmo_total_variation",
    "is_removed"
]

ANGLEZ_MEAN = -8.810476
ANGLEZ_STD = 35.521877
ENMO_MEAN = 0.041315
ENMO_STD = 0.101829

_logger = logging.getLogger(__name__)


def to_coord(x: pl.Expr, max_: int, name: str) -> list[pl.Expr]:
    rad = 2 * np.pi * x / max_
    x_sin = rad.sin()
    x_cos = rad.cos()

    return [x_sin.alias(f"{name}_sin"), x_cos.alias(f"{name}_cos")]


def deg_to_rad(x: pl.Expr) -> pl.Expr:
    return np.pi / 180 * x


def total_variation(x: pl.Expr) -> pl.Expr:
    return x.diff().slice(offset=1).abs().sum()


def rolling_total_variation(series_df: pl.DataFrame, period: timedelta) -> tuple[pl.Series, pl.Series]:
    df_series_rolling_var = (
        series_df
        .set_sorted("timestamp")
        .rolling(index_column="timestamp", period=period)
        .agg([
            total_variation(pl.col("anglez")).alias("anglez_total_variation"),
            total_variation(pl.col("enmo")).alias("enmo_total_variation")
        ])
    )
    return (df_series_rolling_var.get_column("anglez_total_variation"),
            df_series_rolling_var.get_column("enmo_total_variation"))


@jit(nopython=True)
def calculate_removed_periods(
        anglez: np.ndarray,
        window_size: int,
        step: int
) -> np.ndarray:
    g_size = window_size
    step = step
    is_removed = np.repeat(False, len(anglez))
    for first_window_start in range(2 * g_size, len(anglez) - g_size, g_size):

        first_window_end = first_window_start + g_size

        for second_window_start in range(0, first_window_start - g_size, step):

            second_window_end = second_window_start + g_size

            if np.all(
                    anglez[first_window_start:first_window_end] == anglez[second_window_start:second_window_end]):
                is_removed[first_window_start:first_window_end] = True
                is_removed[second_window_start:second_window_end] = True
    return is_removed


def add_feature(
        series_df: pl.DataFrame,
        cfg: PrepareDataConfig,
        period: timedelta,
        is_removed_window_size: int,
        is_removed_step_size: int
) -> pl.DataFrame:

    feature_expressions = []

    if "day_of_week_sin" in cfg.features and "day_of_week_cos" in cfg.features:
        feature_expressions.extend(
            to_coord(pl.col("timestamp").dt.weekday(), 7, "day_of_week")
        )
    if "hour_sin" in cfg.features and "hour_cos" in cfg.features:
        feature_expressions.extend(
            to_coord(pl.col("timestamp").dt.hour(), 24, "hour")
        )
    if "minute_sin" in cfg.features and "minute_cos" in cfg.features:
        feature_expressions.extend(
            to_coord(pl.col("timestamp").dt.minute(), 60, "minute")
        )
    if "is_removed" in cfg.features:
        is_removed = calculate_removed_periods(
            anglez=series_df.get_column("anglez").to_numpy(),
            window_size=is_removed_window_size,
            step=is_removed_step_size
        ).astype(int)
        feature_expressions.append(pl.Series(name="is_removed", values=is_removed))
    if "anglez_total_variation" in cfg.features and "enmo_total_variation" in cfg.features:
        feature_expressions.extend(rolling_total_variation(series_df, period))

    series_df = (
        series_df
        .set_sorted(column="timestamp")
        .with_columns(*feature_expressions)
        .select("series_id", *cfg.features)
    )

    return series_df


def save_each_series(this_series_df: pl.DataFrame, columns: list[str], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    for col_name in columns:
        x = this_series_df.get_column(col_name).to_numpy()
        np.save(output_dir / f"{col_name}.npy", x)


def drop_dark_times(this_series_df: pl.DataFrame, window_size: int, th: int, df_train_events: pd.DataFrame):
    train_series = this_series_df.to_pandas()
    l1 = train_series.shape[0]
    train_events = df_train_events

    # cleaning etc.
    train_events = train_events.dropna()
    train_events["step"] = train_events["step"].astype("int")
    train_events["awake"] = train_events["event"].replace({"onset": 1, "wakeup": 0})

    train = pd.merge(train_series, train_events[['step', 'awake']], on='step', how='left')
    train["awake"] = train["awake"].bfill(axis='rows')

    train['anglez_backward_max'] = train_series['anglez'].rolling(window=window_size, min_periods=window_size).max()
    train['anglez_forward_max'] = train_series['anglez'].rolling(
        FixedForwardWindowIndexer(window_size=window_size),
        min_periods=window_size
    ).max()
    cond = (train.anglez_backward_max < th) | (train.anglez_forward_max < th)
    train.loc[cond, 'awake'] = 2
    # Result: the last event is always a "wakeup"
    train['awake'] = train['awake'].fillna(1)  # awake
    train["awake"] = train["awake"].astype("int")
    train = train.loc[train["awake"] != 2].drop(["awake", 'anglez_backward_max', 'anglez_forward_max'], axis="columns")
    l2 = len(train)
    if l1 - l2 > 0:
        _logger.info(f"Dropped {l1 - l2} ({(l1 - l2) / l1}) rows")
    return polars.from_pandas(train)


def process_events_df(df_events: pd.DataFrame) -> pd.DataFrame:
    # *** Modify some Events ***
    #
    # --- 655f19eabf1e : Remove the first night 3 onset:
    train_events = df_events.drop([5362], axis=0)
    #
    # --- 8a306e0890c0 : Remove the extra night 11 onset
    train_events = train_events.drop([7358], axis=0)
    #
    # --- c3072a759efb : Insert the missing onset for night 20, at 338000
    # Want it at index 10132
    # Add 1 to current 10132 and ones above, put it where desired using float index:
    train_events.loc[10131.5] = ['c3072a759efb', 20, 'onset', 339000,
                                 '2018-03-18T23:00:00-0500']
    # Re-do integer indices
    train_events = train_events.sort_index().reset_index(drop=True)
    #
    # --- c6788e579967 : Remove the extra onset for night 20
    train_events = train_events.drop([10348], axis=0)
    # Re-do integer indices
    train_events = train_events.sort_index().reset_index(drop=True)

    df_events.dropna(inplace=True)

    return train_events


@hydra.main(config_path="../../../config", config_name="prepare_data", version_base="1.2")
def main(cfg: PrepareDataConfig):
    processed_dir: Path = Path(cfg.dir.processed_dir) / cfg.phase

    if cfg.phase == "train":
        train_events_df = process_events_df(pd.read_csv(Path(cfg.dir.data_dir) / "train_events.csv"))
        train_events_df.to_csv(Path(cfg.dir.processed_dir) / "train_events.csv")

    if processed_dir.exists():
        shutil.rmtree(processed_dir)
        print(f"Removed {cfg.phase} dir: {processed_dir}")

    with trace("Load series"):
        # scan parquet
        if cfg.phase in ["train", "test"]:
            series_lf = pl.scan_parquet(
                Path(cfg.dir.data_dir) / f"{cfg.phase}_series.parquet",
                low_memory=True,
            )
        elif cfg.phase == "dev":
            series_lf = pl.scan_parquet(
                Path(cfg.dir.processed_dir) / f"{cfg.phase}_series.parquet",
                low_memory=True,
            )
        else:
            raise ValueError(f"Invalid phase: {cfg.phase}")

        # preprocess
        series_df = (
            series_lf
            .with_columns(
                pl.col("timestamp").str.to_datetime("%Y-%m-%dT%H:%M:%S%z"),
                deg_to_rad(pl.col("anglez")).alias("anglez_rad"),
                (pl.col("anglez") - ANGLEZ_MEAN) / ANGLEZ_STD,
                (pl.col("enmo") - ENMO_MEAN) / ENMO_STD,
            )
            .select(
                [
                    pl.col("series_id"),
                    pl.col("anglez"),
                    pl.col("enmo"),
                    pl.col("step"),
                    pl.col("timestamp"),
                    pl.col("anglez_rad"),
                ]
            )
            .collect(streaming=True)
            .sort(by=["series_id", "timestamp"])
        )
        n_unique = series_df.get_column("series_id").n_unique()

    with trace("Save features"):
        for series_id, this_series_df in tqdm(series_df.group_by("series_id"), total=n_unique):
            this_series_df = add_feature(
                this_series_df,
                cfg=cfg,
                period=timedelta(minutes=cfg.rolling_var_period),
                is_removed_step_size=2,
                is_removed_window_size=720
            )
            series_dir = processed_dir / series_id  # type: ignore
            save_each_series(this_series_df, cfg.features, series_dir)


if __name__ == "__main__":
    main()
