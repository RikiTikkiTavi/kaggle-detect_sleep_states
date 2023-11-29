import logging
import shutil
from datetime import timedelta
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import polars
import polars as pl
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
    "enmo_total_variation"
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


def add_feature(
        series_df: pl.DataFrame,
        period: timedelta
) -> pl.DataFrame:
    series_df = (
        series_df
        .set_sorted(column="timestamp")
        .with_columns(
            *to_coord(pl.col("timestamp").dt.weekday(), 7, "day_of_week"),
            *to_coord(pl.col("timestamp").dt.hour(), 24, "hour"),
            *to_coord(pl.col("timestamp").dt.month(), 12, "month"),
            *to_coord(pl.col("timestamp").dt.minute(), 60, "minute"),
            pl.col("step") / pl.count("step"),
            pl.col('anglez_rad').sin().alias('anglez_sin'),
            pl.col('anglez_rad').cos().alias('anglez_cos'),
            *rolling_total_variation(series_df, period)
        )
        .select("series_id", *FEATURE_NAMES)
    )

    return series_df


def save_each_series(this_series_df: pl.DataFrame, columns: list[str], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    for col_name in columns:
        x = this_series_df.get_column(col_name).to_numpy(zero_copy_only=True)
        np.save(output_dir / f"{col_name}.npy", x)


def drop_dark_times(this_series_df: pl.DataFrame, window_size: int, th: int, df_train_events: pl.DataFrame):
    l1 = len(this_series_df)
    govno = this_series_df.get_column("anglez").to_pandas()
    govno_rolling = govno.rolling(FixedForwardWindowIndexer(window_size=window_size),
                                  min_periods=window_size).max().rename("anglez_forward_max")

    this_series_df = (
        this_series_df
        .with_columns(
            pl.col("anglez").rolling_max(window_size=window_size, min_periods=window_size).alias("anglez_backward_max"),
            pl.from_pandas(govno_rolling)
        )
        .join(df_train_events.select([pl.col("step", "awake")]), on="step", how="left")
        .with_columns(
            pl.col("awake").fill_null(strategy="backward"),
        )
    )
    this_series_df = (
        this_series_df
        .with_columns(
            pl
            .when((pl.col("anglez_backward_max") < th) | (pl.col("anglez_forward_max") < th))
            .then(2)
            .otherwise(pl.col("awake"))
            .alias("awake")
            .fill_null(1)
        )
        .select(
            pl.col("series_id"),
            pl.col("anglez"),
            pl.col("enmo"),
            pl.col("timestamp"),
            pl.col("step"),
            pl.col("anglez_rad"),
            pl.col("awake") != 2,
        )
        .drop(
            "awake"
        )
    )
    count_dropped = l1-len(this_series_df)
    if count_dropped > 0:
        _logger.info(f"Dropped {count_dropped} ({count_dropped/l1:.2f}) rows as dark times.")
    return this_series_df


@hydra.main(config_path="../../../config", config_name="prepare_data", version_base="1.2")
def main(cfg: PrepareDataConfig):
    processed_dir: Path = Path(cfg.dir.processed_dir) / cfg.phase

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
                    pl.col("timestamp"),
                    pl.col("step"),
                    pl.col("anglez_rad"),
                ]
            )
            .collect(streaming=True)
            .sort(by=["series_id", "timestamp"])
        )

        train_events_df = pl.from_pandas(
            pd.read_csv(Path(cfg.dir.data_dir) / "train_events.csv")
            .dropna()
            .astype({
                "step": np.uint32
            })
        ).with_columns(
            pl.col("event").map_dict({"onset": 1, "wakeup": 0}).alias("awake")
        )

        n_unique = series_df.get_column("series_id").n_unique()

    with trace("Save features"):
        for series_id, this_series_df in tqdm(series_df.group_by("series_id"), total=n_unique):
            train_events_df_series = train_events_df.filter(pl.col("series_id") == series_id)
            this_series_df = drop_dark_times(
                this_series_df,
                window_size=cfg.dark_drop_window_size,
                th=cfg.dark_drop_th,
                df_train_events=train_events_df_series
            )
            this_series_df = add_feature(this_series_df, period=timedelta(minutes=cfg.rolling_var_period))
            series_dir = processed_dir / series_id  # type: ignore
            save_each_series(this_series_df, FEATURE_NAMES, series_dir)


if __name__ == "__main__":
    main()
