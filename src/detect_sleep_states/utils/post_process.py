import numpy as np
import pandas as pd
import polars as pl
from scipy.signal import find_peaks


def extract_largest(df: pd.DataFrame) -> pd.Series:
    return df.loc[df["score"].idxmax()]


def aggregate_series_events_df(df: pd.DataFrame, window_size=360) -> pd.DataFrame:
    return df.groupby([df["series_id"], df["step"] // window_size], as_index=False).apply(extract_largest)


def post_process_for_seg(
        keys: list[str],
        preds: np.ndarray,
        score_th: float = 0.01,
        distance: int = 5000
) -> pd.DataFrame:
    """make submission dataframe for segmentation task

    Args:
        keys (list[str]): list of keys. key is "{series_id}_{chunk_id}"
        preds (np.ndarray): (num_series * num_chunks, duration, 2)
        score_th (float, optional): threshold for score. Defaults to 0.5.

    Returns:
        pl.DataFrame: submission dataframe
    """
    series_ids = np.array(list(map(lambda x: x.split("_")[0], keys)))
    unique_series_ids = np.unique(series_ids)

    sub_dfs = []
    for series_id in unique_series_ids:
        series_idx = np.where(series_ids == series_id)[0]
        this_series_preds = preds[series_idx].reshape(-1, 2)
        series_records = []

        events_count = 0
        for i, event_name in enumerate(["onset", "wakeup"]):
            this_event_preds = this_series_preds[:, i]
            steps = find_peaks(this_event_preds, height=score_th, distance=distance)[0]
            scores = this_event_preds[steps]

            for step, score in zip(steps, scores):
                series_records.append(
                    {
                        "series_id": series_id,
                        "step": step,
                        "event": event_name,
                        "score": score,
                    }
                )
                events_count += 1

        if len(series_records) == 0:
            continue

        series_sub_df = pd.DataFrame(series_records)

        events_count_thr = len(preds) // 2
        window_size = 30
        step = 0

        while True:
            if events_count < events_count_thr:
                break
            else:
                series_sub_df = aggregate_series_events_df(series_sub_df, window_size=window_size + step)
                events_count = len(series_sub_df)

            step += 10

        sub_dfs.append(series_sub_df)

    if len(sub_dfs) == 0:
        sub_dfs.append(
            pd.DataFrame([{
                "series_id": series_id,
                "step": 0,
                "event": "onset",
                "score": 0.0,
            }])
        )

    sub_df = pd.concat(sub_dfs)
    sub_df.sort_values(by=["series_id", "step"], inplace=True)
    sub_df["row_id"] = np.arange(len(sub_df))
    # https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/discussion/458274
    # return sub_df.groupby(by=['series_id', 'event']).head(50)
    return sub_df[["row_id", "series_id", "step", "event", "score"]]
