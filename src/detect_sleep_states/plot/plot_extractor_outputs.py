import numpy as np
from matplotlib import pyplot as plt

import detect_sleep_states.config
import detect_sleep_states.plot.plot_predictions
import seaborn as sns


def plot_cnn_extractor_outputs_chunk(
        predictions: np.ndarray,
        features: np.ndarray,
        labels: np.ndarray,
        extractor_outputs: np.ndarray,  # shape (n channels, height, timesteps)
        cfg: detect_sleep_states.config.TrainConfig
):
    sns.set_style("darkgrid")

    fig: plt.Figure
    ax: plt.Axes
    n_rows = extractor_outputs.shape[0] + 1
    height = n_rows * 6
    fig, axs = plt.subplots(figsize=(12, height), ncols=1, nrows=n_rows)
    axs = axs.flatten()

    detect_sleep_states.plot.plot_predictions.plot_predictions_chunk(
        predictions=predictions,
        features=features,
        labels=labels,
        cfg=cfg,
        ax=axs[0]
    )

    for ax, channel_output in zip(axs[1:], extractor_outputs):
        ax.imshow(channel_output, cmap="gray")

    return fig
