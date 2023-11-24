import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.patheffects as pe
import detect_sleep_states.utils.common
import detect_sleep_states.config


def plot_predictions_chunk(
        predictions: np.ndarray,
        features: np.ndarray,
        labels: np.ndarray,
        cfg: detect_sleep_states.config.TrainConfig
):
    sns.set_style("darkgrid")

    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = sns.color_palette()

    anglez = features[:, 0]
    enmo = features[:, 1]

    ax.plot(enmo, label="enmo", color=colors[0], alpha=0.5)

    # ax.set_xticklabels([i * 5 / 3600 for i in range(len(enmo))])

    ax_anglez: plt.Axes = ax.twinx()
    ax_anglez.plot(anglez, label="anglez", color=colors[1], alpha=0.5)
    ax_anglez.set_yticks([])

    ax_prob: plt.Axes = ax.twinx()
    ax_prob.plot(
        predictions[:, 0],
        label="predictions[onset]",
        color=colors[2],
        linewidth=3,
        alpha=0.8
    )
    ax_prob.plot(predictions[:, 1], label="predictions[awake]", color=colors[3], linewidth=3, alpha=0.8)

    chunk_labels_smoothed = detect_sleep_states.utils.common.gaussian_label(
        labels[:, :],
        sigma=cfg.dataset.sigma,
        radius=cfg.dataset.radius
    )
    ax_prob.plot(
        chunk_labels_smoothed[:, 0],
        label="labels[onset]",
        color=colors[2],
        linestyle="--",
        linewidth=3,
        path_effects=[pe.Stroke(linewidth=3, foreground='white'), pe.Normal()]
    )
    ax_prob.plot(
        chunk_labels_smoothed[:, 1],
        label="labels[awake]",
        color=colors[3],
        linestyle=":",
        linewidth=3,
        path_effects=[pe.Stroke(linewidth=3, foreground='white'), pe.Normal()]
    )

    ax_prob.legend(loc="upper left")
    ax_anglez.legend(loc="upper center")
    ax.legend(loc="upper right")

    return fig, ax
