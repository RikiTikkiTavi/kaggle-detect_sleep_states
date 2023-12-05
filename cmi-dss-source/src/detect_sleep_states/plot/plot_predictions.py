import typing

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
        cfg: detect_sleep_states.config.TrainConfig,
        ax: typing.Optional[plt.Axes] = None
) -> tuple[plt.Figure, plt.Axes]:
    sns.set_style("darkgrid")

    fig: plt.Figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.get_figure()

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
        label="predictions[asleep]",
        color=colors[4],
        linewidth=2.5,
        alpha=0.8
    )
    ax_prob.plot(
        predictions[:, 1],
        label="predictions[onset]",
        color=colors[2],
        linewidth=2.5,
        alpha=1
    )
    ax_prob.plot(
        predictions[:, 2],
        label="predictions[wakeup]",
        color=colors[3],
        linewidth=2.5,
        alpha=1
    )
    ax_prob.set_ylim((-1.05, 1.05))

    labels[:, 0][labels[:, 0] == 1] = -1  # asleep
    labels[:, 0][labels[:, 0] == 0] = 1  # awake
    ax_prob.plot(
        labels[:, 0],
        color="black",
        linestyle="--",
        linewidth=3,
        label="labels"
    )

    ax_prob.axhline(
        y=cfg.pp.score_th,
        color="black",
        alpha=0.7,
        linewidth=2,
        linestyle="--",
        label="score_th"
    )

    ax_prob.legend(loc="upper left")
    ax_anglez.legend(loc="upper center")
    ax.legend(loc="upper right")

    return fig, ax
