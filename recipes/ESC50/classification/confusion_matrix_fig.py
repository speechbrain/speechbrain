#!/usr/bin/env python3
"""Helper to create Confusion Matrix figure

Authors
 * David Whipps 2021
 * Ala Eddine Limame 2021
"""

import numpy as np
import matplotlib.pyplot as plt
import itertools


def create_cm_fig(cm, display_labels):

    fig = plt.figure(figsize=cm.shape, dpi=50, facecolor="w", edgecolor="k")
    ax = fig.add_subplot(1, 1, 1)

    ax.imshow(cm, cmap="Oranges")  # fits with the tensorboard colour scheme

    tick_marks = np.arange(cm.shape[0])

    ax.set_xlabel("Predicted class", fontsize=18)
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(display_labels, ha="center", fontsize=18, rotation=90)
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    ax.set_ylabel("True class", fontsize=18)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(display_labels, va="center", fontsize=18)
    ax.yaxis.set_label_position("left")
    ax.yaxis.tick_left()

    fmt = "d"  # TODO use '.3f' if normalized
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            verticalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
            fontsize=18,
        )

    fig.set_tight_layout(True)
    return fig
