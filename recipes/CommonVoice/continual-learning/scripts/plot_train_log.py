#!/usr/bin/env python3

"""Plot train log.

Authors
 * Luca Della Libera 2022
"""

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt, rc
from numpy import ndarray
from plotly import graph_objects as go


__all__ = [
    "parse_train_log",
    "plot_wer",
]


_EXPECTED_METRICS = [
    "epoch",
    "train loss",
    "valid loss",
    "valid CER",
    "valid WER",
]


def parse_train_log(train_log_file: "str") -> "Dict[str, ndarray]":
    """Parse train log to extract metric names and values.

    Parameters
    ----------
    train_log_file:
        The path to the train log file.

    Returns
    -------
        The metrics, i.e. a dict that maps names of
        the metrics to the metric values themselves.

    Examples
    --------
    >>> metrics = parse_train_log("train_log.txt")

    """
    metrics = defaultdict(list)
    with open(train_log_file) as f:
        for line in f:
            line = line.strip().replace(" - ", ", ")
            if not line:
                continue
            tokens = line.split(", ")
            names, values = zip(*[token.split(": ") for token in tokens])
            names, values = list(names), list(values)
            for name in _EXPECTED_METRICS:
                if name not in names:
                    names.append(name)
                    values.append("nan")
            for name, value in zip(names, values):
                try:
                    metrics[name].append(float(value))
                except Exception:
                    pass
    for name, values in metrics.items():
        metrics[name] = np.array(values)
    return metrics


def plot_wer(
    wers: "ndarray",
    output_image: "str",
    old_locales: "Optional[Sequence[str]]" = None,
    new_locales: "Optional[Sequence[str]]" = None,
    title: "Optional[str]" = None,
    figsize: "Tuple[float, float]" = (7.5, 6.0),
    usetex: "bool" = False,
    style_file_or_name: "str" = "classic",
) -> "None":
    """Plot word error rates extracted from a train log.

    Parameters
    ----------
    wers:
        The word error rates (old + new locales).
    output_image:
        The path to the output image.
    old_locales:
        The old locales.
        Default to ``("en", "zh-CN", "de", "es", "ru", "fr", "pt", "ja", "tr", "pl")``.
    new_locales:
        The new locales.
        Default to ``("rw", "eo", "kab", "lg", "mhr", "ckb", "ab", "kmr", "fy-NL", "ia")``.
    title:
        The plot title.
    figsize:
        The figure size.
    usetex:
        True to render text with LaTeX, False otherwise.
    style_file_or_name:
        The path to a Matplotlib style file or the name of one
        of Matplotlib built-in styles
        (see https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html).

    Examples
    --------
    >>> metrics = parse_train_log("train_log.txt")
    >>> plot_wer(metrics["test WER"], "train_log.png")

    """
    if old_locales is None:
        # fmt: off
        old_locales = (
            "en", "zh-CN", "de", "es", "ru", "fr", "pt", "ja", "tr", "pl",
        )
        # fmt: on
    if new_locales is None:
        # fmt: off
        new_locales = (
            "rw", "eo", "kab", "lg", "mhr", "ckb", "ab", "kmr", "fy-NL", "ia",
        )
        # fmt: on
    if title is None:
        title = os.path.splitext(os.path.basename(output_image))[0]

    # Plot with Matplotlib
    rc("text", usetex=usetex)
    rc("font", family="serif", serif=["Computer Modern"], size=14)
    rc("axes", labelsize=16)
    rc("legend", fontsize=12.5, handletextpad=0.3, handlelength=1)
    colors = sns.color_palette(
        "colorblind", len(old_locales) + len(new_locales)
    )
    if os.path.isfile(style_file_or_name):
        style_file_or_name = os.path.realpath(style_file_or_name)
    with plt.style.context(style_file_or_name):
        fig = plt.figure(figsize=figsize)
        plt.gca().set_prop_cycle("color", colors)
        locales = list(old_locales)
        j = 0
        for i, new_locale in enumerate([None] + list(new_locales)):
            if new_locale is not None:
                locales += [new_locale]
            plt.plot(
                range(len(locales)),
                wers[j : j + len(locales)],
                label=new_locale if new_locale is not None else "base",
                marker=".",
                markersize=7,
            )
            j += len(locales)
        plt.legend()
        plt.grid()
        plt.title(title)
        plt.xlim(-0.25, len(locales) - 1 + 0.25)
        plt.xticks(range(len(locales)), locales)
        plt.xlabel("Language")
        plt.ylabel("WER (%)")
        fig.tight_layout()
        plt.savefig(output_image, bbox_inches="tight")
        plt.savefig(output_image.replace(".png", ".pdf"), bbox_inches="tight")
        plt.close()

    # Plot with Plotly
    fig = go.Figure()
    locales = list(old_locales)
    j = 0
    for i, new_locale in enumerate([None] + list(new_locales)):
        if new_locale is not None:
            locales += [new_locale]
        fig.add_trace(
            go.Scatter(
                x=list(range(len(locales))),
                y=wers[j : j + len(locales)],
                marker={"size": 8},
                mode="lines+markers",
                name=new_locale if new_locale is not None else "base",
            )
        )
        j += len(locales)
    fig.update_layout(
        title={"text": title},
        legend={"traceorder": "normal"},
        template="none",
        font_size=20,
        xaxis={
            "title": "Language",
            "ticktext": locales,
            "tickvals": list(range(len(locales))),
            "ticks": "inside",
            "zeroline": False,
            "linewidth": 1.5,
            "range": [-0.25, len(locales) - 1 + 0.25],
            "showline": True,
            "mirror": "all",
            "gridcolor": "gray",
            "griddash": "dot",
        },
        yaxis={
            "title": "WER (%)",
            "showline": True,
            "ticks": "inside",
            "zeroline": False,
            "linewidth": 1.5,
            "mirror": "all",
            "gridcolor": "gray",
            "griddash": "dot",
        },
        margin={"t": 60, "b": 60},
    )
    fig.write_html(
        output_image.replace(".pdf", ".html").replace(".png", ".html"),
        include_plotlyjs=True,
    )

    # Compute performance metrics (average WER and forgetting,
    # see https://arxiv.org/abs/1801.10112)
    num_tasks = 1 + len(new_locales)
    A = np.full((num_tasks, num_tasks), -float("inf"))
    F = np.full((num_tasks, num_tasks), -float("inf"))
    idx = 0
    for k in range(num_tasks):
        for j in range(k + 1):
            if j == 0:
                A[k, j] = 1 - wers[idx : idx + len(old_locales)].mean() / 100
                idx += len(old_locales)
            else:
                A[k, j] = 1 - wers[idx] / 100
                idx += 1
            if j < k:
                F[k, j] = (A[:k, j] - A[k, j]).max()

    # Average WER
    mask = ~np.isinf(A)
    A[~mask] = 0
    avg_A = np.round((1 - A.sum(axis=-1) / mask.sum(axis=-1)) * 100, 2)

    # Average forgetting
    F = F[1:, :]
    mask = ~np.isinf(F)
    F[~mask] = 0
    avg_F = np.round(100 * F.sum(axis=-1) / mask.sum(axis=-1), 2)

    # Save performance metrics
    with open(
        output_image.replace(".png", ".csv").replace(".pdf", ".csv"), "w"
    ) as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["metric", "base"] + list(new_locales))
        csv_writer.writerow(["avg WER"] + avg_A.tolist())
        csv_writer.writerow(
            ["avg forgetting"] + [float("NaN")] + avg_F.tolist()
        )

    # Plot performance metrics
    with plt.style.context(style_file_or_name):
        fig = plt.figure(figsize=figsize)
        plt.plot(avg_A, label="Average WER", marker="o", markersize=7)
        plt.plot(
            [float("NaN")] + avg_F.tolist(),
            label="Average forgetting",
            marker="d",
            markersize=7,
        )
        plt.legend(loc="upper left")
        plt.grid()
        plt.title(title)
        plt.xlim(-0.25, len(new_locales) + 0.25)
        plt.ylim(0)
        plt.xticks(range(num_tasks), ["base"] + list(new_locales))
        plt.xlabel("Language")
        plt.ylabel("Metric (%)")
        fig.tight_layout()
        plt.savefig(
            output_image.replace(".png", "_metrics.png"), bbox_inches="tight"
        )
        plt.savefig(
            output_image.replace(".png", "_metrics.png").replace(
                ".png", ".pdf"
            ),
            bbox_inches="tight",
        )
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot train log")
    parser.add_argument(
        "train_log", help="path to train log",
    )
    parser.add_argument(
        "-t", "--title", default=None, help="plot title",
    )
    parser.add_argument(
        "-f",
        "--figsize",
        nargs=2,
        default=(10, 6),
        type=float,
        help="figure size",
    )
    parser.add_argument(
        "-u", "--usetex", action="store_true", help="render text with LaTeX",
    )
    parser.add_argument(
        "-s",
        "--style",
        default="classic",
        help="path to a Matplotlib style file or name of one of Matplotlib built-in styles",
        dest="style_file_or_name",
    )
    args = parser.parse_args()
    metrics = parse_train_log(args.train_log)
    output_image = args.train_log.replace(".txt", ".png")
    plot_wer(
        metrics["test WER"],
        output_image,
        figsize=args.figsize,
        usetex=args.usetex,
        style_file_or_name=args.style_file_or_name,
    )

    """
    # Reversed
    new_locales = ("ia", "fy-NL", "kmr", "ab", "ckb", "mhr", "lg", "kab", "eo", "rw")

    # Plot all logs found in the given directory
    train_logs = []
    for root, dirs, files in os.walk("plot_cl"):
        for file in files:
            if file.endswith(".txt"):
                train_logs.append(os.path.join(root, file))

    for train_log in train_logs:
        metrics = parse_train_log(train_log)
        output_image = train_log.replace(".txt", ".png")
        plot_wer(
            metrics["test WER"],
            output_image,
            figsize=(10, 6),
            usetex=True,
            style_file_or_name="classic",
        )
    """
