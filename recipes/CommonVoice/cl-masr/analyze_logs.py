#!/usr/bin/env python3

"""Analyze logs generated from continual learning experiments.

Authors
 * Luca Della Libera 2023
"""

import argparse
import csv
import logging
import os
from collections import defaultdict
from itertools import cycle
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy import ndarray
from tqdm import tqdm


_DEFAULT_METRICS = [
    "epoch",
    "train loss",
    "valid loss",
    "valid CER",
    "valid WER",
]

_COLORS = [
    # Built-in colors
    "#0000ff",
    "#ff0000",
    "#00ff00",
    "#ffc0cb",
    "#00ffff",
    "#ff00ff",
    "#ffa500",
    "#ffff00",
    "#40e0d0",
    "#e6e6fa",
    "#add8e6",
    "#800080",
    "#ffff00",
    "#000000",
    "#a52a2a",
    "#ffa500",
    "#008080",
    "#006400",
    "#d2b48c",
    "#fa8072",
    "#ffd700",
    "#8b0000",
    "#00008b",
    "#008000",
]

_MARKERS = ["o", "^", "p", "s", "d", "P", "v", "8", "<", "D", ">"]


def parse_train_log(train_log: "str") -> "Dict[str, ndarray]":
    """Parse a train log to extract metric names and values.

    Parameters
    ----------
    train_log:
        The path to the train log.

    Returns
    -------
        The metrics, i.e. a dict that maps names of
        the metrics to their corresponding values.

    Examples
    --------
    >>> metrics = parse_train_log("train_log.txt")

    """
    metrics = defaultdict(list)
    with open(train_log, encoding="utf-8") as f:
        for line in f:
            line = line.strip().replace(" - ", ", ")
            if not line:
                continue
            tokens = line.split(", ")
            names, values = zip(*[token.split(": ") for token in tokens])
            names, values = list(names), list(values)
            for name in _DEFAULT_METRICS:
                if name not in names:
                    names.append(name)
                    values.append("NaN")
            for name, value in zip(names, values):
                try:
                    metrics[name].append(float(value))
                except Exception:
                    pass
    for name, values in metrics.items():
        metrics[name] = np.array(values)
    return metrics


def compute_wer_matrix(
    wers: "ndarray", num_base_locales: "int" = 10, num_new_locales: "int" = 10
) -> "ndarray":
    """Compute the word error rate matrix.

    Parameters
    ----------
    wers:
        The word error rate for each locale.
    num_base_locales:
        The number of base locales.
    num_new_locales:
        The number of new locales.

    Returns
    -------
        The word error rate matrix.

    Raises
    ------
    RuntimeError
        If the number of locales is not consistent.

    Examples
    --------
    >>> wers = parse_train_log("train_log.txt")["test WER"]
    >>> wer_matrix = compute_wer_matrix(wers)

    """
    num_tasks = 1 + num_new_locales
    wer_matrix = np.full((num_tasks, num_tasks), float("inf"))
    idx = 0
    for k in range(num_tasks):
        for j in range(k + 1):
            if idx > len(wers) - 1:
                raise RuntimeError("Fewer locales than expected")
            if j == 0:
                wer_matrix[k, j] = (
                    wers[idx : idx + num_base_locales].mean() / 100
                )
                idx += num_base_locales
            else:
                wer_matrix[k, j] = wers[idx] / 100
                idx += 1
    return wer_matrix


def compute_awer(wer_matrix: "ndarray") -> "ndarray":
    """Compute the average word error rate.

    Parameters
    ----------
    wer_matrix:
        The word error rate matrix.

    Returns
    -------
        The average word error rate.

    References
    ----------
    .. [1] D. Lopez-Paz and M. Ranzato.
           "Gradient Episodic Memory for Continual Learning".
           In: NeurIPS. 2017, pp. 6470-6479.
           URL: https://arxiv.org/abs/1706.08840v6

    Examples
    --------
    >>> wers = parse_train_log("train_log.txt")["test WER"]
    >>> wer_matrix = compute_wer_matrix(wers)
    >>> awer = compute_awer(wer_matrix)

    """
    wer_matrix = wer_matrix.copy()
    mask = ~np.isinf(wer_matrix)
    wer_matrix[~mask] = 0
    awer = np.round(wer_matrix.sum(axis=-1) / mask.sum(axis=-1) * 100, 2)
    return awer


def compute_bwt(wer_matrix: "ndarray") -> "ndarray":
    """Compute the backward transfer.

    Parameters
    ----------
    wer_matrix:
        The word error rate matrix.

    Returns
    -------
        The backward transfer.

    References
    ----------
    .. [1] D. Lopez-Paz and M. Ranzato.
           "Gradient Episodic Memory for Continual Learning".
           In: NeurIPS. 2017, pp. 6470-6479.
           URL: https://arxiv.org/abs/1706.08840v6

    Examples
    --------
    >>> wers = parse_train_log("train_log.txt")["test WER"]
    >>> wer_matrix = compute_wer_matrix(wers)
    >>> bwt = compute_bwt(wer_matrix)

    """
    wer_matrix = wer_matrix.copy()
    num_tasks = len(wer_matrix)
    bwt = np.full((num_tasks,), float("NaN"))
    for k in range(1, num_tasks):
        bwt[k] = (np.diag(wer_matrix[:k, :k] - wer_matrix[k, :k])).mean()
    bwt[1:] = np.round(100 * bwt[1:], 2)
    return bwt


def compute_im(wer_matrix: "ndarray", refs: "ndarray") -> "ndarray":
    """Compute the intransigence measure.

    Parameters
    ----------
    wer_matrix:
        The word error rate matrix.
    refs:
        The intransigence measure references (joint fine-tuning).

    Returns
    -------
        The intransigence measure.

    References
    ----------
    .. [1] A. Chaudhry, P. K. Dokania, T. Ajanthan, and P. H. S. Torr.
           "Riemannian Walk for Incremental Learning: Understanding Forgetting and Intransigence".
           In: ECCV. 2018.
           URL: https://arxiv.org/abs/1801.10112v3

    Examples
    --------
    >>> wers = parse_train_log("train_log.txt")["test WER"]
    >>> wer_matrix = compute_wer_matrix(wers)
    >>> im = compute_im(wer_matrix, np.zeros(len(wer_matrix) - 1))

    """
    wer_matrix = wer_matrix.copy()
    num_tasks = len(wer_matrix)
    im = np.full(num_tasks, float("NaN"))
    im[1:] = np.round(np.diag(wer_matrix)[1:] * 100 - refs, 2)
    return im


def compute_fwt(wer_matrix: "ndarray", refs: "ndarray") -> "ndarray":
    """Compute the forward transfer.

    Parameters
    ----------
    wer_matrix:
        The word error rate matrix.
    refs:
        The forward transfer references (single task fine-tuning).

    Returns
    -------
        The forward transfer.

    Examples
    --------
    >>> wers = parse_train_log("train_log.txt")["test WER"]
    >>> wer_matrix = compute_wer_matrix(wers)
    >>> fwt = compute_fwt(wer_matrix, np.zeros(len(wer_matrix) - 1))

    """
    wer_matrix = wer_matrix.copy()
    num_tasks = len(wer_matrix)
    fwt = np.full(num_tasks, float("NaN"))
    fwt[1:] = np.round(refs - np.diag(wer_matrix)[1:] * 100, 2)
    return fwt


def plot_wer(
    wers: "ndarray",
    output_image: "str",
    base_locales: "Sequence[str]",
    new_locales: "Sequence[str]",
    xlabel: "Optional[str]" = None,
    figsize: "Tuple[float, float]" = (7.5, 6.0),
    title: "Optional[str]" = None,
    usetex: "bool" = False,
    hide_legend: "bool" = False,
    style_file_or_name: "str" = "classic",
) -> "None":
    """Plot word error rates extracted from a
    continual learning train log.

    Parameters
    ----------
    wers:
        The word error rates (base + new locales).
    output_image:
        The path to the output image.
    base_locales:
        The base locales.
    new_locales:
        The new locales.
    xlabel:
        The x-axis label.
    figsize:
        The figure size.
    title:
        The plot title.
    usetex:
        True to render text with LaTeX, False otherwise.
    hide_legend:
        True to hide the legend, False otherwise.
    style_file_or_name:
        The path to a Matplotlib style file or the name of one
        of Matplotlib built-in styles
        (see https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html).

    Examples
    --------
    >>> wers = parse_train_log("train_log.txt")["test WER"]
    >>> plot_wer(wers, "train_log.png")

    """
    plot_title = title != ""
    if title is None:
        title = os.path.splitext(os.path.basename(output_image))[0]

    # Plot with Matplotlib
    try:
        from matplotlib import pyplot as plt, rc

        if os.path.isfile(style_file_or_name):
            style_file_or_name = os.path.realpath(style_file_or_name)

        with plt.style.context(style_file_or_name):
            # Customize style
            rc("text", usetex=usetex)
            rc("font", family="serif", serif=["Computer Modern"], size=13)
            rc("axes", labelsize=15, titlesize=15)
            rc("legend", fontsize=12)
            rc("xtick", direction="in")
            rc("ytick", direction="in")
            fig = plt.figure(figsize=figsize)
            markers_iter = cycle(_MARKERS)
            locales = list(base_locales)
            j = 0
            for i, new_locale in enumerate([None] + list(new_locales)):
                if new_locale is not None:
                    locales += [new_locale]
                current_wers = wers[j : j + len(locales)]
                plt.plot(
                    range(len(locales)),
                    current_wers,
                    label=new_locale if new_locale is not None else "base",
                    marker=next(markers_iter),
                    markersize=6,
                    color=_COLORS[i % len(_COLORS)],
                )
                j += len(locales)
            if not hide_legend:
                plt.legend(fancybox=True)
            plt.grid()
            if plot_title:
                plt.title(title)
            plt.xlim(-0.25, len(locales) - 1 + 0.25)
            yrange = abs(plt.ylim()[0] - plt.ylim()[1])
            plt.ylim(
                plt.ylim()[0] - 0.025 * yrange, plt.ylim()[1] + 0.025 * yrange
            )
            plt.xticks(range(len(locales)), locales, rotation=90)
            if xlabel is not None:
                plt.xlabel(xlabel)
            plt.ylabel("WER (\%)" if usetex else "WER (%)")  # noqa: W605
            fig.tight_layout()
            plt.savefig(output_image, bbox_inches="tight")
            plt.close()
    except ImportError:
        logging.warning(
            "Install Matplotlib to generate the WER plots (e.g. `pip install matplotlib`)"
        )

    # Plot with Plotly
    try:
        from plotly import graph_objects as go

        fig = go.Figure()
        locales = list(base_locales)
        j = 0
        for i, new_locale in enumerate([None] + list(new_locales)):
            if new_locale is not None:
                locales += [new_locale]
            current_wers = wers[j : j + len(locales)]
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(locales))),
                    y=current_wers,
                    marker={"size": 8, "color": _COLORS[i % len(_COLORS)]},
                    mode="lines+markers",
                    name=new_locale if new_locale is not None else "base",
                )
            )
            j += len(locales)
        fig.update_layout(
            title={"text": title if plot_title else None},
            legend={"traceorder": "normal"},
            template="none",
            font_size=20,
            xaxis={
                "title": xlabel,
                "ticktext": locales,
                "tickvals": list(range(len(locales))),
                "tickangle": -90,
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
            f"{output_image.rsplit('.', 1)[0]}.html", include_plotlyjs=True,
        )
    except ImportError:
        logging.warning(
            "Install Plotly to generate the interactive WER plots (e.g. `pip install plotly`)"
        )


def plot_metric(
    metric_csv_file: "str",
    output_image: "str",
    xlabel: "Optional[str]" = None,
    ylabel: "Optional[str]" = None,
    xticks: "Optional[List[str]]" = None,
    figsize: "Tuple[float, float]" = (7.5, 6.0),
    title: "Optional[str]" = None,
    opacity: "float" = 0.5,
    usetex: "bool" = False,
    hide_legend: "bool" = False,
    style_file_or_name: "str" = "classic",
) -> "None":
    """Plot a continual learning metric.

    Parameters
    ----------
    metric_csv_file:
        The path to the continual learning metric CSV file.
    output_image:
        The path to the output image.
    xlabel:
        The x-axis label.
    ylabel:
        The y-axis label.
    xticks:
        The x-ticks.
    figsize:
        The figure size.
    title:
        The plot title.
    opacity:
        The confidence interval opacity.
    usetex:
        True to render text with LaTeX, False otherwise.
    hide_legend:
        True to hide the legend, False otherwise.
    style_file_or_name:
        The path to a Matplotlib style file or the name of one
        of Matplotlib built-in styles
        (see https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html).

    Examples
    --------
    >>> awer_file = "AWER.csv"
    >>> plot_metric(awer_file, "AWER.png")

    """
    traces = []
    with open(metric_csv_file, encoding="utf-8") as f:
        csv_reader = csv.reader(f)
        _ = next(csv_reader)
        for line in csv_reader:
            name, tasks, _ = line[0], line[1:-1], line[-1]
            mean, stddev = [], []
            for task in tasks:
                m, s = task.split(" +- ")
                mean.append(float(m))
                stddev.append(float(s))
            traces.append((name, mean, stddev))

    plot_title = title != ""
    if title is None:
        title = os.path.splitext(os.path.basename(output_image))[0]

    # Plot performance metrics with Matplotlib
    try:
        from matplotlib import pyplot as plt, rc

        if os.path.isfile(style_file_or_name):
            style_file_or_name = os.path.realpath(style_file_or_name)

        with plt.style.context(style_file_or_name):
            # Customize style
            rc("text", usetex=usetex)
            rc("font", family="serif", serif=["Computer Modern"], size=13)
            rc("axes", labelsize=15, titlesize=15)
            rc("legend", fontsize=12)
            rc("xtick", direction="in")
            rc("ytick", direction="in")
            fig = plt.figure(figsize=figsize)
            markers_iter = cycle(_MARKERS)
            for i, (name, mean, stddev) in enumerate(traces):
                plt.plot(
                    mean,
                    label=name,
                    marker=next(markers_iter),
                    markersize=6,
                    color=_COLORS[i % len(_COLORS)],
                )
                shift = stddev
                plt.fill_between(
                    range(len(mean)),
                    y1=[m - s for m, s in zip(mean, shift)],
                    y2=[m + s for m, s in zip(mean, shift)],
                    color=_COLORS[i % len(_COLORS)],
                    alpha=opacity,
                )
            if not hide_legend:
                plt.legend(
                    loc="upper left",
                    ncols=2 if len(traces) > 10 else 1,
                    fancybox=True,
                )
            plt.grid()
            if plot_title:
                plt.title(title)
            plt.xlim(-0.25, len(xticks) - 1 + 0.25)
            yrange = abs(plt.ylim()[0] - plt.ylim()[1])
            plt.ylim(
                plt.ylim()[0] - 0.025 * yrange, plt.ylim()[1] + 0.025 * yrange
            )
            if xticks is not None:
                plt.xticks(range(len(xticks)), xticks, rotation=90)
            if xlabel is not None:
                plt.xlabel(xlabel)
            if ylabel is not None:
                plt.ylabel(ylabel)
            fig.tight_layout()
            plt.savefig(output_image, bbox_inches="tight")
            plt.close()
    except ImportError:
        logging.warning(
            "Install Matplotlib to generate the performance metrics plots (e.g. `pip install matplotlib`)"
        )

    # Plot with Plotly
    try:
        from plotly import graph_objects as go

        def hex_to_rgb(hex_color: "str") -> "Tuple":
            hex_color = hex_color.lstrip("#")
            if len(hex_color) == 3:
                hex_color = hex_color * 2
            return (
                int(hex_color[0:2], 16),
                int(hex_color[2:4], 16),
                int(hex_color[4:6], 16),
            )

        fig = go.Figure()
        for i, (name, mean, stddev) in enumerate(traces):
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(mean))),
                    y=mean,
                    marker={"size": 8},
                    mode="lines+markers",
                    name=name,
                )
            )
            shift = stddev
            scatter_kwargs = {
                "legendgroup": i,
                "line": {"width": 0},
                "marker": {"color": _COLORS[i % len(_COLORS)]},
                "mode": "lines",
                "showlegend": False,
            }
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(mean))),
                    y=[m + s for m, s in zip(mean, shift)],
                    **scatter_kwargs,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(mean))),
                    y=[m - s for m, s in zip(mean, shift)],
                    fill="tonexty",
                    fillcolor=f"rgba{hex_to_rgb(_COLORS[i % len(_COLORS)]) + (opacity,)}",
                    **scatter_kwargs,
                )
            )
        fig.update_layout(
            title={"text": title if plot_title else None},
            legend={"traceorder": "normal"},
            template="none",
            font_size=20,
            xaxis={
                "title": xlabel,
                "ticktext": xticks,
                "tickvals": list(range(len(xticks))),
                "tickangle": -90,
                "ticks": "inside",
                "zeroline": False,
                "linewidth": 1.5,
                "range": [-0.25, len(xticks) - 1 + 0.25],
                "showline": True,
                "mirror": "all",
                "gridcolor": "gray",
                "griddash": "dot",
            },
            yaxis={
                "title": ylabel.replace("\\", ""),
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
            f"{output_image.rsplit('.', 1)[0]}.html", include_plotlyjs=True,
        )
    except ImportError:
        logging.warning(
            "Install Plotly to generate the interactive performance metrics plots (e.g. `pip install plotly`)"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze logs")
    parser.add_argument(
        "input_dir",
        help="path to directory containing the continual learning train logs. Filenames must be formatted as "
        "<method-name>_base=<comma-separated-base-locales>_new=<comma-separated-new-locales>",
    )
    parser.add_argument(
        "--im_refs",
        # fmt: off
        # whisper-large-v2
        default='{"ab": 64.33, "ckb": 57.51, "eo": 20.14, "fy-NL": 35.01, "ia": 18.70, "kab": 73.57, "kmr": 47.28, "lg": 60.31, "mhr": 37.94, "rw": 69.22}',
        # wavlm-large
        # default='{"ab": 69.39, "ckb": 69.27, "eo": 36.53, "fy-NL": 52.00, "ia": 53.71, "kab": 83.05, "kmr": 66.22, "lg": 66.81, "mhr": 53.53, "rw": 82.34}',
        # whisper-tiny
        # default='{"ab": 77.22, "ckb": 71.33, "eo": 39.12, "fy-NL": 54.84, "ia": 37.24, "kab": 89.74, "kmr": 63.88, "lg": 74.23, "mhr": 54.08, "rw": 85.24}',
        # fmt: on
        help="intransigence measure references",
    )
    parser.add_argument(
        "--fwt_refs",
        # fmt: off
        # whisper-large-v2
        default='{"ab": 58.96, "ckb": 54.51, "eo": 18.45, "fy-NL": 28.26, "ia": 15.22, "kab": 64.51, "kmr": 39.84, "lg": 55.72, "mhr": 31.64, "rw": 67.04}',
        # wavlm-large
        # default='{"ab": 62.31, "ckb": 62.67, "eo": 30.04, "fy-NL": 43.82, "ia": 28.36, "kab": 72.80, "kmr": 50.60, "lg": 58.90, "mhr": 44.69, "rw": 74.70}',
        # whisper-tiny
        # default='{"ab": 70.66, "ckb": 63.18, "eo": 32.21, "fy-NL": 43.84, "ia": 24.74, "kab": 75.08, "kmr": 52.70, "lg": 69.43, "mhr": 45.23, "rw": 85.64}',
        # fmt: on
        help="forward transfer references",
    )
    parser.add_argument(
        "-f", "--format", default="png", help="image format",
    )
    parser.add_argument(
        "-s",
        "--figsize",
        nargs=2,
        default=(7.50, 6.50),
        type=float,
        help="figure size",
    )
    parser.add_argument(
        "-t", "--title", default=None, help="title",
    )
    parser.add_argument(
        "-o", "--opacity", default=0.15, help="confidence interval opacity",
    )
    parser.add_argument(
        "--hide_legend", action="store_true", help="hide legend",
    )
    parser.add_argument(
        "-u", "--usetex", action="store_true", help="render text with LaTeX",
    )
    parser.add_argument(
        "--order",
        nargs="+",
        help="train log processing order e.g. `FT ER A-GEM PNN PB EWC LwF'",
    )
    parser.add_argument(
        "--style",
        default="classic",
        help="path to a Matplotlib style file or name of one of Matplotlib built-in styles",
        dest="style_file_or_name",
    )
    args = parser.parse_args()

    # Retrieve all continual learning train logs in the input directory
    train_logs = []
    for root, dirs, files in os.walk(args.input_dir):
        for file in files:
            if file.endswith(".txt"):
                train_log = os.path.join(root, file)
                train_logs.append(train_log)
    train_logs = sorted(train_logs)

    # Group train logs by name
    groups = {}
    for train_log in train_logs:
        filename = os.path.basename(train_log)
        prefix = filename.split("_")[0]
        if prefix not in groups:
            groups[prefix] = []
        groups[prefix].append(train_log)

    # Sort
    if args.order is not None:
        groups = {k: groups[k] for k in args.order}

    # Compute metrics
    metrics = defaultdict(lambda: defaultdict(list))
    with tqdm(total=len(groups)) as progress_bar:
        for group_name, train_logs in groups.items():
            progress_bar.set_description(group_name)
            for train_log in train_logs:
                # Extract base + new locales from file name
                locales = (
                    os.path.basename(train_log)
                    .replace(".txt", "")
                    .split("_base=")
                )[1]
                base_locales, new_locales = locales.split("_new=")
                base_locales = [x.strip() for x in base_locales.split(",")]
                new_locales = [x.strip() for x in new_locales.split(",")]

                # Compute metrics
                wers = parse_train_log(train_log)["test WER"]
                wer_matrix = compute_wer_matrix(
                    wers, len(base_locales), len(new_locales)
                )

                awer = compute_awer(wer_matrix)
                metrics["Average WER"][group_name].append(awer)

                bwt = compute_bwt(wer_matrix)
                metrics["Backward transfer"][group_name].append(bwt)

                im_refs = eval(args.im_refs)
                im = compute_im(
                    wer_matrix, np.asarray([im_refs[k] for k in new_locales])
                )
                metrics["Intransigence measure"][group_name].append(im)

                fwt_refs = eval(args.fwt_refs)
                fwt = compute_fwt(
                    wer_matrix, np.asarray([fwt_refs[k] for k in new_locales])
                )
                metrics["Forward transfer"][group_name].append(fwt)

                # Plot WERs
                output_image = train_log.replace(".txt", f".{args.format}")
                plot_wer(
                    wers,
                    output_image,
                    base_locales=base_locales,
                    new_locales=new_locales,
                    xlabel=None,
                    figsize=args.figsize,
                    title=args.title,
                    usetex=args.usetex,
                    hide_legend=args.hide_legend,
                    style_file_or_name=args.style_file_or_name,
                )
            progress_bar.update()

    # Store metrics
    for name in metrics:
        with open(
            os.path.join(args.input_dir, f"{name}.csv"), "w", encoding="utf-8"
        ) as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(
                ["name", "base"]
                + [str(i) for i in range(1, 1 + len(new_locales))]
                + ["avg"]
            )
            for group_name, traces in metrics[name].items():
                mean = np.mean(traces, axis=0)
                stddev = np.std(traces, axis=0)
                avg = np.nanmean(traces, axis=1)
                avg_mean = np.mean(avg)
                # Assuming independence, sigma^2 = sum_1^n sigma_i^2 / n^2
                avg_stddev = np.sqrt(
                    np.nansum(stddev ** 2) / (~np.isnan(stddev)).sum() ** 2
                )
                csv_writer.writerow(
                    [group_name]
                    + [f"{m:.2f} +- {s:.2f}" for m, s in zip(mean, stddev)]
                    + [f"{avg_mean:.2f} +- {avg_stddev:.2f}"]
                )

    # Plot metrics
    for name in metrics:
        metric_csv_file = os.path.join(args.input_dir, f"{name}.csv")
        plot_metric(
            metric_csv_file,
            output_image=os.path.join(args.input_dir, f"{name}.{args.format}"),
            xlabel=None,
            ylabel=f"{name} (\%)"
            if args.usetex
            else f"{name} (%)",  # noqa: W605
            xticks=["base"] + [f"L{i}" for i in range(1, 1 + len(new_locales))],
            figsize=args.figsize,
            title=args.title,
            opacity=args.opacity,
            usetex=args.usetex,
            hide_legend=args.hide_legend,
            style_file_or_name=args.style_file_or_name,
        )
