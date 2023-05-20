#!/usr/bin/env python3

"""Analyze logs.

Authors
 * Luca Della Libera 2023
"""

import argparse
import csv
import logging
import os
from collections import defaultdict
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from numpy import ndarray


_DEFAULT_METRICS = [
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
    with open(train_log_file, encoding="utf-8") as f:
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


def compute_cl_metrics(
    all_wers: "Dict[str, ndarray]",
    # fmt: off
    base_locales: "Sequence[str]" = ("en", "zh-CN", "de", "es", "ru", "fr", "pt", "ja", "tr", "pl"),
    new_locales: "Sequence[str]" = ("rw", "eo", "kab", "lg", "mhr", "ckb", "ab", "kmr", "fy-NL", "ia"),
    # fmt: on
):
    """Compute continual learning metrics such as average
    word error rate, average forgetting and average learning
    from word error rates extracted from multiple train logs
    in continual learning format.

    Parameters
    ----------
    all_wers:
        The word error rates, i.e. a dict that maps
        names of the per-log word error rates to their
        corresponding values (base + new locales).
    base_locales:
        The base locales.
    new_locales:
        The new locales.

    Returns
    -------
        - The average word error rate;
        - the average forgetting;
        - the average learning.

    References
    ----------
    .. [1] A. Chaudhry, P. K. Dokania, T. Ajanthan, and P. H. S. Torr.
           "Riemannian Walk for Incremental Learning: Understanding Forgetting and Intransigence".
           In: European Conference on Computer Vision (ECCV). 2018.
           URL: https://arxiv.org/abs/1801.10112v3

    Examples
    --------
    >>> all_wers = {
    ...     "ft": parse_train_log("train_log_ft.txt")["test WER"],
    ...     "ewc": parse_train_log("train_log_ewc.txt")["test WER"],
    ... }
    >>> compute_cl_metrics(all_wers)

    """
    avg_As, avg_Fs, avg_Ls = [], [], []
    for wers in all_wers.values():
        num_tasks = 1 + len(new_locales)
        A = np.full((num_tasks, num_tasks), -float("inf"))
        F = np.full((num_tasks, num_tasks), -float("inf"))
        idx = 0
        for k in range(num_tasks):
            for j in range(k + 1):
                if idx > len(wers) - 1:
                    # Fewer languages than expected
                    break
                if j == 0:
                    A[k, j] = wers[idx : idx + len(base_locales)].mean() / 100
                    idx += len(base_locales)
                else:
                    A[k, j] = wers[idx] / 100
                    idx += 1
                if j < k:
                    F[k, j] = (A[k, j] - A[:k, j]).max()

        # Average learning
        avg_L = np.diag(A)[1:].copy()
        mask = ~np.isinf(avg_L)
        avg_L[~mask] = 0
        avg_L = -np.round(avg_L * 100, 2)

        # Average WER
        mask = ~np.isinf(A)
        A[~mask] = 0
        avg_A = np.round(A.sum(axis=-1) / mask.sum(axis=-1) * 100, 2)

        # Average forgetting
        F = F[1:, :]
        mask = ~np.isinf(F)
        F[~mask] = 0
        avg_F = np.round(100 * F.sum(axis=-1) / mask.sum(axis=-1), 2)

        avg_As.append(avg_A)
        avg_Fs.append(avg_F)
        avg_Ls.append(avg_L)

    return avg_As, avg_Fs, avg_Ls


def plot_wer(
    wers: "ndarray",
    output_image: "str",
    # fmt: off
    base_locales: "Sequence[str]" = ("en", "zh-CN", "de", "es", "ru", "fr", "pt", "ja", "tr", "pl"),
    new_locales: "Sequence[str]" = ("rw", "eo", "kab", "lg", "mhr", "ckb", "ab", "kmr", "fy-NL", "ia"),
    # fmt: on
    title: "Optional[str]" = None,
    figsize: "Tuple[float, float]" = (7.5, 6.0),
    usetex: "bool" = False,
    hide_legend: "bool" = False,
    style_file_or_name: "str" = "classic",
) -> "None":
    """Plot word error rates extracted from a single train log
    in continual learning format.

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
    title:
        The plot title.
    figsize:
        The figure size.
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
    >>> metrics = parse_train_log("train_log.txt")
    >>> plot_wer(metrics["test WER"], "train_log.png")

    """
    if title is None:
        title = os.path.splitext(os.path.basename(output_image))[0]

    # Plot with Matplotlib
    try:
        from matplotlib import pyplot as plt, rc

        if os.path.isfile(style_file_or_name):
            style_file_or_name = os.path.realpath(style_file_or_name)

        with plt.style.context(style_file_or_name):
            # Custom style
            rc("text", usetex=usetex)
            rc("font", family="serif", serif=["Computer Modern"], size=13)
            rc("axes", labelsize=15)
            rc("legend", fontsize=12)
            rc("xtick", direction="in")
            rc("ytick", direction="in")
            rc(
                "axes",
                prop_cycle=plt.cycler(
                    "color", ((0.0, 0.0, 0.0),) + plt.cm.tab10.colors
                ),
            )
            fig = plt.figure(figsize=figsize)
            locales = list(base_locales)
            j = 0
            for i, new_locale in enumerate([None] + list(new_locales)):
                if new_locale is not None:
                    locales += [new_locale]
                current_wers = wers[j : j + len(locales)]
                if len(locales) != len(current_wers):
                    # Fewer languages than expected
                    break
                plt.plot(
                    range(len(locales)),
                    current_wers,
                    label=new_locale if new_locale is not None else "base",
                    marker="d",
                    markersize=5,
                )
                j += len(locales)
            if not hide_legend:
                plt.legend(fancybox=True)
            plt.grid()
            plt.title(title)
            plt.xlim(-0.25, len(locales) - 1 + 0.25)
            plt.ylim(-0.025 * plt.ylim()[1])
            plt.xticks(range(len(locales)), locales)
            plt.xlabel("Language")
            plt.ylabel("WER (\%)" if usetex else "WER (%)")
            fig.tight_layout()
            plt.savefig(output_image, bbox_inches="tight")
            plt.close()
    except ImportError:
        logging.warning(
            "Install Matplotlib to generate the WER plot (e.g. `pip install matplotlib`)"
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
            if len(locales) != len(current_wers):
                # Fewer languages than expected
                break
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(locales))),
                    y=current_wers,
                    marker={"size": 7},
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
            f"{output_image.rsplit('.', 1)[0]}.html", include_plotlyjs=True,
        )
    except ImportError:
        logging.warning(
            "Install Plotly to generate the interactive WER plot (e.g. `pip install plotly`)"
        )


def plot_cl_metrics(
    all_wers: "Dict[str, ndarray]",
    output_dir: "str",
    # fmt: off
    base_locales: "Sequence[str]" = ("en", "zh-CN", "de", "es", "ru", "fr", "pt", "ja", "tr", "pl"),
    new_locales: "Sequence[str]" = ("rw", "eo", "kab", "lg", "mhr", "ckb", "ab", "kmr", "fy-NL", "ia"),
    # fmt: on
    figsize: "Tuple[float, float]" = (7.5, 6.0),
    format: "str" = "png",
    usetex: "bool" = False,
    hide_legend: "bool" = False,
    style_file_or_name: "str" = "classic",
) -> "None":
    """Plot continual learning metrics from word error
    rates extracted from multiple train logs in continual
    learning format.

    Parameters
    ----------
    all_wers:
        The word error rates, i.e. a dict that maps
        names of the per-log word error rates to their
        corresponding values (base + new locales).
    output_dir:
        The path to the output directory.
    base_locales:
        The base locales.
    new_locales:
        The new locales.
    figsize:
        The figure size.
    format:
        The image format.
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
    >>> all_wers = {
    ...     "ft": parse_train_log("train_log_ft.txt")["test WER"],
    ...     "ewc": parse_train_log("train_log_ewc.txt")["test WER"],
    ... }
    >>> plot_cl_metrics(all_wers, "plots")

    """
    # Compute performance metrics
    avg_As, avg_Fs, avg_Ls = compute_cl_metrics(
        all_wers, base_locales, new_locales
    )

    # Save performance metrics
    model = os.path.basename(output_dir).replace(".txt", "")
    with open(
        os.path.join(output_dir, f"{model}_avg_wer.csv"), "w", encoding="utf-8"
    ) as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["name", "base"] + list(new_locales) + ["avg"])
        for name, avg_A in zip(all_wers.keys(), avg_As):
            csv_writer.writerow(
                [name] + avg_A.tolist() + [round(np.nanmean(avg_A), 2)]
            )
    with open(
        os.path.join(output_dir, f"{model}_avg_forgetting.csv"),
        "w",
        encoding="utf-8",
    ) as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["name", "base"] + list(new_locales) + ["avg"])
        for name, avg_F in zip(all_wers.keys(), avg_Fs):
            csv_writer.writerow(
                [name]
                + [float("NaN")]
                + avg_F.tolist()
                + [round(np.nanmean(avg_F), 2)]
            )
    with open(
        os.path.join(output_dir, f"{model}_avg_learning.csv"),
        "w",
        encoding="utf-8",
    ) as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["name", "base"] + list(new_locales) + ["avg"])
        for name, avg_L in zip(all_wers.keys(), avg_Ls):
            csv_writer.writerow(
                [name]
                + [float("NaN")]
                + avg_L.tolist()
                + [round(np.nanmean(avg_L), 2)]
            )

    # Plot performance metrics with Matplotlib
    try:
        from matplotlib import pyplot as plt, rc

        if os.path.isfile(style_file_or_name):
            style_file_or_name = os.path.realpath(style_file_or_name)

        output_image = os.path.join(output_dir, f"{model}_avg_wer.{format}")
        markers = ["o", "^", "d", "s", "p", "*", "+", 0, 1, 2, 3, 4] * 2
        with plt.style.context(style_file_or_name):
            # Customize style
            rc("text", usetex=usetex)
            rc("font", family="serif", serif=["Computer Modern"], size=13)
            rc("axes", labelsize=15)
            rc("legend", fontsize=12)
            rc("xtick", direction="in")
            rc("ytick", direction="in")
            rc("axes", prop_cycle=plt.cycler("color", plt.cm.tab10.colors))
            fig = plt.figure(figsize=figsize)
            markers_iter = iter(markers)
            for name, avg_A in zip(all_wers.keys(), avg_As):
                name = name.replace(".txt", "")
                plt.plot(
                    avg_A, label=name, marker=next(markers_iter), markersize=5
                )
            if not hide_legend:
                plt.legend(
                    loc="upper left",
                    ncols=2 if len(all_wers) > 10 else 1,
                    fancybox=True,
                )
            plt.grid()
            plt.title(model)
            plt.xlim(-0.25, len(new_locales) + 0.25)
            # plt.ylim(-0.025 * 160.0, 160.0)
            plt.ylim(0.975 * plt.ylim()[0], 1.025 * plt.ylim()[1])
            plt.xticks(
                range(1 + len(new_locales)), ["base"] + list(new_locales)
            )
            plt.xlabel("Language")
            plt.ylabel("Average WER (\%)" if usetex else "Average WER (%)")
            fig.tight_layout()
            plt.savefig(output_image, bbox_inches="tight")
            plt.close()

        output_image = os.path.join(
            output_dir, f"{model}_avg_forgetting.{format}"
        )
        with plt.style.context(style_file_or_name):
            # Customize style
            rc("text", usetex=usetex)
            rc("font", family="serif", serif=["Computer Modern"], size=13)
            rc("axes", labelsize=15)
            rc("legend", fontsize=12)
            rc("xtick", direction="in")
            rc("ytick", direction="in")
            rc("axes", prop_cycle=plt.cycler("color", plt.cm.tab10.colors))
            fig = plt.figure(figsize=figsize)
            markers_iter = iter(markers)
            for name, avg_F in zip(all_wers.keys(), avg_Fs):
                name = name.replace(".txt", "")
                plt.plot(
                    [float("NaN")] + avg_F.tolist(),
                    label=name,
                    marker=next(markers_iter),
                    markersize=5,
                )
            if not hide_legend:
                plt.legend(
                    loc="upper left",
                    ncols=2 if len(all_wers) > 10 else 1,
                    fancybox=True,
                )
            plt.grid()
            plt.title(model)
            plt.xlim(-0.25, len(new_locales) + 0.25)
            # plt.ylim(-0.025 * 160.0, 160.0)
            plt.ylim(0.975 * plt.ylim()[0], 1.025 * plt.ylim()[1])
            plt.xticks(
                range(1 + len(new_locales)), ["base"] + list(new_locales)
            )
            plt.xlabel("Language")
            plt.ylabel(
                "Average forgetting (\%)"
                if usetex
                else "Average forgetting (%)"
            )
            fig.tight_layout()
            plt.savefig(output_image, bbox_inches="tight")
            plt.close()

        output_image = os.path.join(
            output_dir, f"{model}_avg_learning.{format}"
        )
        with plt.style.context(style_file_or_name):
            # Customize style
            rc("text", usetex=usetex)
            rc("font", family="serif", serif=["Computer Modern"], size=13)
            rc("axes", labelsize=15)
            rc("legend", fontsize=12)
            rc("xtick", direction="in")
            rc("ytick", direction="in")
            rc("axes", prop_cycle=plt.cycler("color", plt.cm.tab10.colors))
            fig = plt.figure(figsize=figsize)
            markers_iter = iter(markers)
            for name, avg_L in zip(all_wers.keys(), avg_Ls):
                name = name.replace(".txt", "")
                plt.plot(
                    [float("NaN")] + avg_L.tolist(),
                    label=name,
                    marker=next(markers_iter),
                    markersize=5,
                )
            if not hide_legend:
                plt.legend(
                    loc="upper left",
                    ncols=2 if len(all_wers) > 10 else 1,
                    fancybox=True,
                )
            plt.grid()
            plt.title(model)
            plt.xlim(-0.25, len(new_locales) + 0.25)
            # plt.ylim(-160.0, 0.025 * 160.0)
            plt.ylim(0.975 * plt.ylim()[0], 1.025 * plt.ylim()[1])
            plt.xticks(
                range(1 + len(new_locales)), ["base"] + list(new_locales)
            )
            plt.xlabel("Language")
            plt.ylabel(
                "Average learning (\%)" if usetex else "Average learning (%)"
            )
            fig.tight_layout()
            plt.savefig(output_image, bbox_inches="tight")
            plt.close()
    except ImportError:
        logging.warning(
            "Install Matplotlib to generate the performance metrics plots (e.g. `pip install matplotlib`)"
        )

    # Plot performance metrics with Plotly
    try:
        from plotly import graph_objects as go

        output_image = os.path.join(output_dir, f"{model}_avg_wer.html")
        fig = go.Figure()
        for name, avg_A in zip(all_wers.keys(), avg_As):
            name = name.replace(".txt", "")
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(avg_A) + 1)),
                    y=avg_A.tolist(),
                    marker={"size": 7},
                    mode="lines+markers",
                    name=name,
                )
            )
        fig.update_layout(
            title={"text": model},
            legend={"traceorder": "normal"},
            template="none",
            font_size=20,
            xaxis={
                "title": "Language",
                "ticktext": ["base"] + list(new_locales),
                "tickvals": list(range(len(new_locales) + 1)),
                "ticks": "inside",
                "zeroline": False,
                "linewidth": 1.5,
                "range": [-0.25, len(new_locales) + 0.25],
                "showline": True,
                "mirror": "all",
                "gridcolor": "gray",
                "griddash": "dot",
            },
            yaxis={
                "title": "Average WER (%)",
                "showline": True,
                "ticks": "inside",
                "zeroline": False,
                "linewidth": 1.5,
                "rangemode": "tozero",
                "mirror": "all",
                "gridcolor": "gray",
                "griddash": "dot",
            },
            margin={"t": 60, "b": 60},
        )
        fig.write_html(
            output_image, include_plotlyjs=True,
        )

        output_image = os.path.join(output_dir, f"{model}_avg_forgetting.html")
        fig = go.Figure()
        for name, avg_F in zip(all_wers.keys(), avg_Fs):
            name = name.replace(".txt", "")
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(avg_F) + 1)),
                    y=[float("NaN")] + avg_F.tolist(),
                    marker={"size": 7},
                    mode="lines+markers",
                    name=name,
                )
            )
        fig.update_layout(
            title={"text": model},
            legend={"traceorder": "normal"},
            template="none",
            font_size=20,
            xaxis={
                "title": "Language",
                "ticktext": ["base"] + list(new_locales),
                "tickvals": list(range(len(new_locales) + 1)),
                "ticks": "inside",
                "zeroline": False,
                "linewidth": 1.5,
                "range": [-0.25, len(new_locales) + 0.25],
                "showline": True,
                "mirror": "all",
                "gridcolor": "gray",
                "griddash": "dot",
            },
            yaxis={
                "title": "Average forgetting (%)",
                "showline": True,
                "ticks": "inside",
                "zeroline": False,
                "linewidth": 1.5,
                "rangemode": "tozero",
                "mirror": "all",
                "gridcolor": "gray",
                "griddash": "dot",
            },
            margin={"t": 60, "b": 60},
        )
        fig.write_html(
            output_image, include_plotlyjs=True,
        )

        output_image = os.path.join(output_dir, f"{model}_avg_learning.html")
        fig = go.Figure()
        for name, avg_L in zip(all_wers.keys(), avg_Ls):
            name = name.replace(".txt", "")
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(avg_L) + 1)),
                    y=[float("NaN")] + avg_L.tolist(),
                    marker={"size": 7},
                    mode="lines+markers",
                    name=name,
                )
            )
        fig.update_layout(
            title={"text": model},
            legend={"traceorder": "normal"},
            template="none",
            font_size=20,
            xaxis={
                "title": "Language",
                "ticktext": ["base"] + list(new_locales),
                "tickvals": list(range(len(new_locales) + 1)),
                "ticks": "inside",
                "zeroline": False,
                "linewidth": 1.5,
                "range": [-0.25, len(new_locales) + 0.25],
                "showline": True,
                "mirror": "all",
                "gridcolor": "gray",
                "griddash": "dot",
            },
            yaxis={
                "title": "Average learning (%)",
                "showline": True,
                "ticks": "inside",
                "zeroline": False,
                "linewidth": 1.5,
                "rangemode": "tozero",
                "mirror": "all",
                "gridcolor": "gray",
                "griddash": "dot",
            },
            margin={"t": 60, "b": 60},
        )
        fig.write_html(
            output_image, include_plotlyjs=True,
        )
    except ImportError:
        logging.warning(
            "Install Plotly to generate the interactive performance metrics plots (e.g. `pip install plotly`)"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze logs")
    parser.add_argument(
        "input_dir",
        help="path to directory containing the train logs in continual learning format",
    )
    parser.add_argument(
        "-b",
        "--base_locales",
        nargs="+",
        # fmt: off
        default=("en", "zh-CN", "de", "es", "ru", "fr", "pt", "ja", "tr", "pl"),
        # fmt: on
        help="base locales",
    )
    parser.add_argument(
        "-n",
        "--new_locales",
        nargs="+",
        # fmt: off
        default=("rw", "eo", "kab", "lg", "mhr", "ckb", "ab", "kmr", "fy-NL", "ia"),
        # fmt: on
        help="new locales",
    )
    parser.add_argument(
        "-s",
        "--figsize",
        nargs=2,
        default=(10, 6),
        type=float,
        help="figure size",
    )
    parser.add_argument(
        "-f", "--format", default="png", help="image format",
    )
    parser.add_argument(
        "-u", "--usetex", action="store_true", help="render text with LaTeX",
    )
    parser.add_argument(
        "-l", "--hide_legend", action="store_true", help="hide legend",
    )
    parser.add_argument(
        "--style",
        default="classic",
        help="path to a Matplotlib style file or name of one of Matplotlib built-in styles",
        dest="style_file_or_name",
    )
    args = parser.parse_args()

    # Retrieve all train logs in the given directory
    all_wers = {}
    for root, dirs, files in os.walk(args.input_dir):
        for file in files:
            if file.endswith(".txt"):
                train_log = os.path.join(root, file)
                metrics = parse_train_log(train_log)
                wers = metrics["test WER"]
                all_wers[file] = wers
                output_image = train_log.replace(".txt", f".{args.format}")
                new_locales = args.new_locales
                if "order=" in os.path.basename(train_log):
                    new_locales = (
                        os.path.basename(train_log)
                        .replace(".txt", "")
                        .split("order=")[1]
                        .split(", ")
                    )
                # Plot WER
                plot_wer(
                    wers,
                    output_image,
                    base_locales=args.base_locales,
                    new_locales=new_locales,
                    figsize=args.figsize,
                    usetex=args.usetex,
                    hide_legend=args.hide_legend,
                    style_file_or_name=args.style_file_or_name,
                )
    all_wers = dict(sorted(all_wers.items()))

    # Plot metrics
    plot_cl_metrics(
        all_wers,
        args.input_dir,
        base_locales=args.base_locales,
        new_locales=args.new_locales,
        figsize=args.figsize,
        format=args.format,
        usetex=args.usetex,
        hide_legend=args.hide_legend,
        style_file_or_name=args.style_file_or_name,
    )
