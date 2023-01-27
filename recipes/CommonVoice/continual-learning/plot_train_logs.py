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
    with open(train_log_file, encoding="utf-8") as f:
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
    """Plot word error rates extracted from a single train log.

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
    rc("legend", fontsize=10, handletextpad=0.3, handlelength=1)
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
            current_wers = wers[j : j + len(locales)]
            if len(locales) != len(current_wers):
                # Less languages than expected
                break
            plt.plot(
                range(len(locales)),
                current_wers,
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
        current_wers = wers[j : j + len(locales)]
        if len(locales) != len(current_wers):
            # Less languages than expected
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
        output_image.replace(".pdf", ".html").replace(".png", ".html"),
        include_plotlyjs=True,
    )


def plot_cl_metrics(
    all_wers: "Dict[str, ndarray]",
    output_dir: "str",
    old_locales: "Optional[Sequence[str]]" = None,
    new_locales: "Optional[Sequence[str]]" = None,
    figsize: "Tuple[float, float]" = (7.5, 6.0),
    usetex: "bool" = False,
    style_file_or_name: "str" = "classic",
) -> "None":
    """Plot continual learning metrics from word error
    rates extracted from multiple train logs.

    Parameters
    ----------
    all_wers:
        The word error rates, i.e. a dict that maps
        names of the per-log word error rates to their
        corresponding values (old + new locales).
    output_dir:
        The path to the output directory.
    old_locales:
        The old locales.
        Default to ``("en", "zh-CN", "de", "es", "ru", "fr", "pt", "ja", "tr", "pl")``.
    new_locales:
        The new locales.
        Default to ``("rw", "eo", "kab", "lg", "mhr", "ckb", "ab", "kmr", "fy-NL", "ia")``.
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
    >>> all_wers = [
    ...     parse_train_log("train_log1.txt")["test WER"],
    ...     parse_train_log("train_log2.txt")["test WER"],
    ... ]
    >>> plot_cl_metrics(all_wers, "plots")

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

    # Plot with Matplotlib
    rc("text", usetex=usetex)
    rc("font", family="serif", serif=["Computer Modern"], size=14)
    rc("axes", labelsize=16)
    rc("legend", fontsize=10, handletextpad=0.3, handlelength=1)
    colors = sns.color_palette(
        "colorblind", len(old_locales) + len(new_locales)
    )
    if os.path.isfile(style_file_or_name):
        style_file_or_name = os.path.realpath(style_file_or_name)

    # Compute performance metrics (average WER and forgetting,
    # see https://arxiv.org/abs/1801.10112)
    avg_As, avg_Fs = [], []
    for wers in all_wers.values():
        num_tasks = 1 + len(new_locales)
        A = np.full((num_tasks, num_tasks), -float("inf"))
        F = np.full((num_tasks, num_tasks), -float("inf"))
        idx = 0
        for k in range(num_tasks):
            for j in range(k + 1):
                if idx > len(wers) - 1:
                    # Less languages than expected
                    break
                if j == 0:
                    A[k, j] = (
                        1 - wers[idx : idx + len(old_locales)].mean() / 100
                    )
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

        avg_As.append(avg_A)
        avg_Fs.append(avg_F)

    # Save performance metrics
    model = os.path.basename(output_dir).replace(".txt", "")
    with open(
        os.path.join(output_dir, f"{model}_metrics.csv"), "w", encoding="utf-8"
    ) as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["name", "metric", "base"] + list(new_locales))
        for name, avg_A, avg_F in zip(all_wers.keys(), avg_As, avg_As):
            csv_writer.writerow([name, "avg WER"] + avg_A.tolist())
            csv_writer.writerow(
                [name, "avg forgetting"] + [float("NaN")] + avg_F.tolist()
            )

    # Plot performance metrics with Matplotlib
    output_image = os.path.join(output_dir, f"{model}_average_wer.png")
    with plt.style.context(style_file_or_name):
        fig = plt.figure(figsize=figsize)
        plt.gca().set_prop_cycle("color", colors)
        for name, avg_A in zip(all_wers.keys(), avg_As):
            name = name.replace(".txt", "").split("_", maxsplit=1)[-1]
            plt.plot(avg_A, label=name, marker="d", markersize=7)
        plt.legend(loc="upper left")
        if len(all_wers) > 10:
            plt.legend(loc="upper left", ncols=2)
        plt.grid()
        plt.title(model)
        plt.xlim(-0.25, len(new_locales) + 0.25)
        plt.ylim(0)
        plt.xticks(range(num_tasks), ["base"] + list(new_locales))
        plt.xlabel("Language")
        plt.ylabel("Average WER (%)")
        fig.tight_layout()
        plt.savefig(output_image, bbox_inches="tight")
        plt.savefig(output_image.replace(".png", ".pdf"), bbox_inches="tight")
        plt.close()

    output_image = os.path.join(output_dir, f"{model}_average_forgetting.png")
    with plt.style.context(style_file_or_name):
        fig = plt.figure(figsize=figsize)
        plt.gca().set_prop_cycle("color", colors)
        for name, avg_F in zip(all_wers.keys(), avg_Fs):
            name = name.replace(".txt", "").split("_", maxsplit=1)[-1]
            plt.plot(
                [float("NaN")] + avg_F.tolist(),
                label=name,
                marker="d",
                markersize=7,
            )
        plt.legend(loc="upper left")
        if len(all_wers) > 10:
            plt.legend(loc="upper left", ncols=2)
        plt.grid()
        plt.title(model)
        plt.xlim(-0.25, len(new_locales) + 0.25)
        plt.ylim(0)
        plt.xticks(range(num_tasks), ["base"] + list(new_locales))
        plt.xlabel("Language")
        plt.ylabel("Average forgetting (%)")
        fig.tight_layout()
        plt.savefig(output_image, bbox_inches="tight")
        plt.savefig(output_image.replace(".png", ".pdf"), bbox_inches="tight")
        plt.close()

    # Plot performance metrics with Plotly
    output_image = os.path.join(output_dir, f"{model}_average_wer.html")
    fig = go.Figure()
    for name, avg_A in zip(all_wers.keys(), avg_As):
        name = name.replace(".txt", "").split("_", maxsplit=1)[-1]
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

    output_image = os.path.join(output_dir, f"{model}_average_forgetting.html")
    fig = go.Figure()
    for name, avg_F in zip(all_wers.keys(), avg_Fs):
        name = name.replace(".txt", "").split("_", maxsplit=1)[-1]
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot train logs")
    parser.add_argument(
        "input_dir", help="path to directory containing the train logs",
    )
    parser.add_argument(
        "-o",
        "--old_locales",
        nargs="+",
        default=("en", "zh-CN", "de", "es", "ru", "fr", "pt", "ja", "tr", "pl"),
        help="old locales",
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
        "-f",
        "--figsize",
        nargs=2,
        default=(10, 7.5),
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

    # Retrieve all train logs in the given directory
    all_wers = {}
    for root, dirs, files in os.walk(args.input_dir):
        for file in files:
            if file.endswith(".txt"):
                train_log = os.path.join(root, file)
                metrics = parse_train_log(train_log)
                wers = metrics["test WER"]
                all_wers[file] = wers
                output_image = train_log.replace(".txt", ".png")
                # Plot WER
                plot_wer(
                    wers,
                    output_image,
                    old_locales=args.old_locales,
                    new_locales=args.new_locales,
                    figsize=args.figsize,
                    usetex=args.usetex,
                    style_file_or_name=args.style_file_or_name,
                )
    all_wers = dict(sorted(all_wers.items()))

    # Plot metrics
    plot_cl_metrics(
        all_wers,
        args.input_dir,
        old_locales=args.old_locales,
        new_locales=args.new_locales,
        figsize=args.figsize,
        usetex=args.usetex,
        style_file_or_name=args.style_file_or_name,
    )
