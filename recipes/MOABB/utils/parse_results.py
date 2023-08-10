#!/usr/bin/python
"""
Aggregate EEG training results from different sessions and subjects.

This script aggregates results for four EEG training paradigms:
1. Within-Session
2. Cross-Session
3. Leave-One-Session-Out
4. Leave-One-Subject-Out

Usage:
python parse_results.py <results_dir> <metric_file> <metrics>...

Parameters:
- results_dir: Path to the directory containing experiment results.
- metric_file: Path to the metric file (generated during training).
- metrics: List of metrics to aggregate (e.g., "acc", "loss", "f1").

Example:
python3 parse_results.py results/MOABB/EEGNet_BNCI2014001/1234 test_metrics.pkl acc loss f1

For validation results, use valid_metrics.pkl.

Author:
Francesco Paissan, 2021
"""


from pathlib import Path
from pickle import load
from typing import Tuple
from numpy import mean, std, round
import sys


def load_metrics(filepath: Path) -> dict:

    """
    Loads pickles and parses into a dictionary

    Arguments
    ---------
    filepath : path
        Path of the metric file.

    Returns
    -------
    rtype: dict
    """

    try:
        with open(filepath, "rb") as f:
            temp = load(f)

        return temp
    except Exception as e:
        print(f"Error on {str(filepath)} - {str(e)}")
        return None


def visualize_results(paradigm: str, results: dict, vis_metrics: list) -> None:
    """
    Function to visualize the results.

    Arguments
    ---------
    paradigm : str
    results: dict
    vis_metrics: list
    """
    print("\n----", paradigm.name, "----")
    for key in results:
        if type(results[key]) == dict:
            for m in vis_metrics:
                print(
                    key,
                    m,
                    round(mean(results[key][m]), 4),
                    "±",
                    round(std(results[key][m]), 4),
                )
        else:
            if key in vis_metrics:
                print(
                    key,
                    round(mean(results[key]), 4),
                    "±",
                    round(std(results[key]), 4),
                )


def parse_one_session_out(
    paradigm: Path,
    vis_metrics: list = [],
    stat_metrics: list = ["loss", "f1", "acc"],
    metric_file: str = "test_metrics.pkl",
) -> dict:

    """
    Aggregates results obtain by helding back one session as test set and
    using the remaining ones to train the neural nets

    Arguments
    ---------
    paradigm: path
    vis_metrics: list
    stat_metrics: list
    metric_file: str

    Returns
    -------
    out_stat: dict
    """

    sub_folders = [
        folder
        for folder in paradigm.iterdir()
        if folder.is_dir() and folder.name.startswith("sub-")
    ]
    folds = sorted(sub_folders)

    out_stat = {
        key.name: {metric: [] for metric in stat_metrics}
        for key in sorted(
            folder
            for folder in folds[0].iterdir()
            if folder.is_dir() and folder.name.startswith("session")
        )
    }

    for f in folds:
        child = sorted(
            [
                folder
                for folder in f.iterdir()
                if folder.is_dir() and folder.name.startswith("session")
            ]
        )
        for sess in child:
            metrics = load_metrics(sess.joinpath(metric_file))
            if metrics is not None:
                for m in stat_metrics:
                    out_stat[sess.name][m].append(metrics[m])
            else:
                print("Something was wrong when computing ", paradigm)

    if len(vis_metrics) != 0:
        visualize_results(paradigm, out_stat, vis_metrics)

    return out_stat


def parse_one_sub_out(
    paradigm: Path,
    vis_metrics: list = [],
    stat_metrics: list = ["loss", "f1", "acc"],
    metric_file: str = "test_metrics.pkl",
) -> dict:
    """
    Aggregates results obtained helding out one subject
    as test set and using the remaining ones for training.

    Arguments
    ---------
    paradigm: path
    vis_metrics: list
    stat_metrics: list
    metric_file: str

    Returns
    -------
    out_stat: dict
    """

    sub_folders = [
        folder
        for folder in paradigm.iterdir()
        if folder.is_dir() and folder.name.startswith("sub-")
    ]
    folds = sorted(sub_folders)

    out_stat = {metric: [] for metric in stat_metrics}

    for f in folds:
        metrics = load_metrics(f.joinpath(metric_file))
        if metrics is not None:
            for m in stat_metrics:
                out_stat[m].append(metrics[m])
        else:
            print("Something was wrong when computing ", f)

    if len(vis_metrics) != 0:
        visualize_results(paradigm, out_stat, vis_metrics)

    return out_stat


def aggregate_nested(
    results: dict,
    metric_file: str = "test_metrics.pkl",
    stat_metrics: list = ["loss", "f1", "acc"],
):
    """
    Add description

    Arguments
    ---------
    results: dict
    metric_file: str
    stat_metrics: list

    Returns
    -------
    temp: dict
    """
    temp = {key: [] for key in stat_metrics}
    for _, v in results.items():
        for k, r in v.items():
            temp[k].append(mean(r))

    return temp


def aggregate_single(
    results: dict,
    metric_file: str = "test_metrics.pkl",
    stat_metrics: list = ["loss", "f1", "acc"],
):
    """
    Add description

    Arguments
    ---------
    results: dict
    metric_file: str
    stat_metrics: list

    Returns
    -------
    temp: dict
    """
    temp = {key: [] for key in stat_metrics}
    for k, r in results.items():
        temp[k].append(mean(r))

    return temp


available_parsers = {
    "leave-one-session-out": parse_one_session_out,
    "leave-one-subject-out": parse_one_sub_out,
}

available_aggrs = {
    "leave-one-session-out": aggregate_nested,
    "leave-one-subject-out": aggregate_single,
}


def aggregate_metrics(
    verbose=1,
    metric_file="test_metrics.pkl",
    stat_metrics=["loss", "f1", "acc"],
) -> Tuple:
    """
    Parses results and computes statistics over all
    paradigms.

    Arguments
    ---------
    verbode: int
    metric_file: str
    stat_metrics: list

    Returns
    -------
    overall_stat: tuple
    """
    results_folder = Path(sys.argv[1])
    vis_metrics = stat_metrics

    available_paradigms = list(
        map(lambda x: x.stem, sorted(results_folder.iterdir()))
    )

    available_paradigms = [
        paradigm
        for paradigm in available_paradigms
        if paradigm in available_parsers.keys()
    ]

    overall_stat = {key: [] for key in stat_metrics}

    parsers = {k: available_parsers[k] for k in available_paradigms}
    aggr = {k: available_aggrs[k] for k in available_paradigms}

    for paradigm in sorted(results_folder.iterdir()):

        if paradigm.name not in available_paradigms:
            continue

        results = parsers[paradigm.name](
            paradigm,
            vis_metrics,
            metric_file=metric_file,
            stat_metrics=stat_metrics,
        )
        temp = aggr[paradigm.name](
            results, metric_file=metric_file, stat_metrics=stat_metrics
        )

        for k in temp.keys():
            overall_stat[k].extend(temp[k])

    for k in stat_metrics:
        overall_stat[k + "_std"] = std(overall_stat[k])
        overall_stat[k] = mean(overall_stat[k])

    return overall_stat


if __name__ == "__main__":
    metric_file = sys.argv[2]
    stat_metrics = sys.argv[3:]
    temp = aggregate_metrics(
        verbose=1, metric_file=metric_file, stat_metrics=stat_metrics
    )

    print("\nAggregated results")
    for k in stat_metrics:
        print(k, temp[k], "+-", temp[k + "_std"])
