#!/usr/bin/python
"""
Snippet to aggregate results based on four different EEG training paradigms
(within-session, cross-session, leave-one-session-out, leave-one-subject-out).

To run this script (e.g., exp_results: results/MOABB/EEGNet_BNCI2014001/<seed>; required metrics: ["acc", "loss", "f1"]):

    > python3 parse_results.py results/MOABB/EEGNet_BNCI2014001/1234 acc loss f1

The dataset will be automatically downloaded in the specified folder.

Author
------
Francesco Paissan, 2021
"""

from pathlib import Path
from pickle import load
from numpy import mean, std, round
import sys


def load_metrics(filepath: Path) -> dict:
    """Loads pickles and parses into a dictionary

    :param filepath: [description]
    :type filepath: Path
    :return: [description]
    :rtype: dict
    """

    try:
        with open(filepath, "rb") as f:
            temp = load(f)

        return temp
    except Exception as e:
        print(f"Error on {str(filepath)} - {str(e)}")
        return None


def visualize_results(paradigm: str, results: dict, vis_metrics: list) -> None:
    """Prints aggregated strings

    :param paradigm: [description]
    :type paradigm: str
    :param results: [description]
    :type results: dict
    :param vis_metrics: [description]
    :type vis_metrics: list
    """
    print("----", paradigm.name, "----")
    for key in results:
        if type(results[key]) == dict:
            for m in vis_metrics:
                print(
                    key,
                    m,
                    round(mean(results[key][m]), 4),
                    "+-",
                    round(std(results[key][m]), 4),
                )
        else:
            if key in vis_metrics:
                print(
                    key,
                    round(mean(results[key]), 4),
                    "+-",
                    round(std(results[key]), 4),
                )


def parse_one_session_out(paradigm: Path) -> dict:
    """Aggregates results obtain by helding back one session as test set and
    using the remaining ones to train the neural nets

    :param paradigm: [description]
    :type paradigm: Path
    :return: [description]
    :rtype: dict
    """
    out_stat = {
        key.name: {metric: [] for metric in stat_metrics}
        for key in sorted(folds[0].iterdir())
    }

    for f in folds:
        child = sorted(f.iterdir())
        for sess in child:
            metrics = load_metrics(sess.joinpath("metrics.pkl"))
            if metrics is not None:
                for m in stat_metrics:
                    out_stat[sess.name][m].append(metrics[m])
            else:
                print("Something was wrong when computing ", paradigm)

    return out_stat


def parse_cross_section(paradigm: Path) -> dict:
    """Aggregates results obtained using all session' signals merged together.
    Training and test sets are defined using a stratified cross-validation partitioning.

    :param paradigm: [description]
    :type paradigm: Path
    :return: [description]
    :rtype: dict
    """
    out_stat = {metric: [] for metric in stat_metrics}

    for f in folds:
        child = sorted(f.iterdir())
        sess_metrics = {metric: [] for metric in stat_metrics}

        for sess in child:
            metrics = load_metrics(sess.joinpath("metrics.pkl"))
            if metrics is not None:
                for m in stat_metrics:
                    sess_metrics[m].append(metrics[m])

            else:
                print("Something was wrong when computing ", f)

        for m in stat_metrics:
            out_stat[m].append(mean(sess_metrics[m]))

    return out_stat


def parse_one_sub_out(paradigm: Path) -> dict:
    """Aggregates results obtained helding out one subject
    as test set and using the remaining ones for training.

    :param paradigm: [description]
    :type paradigm: Path
    :return: [description]
    :rtype: dict
    """
    out_stat = {metric: [] for metric in stat_metrics}

    for f in folds:
        metrics = load_metrics(f.joinpath("metrics.pkl"))
        if metrics is not None:
            for m in stat_metrics:
                out_stat[m].append(metrics[m])
        else:
            print("Something was wrong when computing ", f)

    return out_stat


def parse_within_session(paradigm: Path) -> dict:
    """For each subject and for each session, the training
    and test sets were defined using a stratified cross-validation partitioning.

    :param paradigm: [description]
    :type paradigm: Path
    :return: [description]
    :rtype: dict
    """
    out_stat = {
        key.name: {metric: [] for metric in stat_metrics}
        for key in sorted(folds[0].iterdir())
    }

    for f in folds:
        child = sorted(f.iterdir())
        for sess in child:
            sub_perf = {metric: [] for metric in stat_metrics}

            for sub in sess.iterdir():
                metrics = load_metrics(sub.joinpath("metrics.pkl"))
                if metrics is not None:
                    for m in stat_metrics:
                        sub_perf[m].append(metrics[m])
                else:
                    print("Something was wrong when computing ", f)

            for k in sub_perf.keys():
                out_stat[sess.name][k].append(mean(sub_perf[k]))

    return out_stat


stat_metrics = ["loss", "f1", "acc", "auc"]

if __name__ == "__main__":
    results_folder = Path(sys.argv[1])
    vis_metrics = sys.argv[2:]

    for paradigm in results_folder.iterdir():
        folds = sorted(paradigm.iterdir())

        if paradigm.name == "leave-one-session-out":
            results = parse_one_session_out(paradigm)
            visualize_results(paradigm, results, vis_metrics)

        elif paradigm.name == "cross-session":
            results = parse_cross_section(paradigm)
            visualize_results(paradigm, results, vis_metrics)

        elif paradigm.name == "leave-one-subject-out":
            results = parse_one_sub_out(paradigm)
            visualize_results(paradigm, results, vis_metrics)

        elif paradigm.name == "within-session":
            results = parse_within_session(paradigm)
            visualize_results(paradigm, results, vis_metrics)
