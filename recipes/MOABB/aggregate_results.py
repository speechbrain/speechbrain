#!/usr/bin/python
"""
Snippet to aggregate the results over multiple runs of the same experiment.
This is useful when we run multiple experiments with different seeds and we
want to compute the average performance. The script also reports the final
metric to Orion (when needed for hyperparameter tuning).

The script searches for the result files (_results.txt) and computes the mean
and the standard deviation of the given evaluation metrics (e.g., acc or f1).
The results must have an identical format (with only different performance
numbers).

To run this script:

    > python aggregate_results.py your_result_folder acc

Author
------
Mirco Ravanelli, 2022
"""

import sys
import numpy as np
from orion.client import report_objective
from speechbrain.utils.data_utils import get_all_files


def get_prototype(res_file, eval_metric):
    """Parses a result file and adds a placeholder where the aggregated metrics
    should be printed. It also returns the number of detected metrics.

    Arguments
    ---------
    res_file: path
        Path of the result file to parse.
    eval_metric: path
        Metric of interest (e.g, acc or f1).

    Returns
    ---------
    prototype: list
        List of the lines of the result file (with <values> as placeholder).
    n_metrics: int
        Number of metrics to replace in the result files.
    """
    prototype = []
    n_metrics = 0

    # Open the first res file and figure out where the metrics are
    with open(res_file) as file_in:
        for line in file_in:
            if eval_metric in line:
                line = line.split(eval_metric)[0]
                # The placeholder for the metric is <values>
                line = line + eval_metric + " <values>"
                n_metrics = n_metrics + 1
            prototype.append(line)
    return prototype, n_metrics


def get_metrics(res_files, eval_metric):
    """Summarizes the metrics of interest in a matrix.

    Arguments
    ---------
    res_files: list
        List of all the result files.
    eval_metric: path
        Metric of interest (e.g, acc or f1).

    Returns
    ---------
    metrics: np.array
        Matrix (n_metrics, n_files) containing the metrics of interest.
    """

    # Metric initialization
    metrics = np.zeros([n_metrics, len(res_files)])

    # Loop over files
    for i in range(len(res_files)):
        cnt = 0
        # Metric extraction
        with open(res_files[i]) as file_in:
            for line in file_in:
                if eval_metric in line:
                    value = line.split(eval_metric + " ")[1]
                    value = value.split(" ")[0]
                    value = float(value)
                    metrics[cnt, i] = value
                    cnt = cnt + 1
    return metrics


def aggregate_metrics(prototype, metrics):
    """Prints the aggregated metrics.It replaces the <values> placeholders with
    the corresponding metrics.

    Arguments
    ---------
    prototype: list
        List of the lines of the result file (with <values> as placeholder).
    metrics: np.array
        Matrix (n_metrics, n_files) containing the metrics of interest.
    """
    cnt = 0
    for line in prototype:
        if eval_metric in line:
            values_line = "["
            for i in range(len(res_files)):
                values_line = values_line + "%f " % float(metrics[cnt, i])
            values_line = values_line[:-1]
            values_line = values_line + "] avg: %f ± %f " % (
                float(metrics[cnt, :].mean()),
                float(metrics[cnt, :].std()),
            )
            line = line.replace("<values>", values_line)
            cnt = cnt + 1
        print(line)


if __name__ == "__main__":
    output_folder = sys.argv[1]
    eval_metric = sys.argv[2]

    # Getting the list of the result files in the output folder
    res_files = get_all_files(output_folder, match_and=["_results.txt"])

    # Gettin a prototype file
    prototype, n_metrics = get_prototype(res_files[0], eval_metric)

    # Extracting the metrics of interest
    metrics = get_metrics(res_files, eval_metric)

    # print aggregated metrics
    aggregate_metrics(prototype, metrics)

    final_metric = metrics[-1, :].mean()

    # Report final metric to Orion
    # Remember: orion expects metrics to be minimized!
    if eval_metric == "acc" or eval_metric == "f1":
        final_metric = 1 - final_metric
    report_objective(final_metric)
