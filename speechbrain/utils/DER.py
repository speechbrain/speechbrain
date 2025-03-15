"""Calculates Diarization Error Rate (DER) which is the sum of Missed Speaker (MS),
False Alarm (FA), and Speaker Error Rate (SER) using md-eval-22.pl from NIST RT Evaluation.

Authors
 * Neville Ryant 2018
 * Nauman Dawalatabad 2020

Credits
 This code is adapted from https://github.com/nryant/dscore
"""

import os
import re
import subprocess

import numpy as np

FILE_IDS = re.compile(r"(?<=Speaker Diarization for).+(?=\*\*\*)")
SCORED_SPEAKER_TIME = re.compile(r"(?<=SCORED SPEAKER TIME =)[\d.]+")
MISS_SPEAKER_TIME = re.compile(r"(?<=MISSED SPEAKER TIME =)[\d.]+")
FA_SPEAKER_TIME = re.compile(r"(?<=FALARM SPEAKER TIME =)[\d.]+")
ERROR_SPEAKER_TIME = re.compile(r"(?<=SPEAKER ERROR TIME =)[\d.]+")


def rectify(arr):
    """Corrects corner cases and converts scores into percentage."""
    # Numerator and denominator both 0.
    arr[np.isnan(arr)] = 0

    # Numerator > 0, but denominator = 0.
    arr[np.isinf(arr)] = 1
    arr *= 100.0

    return arr


def DER(
    ref_rttm,
    sys_rttm,
    ignore_overlap=False,
    collar=0.25,
    individual_file_scores=False,
):
    """Computes Missed Speaker percentage (MS), False Alarm (FA),
    Speaker Error Rate (SER), and Diarization Error Rate (DER).

    Arguments
    ---------
    ref_rttm : str
        The path of reference/groundtruth RTTM file.
    sys_rttm : str
        The path of the system generated RTTM file.
    ignore_overlap : bool
        If True, ignores overlapping speech during evaluation.
    collar : float
        Forgiveness collar.
    individual_file_scores : bool
        If True, returns scores for each file in order.

    Returns
    -------
    MS : float array
        Missed Speech.
    FA : float array
        False Alarms.
    SER : float array
        Speaker Error Rates.
    DER : float array
        Diarization Error Rates.

    Example
    -------
    >>> import pytest
    >>> pytest.skip('Skipping because of Perl dependency')
    >>> ref_rttm = "../../tests/samples/rttm/ref_rttm/ES2014c.rttm"
    >>> sys_rttm = "../../tests/samples/rttm/sys_rttm/ES2014c.rttm"
    >>> ignore_overlap = True
    >>> collar = 0.25
    >>> individual_file_scores = True
    >>> Scores = DER(ref_rttm, sys_rttm, ignore_overlap, collar, individual_file_scores)
    >>> print (Scores)
    (array([0., 0.]), array([0., 0.]), array([7.16923618, 7.16923618]), array([7.16923618, 7.16923618]))
    """
    curr = os.path.abspath(os.path.dirname(__file__))
    mdEval = os.path.join(curr, "../../tools/der_eval/md-eval.pl")

    cmd = [
        mdEval,
        "-af",
        "-r",
        ref_rttm,
        "-s",
        sys_rttm,
        "-c",
        str(collar),
    ]
    if ignore_overlap:
        cmd.append("-1")

    try:
        stdout = subprocess.check_output(cmd, stderr=subprocess.STDOUT)

    except subprocess.CalledProcessError as ex:
        stdout = ex.output

    else:
        stdout = stdout.decode("utf-8")

        # Get all recording IDs
        file_ids = [m.strip() for m in FILE_IDS.findall(stdout)]
        file_ids = [
            file_id[2:] if file_id.startswith("f=") else file_id for file_id in file_ids
        ]

        scored_speaker_times = np.array(
            [float(m) for m in SCORED_SPEAKER_TIME.findall(stdout)]
        )

        miss_speaker_times = np.array(
            [float(m) for m in MISS_SPEAKER_TIME.findall(stdout)]
        )

        fa_speaker_times = np.array([float(m) for m in FA_SPEAKER_TIME.findall(stdout)])

        error_speaker_times = np.array(
            [float(m) for m in ERROR_SPEAKER_TIME.findall(stdout)]
        )

        with np.errstate(invalid="ignore", divide="ignore"):
            tot_error_times = (
                miss_speaker_times + fa_speaker_times + error_speaker_times
            )
            miss_speaker_frac = miss_speaker_times / scored_speaker_times
            fa_speaker_frac = fa_speaker_times / scored_speaker_times
            sers_frac = error_speaker_times / scored_speaker_times
            ders_frac = tot_error_times / scored_speaker_times

        # Values in percentage of scored_speaker_time
        miss_speaker = rectify(miss_speaker_frac)
        fa_speaker = rectify(fa_speaker_frac)
        sers = rectify(sers_frac)
        ders = rectify(ders_frac)

        if individual_file_scores:
            return miss_speaker, fa_speaker, sers, ders
        else:
            return miss_speaker[-1], fa_speaker[-1], sers[-1], ders[-1]
