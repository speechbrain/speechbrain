"""
The functions to create the .csv files for LibriCSS

Author
 * Martin Kocour 2022
"""

import os
import csv
import re
import torchaudio

from glob import glob
from pathlib import Path


def prepare_libricss(
    datapath,
    savepath,
    partitions=["utterances"],
    skip_prep=False,
    fs=16000,
):
    """

    Prepare .csv files for librimix

    Arguments:
    ----------
        datapath (str) : path for the LibriCSS dataset.
        savepath (str) : path where we save the csv file.
        partitions (list): LibriCSS partition (`utterances`, `segments`)
        skip_prep (bool): If True, skip data preparation
        fs (int): sampling rate
    """

    if skip_prep and os.path.exists(os.path.join(savepath, "librigm_dev.csv")):
        return

    if "LibriCSS" in datapath:
        create_libricss_csv(datapath, savepath, partitions, fs)
    else:
        raise ValueError("Unsupported Dataset: {}".format(datapath))


def create_libricss_csv(
    datapath,
    savepath,
    partitions=["utterances"],
    fs=16000,
):
    """
    This functions creates the .csv file for the LibriCSS dataset
    """
    ptrn = re.compile(r".*overlap_ratio_([^/_]+)_sil(.*)_session.*")

    for set_type in partitions:
        mix_path = os.path.join(datapath, "monaural" , set_type)
        files = glob(os.path.join(mix_path, "**", "*.wav"))

        csv_columns = [
            "ID",
            "duration",
            "num_samples",
            "overlap_ratio",
            "silence",
            "mix_wav",
        ]

        with open(
            os.path.join(savepath, "libricss_" + set_type + ".csv"), "w"
        ) as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for mix_path in files:
                mix_info = torchaudio.info(mix_path)
                assert mix_info.sample_rate == fs
                mix_id = create_id(mix_path)
                m = ptrn.match(mix_id)

                row = {
                    "ID": mix_id,
                    "duration": mix_info.num_frames / fs,
                    "num_samples": mix_info.num_frames,
                    "overlap_ratio": m.group(1),
                    "silence": m.group(2),
                    "mix_wav": mix_path,
                }
                writer.writerow(row)


def create_id(path):
    path = Path(path)
    session = path.parent.stem
    fname = path.stem
    return f"LibriCSS_{session}_{fname}"
