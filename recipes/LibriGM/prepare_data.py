"""
The functions to create the .csv files for LibriMix

Author
 * Martin Kocour 2022
"""

import os
import csv
import json

from glob import glob
from pathlib import Path

from recipes.LibriSpeech.librispeech_prepare import prepare_librispeech


def prepare_librigm(
    datapath,
    dmsource,
    savepath,
    skip_prep=False,
    fs=16000,
):
    """

    Prepare .csv files for librimix

    Arguments:
    ----------
        datapath (str) : path for the wsj0-mix dataset.
        dmsource (str) : path for the dynamic mixing train dataset.
        savepath (str) : path where we save the csv file.
        n_spks (int): number of speakers
        skip_prep (bool): If True, skip data preparation
        fs (int): sampling rate
    """

    if skip_prep and os.path.exists(os.path.join(savepath, "librigm_dev.csv")):
        return

    if "LibriGM" in datapath:
        # Libri 2/3Mix datasets
        create_librigm_csv(datapath, savepath)
    else:
        raise ValueError("Unsupported Dataset: {}".format(datapath))

    if "LibriSpeech" in dmsource:
        prepare_librispeech(
            dmsource,
            savepath,
            tr_splits=["train-clean-360"],
            skip_prep=skip_prep,
        )
    else:
        raise ValueError("Unsupported Dataset: {}".format(dmsource))


def create_librigm_csv(
    datapath,
    savepath,
    addnoise=False,
    set_types=["dev", "test"],
):
    """
    This functions creates the .csv file for the LibriGM dataset
    """
    for set_type in set_types:
        mix_path = os.path.join(datapath, set_type, "audio")
        files = list(
            map(
                lambda x: str(
                    Path(x).parent / Path(x).stem.removesuffix("_mix")
                ),
                glob(os.path.join(mix_path, "**", "**.json")),
            )
        )

        mix_ids = [f"{os.path.basename(fl)}_{i}" for i, fl in enumerate(files)]
        mix_fl_paths = [os.path.join(mix_path, fl + "_mix.wav") for fl in files]
        s1_fl_paths = [os.path.join(mix_path, fl + "_src1.wav") for fl in files]
        s2_fl_paths = [os.path.join(mix_path, fl + "_src2.wav") for fl in files]
        noise_fl_paths = [
            os.path.join(mix_path, fl + "_noise.wav") for fl in files
        ]
        info_fl_paths = [
            os.path.join(mix_path, fl + "_mix.json") for fl in files
        ]

        csv_columns = [
            "ID",
            "duration",
            "mix_wav",
            "mix_wav_format",
            "mix_wav_opts",
            "s1_wav",
            "s1_wav_format",
            "s1_wav_opts",
            "s2_wav",
            "s2_wav_format",
            "s2_wav_opts",
            "noise_wav",
            "noise_wav_format",
            "noise_wav_opts",
        ]

        with open(savepath + "/librigm_" + set_type + ".csv", "w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for i, (
                mix_id,
                mix_path,
                s1_path,
                s2_path,
                noise_path,
                info_path,
            ) in enumerate(
                zip(
                    mix_ids,
                    mix_fl_paths,
                    s1_fl_paths,
                    s2_fl_paths,
                    noise_fl_paths,
                    info_fl_paths,
                )
            ):
                with open(info_path) as f:
                    mix_info = json.load(f)

                row = {
                    "ID": mix_id,
                    "duration": mix_info["duration"],
                    "mix_wav": mix_path,
                    "mix_wav_format": "wav",
                    "mix_wav_opts": info_path,
                    "s1_wav": s1_path,
                    "s1_wav_format": "wav",
                    "s1_wav_opts": None,
                    "s2_wav": s2_path if os.path.exists(s2_path) else None,
                    "s2_wav_format": "wav",
                    "s2_wav_opts": None,
                    "noise_wav": noise_path if os.path.exists(noise_path) else None,
                    "noise_wav_format": "wav",
                    "noise_wav_opts": None,
                }
                writer.writerow(row)
