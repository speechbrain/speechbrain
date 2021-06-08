"""
Author
 * Cem Subakan 2020

The .csv preperation functions for WSJ0-Mix.
"""

import os
import csv


def prepare_wham_whamr_csv(
    datapath, savepath, skip_prep=False, fs=8000,
):
    """
    Prepares the csv files for wham or whamr dataset

    Arguments:
    ----------
        datapath (str) : path for the wsj0-mix dataset.
        savepath (str) : path where we save the csv file.
        skip_prep (bool): If True, skip data preparation
    """

    if skip_prep:
        return

    if "wham_original" in datapath:
        # if we want to train a model on the original wham dataset
        create_wham_whamr_csv(
            datapath, savepath, fs, savename="whamorg_", add_reverb=False
        )
    elif "whamr" in datapath:
        # if we want to train a model on the whamr dataset
        create_wham_whamr_csv(datapath, savepath, fs)
    else:
        raise ValueError("Unsupported Dataset")


def create_wham_whamr_csv(
    datapath,
    savepath,
    fs,
    version="min",
    savename="whamr_",
    set_types=["tr", "cv", "tt"],
    add_reverb=True,
):
    """
    This function creates the csv files to get the speechbrain data loaders for the whamr dataset.

    Arguments:
        datapath (str) : path for the wsj0-mix dataset.
        savepath (str) : path where we save the csv file
        fs (int) : the sampling rate
        version (str) : min or max
        savename (str) : the prefix to use for the .csv files
        set_types (list) : the sets to create
    """
    if fs == 8000:
        sample_rate = "8k"
    elif fs == 16000:
        sample_rate = "16k"
    else:
        raise ValueError("Unsupported sampling rate")

    if add_reverb:
        mix_both = "mix_both_reverb/"
        s1 = "s1_anechoic/"
        s2 = "s2_anechoic/"
    else:
        mix_both = "mix_both/"
        s1 = "s1/"
        s2 = "s2/"

    for set_type in set_types:
        mix_path = os.path.join(
            datapath, "wav{}".format(sample_rate), version, set_type, mix_both,
        )
        s1_path = os.path.join(
            datapath, "wav{}".format(sample_rate), version, set_type, s1,
        )
        s2_path = os.path.join(
            datapath, "wav{}".format(sample_rate), version, set_type, s2,
        )
        noise_path = os.path.join(
            datapath, "wav{}".format(sample_rate), version, set_type, "noise/"
        )
        # rir_path = os.path.join(
        #     datapath, "wav{}".format(sample_rate), version, set_type, "rirs/"
        # )

        files = os.listdir(mix_path)

        mix_fl_paths = [mix_path + fl for fl in files]
        s1_fl_paths = [s1_path + fl for fl in files]
        s2_fl_paths = [s2_path + fl for fl in files]
        noise_fl_paths = [noise_path + fl for fl in files]
        # rir_fl_paths = [rir_path + fl + ".t" for fl in files]

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
            # "rir_t",
            # "rir_format",
            # "rir_opts",
        ]

        with open(
            os.path.join(savepath, savename + set_type + ".csv"), "w"
        ) as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for (i, (mix_path, s1_path, s2_path, noise_path),) in enumerate(
                zip(
                    mix_fl_paths,
                    s1_fl_paths,
                    s2_fl_paths,
                    noise_fl_paths,
                    # rir_fl_paths,
                )
            ):

                row = {
                    "ID": i,
                    "duration": 1.0,
                    "mix_wav": mix_path,
                    "mix_wav_format": "wav",
                    "mix_wav_opts": None,
                    "s1_wav": s1_path,
                    "s1_wav_format": "wav",
                    "s1_wav_opts": None,
                    "s2_wav": s2_path,
                    "s2_wav_format": "wav",
                    "s2_wav_opts": None,
                    "noise_wav": noise_path,
                    "noise_wav_format": "wav",
                    "noise_wav_opts": None,
                    # "rir_t": rir_path,
                    # "rir_format": ".t",
                    # "rir_opts": None,
                }
                writer.writerow(row)


def create_whamr_rir_csv(datapath, savepath):
    """
    This function creates the csv files to get the data loaders for the whamr  dataset.

    Arguments:
        datapath (str) : path for the whamr rirs.
        savepath (str) : path where we save the csv file
    """

    csv_columns = ["ID", "duration", "wav", "wav_format", "wav_opts"]

    files = os.listdir(datapath)
    all_paths = [os.path.join(datapath, fl) for fl in files]

    with open(savepath + "/whamr_rirs.csv", "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for i, wav_path in enumerate(all_paths):

            row = {
                "ID": i,
                "duration": 2.0,
                "wav": wav_path,
                "wav_format": "wav",
                "wav_opts": None,
            }
            writer.writerow(row)
