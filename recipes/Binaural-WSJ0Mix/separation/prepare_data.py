"""
The .csv preperation functions for Binaural-WSJ0Mix.

Author
 * Cem Subakan 2020
 * Zijian

 """

import os
import csv


def prepare_binaural_wsj0mix(
    datapath, savepath, n_spks=2, skip_prep=False, fs=8000, version="min",
):
    """
    Prepared binaural wsj2mix if n_spks=2 and binaural wsj3mix if n_spks=3.

    Arguments:
    ----------
        datapath (str) : path for the binaural-wsj0mix dataset.
        savepath (str) : path where we save the csv file.
        n_spks (int): number of speakers
        skip_prep (bool): If True, skip data preparation
    """

    if skip_prep:
        return

    if n_spks == 2:
        create_binaural_wsj0mix2_csv(datapath, savepath, fs, version)
    elif n_spks == 3:
        create_binaural_wsj0mix3_csv(datapath, savepath, fs, version)
    else:
        raise ValueError("Unsupported Number of Speakers")


def create_binaural_wsj0mix2_csv(
    datapath,
    savepath,
    fs,
    version,
    savename="binaural_wsj0-2mix_",
    set_types=["tr", "cv", "tt"],
):
    """
    This function creates the csv files to get the speechbrain data loaders for the binaural-wsj0mix2 dataset.

    Arguments:
        datapath (str) : path for the binaural-wsj0mix dataset.
        savepath (str) : path where we save the csv file
    """
    if fs == 8000:
        sample_rate = "8k"
    elif fs == 16000:
        sample_rate = "16k"
    else:
        raise ValueError("Unsupported sampling rate")

    for set_type in set_types:

        mix_path = os.path.join(
            datapath, "wav{}".format(sample_rate), version, set_type, "mix/",
        )
        s1_path = os.path.join(
            datapath, "wav{}".format(sample_rate), version, set_type, "s1/",
        )
        s2_path = os.path.join(
            datapath, "wav{}".format(sample_rate), version, set_type, "s2/",
        )

        files = os.listdir(mix_path)

        mix_fl_paths = [mix_path + fl for fl in files]
        s1_fl_paths = [s1_path + fl for fl in files]
        s2_fl_paths = [s2_path + fl for fl in files]

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
        ]

        with open(
            os.path.join(savepath, savename + set_type + ".csv"), "w"
        ) as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for i, (mix_path, s1_path, s2_path) in enumerate(
                zip(mix_fl_paths, s1_fl_paths, s2_fl_paths)
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
                }
                writer.writerow(row)


def create_binaural_wsj0mix3_csv(
    datapath,
    savepath,
    fs,
    version,
    savename="binaural_wsj0-3mix_",
    set_types=["tr", "cv", "tt"],
):
    """
    This function creates the csv files to get the speechbrain data loaders for the binaural-wsj0mix3 dataset.

    Arguments:
        datapath (str) : path for the binaural-wsj0mix dataset.
        savepath (str) : path where we save the csv file
    """
    if fs == 8000:
        sample_rate = "8k"
    elif fs == 16000:
        sample_rate = "16k"
    else:
        raise ValueError("Unsupported sampling rate")

    for set_type in set_types:

        mix_path = os.path.join(
            datapath, "wav{}".format(sample_rate), version, set_type, "mix/",
        )
        s1_path = os.path.join(
            datapath, "wav{}".format(sample_rate), version, set_type, "s1/",
        )
        s2_path = os.path.join(
            datapath, "wav{}".format(sample_rate), version, set_type, "s2/",
        )
        s3_path = os.path.join(
            datapath, "wav{}".format(sample_rate), version, set_type, "s3/",
        )

        files = os.listdir(mix_path)

        mix_fl_paths = [mix_path + fl for fl in files]
        s1_fl_paths = [s1_path + fl for fl in files]
        s2_fl_paths = [s2_path + fl for fl in files]
        s3_fl_paths = [s3_path + fl for fl in files]

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
            "s3_wav",
            "s3_wav_format",
            "s3_wav_opts",
        ]

        with open(
            os.path.join(savepath, savename + set_type + ".csv"), "w"
        ) as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for i, (mix_path, s1_path, s2_path, s3_path) in enumerate(
                zip(mix_fl_paths, s1_fl_paths, s2_fl_paths, s3_fl_paths)
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
                    "s3_wav": s3_path,
                    "s3_wav_format": "wav",
                    "s3_wav_opts": None,
                }
                writer.writerow(row)
