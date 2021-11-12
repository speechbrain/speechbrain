"""
The functions to create the .csv files for Aishell1Mix

Author
 * Cem Subakan 2020
"""

import os
import csv


def prepare_aishell1mix(
    datapath,
    savepath,
    n_spks=2,
    skip_prep=False,
    aishell1mix_addnoise=False,
    fs=8000,
):
    """

    Prepare .csv files for aishell1mix

    Arguments:
    ----------
        datapath (str) : path for the wsj0-mix dataset.
        savepath (str) : path where we save the csv file.
        n_spks (int): number of speakers
        skip_prep (bool): If True, skip data preparation
        aishell1mix_addnoise: If True, add whamnoise to aishell1mix datasets
    """

    if skip_prep:
        return

    if "Aishell1" in datapath:
        # Aishell1 Mix2/3 datasets
        if n_spks == 2:
            assert (
                "Aishell1Mix2" in datapath
            ), "Inconsistent number of speakers and datapath"
            create_aishell1mix2_csv(datapath, savepath, addnoise=aishell1mix_addnoise)
        elif n_spks == 3:
            assert (
                "Aishell1Mix3" in datapath
            ), "Inconsistent number of speakers and datapath"
            create_aishell1mix3_csv(datapath, savepath, addnoise=aishell1mix_addnoise)
        else:
            raise ValueError("Unsupported Number of Speakers")
    else:
        raise ValueError("Unsupported Dataset")


def create_aishell1mix2_csv(
    datapath,
    savepath,
    addnoise=False,
    version="wav8k/min/",
    set_types=["train", "dev", "test"],
):
    """
    This functions creates the .csv file for the aishell1mix2 dataset
    """

    for set_type in set_types:
        if addnoise:
            mix_path = os.path.join(datapath, version, set_type, "mix_both/")
        else:
            mix_path = os.path.join(datapath, version, set_type, "mix_clean/")

        s1_path = os.path.join(datapath, version, set_type, "s1/")
        s2_path = os.path.join(datapath, version, set_type, "s2/")
        noise_path = os.path.join(datapath, version, set_type, "noise/")

        files = os.listdir(mix_path)

        mix_fl_paths = [mix_path + fl for fl in files]
        s1_fl_paths = [s1_path + fl for fl in files]
        s2_fl_paths = [s2_path + fl for fl in files]
        noise_fl_paths = [noise_path + fl for fl in files]

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

        with open(savepath + "/aishell1mix2_" + set_type + ".csv", "w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for i, (mix_path, s1_path, s2_path, noise_path) in enumerate(
                zip(mix_fl_paths, s1_fl_paths, s2_fl_paths, noise_fl_paths)
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
                }
                writer.writerow(row)


def create_aishell1mix3_csv(
    datapath,
    savepath,
    addnoise=False,
    version="wav8k/min/",
    set_types=["train", "dev", "test"],
):
    """
    This functions creates the .csv file for the aishell1mix3 dataset
    """

    for set_type in set_types:
        if addnoise:
            mix_path = os.path.join(datapath, version, set_type, "mix_both/")
        else:
            mix_path = os.path.join(datapath, version, set_type, "mix_clean/")

        s1_path = os.path.join(datapath, version, set_type, "s1/")
        s2_path = os.path.join(datapath, version, set_type, "s2/")
        s3_path = os.path.join(datapath, version, set_type, "s3/")
        noise_path = os.path.join(datapath, version, set_type, "noise/")

        files = os.listdir(mix_path)

        mix_fl_paths = [mix_path + fl for fl in files]
        s1_fl_paths = [s1_path + fl for fl in files]
        s2_fl_paths = [s2_path + fl for fl in files]
        s3_fl_paths = [s3_path + fl for fl in files]
        noise_fl_paths = [noise_path + fl for fl in files]

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
            "noise_wav",
            "noise_wav_format",
            "noise_wav_opts",
        ]

        with open(savepath + "/aishell1mix3_" + set_type + ".csv", "w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for (
                i,
                (mix_path, s1_path, s2_path, s3_path, noise_path),
            ) in enumerate(
                zip(
                    mix_fl_paths,
                    s1_fl_paths,
                    s2_fl_paths,
                    s3_fl_paths,
                    noise_fl_paths,
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
                    "s3_wav": s3_path,
                    "s3_wav_format": "wav",
                    "s3_wav_opts": None,
                    "noise_wav": noise_path,
                    "noise_wav_format": "wav",
                    "noise_wav_opts": None,
                }
                writer.writerow(row)
