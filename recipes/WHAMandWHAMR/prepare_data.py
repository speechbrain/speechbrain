"""
Author
 * Cem Subakan 2020

 This script is based on create_wav_2_speakers.m from wsj0-mix dataset.
 This script creates mixtures from wsj0 dataset.
 Create 2-speaker mixtures
 Note that we use octave to call functions from the voicebox MATLAB toolkit.

 This script assumes that WSJ0's wv1 sphere files have already
 been converted to wav files, using the original folder structure
 under wsj0/, e.g.,
 11-1.1/wsj0/si_tr_s/01t/01to030v.wv1 is converted to wav and
 stored in YOUR_PATH/wsj0/si_tr_s/01t/01to030v.wav, and
 11-6.1/wsj0/si_dt_05/050/050a0501.wv1 is converted to wav and
 stored in YOUR_PATH/wsj0/si_dt_05/050/050a0501.wav.
 Relevant data from all disks are assumed merged under YOUR_PATH/wsj0/

 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Copyright (C) 2016 Mitsubishi Electric Research Labs
                           (Jonathan Le Roux, John R. Hershey, Zhuo Chen)
    Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

import os
import numpy as np
from speechbrain.dataio.dataio import write_audio
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


def save_mixture(
    s1,
    s2,
    min_max,
    weight_1,
    weight_2,
    num_files,
    lev1,
    lev2,
    save_fs,
    output_dir,
    data_type,
    mix_name,
    i,
):
    """
    This function creates the mixtures, and saves them

    Arguments:
        s1, s1 (numpy array): source1 and source2 wav files in numpy array.
        weight_1, weight_2 (float): weights for source1 and source2 respectively.
        num_files (int): number of files
        lev1, lev2 (float): levels for each souce obtained with octave.activlev() function
        save_fs (str): in ['wav8k', 'wav16k']
        output_dir (str): the save directory
        data_type (str): in ['tr', 'cv', 'tt']
        mix_name (str): name given to the mixture. (see the main function get_wsj_files())
        i (int): number of the mixture. (see the main function get_wsj_files())

    """
    scaling = np.zeros((num_files, 2))
    scaling16bit = np.zeros((num_files, 1))

    if min_max == "max":
        mix_len = max(s1.shape[0], s2.shape[0])

        s1 = np.pad(
            s1, (0, mix_len - s1.shape[0]), "constant", constant_values=(0, 0),
        )
        s2 = np.pad(
            s2, (0, mix_len - s2.shape[0]), "constant", constant_values=(0, 0),
        )
    else:
        mix_len = min(s1.shape[0], s2.shape[0])

        s1 = s1[:mix_len]
        s2 = s2[:mix_len]

    mix = s1 + s2

    max_amp = max(np.abs(mix).max(), np.abs(s1).max(), np.abs(s2).max(),)
    mix_scaling = 1 / max_amp * 0.9
    s1 = mix_scaling * s1
    s2 = mix_scaling * s2
    mix = mix_scaling * mix

    scaling[i, 0] = weight_1 * mix_scaling / np.sqrt(lev1)
    scaling[i, 1] = weight_2 * mix_scaling / np.sqrt(lev2)
    scaling16bit[i] = mix_scaling

    sampling_rate = 8000 if save_fs == "wav8k" else 16000

    write_audio(
        s1,
        output_dir
        + "/"
        + save_fs
        + "/"
        + min_max
        + "/"
        + data_type
        + "/s1/"
        + mix_name
        + ".wav",
        sampling_rate=sampling_rate,
    )
    write_audio(
        s2,
        output_dir
        + "/"
        + save_fs
        + "/"
        + min_max
        + "/"
        + data_type
        + "/s2/"
        + mix_name
        + ".wav",
        sampling_rate=sampling_rate,
    )
    write_audio(
        mix,
        output_dir
        + "/"
        + save_fs
        + "/"
        + min_max
        + "/"
        + data_type
        + "/mix/"
        + mix_name
        + ".wav",
        sampling_rate=sampling_rate,
    )
    return scaling, scaling16bit
