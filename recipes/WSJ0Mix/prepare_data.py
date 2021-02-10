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
from tqdm import tqdm
from speechbrain.dataio.dataio import read_audio, write_audio
from speechbrain.utils.data_utils import download_file
from scipy.io import wavfile
from scipy import signal
import pickle
import csv


def prepare_wsjmix(datapath, savepath, n_spks=2, skip_prep=False):
    """
    Prepared wsj2mix if n_spks=2 and wsj3mix if n_spks=3.

    Arguments:
    ----------
        datapath (str) : path for the wsj0-mix dataset.
        savepath (str) : path where we save the csv file.
        n_spks (int): number of speakers
        skip_prep (bool): If True, skip data preparation
    """
    if skip_prep:
        return
    if n_spks == 2:
        create_wsj_csv(datapath, savepath)
    if n_spks == 3:
        create_wsj_csv_3spks(datapath, savepath)


# load or create the csv files for the data
def create_wsj_csv(datapath, savepath):
    """
    This function creates the csv files to get the speechbrain data loaders.

    Arguments:
        datapath (str) : path for the wsj0-mix dataset.
        savepath (str) : path where we save the csv file
    """
    for set_type in ["tr", "cv", "tt"]:
        mix_path = os.path.join(datapath, "wav8k/min/" + set_type + "/mix/")
        s1_path = os.path.join(datapath, "wav8k/min/" + set_type + "/s1/")
        s2_path = os.path.join(datapath, "wav8k/min/" + set_type + "/s2/")

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

        with open(savepath + "/wsj_" + set_type + ".csv", "w") as csvfile:
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


def create_wsj_csv_3spks(datapath, savepath):
    """
    This function creates the csv files to get the speechbrain data loaders.
    Arguments:
        datapath (str) : path for the wsj0-mix dataset.
        savepath (str) : path where we save the csv file
    """
    for set_type in ["tr", "cv", "tt"]:
        mix_path = os.path.join(datapath, "wav8k/min/" + set_type + "/mix/")
        s1_path = os.path.join(datapath, "wav8k/min/" + set_type + "/s1/")
        s2_path = os.path.join(datapath, "wav8k/min/" + set_type + "/s2/")
        s3_path = os.path.join(datapath, "wav8k/min/" + set_type + "/s3/")

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

        with open(savepath + "/wsj_" + set_type + ".csv", "w") as csvfile:
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


def arrange_task_files(TaskFile, min_max, data_type, log_dir):
    """
    This function gets the specifications on on what file to read
    and also opens the files for the logs.

    Arguments:
        TaskFile (str): The path to the file that specifies the sources.
        min_max (list): Specifies whether we use min. or max. of the sources,
                        while creating mixtures
        data_type (list): Specifies which set to create, in ['tr', 'cv', 'tt']
        log_dir (str): The string which points to the logs for data creation.
    """
    with open(TaskFile, "r") as fid:
        lines = fid.read()
        C = []

        for i, line in enumerate(lines.split("\n")):
            # print(i)
            if not len(line) == 0:
                C.append(line.split())

    Source1File = os.path.join(
        log_dir, "mix_2_spk_" + min_max + "_" + data_type + "_1"
    )
    Source2File = os.path.join(
        log_dir, "mix_2_spk_" + min_max + "_" + data_type + "_2"
    )
    MixFile = os.path.join(
        log_dir, "mix_2_spk_" + min_max + "_" + data_type + "_mix"
    )
    return Source1File, Source2File, MixFile, C


def get_wsj_files(wsj0root, output_dir, save_fs="wav8k", min_maxs=["min"]):
    """
    This function constructs the wsj0-2mix dataset out of wsj0 dataset.
    (We are assuming that we have the wav files and not the sphere format)

    Argument:
        wsj0root (str): This string specifies the root folder for the wsj0 dataset.
        output_dir (str): The string that species the save folder.
        save_fs (str): The string that specifies the saving sampling frequency, in ['wav8k', 'wav16k']
        min_maxs (list): The list that contains the specification on whether we take min. or max. of signals
                         to construct the mixtures. example: ["min", "max"]
    """

    data_types = ["tr", "cv", "tt"]  # train, valid and test sets

    from oct2py import octave

    filedir = os.path.dirname(os.path.realpath(__file__))
    octave.addpath(
        filedir + "/meta"
    )  # add the matlab functions to octave dir here

    fs_read = 8000 if save_fs == "wav8k" else 16000

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if not os.path.exists(os.path.join(output_dir, save_fs)):
        os.mkdir(os.path.join(output_dir, save_fs))

    log_dir = os.path.join(output_dir, save_fs + "/mixture_definitions_log")
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    # get the the text files in the current working directory
    filelinks = [
        "https://www.dropbox.com/s/u5gk5h3htzw4cgo/mix_2_spk_tr.txt?dl=1",
        "https://www.dropbox.com/s/s3s6311d95n4sip/mix_2_spk_cv.txt?dl=1",
        "https://www.dropbox.com/s/9kdxb2uz18a5k9d/mix_2_spk_tt.txt?dl=1",
    ]
    for filelink, data_type in zip(filelinks, data_types):
        filepath = os.path.join(
            filedir, "meta", "mix_2_spk_" + data_type + ".txt"
        )
        if not os.path.exists(filepath):
            download_file(filelink, filepath)

    inner_folders = ["s1", "s2", "mix"]
    for min_max in min_maxs:
        for data_type in data_types:
            save_dir = os.path.join(
                output_dir, save_fs + "/" + min_max + "/" + data_type
            )

            if not os.path.exists(
                os.path.join(output_dir, save_fs + "/" + min_max)
            ):
                os.mkdir(os.path.join(output_dir, save_fs + "/" + min_max))

            if not os.path.exists(save_dir):
                os.mkdir(save_dir)

            for inner_folder in inner_folders:
                if not os.path.exists(os.path.join(save_dir, inner_folder)):
                    os.mkdir(os.path.join(save_dir, inner_folder))

            TaskFile = os.path.join(
                filedir, "meta", "mix_2_spk_" + data_type + ".txt"
            )
            Source1File, Source2File, MixFile, C = arrange_task_files(
                TaskFile, min_max, data_type, log_dir
            )

            fid_s1 = open(Source1File, "w")
            fid_s2 = open(Source2File, "w")
            fid_m = open(MixFile, "w")

            num_files = len(C)

            print("{} \n".format(min_max + "_" + data_type))

            for i, line in tqdm(enumerate(C)):

                _, inwav1_dir, _, inwav1_name = line[0].split("/")
                _, inwav2_dir, _, inwav2_name = line[2].split("/")

                # write the log data to the log files
                fid_s1.write("{}\n".format(line[0]))
                fid_s2.write("{}\n".format(line[2]))

                inwav1_snr = line[1]
                inwav2_snr = line[3]

                mix_name = (
                    inwav1_name
                    + "_"
                    + str(inwav1_snr)
                    + "_"
                    + inwav2_name
                    + "_"
                    + str(inwav2_snr)
                )
                fid_m.write("{}\n".format(mix_name))

                fs, _ = wavfile.read(os.path.join(wsj0root, line[0]))
                s1 = read_audio(os.path.join(wsj0root, line[0]))
                s2 = read_audio(os.path.join(wsj0root, line[2]))

                # resample, determine levels for source 1
                s1_8k = signal.resample(s1, int((fs_read / fs) * len(s1)))
                out = octave.activlev(s1_8k, fs_read, "n")
                s1_8k, lev1 = out[:-1].squeeze(), out[-1]
                # print('lev1 {}'.format(lev1))

                # resample, determine levels for source 2
                s2_8k = signal.resample(s2, int((fs_read / fs) * len(s2)))
                out = octave.activlev(s2_8k, fs_read, "n")
                s2_8k, lev2 = out[:-1].squeeze(), out[-1]

                weight_1 = 10 ** (float(inwav1_snr) / 20)
                weight_2 = 10 ** (float(inwav2_snr) / 20)

                # apply same gain to 16 kHz file
                if save_fs == "wav8k":
                    s1_8k = weight_1 * s1_8k
                    s2_8k = weight_2 * s2_8k

                    scaling_8k, scaling16bit_8k = save_mixture(
                        s1_8k,
                        s2_8k,
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
                    )
                elif save_fs == "wav16k":
                    s1_16k = weight_1 * s1 / np.sqrt(lev1)
                    s2_16k = weight_2 * s2 / np.sqrt(lev2)

                    scaling_16k, scaling16bit_16k = save_mixture(
                        s1_16k,
                        s2_16k,
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
                    )
                else:
                    raise ValueError("Incorrect sampling frequency for saving")

            if save_fs == "wav8k":
                pickle.dump(
                    {
                        "scaling_8k": scaling_8k,
                        "scaling8bit_8k": scaling16bit_8k,
                    },
                    open(
                        output_dir
                        + "/"
                        + save_fs
                        + "/"
                        + min_max
                        + "/"
                        + data_type
                        + "/scaling.pkl",
                        "wb",
                    ),
                )
            elif save_fs == "wav16k":
                pickle.dump(
                    {
                        "scaling_16k": scaling_16k,
                        "scaling16bit_16k": scaling16bit_16k,
                    },
                    open(
                        output_dir
                        + "/"
                        + save_fs
                        + "/"
                        + min_max
                        + "/"
                        + data_type
                        + "/scaling.pkl",
                        "wb",
                    ),
                )
            else:
                raise ValueError("Incorrect sampling frequency for saving")


if __name__ == "__main__":
    wsj0root = "/network/tmp1/subakany/wsj0-mix"
    output_dir = "."
    get_wsj_files(wsj0root, output_dir)
