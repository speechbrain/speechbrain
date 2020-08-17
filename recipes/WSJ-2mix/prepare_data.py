# This script is based on create_wav_2_speakers.m from wsj0-mix dataset.
# This script creates mixtures from wsj0 dataset.
# Create 2-speaker mixtures
#
# This script assumes that WSJ0's wv1 sphere files have already
# been converted to wav files, using the original folder structure
# under wsj0/, e.g.,
# 11-1.1/wsj0/si_tr_s/01t/01to030v.wv1 is converted to wav and
# stored in YOUR_PATH/wsj0/si_tr_s/01t/01to030v.wav, and
# 11-6.1/wsj0/si_dt_05/050/050a0501.wv1 is converted to wav and
# stored in YOUR_PATH/wsj0/si_dt_05/050/050a0501.wav.
# Relevant data from all disks are assumed merged under YOUR_PATH/wsj0/
#
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#    Copyright (C) 2016 Mitsubishi Electric Research Labs
#                           (Jonathan Le Roux, John R. Hershey, Zhuo Chen)
#    Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import os
import numpy as np
from tqdm import tqdm
from speechbrain.data_io.data_io import read_wav_soundfile, write_wav_soundfile
from oct2py import octave
from scipy.io import wavfile
from scipy import signal
import pickle

data_types = ["tr", "cv", "tt"]
wsj0root = "/network/tmp1/subakany/wsj0-mix/"  # YOUR_PATH/, the folder containing wsj0/

task_dir = "/network/tmp1/subakany/SB-create-speaker-mixtures/"  # this is the path for the TaskFile .txt files, which contains the information on which files to mix
octave.addpath(os.getcwd())  # add the matlab functions to octave dir here

min_maxs = ["min"]  # in ['max', 'min'] if max,
save_fs = "wav_8k"
fs_read = 8000 if save_fs == "wav_8k" else 16000

if save_fs == "wav_8k":
    output_dir = "/network/tmp1/subakany/wsj0-mix-speechbrain-2speakers"
elif save_fs == "wav_16k":
    output_dir = "/network/tmp1/subakany/wsj0-mix-speechbrain-2speakers"
else:
    raise ValueError("Wrong frequency type")

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

if not os.path.exists(task_dir):
    os.mkdir(task_dir)
os.system("cp {}/*.txt {}".format(os.getcwd(), task_dir))

inner_folders = ["/s1", "/s2", "/mix"]
for min_max in min_maxs:
    for data_type in data_types:
        save_dir = output_dir + "/" + save_fs + "/" + min_max + "/" + data_type

        if not os.path.exists(output_dir + "/" + save_fs):
            os.mkdir(output_dir + "/" + save_fs)

        if not os.path.exists(output_dir + "/" + save_fs + "/" + min_max):
            os.mkdir(output_dir + "/" + save_fs + "/" + min_max)

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        for inner_folder in inner_folders:
            if not os.path.exists(save_dir + inner_folder):
                os.mkdir(save_dir + inner_folder)

        TaskFile = task_dir + "mix_2_spk_" + data_type + ".txt"
        with open(TaskFile, "r") as fid:
            lines = fid.read()
            C = []
            for i, line in enumerate(lines.split("\n")):
                if not len(line) == 0:
                    C.append(line.split())

        Source1File = task_dir + "mix_2_spk_" + min_max + "_" + data_type + "_1"
        Source2File = task_dir + "mix_2_spk_" + min_max + "_" + data_type + "_2"
        MixFile = task_dir + "mix_2_spk_" + min_max + "_" + data_type + "_mix"

        fid_s1 = open(Source1File, "w")
        fid_s2 = open(Source2File, "w")
        fid_m = open(MixFile, "w")

        num_files = len(C)

        scaling_16k = np.zeros((num_files, 2))
        scaling_8k = np.zeros((num_files, 2))
        scaling16bit_16k = np.zeros((num_files, 1))
        scaling16bit_8k = np.zeros((num_files, 1))
        print("{} \n".format(min_max + "_" + data_type))

        for i, line in tqdm(enumerate(C)):

            _, inwav1_dir, invwav2_ext, invwav1_name = line[0].split("/")
            _, inwav2_dir, invwav2_ext, invwav2_name = line[2].split("/")

            fid_s1.write("{}\n".format(line[0]))
            fid_s2.write("{}\n".format(line[2]))

            inwav1_snr = line[1]
            inwav2_snr = line[3]

            mix_name = (
                invwav1_name
                + "_"
                + str(inwav1_snr)
                + "_"
                + invwav2_name
                + "_"
                + str(inwav2_snr)
            )
            fid_m.write("{}\n".format(mix_name))

            fs, _ = wavfile.read(wsj0root + line[0])
            s1 = read_wav_soundfile(wsj0root + line[0])
            s2 = read_wav_soundfile(wsj0root + line[2])

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

            s1_8k = weight_1 * s1_8k
            s2_8k = weight_2 * s2_8k

            if save_fs == "wav_8k":

                if min_max == "max":
                    mix_len = max(s1_8k.shape[0], s2_8k.shape[0])

                    s1_8k = np.pad(
                        s1_8k,
                        (0, mix_len - s1_8k.shape[0]),
                        "constant",
                        constant_values=(0, 0),
                    )
                    s2_8k = np.pad(
                        s2_8k,
                        (0, mix_len - s2_8k.shape[0]),
                        "constant",
                        constant_values=(0, 0),
                    )
                else:
                    mix_len = min(s1_8k.shape[0], s2_8k.shape[0])

                    s1_8k = s1_8k[:mix_len]
                    s2_8k = s2_8k[:mix_len]

                mix_8k = s1_8k + s2_8k

                max_amp_8k = max(
                    np.abs(mix_8k).max(),
                    np.abs(s1_8k).max(),
                    np.abs(s2_8k).max(),
                )
                mix_scaling_8k = 1 / max_amp_8k * 0.9
                s1_8k = mix_scaling_8k * s1_8k
                s2_8k = mix_scaling_8k * s2_8k
                mix_8k = mix_scaling_8k * mix_8k

                scaling_8k[i, 0] = weight_1 * mix_scaling_8k / np.sqrt(lev1)
                scaling_8k[i, 1] = weight_2 * mix_scaling_8k / np.sqrt(lev2)
                scaling16bit_8k[i] = mix_scaling_8k

                write_wav_soundfile(
                    s1_8k,
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
                    sampling_rate=8000,
                )
                write_wav_soundfile(
                    s2_8k,
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
                    sampling_rate=8000,
                )
                write_wav_soundfile(
                    mix_8k,
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
                    sampling_rate=8000,
                )
            elif save_fs == "wav_16k":
                # apply same gain to 16 kHz file
                s1_16k = weight_1 * s1 / np.sqrt(lev1)
                s2_16k = weight_2 * s2 / np.sqrt(lev2)

                if min_max == "max":
                    mix_16k_len = max(s1_16k.shape[0], s2_16k.shape[0])

                    s1_16k = np.pad(
                        s1_16k,
                        (0, mix_16k_len - s1_16k.shape[0]),
                        "constant",
                        constant_values=(0, 0),
                    )
                    s2_16k = np.pad(
                        s2_16k,
                        (0, mix_16k_len - s2_16k.shape[0]),
                        "constant",
                        constant_values=(0, 0),
                    )
                else:
                    mix_16k_len = min(s1_16k.shape[0], s2_16k.shape[0])

                    s1_16k = s1_16k[:mix_16k_len]
                    s2_16k = s2_16k[:mix_16k_len]

                mix_16k = s1_16k + s2_16k

                max_amp_16k = max(
                    np.abs(mix_16k).max(),
                    np.abs(s1_16k).max(),
                    np.abs(s2_16k).max(),
                )
                mix_scaling_16k = 1 / max_amp_16k * 0.9
                s1_16k = mix_scaling_16k * s1_16k
                s2_16k = mix_scaling_16k * s2_16k
                mix_16k = mix_scaling_16k * mix_16k

                # the mixtures, as well as necessary scaling factors

                scaling_16k[i, 0] = weight_1 * mix_scaling_16k / np.sqrt(lev1)
                scaling_16k[i, 1] = weight_2 * mix_scaling_16k / np.sqrt(lev2)
                scaling16bit_16k[i] = mix_scaling_16k

                write_wav_soundfile(
                    s1_16k,
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
                    sampling_rate=16000,
                )
                write_wav_soundfile(
                    s2_16k,
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
                    sampling_rate=16000,
                )
                write_wav_soundfile(
                    mix_16k,
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
                    sampling_rate=16000,
                )
            else:
                raise ValueError("Incorrect sampling frequency for saving")

        if save_fs == "wav_8k":
            pickle.dump(
                {"scaling_8k": scaling_8k, "scaling8bit_8k": scaling16bit_8k},
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
        elif save_fs == "wav_16k":
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
