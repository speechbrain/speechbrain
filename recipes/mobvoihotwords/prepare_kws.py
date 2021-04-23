"""
Data preparation for mobvoihotwords.


Author
------
Wang Wei

"""

import os
from os import walk
import glob
import shutil
import logging
import torch
import re
import hashlib
import copy
import numpy as np
from speechbrain.utils.data_utils import download_file
from speechbrain.dataio.dataio import read_audio
from os.path import join, dirname, abspath
import soundfile as sf
import json
import librosa
from tqdm import tqdm
import wave
from wavinfo import WavInfoReader

try:
    import pandas as pd
except ImportError:
    err_msg = (
        "The optional dependency pandas must be installed to run this recipe.\n"
    )
    err_msg += "Install using `pip install pandas`.\n"
    raise ImportError(err_msg)

logger = logging.getLogger(__name__)


def load_json(file_name):
    with open(file_name, "r+") as file:
        return json.load(file)

def save_json(obj, file_name):
    with open(file_name, "w+") as file:
        json.dump(obj, file, indent=4, sort_keys=False)

def glob_all(folder: str, filt: str) -> list:
    """Recursive glob"""
    import os
    import fnmatch
    matches = []
    for root, dirnames, filenames in os.walk(folder, followlinks=True):
        for filename in fnmatch.filter(filenames, filt):
            matches.append(os.path.join(root, filename))
    return matches


def find_wavs(folder: str) -> list:
    """Finds all wavs in folder"""
    wavs = glob_all(folder, '*.wav')

    return wavs

def load_resource(resource_path) -> dict :
    """[summary]

    Parameters
    ----------
    resource_path : [type]
        [description]

    Returns
    -------
    [type]
        [description]

    Examples
    --------
    """
    datasets = {'train':[], 'dev': [], 'test':[]}
    for dataset in ['train', 'dev', 'test']:
        for label in ['p', 'n']:
            json_path = os.path.join(resource_path, label + '_' + dataset + '.json')
            data = load_json(json_path)
            datasets[dataset].extend(data)

    return datasets


def prepare_kws(
    data_folder='/home/wangwei/work/corpus/kws/speechbrain/himia',
    save_folder='results/save',
    words_wanted=[
        "hixiaowen",
        "nihaowenwen",
        "unknown"
    ],
    skip_prep=False,
):
    """
    Prepares the Google Speech Commands V2 dataset.

    Arguments
    ---------
    data_folder : str
        path to dataset. If not present, it will be downloaded here.
    save_folder: str
        folder where to store the data manifest files.
    skip_prep: bool
        If True, skip data preparation.

    Example
    -------
    >>> data_folder = '/path/to/GSC'
    >>> prepare_GSC(data_folder)
    """
    if skip_prep:
        return

    datasets = load_resource(data_folder + '_resources')

    keywords = {
        0: 'hixiaowen',
        1: 'nihaowenwen',
        -1: 'unknown'
    }
    fields = {
        "ID": [],
        "duration": [],
        "start": [],
        "stop": [],
        "wav": [],
        "spk_id": [],
        "command": [],
        # "transcript": [],
    }

    splits = {
        "train": copy.deepcopy(fields),
        "dev": copy.deepcopy(fields),
        "test": copy.deepcopy(fields),
    }

    for dataset in ['train', 'dev', 'test']:
        for utt_dict in tqdm(datasets[dataset]):
            splits[dataset]['ID'].append(utt_dict['utt_id'])
            splits[dataset]['spk_id'].append(utt_dict['speaker_id'])

            wav = os.path.join(data_folder, utt_dict['utt_id'] + '.wav')
            splits[dataset]["wav"].append(wav)

            fs = librosa.get_samplerate(wav)
            assert fs == 16000

            duration = librosa.get_duration(filename=wav, sr=fs)
            sig_len = int(duration*fs)
            splits[dataset]["duration"].append(duration)

            splits[dataset]["start"].append(0)
            stop_len = sig_len if sig_len < 24000 else 24000
            splits[dataset]["stop"].append(stop_len)

            splits[dataset]["command"].append(keywords[utt_dict['keyword_id']])

            # fs = librosa.get_samplerate(wav)
            # with wave.open(wav, 'r') as f:
            #     fs = f.getframerate()
            #     assert fs == 16000

            #     n_frames = f.getnframes()
            #     duration = float(n_frames / fs)

            #     # duration = librosa.get_duration(filename=wav, sr=fs)
            #     sig_len = int(duration*fs)
            #     splits[dataset]["duration"].append(duration)

            #     splits[dataset]["start"].append(0)
            #     stop_len = sig_len if sig_len < 24000 else 24000
            #     splits[dataset]["stop"].append(stop_len)

            #     splits[dataset]["command"].append(keywords[utt_dict['keyword_id']])

            info = WavInfoReader(wav)
            fs = info.fmt.sample_rate
            assert fs == 16000
            n_frames = info.data.frame_count
            duration = float(n_frames / fs)

            splits[dataset]["duration"].append(duration)

            splits[dataset]["start"].append(0)
            stop_len = n_frames if n_frames < 24000 else 24000
            splits[dataset]["stop"].append(stop_len)
            splits[dataset]["command"].append(keywords[utt_dict['keyword_id']])

        print('datasets[{}]:{}'.format(dataset, len(datasets[dataset])))


    for split in splits:
        os.makedirs(save_folder, exist_ok=True)
        new_filename = os.path.join(save_folder, split) + ".csv"
        if os.path.exists(new_filename):
            print("{} exists, skip...".format(new_filename))
            continue
        new_df = pd.DataFrame(splits[split])
        new_df.to_csv(new_filename, index=False)


    return

    # # If the data folders do not exist, we need to extract the data
    # if not os.path.isdir(os.path.join(data_folder, "train-synth")):
    #     # Check for zip file and download if it doesn't exist
    #     tar_location = os.path.join(data_folder, "speech_commands_v0.02.tar.gz")
    #     if not os.path.exists(tar_location):
    #         download_file(GSC_URL, tar_location, unpack=True)
    #     else:
    #         download_file(GSC_URL, tar_location)
    #         # logger.info("Extracting speech_commands_v0.02.tar.gz...")
    #         # shutil.unpack_archive(tar_location, data_folder)

    # Define the words that we do not want to identify
    unknown_words = list(np.setdiff1d(all_words, words_wanted))
    print(unknown_words)

    # All metadata fields to appear within our dataset annotation files (i.e. train.csv, valid.csv, test.cvs)
    fields = {
        "ID": [],
        "duration": [],
        "start": [],
        "stop": [],
        "wav": [],
        "spk_id": [],
        "command": [],
        "transcript": [],
    }

    splits = {
        "train": copy.deepcopy(fields),
        "valid": copy.deepcopy(fields),
        "test": copy.deepcopy(fields),
    }

    num_known_samples_per_split = {"train": 0, "valid": 0, "test": 0}
    words_wanted_parsed = False
    commands = words_wanted + unknown_words
    logger.info("commands:{}".format(commands))
    print("commands:{}".format(commands))

    data_type = ['train', 'dev', 'test']
    data_type_gsc = ['train', 'valid', 'test']

    for index, data_set in enumerate(data_type):
        # if data_set is not 'test':
        #     continue
        for i, command in enumerate(commands):
            # Read all files under a specific class (i.e. command)
            files = []
            filenames = find_wavs(os.path.join(data_folder, data_set, command))
            files.extend(filenames)
            print("{}/{}:{}".format(data_set, command, len(filenames)))

            # Fill in all fields with metadata for each audio sample file under a specific class
            for filename in files:
                # print(filename)

                # select the required split (i.e. set) for the sample
                split = data_type_gsc[index]

                splits[split]["ID"].append(
                    command + "/" + re.sub(r".wav", "", filename)
                )

                wav_path = os.path.join(data_folder, data_set, command, filename)

                sig, sr = sf.read(wav_path)
                duration = float(len(sig)) / sr

                # We know that all recordings are 1 second long (i.e.16000 frames). No need to compute the duration.
                splits[split]["duration"].append(duration)
                splits[split]["start"].append(0)
                stop_len = len(sig) if len(sig) < 24000 else 24000
                splits[split]["stop"].append(stop_len)

                splits[split]["wav"].append(
                    os.path.join(data_folder, data_set, command, filename)
                )

                splits[split]["spk_id"].append(re.sub(r"_.*", "", filename))

                if command in words_wanted:
                    splits[split]["command"].append(command)

                    num_known_samples_per_split[split] += 1
                else:
                    splits[split]["command"].append("unknown")

                splits[split]["transcript"].append(command)


    for split in splits:
        new_filename = os.path.join(save_folder, split) + ".csv"
        if os.path.exists(new_filename):
            print("{} exists, skip...".format(new_filename))
            continue
        new_df = pd.DataFrame(splits[split])
        new_df.to_csv(new_filename, index=False)

if __name__ == "__main__":
    import argparse

    data_folder = '/home/wangwei/diskdata/corpus/kws/mobvoi_hotword_dataset/mobvoi_hotword_dataset'

    # load_resource(data_folder + '_resources')

    prepare_kws(data_folder=data_folder,
                  save_folder='results/save',
                  percentage_silence=0,
                  words_wanted=["wake-word"])
