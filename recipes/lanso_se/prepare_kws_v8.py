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
from typing import Coroutine
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
import random

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


def move_to_folder(data_folder='/home/wangwei/work/corpus/kws/lanso/LS-ASR-data_16k_trim_command_folder_distance_rnd2',
                save_folder='results/save',
                words_wanted=['小蓝小蓝', '管家管家', '物业物业'],
                skip_prep=False,
                use_others=True):

    if skip_prep:
        print('skipping prepare kws!')
        return

    dataset_csv = os.path.join(save_folder, 'dataset') + ".csv"
    if os.path.exists(dataset_csv):
        print('dataset csv exist...')
        datasets = pd.read_csv(dataset_csv)
        all_spk = []
        for spk in list(datasets['spk_id']):
            if spk not in all_spk:
                all_spk.append(spk)
    # shuffle speaker
    np.random.shuffle(all_spk)

    train_percentage = 80
    dev_percentage = 18
    test_percentage = 2

    data_split = ['train', 'dev', 'test']

    new_path_root = data_folder

    for i in tqdm(range(len(datasets['wav']))):
        spk_index = all_spk.index(datasets['spk_id'][i])
        if spk_index < int(train_percentage / 100.0 * len(all_spk)):
            split = 0  # train
        elif spk_index > int((train_percentage + dev_percentage) / 100.0 * len(all_spk)):
            split = 2  # test
        else:
            split = 1  # dev
        if datasets['command'][i] in words_wanted:
            if random.random() < 1:
                old_path = datasets['wav'][i]
                distance = datasets['ID'][i].split('-')[-2]
                assert distance in ['00', '01', '03', '04', '05'], print("{},{} has no distance info".format(datasets['command'][i], old_path))
                new_path = os.path.join(new_path_root, data_split[split], datasets['command'][i], distance)
                new_fullname = os.path.join(new_path, datasets['ID'][i] + '.wav')
                if os.path.exists(new_path) is not True:
                    os.makedirs(new_path, exist_ok=True)
                shutil.copy(old_path, new_fullname)


def dataset_summary(data_path, save_folder):
    """[summary]

    Parameters
    ----------
    data_path : str
        dataset path
    save_folder : str
        where to save csv file

    Returns
    -------
    dict
        [description]
    """

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

    dataset = fields.copy()

    all_wavs = find_wavs(data_path)
    print("find wavs:{}".format(len(all_wavs)))

    all_words = []
    all_words_dict = {}    # keyword to num
    all_spks = []
    all_spks_num = []      # spk to num
    utt2spk = {}
    spk2utt = {}
    keyword_to_spk = []
    keyword_to_spk = {}

    # iteration
    for fullname in tqdm(all_wavs):
        txt_name = fullname[:-3] + 'txt'

        filename = os.path.basename(fullname)
        spk = filename.split('-')[0]
        utt_id = filename[:-4]

        dataset['spk_id'].append(spk)   # speaker
        dataset['ID'].append(utt_id)                # no extension
        fullname_trim = fullname.replace("LS-ASR-data", "LS-ASR-data_16k_trim", 1)
        dataset['wav'].append(fullname_trim)                    # fullpath

        keyword = str(np.genfromtxt(txt_name, dtype='str'))

        length = 24000       # audio already trimed to 1.5s

        dataset['command'].append(keyword)                    # command
        # dataset['transcript'].append(keyword)                    # command
        dataset['duration'].append(1.5)
        dataset['start'].append(0)
        dataset['stop'].append(24000)

        if spk not in all_spks:
            all_spks.append(spk)

        if keyword not in all_words:            # new keyword founded
            all_words.append(keyword)
            all_words_dict[keyword] = 0

            keyword_to_spk[keyword] = []
            keyword_to_spk[keyword].append(spk)

        else:
            all_words_dict[keyword] += 1

            if spk not in keyword_to_spk[keyword]:
                keyword_to_spk[keyword].append(spk)

        utt2spk[utt_id] = spk

    info = {}
    info['datasets'] = dataset
    info['all_spk'] = all_spks

    print("find keywords:{}".format(len(all_words)))

    new_df = pd.DataFrame(dataset)
    new_filename = os.path.join(save_folder, 'dataset') + ".csv"
    new_df.to_csv(new_filename, index=False)

    keywords_filename = os.path.join(save_folder, 'keywords') + ".txt"
    with open(keywords_filename, "w") as f:
        for keyword in all_words:
            print("{}:{}".format(keyword, all_words_dict[keyword]))
            f.writelines(keyword + '\n')    # write to file

    print("find speakers:{}".format(len(all_spks)))
    for keyword in keyword_to_spk.keys():
        print("{}:{}".format(keyword, len(keyword_to_spk[keyword])))

    return info


def prepare_kws_v8(data_folder='/home/wangwei/work/corpus/kws/lanso/LS-ASR-data_16k_trim_command',
                save_folder='results/save',
                words_wanted=['小蓝小蓝', '管家管家', '物业物业'],
                skip_prep=False,
                use_others=True):

    if skip_prep:
        print('skipping prepare kws!')
        return

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

    data_split = ['train', 'dev', 'test']

    for split in data_split:
        for command in words_wanted:
            word_path = os.path.join(data_folder, split, command)
            word_wavs = find_wavs(word_path)
            print("find {} {} {} wavs".format(split, command, len(word_wavs)))

            for fullname in tqdm(word_wavs):
                filename = os.path.basename(fullname)
                spk = filename.split('-')[0]
                utt_id = filename[:-4]

                splits[split]['command'].append(command)

                splits[split]['ID'].append(utt_id)
                splits[split]['duration'].append(1.5)
                splits[split]['start'].append(0)
                splits[split]['stop'].append(24000)
                splits[split]['wav'].append(fullname)
                splits[split]['spk_id'].append(spk)

    for split in data_split:
        new_filename = os.path.join(save_folder, split) + ".csv"
        new_df = pd.DataFrame(splits[split])
        new_df.to_csv(new_filename, index=False)


if __name__ == "__main__":
    import argparse

    data_folder = '/home/wangwei/work/corpus/kws/lanso/LS-ASR-data_16k_trim_command_folder_distance_rnd2'

    # move_to_folder(data_folder=data_folder,
    #             save_folder='results/save7',
    #             words_wanted=['小蓝小蓝', '管家管家', '物业物业'],
    #             skip_prep=False)

    save_folder = 'results/save8'

    prepare_kws_v8(data_folder=data_folder, save_folder=save_folder,
                words_wanted=['小蓝小蓝', '管家管家', '物业物业'], use_others=False)


