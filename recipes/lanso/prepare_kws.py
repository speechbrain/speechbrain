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


def prepare_kws(data_folder='/home/wangwei/work/corpus/kws/speechbrain/himia',
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
    else:
        info = dataset_summary(data_folder, save_folder)
        datasets = info['datasets']
        all_spk = info['all_spk'].copy()

    # shuffle speaker
    np.random.shuffle(all_spk)

    train_percentage = 60
    dev_percentage = 20
    test_percentage = 20

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

    for i in tqdm(range(len(datasets['wav']))):
        spk_index = all_spk.index(datasets['spk_id'][i])
        if spk_index < int(train_percentage / 100.0 * len(all_spk)):
            split = 0  # train
        elif spk_index > int((train_percentage + dev_percentage) / 100.0 * len(all_spk)):
            split = 2  # test
        else:
            split = 1  # dev
        if datasets['command'][i] in words_wanted:
            splits[data_split[split]]['command'].append(datasets['command'][i])
        elif use_others:
            splits[data_split[split]]['command'].append('unknown')
        else:
            continue
        # print(splits[data_split[split]]['command'])

        splits[data_split[split]]['ID'].append(datasets['ID'][i])
        splits[data_split[split]]['duration'].append(datasets['duration'][i])
        splits[data_split[split]]['start'].append(datasets['start'][i])
        splits[data_split[split]]['stop'].append(datasets['stop'][i])
        splits[data_split[split]]['wav'].append(datasets['wav'][i])
        splits[data_split[split]]['spk_id'].append(datasets['spk_id'][i])
        # if datasets['command'][i] in words_wanted:
        #     splits[data_split[split]]['command'].append(datasets['command'][i])
        # else:
        #     splits[data_split[split]]['command'].append('unknown')
        # splits[data_split[split]]['transcript'].append(
        #     datasets['transcript'][i])
        # print(len(splits[data_split[split]]))

    for split in data_split:
        new_filename = os.path.join(save_folder, split) + ".csv"
        if os.path.exists(new_filename):
            print("{} exists, skip...".format(new_filename))
            continue
        new_df = pd.DataFrame(splits[split])
        new_df.to_csv(new_filename, index=False)


if __name__ == "__main__":
    import argparse

    data_folder = '/home/wangwei/work/corpus/kws/lanso/LS-ASR-data'

    prepare_kws(data_folder=data_folder, save_folder='results/save1',
                words_wanted=['小蓝小蓝', '管家管家', '物业物业'], use_others=False)

    # combine with mobvoihotwords negative samples
    mobvoi_path = '/home/wangwei/work/speechbrain/recipes/mobvoihotwords/results/save'
    data_split = ['train', 'dev', 'test']
    for split in data_split:
        mobvoi_split_path = os.path.join(mobvoi_path, split + '.csv')
        datasets = pd.read_csv(mobvoi_split_path)
        datasets_unk = datasets[datasets['command'].str.contains('unknown')]
        unk_csv = datasets.loc[datasets_unk.index]
        print("unknown {}:{}".format(split, len(unk_csv)))

        target_split_path = os.path.join('results/save1', split + '.csv')

        unk_csv.to_csv(target_split_path, mode='a', header=False, index=None)


