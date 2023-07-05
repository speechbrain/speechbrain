"""
Data preparation for emotion diarization.

Training set involves Emov-DB, ESD, IEMOCAP, JL-Corpus, RAVDESS
Test set used is Zaion Emotion Dataset

Author
------
Yingzhi Wang 2023
"""

import os
import random
import json
import logging
from datasets.prepare_EMOVDB import prepare_emovdb
from datasets.prepare_ESD import prepare_esd
from datasets.prepare_IEMOCAP import prepare_iemocap
from datasets.prepare_JLCORPUS import prepare_jlcorpus
from datasets.prepare_RAVDESS import prepare_ravdess

logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)


def getOverlap(a, b):
    """ get the overlap length of two intervals

    Arguments
    ---------
    a : list
    b : list
    """
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def get_labels(data, win_len=0.02, stride=0.02):
    """ make labels for training/test

    Arguments
    ---------
    data (dict): a dictionary that contains:
        {
            'wav': path,
            'duration': dur,
            'emotion': [{'emo': emo, 'start': s, 'end': e}],
            'transcription': trans,
        }
    win_len (float): the frame length used for frame-wise prediction
    stride (float): the frame length used for frame-wise prediction

    """
    emo_list = data["emotion"]
    assert len(emo_list) == 1

    duration = data["duration"]
    emotion = data["emotion"][0]["emo"]
    emo_start = data["emotion"][0]["start"]
    emo_end = data["emotion"][0]["end"]

    number_frames = int(duration / stride) + 1

    intervals = []
    labels = []
    if emo_start != 0:
        intervals.append([0.0, emo_start])
        labels.append("n")
    intervals.append([emo_start, emo_end])
    labels.append(emotion[0])
    if emo_end != duration:
        intervals.append([emo_end, duration])
        labels.append("n")

    start = 0.0
    frame_labels = []
    for i in range(number_frames):
        win_start = start + i * stride
        win_end = win_start + win_len

        # make sure that every sample exists in a window
        if win_end >= duration:
            win_end = duration
            win_start = max(duration - win_len, 0)

        for j in range(len(intervals)):
            if getOverlap([win_start, win_end], intervals[j]) >= 0.5 * (
                win_end - win_start
            ):
                emo_frame = labels[j]
                break
        frame_labels.append(emo_frame)
        if win_end >= duration:
            break
    return intervals, labels, frame_labels


def prepare_train(
    save_json_train,
    save_json_valid,
    save_json_test=None,
    split_ratio=[80, 20],
    win_len=0.02,
    stride=0.02,
    seed=12,
    emovdb_folder=None,
    esd_folder=None,
    iemocap_folder=None,
    jlcorpus_folder=None,
    ravdess_folder=None,
):
    """ training sets preparation

    Args:
        save_json_train (str): train json path
        save_json_valid (str): valid json path
        save_json_test (str, None): Defaults to None for the current recipe.
        split_ratio (list, optional): train/valid split. Defaults to [80, 20].
        win_len (float, optional):
            window length for generating emotion labels. Defaults to 0.02.
        stride (float, optional):
            stride for generating emotion labels. Defaults to 0.02.
        seed (int, optional): train/valid test. Defaults to 12.
        emovdb_folder (str, optional): path to dataset. Defaults to None.
        esd_folder (str, optional): path to dataset. Defaults to None.
        iemocap_folder (str, optional): path to dataset. Defaults to None.
        jlcorpus_folder (str, optional): path to dataset. Defaults to None.
        ravdess_folder (str, optional): path to dataset. Defaults to None.
    """
    # setting seeds for reproducible code.
    random.seed(seed)

    if os.path.exists(save_json_train) and os.path.exists(save_json_valid):
        logger.info("train/valid json both exist, skipping preparation.")
        return

    all_dict = {}
    check_and_prepare_dataset(
        emovdb_folder, "EMOV-DB", prepare_emovdb, all_dict, seed
    )

    check_and_prepare_dataset(esd_folder, "ESD", prepare_esd, all_dict, seed)

    check_and_prepare_dataset(
        iemocap_folder, "IEMOCAP", prepare_iemocap, all_dict, seed
    )
    check_and_prepare_dataset(
        jlcorpus_folder, "JL_CORPUS", prepare_jlcorpus, all_dict, seed
    )
    check_and_prepare_dataset(
        ravdess_folder, "RAVDESS", prepare_ravdess, all_dict, seed
    )

    bad_keys = []
    for key in all_dict.keys():
        try:
            intervals, ctc_label, frame_label = get_labels(
                all_dict[key], win_len, stride
            )
            all_dict[key]["frame_label"] = frame_label
            all_dict[key]["ctc_label"] = ctc_label
        except ValueError:
            logger.info(
                f"Impossible to get labels for id {key} because the window is too large."
            )
            bad_keys.append(key)
            continue
    for key in bad_keys:
        del all_dict[key]

    data_split = split_sets(all_dict, split_ratio)

    train_ids = data_split["train"]
    train_split = {}
    for id in train_ids:
        train_split[id] = all_dict[id]

    valid_ids = data_split["valid"]
    valid_split = {}
    for id in valid_ids:
        valid_split[id] = all_dict[id]

    create_json(train_split, save_json_train)
    create_json(valid_split, save_json_valid)
    if save_json_test is not None:
        test_ids = data_split["test"]
        test_split = {}
        for id in test_ids:
            test_split[id] = all_dict[id]
        create_json(test_split, save_json_test)


def prepare_test(
    ZED_folder, save_json_test, win_len, stride,
):
    """test(ZED) set preparation

    Args:
        ZED_folder (str): path to ZED folder
        save_json_test (str): test json path
        win_len (float):
            window length for generating emotion labels. Defaults to 0.02.
        stride (float):
            stride for generating emotion labels. Defaults to 0.02.
    """
    if os.path.exists(save_json_test):
        logger.info("test json exists, skipping preparation.")
        return

    try:
        zed_json_path = os.path.join(ZED_folder, "ZED.json")
        with open(zed_json_path, "r") as f:
            all_dict = json.load(f)
    except OSError:
        logger.info(f"ZED.json can't be found under {ZED_folder}")
        return

    bad_keys = []
    for key in all_dict.keys():
        try:
            all_dict[key]["wav"] = all_dict[key]["wav"].replace(
                "datafolder", ZED_folder
            )
            intervals, ctc_label, frame_label = get_labels(
                all_dict[key], win_len, stride
            )
            all_dict[key]["frame_label"] = frame_label
            all_dict[key]["ctc_label"] = ctc_label
        except ValueError:
            logger.info(
                f"Impossible to get labels for id {key} because the window is too large."
            )
            bad_keys.append(key)
            continue
    for key in bad_keys:
        del all_dict[key]

    create_json(all_dict, save_json_test)


def split_sets(data_dict, split_ratio, splits=["train", "valid"]):
    """Randomly splits the wav list into training, validation, and test lists.
    Arguments
    ---------
    data_dict : list
        a dictionary of id and its corresponding audio information
    split_ratio: list
        List composed of three integers that sets split ratios for train,
        valid, and test sets, respectively.
        For instance split_ratio=[80, 10, 10] will assign 80% of the sentences
        to training, 10% for validation, and 10% for test.
    splits : list
        List of splits.
    Returns
    ------
    dictionary containing train, valid, and test splits.
    """
    assert len(splits) == len(split_ratio)
    id_list = list(data_dict.keys())
    # Random shuffle of the list
    random.shuffle(id_list)
    tot_split = sum(split_ratio)
    tot_snts = len(id_list)
    data_split = {}
    splits = ["train", "valid"]

    for i, split in enumerate(splits):
        n_snts = int(tot_snts * split_ratio[i] / tot_split)
        data_split[split] = id_list[0:n_snts]
        del id_list[0:n_snts]
    if len(split_ratio) == 3:
        data_split["test"] = id_list
    return data_split


def create_json(data, json_file):
    """
    Creates the json file given a list of wav information.
    Arguments
    ---------
    data : dict
        The dict of wav information (path, label, emotion).
    json_file : str
        The path of the output json file
    """
    # Writing the dictionary to the json file
    with open(json_file, mode="w") as json_f:
        json.dump(data, json_f, indent=2)
    logger.info(f"{json_file} successfully created!")


def check_and_prepare_dataset(
    data_folder, data_name, prepare_function, dictonary, seed
):
    """check if the preparation is done, do it if not

    Args:
        data_folder (str): path to dataset
        data_name (str): name of the dataset
        prepare_function (function): the preparation function
        dictonary (dict): the overall dictionary to be updated
        seed (int): the random seed for reproduction
    """
    if data_folder is not None:
        if not os.path.exists(os.path.join(data_folder, f"{data_name}.json")):
            data = prepare_function(
                data_folder,
                os.path.join(data_folder, f"{data_name}.json"),
                seed,
            )
        else:
            json_path = os.path.join(data_folder, data_name + ".json")
            logger.info(
                f"{json_path} exists, skipping f{data_name} preparation."
            )
            with open(json_path, "r") as f:
                data = json.load(f)
        dictonary.update(data.items())
    else:
        logger.info(f"{data_name} is not used in this exp.")
