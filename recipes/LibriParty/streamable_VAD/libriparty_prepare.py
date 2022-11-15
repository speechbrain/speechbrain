""" This script prepares the data-manifest files (in JSON format)
for training and testing a Voice Activity Detection system with the
LibriParty dataset.

The dataset contains sequences of 1-minutes of LibiSpeech sentences
corrupted by noise and reverberation. The dataset can be downloaded
from here:

https://drive.google.com/file/d/1--cAS5ePojMwNY5fewioXAv9YlYAWzIJ/view?usp=sharing

Authors
 * Mohamed Kleit 2021
  * Arjun V 2021
"""

import numpy as np
import pandas as pd
import json
import logging
from collections import OrderedDict


""" Global variables"""
logger = logging.getLogger(__name__)
valid_json_dataset = {}


def load_data_json(path):
    with open(path) as f:
        json_file = json.load(f)
    return json_file


def clean_dataframe(df):
    # Drop unecessary columns
    df.drop(
        [
            "utt_id",
            "file",
            "lvl",
            "channel",
            "rir_channel",
            "words",
            "file",
            "rir",
        ],
        axis=1,
        inplace=True,
    )

    # Sort by start/stop
    df = df.groupby("session_id").apply(lambda x: x.sort_values("start"))
    df = pd.melt(df, id_vars=["session_id", "start"])
    df.drop(["variable"], axis=1, inplace=True)
    df.rename({"value": "stop"}, axis=1, inplace=True)
    return df


def create_dataframe(df, json):
    sessions = list(json.keys())
    session_id = 0
    for session in sessions:
        sub_section = list(json[session].keys())
        for sub in sub_section:
            if sub != "noises" and sub != "background":
                length = len(json[session][sub])
                for i in range(length):
                    temp_dict = json[session][sub][i]
                    temp_dict["session_id"] = session_id
                    df = df.append([temp_dict])
        session_id += 1

    df = clean_dataframe(df)
    return df


def merge_overlapping_intervals(df):
    for i in range(len(df) - 1):
        stop = df["stop"][i]
        start = df["start"][i + 1]
        id_stop = df["session_id"][i]
        id_start = df["session_id"][i + 1]
        if (stop - start) > 0 and id_stop == id_start:
            start_new = df["start"][i]
            df.at[i + 1, "start"] = start_new
            df.at[i, "start"] = -1

    # Delete redundant rows
    df = df[df["start"] != -1].copy()
    df.reset_index(inplace=True)
    df.drop(["index"], axis=1, inplace=True)
    return df


def create_json_structure(df, data_folder):
    session_ids = df["session_id"].unique().tolist()
    json = {}
    for session_id in session_ids:
        path = (
            data_folder
            + str(session_id)
            + "/session_"
            + str(session_id)
            + "_mixture.wav"
        )
        df_session = df[df["session_id"] == session_id]
        speech = list(zip(df_session["start"], df_session["stop"]))
        json["session_" + str(session_id)] = {"file": path, "speech": speech}
    return json


def duplicates(lst, item):
    dup = [i for i, x in enumerate(lst) if x == item]
    return dup


def create_window_splits(values, compare_list, reference_list, window_size):
    start = values[0]
    stop = values[1]
    start_floor = int(np.floor(start / window_size))
    stop_floor = int(np.ceil(stop / window_size))
    reference_sub_list = []
    # to create reference list
    iter_values = [
        i
        for i in range(start_floor * window_size, stop_floor * window_size + 1)
        if i % window_size == 0
    ]
    for i in range(len(iter_values) - 1):
        temp = [iter_values[i], iter_values[i + 1]]
        reference_sub_list.append(temp.copy())
        compare_list.append(temp.copy())

    reference_sub_list[0][0] = start
    reference_sub_list[-1][1] = stop
    reference_list.append(reference_sub_list)
    return reference_list, compare_list


def remove_duplicates_sort(reference_list, compare_list):
    seq_timing = []
    for sub in reference_list:
        for pair in sub:
            seq_timing.append(pair)
    res = []
    # Removing duplicates in the list by comparing
    for i in compare_list:
        res.append(duplicates(compare_list, i))
    # Sort it by using the ordered dictionary
    unq_lst = OrderedDict()
    for e in res:
        unq_lst.setdefault(frozenset(e), []).append(e)
    unique_list = [list(x) for x in unq_lst]
    return unique_list, seq_timing


def add_example(file_path, speech, window, example, sample_rate, json_dataset):
    example = "example_" + str(example)
    json_dataset[example] = {}
    json_dataset[example]["wav"] = {}
    json_dataset[example]["wav"]["file"] = file_path
    json_dataset[example]["wav"]["start"] = window[0] * sample_rate
    json_dataset[example]["wav"]["stop"] = window[1] * sample_rate
    for interval in speech:
        interval[0] -= window[0]
        interval[1] -= window[0]
    json_dataset[example]["speech"] = speech
    return json_dataset


def create_json_dataset(dic, sample_rate, window_size):
    """Creates JSON file for Voice Activity Detection.
    Data are chunked in shorter clips of duration window_size"""
    sessions_iteration = dic.keys()
    example_counter = 1
    json_dataset = {}
    for sessions in sessions_iteration:
        speech_timings = dic[sessions]["speech"]
        reference_list = []
        compare_list = []
        for values in speech_timings:
            reference_list, compare_list = create_window_splits(
                values, compare_list, reference_list, window_size
            )
            unique_list, seq_timing = remove_duplicates_sort(
                reference_list, compare_list
            )
        speech_sequence_cleaned = []
        overlap = []
        for i, values in enumerate(unique_list):
            if len(values) == 1:
                speech_sequence_cleaned.append(seq_timing[values[0]])
                json_dataset = json_dataset = add_example(
                    dic[sessions]["file"],
                    [seq_timing[values[0]]],
                    compare_list[values[0]],
                    example_counter,
                    sample_rate,
                    json_dataset,
                )
                example_counter += 1
            else:
                for iter in values:
                    overlap.append(seq_timing[iter])
                json_dataset = add_example(
                    dic[sessions]["file"],
                    overlap,
                    compare_list[values[0]],
                    example_counter,
                    sample_rate,
                    json_dataset,
                )
                speech_sequence_cleaned.append(overlap)
                overlap = []
                example_counter += 1
            dic[sessions]["speech_segments"] = speech_sequence_cleaned
    return json_dataset


def save_dataset(json_save_path, json_dataset):
    """Saves a JSON file."""
    with open(json_save_path, "w+") as fp:
        json.dump(json_dataset, fp, indent=4)


def prepare_libriparty(
    data_folder,
    save_json_folder,
    sample_rate=16000,
    window_size=5,
    skip_prep=False,
):
    """
    Prepares the json files for the LibriParty dataset.

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original LibriSpeech dataset is stored.
    data_folder : str
        The path where to store the training json file.
    save_json_valid : str
        The path where to store the valid json file.
    save_json_test : str
        The path where to store the test json file.
    skip_prep: bool
        Default: False
        If True, the data preparation is skipped.

    Example
    -------
    >>> from recipes.LibriParty.libriparty_prepare import prepare_libriparty
    >>> data_folder = 'datasets/LibriParty'
    >>> prepare_libriparty(data_folder, 'train.json', 'valid.json', 'test.json')
    """

    # Skip if needed
    if skip_prep:
        return

    # Load datasets as json
    train_json = load_data_json(data_folder + "/metadata/train.json")
    valid_json = load_data_json(data_folder + "/metadata/dev.json")
    eval_json = load_data_json(data_folder + "/metadata/eval.json")

    # Create dummy dataframe
    df_columns = pd.DataFrame(
        columns=[
            "start",
            "stop",
            "words",
            "rir",
            "utt_id",
            "file",
            "lvl",
            "channel",
            "rir_channel",
            "session_id",
        ]
    )

    # Create dataframes
    df_train = create_dataframe(df_columns, train_json)
    df_valid = create_dataframe(df_columns, valid_json)
    df_test = create_dataframe(df_columns, eval_json)

    # Merge overlapping intervals
    df_train = merge_overlapping_intervals(df_train)
    df_valid = merge_overlapping_intervals(df_valid)
    df_test = merge_overlapping_intervals(df_test)

    # Create json structure
    train_dict = create_json_structure(
        df_train, data_folder + "/train/session_"
    )
    valid_dict = create_json_structure(df_valid, data_folder + "/dev/session_")
    test_dict = create_json_structure(df_test, data_folder + "/eval/session_")

    # Create datasets as json
    train_dataset = create_json_dataset(train_dict, sample_rate, window_size)
    valid_dataset = create_json_dataset(valid_dict, sample_rate, window_size)
    test_dataset = create_json_dataset(test_dict, sample_rate, window_size)

    # Save datasets
    save_dataset(save_json_folder + "/train.json", train_dataset)
    save_dataset(save_json_folder + "/valid.json", valid_dataset)
    save_dataset(save_json_folder + "/test.json", test_dataset)
