"""
Downloads and creates data manifest files for IEMOCAP
(https://sail.usc.edu/iemocap/).

Authors:
 * Mirco Ravanelli, 2021
 * Modified by Pierre-Yves Yanni, 2021
"""

import os
import json
import random
import logging
from speechbrain.utils.data_utils import get_all_files
from speechbrain.dataio.dataio import read_audio

logger = logging.getLogger(__name__)
SAMPLERATE = 16000


def prepare_data(
    data_folder,
    save_json_train,
    save_json_valid,
    save_json_test,
    split_ratio=[80, 10, 10],
    different_speakers=False,
    seed=12,
):
    """
    Prepares the json files for the IEMOCAP dataset.

    We here use only the audio part of the dataset. The assumpion is
    that the data folder is structured as:

    <session_id>/<emotion>/<file:name>.wav

    e.g.
    session1/ang/psno1_ang_s084_orgn.wav

    Please, process the original IEMOCAP folder to match the expected
    folder structure.


    Arguments
    ---------
    data_folder : str
        Path to the folder where the transformed IEMOCAP dataset is stored.
    save_json_train : str
        Path where the train data specification file will be saved.
    save_json_valid : str
        Path where the validation data specification file will be saved.
    save_json_test : str
        Path where the test data specification file will be saved.
    split_ratio: list
        List composed of three integers that sets split ratios for train,
        valid, and test sets, respecively.
        For instance split_ratio=[80, 10, 10] will assign 80% of the sentences
        to training, 10% for validation, and 10% for test.
    seed : int
        Seed for reproducibility

    Example
    -------
    >>> data_folder = '/path/to/iemocap'
    >>> prepare_data(data_path, data_folder, 'train.json', 'valid.json',
        'test.json')
    """

    # setting seeds for reproducible code.
    random.seed(seed)

    # Check if this phase is already done (if so, skip it)
    if skip(save_json_train, save_json_valid, save_json_test):
        logger.info("Preparation completed in previous run, skipping.")
        return

    if not check_folders(data_folder + "session1/ang/psno1_ang_s084_orgn.wav"):
        logger.info(
            "The data folder is not in the expected format. Expected <session_id>/<emo_id>/<file_name>.wav (e.g., session1/ang/psno1_ang_s084_orgn.wav)"
        )

    # List files and create manifest from list
    logger.info(
        f"Creating {save_json_train}, {save_json_valid}, and {save_json_test}"
    )
    extension = [".wav"]

    # Randomly split the signal list into train, valid, and test sets.
    wav_list = get_all_files(data_folder, match_and=extension)
    if different_speakers:
        data_split = split_different_speakers(wav_list)
    else:
        data_split = split_sets(wav_list, split_ratio)

    # Creating json files
    create_json(data_split["train"], save_json_train)
    create_json(data_split["valid"], save_json_valid)
    create_json(data_split["test"], save_json_test)


def create_json(wav_list, json_file):
    """
    Creates the json file given a list of wav files.

    Arguments
    ---------
    wav_list : list of str
        The list of wav files.
    json_file : str
        The path of the output json file
    """
    # Processing all the wav files in the list

    json_dict = {}
    for wav_file in wav_list:

        # Reading the signal (to retrieve duration in seconds)
        signal = read_audio(wav_file)
        duration = signal.shape[0] / SAMPLERATE

        # Manipulate path to get relative path and uttid
        path_parts = wav_file.split(os.path.sep)
        uttid, _ = os.path.splitext(path_parts[-1])
        relative_path = os.path.join("{data_root}", *path_parts[-3:])

        # Getting emotion
        emo = path_parts[-2]

        # Create entry for this utterance
        json_dict[uttid] = {
            "wav": relative_path,
            "length": duration,
            "emo": emo,
        }

    # Writing the dictionary to the json file
    with open(json_file, mode="w") as json_f:
        json.dump(json_dict, json_f, indent=2)

    logger.info(f"{json_file} successfully created!")


def skip(*filenames):
    """
    Detects if the data preparation has been already done.
    If the preparation has been done, we can skip it.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """
    for filename in filenames:
        if not os.path.isfile(filename):
            return False
    return True


def check_folders(*folders):
    """Returns False if any passed folder does not exist."""
    for folder in folders:
        if not os.path.exists(folder):
            return False
    return True


def split_different_speakers(wav_list):
    """"Constructs train, validation and test sets that do not share common
    speakers. There are two different speakers in each session. Train set is
    constituted of 3 sessions, validation set another session and test set the
    remaining session.

    Arguments
    ---------
    wav_list: list
        list of all signals in the dataset

    Returns
    ------
    dictionary containing train, valid, and test splits.
    """
    data_split = {k: [] for k in ["train", "valid", "test"]}
    sessions = list(range(1, 6))
    random.shuffle(sessions)
    random.shuffle(wav_list)

    for path_wav in wav_list:
        session = int(os.path.split(path_wav)[-1][4])
        if session in sessions[:3]:
            data_split["train"].append(path_wav)
        elif session == sessions[3]:
            data_split["valid"].append(path_wav)
        else:
            data_split["test"].append(path_wav)
    return data_split


def split_sets(wav_list, split_ratio):
    """Randomly splits the wav list into training, validation, and test lists.
    Note that a better approach is to make sure that all the classes have the
    same proportion of samples (e.g, spk01 should have 80% of samples in
    training, 10% validation, 10% test, the same for speaker2 etc.). This
    is the approach followed in some recipes such as the Voxceleb one. For
    simplicity, we here simply split the full list without necessarly
    respecting the split ratio within each class.

    Arguments
    ---------
    wav_list : list
        list of all the signals in the dataset
    split_ratio: list
        List composed of three integers that sets split ratios for train,
        valid, and test sets, respectively.
        For instance split_ratio=[80, 10, 10] will assign 80% of the sentences
        to training, 10% for validation, and 10% for test.

    Returns
    ------
    dictionary containing train, valid, and test splits.
    """
    # Random shuffle of the list
    random.shuffle(wav_list)
    tot_split = sum(split_ratio)
    tot_snts = len(wav_list)
    data_split = {}
    splits = ["train", "valid"]

    for i, split in enumerate(splits):
        n_snts = int(tot_snts * split_ratio[i] / tot_split)
        data_split[split] = wav_list[0:n_snts]
        del wav_list[0:n_snts]
    data_split["test"] = wav_list

    return data_split
