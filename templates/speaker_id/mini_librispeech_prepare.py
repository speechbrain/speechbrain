"""
Downloads and creates data manifest files for Mini LibriSpeech (spk-id).
For speaker-id, different sentences of the same speaker must appear in train,
validation, and test sets. In this case, these sets are thus derived from
splitting the original training set intothree chunks.

Authors:
 * Mirco Ravanelli, 2021
"""

import os
import json
import shutil
import random
import logging
from speechbrain.utils.data_utils import get_all_files, download_file
from speechbrain.dataio.dataio import read_audio

logger = logging.getLogger(__name__)
MINILIBRI_TRAIN_URL = "http://www.openslr.org/resources/31/train-clean-5.tar.gz"
SAMPLERATE = 16000


def prepare_mini_librispeech(
    data_folder,
    save_json_train,
    save_json_valid,
    save_json_test,
    split_ratio=[80, 10, 10],
):
    """
    Prepares the json files for the Mini Librispeech dataset.

    Downloads the dataset if it is not found in the `data_folder`.

    Arguments
    ---------
    data_folder : str
        Path to the folder where the Mini Librispeech dataset is stored.
    save_json_train : str
        Path where the train data specification file will be saved.
    save_json_valid : str
        Path where the validation data specification file will be saved.
    save_json_test : str
        Path where the test data specification file will be saved.
    split_ratio: list
        List composed of three integers that sets split ratios for train, valid,
        and test sets, respectively. For instance split_ratio=[80, 10, 10] will
        assign 80% of the sentences to training, 10% for validation, and 10%
        for test.

    Example
    -------
    >>> data_folder = '/path/to/mini_librispeech'
    >>> prepare_mini_librispeech(data_folder, 'train.json', 'valid.json', 'test.json')
    """

    # Check if this phase is already done (if so, skip it)
    if skip(save_json_train, save_json_valid, save_json_test):
        logger.info("Preparation completed in previous run, skipping.")
        return

    # If the dataset doesn't exist yet, download it
    train_folder = os.path.join(data_folder, "LibriSpeech", "train-clean-5")
    if not check_folders(train_folder):
        download_mini_librispeech(data_folder)

    # List files and create manifest from list
    logger.info(
        f"Creating {save_json_train}, {save_json_valid}, and {save_json_test}"
    )
    extension = [".flac"]
    wav_list = get_all_files(train_folder, match_and=extension)

    # Random split the signal list into train, valid, and test sets.
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
        relative_path = os.path.join("{data_root}", *path_parts[-5:])

        # Getting speaker-id from utterance-id
        spk_id = uttid.split("-")[0]

        # Create entry for this utterance
        json_dict[uttid] = {
            "wav": relative_path,
            "length": duration,
            "spk_id": spk_id,
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


def split_sets(wav_list, split_ratio):
    """Randomly splits the wav list into training, validation, and test lists.
    Note that a better approach is to make sure that all the classes have the
    same proportion of samples (e.g, spk01 should have 80% of samples in
    training, 10% validation, 10% test, the same for speaker2 etc.). This
    is the approach followed in some recipes such as the Voxceleb one. For
    simplicity, we here simply split the full list without necessarily respecting
    the split ratio within each class.

    Arguments
    ---------
    wav_lst : list
        list of all the signals in the dataset
    split_ratio: list
        List composed of three integers that sets split ratios for train, valid,
        and test sets, respectively. For instance split_ratio=[80, 10, 10] will
        assign 80% of the sentences to training, 10% for validation, and 10%
        for test.

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


def download_mini_librispeech(destination):
    """Download dataset and unpack it.

    Arguments
    ---------
    destination : str
        Place to put dataset.
    """
    train_archive = os.path.join(destination, "train-clean-5.tar.gz")
    download_file(MINILIBRI_TRAIN_URL, train_archive)
    shutil.unpack_archive(train_archive, destination)
