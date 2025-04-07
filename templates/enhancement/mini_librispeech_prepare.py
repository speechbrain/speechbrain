"""
Downloads and creates manifest files for Mini LibriSpeech.
Noise is automatically added to samples, managed by the EnvCorrupt class.

Authors:
 * Peter Plantinga, 2020
"""

import json
import os
import shutil

from speechbrain.dataio.dataio import read_audio
from speechbrain.utils.data_utils import download_file, get_all_files
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)
MINILIBRI_TRAIN_URL = "http://www.openslr.org/resources/31/train-clean-5.tar.gz"
MINILIBRI_VALID_URL = "http://www.openslr.org/resources/31/dev-clean-2.tar.gz"
MINILIBRI_TEST_URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"
SAMPLERATE = 16000


def prepare_mini_librispeech(
    data_folder, save_json_train, save_json_valid, save_json_test
):
    """
    Prepares the json files for the Mini Librispeech dataset.

    Downloads the dataset if its not found in the `data_folder`.

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

    Returns
    -------
    None

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
    valid_folder = os.path.join(data_folder, "LibriSpeech", "dev-clean-2")
    test_folder = os.path.join(data_folder, "LibriSpeech", "test-clean")
    if not check_folders(train_folder, valid_folder, test_folder):
        download_mini_librispeech(data_folder)

    # List files and create manifest from list
    logger.info(
        f"Creating {save_json_train}, {save_json_valid}, and {save_json_test}"
    )
    extension = [".flac"]
    wav_list_train = get_all_files(train_folder, match_and=extension)
    wav_list_valid = get_all_files(valid_folder, match_and=extension)
    wav_list_test = get_all_files(test_folder, match_and=extension)
    create_json(wav_list_train, save_json_train)
    create_json(wav_list_valid, save_json_valid)
    create_json(wav_list_test, save_json_test)


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

        # Create entry for this utterance
        json_dict[uttid] = {"wav": relative_path, "length": duration}

    # Writing the dictionary to the json file
    with open(json_file, mode="w", encoding="utf-8") as json_f:
        json.dump(json_dict, json_f, indent=2)

    logger.info(f"{json_file} successfully created!")


def skip(*filenames):
    """
    Detects if the data preparation has been already done.
    If the preparation has been done, we can skip it.

    Arguments
    ---------
    *filenames : tuple
        The file paths passed here should already exist
        in order for the preparation to be considered done.

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


def download_mini_librispeech(destination):
    """Download dataset and unpack it.

    Arguments
    ---------
    destination : str
        Place to put dataset.
    """
    train_archive = os.path.join(destination, "train-clean-5.tar.gz")
    valid_archive = os.path.join(destination, "dev-clean-2.tar.gz")
    test_archive = os.path.join(destination, "test-clean.tar.gz")
    download_file(MINILIBRI_TRAIN_URL, train_archive)
    download_file(MINILIBRI_VALID_URL, valid_archive)
    download_file(MINILIBRI_TEST_URL, test_archive)
    shutil.unpack_archive(train_archive, destination)
    shutil.unpack_archive(valid_archive, destination)
    shutil.unpack_archive(test_archive, destination)
