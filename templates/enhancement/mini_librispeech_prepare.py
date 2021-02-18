"""
Downloads and creates manifest files for Mini LibriSpeech.
Noise is automatically added to samples.

Download and resample, use ``download_vctk`` below.
https://datashare.is.ed.ac.uk/handle/10283/2791

Authors:
 * Peter Plantinga, 2020
"""

import os
import json
import shutil
import logging
from speechbrain.utils.data_utils import get_all_files, download_file
from speechbrain.dataio.dataio import read_audio

logger = logging.getLogger(__name__)
MINILIBRI_TRAIN_URL = "http://www.openslr.org/resources/31/train-clean-5.tar.gz"
MINILIBRI_VALID_URL = "http://www.openslr.org/resources/31/dev-clean-2.tar.gz"
TRAIN_JSON = "train.json"
VALID_JSON = "valid.json"
SAMPLERATE = 16000


def prepare_mini_librispeech(data_folder, save_folder):
    """
    Prepares the json files for the Mini Librispeech dataset.

    Downloads the dataset if its not found in the `data_folder`.

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original Voicebank dataset is stored.
    save_folder : str
        The directory where to store the json files.

    Example
    -------
    >>> data_folder = '/path/to/mini_librispeech'
    >>> save_folder = '.'
    >>> prepare_voicebank(data_folder, save_folder)
    """

    # Setting ouput files
    save_json_train = os.path.join(save_folder, TRAIN_JSON)
    save_json_valid = os.path.join(save_folder, VALID_JSON)

    # Check if this phase is already done (if so, skip it)
    if skip(save_json_train, save_json_valid):
        logger.info("Preparation completed in previous run, skipping.")
        return

    # Setting the save folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # If the dataset doesn't exist yet, download it
    train_folder = os.path.join(data_folder, "LibriSpeech", "train-clean-5")
    valid_folder = os.path.join(data_folder, "LibriSpeech", "dev-clean-2")
    if not check_folders(train_folder, valid_folder):
        download_mini_librispeech(data_folder)

    # List files and create manifest from list
    logger.info(f"Creating {save_json_train} and {save_json_valid}")
    extension = [".flac"]
    wav_list_train = get_all_files(train_folder, match_and=extension)
    wav_list_valid = get_all_files(valid_folder, match_and=extension)
    create_json(wav_list_train, save_json_train)
    create_json(wav_list_valid, save_json_valid)


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
    with open(json_file, mode="w") as json_f:
        json.dump(json_dict, json_f, indent=2)

    logger.info(f"{json_file} successfully created!")


def skip(*filenames):
    """
    Detects if the Voicebank data_preparation has been already done.
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


def download_mini_librispeech(destination):
    """Download dataset and unpack it.

    Arguments
    ---------
    destination : str
        Place to put dataset.
    """
    train_archive = os.path.join(destination, "train-clean-5.tar.gz")
    valid_archive = os.path.join(destination, "dev-clean-2.tar.gz")
    download_file(MINILIBRI_TRAIN_URL, train_archive)
    download_file(MINILIBRI_VALID_URL, valid_archive)
    shutil.unpack_archive(train_archive, destination)
    shutil.unpack_archive(valid_archive, destination)
