"""
LibriTTS data preparation

Authors
 * Pradnya Kandarkar 2022
"""

import json
import os
import random

import torch
import torchaudio
from tqdm import tqdm

from speechbrain.inference.text import GraphemeToPhoneme
from speechbrain.utils.data_utils import get_all_files
from speechbrain.utils.logger import get_logger
from speechbrain.utils.text_to_sequence import _g2p_keep_punctuations

logger = get_logger(__name__)
LIBRITTS_URL_PREFIX = "https://www.openslr.org/resources/60/"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_libritts(
    data_folder,
    save_json_train,
    save_json_valid,
    save_json_test,
    sample_rate,
    split_ratio=[80, 10, 10],
    libritts_subsets=None,
    train_split=None,
    valid_split=None,
    test_split=None,
    seed=1234,
    model_name=None,
    skip_prep=False,
):
    """
    Prepares the json files for the LibriTTS dataset.
    Downloads the dataset if it is not found in the `data_folder` as expected.

    Arguments
    ---------
    data_folder : str
        Path to the folder where the LibriTTS dataset is stored.
    save_json_train : str
        Path where the train data specification file will be saved.
    save_json_valid : str
        Path where the validation data specification file will be saved.
    save_json_test : str
        Path where the test data specification file will be saved.
    sample_rate : int
        The sample rate to be used for the dataset
    split_ratio : list
        List composed of three integers that sets split ratios for train, valid,
        and test sets, respectively. For instance split_ratio=[80, 10, 10] will
        assign 80% of the sentences to training, 10% for validation, and 10%
        for test.
    libritts_subsets: list
        List of librispeech subsets to use (e.g., dev-clean, train-clean-100, ...) for the experiment.
        This parameter will be ignored if explicit data splits are provided.
        Explicit data splits parameters: "train_split", "valid_split", "test_split"
    train_split : list
        List of librispeech subsets to use (e.g.,train-clean-100, train-clean-360) for the experiment training stage.
    valid_split : list
        List of librispeech subsets to use (e.g., dev-clean) for the experiment validation stage.
    test_split : list
        List of librispeech subsets to use (e.g., test-clean) for the experiment testing stage.
    seed : int
        Seed value
    model_name : str
        Model name (used to prepare additional model specific data)
    skip_prep: Bool
        If True, skip preparation.

    Returns
    -------
    None
    """

    if skip_prep:
        return

    # Setting the seed value
    random.seed(seed)

    # Checks if this phase is already done (if so, skips it)
    if skip(save_json_train, save_json_valid, save_json_test):
        logger.info("Preparation completed in previous run, skipping.")
        return

    logger.info(
        f"Creating {save_json_train}, {save_json_valid}, and {save_json_test}"
    )

    # If specific splits are provided, creates data manifest files accordingly
    if train_split:
        wav_list = prepare_split(data_folder, train_split)
        create_json(wav_list, save_json_train, sample_rate, model_name)
    if valid_split:
        wav_list = prepare_split(data_folder, valid_split)
        # TODO add better way to speedup evaluation
        wav_list = random.sample(wav_list, 500)
        create_json(wav_list, save_json_valid, sample_rate, model_name)
    if test_split:
        wav_list = prepare_split(data_folder, test_split)
        create_json(wav_list, save_json_test, sample_rate, model_name)

    if skip(save_json_train, save_json_valid, save_json_test):
        logger.info("Preparation completed.")
        return

    # If specific splits are not provided, and a list of subsets if provided, creates train, valid, test splits
    # Creates data manifest files according to the data splits
    if libritts_subsets:
        wav_list = prepare_split(data_folder, libritts_subsets)
        # Random split the signal list into train, valid, and test sets.
        data_split = split_sets(wav_list, split_ratio)
        # Creating json files
        create_json(
            data_split["train"], save_json_train, sample_rate, model_name
        )
        create_json(
            data_split["valid"], save_json_valid, sample_rate, model_name
        )
        create_json(data_split["test"], save_json_test, sample_rate, model_name)


def prepare_split(data_folder, split_list):
    """
    Processes the provided list of LibriTTS subsets and creates a list of all the .wav files present in the subsets.
    Downloads the LibriTTS subsets as required.

    Arguments
    ---------
    data_folder : str
        Path to the folder where the LibriTTS dataset is stored
    split_list : list
        List of librispeech subsets to process (e.g., dev-clean, train-clean-100, ...)

    Returns
    -------
    wav_list : list
        List of all .wav files to be processed
    """
    extension = [".wav"]  # The expected extension for audio files
    wav_list = list()  # Stores all audio file paths for the dataset

    # For every subset of the dataset, if it doesn't exist, downloads it
    for subset_name in split_list:
        subset_folder = os.path.join(data_folder, subset_name)
        subset_archive = os.path.join(subset_folder, subset_name + ".tar.gz")

        if not check_folders(subset_folder):
            logger.info(
                f"No data found for {subset_name}. Checking for an archive file."
            )
            if not os.path.isfile(subset_archive):
                logger.info(
                    f"No archive file found for {subset_name}. Downloading and unpacking."
                )
                quit()
        # Collects all files matching the provided extension
        wav_list.extend(get_all_files(subset_folder, match_and=extension))

    return wav_list


def create_json(wav_list, json_file, sample_rate, model_name=None):
    """
    Creates the json file given a list of wav files.
    Arguments
    ---------
    wav_list : list of str
        The list of wav files.
    json_file : str
        The path of the output json file
    sample_rate : int
        The sample rate to be used for the dataset
    model_name : str
        Model name (used to prepare additional model specific data)
    """

    # Downloads and initializes the G2P model to compute the phonemes if data is being prepared for Tacotron2 experiments
    if model_name == "Tacotron2":
        logger.info(
            "Computing phonemes for labels using SpeechBrain G2P. This may take a while."
        )
        g2p = GraphemeToPhoneme.from_hparams(
            "speechbrain/soundchoice-g2p", run_opts={"device": DEVICE}
        )

    json_dict = {}

    # Processes all the wav files in the list
    for wav_file in tqdm(wav_list):
        # Reads the signal
        signal, sig_sr = torchaudio.load(wav_file)
        duration = signal.shape[1] / sig_sr

        # TODO add better way to filter short utterances
        if duration < 1.0:
            continue

        # Manipulates path to get relative path and uttid
        path_parts = wav_file.split(os.path.sep)
        uttid, _ = os.path.splitext(path_parts[-1])
        # relative_path = os.path.join("{data_root}", *path_parts[-4:])

        # Gets the path for the text files and extracts the input text
        normalized_text_path = os.path.join(
            "/", *path_parts[:-1], uttid + ".normalized.txt"
        )
        try:
            with open(normalized_text_path, encoding="utf-8") as f:
                normalized_text = f.read()
                if normalized_text.__contains__("{"):
                    normalized_text = normalized_text.replace("{", "")
                if normalized_text.__contains__("}"):
                    normalized_text = normalized_text.replace("}", "")
        except FileNotFoundError:
            print(f"Warning: The file {normalized_text_path} does not exist.")
            continue

        # Resamples the audio file if required
        if sig_sr != sample_rate:
            resampled_signal = torchaudio.functional.resample(
                signal, sig_sr, sample_rate
            )
            os.unlink(wav_file)
            torchaudio.save(wav_file, resampled_signal, sample_rate=sample_rate)

        # Gets the speaker-id from the utterance-id
        spk_id = uttid.split("_")[0]

        # Creates an entry for the utterance
        json_dict[uttid] = {
            "uttid": uttid,
            "wav": wav_file,
            "duration": duration,
            "spk_id": spk_id,
            "label": normalized_text,
            "segment": True if "train" in json_file else False,
        }

        # Characters are used for Tacotron2, phonemes may be needed for other models
        if model_name not in ["Tacotron2", "HiFi-GAN"]:
            # Computes phoneme labels using SpeechBrain G2P and keeps the punctuations
            phonemes = _g2p_keep_punctuations(g2p, normalized_text)
            json_dict[uttid].update({"label_phoneme": phonemes})

    # Writes the dictionary to the json file
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
        Set of filenames to check for existence.

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


def split_sets(wav_list, split_ratio):
    """Randomly splits the wav list into training, validation, and test lists.

    Arguments
    ---------
    wav_list : list
        list of all the signals in the dataset
    split_ratio: list
        List composed of three integers that sets split ratios for train, valid,
        and test sets, respectively. For instance split_ratio=[80, 10, 10] will
        assign 80% of the sentences to training, 10% for validation, and 10%
        for test.

    Returns
    -------
    dictionary containing train, valid, and test splits.
    """
    # Random shuffles the list
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


def check_folders(*folders):
    """Returns False if any passed folder does not exist."""
    for folder in folders:
        if not os.path.exists(folder):
            return False
    return True
