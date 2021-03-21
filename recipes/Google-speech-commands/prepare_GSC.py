"""
Data preparation for Google Speech Commands v0.02.

Download: http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz

Author
------
David Raby-Pepin 2021

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

try:
    import pandas as pd
except ImportError:
    err_msg = (
        "The optional dependency pandas must be installed to run this recipe.\n"
    )
    err_msg += "Install using `pip install pandas`.\n"
    raise ImportError(err_msg)

logger = logging.getLogger(__name__)

GSC_URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"

# List of all the words (i.e. classes) within the GSC v2 dataset
all_words = [
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "bed",
    "bird",
    "cat",
    "dog",
    "happy",
    "house",
    "marvin",
    "sheila",
    "tree",
    "wow",
    "backward",
    "forward",
    "follow",
    "learn",
    "visual",
]


def prepare_GSC(
    data_folder,
    save_folder,
    validation_percentage=10,
    testing_percentage=10,
    percentage_unknown=10,
    percentage_silence=10,
    words_wanted=[
        "yes",
        "no",
        "up",
        "down",
        "left",
        "right",
        "on",
        "off",
        "stop",
        "go",
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
    validation_percentage: int
        How much of the data set to use for validation.
    testing_percentage: int
        How much of the data set to use for testing.
    percentage unknown: int.
        How much data outside of the known (i.e wanted) words to preserve; relative to the total number of known words.
    percentage silence: int
        How many silence samples to generate; relative to the total number of known words.
    skip_prep: bool
        If True, skip data preparation.

    Example
    -------
    >>> data_folder = '/path/to/GSC'
    >>> prepare_GSC(data_folder)
    """

    if skip_prep:
        return

    # If the data folders do not exist, we need to extract the data
    if not os.path.isdir(os.path.join(data_folder, "train-synth")):
        # Check for zip file and download if it doesn't exist
        tar_location = os.path.join(data_folder, "speech_commands_v0.02.tar.gz")
        if not os.path.exists(tar_location):
            download_file(GSC_URL, tar_location, unpack=True)
        else:
            logger.info("Extracting speech_commands_v0.02.tar.gz...")
            shutil.unpack_archive(tar_location, data_folder)

    # Define the words that we do not want to identify
    unknown_words = list(np.setdiff1d(all_words, words_wanted))

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
    for i, command in enumerate(commands):
        # logger.info("Preparing {}/{} commands...".format(i, len(commands)))

        # Indicate once all wanted words are parsed
        if i >= len(words_wanted) and not words_wanted_parsed:
            num_known_samples_total = np.sum(
                list(num_known_samples_per_split.values())
            )
            num_unknown_samples_total = 105829 - num_known_samples_total
            percentage_applied_to_unknown_samples = (
                percentage_unknown * num_known_samples_total
            ) / num_unknown_samples_total
            words_wanted_parsed = True

        # Read all files under a specific class (i.e. command)
        files = []
        for (dirpath, dirnames, filenames) in walk(
            os.path.join(data_folder, command)
        ):
            files.extend(filenames)
            break

        # Fill in all fields with metadata for each audio sample file under a specific class
        for filename in files:
            # Once all wanted words are parsed, only retain the required percentage of unknown words
            if (
                words_wanted_parsed
                and torch.rand(1)[0].tolist()
                > percentage_applied_to_unknown_samples / 100
            ):
                continue

            # select the required split (i.e. set) for the sample
            split = which_set(
                filename, validation_percentage, testing_percentage
            )

            splits[split]["ID"].append(
                command + "/" + re.sub(r".wav", "", filename)
            )

            # We know that all recordings are 1 second long (i.e.16000 frames). No need to compute the duration.
            splits[split]["duration"].append(1.0)
            splits[split]["start"].append(0)
            splits[split]["stop"].append(16000)

            splits[split]["wav"].append(
                os.path.join(data_folder, command, filename)
            )

            splits[split]["spk_id"].append(re.sub(r"_.*", "", filename))

            if command in words_wanted:
                splits[split]["command"].append(command)

                num_known_samples_per_split[split] += 1
            else:
                splits[split]["command"].append("unknown")

            splits[split]["transcript"].append(command)

    if percentage_silence > 0:
        generate_silence_data(
            num_known_samples_per_split,
            splits,
            data_folder,
            percentage_silence=percentage_silence,
        )

    for split in splits:
        new_filename = os.path.join(save_folder, split) + ".csv"
        new_df = pd.DataFrame(splits[split])
        new_df.to_csv(new_filename, index=False)


MAX_NUM_WAVS_PER_CLASS = 2 ** 27 - 1  # ~134M


def which_set(filename, validation_percentage, testing_percentage):
    """Determines which data partition the file should belong to.

  We want to keep files in the same training, validation, or testing sets even
  if new ones are added over time. This makes it less likely that testing
  samples will accidentally be reused in training when long runs are restarted
  for example. To keep this stability, a hash of the filename is taken and used
  to determine which set it should belong to. This determination only depends on
  the name and the set proportions, so it won't change as other files are added.

  It's also useful to associate particular files as related (for example words
  spoken by the same person), so anything after '_nohash_' in a filename is
  ignored for set determination. This ensures that 'bobby_nohash_0.wav' and
  'bobby_nohash_1.wav' are always in the same set, for example.

    Arguments
    ---------
    filename: path
        File path of the data sample.
    validation_percentage: int
        How much of the data set to use for validation.
    testing_percentage: int
        How much of the data set to use for testing.

    Returns
    ---------
    result: str
        one of 'training', 'validation', or 'testing'.
  """
    base_name = os.path.basename(filename)
    # We want to ignore anything after '_nohash_' in the file name when
    # deciding which set to put a wav in, so the data set creator has a way of
    # grouping wavs that are close variations of each other.
    hash_name = re.sub(r"_nohash_.*$", "", base_name).encode("utf-8")
    # This looks a bit magical, but we need to decide whether this file should
    # go into the training, testing, or validation sets, and we want to keep
    # existing files in the same set even if more files are subsequently
    # added.
    # To do that, we need a stable way of deciding based on just the file name
    # itself, so we do a hash of that and then use that to generate a
    # probability value that we use to assign it.
    hash_name_hashed = hashlib.sha1(hash_name).hexdigest()
    percentage_hash = (
        int(hash_name_hashed, 16) % (MAX_NUM_WAVS_PER_CLASS + 1)
    ) * (100.0 / MAX_NUM_WAVS_PER_CLASS)
    if percentage_hash < validation_percentage:
        result = "valid"
    elif percentage_hash < (testing_percentage + validation_percentage):
        result = "test"
    else:
        result = "train"
    return result


def generate_silence_data(
    num_known_samples_per_split, splits, data_folder, percentage_silence=26
):
    """Generates silence samples.

    Arguments
    ---------
    num_known_samples_per_split: int
        Total number of samples of known words for each split (i.e. set).
    splits: str
        Training, validation and test sets.
    data_folder: str
        path to dataset.
    percentage_silence: int
        How many silence samples to generate; relative to the total number of known words.
  """
    for split in splits:
        num_silence_samples = int(
            (percentage_silence / 100.0) * num_known_samples_per_split[split]
        )

        # Fetch all background noise wav files used to generate silence samples
        search_path = os.path.join(data_folder, "_background_noise_", "*.wav")
        silence_paths = []
        for wav_path in glob.glob(search_path):
            silence_paths.append(wav_path)

        # Generate random silence samples
        # Assumes that the pytorch seed has been defined in the HyperPyYaml file
        num_silence_samples_per_path = int(
            num_silence_samples / len(silence_paths)
        )
        for silence_path in silence_paths:
            signal = read_audio(silence_path)
            random_starts = (
                (
                    torch.rand(num_silence_samples_per_path)
                    * (signal.shape[0] - 16001)
                )
                .type(torch.int)
                .tolist()
            )

            for i, random_start in enumerate(random_starts):
                splits[split]["ID"].append(
                    re.sub(
                        r".wav",
                        "/" + str(random_start) + "_" + str(i),
                        re.sub(r".+?(?=_background_noise_)", "", silence_path),
                    )
                )

                splits[split]["duration"].append(1.0)
                splits[split]["start"].append(random_start)
                splits[split]["stop"].append(random_start + 16000)
                splits[split]["wav"].append(silence_path)
                splits[split]["spk_id"].append(None)
                splits[split]["command"].append("silence")
                splits[split]["transcript"].append(None)
