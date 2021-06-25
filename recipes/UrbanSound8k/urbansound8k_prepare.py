"""
Creates data manifest files from UrbanSound8k, suitable for use in SpeechBrain.

https://urbansounddataset.weebly.com/urbansound8k.html

From the authors of UrbanSound8k:

1. Don't reshuffle the data! Use the predefined 10 folds and perform 10-fold (not 5-fold) cross validation
The experiments conducted by vast majority of publications using UrbanSound8K (by ourselves and others)
evaluate classification models via 10-fold cross validation using the predefined splits*.
We strongly recommend following this procedure.

Why?
If you reshuffle the data (e.g. combine the data from all folds and generate a random train/test split)
you will be incorrectly placing related samples in both the train and test sets, leading to inflated
scores that don't represent your model's performance on unseen data. Put simply, your results will be wrong.
Your results will NOT be comparable to previous results in the literature, meaning any claims to an
improvement on previous research will be invalid. Even if you don't reshuffle the data, evaluating using
different splits (e.g. 5-fold cross validation) will mean your results are not comparable to previous research.

2. Don't evaluate just on one split! Use 10-fold (not 5-fold) cross validation and average the scores
We have seen reports that only provide results for a single train/test split, e.g. train on folds 1-9,
test on fold 10 and report a single accuracy score. We strongly advise against this. Instead, perform
10-fold cross validation using the provided folds and report the average score.

Why?
Not all the splits are as "easy". That is, models tend to obtain much higher scores when trained on folds
1-9 and tested on fold 10, compared to (e.g.) training on folds 2-10 and testing on fold 1. For this reason,
it is important to evaluate your model on each of the 10 splits and report the average accuracy.
Again, your results will NOT be comparable to previous results in the literature.

​* 10-fold cross validation using the predefined folds: train on data from 9 of the 10 predefined folds and
test on data from the remaining fold. Repeat this process 10 times (each time using a different set of
9 out of the 10 folds for training and the remaining fold for testing). Finally report the average classification
accuracy over all 10 experiments (as an average score + standard deviation, or, even better, as a boxplot).

Authors:
 * David Whipps, 2021
"""

import os
import json
import logging
import ntpath
import torchaudio
from speechbrain.dataio.dataio import read_audio
from speechbrain.dataio.dataio import load_data_csv

logger = logging.getLogger(__name__)

URBAN_SOUND_8K_DOWNLOAD_FORM_URL = (
    "https://urbansounddataset.weebly.com/download-urbansound8k.html"
)
MODIFIED_METADATA_FILE_NAME = "UrbanSound8k_speechbrain.csv"

ACCEPTABLE_FOLD_NUMS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def prepare_urban_sound_8k(
    data_folder,
    audio_data_folder,
    save_json_train,
    save_json_valid,
    save_json_test,
    train_fold_nums=[1, 2, 3, 4, 5, 6, 7, 8],
    valid_fold_nums=[9],
    test_fold_nums=[10],
    skip_manifest_creation=False,
):
    """
    Prepares the json files for the UrbanSound8k dataset.
    Prompts to download the dataset if it is not found in the `data_folder`.
    Arguments
    ---------
    data_folder : str
        Path to the folder where the UrbanSound8k dataset metadata is stored.
    audio_data_folder: str
        Path to the folder where the UrbanSound8k dataset audio files are stored.
    save_json_train : str
        Path where the train data specification file will be saved.
    save_json_valid : str
        Path where the validation data specification file will be saved.
    save_json_test : str
        Path where the test data specification file will be saved.
    train_folds: list or int (integers [1,10])
        A list of integers defining which pre-defined "folds" to use for training. Must be
        exclusive of valid_folds and test_folds.
    valid_folds: list or int (integers [1,10])
        A list of integers defining which pre-defined "folds" to use for validation. Must be
        exclusive of train_folds and test_folds.
    test_folds: list or int (integers [1,10])
        A list of integers defining which pre-defined "folds" to use for test. Must be
        exclusive of train_folds and valid_folds.
    Example
    -------
    >>> data_folder = '/path/to/UrbanSound8k'
    >>> prepare_urban_sound_8k(data_folder, 'train.json', 'valid.json', 'test.json', [1,2,3,4,5,6,7,8], [9], [10])
    """

    # Tease params to correct type if necessary
    if type(train_fold_nums) is int:
        train_fold_nums = [train_fold_nums]
    if type(valid_fold_nums) is int:
        valid_fold_nums = [valid_fold_nums]
    if type(test_fold_nums) is int:
        test_fold_nums = [test_fold_nums]

    # Validate passed fold params
    for fold_num in train_fold_nums:
        if fold_num not in ACCEPTABLE_FOLD_NUMS:
            print(
                f"Train fold numbers {train_fold_nums}, contains an invalid value. Must be in {ACCEPTABLE_FOLD_NUMS}"
            )
            logger.info(
                f"Train fold numbers {train_fold_nums}, contains an invalid value. Must be in {ACCEPTABLE_FOLD_NUMS}"
            )
            return
    for fold_num in valid_fold_nums:
        if fold_num not in ACCEPTABLE_FOLD_NUMS:
            print(
                f"Validation fold numbers {valid_fold_nums}, contains an invalid value. Must be in {ACCEPTABLE_FOLD_NUMS}"
            )
            logger.info(
                f"Validation fold numbers {valid_fold_nums}, contains an invalid value. Must be in {ACCEPTABLE_FOLD_NUMS}"
            )
            return
    for fold_num in test_fold_nums:
        if fold_num not in ACCEPTABLE_FOLD_NUMS:
            print(
                f"Test fold numbers {test_fold_nums}, contains an invalid value. Must be in {ACCEPTABLE_FOLD_NUMS}"
            )
            logger.info(
                f"Test fold numbers {test_fold_nums}, contains an invalid value. Must be in {ACCEPTABLE_FOLD_NUMS}"
            )
            return

    # Check if train, and valid and train and test folds are exclusive
    if folds_overlap(train_fold_nums, valid_fold_nums):
        print(
            f"Train {train_fold_nums}, and Valid {valid_fold_nums} folds must be mutually exclusive!"
        )
        logger.info(
            f"Train {train_fold_nums}, and Valid {valid_fold_nums} folds must be mutually exclusive!"
        )
        return
    if folds_overlap(train_fold_nums, test_fold_nums):
        print(
            f"Train {train_fold_nums} and Test {test_fold_nums} folds must be mutually exclusive!"
        )
        logger.info(
            f"Train {train_fold_nums} and Test {test_fold_nums} folds must be mutually exclusive!"
        )
        return

    # If the dataset doesn't exist yet, prompt the user to set or download it
    if not check_folders(audio_data_folder):
        prompt_download_urban_sound_8k(audio_data_folder)
        return

    # Don't need to do this every single time
    if skip_manifest_creation is True:
        return

    # If our modified metadata file does not exist, create it
    urban_sound_8k_speechbrain_metadata_csv_path = os.path.join(
        os.path.abspath(data_folder), "metadata/", MODIFIED_METADATA_FILE_NAME
    )
    if not os.path.exists(urban_sound_8k_speechbrain_metadata_csv_path):
        urban_sound_8k_speechbrain_metadata_csv_path = create_metadata_speechbrain_file(
            data_folder
        )
        # TODO: If it does not exist, we create it, but next step will certainly fail?

    # Read the metadata into a dictionary
    # Every key of this dictionary is now one of the sound filenames, without the ".wav" suffix
    metadata = load_data_csv(urban_sound_8k_speechbrain_metadata_csv_path)

    # List files and create manifest from list
    logger.info(
        f"Creating {save_json_train}, {save_json_valid}, and {save_json_test}"
    )

    # Creating json files
    create_json(metadata, audio_data_folder, train_fold_nums, save_json_train)
    create_json(metadata, audio_data_folder, valid_fold_nums, save_json_valid)
    create_json(metadata, audio_data_folder, test_fold_nums, save_json_test)


def create_json(metadata, audio_data_folder, folds_list, json_file):
    """
    Creates the json file given a list of wav files.
    Arguments
    ---------
    metadata: dict
        A dictionary containing the UrbanSound8k metadata file modified for the
        SpeechBrain, such that keys are IDs (which are the .wav file names without the file extension).
    folds_list : list of int
        The list of folds [1,10] to include in this batch
    json_file : str
        The path of the output json file
    """
    # Processing all the wav files in the list
    json_dict = {}

    for ID, sample_metadata in metadata.items():
        fold_num = int(sample_metadata["fold"])
        if fold_num in folds_list:
            # Reading the signal (to retrieve duration in seconds)
            wav_file = os.path.join(
                os.path.abspath(audio_data_folder),
                "fold" + str(fold_num) + "/",
                sample_metadata["slice_file_name"],
            )
            try:

                signal = read_audio(wav_file)
                file_info = torchaudio.info(wav_file)

                # If we're using sox/soundfile backend, file_info will have the old type
                if isinstance(
                    file_info, torchaudio.backend.common.AudioMetaData
                ):
                    duration = signal.shape[0] / file_info.sample_rate
                else:
                    duration = signal.shape[0] / file_info[0].rate

                # Create entry for this sample ONLY if we have successfully read-in the file using SpeechBrain/torchaudio
                json_dict[ID] = {
                    "wav": sample_metadata["slice_file_name"],
                    "classID": int(sample_metadata["classID"]),
                    "class_string": sample_metadata["class_string"],
                    "salience": int(sample_metadata["salience"]),
                    "fold": sample_metadata["fold"],
                    "duration": duration,
                }
            except Exception:
                print(
                    f"There was a problem reading the file:{wav_file}. Skipping duration field for it."
                )
                logger.exception(
                    f"There was a problem reading the file:{wav_file}. Skipping it."
                )

    # Writing the dictionary to the json file
    # Need to make sure sub folder "manifest" exists, if not create it
    parent_dir = os.path.dirname(json_file)
    if not os.path.exists(parent_dir):
        os.mkdir(parent_dir)

    with open(json_file, mode="w") as json_f:
        json.dump(json_dict, json_f, indent=2)

    logger.info(f"{json_file} successfully created!")


def folds_overlap(list1, list2):
    """Returns True if any passed lists has incorrect type OR has items in common."""
    if (type(list1) != list) or (type(list2) != list):
        return True
    if any(item in list1 for item in list2):
        return True
    return False


def check_folders(*folders):
    """Returns False if any passed folder does not exist."""
    for folder in folders:
        if not os.path.exists(folder):
            return False
    return True


def full_path_to_audio_file(data_folder, slice_file_name, fold_num):
    """Get path to file given slice file name and fold number
    Arguments
    ---------
    slice_file_name : str
        Filename.
    fold_num : int
        Fold number.
    Returns
    ------
    string containing absolute path to corresponding file
    """
    return os.path.join(
        os.path.abspath(data_folder),
        "audio/",
        "fold" + str(fold_num) + "/",
        slice_file_name,
    )


def create_metadata_speechbrain_file(data_folder):
    """Get path to file given slice file name and fold number
    Arguments
    ---------
    data_folder : str
        UrbanSound8k data folder.
    Returns
    ------
    string containing absolute path to metadata csv file modified for SpeechBrain or None if source file not found
    """
    import pandas as pd

    urban_sound_8k_metadata_csv_path = os.path.join(
        os.path.abspath(data_folder), "metadata/UrbanSound8K.csv"
    )
    if not os.path.exists(urban_sound_8k_metadata_csv_path):
        return None

    urbansound_metadata_df = pd.read_csv(urban_sound_8k_metadata_csv_path)
    # SpeechBrain wants an ID column
    urbansound_metadata_df["ID"] = urbansound_metadata_df.apply(
        lambda row: removesuffix(row["slice_file_name"], ".wav"), axis=1
    )
    urbansound_metadata_df = urbansound_metadata_df.rename(
        columns={"class": "class_string"}
    )

    urban_sound_speechbrain_metadata_csv_path = os.path.join(
        os.path.abspath(data_folder), "metadata/", MODIFIED_METADATA_FILE_NAME
    )
    urbansound_metadata_df.to_csv(
        urban_sound_speechbrain_metadata_csv_path, index=False
    )
    return urban_sound_speechbrain_metadata_csv_path


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def removesuffix(somestring, suffix):
    """Removed a suffix from a string
    Arguments
    ---------
    somestring : str
        Any string.
    suffix : str
        Suffix to be removed from somestring.
    Returns
    ------
    string resulting from suffix removed from somestring, if found, unchanged otherwise
    """
    if somestring.endswith(suffix):
        return somestring[: -1 * len(suffix)]
    else:
        return somestring


def prompt_download_urban_sound_8k(destination):
    """Prompt to download dataset
    Arguments
    ---------
    destination : str
        Place to put dataset.
    """
    print(
        "UrbanSound8k data is missing from {}!\nRequest it from here: {}".format(
            destination, URBAN_SOUND_8K_DOWNLOAD_FORM_URL
        )
    )


# Testing
if __name__ == "__main__":
    import speechbrain

    # Data preparation, to be run on only one process.
    speechbrain.utils.distributed.run_on_main(
        prepare_urban_sound_8k,
        kwargs={
            "data_folder": "/Volumes/BigMule/BigDevDocuments/UrbanSound8K",
            "audio_data_folder": "/Volumes/BigMule/BigDevDocuments/UrbanSound8K/audio",
            "save_json_train": "./UrbanSound8k/manifest/train.json",
            "save_json_valid": "./UrbanSound8k/manifest/valid.json",
            "save_json_test": "./UrbanSound8k/manifest/test.json",
            "train_fold_nums": [1, 2, 3, 4, 5, 6, 7, 8],
            "valid_fold_nums": [9],
            "test_fold_nums": [10],
        },
    )
