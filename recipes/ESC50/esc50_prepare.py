"""
Creates data manifest files for ESC50
If the data does not exist in the specified --data_folder, we download the data automatically.

https://github.com/karoldvl/ESC-50/

Authors:
 * Cem Subakan 2022, 2023
 * Francesco Paissan 2022, 2023

 Adapted from the Urbansound8k recipe.
"""

import os
import shutil
import json
import logging
import torchaudio
from speechbrain.dataio.dataio import read_audio
from speechbrain.dataio.dataio import load_data_csv
from speechbrain.pretrained import fetch

logger = logging.getLogger(__name__)

ESC50_DOWNLOAD_URL = "https://github.com/karoldvl/ESC-50/archive/master.zip"
MODIFIED_METADATA_FILE_NAME = "esc50_speechbrain.csv"

ACCEPTABLE_FOLD_NUMS = [1, 2, 3, 4, 5]


def download_esc50(data_path):
    """
    This function automatically downloads the ESC50 dataset to the specified data path in the data_path variable

    Arguments
    ---------
    data_path: str or Path
        Directory used to save the dataset.
    """
    if not os.path.exists(os.path.join(data_path, "meta")):
        print(
            f"ESC50 is missing. We are now downloading it. Be patient it's a 600M file. You can check {data_path}/temp_download to see the download progression"
        )
        temp_path = os.path.join(data_path, "temp_download")

        # download the data
        fetch(
            "master.zip",
            "https://github.com/karoldvl/ESC-50/archive/",
            savedir=temp_path,
        )

        # unpack the .zip file
        shutil.unpack_archive(os.path.join(temp_path, "master.zip"), data_path)

        # move the files up to the datapath
        files = os.listdir(os.path.join(data_path, "ESC-50-master"))
        for fl in files:
            shutil.move(os.path.join(data_path, "ESC-50-master", fl), data_path)

        # remove the unused datapath
        shutil.rmtree(os.path.join(data_path, "temp_download"))
        shutil.rmtree(os.path.join(data_path, "ESC-50-master"))

        print(f"ESC50 is downloaded in {data_path}")


def prepare_esc50(
    data_folder,
    audio_data_folder,
    save_json_train,
    save_json_valid,
    save_json_test,
    train_fold_nums=[1, 2, 3],
    valid_fold_nums=[4],
    test_fold_nums=[5],
    skip_manifest_creation=False,
):
    """
    Prepares the json files for the ESC50 dataset.
    Prompts to download the dataset if it is not found in the `data_folder`.
    Arguments
    ---------
    data_folder : str
        Path to the folder where the ESC50 dataset (including the metadata) is stored.
    audio_data_folder: str
        Path to the folder where the ESC50 dataset audio files are stored.
    save_json_train : str
        Path where the train data specification file will be saved.
    save_json_valid : str
        Path where the validation data specification file will be saved.
    save_json_test : str
        Path where the test data specification file will be saved.
    train_folds: list or int (integers [1,5])
        A list of integers defining which pre-defined "folds" to use for training. Must be
        exclusive of valid_folds and test_folds.
    valid_folds: list or int (integers [1,5])
        A list of integers defining which pre-defined "folds" to use for validation. Must be
        exclusive of train_folds and test_folds.
    test_folds: list or int (integers [1,5])
        A list of integers defining which pre-defined "folds" to use for test. Must be
        exclusive of train_folds and valid_folds.
    Example
    -------
    >>> data_folder = '/path/to/ESC-50-master'
    >>> prepare_urban_sound_8k(data_folder, 'train.json', 'valid.json', 'test.json', [1,2,3], [4], [5])
    """
    download_esc50(data_folder)

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

    # Don't need to do this every single time
    if skip_manifest_creation is True:
        return

    # If our modified metadata file does not exist, create it
    esc50_speechbrain_metadata_csv_path = os.path.join(
        os.path.abspath(data_folder), "metadata/", MODIFIED_METADATA_FILE_NAME
    )
    if not os.path.exists(esc50_speechbrain_metadata_csv_path):
        esc50_speechbrain_metadata_csv_path = create_metadata_speechbrain_file(
            data_folder
        )

    # Read the metadata into a dictionary
    # Every key of this dictionary is now one of the sound filenames, without the ".wav" suffix
    metadata = load_data_csv(esc50_speechbrain_metadata_csv_path)

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
        A dictionary containing the ESC50 metadata file modified for the
        SpeechBrain, such that keys are IDs (which are the .wav file names without the file extension).
    audio_data_folder : str or Path
        Data folder that stores ESC50 samples.
    folds_list : list of int
        The list of folds [1,5] to include in this batch
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
                # "fold" + str(fold_num) + "/",
                sample_metadata["filename"],
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
                    "wav": sample_metadata["filename"],
                    "classID": int(sample_metadata["target"]),
                    "class_string": sample_metadata["class_string"],
                    # "salience": int(sample_metadata["salience"]),
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
    """Returns True if any passed lists has incorrect type OR has items in common.

    Arguments
    ----------
    list1 : list
        First list for comparison.
    list2 : list
        Second list for comparison.
    """
    if (type(list1) != list) or (type(list2) != list):
        return True
    if any(item in list1 for item in list2):
        return True
    return False


def check_folders(*folders):
    """Returns False if any passed folder does not exist.

    Arguments
    ---------
    folders: list
        Folders to check.
    """
    for folder in folders:
        if not os.path.exists(folder):
            return False
    return True


def full_path_to_audio_file(data_folder, slice_file_name, fold_num):
    """Get path to file given slice file name and fold number
    Arguments
    ---------
    data_foder : str
        Folder that contains the dataset.
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
        ESC50 data folder.
    Returns
    ------
    string containing absolute path to metadata csv file modified for SpeechBrain or None if source file not found
    """
    import pandas as pd

    esc50_metadata_csv_path = os.path.join(
        os.path.abspath(data_folder), "meta/esc50.csv"
    )
    if not os.path.exists(esc50_metadata_csv_path):
        return None

    esc50_metadata_df = pd.read_csv(esc50_metadata_csv_path)
    # SpeechBrain wants an ID column
    esc50_metadata_df["ID"] = esc50_metadata_df.apply(
        lambda row: removesuffix(row["filename"], ".wav"), axis=1
    )
    esc50_metadata_df = esc50_metadata_df.rename(
        columns={"category": "class_string"}
    )

    esc50_speechbrain_metadata_csv_path = os.path.join(
        os.path.abspath(data_folder), "meta/", MODIFIED_METADATA_FILE_NAME
    )
    esc50_metadata_df.to_csv(esc50_speechbrain_metadata_csv_path, index=False)
    return esc50_speechbrain_metadata_csv_path


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
