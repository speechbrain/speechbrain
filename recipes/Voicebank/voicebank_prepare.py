# -*- coding: utf-8 -*-
"""
Data preparation.

Download: https://datashare.is.ed.ac.uk/handle/10283/2791

Author
------
Szu-Wei Fu, 2020
"""

import os
import csv
import logging
from speechbrain.utils.data_utils import get_all_files
from speechbrain.data_io.data_io import read_wav_soundfile

logger = logging.getLogger(__name__)
TRAIN_CSV = "train.csv"
TEST_CSV = "test.csv"
VALID_CSV = "valid.csv"
SAMPLERATE = 16000


def prepare_voicebank(data_folder, save_folder):
    """
    Prepares the csv files for the Voicebank dataset.
    Arguments
    ---------
    data_folder : str
        Path to the folder where the original Voicebank dataset is stored.
    save_folder : str
        The directory where to store the csv files.

    Example
    -------
    This example requires the actual Voicebank dataset.
    The noisy_vctk_prepare.py can be used to download the dataset.
    ```
    data_folder = '/path/to/datasets/Voicebank'
    save_folder = 'exp/Voicebank_exp'
    VoicebankPreparer(data_folder, save_folder)
    ```
    """

    train_clean_folder = os.path.join(
        data_folder, "clean_trainset_28spk_wav_16k/"
    )
    train_noisy_folder = os.path.join(
        data_folder, "noisy_trainset_28spk_wav_16k/"
    )
    test_clean_folder = os.path.join(data_folder, "clean_testset_wav_16k/")
    test_noisy_folder = os.path.join(data_folder, "noisy_testset_wav_16k/")

    # Setting file extension.
    extension = [".wav"]

    # Setting the save folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Setting ouput files
    save_csv_train = os.path.join(save_folder, TRAIN_CSV)
    save_csv_test = os.path.join(save_folder, TEST_CSV)
    save_csv_valid = os.path.join(save_folder, VALID_CSV)

    # Check if this phase is already done (if so, skip it)
    if skip(save_csv_train, save_csv_test, save_csv_valid):

        msg = "\t%s sucessfully created!" % (save_csv_train)
        logger.debug(msg)

        msg = "\t%s sucessfully created!" % (save_csv_test)
        logger.debug(msg)

        msg = "\t%s sucessfully created!" % (save_csv_valid)
        logger.debug(msg)

        return

    # Additional checks to make sure the data folder contains Voicebank
    check_Voicebank_folders(
        train_clean_folder,
        train_noisy_folder,
        test_clean_folder,
        test_noisy_folder,
    )

    msg = "\tCreating csv file for the Voicebank Dataset.."
    logger.debug(msg)

    # Creating csv file for training data
    wav_lst_train = get_all_files(
        train_noisy_folder,
        match_and=extension,
        exclude_or=[
            "p226",
            "p287",
        ],  # These two speakers are used for validation set
    )

    create_csv(
        wav_lst_train,
        save_csv_train,
        train_clean_folder,
        test_clean_folder,
        is_train_folder=True,
    )

    # Creating csv file for validation data
    wav_lst_valid = get_all_files(
        train_noisy_folder, match_and=extension, match_or=["p226", "p287"],
    )

    create_csv(
        wav_lst_valid,
        save_csv_valid,
        train_clean_folder,
        test_clean_folder,
        is_train_folder=True,
    )

    # Creating csv file for testing data
    wav_lst_test = get_all_files(test_noisy_folder, match_and=extension,)

    create_csv(
        wav_lst_test,
        save_csv_test,
        train_clean_folder,
        test_clean_folder,
        is_train_folder=False,
    )


def skip(save_csv_train, save_csv_test, save_csv_valid):
    """
    Detects if the Voicebank data_preparation has been already done.
    If the preparation has been done, we can skip it.
    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """

    # Checking folders and save options
    skip = False

    if (
        os.path.isfile(save_csv_train)
        and os.path.isfile(save_csv_test)
        and os.path.isfile(save_csv_valid)
    ):
        skip = True

    return skip


def create_csv(
    wav_lst, csv_file, train_clean_folder, test_clean_folder, is_train_folder
):
    """
    Creates the csv file given a list of wav files.
    Arguments
    ---------
    wav_lst : list
        The list of wav files.
    csv_file : str
        The path of the output csv file
    Returns
    -------
    None
    """

    # Adding some Prints
    msg = '\t"Creating csv lists in  %s..."' % (csv_file)
    logger.debug(msg)

    csv_lines = [
        [
            "ID",
            "duration",
            "noisy_wav",
            "noisy_wav_format",
            "noisy_wav_opts",
            "clean_wav",
            "clean_wav_format",
            "clean_wav_opts",
        ]
    ]

    # Processing all the wav files in the list
    for wav_file in wav_lst:  # ex:p203_122.wav

        # Example wav_file: p232_001.wav
        snt_id = wav_file.split("/")[-1]

        if is_train_folder:
            clean_folder = train_clean_folder
        else:
            clean_folder = test_clean_folder
        clean_wav = clean_folder + snt_id

        # Reading the signal (to retrieve duration in seconds)
        signal = read_wav_soundfile(wav_file)
        duration = signal.shape[0] / SAMPLERATE

        # Composition of the csv_line
        csv_line = [
            snt_id.replace(".wav", ""),
            str(duration),
            wav_file,
            "wav",
            "",
            clean_wav,
            "wav",
            "",
        ]

        # Adding this line to the csv_lines list
        csv_lines.append(csv_line)

    # Writing the csv lines
    with open(csv_file, mode="w") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )

        for line in csv_lines:
            csv_writer.writerow(line)

    # Final prints
    msg = "\t%s sucessfully created!" % (csv_file)
    logger.debug(msg)


def check_Voicebank_folders(
    train_clean_folder, train_noisy_folder, test_clean_folder, test_noisy_folder
):
    """
    Check if the data folder actually contains the Voicebank dataset.
    If not, raises an error.
    Returns
    -------
    None
    Raises
    ------
    FileNotFoundError
        If data folder doesn't contain Voicebank dataset.
    """

    # Checking train_clean folder
    if not os.path.exists(train_clean_folder):

        err_msg = (
            "the folder %s does not exist (it is expected in "
            "the Voicebank dataset)" % (train_clean_folder)
        )
        raise FileNotFoundError(err_msg)

    # Checking train_noisy folder
    if not os.path.exists(train_noisy_folder):

        err_msg = (
            "the folder %s does not exist (it is expected in "
            "the Voicebank dataset)" % (train_noisy_folder)
        )
        raise FileNotFoundError(err_msg)

    # Checking test_clean folder
    if not os.path.exists(test_clean_folder):

        err_msg = (
            "the folder %s does not exist (it is expected in "
            "the Voicebank dataset)" % (test_clean_folder)
        )
        raise FileNotFoundError(err_msg)

    # Checking test_noisy folder
    if not os.path.exists(test_noisy_folder):

        err_msg = (
            "the folder %s does not exist (it is expected in "
            "the Voicebank dataset)" % (test_noisy_folder)
        )
        raise FileNotFoundError(err_msg)
