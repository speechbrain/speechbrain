"""
Data preparation.

Download: https://github.com/microsoft/DNS-Challenge

Author
------
Chien-Feng Liao 2020
"""

import os
import csv
import logging
from speechbrain.utils.data_utils import get_all_files
from speechbrain.data_io.data_io import read_wav_soundfile

logger = logging.getLogger(__name__)
TRAIN_CSV = "train.csv"
TEST_CSV = "test.csv"
SAMPLERATE = 16000


def prepare_dns(data_folder, save_folder, seg_size=10.0):
    """
    prepares the csv files for the DNS challenge dataset.

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original DNS dataset is stored.
    save_folder : str
        The directory where to store the csv files.
    seg_size : float
        Split the file into multiple fix length segments (ms).

    Example
    -------
    This example requires the actual DNS dataset:
    The "training" folder is expected after the dataset is downloaded and processed.

    >>> from recipes.DNS.dns_prepare import prepare_dns
    >>> data_folder='datasets/DNS-Challenge'
    >>> save_folder='DNS_prepared'
    >>> prepare_dns(data_folder,save_folder)
    """

    train_noisy_folder = os.path.join(data_folder, "training/noisy")
    test_folder = os.path.join(
        data_folder, "datasets/test_set/synthetic/no_reverb"
    )

    # Setting file extension.
    extension = [".wav"]

    # Setting the save folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Setting ouput files
    save_csv_train = os.path.join(save_folder, TRAIN_CSV)
    save_csv_test = os.path.join(save_folder, TEST_CSV)

    # Additional checks to make sure the data folder contains DNS
    _check_DNS_folders(data_folder)

    msg = "\tCreating csv file for the ms_DNS Dataset.."
    logger.debug(msg)

    # Check if this phase is already done (if so, skip it)
    if skip(save_folder):

        msg = "\t%s sucessfully created!" % (save_csv_train)
        logger.debug(msg)

        msg = "\t%s sucessfully created!" % (save_csv_test)
        logger.debug(msg)

        return

    # Creating csv file for training data
    wav_lst_train = get_all_files(train_noisy_folder, match_and=extension,)

    create_csv(
        wav_lst_train, save_csv_train, is_noise_folder=True, seg_size=seg_size,
    )

    # Creating csv file for test data
    wav_lst_test = get_all_files(
        test_folder, match_and=extension, exclude_or=["/clean/"],
    )

    create_csv(
        wav_lst_test, save_csv_test,
    )

    return


def skip(save_folder):
    """
    Detects if the DNS data_preparation has been already done.

    If the preparation has been done, we can skip it.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """

    # Checking folders and save options
    skip = True

    split_files = [TRAIN_CSV, TEST_CSV]
    for split in split_files:
        if not os.path.isfile(os.path.join(save_folder, split)):
            skip = False

    return skip


def create_csv(wav_lst, csv_file, is_noise_folder=False, seg_size=None):
    """
    Creates the csv file given a list of wav files.

    Arguments
    ---------
    wav_lst : list
        The list of wav files of a given data split.
    csv_file : str
        The path of the output csv file
    is_noise_folder : boolean
        True if noise files are included
    seg_size: int
        Split the file into multiple fix length segments (ms).

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
            "noise_wav",
            "noise_wav_format",
            "noise_wav_opts",
        ]
    ]

    # Processing all the wav files in the list
    for wav_file in wav_lst:

        # Getting fileids
        full_file_name = wav_file.split("/")[-1]
        fileid = full_file_name.split("_")[-1]

        clean_folder = os.path.join(
            os.path.split(os.path.split(wav_file)[0])[0], "clean"
        )
        clean_wav = clean_folder + "/clean_fileid_" + fileid

        if is_noise_folder:
            noise_folder = os.path.join(
                os.path.split(os.path.split(wav_file)[0])[0], "noise"
            )
            noise_wav = noise_folder + "/noise_fileid_" + fileid
        else:
            noise_wav = ""

        # Reading the signal (to retrieve duration in seconds)
        signal = read_wav_soundfile(wav_file)
        duration = signal.shape[0] / SAMPLERATE

        # Composition of the csv_line
        if not seg_size:
            csv_line = [
                "fileid_" + fileid,
                str(duration),
                wav_file,
                "wav",
                "",
                clean_wav,
                "wav",
                "",
                noise_wav,
                "wav",
                "",
            ]

            # Adding this line to the csv_lines list
            csv_lines.append(csv_line)

        else:
            for idx in range(int(duration // seg_size)):
                start = int(idx * seg_size * SAMPLERATE)
                stop = int((idx + 1) * seg_size * SAMPLERATE)
                csv_line = [
                    "fileid_{}_{}".format(idx, fileid),
                    str(seg_size),
                    wav_file,
                    "wav",
                    "start:{} stop:{}".format(start, stop),
                    clean_wav,
                    "wav",
                    "start:{} stop:{}".format(start, stop),
                    noise_wav,
                    "wav",
                    "start:{} stop:{}".format(start, stop),
                ]

                # Adding this line to the csv_lines list
                csv_lines.append(csv_line)

    # Writing the csv lines
    _write_csv(csv_lines, csv_file)
    # Final prints
    msg = "\t%s sucessfully created!" % (csv_file)
    logger.debug(msg)


def _write_csv(csv_lines, csv_file):
    """
    Writes on the specified csv_file the given csv_files.
    """
    with open(csv_file, mode="w") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )

        for line in csv_lines:
            csv_writer.writerow(line)


def _check_DNS_folders(data_folder):
    """
    Check if the data folder actually contains the DNS training dataset.

    If not, raises an error.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If data folder doesn't contain DNS dataset (training and testset included).
    """
    train_clean_folder = os.path.join(data_folder, "training/clean")
    train_noise_folder = os.path.join(data_folder, "training/noise")
    train_noisy_folder = os.path.join(data_folder, "training/noisy")
    test_folder = os.path.join(
        data_folder, "datasets/test_set/synthetic/no_reverb"
    )

    # Checking clean folder
    if not os.path.exists(train_clean_folder):

        err_msg = (
            "the folder %s does not exist (it is expected in "
            "the DNS dataset)" % (train_clean_folder)
        )
        raise FileNotFoundError(err_msg)

    # Checking noise folder
    if not os.path.exists(train_noise_folder):

        err_msg = (
            "the folder %s does not exist (it is expected in "
            "the DNS dataset)" % (train_noise_folder)
        )
        raise FileNotFoundError(err_msg)

    # Checking noisy folder
    if not os.path.exists(train_noisy_folder):

        err_msg = (
            "the folder %s does not exist (it is expected in "
            "the DNS dataset)" % (train_noisy_folder)
        )
        raise FileNotFoundError(err_msg)

    # Checking testset folder
    if not os.path.exists(test_folder):

        err_msg = (
            "the folder %s does not exist (it is expected in "
            "the DNS dataset)" % (test_folder)
        )
        raise FileNotFoundError(err_msg)
