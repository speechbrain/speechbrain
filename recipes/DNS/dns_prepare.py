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
NOISE_CSV = "tr_noise.csv"
CLEAN_CSV = "tr_clean.csv"
TEST_CSV = "test.csv"
SAMPLERATE = 16000


def prepare_dns(data_folder, save_folder, seg_size=10.0, mode="dynamic"):
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
    mode : str
        'dynamic' or 'static' for dynamically mixing clean and noise data
        during training or statically mixing before training.

    Example
    -------
    This example requires the actual DNS dataset:
    The "training" folder is expected after the dataset is downloaded and processed.

    >>> from recipes.DNS.dns_prepare import prepare_dns
    >>> data_folder='datasets/DNS-Challenge'
    >>> save_folder='DNS_prepared'
    >>> prepare_dns(data_folder,save_folder)
    """

    # Additional checks to make sure the data folder contains DNS
    _check_DNS_folders(data_folder, mode)

    train_folder = os.path.join(data_folder, "datasets")
    test_folder = os.path.join(
        data_folder, "datasets/test_set/synthetic/no_reverb"
    )

    # Setting file extension.
    extension = [".wav"]

    # Setting the save folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Setting ouput files
    save_csv_noise = os.path.join(save_folder, NOISE_CSV)
    save_csv_clean = os.path.join(save_folder, CLEAN_CSV)
    save_csv_test = os.path.join(save_folder, TEST_CSV)

    msg = "\tCreating csv file for the ms_DNS Dataset.."
    logger.debug(msg)

    # Check if this phase is already done (if so, skip it)
    if skip(save_folder):

        msg = "\t%s sucessfully created!" % (save_csv_noise)
        logger.debug(msg)

        msg = "\t%s sucessfully created!" % (save_csv_clean)
        logger.debug(msg)

        msg = "\t%s sucessfully created!" % (save_csv_test)
        logger.debug(msg)

        return

    # Creating csv file for training data
    wav_lst_noise = get_all_files(
        os.path.join(train_folder, "noise"), match_and=extension
    )
    wav_lst_clean = get_all_files(
        os.path.join(train_folder, "clean"), match_and=extension
    )

    create_csv(
        wav_lst_noise, save_csv_noise,
    )
    create_csv(
        wav_lst_clean, save_csv_clean, seg_size=seg_size,
    )

    # Creating csv file for test data
    wav_lst_test = get_all_files(
        test_folder, match_and=extension, exclude_or=["/clean/"],
    )

    create_csv(
        wav_lst_test, save_csv_test, has_target=True,
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

    split_files = [NOISE_CSV, CLEAN_CSV, TEST_CSV]
    for split in split_files:
        if not os.path.isfile(os.path.join(save_folder, split)):
            skip = False

    return skip


def create_csv(wav_lst, csv_file, seg_size=None, has_target=False):
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
            "wav",
            "wav_format",
            "wav_opts",
            "target",
            "target_format",
            "target_opts",
        ]
    ]

    # Processing all the wav files in the list
    fileid = 0
    for wav_file in wav_lst:
        # Getting fileids
        full_file_name = wav_file.split("/")[-1]

        if has_target:
            fileid = full_file_name.split("_")[-1]

            target_folder = os.path.join(
                os.path.split(os.path.split(wav_file)[0])[0], "clean"
            )
            target_file = target_folder + "/clean_fileid_" + fileid
        else:
            target_file = ""

        # Reading the signal (to retrieve duration in seconds)
        signal = read_wav_soundfile(wav_file)
        duration = signal.shape[0] / SAMPLERATE

        # Composition of the csv_line
        if not seg_size:
            csv_line = [
                full_file_name,
                str(duration),
                wav_file,
                "wav",
                "",
                target_file,
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
                    full_file_name + str(idx),
                    str(seg_size),
                    wav_file,
                    "wav",
                    "start:{} stop:{}".format(start, stop),
                    target_file,
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


def _check_DNS_folders(data_folder, mode):
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
    test_folder = os.path.join(
        data_folder, "datasets/test_set/synthetic/no_reverb"
    )

    # Checking testset folder
    if not os.path.exists(test_folder):

        err_msg = (
            "the folder %s does not exist (it is expected in "
            "the DNS dataset)" % (test_folder)
        )
        raise FileNotFoundError(err_msg)

    if mode == "static":
        train_clean_folder = os.path.join(data_folder, "training/clean")
        train_noise_folder = os.path.join(data_folder, "training/noise")
        train_noisy_folder = os.path.join(data_folder, "training/noisy")

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

    elif mode == "dynamic":
        train_clean_folder = os.path.join(data_folder, "datasets/clean")
        train_noise_folder = os.path.join(data_folder, "datasets/noise")

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
