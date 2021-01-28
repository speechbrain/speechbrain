"""
Data preparation.

Download: https://github.com/microsoft/DNS-Challenge

Author
------
Chien-Feng Liao 2020
"""

import os
import csv
import torch
import logging
import torchaudio
from speechbrain.utils.data_utils import get_all_files
from speechbrain.dataio.dataio import read_audio
from speechbrain.processing.speech_augmentation import AddNoise

logger = logging.getLogger(__name__)
NOISE_CSV = "tr_noise.csv"
CLEAN_CSV = "tr_clean.csv"
VALID_CSV = "valid.csv"
TEST_CSV = "test.csv"
SAMPLERATE = 16000


def prepare_dns(
    data_folder,
    save_folder,
    seg_size=10.0,
    valid_folder=None,
    valid_ratio=0.002,
    valid_snr_low=0,
    valid_snr_high=40,
    skip_prep=False,
):
    """
    Prepares the csv files for the DNS challenge dataset.

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original DNS dataset is stored.
    save_folder : str
        The directory where to store the csv files.
    seg_size : float
        Split the file into multiple fix length segments (ms).
    valid_ratio : float
        Use this fraction of the training data as a validation set.
    valid_folder : str
        Location for storing mixed validation samples.
    valid_snr_low : float
        Lowest SNR to use when mixing the validation set.
    valid_snr_high : float
        Highest SNR to use when mixing the validiation set.
    skip_prep: bool
        If False, skip data preparation.

    Example
    -------
    >>> # This example requires the actual DNS dataset:
    >>> data_folder = 'datasets/DNS-Challenge'
    >>> save_folder = 'DNS_prepared'
    >>> prepare_dns(data_folder, save_folder)
    """
    if skip_prep:
        return

    if valid_ratio > 0 and valid_folder is None:
        raise ValueError("Must provide folder for storing validation data")

    # Additional checks to make sure the data folder contains DNS
    _check_DNS_folders(data_folder)

    train_folder = os.path.join(data_folder, "datasets")
    test_folder = os.path.join(
        data_folder, "datasets", "test_set", "synthetic", "no_reverb"
    )

    # Setting file extension.
    extension = [".wav"]

    # Setting the save folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Check if this phase is already done (if so, skip it)
    if skip(save_folder):
        logger.info("Preparation completed in previous run.")
        return

    logger.info("Creating csv files for the DNS Dataset...")

    # Setting ouput files
    save_csv_noise = os.path.join(save_folder, NOISE_CSV)
    save_csv_clean = os.path.join(save_folder, CLEAN_CSV)
    save_csv_valid = os.path.join(save_folder, VALID_CSV)
    save_csv_test = os.path.join(save_folder, TEST_CSV)

    # Get the list of files
    wav_lst_noise = get_all_files(
        os.path.join(train_folder, "noise"), match_and=extension
    )
    wav_lst_clean = get_all_files(
        os.path.join(train_folder, "clean"), match_and=extension
    )

    # Clean is excluded here, but will be picked up by `create_csv`
    wav_lst_test = get_all_files(
        test_folder, match_and=extension, exclude_or=["/clean/"],
    )

    # Split training into validation and training
    if valid_ratio > 0:

        # Sort to ensure same validation set for each run.
        wav_lst_noise.sort()
        wav_lst_clean.sort()

        # Split
        valid_count = int(valid_ratio * len(wav_lst_clean))
        valid_lst_noise = wav_lst_noise[:valid_count]
        valid_lst_clean = wav_lst_clean[:valid_count]
        wav_lst_noise = wav_lst_noise[valid_count:]
        wav_lst_clean = wav_lst_clean[valid_count:]

        # Create noise csv to use when adding noise to validation samples.
        save_valid_noise = os.path.join(save_folder, "valid_noise.csv")
        create_csv(save_valid_noise, valid_lst_noise)
        create_csv(
            save_csv_valid,
            valid_lst_clean,
            seg_size=seg_size,
            noise_csv=save_valid_noise,
            noisy_folder=valid_folder,
            noise_snr_low=valid_snr_low,
            noise_snr_high=valid_snr_high,
        )

    # Test set has target in parallel "clean" directory
    create_csv(save_csv_test, wav_lst_test, has_target=True)

    # Create tr_clean.csv and tr_noise.csv for dynamic mixing the training data
    create_csv(save_csv_noise, wav_lst_noise)
    create_csv(save_csv_clean, wav_lst_clean, seg_size=seg_size)


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

    split_files = [NOISE_CSV, CLEAN_CSV, VALID_CSV, TEST_CSV]
    for split in split_files:
        if not os.path.isfile(os.path.join(save_folder, split)):
            skip = False

    return skip


def create_csv(
    csv_file,
    wav_lst,
    seg_size=None,
    has_target=False,
    noise_csv=None,
    noisy_folder=None,
    noise_snr_low=0,
    noise_snr_high=0,
):
    """
    Creates the csv file given a list of wav files.

    Arguments
    ---------
    csv_file : str
        The path of the output csv file
    wav_lst : list
        The list of wav files of a given data split.
    seg_size : int
        Split the file into multiple fix length segments (ms).
    has_target : bool
        Whether clean utterances are present in a similar directory.
    noise_csv : str
        A set of noise files to mix with the signals in `wav_lst`.
    noisy_folder : str
        A location for storing the mixed samples, if `noise_csv` is provided.
    noise_snr_low : float
        The lowest amplitude ratio to use when mixing `noise_csv`.
    noise_snr_high : float
        The highest amplitude ratio to use when mixing `noise_csv`.
    """

    if noise_csv and has_target:
        raise ValueError("Expected only one of `noise_csv` and `has_target`")

    logger.info("Creating csv list: %s" % csv_file)

    csv_lines = [["ID", "duration", "wav", "wav_format", "wav_opts"]]
    if noise_csv or has_target:
        csv_lines[0].extend(["target", "target_format", "target_opts"])

    if noise_csv:
        if not os.path.exists(noisy_folder):
            os.makedirs(noisy_folder)

        noise_adder = AddNoise(
            csv_file=noise_csv,
            snr_low=noise_snr_low,
            snr_high=noise_snr_high,
            pad_noise=True,
            normalize=True,
        )

    # Processing all the wav files in the list
    fileid = 0
    for wav_file in wav_lst:
        full_file_name = os.path.basename(wav_file)

        if has_target:
            fileid = full_file_name.split("_")[-1]
            target_folder = os.path.join(
                os.path.split(os.path.split(wav_file)[0])[0], "clean"
            )
            target_file = os.path.join(target_folder, "clean_fileid_" + fileid)

        # Reading the signal (to retrieve duration in seconds)
        signal = read_audio(wav_file)
        duration = signal.shape[0] / SAMPLERATE

        if noise_csv:
            target = torch.Tensor(signal).unsqueeze(0)
            signal = noise_adder(target, torch.ones(1))
            filepath = os.path.join(noisy_folder, full_file_name)
            torchaudio.save(filepath, signal, SAMPLERATE)
            target_file = wav_file
            wav_file = filepath

        # Composition of the csv_line
        if not seg_size or duration < seg_size:
            csv_line = [full_file_name, str(duration), wav_file, "wav", ""]
            if noise_csv or has_target:
                csv_line.extend([target_file, "wav", ""])
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
                ]
                if noise_csv or has_target:
                    csv_line.extend(
                        [
                            target_file,
                            "wav",
                            "start:{} stop:{}".format(start, stop),
                        ]
                    )

                # Adding this line to the csv_lines list
                csv_lines.append(csv_line)

    # Writing the csv lines
    _write_csv(csv_lines, csv_file)
    logger.info("%s successfully created!" % csv_file)


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
    test_folder = os.path.join(
        data_folder, "datasets", "test_set", "synthetic", "no_reverb"
    )

    # Checking testset folder
    if not os.path.exists(test_folder):

        err_msg = (
            "the folder %s does not exist (it is expected in "
            "the DNS dataset)" % (test_folder)
        )
        raise FileNotFoundError(err_msg)

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
