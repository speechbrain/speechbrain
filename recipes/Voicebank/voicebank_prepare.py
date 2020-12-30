# -*- coding: utf-8 -*-
"""
Data preparation.

Download and resample, use ``download_vctk`` below.
https://datashare.is.ed.ac.uk/handle/10283/2791

Authors:
 * Szu-Wei Fu, 2020
 * Peter Plantinga, 2020
"""

import os
import re
import csv
import string
import urllib
import shutil
import logging
import tempfile
import torchaudio
from torchaudio.transforms import Resample
from speechbrain.utils.data_utils import get_all_files
from speechbrain.data_io.data_io import read_audio

logger = logging.getLogger(__name__)
TRAIN_CSV = "train.csv"
TEST_CSV = "test.csv"
VALID_CSV = "valid.csv"
SAMPLERATE = 16000
TRAIN_SPEAKERS = [
    "p226",
    "p287",
    "p227",
    "p228",
    "p230",
    "p231",
    "p233",
    "p236",
    "p239",
    "p243",
    "p244",
    "p250",
    "p254",
    "p256",
    "p258",
    "p259",
    "p267",
    "p268",
    "p269",
    "p270",
    "p273",
    "p274",
    "p276",
    "p277",
    "p278",
    "p279",
    "p282",
    "p286",
]


def prepare_voicebank(data_folder, save_folder, valid_speaker_count=2):
    """
    Prepares the csv files for the Voicebank dataset.

    Expects the data folder to be the same format as the output of
    ``download_vctk()`` below.

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original Voicebank dataset is stored.
    save_folder : str
        The directory where to store the csv files.
    valid_speaker_count : int
        The number of validation speakers to use (out of 28 in train set).

    Example
    -------
    >>> data_folder = '/path/to/datasets/Voicebank'
    >>> save_folder = 'exp/Voicebank_exp'
    >>> prepare_voicebank(data_folder, save_folder)
    """

    # Setting ouput files
    save_csv_train = os.path.join(save_folder, TRAIN_CSV)
    save_csv_test = os.path.join(save_folder, TEST_CSV)
    save_csv_valid = os.path.join(save_folder, VALID_CSV)

    # Check if this phase is already done (if so, skip it)
    if skip(save_csv_train, save_csv_test, save_csv_valid):
        print("Preparation completed in previous run, skipping.")
        return

    train_clean_folder = os.path.join(
        data_folder, "clean_trainset_28spk_wav_16k"
    )
    train_noisy_folder = os.path.join(
        data_folder, "noisy_trainset_28spk_wav_16k"
    )
    train_txts = os.path.join(data_folder, "trainset_28spk_txt")
    test_clean_folder = os.path.join(data_folder, "clean_testset_wav_16k")
    test_noisy_folder = os.path.join(data_folder, "noisy_testset_wav_16k")
    test_txts = os.path.join(data_folder, "testset_txt")

    # Setting the save folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Additional checks to make sure the data folder contains Voicebank
    check_voicebank_folders(
        train_clean_folder,
        train_noisy_folder,
        train_txts,
        test_clean_folder,
        test_noisy_folder,
        test_txts,
    )

    print("Creating csv files for noisy VoiceBank...")

    # Creating csv file for training data
    extension = [".wav"]
    valid_speakers = TRAIN_SPEAKERS[:valid_speaker_count]
    wav_lst_train = get_all_files(
        train_noisy_folder, match_and=extension, exclude_or=valid_speakers,
    )
    create_csv(wav_lst_train, save_csv_train, train_clean_folder, train_txts)

    # Creating csv file for validation data
    wav_lst_valid = get_all_files(
        train_noisy_folder, match_and=extension, match_or=valid_speakers,
    )
    create_csv(wav_lst_valid, save_csv_valid, train_clean_folder, train_txts)

    # Creating csv file for testing data
    wav_lst_test = get_all_files(test_noisy_folder, match_and=extension)
    create_csv(wav_lst_test, save_csv_test, test_clean_folder, test_txts)


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


def create_csv(wav_lst, csv_file, clean_folder, txt_folder):
    """
    Creates the csv file given a list of wav files.

    Arguments
    ---------
    wav_lst : list
        The list of wav files.
    csv_file : str
        The path of the output csv file
    clean_folder : str
        The location of parallel clean samples.
    txt_folder : str
        The location of the transcript files.
    """
    logger.debug(f"Creating csv lists in {csv_file}")

    csv_lines = [["ID", "duration"]]
    csv_lines[0].extend(["noisy_wav", "noisy_wav_format", "noisy_wav_opts"])
    csv_lines[0].extend(["clean_wav", "clean_wav_format", "clean_wav_opts"])
    csv_lines[0].extend(["char", "char_format", "char_opts"])

    # Processing all the wav files in the list
    for wav_file in wav_lst:  # ex:p203_122.wav

        # Example wav_file: p232_001.wav
        snt_id = os.path.basename(wav_file).replace(".wav", "")
        clean_wav = os.path.join(clean_folder, snt_id + ".wav")

        # Reading the signal (to retrieve duration in seconds)
        signal = read_audio(wav_file)
        duration = signal.shape[0] / SAMPLERATE

        # Reading the transcript
        with open(os.path.join(txt_folder, snt_id + ".txt")) as f:
            words = f.read()

        # Strip punctuation and add spaces (excluding repeats).
        words = words.translate(str.maketrans("", "", string.punctuation))
        chars = " ".join(words.strip().upper())
        chars = chars.replace("   ", " <SP> ")
        chars = re.sub(r"\s{2,}", r" ", chars)
        chars = re.sub(r"(.) \1", r"\1\1", chars)

        # Composition of the csv_line
        csv_line = [snt_id, str(duration)]
        csv_line.extend([wav_file, "wav", ""])
        csv_line.extend([clean_wav, "wav", ""])
        csv_line.extend([chars, "string", ""])

        # Adding this line to the csv_lines list
        csv_lines.append(csv_line)

    # Writing the csv lines
    with open(csv_file, mode="w") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )

        for line in csv_lines:
            csv_writer.writerow(line)

    print(f"{csv_file} successfully created!")


def check_voicebank_folders(*folders):
    """Raises FileNotFoundError if any passed folder does not exist."""
    for folder in folders:
        if not os.path.exists(folder):
            raise FileNotFoundError(
                f"the folder {folder} does not exist (it is expected in "
                "the Voicebank dataset)"
            )


def download_vctk(destination, tmp_dir=None, device="cpu"):
    """Download dataset and perform resample to 16000 Hz.

    Arguments
    ---------
    destination : str
        Place to put final zipped dataset.
    tmp_dir : str
        Location to store temporary files. Will use `tempfile` if not provided.
    device : str
        Passed directly to pytorch's ``.to()`` method. Used for resampling.
    """
    dataset_name = "noisy-vctk-16k"
    if tmp_dir is None:
        tmp_dir = tempfile.gettempdir()
    final_dir = os.path.join(tmp_dir, dataset_name)

    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)

    if not os.path.isdir(final_dir):
        os.mkdir(final_dir)

    prefix = "https://datashare.is.ed.ac.uk/bitstream/handle/10283/2791/"
    noisy_vctk_urls = [
        prefix + "clean_testset_wav.zip",
        prefix + "noisy_testset_wav.zip",
        prefix + "testset_txt.zip",
        prefix + "clean_trainset_28spk_wav.zip",
        prefix + "noisy_trainset_28spk_wav.zip",
        prefix + "trainset_28spk_txt.zip",
    ]

    zip_files = []
    for url in noisy_vctk_urls:
        filename = os.path.join(tmp_dir, url.split("/")[-1])
        zip_files.append(filename)
        if not os.path.isfile(filename):
            print("Downloading " + url)
            with urllib.request.urlopen(url) as response:
                with open(filename, "wb") as tmp_file:
                    print("... to " + tmp_file.name)
                    shutil.copyfileobj(response, tmp_file)

    # Unzip
    for zip_file in zip_files:
        print("Unzipping " + zip_file)
        shutil.unpack_archive(zip_file, tmp_dir, "zip")
        os.remove(zip_file)

    # Move transcripts to final dir
    shutil.move(os.path.join(tmp_dir, "testset_txt"), final_dir)
    shutil.move(os.path.join(tmp_dir, "trainset_28spk_txt"), final_dir)

    # Downsample
    dirs = [
        "noisy_testset_wav",
        "clean_testset_wav",
        "noisy_trainset_28spk_wav",
        "clean_trainset_28spk_wav",
    ]

    downsampler = Resample(orig_freq=48000, new_freq=16000)

    for directory in dirs:
        print("Resampling " + directory)
        dirname = os.path.join(tmp_dir, directory)

        # Make directory to store downsampled files
        dirname_16k = os.path.join(final_dir, directory + "_16k")
        if not os.path.isdir(dirname_16k):
            os.mkdir(dirname_16k)

        # Load files and downsample
        for filename in get_all_files(dirname, match_and=[".wav"]):
            signal, rate = torchaudio.load(filename)
            downsampled_signal = downsampler(signal.view(1, -1).to(device))

            # Save downsampled file
            torchaudio.save(
                os.path.join(dirname_16k, filename[-12:]),
                downsampled_signal[0].cpu(),
                sample_rate=16000,
                channels_first=False,
            )

            # Remove old file
            os.remove(filename)

        # Remove old directory
        os.rmdir(dirname)

    print("Zipping " + final_dir)
    final_zip = shutil.make_archive(
        base_name=final_dir,
        format="zip",
        root_dir=os.path.dirname(final_dir),
        base_dir=os.path.basename(final_dir),
    )

    print(f"Moving {final_zip} to {destination}")
    shutil.move(final_zip, os.path.join(destination, dataset_name + ".zip"))
