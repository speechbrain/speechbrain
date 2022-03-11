"""
Data preparation.
Download: https://dvoice.ma/
Author
------
Abdou Mohamed Naira
"""

import os
import csv
import re
import logging
import torchaudio
import unicodedata
from tqdm.contrib import tzip

import torch
import random
import pandas as pd
from tqdm import tqdm
import numpy as np
from speechbrain.dataio.dataio import read_audio
from speechbrain.processing.speech_augmentation import SpeedPerturb
from speechbrain.processing.speech_augmentation import DropChunk
from speechbrain.processing.speech_augmentation import DropFreq
from speechbrain.processing.speech_augmentation import DoClip
from speechbrain.lobes.augment import TimeDomainSpecAugment



logger = logging.getLogger(__name__)

def prepare_dvoice(
    data_folder,
    save_folder,
    train_csv_file=None,
    dev_csv_file=None,
    test_csv_file=None,
    accented_letters=False,
    language="dar",
    skip_prep=False,

):

    if skip_prep:
        return

    # If not specified point toward standard location w.r.t CommonVoice tree
    if train_csv_file is None:
        train_csv_file = data_folder + "texts/train.csv"
    else:
        train_csv_file = train_csv_file

    if dev_csv_file is None:
        dev_csv_file = data_folder + "texts/dev.csv"
    else:
        dev_csv_file = dev_csv_file

    if test_csv_file is None:
        test_csv_file = data_folder + "texts/test.csv"
    else:
        test_csv_file = test_csv_file

    # Setting the save folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        
    # Setting ouput files
    save_csv_train = save_folder + "/train.csv"
    save_csv_dev = save_folder + "/dev.csv"
    save_csv_test = save_folder + "/test.csv"

    # If csv already exists, we skip the data preparation
    if skip(save_csv_train, save_csv_dev, save_csv_test):

        msg = "%s already exists, skipping data preparation!" % (save_csv_train)
        logger.info(msg)

        msg = "%s already exists, skipping data preparation!" % (save_csv_dev)
        logger.info(msg)

        msg = "%s already exists, skipping data preparation!" % (save_csv_test)
        logger.info(msg)

        return

    # Additional checks to make sure the data folder contains Common Voice
    check_commonvoice_folders(data_folder)

    # Creating csv file for training data
    if train_csv_file is not None:

        create_csv(
            train_csv_file,
            save_csv_train,
            data_folder,
            accented_letters,
            language,
        )

    # Creating csv file for dev data
    if dev_csv_file is not None:

        create_csv(
            dev_csv_file,
            save_csv_dev,
            data_folder,
            accented_letters,
            language,
        )

    # Creating csv file for test data
    if test_csv_file is not None:

        create_csv(
            test_csv_file,
            save_csv_test,
            data_folder,
            accented_letters,
            language,
        )


def train_validate_test_split(
    df, train_percent=0.6, validate_percent=0.2, seed=None
):    
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.iloc[perm[:train_end]]
    validate = df.iloc[perm[train_end:validate_end]]
    test = df.iloc[perm[validate_end:]]
    return train, validate, test


def skip(save_csv_train, save_csv_dev, save_csv_test):
    """
    Detects if the DVoice data preparation has been already done.
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
        and os.path.isfile(save_csv_dev)
        and os.path.isfile(save_csv_test)
    ):
        skip = True

    return skip


def create_csv(
    orig_csv_file, csv_file, data_folder, accented_letters=False, language="dar"
):
    """
    Creates the csv file given a list of wav files.
    Arguments
    ---------
    orig_csv_file : str
        Path to the DVoice csv file (standard file).
    data_folder : str
        Path of the DVoice dataset.
    accented_letters : bool, optional
        Defines if accented letters will be kept as individual letters or
        transformed to the closest non-accented letters.
    Returns
    -------
    None
    """

    # Check if the given files exists
    if not os.path.isfile(orig_csv_file):
        msg = "\t%s doesn't exist, verify your dataset!" % (orig_csv_file)
        logger.info(msg)
        raise FileNotFoundError(msg)

    # We load and skip the header
    loaded_csv = open(orig_csv_file, "r").readlines()[1:]
    nb_samples = str(len(loaded_csv))
    msg = "Preparing CSV files for %s samples ..." % (str(nb_samples))
    logger.info(msg)

    # Adding some Prints
    msg = "Creating csv lists in %s ..." % (csv_file)
    logger.info(msg)

    csv_lines = [["ID", "duration", "wav", "spk_id", "wrd"]]

    # Start processing lines
    total_duration = 0.0
    for line in tzip(loaded_csv):

        line = line[0]
        # Path is at indice 1 in DVoice csv files. And .mp3 files
        # are located in datasets/lang/clips/

        mp3_path = data_folder + "/wavs/" + line.split("\t")[0]
        file_name = line.split("\t")[0]
        spk_id = line.split("\t")[0].replace(".wav", "")
        snt_id = file_name

        # Setting torchaudio backend to sox-io (needed to read mp3 files)
        if torchaudio.get_audio_backend() != "sox_io":
            logger.warning("This recipe needs the sox-io backend of torchaudio")
            logger.warning("The torchaudio backend is changed to sox_io")
            torchaudio.set_audio_backend("sox_io")

        duration = float(line.split("\t")[2])
        total_duration += duration

        # Getting transcript
        words = line.split("\t")[1]

        # Unicode Normalization
        # words = unicode_normalisation(words)

        # !! Language specific cleaning !!
        # Important: feel free to specify the text normalization
        # corresponding to your alphabet.

        if language == "dar":
            HAMZA = "\u0621"
            ALEF_MADDA = "\u0622"
            ALEF_HAMZA_ABOVE = "\u0623"
            letters = (
                "ابتةثجحخدذرزسشصضطظعغفقكلمنهويءآأؤإئ"
                + HAMZA
                + ALEF_MADDA
                + ALEF_HAMZA_ABOVE
            )
            words = re.sub("[^" + letters + "]+", " ", words).upper()

        # # Remove accents if specified
        # if not accented_letters:
        #     words = strip_accents(words)
        #     words = words.replace("'", " ")
        #     words = words.replace("’", " ")

        # # Remove multiple spaces
        # words = re.sub(" +", " ", words)

        # # Remove spaces at the beginning and the end of the sentence
        # words = words.lstrip().rstrip()

        # # Getting chars
        # chars = words.replace(" ", "_")
        # chars = " ".join([char for char in chars][:])

        # Remove too short sentences (or empty):
        # if len(words.split(" ")) < 3:
        #     continue

        # Composition of the csv_line
        csv_line = [snt_id, str(duration), mp3_path, spk_id, str(words)]

        # Adding this line to the csv_lines list
        csv_lines.append(csv_line)

    # Writing the csv lines
    with open(csv_file, mode="w", encoding="utf-8") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )

        for line in csv_lines:
            csv_writer.writerow(line)

    # Final prints
    msg = "%s successfully created!" % (csv_file)
    logger.info(msg)
    msg = "Number of samples: %s " % (str(len(loaded_csv)))
    logger.info(msg)
    msg = "Total duration: %s Hours" % (str(round(total_duration / 3600, 2)))
    logger.info(msg)


def check_commonvoice_folders(data_folder):
    """
    Check if the data folder actually contains the DVoice dataset.
    If not, raises an error.
    Returns
    -------
    None
    Raises
    ------
    FileNotFoundError
        If data folder doesn't contain DVoice dataset.
    """

    files_str = "/wavs"

    # Checking clips
    if not os.path.exists(data_folder + files_str):

        err_msg = (
            "the folder %s does not exist (it is expected in "
            "the DVoice dataset)" % (data_folder + files_str)
        )
        raise FileNotFoundError(err_msg)


def unicode_normalisation(text):

    try:
        text = unicode(text, "utf-8")
    except NameError:  # unicode is a default on python 3
        pass
    return str(text)


def strip_accents(text):

    text = (
        unicodedata.normalize("NFD", text)
        .encode("ascii", "ignore")
        .decode("utf-8")
    )

    return str(text)

