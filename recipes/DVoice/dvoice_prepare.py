"""
Data preparation.
Download: https://zenodo.org/record/5482551

Author
------
Abdou Mohamed Naira 2022
"""

import csv
import glob
import os
import random
import re
import unicodedata

import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.contrib import tzip

from speechbrain.dataio.dataio import read_audio_info
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


def prepare_dvoice(
    data_folder,
    save_folder,
    train_csv_file=None,
    dev_csv_file=None,
    test_csv_file=None,
    accented_letters=False,
    language="fongbe",
    skip_prep=False,
):
    if skip_prep:
        return

    # If not specified point toward standard location w.r.t DVoice tree
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

    # Setting the ALFFA-Dataset csv files
    ALFFA_LANGUAGES = ["amharic", "fongbe", "wolof"]
    if language in ALFFA_LANGUAGES:
        df = alffa_public_prepare(language, data_folder)
        train, dev, test = train_validate_test_split(df)
        train.to_csv(f"{data_folder}/train.csv", index=False, sep="\t")
        dev.to_csv(f"{data_folder}/dev.csv", index=False, sep="\t")
        test.to_csv(f"{data_folder}/test.csv", index=False, sep="\t")

    if language == "swahili":
        df = swahili_prepare(data_folder)
        train, dev, test = train_validate_test_split(df)
        train.to_csv(f"{data_folder}/train.csv", index=False, sep="\t")
        dev.to_csv(f"{data_folder}/dev.csv", index=False, sep="\t")
        test.to_csv(f"{data_folder}/test.csv", index=False, sep="\t")

    if language == "multilingual":
        ALFFA_LANGUAGES = ["amharic", "wolof"]
        df_alffa = pd.DataFrame()
        for lang in ALFFA_LANGUAGES:
            data_folder2 = (
                data_folder + f"/ALFFA_PUBLIC/ASR/{lang.upper()}/data"
            )
            df_l = alffa_public_prepare(lang, data_folder2)
            df_l["wav"] = df_l["wav"].map(
                lambda x: f"ALFFA_PUBLIC/ASR/{lang.upper()}/data/"
                + x.replace(f"{data_folder}/", "")
            )
            df_alffa = pd.concat([df_alffa, df_l], ignore_index=True)
        df_sw = swahili_prepare(data_folder)

        train_darija = pd.read_csv(
            f"{data_folder}/DVOICE/darija/texts/train.csv", sep="\t"
        )
        dev_darija = pd.read_csv(
            f"{data_folder}/DVOICE/darija/texts/dev.csv", sep="\t"
        )
        test_darija = pd.read_csv(
            f"{data_folder}/DVOICE/darija/texts/test.csv", sep="\t"
        )
        df_dar = pd.concat(
            [train_darija, dev_darija, test_darija], ignore_index=True
        )
        df_dar["wav"] = df_dar["wav"].map(lambda x: "DVOICE/darija/wavs/" + x)
        df = pd.concat([df_alffa, df_sw, df_dar], ignore_index=True)
        train, dev, test = train_validate_test_split(df)
        train.to_csv(f"{data_folder}/train.csv", index=False, sep="\t")
        dev.to_csv(f"{data_folder}/dev.csv", index=False, sep="\t")
        test.to_csv(f"{data_folder}/test.csv", index=False, sep="\t")

    # Setting output files
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

    # Additional checks to make sure the folder contains the data
    check_dvoice_folders(data_folder, language)

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


def alffa_public_prepare(language, data_folder):
    if language == "amharic":
        wavs = glob.glob(f"{data_folder}/*/*/*.wav")
        f_train = open(f"{data_folder}/train/text", "r", encoding="utf-8")
        f_test = open(f"{data_folder}/test/text", "r", encoding="utf-8")
        text = f_train.readlines() + f_test.readlines()
        random.shuffle(text)

    if language == "fongbe":
        wavs = glob.glob(f"{data_folder}/*/wav/*/*.wav")
        f_train = open(f"{data_folder}/train/text", "r", encoding="utf-8")
        f_test = open(f"{data_folder}/test/text", "r", encoding="utf-8")
        text = f_train.readlines() + f_test.readlines()
        random.shuffle(text)

    if language == "wolof":
        wavs_train = glob.glob(f"{data_folder}/train/*/*.wav")
        wavs_dev = glob.glob(f"{data_folder}/dev/wav/*/*.wav")
        wavs_test = glob.glob(f"{data_folder}/test/wav/*/*.wav")
        wavs = wavs_train + wavs_dev + wavs_test
        f_train = open(f"{data_folder}/train/text", "r", encoding="utf-8")
        f_test = open(f"{data_folder}/test/text", "r", encoding="utf-8")
        f_dev = open(f"{data_folder}/dev/text", "r", encoding="utf-8")
        text = f_train.readlines() + f_dev.readlines() + f_test.readlines()
        random.shuffle(text)

    data = []
    for i in tqdm(range(len(text))):
        text[i] = text[i].replace("   ", " ")
        text[i] = text[i].replace("  ", " ")
        text[i] = text[i].split(" ")
        file_name = text[i][0]
        words = " ".join(text[i][1:])
        for j in range(len(wavs)):
            if wavs[j].split("/")[-1] == file_name + ".wav":
                wav = wavs[j]
                info = read_audio_info(wav)
                duration = info.num_frames / info.sample_rate
                dic = {
                    "wav": wavs[j].replace(data_folder + "/", ""),
                    "words": str(words).replace("\n", ""),
                    "duration": duration,
                }
                data.append(dic)
                break

    random.shuffle(data)
    df = pd.DataFrame(data)
    return df


def swahili_prepare(data_folder):
    wavs_alffa = glob.glob(
        f"{data_folder}/ALFFA_PUBLIC/ASR/SWAHILI/data/*/*/*/*"
    )
    train_dvoice = pd.read_csv(
        f"{data_folder}/DVOICE/swahili/texts/train.csv", sep="\t"
    )
    dev_dvoice = pd.read_csv(
        f"{data_folder}/DVOICE/swahili/texts/dev.csv", sep="\t"
    )
    test_dvoice = pd.read_csv(
        f"{data_folder}/DVOICE/swahili/texts/test.csv", sep="\t"
    )
    text_dvoice = pd.concat(
        [train_dvoice, dev_dvoice, test_dvoice], ignore_index=True
    )
    text_dvoice["wav"] = text_dvoice["wav"].map(
        lambda x: "DVOICE/swahili/wavs/" + x
    )

    f_train_alffa = open(
        f"{data_folder}/ALFFA_PUBLIC/ASR/SWAHILI/data/train/text",
        "r",
        encoding="utf-8",
    )
    f_test_alffa = open(
        f"{data_folder}/ALFFA_PUBLIC/ASR/SWAHILI/data/test/text",
        "r",
        encoding="utf-8",
    )
    train_alffa = f_train_alffa.readlines()
    test_alffa = f_test_alffa.readlines()
    text_alffa = train_alffa + test_alffa
    random.shuffle(text_alffa)

    data_alffa = []
    for i in tqdm(range(len(text_alffa))):
        if "\t" in text_alffa[i]:
            text_alffa[i] = text_alffa[i].split("\t")
            file_name = text_alffa[i][0]
            words = text_alffa[i][1]
        else:
            text_alffa[i] = text_alffa[i].split(" ")
            file_name = text_alffa[i][0]
            words = " ".join(text_alffa[i][1:])
        for j in range(len(wavs_alffa)):
            if wavs_alffa[j].split("/")[-1] == file_name + ".wav":
                wav = wavs_alffa[j]
                info = read_audio_info(wav)
                duration = info.num_frames / info.sample_rate
                dic = {
                    "wav": wavs_alffa[j].replace(data_folder + "/", ""),
                    "words": str(words).replace("\n", ""),
                    "duration": duration,
                }
                data_alffa.append(dic)
                break

    text_alffa = pd.DataFrame(data_alffa)

    df = pd.concat([text_dvoice, text_alffa], ignore_index=True)
    return df


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

    Arguments
    ---------
    save_csv_train : str
        Path to the train csv
    save_csv_dev : str
        Path to the dev csv
    save_csv_test : str
        Path to the test csv

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
    orig_csv_file,
    csv_file,
    data_folder,
    accented_letters=False,
    language="darija",
):
    """
    Creates the csv file given a list of wav files.

    Arguments
    ---------
    orig_csv_file : str
        Path to the DVoice csv file (standard file).
    csv_file : str
        Path to the new DVoice csv file.
    data_folder : str
        Path of the DVoice dataset.
    accented_letters : bool, optional
        Defines if accented letters will be kept as individual letters or
        transformed to the closest non-accented letters.
    language : str
        Language to prepare.
    """

    # Check if the given files exists
    if not os.path.isfile(orig_csv_file):
        msg = "\t%s doesn't exist, verify your dataset!" % (orig_csv_file)
        logger.info(msg)
        raise FileNotFoundError(msg)

    # We load and skip the header
    loaded_csv = open(orig_csv_file, "r", encoding="utf-8").readlines()[1:]
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
        ALFFA_LANGUAGES = ["amharic", "fongbe"]
        if language in ALFFA_LANGUAGES:
            mp3_path = line.split("\t")[0]
        elif (
            language == "multilingual"
            or language == "swahili"
            or language == "wolof"
        ):
            mp3_path = data_folder + "/" + line.split("\t")[0]
        else:
            mp3_path = data_folder + "/wavs/" + line.split("\t")[0]

        file_name = line.split("\t")[0]
        spk_id = line.split("\t")[0].replace(".wav", "")
        snt_id = os.path.basename(file_name)

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
                "ابتةثجحخدذرزسشصضطظعغفقكلمنهويءآأؤإئ"  # cspell:disable-line
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


def check_dvoice_folders(data_folder, language):
    """
    Check if the data folder actually contains the DVoice dataset.
    If not, raises an error.

    Arguments
    ---------
    data_folder : str
        Path to directory with data.
    language : str
        The language to check.

    Raises
    ------
    FileNotFoundError
        If data folder doesn't contain DVoice dataset.
    """

    ALFFA_LANGUAGES = ["amharic", "fongbe", "wolof"]
    if (
        language in ALFFA_LANGUAGES
        or language == "swahili"
        or language == "multilingual"
    ):
        files_str = "/"
    else:
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
