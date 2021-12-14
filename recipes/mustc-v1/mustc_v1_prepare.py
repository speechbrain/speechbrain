"""
Data preparation.

Download: https://ict.fbk.eu/must-c-releases/

Author
------
Titouan Parcollet
YAO-FEI, CHENG
"""

import os
import json
import re
import string
import logging
import unicodedata

from tqdm.contrib import tzip

logger = logging.getLogger(__name__)


def prepare_mustc_v1(
    data_folder: str,
    save_folder: str,
    font_case: str = "lc",
    accented_letters: bool = True,
    punctuation: bool = False,
    non_verbal: bool = False,
    tgt_language: str = "de",
    skip_prep: bool = False,
):
    """
    Prepares the json files for the MuST-C (V1) Corpus.
    Download: https://ict.fbk.eu/must-c/

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original MuST-C dataset is stored.
        This path should include the lang: https://ict.fbk.eu/must-c-releases/
    save_folder : str
        The directory where to store the json files.
    font_case : str, optional
        Can be tc, lc, uc for True Case, Low Case or Upper Case respectively.
    accented_letters : bool, optional
        Defines if accented letters will be kept as individual letters or
        transformed to the closest non-accented letters.
    punctuation : bool, optional
        If set to True, the punctuation will be removed.
    non_verbal : bool, optional
        If set to True, non-verbal tags will be removed e.g. ( Applause ).
    tgt_language : str, optional
        Can be "de", "en", "fr", "es", "it", "nl", "pt", "ro" or "ru".
    skip_prep: bool
        If True, skip data preparation.

    Example
    -------
    >>> from recipes.mustc-v1.mustc_v1_prepare import prepare_mustc_v1
    >>> data_folder = '/datasets/mustc-v1/en-de/data'
    >>> save_folder = 'exp/mustc'
    >>> prepare_mustc_v1( \
                 data_folder, \
                 save_folder, \
                 tgt_language="de" \
                 )
    """

    if skip_prep or skip(save_folder, tgt_language):
        logger.info("Skipping data preparation.")
        return
    else:
        logger.info("Data preparation ...")

    mustc_v1_languages = ["de", "en", "fr", "es", "it", "nl", "pt", "ro", "ru"]

    if tgt_language not in mustc_v1_languages:
        msg = "tgt_language must be one of:" + str(mustc_v1_languages)
        raise ValueError(msg)

    # Setting the save folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Additional checks to make sure the data folder contains Common Voice
    check_mustc_folders(data_folder)

    # Setting the official MuST-C file paths
    train_yaml = os.path.join(data_folder, "train/txt/train.yaml")
    train_src = os.path.join(data_folder, "train/txt/train.en")
    train_tgt = os.path.join(
        data_folder, "train/txt/train." + str(tgt_language)
    )
    train_wav = os.path.join(data_folder, "train/wav")

    dev_yaml = os.path.join(data_folder, "dev/txt/dev.yaml")
    dev_src = os.path.join(data_folder, "dev/txt/dev.en")
    dev_tgt = os.path.join(data_folder, "dev/txt/dev." + str(tgt_language))
    dev_wav = os.path.join(data_folder, "dev/wav")

    test_he_yaml = os.path.join(data_folder, "tst-HE/txt/tst-HE.yaml")
    test_he_src = os.path.join(data_folder, "tst-HE/txt/tst-HE.en")
    test_he_tgt = os.path.join(
        data_folder, "tst-HE/txt/tst-HE." + str(tgt_language)
    )
    test_he_wav = os.path.join(data_folder, "tst-HE/wav")

    test_com_yaml = os.path.join(data_folder, "tst-COMMON/txt/tst-COMMON.yaml")
    test_com_src = os.path.join(data_folder, "tst-COMMON/txt/tst-COMMON.en")
    test_com_tgt = os.path.join(
        data_folder, "tst-COMMON/txt/tst-COMMON." + str(tgt_language)
    )
    test_com_wav = os.path.join(data_folder, "tst-COMMON/wav")

    # Path for preparated json files
    train = os.path.join(save_folder, "train_en-" + str(tgt_language) + ".json")
    dev = os.path.join(save_folder, "dev_en-" + str(tgt_language) + ".json")
    test_he = os.path.join(
        save_folder, "test_he_en-" + str(tgt_language) + ".json"
    )
    test_com = os.path.join(
        save_folder, "test_com_en-" + str(tgt_language) + ".json"
    )

    datasets = [
        (train, train_yaml, train_src, train_tgt, train_wav),
        (dev, dev_yaml, dev_src, dev_tgt, dev_wav),
        (test_he, test_he_yaml, test_he_src, test_he_tgt, test_he_wav),
        (test_com, test_com_yaml, test_com_src, test_com_tgt, test_com_wav),
    ]

    # Creating json files based on the dataset
    for dataset in datasets:
        json_path, yaml_path, src_path, tgt_path, wav_path = dataset

        create_json(
            json_path,
            yaml_path,
            src_path,
            tgt_path,
            wav_path,
            data_folder,
            font_case,
            accented_letters,
            punctuation,
            non_verbal,
            tgt_language,
        )


def skip(save_folder: str, tgt_language: str):
    """
    Detects if the MuST-C data preparation has been already done.

    If the preparation has been done, we can skip it.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """

    # File that should exists if already created
    train = os.path.join(save_folder, "train_en-" + str(tgt_language) + ".json")
    dev = os.path.join(save_folder, "dev_en-" + str(tgt_language) + ".json")
    test_he = os.path.join(
        save_folder, "test_he_en-" + str(tgt_language) + ".json"
    )
    test_com = os.path.join(
        save_folder, "test_com_en-" + str(tgt_language) + ".json"
    )

    skip = False

    if (
        os.path.isfile(train)
        and os.path.isfile(dev)
        and os.path.isfile(test_he)
        and os.path.isfile(test_com)
    ):
        skip = True

    return skip


def create_json(
    json_path: str,
    yaml_path: str,
    src_path: str,
    tgt_path: str,
    wav_path: str,
    data_folder: str,
    font_case: str,
    accented_letters: bool,
    punctuation: bool,
    non_verbal: bool,
    tgt_language: str,
):
    """
    Creates the a json file.

    See the prepare_mustc_v1 function for the arguments definition.
    """

    # We load all files and check that the number of samples correspond
    with open(yaml_path, "r", encoding="utf-8") as yaml_file, open(
        src_path, "r", encoding="utf-8"
    ) as src_file, open(tgt_path, "r", encoding="utf-8") as tgt_file:

        loaded_yaml = yaml_file.readlines()
        loaded_src = src_file.readlines()
        loaded_tgt = tgt_file.readlines()

        if not (len(loaded_yaml) == len(loaded_src) == len(loaded_tgt)):
            msg = (
                "The number of lines in yaml, src and tgt files are different!"
            )
            raise ValueError(msg)

        nb_samples = len(loaded_yaml)

        # Adding some Prints
        msg = f"Creating json lists in {json_path} ..."
        logger.info(msg)

        msg = f"Preparing JSON files for {nb_samples} samples ..."
        logger.info(msg)

        # Start processing lines
        total_duration = 0.0
        sample = 0
        cpt = 0

        json_dict = {}

        for line in tzip(loaded_yaml):
            line_yaml = line[0]
            src_trs = loaded_src[cpt]
            tgt_trs = loaded_tgt[cpt]
            cpt += 1

            yaml_split = line_yaml.split(" ")

            # Extract the needed fields
            wav = os.path.join(wav_path, yaml_split[-1].split("}")[0])
            spk_id = yaml_split[-2].split(",")[0]
            snt_id = sample
            duration = float(yaml_split[2].split(",")[0])
            offset = float(yaml_split[4].split(",")[0])
            total_duration += float(duration)

            # Getting transcripts and normalize according to parameters
            normalized_src = src_trs.rstrip()
            normalized_tgt = tgt_trs.rstrip()

            # 1. Case
            if font_case == "lc":
                normalized_src = normalized_src.lower()
                normalized_tgt = normalized_tgt.lower()
            elif font_case == "uc":
                normalized_src = normalized_src.upper()
                normalized_tgt = normalized_tgt.upper()

            # 2. Replace contraction with space
            normalized_src = normalized_src.replace("'", " '")
            normalized_tgt = normalized_tgt.replace("'", " '")

            # 3. Non verbal
            if not non_verbal:
                normalized_src = re.sub(r"\([^()]*\)", "", normalized_src)
                normalized_tgt = re.sub(r"\([^()]*\)", "", normalized_tgt)

            # 4. Punctuation
            if not punctuation:
                normalized_src = normalized_src.translate(
                    str.maketrans("", "", string.punctuation)
                )
                normalized_tgt = normalized_tgt.translate(
                    str.maketrans("", "", string.punctuation)
                )

            # 5. Accented letters
            if not accented_letters:
                normalized_src = strip_accents(normalized_src)
                normalized_tgt = strip_accents(normalized_tgt)

            # 6. We remove all examples that contains a single word
            if (
                len(normalized_tgt.split(" ")) < 2
                or len(normalized_src.split(" ")) < 2
            ):
                continue

            start = int(offset * 16000)
            stop = int((offset + duration) * 16000)

            json_dict[snt_id] = {
                "start": start,
                "stop": stop,
                "duration": duration,
                "wav": wav,
                "spk_id": spk_id,
                "transcription": normalized_src,
                "translation": normalized_tgt,
                "transcription_and_translation": f"{normalized_src}\n{normalized_tgt}",
            }

            sample += 1

    # Writing the dictionary to the json file
    with open(json_path, mode="w", encoding="utf8") as json_file:
        json.dump(json_dict, json_file, indent=2, ensure_ascii=False)

    # Final prints
    msg = f"{json_path} successfully created!"
    logger.info(msg)
    msg = f"Number of samples: {sample} "
    logger.info(msg)
    total_duration = round(total_duration / 3600, 2)
    msg = f"Total duration: {total_duration} Hours"
    logger.info(msg)


def check_mustc_folders(data_folder: str):
    """
    Check if the data folder actually contains the must-c dataset.

    If not, raises an error.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If data folder doesn't contain must-c
    """

    train = "/train"

    # Checking clips
    if not os.path.exists(data_folder + train):
        train_path = data_folder + train
        err_msg = (
            f"the folder {train_path} does not exist (it is expected in "
            "the must-c)"
        )
        raise FileNotFoundError(err_msg)


def strip_accents(text):
    text = (
        unicodedata.normalize("NFD", text)
        .encode("ascii", "ignore")
        .decode("utf-8")
    )

    return str(text)
