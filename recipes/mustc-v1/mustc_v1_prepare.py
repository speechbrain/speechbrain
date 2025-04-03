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
    source_font_case: str = "lc",
    target_font_case: str = "tc",
    is_accented_letters: bool = True,
    is_remove_punctuation: bool = False,
    is_remove_verbal: bool = False,
    target_language: str = "de",
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
    source_font_case : str, optional
        Can be tc, lc, uc for True Case, Low Case or Upper Case respectively.
    target_font_case : str, optional
        Can be tc, lc, uc for True Case, Low Case or Upper Case respectively.
    is_accented_letters : bool, optional
        Defines if accented letters will be kept as individual letters or
        transformed to the closest non-accented letters.
    is_remove_punctuation : bool, optional
        If set to True, the punctuation will be removed.
    is_remove_verbal : bool, optional
        If set to True, non-verbal tags will be removed e.g. ( Applause ).
    target_language : str, optional
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
                 target_language="de" \
                 )
    """

    if skip_prep or skip(save_folder, target_language):
        logger.info("Skipping data preparation.")
        return
    else:
        logger.info("Data preparation ...")

    mustc_v1_languages = ["de", "en", "fr", "es", "it", "nl", "pt", "ro", "ru"]

    if target_language not in mustc_v1_languages:
        msg = "target_language must be one of:" + str(mustc_v1_languages)
        raise ValueError(msg)

    # Setting the save folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Additional checks to make sure the data folder contains Common Voice
    _check_mustc_folders(data_folder)

    # Setting the official MuST-C file paths
    train_yaml = os.path.join(data_folder, "train/txt/train.yaml")
    train_source = os.path.join(data_folder, "train/txt/train.en")
    train_target = os.path.join(
        data_folder, "train/txt/train." + str(target_language)
    )
    train_wav = os.path.join(data_folder, "train/wav")

    dev_yaml = os.path.join(data_folder, "dev/txt/dev.yaml")
    dev_source = os.path.join(data_folder, "dev/txt/dev.en")
    dev_target = os.path.join(
        data_folder, "dev/txt/dev." + str(target_language)
    )
    dev_wav = os.path.join(data_folder, "dev/wav")

    test_he_yaml = os.path.join(data_folder, "tst-HE/txt/tst-HE.yaml")
    test_he_source = os.path.join(data_folder, "tst-HE/txt/tst-HE.en")
    test_he_target = os.path.join(
        data_folder, "tst-HE/txt/tst-HE." + str(target_language)
    )
    test_he_wav = os.path.join(data_folder, "tst-HE/wav")

    test_com_yaml = os.path.join(data_folder, "tst-COMMON/txt/tst-COMMON.yaml")
    test_com_source = os.path.join(data_folder, "tst-COMMON/txt/tst-COMMON.en")
    test_com_target = os.path.join(
        data_folder, "tst-COMMON/txt/tst-COMMON." + str(target_language)
    )
    test_com_wav = os.path.join(data_folder, "tst-COMMON/wav")

    # Path for preparated json files
    train = os.path.join(
        save_folder, "train_en-" + str(target_language) + ".json"
    )
    dev = os.path.join(save_folder, "dev_en-" + str(target_language) + ".json")
    test_he = os.path.join(
        save_folder, "test_he_en-" + str(target_language) + ".json"
    )
    test_com = os.path.join(
        save_folder, "test_com_en-" + str(target_language) + ".json"
    )

    datasets = [
        (train, train_yaml, train_source, train_target, train_wav),
        (dev, dev_yaml, dev_source, dev_target, dev_wav),
        (test_he, test_he_yaml, test_he_source, test_he_target, test_he_wav),
        (
            test_com,
            test_com_yaml,
            test_com_source,
            test_com_target,
            test_com_wav,
        ),
    ]

    # Creating json files based on the dataset
    for dataset in datasets:
        json_path, yaml_path, source_path, target_path, wav_path = dataset

        create_json(
            json_path,
            yaml_path,
            source_path,
            target_path,
            wav_path,
            data_folder,
            source_font_case,
            target_font_case,
            is_accented_letters,
            is_remove_punctuation,
            is_remove_verbal,
            target_language,
        )


def skip(save_folder: str, target_language: str):
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
    train = os.path.join(
        save_folder, "train_en-" + str(target_language) + ".json"
    )
    dev = os.path.join(save_folder, "dev_en-" + str(target_language) + ".json")
    test_he = os.path.join(
        save_folder, "test_he_en-" + str(target_language) + ".json"
    )
    test_com = os.path.join(
        save_folder, "test_com_en-" + str(target_language) + ".json"
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
    source_path: str,
    target_path: str,
    wav_path: str,
    data_folder: str,
    source_font_case: str,
    target_font_case: str,
    is_accented_letters: bool,
    is_remove_punctuation: bool,
    is_remove_verbal: bool,
    target_language: str,
):
    """
    Creates the a json file.

    See the prepare_mustc_v1 function for the arguments definition.
    """

    # We load all files and check that the number of samples correspond
    with open(yaml_path, "r", encoding="utf-8") as yaml_file, open(
        source_path, "r", encoding="utf-8"
    ) as source_file, open(target_path, "r", encoding="utf-8") as target_file:

        loaded_yaml = yaml_file.readlines()
        loaded_source = source_file.readlines()
        loaded_target = target_file.readlines()

        if not (len(loaded_yaml) == len(loaded_source) == len(loaded_target)):
            msg = "The number of lines in yaml, source and target files are different!"
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
            source_trs = loaded_source[cpt]
            target_trs = loaded_target[cpt]
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
            source_trs = source_trs.rstrip()
            target_trs = target_trs.rstrip()

            normalized_source = _normalize_text(
                text=source_trs,
                font_case=source_font_case,
                is_accented_letters=is_accented_letters,
                is_remove_verbal=is_remove_verbal,
                is_remove_punctuation=is_remove_punctuation,
            )

            normalized_target = _normalize_text(
                text=target_trs,
                font_case=target_font_case,
                is_accented_letters=is_accented_letters,
                is_remove_verbal=is_remove_verbal,
                is_remove_punctuation=is_remove_punctuation,
            )

            if (
                len(normalized_target.split(" ")) < 2
                or len(normalized_source.split(" ")) < 2
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
                "transcription": normalized_source,
                "translation": normalized_target,
                "transcription_and_translation": f"{normalized_source}\n{normalized_target}",
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


def _normalize_text(
    text: str,
    is_accented_letters: bool = False,
    is_remove_punctuation: bool = True,
    is_remove_verbal: bool = False,
    language: str = "en",
    font_case: str = "lc",
) -> str:
    """Language specific normalization for the given text"""
    if language in ["en", "fr", "it", "rw"]:
        text = re.sub("[^’'A-Za-z0-9À-ÖØ-öø-ÿЀ-ӿéæœâçèàûî]+", " ", text)

    if language == "fr":
        # Replace J'y D'hui etc by J_ D_hui
        text = text.replace("'", " ")
        text = text.replace("’", " ")

    elif language == "ar":
        HAMZA = "\u0621"
        ALEF_MADDA = "\u0622"
        ALEF_HAMZA_ABOVE = "\u0623"
        letters = (
            "ابتةثجحخدذرزسشصضطظعغفقكلمنهويءآأؤإئ"
            + HAMZA
            + ALEF_MADDA
            + ALEF_HAMZA_ABOVE
        )
        text = re.sub("[^" + letters + "]+", " ", text).upper()
    elif language == "ga-IE":
        # Irish lower() is complicated, but upper() is nondeterministic, so use lowercase
        def pfxuc(a):
            return len(a) >= 2 and a[0] in "tn" and a[1] in "AEIOUÁÉÍÓÚ"

        def galc(w):
            return w.lower() if not pfxuc(w) else w[0] + "-" + w[1:].lower()

        text = re.sub("[^-A-Za-z'ÁÉÍÓÚáéíóú]+", " ", text)
        text = " ".join(map(galc, text.split(" ")))

    # Remove accents if specified
    if not is_accented_letters:
        text = _strip_accents(text)
        text = text.replace("'", " ")
        text = text.replace("’", " ")

    # Remove spaces at the beginning and the end of the sentence
    text = text.lstrip().rstrip()

    # Replace contraction with space
    text = text.replace("'", " '")

    # Remove verbal
    text = re.sub(r"\([^()]*\)", "", text)

    # Normalize punctuations
    if is_remove_punctuation:
        text = _normalize_punctuation(
            text=text,
            font_case=font_case,
            is_remove_punctuation=is_remove_punctuation,
        )
    text = text.strip()

    # Remove multiple spaces
    text = re.sub(" +", " ", text)

    return text


def _normalize_punctuation(
    text: str, font_case: str = "lc", is_remove_punctuation: bool = True,
) -> str:
    if font_case == "tc":
        return text
    elif font_case == "lc":
        text = text.lower()
    elif font_case == "uc":
        text = text.upper()
    else:
        raise ValueError(
            f"font case must be lc/tc/uc, please check the value of font case"
        )

    if is_remove_punctuation:
        text = text.translate(str.maketrans("", "", string.punctuation))

    return text


def _strip_accents(text: str) -> str:
    text = (
        unicodedata.normalize("NFD", text)
        .encode("ascii", "ignore")
        .decode("utf-8")
    )

    return str(text)


def _check_mustc_folders(data_folder: str):
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
