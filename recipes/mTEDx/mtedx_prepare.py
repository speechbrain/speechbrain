"""
Data preparation.

Download: http://www.openslr.org/100

Author
------
Mohamed Anwar (Anwarvic) 2022
"""

import os
import json
import string
import logging

logger = logging.getLogger(__name__)


def prepare_mtedx(
    data_folder, save_folder, langs=[],
):
    """
    This function prepares the json files for the mTEDx dataset.

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original mTEDx dataset is stored.
    save_folder: str
        Path to the folder where the resulting files will be stored.
    langs: list(str)
        A list of language-codes to be prepared.

    Example
    -------
    >>> data_folder = 'datasets/mTEDx'
    >>> save_folder = 'mTEDx_prepared'
    >>> langs = ["fr", "es"]
    >>> prepare_mtedx(data_folder, save_folder, langs)
    """
    # Saving folder
    os.makedirs(save_folder, exist_ok=True)

    langs.sort()
    # checks for one json file that have all needed data for give langs.
    if skip(save_folder, langs):
        logger.info("Skipping preparation, completed in previous run.")
        return
    else:
        logger.info("Data_preparation...")

    # Additional checks to make sure the data folder contains mTEDx
    for lang in langs:
        for group in ["train", "valid", "test"]:
            check_mtedx_folders(data_folder, lang, group)

    # create json files for each group
    for group in ["test", "valid", "train"]:
        create_json(data_folder, save_folder, langs, group)


def create_json(data_folder, save_folder, langs, group):
    """
    Create the dataset json file given a list of wav files.

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original mTEDx dataset is stored.
    save_folder : str
        Location of the folder for storing the csv.
    wav_lst : list
        The list of wav files of a given data split.
    text_dict : list
        The dictionary containing the text of each sentence.
    split : str
        The name of the current data split.
    """
    # Setting path for the json file
    json_file = os.path.join(save_folder, f"{group}_{'_'.join(langs)}.json")

    # skip if the file already exists
    if os.path.exists(json_file):
        logger.info(f"{json_file} already exists. Skipping!!")
        return

    logger.info(f"Creating json file in {json_file}")
    group_data = {}
    for lang in langs:
        logger.info(f"Processing language: {lang}, group: {group}")
        with open(os.path.join(data_folder, lang, f"{group}.json")) as fin:
            group_data.extend(json.loads(fin))
    # write dict into json file
    with open(json_file, "w", encoding="utf8") as fout:
        fout.write(json.dumps(group_data, indent=4, ensure_ascii=False))
    logger.info(f"{json_file} successfully created!")


def skip(save_folder, langs):
    """
    Detect when the mTEDx data preparation can be skipped. The preparation
    can be skipped iff `train.json`, `valid.json` and `test.json` exist.

    Arguments
    ---------
    save_folder : str
        The location of the save directory.
    langs : list(str)
        A list of languages expected in the preparation.

    Returns
    -------
    bool
        If True, the preparation is skipped. Otherwise, it must be done.
    """
    langs = "_".join(langs)
    return (
        os.path.exists(os.path.join(save_folder, f"train_{langs}.json"))
        and os.path.exists(os.path.join(save_folder, f"valid_{langs}.json"))
        and os.path.exists(os.path.join(save_folder, f"test_{langs}.json"))
    )


def check_mtedx_folders(data_folder, lang, group):
    """
    Check if the data folder actually contains the mTEDx dataset.
    If it does not, an error is raised.

    Arguments
    ---------
    data_folder: str
        Path to the folder where the original mTEDx dataset is stored.
    lang: str
        A string describing the language code.
    group: str
        A string describing the data group, e.g "test".

    Raises
    ------
    ValueError
        If data in json file didn't match real data.
    """
    with open(os.path.join(data_folder, lang, f"{group}.json")) as fin:
        json_data = json.load(fin)
    audio_data = os.listdir(os.path.join(data_folder, lang, group))
    if len(json_data) != len(audio_data):
        raise ValueError(
            f"{lang}/{group} data doesn't match!! Actual audio data is"
            + f"{len(audio_data)}, while audio data written in json file is"
            + f"{len(json_data)}!"
        )
    return True


# helpful function
def remove_punctuations(text):
    """Removes punctuations from text"""
    PUNCS = string.punctuation + "،؟؛¿¡—÷×»«‹›"
    return text.translate(str.maketrans("", "", PUNCS)).strip()
