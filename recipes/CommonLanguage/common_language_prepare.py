"""
Data preparation of CommonLangauge dataset for LID.

Download: https://zenodo.org/record/5036977#.YNo1mHVKg5k

Author
------
Pavlo Ruban 2021
"""

import os
import csv
import logging
import torchaudio
from tqdm.contrib import tzip
from speechbrain.utils.data_utils import get_all_files

logger = logging.getLogger(__name__)


LANGUAGES = [
    "Arabic",
    "Basque",
    "Breton",
    "Catalan",
    "Chinese_China",
    "Chinese_Hongkong",
    "Chinese_Taiwan",
    "Chuvash",
    "Czech",
    "Dhivehi",
    "Dutch",
    "English",
    "Esperanto",
    "Estonian",
    "French",
    "Frisian",
    "Georgian",
    "German",
    "Greek",
    "Hakha_Chin",
    "Indonesian",
    "Interlingua",
    "Italian",
    "Japanese",
    "Kabyle",
    "Kinyarwanda",
    "Kyrgyz",
    "Latvian",
    "Maltese",
    "Mangolian",
    "Persian",
    "Polish",
    "Portuguese",
    "Romanian",
    "Romansh_Sursilvan",
    "Russian",
    "Sakha",
    "Slovenian",
    "Spanish",
    "Swedish",
    "Tamil",
    "Tatar",
    "Turkish",
    "Ukranian",
    "Welsh",
]


def prepare_common_language(data_folder, save_folder, skip_prep=False):
    """
    Prepares the csv files for the CommonLanguage dataset for LID.
    Download: https://drive.google.com/uc?id=1Vzgod6NEYO1oZoz_EcgpZkUO9ohQcO1F

    Arguments
    ---------
    data_folder : str
        Path to the folder where the CommonLanguage dataset for LID is stored.
        This path should include the multi: /datasets/CommonLanguage
    save_folder : str
        The directory where to store the csv files.
    max_duration : int, optional
        Max duration (in seconds) of training uterances.
    skip_prep: bool
        If True, skip data preparation.

    Example
    -------
    >>> from recipes.CommonLanguage.common_language_prepare import prepare_common_language
    >>> data_folder = '/datasets/CommonLanguage'
    >>> save_folder = 'exp/CommonLanguage_exp'
    >>> prepare_common_language(\
            data_folder,\
            save_folder,\
            skip_prep=False\
        )
    """

    if skip_prep:
        return

    # Setting the save folder
    os.makedirs(save_folder, exist_ok=True)

    # Setting ouput files
    save_csv_train = os.path.join(save_folder, "train.csv")
    save_csv_dev = os.path.join(save_folder, "dev.csv")
    save_csv_test = os.path.join(save_folder, "test.csv")

    # If csv already exists, we skip the data preparation
    if skip(save_csv_train, save_csv_dev, save_csv_test):
        csv_exists = " already exists, skipping data preparation!"
        msg = save_csv_train + csv_exists
        logger.info(msg)
        msg = save_csv_dev + csv_exists
        logger.info(msg)
        msg = save_csv_test + csv_exists
        logger.info(msg)

        return

    # Additional checks to make sure the data folder contains Common Language
    check_common_language_folder(data_folder)

    # Audio files extensions
    extension = [".wav"]

    # Create the signal list of train, dev, and test sets.
    data_split = create_sets(data_folder, extension)

    # Creating csv files for training, dev and test data
    create_csv(wav_list=data_split["train"], csv_file=save_csv_train)
    create_csv(wav_list=data_split["dev"], csv_file=save_csv_dev)
    create_csv(wav_list=data_split["test"], csv_file=save_csv_test)


def skip(save_csv_train, save_csv_dev, save_csv_test):
    """
    Detects if the CommonLanguage data preparation for LID has been already done.

    If the preparation has been done, we can skip it.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """

    # Checking folders and save options
    skip = (
        os.path.isfile(save_csv_train)
        and os.path.isfile(save_csv_dev)
        and os.path.isfile(save_csv_test)
    )

    return skip


def create_sets(data_folder, extension):
    """
    Creates lists for train, dev and test sets with data from the data_folder

    Arguments
    ---------
    data_folder : str
        Path of the CommonLanguage dataset.
    extension: list of file extentions
        List of strings with file extentions that correspond to the audio files
        in the CommonLanguage dataset

    Returns
    -------
    dictionary containing train, dev, and test splits.
    """

    # Datasets initialization
    datasets = {"train", "dev", "test"}
    data_split = {dataset: [] for dataset in datasets}

    # Get the list of languages from the dataset folder
    languages = [
        name
        for name in os.listdir(data_folder)
        if os.path.isdir(os.path.join(data_folder, name))
        and datasets.issubset(os.listdir(os.path.join(data_folder, name)))
    ]

    msg = f"{len(languages)} languages detected!"
    logger.info(msg)

    # Fill the train, dev and test datasets with audio filenames
    for language in languages:
        for dataset in datasets:
            curr_folder = os.path.join(data_folder, language, dataset)
            wav_list = get_all_files(curr_folder, match_and=extension)
            data_split[dataset].extend(wav_list)

    msg = "Data successfully split!"
    logger.info(msg)

    return data_split


def create_csv(wav_list, csv_file):
    """
    Creates the csv file given a list of wav files.

    Arguments
    ---------
    wav_list : list of str
        The list of wav files.
    csv_file : str
        The path of the output json file
    """

    # Adding some Prints
    msg = f"Creating csv lists in {csv_file} ..."
    logger.info(msg)

    csv_lines = []

    # Start processing lines
    total_duration = 0.0

    # Starting index
    idx = 0

    for wav_file in tzip(wav_list):
        wav_file = wav_file[0]

        path_parts = wav_file.split(os.path.sep)
        file_name, wav_format = os.path.splitext(path_parts[-1])

        # Peeking at the signal (to retrieve duration in seconds)
        if os.path.isfile(wav_file):
            info = torchaudio.info(wav_file)
        else:
            msg = "\tError loading: %s" % (str(len(file_name)))
            logger.info(msg)
            continue

        audio_duration = info.num_frames / info.sample_rate
        total_duration += audio_duration

        # Actual name of the language
        language = path_parts[-4]

        # Create a row with whole utterences
        csv_line = [
            idx,  # ID
            wav_file,  # File name
            wav_format,  # File format
            str(info.num_frames / info.sample_rate),  # Duration (sec)
            language,  # Language
        ]

        # Adding this line to the csv_lines list
        csv_lines.append(csv_line)

        # Increment index
        idx += 1

    # CSV column titles
    csv_header = ["ID", "wav", "wav_format", "duration", "language"]

    # Add titles to the list at indexx 0
    csv_lines.insert(0, csv_header)

    # Writing the csv lines
    with open(csv_file, mode="w", encoding="utf-8") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )

        for line in csv_lines:
            csv_writer.writerow(line)

    # Final prints
    msg = f"{csv_file} sucessfully created!"
    logger.info(msg)
    msg = f"Number of samples: {len(wav_list)}."
    logger.info(msg)
    msg = f"Total duration: {round(total_duration / 3600, 2)} hours."
    logger.info(msg)


def check_common_language_folder(data_folder):
    """
    Check if the data folder actually contains the CommonLanguage dataset.

    If not, raises an error.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If data folder doesn't contain at least two languages.
    """

    # Checking if at least two languages are present in the data
    if len(set(os.listdir(data_folder)) & set(LANGUAGES)) < 2:
        err_msg = f"{data_folder} must have at least two languages from CommonLanguage in it."
        raise FileNotFoundError(err_msg)
