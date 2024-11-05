"""
Data preparation for ASR with VoxPopuli.
Download: https://github.com/facebookresearch/voxpopuli
Author
------
Titouan Parcollet 2024
"""

import csv
import functools
import os
import re
from dataclasses import dataclass

from speechbrain.dataio.dataio import read_audio_info
from speechbrain.utils.logger import get_logger
from speechbrain.utils.parallel import parallel_map

logger = get_logger(__name__)


def prepare_voxpopuli(
    data_folder,
    save_folder,
    train_tsv_file=None,
    dev_tsv_file=None,
    test_tsv_file=None,
    skip_prep=False,
    language="en",
    remove_if_longer_than=100,
):
    """
    Prepares the csv files for the Vox Populi dataset.
    Download: https://github.com/facebookresearch/voxpopuli

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original Vox Populi dataset is stored.
        This path should include the transcribed_data folder.
    save_folder : str
        The directory where to store the csv files.
    train_tsv_file : str, optional
        Path to the Train Vox Populi .tsv file (cs)
    dev_tsv_file : str, optional
        Path to the Dev Vox Populi .tsv file (cs)
    test_tsv_file : str, optional
        Path to the Test Vox Populi .tsv file (cs)
    skip_prep: bool, optional
        If True, skip data preparation.
    language: str, optional
        The language of the voxpopuli dataset. This is used to apply language
        specific text normalisation.
    remove_if_longer_than: int, optional
        Some audio files in VoxPopuli can be very long (200+ seconds). This option
        removes them from the train set.

    Returns
    -------
    None

    Example
    -------
    >>> from recipes.VoxPopuli.ASR.voxpopuli_prepare import prepare_voxpopuli
    >>> data_folder = '/datasets/voxpopuli/en'
    >>> save_folder = 'exp/voxpopuli_exp'
    >>> train_tsv_file = '/datasets/voxpopuli/data/transcribed_data/en/asr_train.tsv'
    >>> dev_tsv_file = '/datasets/voxpopuli/data/transcribed_data/en/asr_dev.tsv'
    >>> test_tsv_file = '/datasets/voxpopuli/data/transcribed_data/en/test.tsv'
    >>> prepare_voxpopuli( \
                 data_folder, \
                 save_folder, \
                 )
    """

    if skip_prep:
        return

    # If not specified point toward standard location w.r.t VoxPopuli tree
    if train_tsv_file is None:
        train_tsv_file = data_folder + "/asr_train.tsv"
    else:
        train_tsv_file = train_tsv_file

    if dev_tsv_file is None:
        dev_tsv_file = data_folder + "/asr_dev.tsv"
    else:
        dev_tsv_file = dev_tsv_file

    if test_tsv_file is None:
        test_tsv_file = data_folder + "/asr_test.tsv"
    else:
        test_tsv_file = test_tsv_file

    # Setting the save folder
    os.makedirs(save_folder, exist_ok=True)

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

    # Additional checks to make sure the data folder contains Common Voice
    check_voxpopuli_folders(data_folder)
    # Creating csv files for {train, dev, test} data
    file_pairs = zip(
        [train_tsv_file, dev_tsv_file, test_tsv_file],
        [save_csv_train, save_csv_dev, save_csv_test],
    )
    for tsv_file, save_csv in file_pairs:
        create_csv(
            tsv_file, save_csv, data_folder, language, remove_if_longer_than
        )


def skip(save_csv_train, save_csv_dev, save_csv_test):
    """
    Detects if the VoxPopuli data preparation has been already done.
    If the preparation has been done, we can skip it.

    Arguments
    ---------
    save_csv_train : str
        Path to train manifest file
    save_csv_dev : str
        Path to dev manifest file
    save_csv_test : str
        Path to test manifest file

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


@dataclass
class VPRow:
    snt_id: str
    duration: float
    ogg_path: str
    spk_id: str
    words: str


def process_line(line, data_folder, language):
    """
    Processes each line of the CSV (most likely happening with multiple threads)

    Arguments
    ---------
    line : str
        Line of the csv file.
    data_folder : str
        Path of the Vox Populi dataset.
    language: str, optional
        The language of the voxpopuli dataset. This is used to apply language
        specific text normalisation.

    Returns
    -------
    VPRow
    """
    year_path = os.path.join(line[0:4], line.split("\t")[0])
    ogg_path = os.path.join(data_folder, year_path) + ".ogg"
    file_name = line.split("\t")[0]
    spk_id = line.split("\t")[3]
    snt_id = file_name

    # Reading the signal (to retrieve duration in seconds)
    if os.path.isfile(ogg_path):
        info = read_audio_info(ogg_path)
    else:
        msg = "\tError loading: %s" % (ogg_path)
        logger.info(msg)
        return None

    duration = info.num_frames / info.sample_rate

    # Getting transcript
    words = line.split("\t")[2]

    # Unicode Normalization
    words = unicode_normalisation(words)

    words = language_specific_preprocess(language, words)

    # Remove multiple spaces
    words = re.sub(" +", " ", words)

    # Remove spaces at the beginning and the end of the sentence
    words = words.lstrip().rstrip()

    if len(words.split(" ")) < 3:
        return None

    # Composition of the csv_line
    return VPRow(snt_id, duration, ogg_path, spk_id, words)


def create_csv(
    orig_tsv_file, csv_file, data_folder, language, remove_if_longer_than
):
    """
    Creates the csv file given a list of ogg files.

    Arguments
    ---------
    orig_tsv_file : str
        Path to the Vox Populi tsv file (standard file).
    csv_file : str
        Path to the csv file where data will be dumped.
    data_folder : str
        Path of the Vox Populi dataset.
    language: str, optional
        The language of the voxpopuli dataset. This is used to apply language
        specific text normalisation.
    remove_if_longer_than: int, optional
        Some audio files in VoxPopuli can be very long (200+ seconds). This option
        removes them from the train set. Information about the discarded data is given.
    """

    # Check if the given files exists
    if not os.path.isfile(orig_tsv_file):
        msg = "\t%s doesn't exist, verify your dataset!" % (orig_tsv_file)
        logger.info(msg)
        raise FileNotFoundError(msg)

    # We load and skip the header
    loaded_csv = open(orig_tsv_file, "r", encoding="utf-8").readlines()[1:]
    nb_samples = len(loaded_csv)

    msg = "Preparing CSV files for %s samples ..." % (str(nb_samples))
    logger.info(msg)

    # Adding some Prints
    msg = "Creating csv lists in %s ..." % (csv_file)
    logger.info(msg)

    # Process and write lines
    total_duration = 0.0
    skipped_duration = 0.0

    line_processor = functools.partial(
        process_line, language=language, data_folder=data_folder
    )

    # Stream into a .tmp file, and rename it to the real path at the end.
    csv_file_tmp = csv_file + ".tmp"

    with open(csv_file_tmp, mode="w", encoding="utf-8") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )

        csv_writer.writerow(["ID", "duration", "wav", "spk_id", "wrd"])

        for row in parallel_map(line_processor, loaded_csv):
            if row is None:
                continue

            if row.duration < remove_if_longer_than:
                total_duration += row.duration
            else:
                skipped_duration += row.duration
                continue

            csv_writer.writerow(
                [
                    row.snt_id,
                    str(row.duration),
                    row.ogg_path,
                    row.spk_id,
                    row.words,
                ]
            )

    os.replace(csv_file_tmp, csv_file)

    # Final prints
    msg = "%s successfully created!" % (csv_file)
    logger.info(msg)
    msg = "Number of samples: %s " % (str(len(loaded_csv)))
    logger.info(msg)
    msg = "Total duration: %s Hours" % (str(round(total_duration / 3600, 2)))
    logger.info(msg)
    msg = "Total skipped duration (too long segments): %s Hours" % (
        str(round(skipped_duration / 3600, 2))
    )
    logger.info(msg)


def check_voxpopuli_folders(data_folder):
    """
    Check if the data folder actually contains the voxpopuli dataset.
    If not, raises an error.

    Arguments
    ---------
    data_folder : str
        Path to data folder to check

    Raises
    ------
    FileNotFoundError
        If data folder doesn't contain Common Voice dataset.
    """
    files_str = "/2020"
    # Checking clips
    if not os.path.exists(data_folder + files_str):
        err_msg = (
            "the folder %s does not exist (it is expected in "
            "the Common Voice dataset)" % (data_folder + files_str)
        )
        raise FileNotFoundError(err_msg)


def unicode_normalisation(text):
    return str(text)


def language_specific_preprocess(language, words):
    """
    Format the input string according to some rules depending on the language.

    Arguments
    ---------
    language : str
        Corresponds to the two letters for language-specific sets
    words : str
        The string to be cleaned.

    Returns
    -------
    str
    """

    # !! Language specific cleaning !!
    # Important: feel free to specify the text normalization
    # corresponding to your alphabet.

    if language in ["en", "fr", "it"]:
        words = re.sub("[^’'A-Za-z0-9À-ÖØ-öø-ÿЀ-ӿéæœâçèàûî]+", " ", words)

    if language == "de":
        # this replacement helps preserve the case of ß
        # (and helps retain solitary occurrences of SS)
        # since python's upper() converts ß to SS.
        words = words.replace("ß", "0000ß0000")
        words = re.sub("[^’'A-Za-z0-9öÖäÄüÜß]+", " ", words)
        words = words.replace("'", " ")
        words = words.replace("’", " ")
        words = words.replace(
            "0000SS0000", "ß"
        )  # replace 0000SS0000 back to ß as its initial presence in the corpus

    elif language == "fr":  # SM
        words = re.sub("[^’'A-Za-z0-9À-ÖØ-öø-ÿЀ-ӿéæœâçèàûî]+", " ", words)
        words = words.replace("’", "'")
        words = words.replace("é", "é")
        words = words.replace("æ", "ae")
        words = words.replace("œ", "oe")
        words = words.replace("â", "â")
        words = words.replace("ç", "ç")
        words = words.replace("è", "è")
        words = words.replace("à", "à")
        words = words.replace("û", "û")
        words = words.replace("î", "î")
        words = words

        # Case of apostrophe collés
        words = words.replace("L'", "L' ")
        words = words.replace("L'  ", "L' ")
        words = words.replace("S'", "S' ")
        words = words.replace("S'  ", "S' ")
        words = words.replace("D'", "D' ")
        words = words.replace("D'  ", "D' ")
        words = words.replace("J'", "J' ")
        words = words.replace("J'  ", "J' ")
        words = words.replace("N'", "N' ")
        words = words.replace("N'  ", "N' ")
        words = words.replace("C'", "C' ")
        words = words.replace("C'  ", "C' ")
        words = words.replace("QU'", "QU' ")
        words = words.replace("QU'  ", "QU' ")
        words = words.replace("M'", "M' ")
        words = words.replace("M'  ", "M' ")

        # Case of apostrophe qui encadre quelques mots
        words = words.replace(" '", " ")
        words = words.replace("A'", "A")
        words = words.replace("B'", "B")
        words = words.replace("E'", "E")
        words = words.replace("F'", "F")
        words = words.replace("G'", "G")
        words = words.replace("K'", "K")
        words = words.replace("Q'", "Q")
        words = words.replace("V'", "V")
        words = words.replace("W'", "W")
        words = words.replace("Z'", "Z")
        words = words.replace("O'", "O")
        words = words.replace("X'", "X")
        words = words.replace("AUJOURD' HUI", "AUJOURD'HUI")

    return words
