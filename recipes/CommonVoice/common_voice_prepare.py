"""
Data preparation.
Download: https://commonvoice.mozilla.org/en/datasets
Author
------
Titouan Parcollet 2021, 2022, 2024
Luca Della Libera 2022
Pooneh Mousavi 2022
Salima Mdhaffar 2023
Adel Moumen 2024
"""

import csv
import functools
import os
import re
import unicodedata
from dataclasses import dataclass

from speechbrain.dataio.dataio import read_audio_info
from speechbrain.utils.logger import get_logger
from speechbrain.utils.parallel import parallel_map

logger = get_logger(__name__)

VERBOSE = False
SAMPLING_RATE = 16_000


def prepare_common_voice(
    data_folder,
    save_folder,
    train_tsv_file=None,
    dev_tsv_file=None,
    test_tsv_file=None,
    accented_letters=False,
    language="en",
    skip_prep=False,
    convert_to_wav=False,
):
    """
    Prepares the csv files for the Mozilla Common Voice dataset.
    Download: https://commonvoice.mozilla.org/en

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original Common Voice dataset is stored.
        This path should include the lang: /datasets/CommonVoice/<language>/
    save_folder : str
        The directory where to store the csv files.
    train_tsv_file : str, optional
        Path to the Train Common Voice .tsv file (cs)
    dev_tsv_file : str, optional
        Path to the Dev Common Voice .tsv file (cs)
    test_tsv_file : str, optional
        Path to the Test Common Voice .tsv file (cs)
    accented_letters : bool, optional
        Defines if accented letters will be kept as individual letters or
        transformed to the closest non-accented letters.
    language: str
        Specify the language for text normalization.
    skip_prep: bool
        If True, skip data preparation.
    convert_to_wav: bool
        If True, `.mp3` files are converted (duplicated) to uncompressed `.wav`.
        Uncompressed `wav`s can be much faster to decode than MP3, at the cost
        of much higher disk usage and bandwidth. This might be useful if you are
        CPU-limited in workers during training.
        This invokes the `ffmpeg` commandline, so ffmpeg must be installed.

    Returns
    -------
    None

    Example
    -------
    >>> from recipes.CommonVoice.common_voice_prepare import prepare_common_voice
    >>> data_folder = '/datasets/CommonVoice/en'
    >>> save_folder = 'exp/CommonVoice_exp'
    >>> train_tsv_file = '/datasets/CommonVoice/en/train.tsv'
    >>> dev_tsv_file = '/datasets/CommonVoice/en/dev.tsv'
    >>> test_tsv_file = '/datasets/CommonVoice/en/test.tsv'
    >>> accented_letters = False
    >>> duration_threshold = 10
    >>> prepare_common_voice( \
                 data_folder, \
                 save_folder, \
                 train_tsv_file, \
                 dev_tsv_file, \
                 test_tsv_file, \
                 accented_letters, \
                 language="en" \
                 )
    """

    if skip_prep:
        return

    # If not specified point toward standard location w.r.t CommonVoice tree
    if train_tsv_file is None:
        train_tsv_file = data_folder + "/train.tsv"
    else:
        train_tsv_file = train_tsv_file

    if dev_tsv_file is None:
        dev_tsv_file = data_folder + "/dev.tsv"
    else:
        dev_tsv_file = dev_tsv_file

    if test_tsv_file is None:
        test_tsv_file = data_folder + "/test.tsv"
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
    check_commonvoice_folders(data_folder)
    # Creating csv files for {train, dev, test} data
    file_pairs = zip(
        [train_tsv_file, dev_tsv_file, test_tsv_file],
        [save_csv_train, save_csv_dev, save_csv_test],
    )
    for tsv_file, save_csv in file_pairs:
        create_csv(
            convert_to_wav,
            tsv_file,
            save_csv,
            data_folder,
            accented_letters,
            language,
        )


def skip(save_csv_train, save_csv_dev, save_csv_test):
    """
    Detects if the Common Voice data preparation has been already done.
    If the preparation has been done, we can skip it.

    Arguments
    ---------
    save_csv_train : str
        The train csv file
    save_csv_dev : str
        The dev csv file
    save_csv_test : str
        The test csv file

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
class CVRow:
    snt_id: str
    duration: float
    audio_path: str
    spk_id: str
    words: str


def process_line(
    line, convert_to_wav, data_folder, language, accented_letters, header_map
):
    """Process a line of CommonVoice tsv file.

    Arguments
    ---------
    line : str
        A line of the CommonVoice tsv file.
    convert_to_wav : bool
        If True, `.mp3` files are converted (duplicated) to uncompressed `.wav`.
        Uncompressed `wav`s can be much faster to decode than MP3, at the cost
        of much higher disk usage and bandwidth. This might be useful if you are
        CPU-limited in workers during training.
        This invokes the `ffmpeg` commandline, so ffmpeg must be installed.
    data_folder : str
        Path to the CommonVoice dataset.
    language : str
        Language code, e.g. "en"
    accented_letters : bool
        Defines if accented letters will be kept as individual letters or
        transformed to the closest non-accented letters.
    header_map : Dict[str, int]
        Map from column name to column indices

    Returns
    -------
    CVRow
        A dataclass containing the information about the line.
    """

    columns = line.strip().split("\t")
    spk_id = columns[header_map["client_id"]]
    audio_path_filename = columns[header_map["path"]]
    words = columns[header_map["sentence"]]

    # Path is at indice 1 in Common Voice tsv files. And .mp3 files
    # are located in datasets/lang/clips/
    audio_path = data_folder + "/clips/" + audio_path_filename

    if convert_to_wav:
        audio_path = convert_mp3_to_wav(audio_path)

    file_name = audio_path.split(".")[-2].split("/")[-1]
    snt_id = file_name

    # Reading the signal (to retrieve duration in seconds)
    if os.path.isfile(audio_path):
        info = read_audio_info(audio_path)
    else:
        msg = "\tError loading: %s" % (str(len(file_name)))
        logger.info(msg)
        return None

    duration = info.num_frames / info.sample_rate

    # Getting transcript

    # Unicode Normalization
    words = unicode_normalisation(words)

    # !! Language specific cleaning !!
    words = language_specific_preprocess(language, words)

    # Remove accents if specified
    if not accented_letters:
        words = strip_accents(words)
        words = words.replace("'", " ")
        words = words.replace("’", " ")

    # Remove multiple spaces
    words = re.sub(" +", " ", words)

    # Remove spaces at the beginning and the end of the sentence
    words = words.lstrip().rstrip()

    # Getting chars
    chars = words.replace(" ", "_")
    chars = " ".join([char for char in chars][:])

    # Remove too short sentences (or empty):
    if language in ["ja", "zh-CN"]:
        if len(chars) < 3:
            return None
    else:
        if len(words.split(" ")) < 3:
            return None

    # Composition of the csv_line
    return CVRow(snt_id, duration, audio_path, spk_id, words)


def create_csv(
    convert_to_wav,
    orig_tsv_file,
    csv_file,
    data_folder,
    accented_letters=False,
    language="en",
):
    """
    Creates the csv file given a list of wav files.

    Arguments
    ---------
    convert_to_wav : bool
        If True, `.mp3` files are converted (duplicated) to uncompressed `.wav`.
        Uncompressed `wav`s can be much faster to decode than MP3, at the cost
        of much higher disk usage and bandwidth. This might be useful if you are
        CPU-limited in workers during training.
        This invokes the `ffmpeg` commandline, so ffmpeg must be installed.
    orig_tsv_file : str
        Path to the Common Voice tsv file (standard file).
    csv_file : str
        New csv file.
    data_folder : str
        Path of the CommonVoice dataset.
    accented_letters : bool, optional
        Defines if accented letters will be kept as individual letters or
        transformed to the closest non-accented letters.
    language : str
        Language code, e.g. "en"
    """

    # Check if the given files exists
    if not os.path.isfile(orig_tsv_file):
        msg = "\t%s doesn't exist, verify your dataset!" % (orig_tsv_file)
        logger.info(msg)
        raise FileNotFoundError(msg)

    # We load and skip the header
    csv_lines = open(orig_tsv_file, "r", encoding="utf-8").readlines()
    header_line = csv_lines[0]
    csv_data_lines = csv_lines[1:]
    nb_samples = len(csv_data_lines)

    header_map = {
        column_name: index
        for index, column_name in enumerate(header_line.split("\t"))
    }

    msg = "Preparing CSV files for %s samples ..." % (str(nb_samples))
    logger.info(msg)

    # Adding some Prints
    msg = "Creating csv lists in %s ..." % (csv_file)
    logger.info(msg)

    # Process and write lines
    total_duration = 0.0

    line_processor = functools.partial(
        process_line,
        convert_to_wav=convert_to_wav,
        data_folder=data_folder,
        language=language,
        accented_letters=accented_letters,
        header_map=header_map,
    )

    # Stream into a .tmp file, and rename it to the real path at the end.
    csv_file_tmp = csv_file + ".tmp"

    with open(csv_file_tmp, mode="w", newline="", encoding="utf-8") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )

        csv_writer.writerow(["ID", "duration", "wav", "spk_id", "wrd"])

        for row in parallel_map(line_processor, csv_data_lines):
            if row is None:
                continue

            total_duration += row.duration
            csv_writer.writerow(
                [
                    row.snt_id,
                    str(row.duration),
                    row.audio_path,
                    row.spk_id,
                    row.words,
                ]
            )

    os.replace(csv_file_tmp, csv_file)

    # Final prints
    msg = "%s successfully created!" % (csv_file)
    logger.info(msg)
    msg = "Number of samples: %s " % (str(len(csv_data_lines)))
    logger.info(msg)
    msg = "Total duration: %s Hours" % (str(round(total_duration / 3600, 2)))
    logger.info(msg)


def convert_mp3_to_wav(audio_mp3_path):
    """Convert an mp3 file to a wav file.

    Parameters
    ----------
    audio_mp3_path : str
        The path to the opus file to be converted.

    Returns
    -------
    str
        The path to the converted wav file.

    Raises
    ------
    subprocess.CalledProcessError
        If the conversion process fails.
    """
    audio_wav_path = audio_mp3_path.replace(".mp3", ".wav")

    if VERBOSE:
        os.system(
            f"ffmpeg -y -i {audio_mp3_path} -ac 1 -ar {SAMPLING_RATE} {audio_wav_path}"
        )
    else:
        os.system(
            f"ffmpeg -y -i {audio_mp3_path} -ac 1 -ar {SAMPLING_RATE} {audio_wav_path} > /dev/null 2>&1"
        )
    return audio_wav_path


def language_specific_preprocess(language, words):
    # !! Language specific cleaning !!
    # Important: feel free to specify the text normalization
    # corresponding to your alphabet.

    if language in ["en", "fr", "it", "rw"]:
        words = re.sub(
            "[^’'A-Za-z0-9À-ÖØ-öø-ÿЀ-ӿéæœâçèàûî]+", " ", words
        ).upper()

    if language == "de":
        # this replacement helps preserve the case of ß
        # (and helps retain solitary occurrences of SS)
        # since python's upper() converts ß to SS.
        words = words.replace("ß", "0000ß0000")
        words = re.sub("[^’'A-Za-z0-9öÖäÄüÜß]+", " ", words).upper()
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
        words = words.upper()

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
        words = words.replace(
            "AUJOURD' HUI", "AUJOURD'HUI"  # cspell:disable-line
        )
    elif language == "ar":
        HAMZA = "\u0621"
        ALEF_MADDA = "\u0622"
        ALEF_HAMZA_ABOVE = "\u0623"
        letters = (
            "ابتةثجحخدذرزژشسصضطظعغفقكلمنهويىءآأؤإئ"  # cspell:disable-line
            + HAMZA
            + ALEF_MADDA
            + ALEF_HAMZA_ABOVE
        )
        words = re.sub("[^" + letters + " ]+", "", words).upper()
    elif language == "fa":
        HAMZA = "\u0621"
        ALEF_MADDA = "\u0622"
        ALEF_HAMZA_ABOVE = "\u0623"
        letters = (
            "ابپتةثجحخچدذرزژسشصضطظعغفقگکلمنهویىءآأؤإئ"  # cspell:disable-line
            + HAMZA
            + ALEF_MADDA
            + ALEF_HAMZA_ABOVE
        )
        words = re.sub("[^" + letters + " ]+", "", words).upper()
    elif language == "ga-IE":
        # Irish lower() is complicated, but upper() is nondeterministic, so use lowercase
        def pfxuc(a):
            return len(a) >= 2 and a[0] in "tn" and a[1] in "AEIOUÁÉÍÓÚ"

        def galc(w):
            return w.lower() if not pfxuc(w) else w[0] + "-" + w[1:].lower()

        words = re.sub("[^-A-Za-z'ÁÉÍÓÚáéíóú]+", " ", words)
        words = " ".join(map(galc, words.split(" ")))
    elif language == "es":
        # Fix the following error in dataset large:
        # KeyError: 'The item En noviembre lanzaron Queen Elizabeth , coproducida por Foreign Noi$e . requires replacements which were not supplied.'
        # cspell:ignore noviembre lanzaron coproducida
        words = words.replace("$", "s")
    return words


def check_commonvoice_folders(data_folder):
    """
    Check if the data folder actually contains the Common Voice dataset.
    If not, raises an error.

    Arguments
    ---------
    data_folder : str
        The folder containing the data to check

    Raises
    ------
    FileNotFoundError
        If data folder doesn't contain Common Voice dataset.
    """
    files_str = "/clips"
    # Checking clips
    if not os.path.exists(data_folder + files_str):
        err_msg = (
            "the folder %s does not exist (it is expected in "
            "the Common Voice dataset)" % (data_folder + files_str)
        )
        raise FileNotFoundError(err_msg)


def unicode_normalisation(text):
    return str(text)


def strip_accents(text):
    text = (
        unicodedata.normalize("NFD", text)
        .encode("ascii", "ignore")
        .decode("utf-8")
    )
    return str(text)
