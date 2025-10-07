"""
Data preparation for the CoVoST dataset. This is heavily inspired
by the CommonVoice data preparation.

GitHub: https://github.com/facebookresearch/covost
Download: https://commonvoice.mozilla.org/en/datasets

Author
------
 * Titouan Parcollet 2025
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

SAMPLING_RATE = 16000
VERBOSE = False


def prepare_covost(
    data_folder,
    save_folder,
    train_tsv_file,
    dev_tsv_file,
    test_tsv_file,
    src_language="en",
    tgt_language="de",
    skip_prep=False,
    convert_to_wav=True,
):
    """
    Prepares the csv files for the CoVoST dataset.
    GitHub: https://github.com/facebookresearch/covost
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
    src_language: str, (default 'en')
        Specify the source language for text normalization.
    tgt_language: str, (default 'de')
        Specify the target language for text normalization.
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
    >>> from recipes.CoVoST.covost_prepare import prepare_covost
    >>> data_folder = '/datasets/CommonVoice/en'
    >>> save_folder = 'exp/CommonVoice_exp'
    >>> train_tsv_file = '/datasets/CommonVoice/en/train.tsv'
    >>> dev_tsv_file = '/datasets/CommonVoice/en/dev.tsv'
    >>> test_tsv_file = '/datasets/CommonVoice/en/test.tsv'
    >>> prepare_common_voice( \
                 data_folder, \
                 save_folder, \
                 train_tsv_file, \
                 dev_tsv_file, \
                 test_tsv_file, \
                 )
    """

    if skip_prep:
        return

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
            src_language,
            tgt_language,
        )


def create_csv(
    convert_to_wav,
    orig_tsv_file,
    csv_file,
    data_folder,
    src_language="en",
    tgt_language="de",
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
        New csv file to be generated.
    data_folder : str
        Path of the CommonVoice dataset.
    src_language : str, (default 'en')
        Source language code, e.g. "en".
    tgt_language : str, (default 'de')
        Target language code, e.g. "en".
    """

    # Check if the given files exists
    if not os.path.isfile(orig_tsv_file):
        msg = "\t%s doesn't exist, verify your dataset!" % (orig_tsv_file)
        logger.info(msg)
        raise FileNotFoundError(msg)

    # We load and skip the header
    csv_lines = open(orig_tsv_file, encoding="utf-8").readlines()
    csv_data_lines = csv_lines[1:]
    nb_samples = len(csv_data_lines)

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
        src_language=src_language,
        tgt_language=tgt_language,
    )

    # Stream into a .tmp file, and rename it to the real path at the end.
    csv_file_tmp = csv_file + ".tmp"

    with open(csv_file_tmp, mode="w", newline="", encoding="utf-8") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )

        csv_writer.writerow(
            ["ID", "duration", "wav", "transcription", "translation"]
        )

        for row in parallel_map(line_processor, csv_data_lines):
            if row is None:
                continue

            total_duration += row.duration
            csv_writer.writerow(
                [
                    row.snt_id,
                    str(row.duration),
                    row.audio_path,
                    row.transcription,
                    row.translation,
                ]
            )

    os.replace(csv_file_tmp, csv_file)

    # Final prints
    msg = "%s successfully created!" % (csv_file)
    logger.info(msg)
    msg = "Number of samples: %s " % (str(len(csv_data_lines)))
    logger.info(msg)
    print(msg)
    msg = "Total duration: %s Hours" % (str(round(total_duration / 3600, 2)))
    print(msg)
    logger.info(msg)


@dataclass
class CoVoSTRow:
    snt_id: str
    duration: float
    audio_path: str
    transcription: str
    translation: str


def process_line(line, convert_to_wav, data_folder, src_language, tgt_language):
    """Process a line of CoVoST tsv file.

    Arguments
    ---------
    line : str
        A line of the CoVoST tsv file.
    convert_to_wav : bool
        If True, `.mp3` files are converted (duplicated) to uncompressed `.wav`.
        Uncompressed `wav`s can be much faster to decode than MP3, at the cost
        of much higher disk usage and bandwidth. This might be useful if you are
        CPU-limited in workers during training.
        This invokes the `ffmpeg` commandline, so ffmpeg must be installed.
    data_folder : str
        Path to the CommonVoice dataset.
    src_language : str
        Source language code, e.g. "en"
    tgt_language : str
        Target language code, e.g. "en"

    Returns
    -------
    CVRow
        A dataclass containing the information about the line.
    """

    columns = line.strip().split("\t")
    audio_path_filename = columns[0]
    transcription = str(columns[1])
    translation = columns[2]

    if src_language == "en":
        # Corrupted files in english.
        if audio_path_filename in [
            "common_voice_fr_19528232.mp3",
            "common_voice_fr_19528233.mp3",
            "common_voice_en_19817845.mp3",
            "common_voice_en_19504777.mp3",
        ]:
            return None

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
    # !! Language specific cleaning !!
    transcription = language_specific_preprocess(src_language, transcription)
    translation = language_specific_preprocess(tgt_language, translation)

    if transcription is None or translation is None:
        return None
    elif len(transcription.split(" ")) < 4 or len(translation.split(" ")) < 4:
        return None

    # Composition of the csv_line
    return CoVoSTRow(snt_id, duration, audio_path, transcription, translation)


def skip(save_csv_train, save_csv_dev, save_csv_test):
    """
    Detects if the CoVoST data preparation has been already done.
    If the preparation has been done, we can skip it.

    Parameters
    ----------
    save_csv_train : str
        Path to train csv file.

    save_csv_dev : str
        Path to dev csv file.
    save_csv_test : str
        Path to test csv file.

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


def language_specific_preprocess(language, sentence):
    """
    Preprocess text based on language. This must be extended with
    other languages if needed.

    Parameters
    ----------
    language : str
        The code of the language to use for normalisation. E.g. "en", "de".
    sentence : str
        The string to normalise.
    Returns
    -------
    str
        The normalised sentence. Returns None if it was not possible to
        normalise the sentence.

    """

    STOP_ACCENTED_CHAR_LANGUAGES = ["en", "de"]

    if language == "en":
        final_characters = set(" abcdefghijklmnopqrstuvwxyz1234567890-&'")
    if language == "de":
        final_characters = set(
            " abcdefghijklmnopqrstuvwxyz1234567890-&ÄäÖöÜüẞß'"
        )
    else:  # Default to english set.
        final_characters = set(" abcdefghijklmnopqrstuvwxyz1234567890-&'")

    if language in STOP_ACCENTED_CHAR_LANGUAGES:
        if language == "en":
            stop_characters = (
                "["
                "áÁàăâåäÄãÃāảạæćčČçÇðéÉèÈêěëęēəğíîÎïīịıłṃńňñóÓòôőõøØōŌœŒřšŠşșȘúÚûūụýžþ"
                # Suggests the sentence is not English but German.
                "öÖßüÜ"
                # All sorts of languages: Greek, Arabic...
                "\u0370-\u1aaf"
                # Chinese/Japanese/Korean.
                "\u4e00-\u9fff"
                # Technical symbols.
                "\u2190-\u23ff"
                # Symbols that could be pronounced in various ways.
                "]"
            )
        elif language == "de":
            stop_characters = (
                "["
                "áÁàăâåãÃāảạæćčČçÇðéÉèÈêěëęēəğíîÎïīịıłṃńňñóÓòôőõøØōŌœŒřšŠşșȘúÚûūụýžþ"
                # All sorts of languages: Greek, Arabic...
                "\u0370-\u1aaf"
                # Chinese/Japanese/Korean.
                "\u4e00-\u9fff"
                # Technical symbols.
                "\u2190-\u23ff"
                # Symbols that could be pronounced in various ways.
                "]"
            )
        else:
            stop_characters = (
                "["
                "áÁàăâåãÃāảạæćčČçÇðéÉèÈêěëęēəğíîÎïīịıłṃńňñóÓòôőõøØōŌœŒřšŠşșȘúÚûūụýžþ"
                # All sorts of languages: Greek, Arabic...
                "\u0370-\u1aaf"
                # Chinese/Japanese/Korean.
                "\u4e00-\u9fff"
                # Technical symbols.
                "\u2190-\u23ff"
                # Symbols that could be pronounced in various ways.
                "]"
            )

        if re.search(stop_characters, sentence) is not None:
            return None

    # These characters mark word boundaries.
    split_character_regex = '[ ",:;!?¡\\.…()\\-—–‑_“”„/«»]'

    # These could all be used as apostrophes in the middle of words.
    # If at the start or end of a word, they will be removed.
    apostrophes_or_quotes = "['`´ʻ‘’]"

    # Some punctuation that indicates a word boundary.
    words_split = re.split(split_character_regex, sentence)
    words_quotes = [
        # Use ' as apostrophe.
        # Remove apostrophes at the start and end of words (probably quotes).
        # Word-internal apostrophes, even where rotated, are retained.
        re.sub(apostrophes_or_quotes, "'", word).strip("'")
        for word in words_split
    ]

    # Processing that does not change the length.
    words_lower = [word.lower() for word in words_quotes]

    words_mapped = [
        # word.translate(character_mapping)
        word
        for word in words_lower
        # Previous processing may have reduced words to nothing.
        # Remove them.
        if word != ""
    ]

    # removing empty strings
    words_mapped = [x for x in words_mapped if x.strip()]
    result = " ".join(words_mapped).rstrip()

    character_set = set(result)

    if not character_set <= final_characters:
        return None
    else:
        return result


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
    if not os.path.isfile(audio_wav_path):
        if VERBOSE:
            os.system(
                f"ffmpeg -y -i {audio_mp3_path} -ac 1 -ar {SAMPLING_RATE} {audio_wav_path}"
            )
        else:
            os.system(
                f"ffmpeg -y -i {audio_mp3_path} -ac 1 -ar {SAMPLING_RATE} {audio_wav_path} > /dev/null 2>&1"
            )
    return audio_wav_path
