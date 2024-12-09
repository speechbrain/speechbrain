"""
Data preparation for libriheavy

Author
------
Titouan Parcollet 2024
Shucong Zhang 2024
"""

import csv
import functools
import gzip
import json
import os
import re
from dataclasses import dataclass

from speechbrain.utils.logger import get_logger
from speechbrain.utils.parallel import parallel_map

logger = get_logger(__name__)

SAMPLING_RATE = 16000
LOWER_DURATION_THRESHOLD_IN_S = 1.0  # Should not happen in that dataset
UPPER_DURATION_THRESHOLD_IN_S = 100  # Should not happen in that dataset
LOWER_WORDS_THRESHOLD = 3


@dataclass
class TheLibriheavyRow:
    ID: str
    duration: float
    start: float
    wav: str
    spk_id: str
    sex: str
    text: str


def prepare_libriheavy(
    data_folder,
    manifest_folder,
    save_folder,
    tr_splits=[],
    te_splits=[],
    skip_prep=False,
):
    """
    Prepares the csv files for the Libriheavy dataset.
    1. Please download the Libri-Light dataset.
    Download: https://github.com/facebookresearch/libri-light/tree/main/data_preparation

    2. Please download the manifests.
    Download: https://github.com/k2-fsa/libriheavy

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original Libri-Light dataset is stored.
        e.g. /my_path/to/libri-light
    manifest_folder : str
        Path to the folder where the Libriheavy manifest (jsonl.gz files) is stored.
        e.g. /my_path/to/libriheavy
    save_folder : str
        The directory where to store the csv files.
    tr_splits : list
        Train split to prepare from.
        ['small'] -> 0.5 hours data
        ['medium'] -> 5k hours data
        ['large'] -> 50k hours data
    te_splits : list
        List of test splits to prepare from ['test_clean','test_others',
        'test_clean_large','test_others_large'].
    skip_prep: bool
        If True, data preparation is skipped.

    Returns
    -------
    None

    """

    if skip_prep:
        return

    splits = tr_splits + ["dev"] + te_splits

    # Setting the save folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for split_index in range(len(splits)):
        split = splits[split_index]
        save_csv = save_folder + f"/{split}.csv"

        if os.path.isfile(save_csv):
            msg = "%s already exists, skipping data preparation!" % (save_csv)
            logger.info(msg)

        csv_corpus = extract_transcripts(
            manifest_folder + f"/libriheavy_cuts_{split}.jsonl.gz"
        )

        if "dev" in split or "test" in split:
            data_folder_for_split = data_folder + "/large"
        else:
            data_folder_for_split = data_folder + f"/{split}"

        create_csv(
            csv_corpus,
            save_csv,
            data_folder_for_split,
        )


def extract_transcripts(jsonl_gz_file_path):
    """Extract the json file into a list.

    Arguments
    ---------
    jsonl_gz_file_path : str
        Path to the jsonl_gz file to extract.

    Returns
    -------
    list
        A list containing SpeechBrain ready lines of data.
    """

    logger.info(
        f"Extracting transcriptions from {jsonl_gz_file_path} to a list. This step can be fairly long."
    )

    csv_corpus = []

    # Open the gzipped JSONL file and the CSV file
    with gzip.open(jsonl_gz_file_path, "rt") as jsonl_file:

        # Write the header to the CSV file
        header = "ID,wav,start,duration,text,spk_id"
        csv_corpus.append(header)

        # Initialize the progress bar
        for cpt, line in enumerate(jsonl_file):
            if (cpt + 1) % 1000000 == 0:
                logger.info(f"{cpt} samples have been loaded!")
            data = json.loads(line)
            snt_id = data["id"]
            wav = data["recording"]["id"]
            start = str(data["start"])
            duration = str(data["duration"])
            texts = data["supervisions"][0]["custom"]["texts"]
            spk_id = str(data["supervisions"][0]["speaker"])
            # Extract transcriptions
            text = texts[1]
            # Write the row to the CSV file
            csv_corpus.append(
                snt_id
                + ","
                + wav
                + ","
                + start
                + ","
                + duration
                + ","
                + text
                + ","
                + spk_id
            )

    return csv_corpus


def process_line(line, data_folder):
    """Process a line of Libryheavy csv list.

    Arguments
    ---------
    line : str
        A line of the Libriheavy csv list.
    data_folder : str
        Path to the Libri-Light dataset.

    Returns
    -------
    TheLibriheavyRow
        A dataclass containing the information about the line.
    """

    if len(line.split(",")) != 6:
        return None

    snt_id, wav, start, duration, text, spk_id = line.split(",")

    start = float(start)
    duration = float(duration)

    # Remove the large / small denomination as already given by user.
    wav = os.path.join(*wav.split("/")[1:])
    sex = None

    # Unicode Normalization
    words = unicode_normalisation(text)

    # !! Language specific cleaning !!
    words = english_specific_preprocess(words)

    if words is None or len(words) < LOWER_WORDS_THRESHOLD:
        return None

    audio_path = os.path.join(data_folder, wav) + ".flac"

    # Reading the signal (to retrieve duration in seconds)
    if not os.path.isfile(audio_path):
        msg = "\tError loading: %s" % (str(audio_path))
        logger.info(msg)
        return None

    if duration < LOWER_DURATION_THRESHOLD_IN_S:
        return None
    elif duration > UPPER_DURATION_THRESHOLD_IN_S:
        return None

    # Composition of the csv_line
    return TheLibriheavyRow(
        snt_id, duration, start, audio_path, spk_id, sex, words
    )


def create_csv(
    filtered_csv_corpus,
    csv_file,
    data_folder,
):
    """
    Creates the csv file given a list of wav files.

    Arguments
    ---------
    filtered_csv_corpus : list
        Pre filtered list containing each sample. Obtained with functions
        extract_transcripts()
    csv_file : str
        New csv file.
    data_folder : str
        Path of the Libri-Light dataset.

    """

    # We load and skip the header
    csv_lines = filtered_csv_corpus
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
        data_folder=data_folder,
    )

    # Stream into a .tmp file, and rename it to the real path at the end.
    csv_file_tmp = csv_file + ".tmp"

    with open(csv_file_tmp, mode="w", newline="", encoding="utf-8") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )

        csv_writer.writerow(
            ["ID", "duration", "start", "wav", "spk_id", "sex", "text"]
        )

        for row in parallel_map(line_processor, csv_data_lines):
            if row is None:
                continue

            total_duration += row.duration
            csv_writer.writerow(
                [
                    row.ID,
                    str(row.duration),
                    str(row.start),
                    row.wav,
                    row.spk_id,
                    row.sex,
                    row.text,
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


def unicode_normalisation(text):
    return str(text)


def english_specific_preprocess(sentence):
    """
    Preprocess English text from the CommonVoice dataset into space-separated
    words.
    This removes various punctuation and treats it as word boundaries.
    It normalises and retains various apostrophes (’‘´) between letters, but not
    other ones, which are probably quotation marks.
    It capitalises all text.
    It removes non-English characters and those with accents, on the basis that
    each of them indicates text that is likely to be pronounced differently by
    different native speakers of English.
    Sometimes this is because the word with the accent is foreign, but often the
    other words in the Sentence are also hard to pronounce.
    An extreme example is ö,ß, and ü, which are often whole German sentences.
    CommonVoice regularly has data added to it.
    This function may error out if new characters show up in the training, dev,
    or test sets.
    If this happens, add the case to test_common_voice_prepare.py, and fix this
    function in a backward-compatible manner.
    """

    # These characters mean we should discard the sentence, because the
    # pronunciation will be too uncertain.
    stop_characters = (
        "["
        "áÁàăâåäÄãÃāảạæćčČçÇðéÉèÈêěëęēəğíîÎïīịıłṃńňñóÓòôőõøØōŌœŒřšŠşșȘúÚûūụýžþ"
        # Suggests the sentence is not English but German.
        "öÖßüÜ"
        # All sorts of languages: Greek, Arabic...
        "\u0370-\u1AAF"
        # Chinese/Japanese/Korean.
        "\u4E00-\u9FFF"
        # Technical symbols.
        "\u2190-\u23FF"
        # Symbols that could be pronounced in various ways.
        "\\[\\]€→=~%§_#"
        "]"
    )

    # These characters mark word boundaries.
    split_character_regex = '[ ",:;!?¡\\.…()\\-—–‑_“”„/«»]'

    # These could all be used as apostrophes in the middle of words.
    # If at the start or end of a word, they will be removed.
    apostrophes_or_quotes = "['`´ʻ‘’]"

    sentence_level_mapping = {"&": " and ", "+": " plus ", "ﬂ": "fl"}

    final_characters = set(" ABCDEFGHIJKLMNOPQRSTUVWXYZ'")

    if not re.search(stop_characters, sentence) is None:
        return None

    sentence_mapped = sentence
    if any((source in sentence) for source in sentence_level_mapping):
        for source, target in sentence_level_mapping.items():
            sentence_mapped = sentence_mapped.replace(source, target)

    # Some punctuation that indicates a word boundary.
    words_split = re.split(split_character_regex, sentence_mapped)
    words_quotes = [
        # Use ' as apostrophe.
        # Remove apostrophes at the start and end of words (probably quotes).
        # Word-internal apostrophes, even where rotated, are retained.
        re.sub(apostrophes_or_quotes, "'", word).strip("'")
        for word in words_split
    ]

    # Processing that does not change the length.
    words_upper = [word.upper() for word in words_quotes]

    words_mapped = [
        # word.translate(character_mapping)
        word
        for word in words_upper
        # Previous processing may have reduced words to nothing.
        # Remove them.
        if word != ""
    ]

    result = " ".join(words_mapped)
    character_set = set(result)
    assert character_set <= final_characters, (
        "Unprocessed characters",
        sentence,
        result,
        character_set - final_characters,
    )
    return result
