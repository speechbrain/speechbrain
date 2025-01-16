"""
Data preparation for The LargeScaleASR Set (done on libriheavy)

Download: https://github.com/k2-fsa/libriheavy

The csvs are stored in the manifest folder while data is in data/libriheavy.

Author
------
Titouan Parcollet 2024
"""

import csv
import functools
import gzip
import json
import os
import shutil
from dataclasses import dataclass

import numpy as np

from speechbrain.utils.logger import get_logger
from speechbrain.utils.parallel import parallel_map
from speechbrain.utils.text_normalisation import TextNormaliser

logger = get_logger(__name__)

SAMPLING_RATE = 16000
LOWER_DURATION_THRESHOLD_IN_S = 1.0  # Should not happen in that dataset
UPPER_DURATION_THRESHOLD_IN_S = 40  # Should not happen in that dataset
LOWER_WORDS_THRESHOLD = 3


@dataclass
class TheLoquaciousRow:
    ID: str
    duration: float
    start: float
    wav: str
    spk_id: str
    sex: str
    text: str


def prepare_libriheavy(
    data_folder,
    huggingface_folder,
    train_jsonl_gz_file,
    dev_jsonl_gz_file,
    test_jsonl_gz_file,
    hours_for_training,
):
    """
    Prepares the csv files for the LibriHeavy dataset.
    Download: https://github.com/k2-fsa/libriheavy

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original Libri-Light dataset is stored.
        e.g. /my_path/to/libri-light/{large/medium/small}
    huggingface_folder : str
        The directory of the HuggingFace LargeScaleASR Set.
    train_jsonl_gz_file : str
        Path to the train .jsonl.gz file from Libriheavy.
    dev_jsonl_gz_file : str
        Path to the dev .jsonl.gz file from Libriheavy.
    test_jsonl_gz_file : str
        Path to the test .jsonl.gz file from Libriheavy.
    hours_for_training : int
        Set the number of hours to take from libriheavy. Samples are sampled
        randomly until the limit is reached.

    Returns
    -------
    None

    """

    # Setting the save folder
    manifest_folder = os.path.join(huggingface_folder, "manifests")
    wav_folder = os.path.join(
        huggingface_folder, os.path.join("data", "libriheavy")
    )
    os.makedirs(wav_folder, exist_ok=True)

    # Setting output files
    save_csv_train = manifest_folder + "/libriheavy_train.csv"

    # If csv already exists, we skip the data preparation
    if os.path.isfile(save_csv_train):
        msg = "%s already exists, skipping data preparation!" % (save_csv_train)
        logger.info(msg)
        return

    train_csv_corpus = extract_transcripts(train_jsonl_gz_file)

    filtered_csv_corpus = select_n_hours(train_csv_corpus, hours_for_training)

    create_csv(
        filtered_csv_corpus,
        save_csv_train,
        data_folder,
        wav_folder,
    )


def select_n_hours(train_csv_corpus, hours_for_training):
    """Randomly samples lines of a given list and return a new list whose duration column sums to 'hours_for_training'.

    Arguments
    ---------
    train_csv_corpus : list
        List of strings obtained with 'extract_transcripts(...)'
    hours_for_training : int
        Set the number of hours to take from libriheavy. Samples are sampled
        randomly until the limit is reached.

    Returns
    -------
    list
        A list containing SpeechBrain ready lines of data.
    """

    logger.info(f"Shuffling and extracting {hours_for_training} hours.")

    rng = np.random.default_rng(666)

    nb_of_samples = len(train_csv_corpus)
    indices = np.arange(1, nb_of_samples)
    rng.shuffle(indices)

    filtered_corpus = []
    filtered_corpus.append(train_csv_corpus[0])
    current_total_duration = 0.0
    for i in range(nb_of_samples - 1):

        line = train_csv_corpus[indices[i]]
        if len(line.split(",")) != 6:
            logger.info("Removed sentence: " + line)
            continue
        current_total_duration += float(line.split(",")[3]) / 3600
        if current_total_duration >= hours_for_training:
            break

        filtered_corpus.append(line)

    logger.info(
        f"Total amount of training: {round(current_total_duration, 1)} h"
    )

    return filtered_corpus


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
        f"Extracting transcriptions from {jsonl_gz_file_path} to a list. This step can be fairly long. The large set has 11M samples."
    )

    csv_corpus = []

    # Open the gzipped JSONL file and the CSV file
    with gzip.open(jsonl_gz_file_path, "rt") as jsonl_file:

        # Write the header to the CSV file
        header = "id,wav,start,duration,text,spk_id"
        csv_corpus.append(header)

        # Initialize the progress bar
        for cpt, line in enumerate(jsonl_file):
            if (cpt + 1) % 1000000 == 0:
                logger.info(f"{cpt} samples have been loaded!")
            data = json.loads(line)
            id = data["id"]
            wav = data["supervisions"][0]["recording_id"]
            start = str(data["start"])
            duration = str(data["duration"])
            texts = data["supervisions"][0]["custom"]["texts"]
            spk_id = str(data["supervisions"][0]["speaker"])
            # Extract transcriptions
            text = texts[1]
            # Write the row to the CSV file
            csv_corpus.append(
                id
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


def process_line(line, data_folder, save_folder, text_normaliser):
    """Process a line of Libryheavy csv list.

    Arguments
    ---------
    line : str
        A line of the Libriheavy csv list.
    data_folder : str
        Path to the Libri-Light dataset.
    save_folder : str
        Where the wav files will be stored.
    text_normaliser : speechbrain.utils.text_normalisation.TextNormaliser

    Returns
    -------
    TheLoquaciousRow
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
    words = text_normaliser.english_specific_preprocess(words)

    if words is None or len(words) < LOWER_WORDS_THRESHOLD:
        return None

    audio_path = os.path.join(data_folder, wav) + ".flac"

    # We create a new filename with both the directory and file name because
    # some audio files have the same name acros multiple directories.
    save_audio_path = (
        os.path.join(save_folder, "_".join(wav.split("/")[-3:])) + ".flac"
    )

    # We make the id even more uniq as it is used later on to split audio files
    snt_id = "_".join(snt_id.split("/")[-3:])

    # Checking the audio file exists.
    if not os.path.isfile(audio_path):
        msg = "\tError loading: %s" % (str(audio_path))
        logger.info(msg)
        return None

    if duration < LOWER_DURATION_THRESHOLD_IN_S:
        return None
    elif duration > UPPER_DURATION_THRESHOLD_IN_S:
        return None

    if not os.path.isfile(save_audio_path):
        if "frankenstein_00" in save_audio_path:
            print(audio_path + " " + save_audio_path)
        shutil.copyfile(audio_path, save_audio_path)

    # file_name = save_audio_path.split(".")[-2].split("/")[-1]

    # Composition of the csv_line
    return TheLoquaciousRow(
        snt_id, duration, start, save_audio_path, spk_id, sex, words
    )


def create_csv(
    filtered_csv_corpus,
    csv_file,
    data_folder,
    wav_folder,
):
    """
    Creates the csv file given a list of wav files.

    Arguments
    ---------
    filtered_csv_corpus : list
        Pre filtered list containing each sample. Obtained with two functions:
        extract_transcripts() and select_n_hours().
    csv_file : str
        New csv file.
    data_folder : str
        Path of the Libri-Light dataset.
    wav_folder : str
        Path to the new folder to copy to wav to.

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

    text_norm = TextNormaliser()
    line_processor = functools.partial(
        process_line,
        data_folder=data_folder,
        save_folder=wav_folder,
        text_normaliser=text_norm,
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
