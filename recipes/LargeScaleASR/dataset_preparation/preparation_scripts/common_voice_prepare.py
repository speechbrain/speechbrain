"""
Data preparation for the LargeScaleASR Set (done on CV 18.0)
Download: https://commonvoice.mozilla.org/en/datasets

The csvs are stored in the manifest folder while data is in data/commonvoice.

Author
------
Titouan Parcollet 2024
"""

import csv
import functools
import os
from dataclasses import dataclass

from nemo_text_processing.text_normalization.normalize import Normalizer

from speechbrain.dataio.dataio import read_audio_info
from speechbrain.utils.logger import get_logger
from speechbrain.utils.parallel import parallel_map
from speechbrain.utils.text_normalisation import TextNormaliser

normaliser = Normalizer(input_case="cased", lang="en")

logger = get_logger(__name__)

VERBOSE = False
SAMPLING_RATE = 16000
LOWER_DURATION_THRESHOLD_IN_S = 1.0
UPPER_DURATION_THRESHOLD_IN_S = 40
LOWER_WORDS_THRESHOLD = 3
MAX_NB_OF_SAMPLES_PER_SPK = 9000


@dataclass
class TheLoquaciousRow:
    ID: str
    duration: float
    start: float
    wav: str
    spk_id: str
    sex: str
    text: str


def prepare_common_voice(
    data_folder,
    huggingface_folder,
    train_tsv_file=None,
    dev_tsv_file=None,
    test_tsv_file=None,
):
    """
    Prepares the csv files for the LargeScaleASR Set dataset.
    Download: https://commonvoice.mozilla.org/en

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original Common Voice dataset is stored.
        This path should include the lang: /datasets/CommonVoice/<language>/
    huggingface_folder : str
        The directory of the HuggingFace LargeScaleASR Set.
    train_tsv_file : str, optional
        Path to the Train Common Voice .tsv file (cs)
    dev_tsv_file : str, optional
        Path to the Dev Common Voice .tsv file (cs)
    test_tsv_file : str, optional
        Path to the Test Common Voice .tsv file (cs)

    """

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
    manifest_folder = os.path.join(huggingface_folder, "manifests")
    wav_folder = os.path.join(
        huggingface_folder, os.path.join("data", "commonvoice18")
    )
    os.makedirs(wav_folder, exist_ok=True)

    # Setting output files
    save_csv_train = manifest_folder + "/commonvoice_train.csv"
    save_csv_dev = manifest_folder + "/commonvoice_dev.csv"
    save_csv_test = manifest_folder + "/commonvoice_test.csv"

    # Additional checks to make sure the data folder contains Common Voice
    check_commonvoice_folders(data_folder)

    # If csv already exists, we skip the data preparation
    if not os.path.isfile(save_csv_train):
        create_csv(
            train_tsv_file,
            save_csv_train,
            data_folder,
            wav_folder,
        )
    else:
        msg = "%s already exists, skipping data preparation!" % (save_csv_train)
        logger.info(msg)

    if not os.path.isfile(save_csv_dev):
        create_csv(
            dev_tsv_file,
            save_csv_dev,
            data_folder,
            wav_folder,
        )
    else:
        msg = "%s already exists, skipping data preparation!" % (save_csv_dev)
        logger.info(msg)

    if not os.path.isfile(save_csv_test):
        create_csv(
            test_tsv_file,
            save_csv_test,
            data_folder,
            wav_folder,
        )
    else:
        msg = "%s already exists, skipping data preparation!" % (save_csv_test)
        logger.info(msg)


def process_line(line, data_folder, save_folder, header_map, text_norm):
    """Process a line of CommonVoice tsv file.

    Arguments
    ---------
    line : str
        A line of the CommonVoice tsv file.
    data_folder : str
        Path to the CommonVoice dataset.
    save_folder : str
        Where the wav files will be stored.
    header_map : Dict[str, int]
        Map from column name to column indices.
    text_norm : speechbrain.utils.text_normalisation.TextNormaliser

    Returns
    -------
    TheLoquaciousRow
        A dataclass containing the information about the line.
    """

    columns = line.strip().split("\t")
    spk_id = columns[header_map["client_id"]]
    audio_path_filename = columns[header_map["path"]]
    words = columns[header_map["sentence"]]
    sex = columns[header_map["gender"]]
    start = -1

    # !! Language specific cleaning !!
    words = normaliser.normalize(words)
    words = text_norm.english_specific_preprocess(words)

    if words is None or len(words) < LOWER_WORDS_THRESHOLD:
        return None

    # Path is at indice 1 in Common Voice tsv files. And .mp3 files
    # are located in datasets/lang/clips/
    audio_path = data_folder + "/clips/" + audio_path_filename

    # Reading the signal (to retrieve duration in seconds)
    if os.path.isfile(audio_path):
        info = read_audio_info(audio_path)
    else:
        msg = "\tError loading: %s" % (str(len(audio_path)))
        logger.info(msg)
        return None

    duration = info.num_frames / info.sample_rate

    if duration < LOWER_DURATION_THRESHOLD_IN_S:
        return None
    elif duration > UPPER_DURATION_THRESHOLD_IN_S:
        return None

    audio_path = convert_to_wav_and_copy(audio_path, save_folder)

    file_name = audio_path.split(".")[-2].split("/")[-1]
    snt_id = file_name

    # Composition of the csv_line
    return TheLoquaciousRow(
        snt_id, duration, start, audio_path, spk_id, sex, words
    )


def create_csv(
    orig_tsv_file,
    csv_file,
    data_folder,
    wav_folder,
):
    """
    Creates the csv file given a list of wav files.

    Arguments
    ---------
    orig_tsv_file : str
        Path to the Common Voice tsv file (standard file).
    csv_file : str
        New csv file.
    data_folder : str
        Path of the CommonVoice dataset.
    wav_folder : str
        Path to the new folder to copy to wav to.

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

    # Getting the list of spk and their number of samples.
    csv_data_lines = remove_sentences_based_on_spk_count(csv_data_lines)

    text_norm = TextNormaliser()
    line_processor = functools.partial(
        process_line,
        data_folder=data_folder,
        save_folder=wav_folder,
        header_map=header_map,
        text_norm=text_norm,
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


def remove_sentences_based_on_spk_count(corpus):
    """
    This function removes samples from spk that have too many of them based on
    the MAX_NB_SAMPLES_PER_SPK variable. In average, a CV sentence is 4s long.

    Arguments
    ---------
    corpus: list
        List of string representing the full corpus to deal with.

    Returns
    -------
    list
        List of strings with speakers talking too much removed.
    """

    seen = {}
    removed_cpt = 0
    new_corpus = []

    for line in corpus:
        spk_id = line.split("\t")[0]
        if spk_id not in seen:
            seen[spk_id] = 0
        else:
            seen[spk_id] += 1

        if seen[spk_id] < MAX_NB_OF_SAMPLES_PER_SPK:
            new_corpus.append(line)
        else:
            removed_cpt += 1

    logger.info("Removed: " + str(removed_cpt) + " samples to balance spks.")

    return new_corpus


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


def convert_to_wav_and_copy(source_audio_file, dest_audio_path):
    """Convert an audio file to a wav file.

    Parameters
    ----------
    source_audio_file : str
        The path to the opus file to be converted.
    dest_audio_path : str
        The path of the folder where to store the converted audio file.

    Returns
    -------
    str
        The path to the converted wav file.

    Raises
    ------
    subprocess.CalledProcessError
        If the conversion process fails.
    """
    audio_wav_path = source_audio_file.replace(".mp3", ".wav")
    audio_wav_path = os.path.join(
        dest_audio_path, audio_wav_path.split("/")[-1]
    )

    if not os.path.isfile(audio_wav_path):
        os.system(
            f"ffmpeg -y -i {source_audio_file} -ac 1 -ar {SAMPLING_RATE} {audio_wav_path} > /dev/null 2>&1"
        )

    return audio_wav_path
