"""
Data preparation of Librispeech for the LargeScaleASR Set (only dev and test).

Download: http://www.openslr.org/12

Author
------
 * Titouan Parcollet, 2024
"""

import csv
import functools
import os
from dataclasses import dataclass

from speechbrain.dataio.dataio import read_audio_info
from speechbrain.utils.data_utils import get_all_files
from speechbrain.utils.logger import get_logger
from speechbrain.utils.parallel import parallel_map

logger = get_logger(__name__)
OPT_FILE = "opt_librispeech_prepare.pkl"
SAMPLING_RATE = 16000
OPEN_SLR_11_LINK = "http://www.openslr.org/resources/11/"
OPEN_SLR_11_NGRAM_MODELs = [
    "3-gram.arpa.gz",
    "3-gram.pruned.1e-7.arpa.gz",
    "3-gram.pruned.3e-7.arpa.gz",
    "4-gram.arpa.gz",
]


def prepare_librispeech(
    data_folder,
    huggingface_folder,
    tr_splits=[],
    dev_splits=[],
    te_splits=[],
    select_n_sentences=None,
    merge_lst=[],
    merge_name=None,
    create_lexicon=False,
):
    """
    This class prepares the csv files for the LibriSpeech dataset.
    Download link: http://www.openslr.org/12

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original LibriSpeech dataset is stored.
    huggingface_folder : str
        The directory where to store the csv files.
    tr_splits : list
        List of train splits to prepare from ['test-others','train-clean-100',
        'train-clean-360','train-other-500'].
    dev_splits : list
        List of dev splits to prepare from ['dev-clean','dev-others'].
    te_splits : list
        List of test splits to prepare from ['test-clean','test-others'].
    select_n_sentences : int
        Default : None
        If not None, only pick this many sentences.
    merge_lst : list
        List of librispeech splits (e.g, train-clean, train-clean-360,..) to
        merge in a single csv file.
    merge_name: str
        Name of the merged csv file.
    create_lexicon: bool
        If True, it outputs csv files containing mapping between grapheme
        to phonemes. Use it for training a G2P system.

    Returns
    -------
    None

    """

    data_folder = data_folder
    splits = tr_splits + dev_splits + te_splits
    save_folder = huggingface_folder
    select_n_sentences = select_n_sentences
    conf = {
        "select_n_sentences": select_n_sentences,
    }

    # Other variables
    # Saving folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Check if this phase is already done (if so, skip it)
    if skip(splits, save_folder, conf):
        logger.info("Skipping preparation, completed in previous run.")
        return
    else:
        logger.info("Data_preparation...")

    # Additional checks to make sure the data folder contains Librispeech
    check_librispeech_folders(data_folder, splits)

    manifest_folder = os.path.join(huggingface_folder, "manifests")
    wav_folder = os.path.join(
        huggingface_folder, os.path.join("data", "librispeech")
    )
    os.makedirs(wav_folder, exist_ok=True)

    # create csv files for each split
    all_texts = {}
    for split_index in range(len(splits)):
        split = splits[split_index]

        wav_lst = get_all_files(
            os.path.join(data_folder, split), match_and=[".flac"]
        )

        text_lst = get_all_files(
            os.path.join(data_folder, split), match_and=["trans.txt"]
        )

        text_dict = text_to_dict(text_lst)
        all_texts.update(text_dict)

        if select_n_sentences is not None:
            n_sentences = select_n_sentences[split_index]
        else:
            n_sentences = len(wav_lst)

        create_csv(
            manifest_folder, wav_folder, wav_lst, text_dict, split, n_sentences
        )


@dataclass
class TheLoquaciousRow:
    ID: str
    duration: float
    start: float
    wav: str
    spk_id: str
    sex: str
    text: str


def process_line(wav_file, text_dict, save_folder) -> TheLoquaciousRow:
    snt_id = wav_file.split("/")[-1].replace(".flac", "")
    spk_id = "-".join(snt_id.split("-")[0:2])
    wrds = text_dict[snt_id]
    wrds = " ".join(wrds.split("_"))

    info = read_audio_info(wav_file)
    duration = info.num_frames / info.sample_rate

    wav_file = convert_to_wav_and_copy(wav_file, save_folder)

    return TheLoquaciousRow(
        ID=snt_id,
        spk_id=spk_id,
        duration=duration,
        start=-1,
        wav=wav_file,
        sex=None,
        text=wrds,
    )


def create_csv(
    save_folder, wav_folder, wav_lst, text_dict, split, select_n_sentences
):
    """
    Create the dataset csv file given a list of wav files.

    Arguments
    ---------
    save_folder : str
        Location of the folder for storing the csv.
    wav_folder : str
        Path to the new folder to copy to wav to.
    wav_lst : list
        The list of wav files of a given data split.
    text_dict : list
        The dictionary containing the text of each sentence.
    split : str
        The name of the current data split.
    select_n_sentences : int, optional
        The number of sentences to select.

    Returns
    -------
    None
    """
    # Setting path for the csv file
    csv_file = os.path.join(save_folder, "libri_" + split + ".csv")
    if os.path.exists(csv_file):
        logger.info("Csv file %s already exists, not recreating." % csv_file)
        return

    # Preliminary prints
    msg = "Creating csv lists in  %s..." % (csv_file)
    logger.info(msg)

    csv_lines = [["ID", "duration", "start", "wav", "spk_id", "sex", "text"]]

    snt_cnt = 0
    line_processor = functools.partial(
        process_line, text_dict=text_dict, save_folder=wav_folder
    )
    # Processing all the wav files in wav_lst
    # FLAC metadata reading is already fast, so we set a high chunk size
    # to limit main thread CPU bottlenecks
    for row in parallel_map(line_processor, wav_lst, chunk_size=128):
        csv_line = [
            row.ID,
            str(row.duration),
            row.start,
            row.wav,
            row.spk_id,
            row.sex,
            row.text,
        ]

        # Appending current file to the csv_lines list
        csv_lines.append(csv_line)

        snt_cnt = snt_cnt + 1

        # parallel_map guarantees element ordering so we're OK
        if snt_cnt == select_n_sentences:
            break

    # Writing the csv_lines
    with open(csv_file, mode="w", newline="", encoding="utf-8") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )

        for line in csv_lines:
            csv_writer.writerow(line)

    # Final print
    msg = "%s successfully created!" % (csv_file)
    logger.info(msg)


def skip(splits, save_folder, conf):
    """
    Detect when the librispeech data prep can be skipped.

    Arguments
    ---------
    splits : list
        A list of the splits expected in the preparation.
    save_folder : str
        The location of the save directory
    conf : dict
        The configuration options to ensure they haven't changed.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """

    # Checking csv files
    skip = True

    for split in splits:
        if not os.path.isfile(os.path.join(save_folder, split + ".csv")):
            skip = False

    return skip


def text_to_dict(text_lst):
    """
    This converts lines of text into a dictionary-

    Arguments
    ---------
    text_lst : str
        Path to the file containing the librispeech text transcription.

    Returns
    -------
    dict
        The dictionary containing the text transcriptions for each sentence.

    """
    # Initialization of the text dictionary
    text_dict = {}
    # Reading all the transcription files is text_lst
    for file in text_lst:
        with open(file, "r", encoding="utf-8") as f:
            # Reading all line of the transcription file
            for line in f:
                line_lst = line.strip().split(" ")
                text_dict[line_lst[0]] = "_".join(line_lst[1:])
    return text_dict


def check_librispeech_folders(data_folder, splits):
    """
    Check if the data folder actually contains the LibriSpeech dataset.

    If it does not, an error is raised.

    Arguments
    ---------
    data_folder : str
        The path to the directory with the data.
    splits : list
        The portions of the data to check.

    Raises
    ------
    OSError
        If LibriSpeech is not found at the specified path.
    """
    # Checking if all the splits exist
    for split in splits:
        split_folder = os.path.join(data_folder, split)
        if not os.path.exists(split_folder):
            err_msg = (
                "the folder %s does not exist (it is expected in the "
                "Librispeech dataset)" % split_folder
            )
            raise OSError(err_msg)


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
    audio_wav_path = source_audio_file.replace(".flac", ".wav")
    audio_wav_path = os.path.join(
        dest_audio_path, audio_wav_path.split("/")[-1]
    )

    if not os.path.isfile(audio_wav_path):
        os.system(
            f"ffmpeg -y -i {source_audio_file} -ac 1 -ar {SAMPLING_RATE} {audio_wav_path} > /dev/null 2>&1"
        )

    return audio_wav_path
