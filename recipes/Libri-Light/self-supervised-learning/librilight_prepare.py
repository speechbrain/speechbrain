"""
Data preparation.

1. Download the Libri-Light dataset through the toolkit in the Libri-Light github repo
    "https://github.com/facebookresearch/libri-light/"

2. Use the data_preparation/cut_by_vad.py script of the Libri-Light repo to do the vad. For example,
    "python cut_by_vad.py --input_dir path_to_Libri-Light/small --output_dir Libri-Light_vad/small_vad --target_len_sec 20"

Author
------
 * Titouan Parcollet 2024
 * Shucong Zhang 2024
"""

import csv
import functools
import os
from dataclasses import dataclass

from speechbrain.dataio.dataio import merge_csvs, read_audio_info
from speechbrain.utils.data_utils import get_all_files
from speechbrain.utils.logger import get_logger
from speechbrain.utils.parallel import parallel_map

logger = get_logger(__name__)


def prepare_librilight(
    data_folder,
    dev_folder,
    save_folder,
    vad_splits=[],
    merge_lst=[],
    merge_name=None,
    skip_prep=False,
):
    """
    This class prepares the csv files for the LibriLight dataset.
    Please do the VAD first before preparing the csv files

    Arguments
    ---------
    data_folder : str
        Path to the folder where LibriLight dataset after VAD is stored.
    dev_folder: str
        Path to the folder which will be used to create the dev csv.
        LibriLight does not have a dev split, so please use something like
        "path_to_LibriSpeech/dev-clean"
    save_folder : str
        The directory where to store the csv files.
    vad_splits : list
        List of train splits. E.g, ['small_vad'] or ['small_vad', 'medium_vad']
        or ['small_vad', 'medium_vad', 'large_vad']. Please ensure to have
        the 'small_vad' or 'medium_vad' or 'large_vad' folders under the data_folder
    merge_lst : list
        List of train splits to merge in a single csv file.
    merge_name: str
        Name of the merged csv file.
    skip_prep: bool
        If True, data preparation is skipped.

    Returns
    -------
    None

    Example
    -------
    >>> data_folder = 'datasets/Libri-Light_VAD'
    >>> dev_folder = 'datasets/LibriSpeech/dev-clean'
    >>> vad_splits = ['small_vad']
    >>> save_folder = 'librilight_prepared'
    >>> prepare_librilight(data_folder, dev_folder, save_folder, vad_splits)
    """

    if skip_prep:
        return

    # Saving folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)

    # Check if this phase is already done (if so, skip it)
    if skip(vad_splits, save_folder):
        logger.info("Skipping preparation, completed in previous run.")
        return
    else:
        logger.info("Data_preparation...")
        logger.info(
            "If you are using the large split, the data preparation may take more than 30 minutes."
        )
        logger.info("Please be patient and do not kill the process.")

    # Additional checks to make sure the data folder contains LibriLight
    check_librilight_folders(data_folder, vad_splits)

    # create csv files for each split
    for split_index in range(len(vad_splits)):
        split = vad_splits[split_index]

        wav_lst = get_all_files(
            os.path.join(data_folder, split), match_and=[".flac"]
        )

        n_sentences = len(wav_lst)

        create_csv(save_folder, wav_lst, split, n_sentences)

    # Merging csv file if needed
    if merge_lst and merge_name is not None:
        merge_files = [split_libri + ".csv" for split_libri in merge_lst]
        merge_csvs(
            data_folder=save_folder, csv_lst=merge_files, merged_csv=merge_name
        )

    # create a dev csv file from the dev_folder
    dev_wav_lst = get_all_files(dev_folder, match_and=[".flac"])

    dev_n_sentences = len(dev_wav_lst)

    create_csv(save_folder, dev_wav_lst, "dev", dev_n_sentences)


@dataclass
class LLRow:
    """Dataclass for handling Libri-Light rows.

    Attributes
    ----------
    snt_id : str
    The segment ID.
    duration : float
    The duration of the segment.
    file_path : str
    The path to the audio file.
    """

    snt_id: str
    duration: float
    file_path: str


def process_line(wav_file) -> LLRow:
    snt_id = "".join(wav_file.split("/")[-3:]).replace(".flac", "")

    info = read_audio_info(wav_file)
    duration = info.num_frames / info.sample_rate

    return LLRow(
        snt_id=snt_id,
        duration=duration,
        file_path=wav_file,
    )


def create_csv(save_folder, wav_lst, split, select_n_sentences):
    """
    Create the dataset csv file given a list of wav files.

    Arguments
    ---------
    save_folder : str
        Location of the folder for storing the csv.
    wav_lst : list
        The list of wav files of a given data split.
    split : str
        The name of the current data split.
    select_n_sentences : int, optional
        The number of sentences to select.

    Returns
    -------
    None
    """
    # Setting path for the csv file
    csv_file = os.path.join(save_folder, split + ".csv")
    if os.path.exists(csv_file):
        logger.info("Csv file %s already exists, not recreating." % csv_file)
        return

    # Preliminary prints
    msg = "Creating csv lists in  %s..." % (csv_file)
    logger.info(msg)

    csv_lines = [["ID", "duration", "wav"]]

    snt_cnt = 0
    line_processor = functools.partial(process_line)
    # Processing all the wav files in wav_lst
    # FLAC metadata reading is already fast, so we set a high chunk size
    # to limit main thread CPU bottlenecks
    for row in parallel_map(line_processor, wav_lst, chunk_size=8192):
        csv_line = [row.snt_id, str(row.duration), row.file_path]

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


def skip(splits, save_folder):
    """
    Detect when the LibriLight data prep can be skipped.

    Arguments
    ---------
    splits : list
        A list of the splits expected in the preparation.
    save_folder : str
        The location of the save directory

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


def check_librilight_folders(data_folder, splits):
    """
    Check if the data folder actually contains the Libri-Light dataset.

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
        If Libri-Light is not found at the specified path.
    """
    # Checking if all the splits exist
    for split in splits:
        split_folder = os.path.join(data_folder, split)
        if not os.path.exists(split_folder):
            err_msg = (
                "the folder %s does not exist (it is expected in the "
                "Libri-Light dataset)" % split_folder
            )
            raise OSError(err_msg)
