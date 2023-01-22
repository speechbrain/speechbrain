"""
Data preparation.

Download: http://www.openslr.org/12

Author
------
* Mirco Ravanelli, Ju-Chieh Chou, Loren Lugosch 2020
* Artem Ploujnikov 2022 (alignments)
"""

import json
import os
import csv
import random
from collections import Counter, namedtuple
from enum import Enum
import logging
import torchaudio
from speechbrain.utils.data_utils import download_file, get_all_files
from speechbrain.dataio.dataio import (
    load_pkl,
    save_pkl,
    merge_csvs,
    merge_jsons
)

logger = logging.getLogger(__name__)
OPT_FILE = "opt_librispeech_prepare.pkl"
SAMPLERATE = 16000


class LibriSpeechMode(Enum):
    BASIC = "basic"
    ALIGNMENT = "alignment"


def prepare_librispeech(
    data_folder,
    save_folder,
    tr_splits=[],
    dev_splits=[],
    te_splits=[],
    select_n_sentences=None,
    merge_lst=[],
    merge_name=None,
    create_lexicon=False,
    skip_prep=False,
    mode=LibriSpeechMode.BASIC,
    alignments_folder=None,
):
    """
    This class prepares the csv files for the LibriSpeech dataset.
    Download link: http://www.openslr.org/12

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original LibriSpeech dataset is stored.
    tr_splits : list
        List of train splits to prepare from ['test-others','train-clean-100',
        'train-clean-360','train-other-500'].
    dev_splits : list
        List of dev splits to prepare from ['dev-clean','dev-others'].
    te_splits : list
        List of test splits to prepare from ['test-clean','test-others'].
    save_folder : str
        The directory where to store the csv files.
    select_n_sentences : int
        Default : None
        If not None, only pick this many sentences.
    merge_lst : list
        List of librispeech splits (e.g, train-clean, train-clean-360,..) to
        merge in a singe csv file.
    merge_name: str
        Name of the merged csv file.
    create_lexicon: bool
        If True, it outputs csv files containing mapping between grapheme
        to phonemes. Use it for training a G2P system.
    skip_prep: bool
        If True, data preparation is skipped.
    alignments_folder: str
        The path to the LibriSpeech-Alignments dataset


    Example
    -------
    >>> data_folder = 'datasets/LibriSpeech'
    >>> tr_splits = ['train-clean-100']
    >>> dev_splits = ['dev-clean']
    >>> te_splits = ['test-clean']
    >>> save_folder = 'librispeech_prepared'
    >>> prepare_librispeech(data_folder, save_folder, tr_splits, dev_splits, te_splits)
    """

    if mode == LibriSpeechMode.ALIGNMENT and (
        not alignments_folder or not os.path.exists(alignments_folder)
    ):
        raise ValueError(
            "Alignment mode requires the LibriSpeech-Alignments dataset"
        )

    if skip_prep:
        return
    data_folder = data_folder
    splits = tr_splits + dev_splits + te_splits
    save_folder = save_folder
    select_n_sentences = select_n_sentences
    conf = {
        "select_n_sentences": select_n_sentences,
    }

    # Other variables
    # Saving folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    save_opt = os.path.join(save_folder, OPT_FILE)

    # Check if this phase is already done (if so, skip it)
    if skip(splits, save_folder, conf):
        logger.info("Skipping preparation, completed in previous run.")
        return
    else:
        logger.info("Data_preparation...")

    # Additional checks to make sure the data folder contains Librispeech
    check_librispeech_folders(data_folder, splits)

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

        alignments_dict = {
            file_name: get_alignment_path(
                data_folder=data_folder,
                alignments_folder=alignments_folder,
                file_name=file_name,
            )
            for file_name in wav_lst
        }

        text_dict = text_to_dict(text_lst)
        all_texts.update(text_dict)

        if select_n_sentences is not None:
            n_sentences = select_n_sentences[split_index]
        else:
            n_sentences = len(wav_lst)

        if mode == LibriSpeechMode.ALIGNMENT:
            create_alignment_json(
                save_folder,
                wav_lst,
                text_dict,
                alignments_dict,
                split,
                n_sentences,
            )
        else:
            create_csv(
                save_folder, wav_lst, text_dict, split, n_sentences,
            )

    # Merging csv file if needed
    if merge_lst and merge_name is not None:
        merge_files = [split_libri + ".json" for split_libri in merge_lst]
        if mode == LibriSpeechMode.ALIGNMENT:
            merge_jsons(
                data_folder=save_folder, json_lst=merge_files, merged_json=merge_name,
            )
        else:
            merge_files = [split_libri + ".csv" for split_libri in merge_lst]
            merge_csvs(
                data_folder=save_folder, csv_lst=merge_files, merged_csv=merge_name,
            )

    # Create lexicon.csv and oov.csv
    if create_lexicon:
        create_lexicon_and_oov_csv(all_texts, data_folder, save_folder)

    # saving options
    save_pkl(conf, save_opt)


def create_lexicon_and_oov_csv(all_texts, data_folder, save_folder):
    """
    Creates lexicon csv files useful for training and testing a
    grapheme-to-phoneme (G2P) model.

    Arguments
    ---------
    all_text : dict
        Dictionary containing text from the librispeech transcriptions
    data_folder : str
        Path to the folder where the original LibriSpeech dataset is stored.
    save_folder : str
        The directory where to store the csv files.
    Returns
    -------
    None
    """
    # If the lexicon file does not exist, download it
    lexicon_url = "http://www.openslr.org/resources/11/librispeech-lexicon.txt"
    lexicon_path = os.path.join(save_folder, "librispeech-lexicon.txt")

    if not os.path.isfile(lexicon_path):
        logger.info(
            "Lexicon file not found. Downloading from %s." % lexicon_url
        )
        download_file(lexicon_url, lexicon_path)

    # Get list of all words in the transcripts
    transcript_words = Counter()
    for key in all_texts:
        transcript_words.update(all_texts[key].split("_"))

    # Get list of all words in the lexicon
    lexicon_words = []
    lexicon_pronunciations = []
    with open(lexicon_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            word = line.split()[0]
            pronunciation = line.split()[1:]
            lexicon_words.append(word)
            lexicon_pronunciations.append(pronunciation)

    # Create lexicon.csv
    header = "ID,duration,char,phn\n"
    lexicon_csv_path = os.path.join(save_folder, "lexicon.csv")
    with open(lexicon_csv_path, "w") as f:
        f.write(header)
        for idx in range(len(lexicon_words)):
            separated_graphemes = [c for c in lexicon_words[idx]]
            duration = len(separated_graphemes)
            graphemes = " ".join(separated_graphemes)
            pronunciation_no_numbers = [
                p.strip("0123456789") for p in lexicon_pronunciations[idx]
            ]
            phonemes = " ".join(pronunciation_no_numbers)
            line = (
                ",".join([str(idx), str(duration), graphemes, phonemes]) + "\n"
            )
            f.write(line)
    logger.info("Lexicon written to %s." % lexicon_csv_path)

    # Split lexicon.csv in train, validation, and test splits
    split_lexicon(save_folder, [98, 1, 1])


def split_lexicon(data_folder, split_ratio):
    """
    Splits the lexicon.csv file into train, validation, and test csv files

    Arguments
    ---------
    data_folder : str
        Path to the folder containing the lexicon.csv file to split.
    split_ratio : list
        List containing the training, validation, and test split ratio. Set it
        to [80, 10, 10] for having 80% of material for training, 10% for valid,
        and 10 for test.

    Returns
    -------
    None
    """
    # Reading lexicon.csv
    lexicon_csv_path = os.path.join(data_folder, "lexicon.csv")
    with open(lexicon_csv_path, "r") as f:
        lexicon_lines = f.readlines()
    # Remove header
    lexicon_lines = lexicon_lines[1:]

    # Shuffle entries
    random.shuffle(lexicon_lines)

    # Selecting lines
    header = "ID,duration,char,phn\n"

    tr_snts = int(0.01 * split_ratio[0] * len(lexicon_lines))
    train_lines = [header] + lexicon_lines[0:tr_snts]
    valid_snts = int(0.01 * split_ratio[1] * len(lexicon_lines))
    valid_lines = [header] + lexicon_lines[tr_snts : tr_snts + valid_snts]
    test_lines = [header] + lexicon_lines[tr_snts + valid_snts :]

    # Saving files
    with open(os.path.join(data_folder, "lexicon_tr.csv"), "w") as f:
        f.writelines(train_lines)
    with open(os.path.join(data_folder, "lexicon_dev.csv"), "w") as f:
        f.writelines(valid_lines)
    with open(os.path.join(data_folder, "lexicon_test.csv"), "w") as f:
        f.writelines(test_lines)


FileRecord = namedtuple("FileRecord", "snt_id duration wav_file spk_id wrd")


def create_csv(
    save_folder, wav_lst, text_dict, split, select_n_sentences,
):
    """
    Create the dataset csv file given a list of wav files.

    Arguments
    ---------
    save_folder : str
        Location of the folder for storing the csv.
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
    csv_file = os.path.join(save_folder, split + ".csv")
    if os.path.exists(csv_file):
        logger.info("Csv file %s already exists, not recreating." % csv_file)
        return

    # Preliminary prints
    msg = "Creating csv lists in  %s..." % (csv_file)
    logger.info(msg)

    csv_lines = [["ID", "duration", "wav", "spk_id", "wrd"]]

    snt_cnt = 0
    # Processing all the wav files in wav_lst
    for wav_file in wav_lst:

        csv_line = get_file_record(wav_file, text_dict)

        #  Appending current file to the csv_lines list
        csv_lines.append(csv_line)
        snt_cnt = snt_cnt + 1

        if snt_cnt == select_n_sentences:
            break

    # Writing the csv_lines
    with open(csv_file, mode="w") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )

        for line in csv_lines:
            csv_writer.writerow(line)

    # Final print
    msg = "%s successfully created!" % (csv_file)
    logger.info(msg)


def get_file_record(wav_file, text_dict):
    snt_id = wav_file.split("/")[-1].replace(".flac", "")
    spk_id = "-".join(snt_id.split("-")[0:2])
    wrds = text_dict[snt_id]

    signal, fs = torchaudio.load(wav_file)
    signal = signal.squeeze(0)
    duration = signal.shape[0] / SAMPLERATE
    wrd = str(" ".join(wrds.split("_")))
    return FileRecord(snt_id, duration, wav_file, spk_id, wrd)


def skip(splits, save_folder, conf):
    """
    Detect when the librispeech data prep can be skipped.

    Arguments
    ---------
    splits : list
        A list of the splits expected in the preparation.
    save_folder : str
        The location of the seave directory
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

    #  Checking saved options
    save_opt = os.path.join(save_folder, OPT_FILE)
    if skip is True:
        if os.path.isfile(save_opt):
            opts_old = load_pkl(save_opt)
            if opts_old == conf:
                skip = True
            else:
                skip = False
        else:
            skip = False

    return skip


def create_alignment_json(
    save_folder,
    wav_lst,
    text_dict,
    alignments_dict,
    split,
    select_n_sentences,
    json_indent=2,
):
    """Creates a JSON file with alignment information given a list
    of wav files

    Arguments
    ---------
    save_folder : str
        Location of the folder for storing the csv.
    wav_lst : list
        The list of wav files of a given data split.
    text_dict : dict
        The dictionary containing the text of each sentence.
    alignments_dict: dict
        The dictionary containing the alignment file for
        each sentence
    split : str
        The name of the current data split.
    select_n_sentences : int, optional
        The number of sentences to select.

    Returns
    -------
    None

    """
    json_file = os.path.join(save_folder, split + ".json")
    if os.path.exists(json_file):
        logger.info("JSON file %s already exists, not recreating." % json_file)
        return

    # Preliminary prints
    logger.info("Creating JSON lists in  %s...", json_file)

    snt_cnt = 0
    data = {}
    # Processing all the wav files in wav_lst
    for wav_file in wav_lst:
        alignment_file_name = alignments_dict.get(wav_file)
        if alignment_file_name and os.path.isfile(alignment_file_name):
            alignment_data = parse_alignments(alignment_file_name)
        else:
            logger.warn("No alignments found for %s, skipping", wav_file)
            continue

        record = get_file_record(wav_file, text_dict)
        record_dict = record._asdict()
        record_dict["wav"] = wav_file
        record_dict["char"] = record.wrd
        del record_dict["wav_file"]
        record_dict.update(alignment_data)
        data[record.snt_id] = record_dict
        snt_cnt = snt_cnt + 1

        if snt_cnt == select_n_sentences:
            break

    with open(json_file, "w") as out_file:
        json.dump(data, out_file, indent=json_indent)


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
        with open(file, "r") as f:
            # Reading all line of the transcription file
            for line in f:
                line_lst = line.strip().split(" ")
                text_dict[line_lst[0]] = "_".join(line_lst[1:])
    return text_dict


def check_librispeech_folders(data_folder, splits):
    """
    Check if the data folder actually contains the LibriSpeech dataset.

    If it does not, an error is raised.

    Returns
    -------
    None

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


def get_alignment_path(data_folder, alignments_folder, file_name):
    """Returns the path in the LibriSpeech-Alignments dataset
    corresponding to the specified file path in LibriSpeech

    Arguments
    ---------
    data_folder: str
        the path to LibriSpeech
    alignments_folder: str
        the path to LibriSpeech-Alignments
    file_name: str
        the file name within LibriSpeech

    Returns
    -------
    file_name: str
        the alignment file path
    """
    file_name_rel = os.path.relpath(file_name, data_folder)
    file_dirname = os.path.dirname(file_name_rel)
    file_basename = os.path.basename(file_name_rel)
    file_basename_noext, _ = os.path.splitext(file_basename)

    file_name_alignments = os.path.join(
        alignments_folder, file_dirname, f"{file_basename_noext}.TextGrid"
    )
    return file_name_alignments


def parse_alignments(file_name):
    """Parses a given LibriSpeech-Alignments TextGrid file and
    converts the results to the desired format (to be used in JSON
    metadata)

    Arguments
    ---------
    file_name: str
        the file name of the TextGrid file

    Returns
    -------
    details: dict
        the metadata details
    """
    try:
        import textgrids
    except ImportError:
        print(
            "Parsing LibriSpeech-alignments requires the"
            "praat-textgrids package"
        )

    text_grid = textgrids.TextGrid()
    text_grid.read(file_name)
    word_intervals = [
        {**word, "label": word["label"].upper()} 
        for word in text_grid.interval_tier_to_array("words")
    ]
    phn_intervals = text_grid.interval_tier_to_array("phones")
    details = {}
    details.update(intervals_to_dict(word_intervals, "wrd"))
    details.update(intervals_to_dict(phn_intervals, "phn"))
    return details


INTERVAL_MAP = [("label", ""), ("begin", "_start"), ("end", "_end")]


def intervals_to_dict(intervals, prefix):
    """
    Converts a parsed list of intervals from PRAAT TextGrid
    to a learning-friendly array

    Arguments
    ---------
    intervals: list
        A list of raw TextGrid intervals, as returned by
        TextGrid.interval_tier_to_array
    prefix: str
        the prefix to add

    Returns
    -------
    result: dict
        A dictionary of the form
            {
                "{prefix}": <list of labels>,
                "{prefix}_start": <list of begin values>,
                "{prefix}_end": <list of end values>,
                "{prefix}_count: <number of intervals>
            }

    """
    result = {
        f"{prefix}{suffix}": [interval[key] for interval in intervals]
        for key, suffix in INTERVAL_MAP
    }
    result[f"{prefix}_count"] = len(intervals)
    return result
