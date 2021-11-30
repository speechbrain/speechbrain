"""
Data preparation.

Download: https://aihub.or.kr/aidata/105/download

Author
------
Dongwon Kim, Dongwoo Kim 2021
"""
import csv
import logging
import os
import re

import torchaudio

from speechbrain.dataio.dataio import load_pkl, merge_csvs, save_pkl
from speechbrain.utils.data_utils import get_all_files

logger = logging.getLogger(__name__)
OPT_FILE = "opt_ksponspeech_prepare.pkl"
SAMPLERATE = 16000


def prepare_ksponspeech(
    data_folder,
    save_folder,
    tr_splits=[],
    dev_splits=[],
    te_splits=[],
    select_n_sentences=None,
    merge_lst=[],
    merge_name=None,
    skip_prep=False,
):
    """
    This class prepares the csv files for the KsponSpeech dataset.
    Download link: https://aihub.or.kr/aidata/105/download

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original KsponSpeech dataset is stored.
    tr_splits : list
        List of train splits to prepare from ['train', 'dev', 'eval_clean',
        'eval_other'].
    dev_splits : list
        List of dev splits to prepare from ['dev'].
    te_splits : list
        List of test splits to prepare from ['eval_clean','eval_other'].
    save_folder : str
        The directory where to store the csv files.
    select_n_sentences : int
        Default : None
        If not None, only pick this many sentences.
    merge_lst : list
        List of KsponSpeech splits (e.g, eval_clean, eval_other) to
        merge in a singe csv file.
    merge_name: str
        Name of the merged csv file.
    skip_prep: bool
        If True, data preparation is skipped.


    Example
    -------
    >>> data_folder = 'datasets/KsponSpeech'
    >>> tr_splits = ['train']
    >>> dev_splits = ['dev']
    >>> te_splits = ['eval_clean']
    >>> save_folder = 'KsponSpeech_prepared'
    >>> prepare_ksponspeech(data_folder, save_folder, tr_splits, dev_splits, \
                            te_splits)
    """

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

    # Additional checks to make sure the data folder contains ksponspeech
    check_ksponspeech_folders(data_folder, splits)

    # parse trn file
    all_texts = {}
    for split_index in range(len(splits)):

        split = splits[split_index]
        dirlist = split2dirs(split)
        wav_lst = []
        for dir in dirlist:
            wav_lst += get_all_files(
                os.path.join(data_folder, dir), match_and=[".wav"]
            )

        trnpath = os.path.join(data_folder, split + ".trn")
        text_dict = text_to_dict(trnpath)
        all_texts.update(text_dict)

        if select_n_sentences is not None:
            n_sentences = select_n_sentences[split_index]
        else:
            n_sentences = len(wav_lst)

        create_csv(
            save_folder, wav_lst, text_dict, split, n_sentences,
        )

    # Merging csv file if needed
    if merge_lst and merge_name is not None:
        merge_files = [split_kspon + ".csv" for split_kspon in merge_lst]
        merge_csvs(
            data_folder=save_folder, csv_lst=merge_files, merged_csv=merge_name,
        )

    # saving options
    save_pkl(conf, save_opt)


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

    # Preliminary prints
    msg = "Creating csv lists in  %s..." % (csv_file)
    logger.info(msg)

    csv_lines = [["ID", "duration", "wav", "spk_id", "wrd"]]

    snt_cnt = 0
    # Processing all the wav files in wav_lst
    for wav_file in wav_lst:

        snt_id = wav_file.split("/")[-1].replace(".wav", "")
        spk_id = snt_id.split("_")[-1]
        wrds = text_dict[snt_id]

        duration = torchaudio.info(wav_file).num_frames / SAMPLERATE

        csv_line = [
            snt_id,
            str(duration),
            wav_file,
            spk_id,
            str(" ".join(wrds.split())),
        ]

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


def skip(splits, save_folder, conf):
    """
    Detect when the ksponspeech data prep can be skipped.

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


def text_to_dict(trnpath):
    """
    This converts lines of text into a dictionary-

    Arguments
    ---------
    text_lst : str
        Path to the file containing the ksponspeech text transcription.

    Returns
    -------
    dict
        The dictionary containing the text transcriptions for each sentence.

    """
    # Initialization of the text dictionary
    text_dict = {}
    # Reading all the transcription files is text_lst
    with open(trnpath, "r") as f:
        # Reading all line of the transcription file
        for line in f:
            filename, raw_script = line.split(" :: ")
            file_id = filename.split("/")[-1].replace(".pcm", "")
            script = normalize(raw_script)
            text_dict[file_id] = script
    return text_dict


def normalize(string):
    """
    This function normalizes a given string according to
    the normalization rule
    The normalization rule removes "/" indicating filler words,
    removes "+" indicating repeated words,
    removes all punctuation marks,
    removes non-speech symbols,
    and extracts orthographic transcriptions.

    Arguments
    ---------
    string : str
        The string to be normalized

    Returns
    -------
    str
        The string normalized according to the rules

    """
    # extracts orthographic transcription
    string = re.sub(r"\(([^)]*)\)\/\(([^)]*)\)", r"\1", string)
    # removes non-speech symbols
    string = re.sub(r"n/|b/|o/|l/|u/", "", string)
    # removes punctuation marks
    string = re.sub(r"[+*/.?!,]", "", string)
    # removes extra spaces
    string = re.sub(r"\s+", " ", string)
    string = string.strip()

    return string


def split2dirs(split):
    """
    This gives directory names for a given data split

    Arguments
    ---------
    split : str
        The split of ksponspeech data

    Returns
    -------
    list
        A list containing directories of the given data split

    """

    if split not in ["eval_other", "eval_clean", "train", "dev"]:
        raise ValueError("Unsupported data split")

    if "eval" in split:
        dirs = ["test/" + split]

    elif split == "dev":
        dirs = [
            "train/KsponSpeech_05/KsponSpeech_{0:>04d}".format(num)
            for num in range(621, 624)
        ]

    elif split == "train":
        dirs = (
            [
                "train/KsponSpeech_01/KsponSpeech_{0:>04d}".format(num)
                for num in range(1, 125)
            ]
            + [
                "train/KsponSpeech_02/KsponSpeech_{0:>04d}".format(num)
                for num in range(125, 249)
            ]
            + [
                "train/KsponSpeech_03/KsponSpeech_{0:>04d}".format(num)
                for num in range(249, 373)
            ]
            + [
                "train/KsponSpeech_04/KsponSpeech_{0:>04d}".format(num)
                for num in range(373, 497)
            ]
            + [
                "train/KsponSpeech_05/KsponSpeech_{0:>04d}".format(num)
                for num in range(497, 621)
            ]
        )

    return dirs


def check_ksponspeech_folders(data_folder, splits):
    """
    Check if the data folder actually contains the ksponspeech dataset.

    If it does not, an error is raised.

    Returns
    -------
    None

    Raises
    ------
    OSError
        If ksponspeech is not found at the specified path.
    """
    # Checking if all the splits exist

    for split in splits:
        if split not in ["eval_other", "eval_clean", "train", "dev"]:
            raise ValueError("Unsupported data split")

        if "eval" in split:
            trn_folder = os.path.join(data_folder, split + ".trn")
            if not os.path.exists(trn_folder):
                err_msg = (
                    "the file %s does not exist (it is expected in the "
                    "ksponspeech dataset)" % trn_folder
                )
                raise OSError(err_msg)

        elif split == "dev":
            trn_folder = os.path.join(data_folder, "train.trn")
            if not os.path.exists(trn_folder):
                err_msg = (
                    "the file %s does not exist (it is expected in the "
                    "ksponspeech dataset)" % trn_folder
                )
                raise OSError(err_msg)

        elif split == "train":
            trn_folder = os.path.join(data_folder, "train.trn")
            if not os.path.exists(trn_folder):
                err_msg = (
                    "the file %s does not exist (it is expected in the "
                    "ksponspeech dataset)" % trn_folder
                )
                raise OSError(err_msg)

        dirs = split2dirs(split)

        for dir in dirs:
            dir_folder = os.path.join(data_folder, dir)
            if not os.path.exists(dir_folder):
                err_msg = (
                    "the file %s does not exist (it is expected in the "
                    "ksponspeech dataset)" % dir_folder
                )
                raise OSError(err_msg)
