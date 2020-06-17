"""
Data preparation for LM corpus.

Author
------
Ju-Chieh Chou 2020
"""

import os
import csv
import logging
from speechbrain.data_io.data_io import (
    load_pkl,
    save_pkl,
)

logger = logging.getLogger(__name__)
OPT_FILE = "opt_librispeech_lm_corpus_prepare.pkl"


def prepare_librispeech_lm_corpus(
    data_folder, save_folder, csv_filename, select_n_sentences=None
):
    """
    This function prepares the csv file for the LibriSpeech LM corpus.
    Download link: http://www.openslr.org/11/

    Arguments:
    data_folder : str
        Path to the folder of LM (normalized) corpus.
    save_folder : str
        folder to store the csv file.
    csv_filename : str
        The filename of csv file.
    select_n_sentences : int
        Default : None
        If not None, only pick this many sentences.

    Example
    -------
    >>> data_folder = 'dataset/LibriSpeech'
    >>> save_folder = 'librispeech_lm'
    >>> prepare_librispeech(data_folder, save_folder)
    """
    conf = {
        "select_n_sentences": select_n_sentences,
    }
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    save_opt = os.path.join(save_folder, OPT_FILE)

    # Check if this phase is already done (if so, skip it)
    if skip(save_folder, csv_filename, conf):
        logger.debug("Skipping preparation, completed in previous run.")
        return

    data_path = os.path.join(data_folder, "librispeech-lm-norm.txt")

    create_csv(
        data_path, save_folder, csv_filename, select_n_sentences,
    )

    # saving options
    save_pkl(conf, save_opt)


def skip(save_folder, csv_filename, conf):
    """
    Detect when the librispeech data prep can be skipped.

    Arguments
    ---------
    save_folder : str
        The location of the seave directory
    csv_filename : str
        The filename of the csv file.
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

    if not os.path.isfile(os.path.join(save_folder, csv_filename + ".csv")):
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


def create_csv(
    data_path, save_folder, csv_filename, select_n_sentences,
):
    """
    Create the csv file.

    Arguments
    ---------
    data_path : str
        The path of LM corpus txt file.
    save_folder : str
        Location of the folder for storing the csv.
    csv_filename : str
        The filename of csv file.
    select_n_sentences : int, optional
        The number of sentences to select.

    Returns
    -------
    None
    """
    # Setting path for the csv file
    csv_file = os.path.join(save_folder, csv_filename)

    # Preliminary prints
    msg = "\tCreating csv in  %s..." % (csv_file)
    logger.debug(msg)

    header = [
        "ID",
        "duration",
        "wrd",
        "wrd_format",
        "wrd_opts",
        "char",
        "char_format",
        "char_opts",
    ]

    snt_cnt = 0
    with open(data_path, "r") as f_in, open(csv_file, mode="w") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        csv_writer.writerow(header)
        for snt_id, line in enumerate(f_in):
            wrds = "_".join(line.strip().split(" "))

            # skip empty sentences
            if len(wrds) == 0:
                continue

            # replace space to <space> token
            chars_lst = [c for c in wrds]
            chars = " ".join(chars_lst)

            # TODO: duration set to char len temporarily
            csv_line = [
                snt_id,
                len(chars),
                str(" ".join(wrds.split("_"))),
                "string",
                "",
                str(chars),
                "string",
                "",
            ]
            csv_writer.writerow(csv_line)
            snt_cnt = snt_cnt + 1

            if snt_cnt == select_n_sentences:
                break

    # Final print
    msg = "\t%s sucessfully created!" % (csv_file)
    logger.debug(msg)
