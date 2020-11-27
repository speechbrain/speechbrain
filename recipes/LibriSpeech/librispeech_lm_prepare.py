"""
Data preparation for LM corpus.

Author
------
Ju-Chieh Chou 2020
"""

import os
import logging
from speechbrain.data_io.data_io import (
    load_pkl,
    save_pkl,
)
import h5py
from speechbrain.utils.data_utils import download_file
import gzip

logger = logging.getLogger(__name__)
OPT_FILE = "opt_librispeech_lm_corpus_prepare.pkl"


def prepare_lm_corpus(
    data_folder, save_folder, filename, select_n_sentences=None, add_txt=None
):
    """
    This function prepares the hdf5 file for the LibriSpeech LM corpus.
    Download link: http://www.openslr.org/11/

    Arguments:
    data_folder : str
        Path to the folder of LM (normalized) corpus.
    save_folder : str
        folder to store the hdf5 file.
    filename : str
        The filename of hdf5 file.
    select_n_sentences : int
        Default : None
        If not None, only pick this many sentences.
    add_txt: list
        Additional test to add in the LM (e.g, text from
        the training transcriptions on LibriSpeech)

    Example
    -------
    >>> data_folder = 'dataset/LibriSpeech'
    >>> save_folder = 'librispeech_lm'
    >>> prepare_librispeech(data_folder, save_folder, 'lm_corpus.h5')
    """
    conf = {
        "select_n_sentences": select_n_sentences,
    }
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    save_opt = os.path.join(save_folder, OPT_FILE)

    # Check if this phase is already done (if so, skip it)
    if skip(save_folder, filename, conf):
        logger.info("Skipping preparation, completed in previous run.")
        return

    data_path = os.path.join(data_folder, "librispeech-lm-norm.txt.gz")
    src = "http://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz"
    download_file(src, data_path)

    create_hdf5(
        data_path, save_folder, filename, select_n_sentences, add_txt=None
    )

    # saving options
    save_pkl(conf, save_opt)


def skip(save_folder, filename, conf):
    """
    Detect when the librispeech data prep can be skipped.

    Arguments
    ---------
    save_folder : str
        The location of the seave directory
    filename : str
        The filename of the file.
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

    if not os.path.isfile(os.path.join(save_folder, filename)):
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


def create_hdf5(
    data_path, save_folder, filename, select_n_sentences, add_txt=None
):
    """
    Create the hdf5 file.

    Arguments
    ---------
    data_path : str
        The path of LM corpus txt file (compressed).
    save_folder : str
        Location of the folder for storing the csv.
    filename : str
        The filename of hdf5 file.
    select_n_sentences : int, optional
        The number of sentences to select.
    add_txt: list
        Additional test to add in the LM (e.g, text from
        the training transcriptions on LibriSpeech)

    Returns
    -------
    None
    """
    # Setting path for the csv file
    hdf5_file = os.path.join(save_folder, filename)

    # Preliminary prints
    msg = "\tCreating hdf5 in  %s..." % (hdf5_file)
    logger.info(msg)

    snt_cnt = 0
    all_wrds, all_chars = [], []

    if add_txt is not None:
        all_wrds = add_txt
        chars_lst = [c for c in "_".join(all_wrds)]
        all_chars = " ".join(chars_lst)

    with gzip.open(data_path, "rt") as f_in:
        for snt_id, line in enumerate(f_in):
            wrds = line.strip()
            wrds_lst = wrds.split(" ")

            # skip empty sentences
            if len(wrds) == 0:
                continue

            # replace space to <space> token
            chars_lst = [c for c in "_".join(wrds_lst)]
            chars = " ".join(chars_lst)

            all_wrds.append(wrds)
            all_chars.append(chars)

            snt_cnt = snt_cnt + 1
            if snt_cnt == select_n_sentences:
                break

    with h5py.File(hdf5_file, "w") as f_h5:
        dset = f_h5.create_dataset(
            "wrd", (len(all_wrds),), dtype=h5py.string_dtype()
        )
        dset[:] = all_wrds
        dset = f_h5.create_dataset(
            "char", (len(all_chars),), dtype=h5py.string_dtype()
        )
        dset[:] = all_chars

    # Final print
    msg = "\t%s sucessfully created!" % (hdf5_file)
    logger.info(msg)
