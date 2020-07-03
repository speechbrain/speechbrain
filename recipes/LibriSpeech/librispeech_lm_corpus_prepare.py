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
import numpy as np

logger = logging.getLogger(__name__)
OPT_FILE = "opt_librispeech_lm_corpus_prepare.pkl"


def prepare_librispeech_lm_corpus(
    data_folder, save_folder, hdf5_filename, label_dict, select_n_sentences=None
):
    """
    This function prepares the hdf5 file for the LibriSpeech LM corpus.
    Download link: http://www.openslr.org/11/

    Arguments:
    data_folder : str
        Path to the folder of LM (normalized) corpus.
    save_folder : str
        folder to store the csv file.
    hdf5_filename : str
        The filename of hdf5 file.
    label_dict : dict
        The dictionary for label2ind.
    select_n_sentences : int
        Default : None
        If not None, only pick this many sentences.

    Example
    -------
    >>> data_folder = 'dataset/LibriSpeech'
    >>> save_folder = 'librispeech_lm'
    >>> prepare_librispeech(data_folder, save_folder, 'lm.h5')
    """
    conf = {
        "select_n_sentences": select_n_sentences,
    }
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    save_opt = os.path.join(save_folder, OPT_FILE)

    # Check if this phase is already done (if so, skip it)
    if skip(save_folder, hdf5_filename, conf):
        logger.debug("Skipping preparation, completed in previous run.")
        return

    data_path = os.path.join(data_folder, "librispeech-lm-norm.txt")

    dump_hdf5(
        data_path, save_folder, hdf5_filename, label_dict, select_n_sentences,
    )

    # saving options
    save_pkl(conf, save_opt)


def skip(save_folder, hdf5_filename, conf):
    """
    Detect when the librispeech lm corpus data prep can be skipped.

    Arguments
    ---------
    save_folder : str
        The location of the seave directory
    hdf5_filename : str
        The filename of the hdf5 file.
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

    if not os.path.isfile(os.path.join(save_folder, hdf5_filename)):
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


def dump_hdf5(
    data_path, save_folder, hdf5_filename, label_dict, select_n_sentences,
):
    """
    dump the hdf5 file.

    Arguments
    ---------
    data_path : str
        The path of LM corpus txt file.
    save_folder : str
        Location of the folder for storing the csv.
    hdf5_filename : str
        The filename of hdf5 file.
    label_dict : dict
        The dictionary for label2ind.
    select_n_sentences : int, optional
        The number of sentences to select.

    Returns
    -------
    None
    """
    # Setting path for the csv file
    hdf5_file = os.path.join(save_folder, hdf5_filename)

    # Preliminary prints
    msg = "Dump hdf5 in  %s..." % (hdf5_file)
    logger.debug(msg)

    snt_cnt, res = 0, []
    with open(data_path, "r") as f_in, h5py.File(hdf5_file, "w") as f_h5:
        for snt_id, line in enumerate(f_in):
            wrds = "_".join(line.strip().split(" "))

            # skip empty sentences
            if len(wrds) == 0:
                continue

            char_lst = np.array([label_dict[c] for c in wrds], dtype=np.int32)
            res.append(char_lst)
            snt_cnt = snt_cnt + 1

            if snt_cnt == select_n_sentences:
                break
        f_h5.create_dataset(
            "data", data=res, dtype=h5py.special_dtype(vlen=np.int32)
        )

    # Final print
    msg = "\t%s sucessfully created!" % (hdf5_file)
    logger.debug(msg)
