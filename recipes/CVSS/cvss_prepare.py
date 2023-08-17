"""
CVSS data preparation.
Download: https://github.com/google-research-datasets/cvss

Authors
 * Jarod DURET 2023
"""

import os
import csv
import json
import logging
import random
import tqdm
import pathlib as pl

import torchaudio
from speechbrain.dataio.dataio import (
    load_pkl,
    save_pkl,
)

OPT_FILE = "opt_cvss_prepare.pkl"

SRC_METADATA = "validated.tsv"
TGT_METADATA = {
    "train": "train.tsv",
    "valid": "dev.tsv",
    "test": "test.tsv",
}

# Need to be set according to your system
SRC_AUDIO = "clips"
TGT_AUDIO = {
    "train": "train",
    "valid": "dev",
    "test": "test",
}

# Number of samples for the small evalution subset
SMALL_EVAL_SIZE = 1000

log_format = "[%(asctime)s] [%(levelname)s]: %(message)s"
logging.basicConfig(format=log_format, level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_cvss(
    src_data_folder,
    tgt_data_folder,
    save_folder,
    splits=["train", "valid", "test"],
    seed=1234,
    skip_prep=False,
):
    """
    Prepares the csv files for the CVSS datasets.

    Arguments
    ---------
    src_data_folder : str
        Path to the folder where the original source CV data is stored.
    tgt_data_folder : str
        Path to the folder where the original target CVSS data is stored.
    save_folder : str
        The directory where to store the csv files.
    splits : list
        List of splits to prepare.
    skip_prep: Bool
        If True, skip preparation.
    seed : int
        Random seed
    """
    # setting seeds for reproducible code.
    random.seed(seed)

    if skip_prep:
        return

    # Create configuration for easily skipping data_preparation stage
    conf = {
        "src_data_folder": src_data_folder,
        "tgt_data_folder": tgt_data_folder,
        "splits": splits,
        "save_folder": save_folder,
        "seed": seed,
    }

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    src_validated = pl.Path(src_data_folder) / SRC_METADATA
    tgt_train = pl.Path(tgt_data_folder) / TGT_METADATA["train"]
    tgt_valid = pl.Path(tgt_data_folder) / TGT_METADATA["valid"]
    tgt_test = pl.Path(tgt_data_folder) / TGT_METADATA["test"]

    src_audio = pl.Path(src_data_folder) / SRC_AUDIO
    tgt_audio_train = pl.Path(tgt_data_folder) / TGT_AUDIO["train"]
    tgt_audio_valid = pl.Path(tgt_data_folder) / TGT_AUDIO["valid"]
    tgt_audio_test = pl.Path(tgt_data_folder) / TGT_AUDIO["test"]

    save_opt = pl.Path(save_folder) / OPT_FILE
    save_json_train = pl.Path(save_folder) / "train.json"
    save_json_valid = pl.Path(save_folder) / "valid.json"
    save_json_valid_small = pl.Path(save_folder) / "valid_small.json"
    save_json_test = pl.Path(save_folder) / "test.json"

    # Check if this phase is already done (if so, skip it)
    if skip(splits, save_folder, conf):
        logger.info("Skipping preparation, completed in previous run.")
        return

    msg = "\tCreating json file for CVSS Dataset.."
    logger.info(msg)

    # Prepare csv
    if "train" in splits:
        prepare_json(
            save_json_train,
            src_audio,
            tgt_audio_train,
            src_validated,
            tgt_train,
        )
    if "valid" in splits:
        prepare_json(
            save_json_valid,
            src_audio,
            tgt_audio_valid,
            src_validated,
            tgt_valid,
        )
        prepare_json(
            save_json_valid_small,
            src_audio,
            tgt_audio_valid,
            src_validated,
            tgt_valid,
            limit_to_n_sample=SMALL_EVAL_SIZE,
        )
    if "test" in splits:
        prepare_json(
            save_json_test, src_audio, tgt_audio_test, src_validated, tgt_test,
        )

    save_pkl(conf, save_opt)


def skip(splits, save_folder, conf):
    """
    Detects if the cvss data_preparation has been already done.
    If the preparation has been done, we can skip it.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """
    # Checking json files
    skip = True

    split_files = {
        "train": "train.json",
        "valid": "valid.json",
        "valid_small": "valid_small.json",
        "test": "test.json",
    }

    for split in splits:
        if not os.path.isfile(os.path.join(save_folder, split_files[split])):
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


def prepare_json(
    json_file,
    src_audio_folder,
    tgt_audio_folder,
    src_validated,
    tgt_split,
    limit_to_n_sample=None,
):
    """
    Creates json file.

    """

    json_dict = {}
    tgt_meta = list(
        csv.reader(open(tgt_split), delimiter="\t", quoting=csv.QUOTE_NONE)
    )

    limit_to_n_sample = (
        len(tgt_meta) if not limit_to_n_sample else limit_to_n_sample
    )

    for i in tqdm.tqdm(range(limit_to_n_sample)):
        session_id = tgt_meta[i][0].split(".")[0]

        tgt_audio = f"{tgt_audio_folder}/{session_id}.mp3.wav"
        src_audio = f"{src_audio_folder}/{session_id}.mp3"

        src_sig, sr = torchaudio.load(src_audio)
        duration = src_sig.shape[1] / sr

        # src_text = meta_dict[session_id]["sentence"]
        tgt_text = tgt_meta[i][1]

        if duration < 1.5 or len(tgt_text) < 10:
            continue

        json_dict[session_id] = {
            "src_audio": src_audio,
            "tgt_audio": tgt_audio,
            "duration": duration,
            # "src_text": src_text,
            "tgt_text": tgt_text,
        }

    # Writing the dictionary to the json file
    with open(json_file, mode="w") as json_f:
        json.dump(json_dict, json_f, indent=2)

    logger.info(f"{json_file} successfully created!")
