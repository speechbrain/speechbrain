"""
LJspeech data preparation.
Download: https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
Authors
 * Sathvik Udupa 2022
"""

import os
import csv
import json
import logging
import random
from speechbrain.dataio.dataio import (
    load_pkl,
    save_pkl,
)

logger = logging.getLogger(__name__)
OPT_FILE = "opt_ljspeech_prepare.pkl"

def prepare_ljspeech(
    data_folder,
    save_folder,
    train,
    valid,
    test,
    duration,
    wavs,
    seed,
):

    random.seed(seed)
    conf = {
        "data_folder": data_folder,
        "save_folder": save_folder,
        "train": train,
        "valid": valid,
        "test":test,
        "duration":duration,
        "wavs":wavs,
        "seed": seed,
    }

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for filename in [train, valid, test, wavs, duration]:
        assert os.path.exists(filename), f"{filename} not found"
    save_opt = os.path.join(save_folder, OPT_FILE)
    save_pkl(conf, save_opt)
