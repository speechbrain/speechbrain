"""
Custom LibriParty creation script with user specified parameters.

Author
------
Samuele Cornell, 2020
"""


import os
import sys
import json
import random
import numpy as np
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.data_utils import get_all_files
from local.create_mixtures_metadata import create_metadata
from local.create_mixtures_from_metadata import create_mixture
from pathlib import Path
from tqdm import tqdm

# Load hyperparameters file with command-line overrides
params_file, run_opts, overrides = sb.core.parse_arguments(sys.argv[1:])
with open(params_file) as fin:
    params = load_hyperpyyaml(fin, overrides)

# setting seeds for reproducible code.
np.random.seed(params["seed"])
random.seed(params["seed"])
# we parse the yaml, and create mixtures for every train, dev and eval split.


def split_list(array, split_factors):
    assert round(sum(split_factors), 6) == 1, "split_factors should sum to one"
    np.random.shuffle(array)
    pivots = [int(len(array) * x) for x in split_factors]
    out = []
    indx = 0
    for i in pivots:
        out.append(array[indx : i + indx])
        indx = i
    return out


def parse_libri_folder(libri_folders):
    # parsing librispeech
    # step 1:  we construct a dictionary with speakers ids
    utterances = []
    txt_files = []
    for libri_dir in libri_folders:
        utterances.extend(get_all_files(libri_dir, match_and=[".flac"]))
        txt_files.extend(get_all_files(libri_dir, match_and=["trans.txt"]))
    # step 2: we then build an hashtable for words for each utterance
    words_dict = {}
    for trans in txt_files:
        with open(trans, "r") as f:
            for line in f:
                splitted = line.split(" ")
                utt_id = splitted[0]
                words = " ".join(splitted[1:])
                words_dict[utt_id] = words.strip("\n")

    # step 3: we build an hashtable for also speakers
    speakers = {}
    for u in utterances:
        spk_id = Path(u).parent.parent.stem
        if spk_id not in speakers:
            speakers[spk_id] = [u]
        else:
            speakers[spk_id].append(u)

    return speakers, words_dict


# split
split_f = params["split_factors"]
# we get all noises and rirs
noises = []
for f in params["noises_folders"]:
    noises.extend(get_all_files(f, match_and=[".wav"]))
rirs = []
for f in params["rirs_folders"]:
    rirs.extend(get_all_files(f, match_and=[".wav"]))
# we split them in training, dev and eval
noises = split_list(noises, split_f)
rirs = split_list(rirs, split_f)
# do the same for background noises
if params["backgrounds_root"]:
    backgrounds = get_all_files(params["backgrounds_root"], match_and=[".wav"])
    backgrounds = split_list(backgrounds, split_f)
else:
    backgrounds = [None] * 3

os.makedirs(os.path.join(params["out_folder"], "metadata"), exist_ok=True)

# we generate metadata for each split
for indx, split in enumerate(["train", "dev", "eval"]):
    print("Generating metadata for {} set".format(split))
    # we parse librispeech utterances for current split
    c_libri_folder = params["librispeech_folders"][split]
    c_utterances, c_words = parse_libri_folder(c_libri_folder)

    create_metadata(
        os.path.join(params["out_folder"], "metadata", split),
        params["n_sessions"][split],
        params,
        c_utterances,
        c_words,
        rirs[indx],
        noises[indx],
        backgrounds[indx],
    )

# from metadata we generate the actual mixtures

for indx, split in enumerate(["train", "dev", "eval"]):
    # load metadata
    with open(
        os.path.join(params["out_folder"], "metadata", split + ".json")
    ) as f:
        c_meta = json.load(f)
    print("Creating {} set".format(split))
    for sess in tqdm(c_meta.keys()):
        c_folder = os.path.join(params["out_folder"], split)
        os.makedirs(c_folder, exist_ok=True)
        create_mixture(sess, c_folder, params, c_meta[sess])
