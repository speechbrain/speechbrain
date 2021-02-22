"""
LibriParty Dataset creation by using official metadata.
Author
------
Samuele Cornell, 2020
Mirco Ravanelli, 2020
"""


import os
import sys
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.data_utils import download_file
from local.create_mixtures_from_metadata import create_mixture
import json
from tqdm import tqdm

URL_METADATA = (
    "https://www.dropbox.com/s/0u6x6ndyedb4rl7/LibriParty_metadata.zip?dl=1"
)

# Load hyperparameters file with command-line overrides
params_file, run_opts, overrides = sb.core.parse_arguments(sys.argv[1:])
with open(params_file) as fin:
    params = load_hyperpyyaml(fin, overrides)

metadata_folder = params["metadata_folder"]
if not os.path.exists(metadata_folder):
    os.makedirs(metadata_folder)

# Download meta data from the web
download_file(
    URL_METADATA,
    metadata_folder + "/meta.zip",
    unpack=True,
    dest_unpack=metadata_folder,
)

for data_split in ["train", "dev", "eval"]:
    with open(os.path.join(metadata_folder, data_split + ".json"), "r") as f:
        metadata = json.load(f)
    print("Creating data for {} set".format(data_split))
    c_folder = os.path.join(params["out_folder"], data_split)
    os.makedirs(c_folder, exist_ok=True)
    for sess in tqdm(metadata.keys()):
        create_mixture(sess, c_folder, params, metadata[sess])
