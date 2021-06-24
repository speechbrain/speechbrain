"""
Data download (dataset available at https://osf.io/etdkz/).
Reference to ERPCore: E. S. Kappenman et al., Neuroimage 2021 (https://doi.org/10.1016/j.neuroimage.2020.117465).

Author
------
Davide Borra, 2021
"""

import argparse
import os
from speechbrain.utils.data_utils import download_file


ERPCORE_P3_URL = "https://files.osf.io/v1/resources/etdkz/providers/osfstorage/60077b04ba010908a78927e9/?zip="
parser = argparse.ArgumentParser(
    "Python script to download required recipe data"
)
parser.add_argument(
    "--data_folder", type=str, required=True, help="Target data directory"
)
args = parser.parse_args()

os.makedirs(args.data_folder, exist_ok=True)
print("Downloading ERPCore P3 dataset...")
download_file(
    ERPCORE_P3_URL,
    os.path.join(args.data_folder, "ERPCore_P3.zip"),
    unpack=True,
)
print("Successfully downloaded and unpacked in {0}".format(args.data_folder))
