"""
Creates data manifest files for SEP-28k
If the data does not exist, we download the data automatically.

Authors:
 * Ilias Maoudj 2024

Adapted from the ESC50 and Urbansound8k recipe.
"""

import logging
import os
import shutil

import pandas as pd
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


def download_dropbox(url, podcast):
    headers = {"user-agent": "Wget/1.16 (linux-gnu)"}
    r = requests.get(url, stream=True, headers=headers)
    with open(podcast, "wb") as f:
        for chunk in tqdm(r.iter_content(chunk_size=1024)):
            if chunk:
                f.write(chunk)


def download_sep28k(data_path):
    """
    This function automatically downloads the SEP-28k dataset to the specified data path in the data_path variable

    Arguments
    ---------
    data_path: str or Path
        Directory used to save the dataset.
    """
    temp_path = os.path.join(data_path, "temp_download")
    if not os.path.exists(temp_path):
        os.mkdir(temp_path)
    if not os.path.exists(os.path.join(data_path, "SEP28k-data.zip")):
        print(
            "SEP-28k is missing. We are now downloading it. Be patient, the total size is 1.9GB. Takes 1,995,736 iterations."
        )
        print("**** NOW DOWNLOADING zip file *******")
        download_dropbox(
            "https://www.dropbox.com/scl/fi/rpavffri0odb2g25bxy58/sep28k_clips.zip?rlkey=zfxpdrek642pxu0rj64qid7gh&st=xum8yxm1&dl=0",
            f"{temp_path}/SEP28k-data.zip",
        )
    if not os.path.exists(
        os.path.join(data_path, "SEP-28k-Extended_clips.csv")
    ):
        print("**** NOW DOWNLOADING csv file *******")
        download_dropbox(
            "https://www.dropbox.com/scl/fi/amzp62bpj8zqo21kpmoqy/SEP-28k-Extended_ \
            clips.csv?rlkey=5ehd8wv1q2gyz32m2pyynlcn2&st=cb76g1r4&dl=0",
            f"{temp_path}/SEP-28k-Extended_clips.csv",
        )
    files = os.listdir(temp_path)
    for fl in files:
        shutil.move(os.path.join(temp_path, fl), data_path)
    shutil.rmtree(os.path.join(temp_path))
    if not os.path.exists(os.path.join(data_path, "SEP28k-data")):
        shutil.unpack_archive(
            os.path.join(data_path, "SEP28k-data.zip"), data_path
        )
    print(f"SEP-28k is downloaded in {data_path}")


def prepare_sep28k(data_folder, split_type="SEP28k-E"):
    """

    Arguments
    ---------
    data_folder: str
        Where to save the dataset
    split_type: str
        Which partitioning to use (can be either SEP12k, SEP28k-E [default], SEP28k-T, SEP28k-D)

    """
    if not os.path.exists(os.path.join(data_folder)):
        os.mkdir(data_folder)
    download_sep28k(data_folder)

    df = pd.read_csv(f"{data_folder}/SEP-28k-Extended_clips.csv")
    df["ID"] = df.index
    df_train = df[df[split_type] == "train"]
    df_valid = df[df[split_type] == "dev"]
    df_test = df[df[split_type] == "test"]

    df_train.to_csv(f"{data_folder}/{split_type}_train.csv")
    df_valid.to_csv(f"{data_folder}/{split_type}_valid.csv")
    df_test.to_csv(f"{data_folder}/{split_type}_test.csv")
    df_all = pd.concat([df_train, df_valid, df_test])
    df_all.to_csv(f"{data_folder}/{split_type}_clean.csv")
