#!/usr/bin/env python3
# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
This file downloads the following LibriSpeech LM files:

    - 3-gram.pruned.1e-7.arpa.gz
    - 4-gram.arpa.gz
    - librispeech-vocab.txt
    - librispeech-lexicon.txt

from http://www.openslr.org/resources/11
and save them in the user provided directory.

Files are not re-downloaded if they already exist.

Usage:
    ./local/download_lm.py --out-dir ./download/lm
"""

import argparse
import gzip
import logging
import os
import shutil
from pathlib import Path

from tqdm.auto import tqdm

def urlretrieve_progress(url, filename=None, data=None, desc=None):
    """
    Works exactly like urllib.request.urlretrieve, but attaches a tqdm hook to display
    a progress bar of the download.
    Use "desc" argument to display a user-readable string that informs what is being downloaded.
    """
    from urllib.request import urlretrieve
    with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=desc) as t:
        reporthook = tqdm_urlretrieve_hook(t)
        return urlretrieve(url=url, 
                           filename=filename, 
                           reporthook=reporthook, 
                           data=data)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, help="Output directory.")

    args = parser.parse_args()
    return args


def main(out_dir: str):
    url = "http://www.openslr.org/resources/11"
    out_dir = Path(out_dir)

    files_to_download = (
        "3-gram.pruned.1e-7.arpa.gz",
        "4-gram.arpa.gz",
        "librispeech-vocab.txt",
        "librispeech-lexicon.txt",
    )

    for f in tqdm(files_to_download, desc="Downloading LibriSpeech LM files"):
        filename = out_dir / f
        if filename.is_file() is False:
            urlretrieve_progress(
                f"{url}/{f}",
                filename=filename,
                desc=f"Downloading {filename}",
            )
        else:
            logging.info(f"{filename} already exists - skipping")

        if ".gz" in str(filename):
            unzipped = Path(os.path.splitext(filename)[0])
            if unzipped.is_file() is False:
                with gzip.open(filename, "rb") as f_in:
                    with open(unzipped, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
            else:
                logging.info(f"{unzipped} already exist - skipping")


if __name__ == "__main__":
    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )

    logging.basicConfig(format=formatter, level=logging.INFO)

    args = get_args()
    logging.info(f"out_dir: {args.out_dir}")

    main(out_dir=args.out_dir)
