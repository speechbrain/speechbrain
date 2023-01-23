#!/usr/bin/env python3

"""
Common Voice data preparation.

Authors
 * Luca Della Libera 2022
"""

import argparse
import csv
import logging
import os
import random
import re
import shutil
import tarfile
from typing import Optional, Sequence

import requests
import torchaudio
from tqdm import tqdm


__all__ = [
    "prepare_common_voice",
]


# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(funcName)s - %(message)s",
)

# Set backend to SoX (needed to read MP3 files)
if torchaudio.get_audio_backend() != "sox_io":
    torchaudio.set_audio_backend("sox_io")


_LOGGER = logging.getLogger(__name__)

_URL_TEMPLATE = (
    "https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com"
    "/cv-corpus-$version/cv-corpus-$version-$locale.tar.gz"
)

_SPLITS = ["train", "dev", "test"]

# Random indices are not generated on the fly but statically read from a predefined
# file to avoid reproducibility issues on different platforms and/or Python versions
with open(os.path.join(os.path.dirname(__file__), "random_idxes.txt")) as f:
    _RANDOM_IDXES = [int(line) for line in f]

# Default seed
random.seed(0)


def prepare_common_voice(
    locales: "Sequence[str]" = ("en",),
    download_dir: "str" = "data",
    version: "str" = "12.0-2022-12-07",
    max_duration: "Optional[float]" = None,
    shuffle: "bool" = False,
) -> "None":
    """Prepare the data manifest CSV files for Common Voice dataset
    (see https://commonvoice.mozilla.org/en/datasets).

    Parameters
    ----------
    locales:
        The dataset locales to download (e.g. "en", "it", etc.).
    download_dir:
        The path to the dataset download directory.
    version:
        The dataset version.
    max_duration:
        The maximum total duration in seconds to
        sample from each locale.
        Default to ``float("inf")``.
    shuffle:
        True to shuffle the data, False otherwise.
        Used only if `max_duration` is less than infinity.

    Raises
    ------
    ValueError
        If `locales` is empty.

    Examples
    --------
    >>> prepare_common_voice(["en", "it"], "data", "11.0-2022-09-21")

    """
    if not locales:
        raise ValueError(f"`locales` ({locales}) must be non-empty")

    for locale in locales:
        _LOGGER.log(
            logging.INFO,
            "----------------------------------------------------------------------",
        )
        _LOGGER.log(logging.INFO, f"Locale: {locale}")
        locale_dir = os.path.join(download_dir, locale)
        if not os.path.isdir(locale_dir):
            try:
                download_locale(locale, locale_dir, version)
            except Exception:
                if not os.listdir(download_dir):
                    shutil.rmtree(download_dir)
                raise
        else:
            _LOGGER.log(logging.INFO, "Data already downloaded")

    _LOGGER.log(
        logging.INFO,
        "----------------------------------------------------------------------",
    )
    _LOGGER.log(logging.INFO, f"Merging TSV files...")
    for split in _SPLITS:
        merge_tsv_files(
            [
                os.path.join(download_dir, locale, f"{split}.tsv")
                for locale in locales
            ],
            os.path.join(download_dir, f"{split}.tsv"),
            max_duration,
            shuffle,
        )

    _LOGGER.log(
        logging.INFO,
        "----------------------------------------------------------------------",
    )
    _LOGGER.log(logging.INFO, f"Creating data manifest CSV files...")
    for split in _SPLITS:
        preprocess_tsv_file(
            os.path.join(download_dir, f"{split}.tsv"),
            os.path.join(download_dir, f"{split}.csv"),
        )


def download_locale(
    locale: "str", download_dir: "str", version: "str",
) -> "None":
    """Download Common Voice dataset locale.

    Parameters
    ----------
    locale:
        The dataset locale to download.
    download_dir:
        The path to the dataset locale download directory.
    version:
        The dataset version.

    Raises
    ------
    RuntimeError
        If an error occurs while downloading the data.

    Examples
    --------
    >>> download_locale("en", os.path.join("data", "en"), "11.0-2022-09-21")

    """
    os.makedirs(download_dir)
    archive = os.path.join(download_dir, "tmp.tar.gz")
    url = _URL_TEMPLATE.replace("$version", version).replace("$locale", locale)
    try:
        _LOGGER.log(logging.INFO, "Downloading data...")
        with requests.get(url, stream=True) as response:
            total_size = int(response.headers.get("content-length", 0))
            chunk_size = 1024 * 1024
            progress_bar = tqdm(total=total_size, unit="B", unit_scale=True)
            with open(archive, "wb") as f:
                for data in response.iter_content(chunk_size):
                    progress_bar.update(len(data))
                    f.write(data)
                progress_bar.close()
        _LOGGER.log(logging.INFO, "Done!")

        _LOGGER.log(logging.INFO, "Extracting data...")
        with tarfile.open(archive) as tar:
            for member in tar.getmembers():
                name = os.path.basename(member.name)
                if name.endswith(".mp3"):
                    member.name = os.path.join(download_dir, "clips", name)
                    tar.extract(member)
                elif os.path.splitext(name)[0] in _SPLITS:
                    member.name = os.path.join(download_dir, name)
                    tar.extract(member)
        os.remove(archive)
        _LOGGER.log(logging.INFO, "Done!")

        _LOGGER.log(logging.INFO, "Computing clip durations...")
        for split in _SPLITS:
            input_tsv_file = os.path.join(download_dir, f"{split}.tsv")
            output_tsv_file = os.path.join(download_dir, f"tmp.tsv")
            with open(input_tsv_file) as fr, open(output_tsv_file, "w") as fw:
                tsv_reader = csv.reader(
                    fr, delimiter="\t", quoting=csv.QUOTE_NONE
                )
                tsv_writer = csv.writer(fw, delimiter="\t")
                header = next(tsv_reader)
                tsv_writer.writerow(header + ["duration"])
                for row in tsv_reader:
                    # Remove "\t" and "\"" to not confuse the TSV writer
                    for i in range(len(row)):
                        row[i] = row[i].replace("\t", " ")
                        row[i] = row[i].replace('"', "")

                    mp3 = row[1]
                    mp3 = os.path.join(download_dir, "clips", mp3)

                    # NOTE: info returns incorrect num_frames on torchaudio==0.12.x
                    info = torchaudio.info(mp3)
                    duration = info.num_frames / info.sample_rate

                    tsv_writer.writerow(row + [duration])
            shutil.move(output_tsv_file, input_tsv_file)
        _LOGGER.log(logging.INFO, "Done!")

    except Exception:
        shutil.rmtree(download_dir)
        raise RuntimeError(f"Could not download locale: {locale}")


def merge_tsv_files(
    input_tsv_files: "Sequence[str]",
    output_tsv_file: "str",
    max_duration: "Optional[float]" = None,
    shuffle: "bool" = False,
) -> "None":
    """Merge input TSV files into a single output TSV file.

    Parameters
    ----------
    input_tsv_files:
        The paths to the input TSV files.
    output_tsv_file:
        The path to the output TSV file.
    max_duration:
        The maximum total duration in seconds to
        sample from each TSV file.
        Default to ``float("inf")``.
    shuffle:
        True to shuffle the data, False otherwise.
        Used only if `max_duration` is less than infinity.

    Raises
    ------
    IndexError
        If `max_duration` is less than infinity and the
        number of rows in any of the TSV files is larger
        than the maximum allowed (2000000).

    Examples
    --------
    >>> merge_tsv_files(["data/en/test.tsv", "data/it/test.tsv"], "data/test.tsv")

    """
    if max_duration is None:
        max_duration = float("inf")
    _LOGGER.log(logging.INFO, f"Writing output TSV file ({output_tsv_file})...")
    os.makedirs(os.path.dirname(output_tsv_file), exist_ok=True)
    with open(output_tsv_file, "w") as fw:
        tsv_writer = csv.writer(
            fw, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\"
        )
        write_header = True
        for input_tsv_file in input_tsv_files:
            _LOGGER.log(
                logging.INFO, f"Reading input TSV file ({input_tsv_file})..."
            )
            with open(input_tsv_file) as fr:
                tsv_reader = csv.reader(fr, delimiter="\t")
                header = next(tsv_reader)
                if write_header:
                    tsv_writer.writerow(header)
                    write_header = False
                if max_duration == float("inf"):
                    for row in tsv_reader:
                        tsv_writer.writerow(row)
                    continue
                rows = list(tsv_reader)

            # Add rows until `max_duration` is reached
            random_idxes = (
                random.sample(_RANDOM_IDXES, len(_RANDOM_IDXES))
                if shuffle
                else _RANDOM_IDXES
            )
            duration, i, num_added_rows = 0.0, 0, 0
            while duration <= max_duration and num_added_rows < len(rows):
                try:
                    idx = random_idxes[i]
                except IndexError:
                    raise IndexError(
                        f"The number of rows ({len(rows) + 1}) in {input_tsv_file} "
                        f"must be in the integer interval [1, {len(random_idxes) + 1}]"
                    )
                i += 1
                try:
                    row = rows[idx]
                except IndexError:
                    continue
                duration += float(row[10])
                num_added_rows += 1
                tsv_writer.writerow(row)
            _LOGGER.log(
                logging.INFO, f"Total duration (s): {duration}",
            )
            _LOGGER.log(logging.INFO, f"Added {num_added_rows} rows")

    _LOGGER.log(logging.INFO, "Done!")


# Adapted from:
# https://github.com/speechbrain/speechbrain/blob/v0.5.13/recipes/CommonVoice/common_voice_prepare.py#L160
def preprocess_tsv_file(
    input_tsv_file: "str", output_csv_file: "str",
) -> "None":
    """Apply minimal Common Voice preprocessing (e.g. rename columns, remove unused columns,
    remove commas, special characters and empty sentences etc.) to each row of an input TSV file.

    Parameters
    ----------
    input_tsv_file:
        The path to the input TSV file.
    output_csv_file:
        The path to the output CSV file.

    Examples
    --------
    >>> preprocess_tsv_file("data/test.tsv", "data/test.csv")

    """
    # Header: client_id path sentence up_votes down_votes age gender accents locale segment duration
    _LOGGER.log(logging.INFO, f"Reading input TSV file ({input_tsv_file})...")
    _LOGGER.log(logging.INFO, f"Writing output CSV file ({output_csv_file})...")
    os.makedirs(os.path.dirname(output_csv_file), exist_ok=True)
    num_clips, total_duration = 0, 0.0
    with open(input_tsv_file) as fr, open(output_csv_file, "w") as fw:
        tsv_reader = csv.reader(fr, delimiter="\t", quoting=csv.QUOTE_NONE)
        csv_writer = csv.writer(fw)
        _ = next(tsv_reader)
        csv_writer.writerow(["ID", "mp3", "wrd", "locale", "duration"])
        for i, row in enumerate(tsv_reader):
            mp3, wrd, locale, duration = row[1], row[2], row[8], row[10]
            id_ = os.path.splitext(mp3)[0]
            mp3 = os.path.join("$data_root", locale, "clips", mp3)

            # Unicode normalization (default in Python 3)
            wrd = str(wrd)

            # Remove commas
            wrd = wrd.replace(",", " ")

            # Replace special characters used by SpeechBrain
            wrd = wrd.replace("$", "s")

            # Remove quotes
            wrd = wrd.replace("'", " ")
            wrd = wrd.replace("’", " ")
            wrd = wrd.replace("`", " ")
            wrd = wrd.replace('"', " ")

            # Remove multiple spaces
            wrd = re.sub(" +", " ", wrd)

            # Remove spaces at the beginning and the end of the sentence
            wrd = wrd.lstrip().rstrip()

            # Remove empty sentences
            if len(wrd) < 1:
                _LOGGER.log(
                    logging.DEBUG,
                    f"Sentence for row {i + 1} is too short, removing...",
                )
                continue

            num_clips += 1
            total_duration += float(duration)
            csv_writer.writerow([id_, mp3, wrd, locale, duration])

    with open(f"{output_csv_file}.stats", "w") as fw:
        fw.write(f"Number of samples: {num_clips}\n")
        fw.write(f"Total duration in seconds: {total_duration}")

    _LOGGER.log(logging.INFO, "Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Common Voice dataset")
    parser.add_argument(
        "-l",
        "--locales",
        nargs="+",
        default=["en"],
        help='dataset locales to download (e.g. "en", "it", etc.)',
    )
    parser.add_argument(
        "-d",
        "--download_dir",
        default="data",
        help="path to dataset download directory",
    )
    parser.add_argument(
        "-v", "--version", default="11.0-2022-09-21", help="dataset version",
    )
    parser.add_argument(
        "-m",
        "--max_duration",
        help="maximum total duration in seconds to sample from each locale",
    )
    parser.add_argument(
        "-s",
        "--shuffle",
        action="store_true",
        help="shuffle the data when `max_duration` is less than infinity",
    )

    args = parser.parse_args()
    prepare_common_voice(
        args.locales,
        args.download_dir,
        args.version,
        args.max_duration,
        args.shuffle,
    )
