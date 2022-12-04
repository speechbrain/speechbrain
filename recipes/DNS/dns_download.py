#!/usr/bin/env/python3
"""
Recipe for downloading DNS dataset- training,
baseline DEV noisyset, blind testset
Source:
https://github.com/microsoft/DNS-Challenge
https://github.com/microsoft/DNS-Challenge/blob/master/download-dns-challenge-4.sh

Disk-space (compressed): 500 GB
Disk-space (decompressed): 1 TB

NOTE:
    Some of the azure links provided by Microsoft are not perfect and data download
    may stop mid-way through the download process. Hence we validate download size
    of each of the file.
Authors
    * Sangeet Sagar 2022
"""

import os
import urllib.request
import shutil
import tarfile
from tqdm.auto import tqdm

BLOB_NAMES = [
    "clean_fullband/datasets_fullband.clean_fullband.VocalSet_48kHz_mono_000_NA_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.emotional_speech_000_NA_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.french_speech_000_NA_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.french_speech_001_NA_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.french_speech_002_NA_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.french_speech_003_NA_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.french_speech_004_NA_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.french_speech_005_NA_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.french_speech_006_NA_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.french_speech_007_NA_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.french_speech_008_NA_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.german_speech_000_0.00_3.47.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.german_speech_001_3.47_3.64.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.german_speech_002_3.64_3.74.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.german_speech_003_3.74_3.81.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.german_speech_004_3.81_3.86.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.german_speech_005_3.86_3.91.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.german_speech_006_3.91_3.96.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.german_speech_007_3.96_4.00.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.german_speech_008_4.00_4.04.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.german_speech_009_4.04_4.08.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.german_speech_010_4.08_4.12.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.german_speech_011_4.12_4.16.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.german_speech_012_4.16_4.21.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.german_speech_013_4.21_4.26.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.german_speech_014_4.26_4.33.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.german_speech_015_4.33_4.43.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.german_speech_016_4.43_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.german_speech_017_NA_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.german_speech_018_NA_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.german_speech_019_NA_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.german_speech_020_NA_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.german_speech_021_NA_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.german_speech_022_NA_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.german_speech_023_NA_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.german_speech_024_NA_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.german_speech_025_NA_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.german_speech_026_NA_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.german_speech_027_NA_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.german_speech_028_NA_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.german_speech_029_NA_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.german_speech_030_NA_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.german_speech_031_NA_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.german_speech_032_NA_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.german_speech_033_NA_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.german_speech_034_NA_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.german_speech_035_NA_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.german_speech_036_NA_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.german_speech_037_NA_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.german_speech_038_NA_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.german_speech_039_NA_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.german_speech_040_NA_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.german_speech_041_NA_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.german_speech_042_NA_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.italian_speech_000_0.00_3.98.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.italian_speech_001_3.98_4.21.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.italian_speech_002_4.21_4.40.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.italian_speech_003_4.40_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.italian_speech_004_NA_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.italian_speech_005_NA_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_000_0.00_3.75.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_001_3.75_3.88.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_002_3.88_3.96.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_003_3.96_4.02.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_004_4.02_4.06.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_005_4.06_4.10.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_006_4.10_4.13.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_007_4.13_4.16.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_008_4.16_4.19.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_009_4.19_4.21.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_010_4.21_4.24.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_011_4.24_4.26.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_012_4.26_4.29.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_013_4.29_4.31.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_014_4.31_4.33.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_015_4.33_4.35.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_016_4.35_4.38.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_017_4.38_4.40.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_018_4.40_4.42.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_019_4.42_4.45.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_020_4.45_4.48.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_021_4.48_4.52.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_022_4.52_4.57.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_023_4.57_4.67.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_024_4.67_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_025_NA_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_026_NA_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_027_NA_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_028_NA_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_029_NA_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_030_NA_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_031_NA_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_032_NA_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_033_NA_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_034_NA_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_035_NA_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_036_NA_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_037_NA_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_038_NA_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.read_speech_039_NA_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.russian_speech_000_0.00_4.31.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.russian_speech_001_4.31_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.spanish_speech_000_0.00_4.09.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.spanish_speech_001_4.09_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.spanish_speech_002_NA_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.spanish_speech_003_NA_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.spanish_speech_004_NA_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.spanish_speech_005_NA_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.spanish_speech_006_NA_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.spanish_speech_007_NA_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.spanish_speech_008_NA_NA.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.vctk_wav48_silence_trimmed_000.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.vctk_wav48_silence_trimmed_001.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.vctk_wav48_silence_trimmed_002.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.vctk_wav48_silence_trimmed_003.tar.bz2",
    "clean_fullband/datasets_fullband.clean_fullband.vctk_wav48_silence_trimmed_004.tar.bz2",
    "noise_fullband/datasets_fullband.noise_fullband.audioset_000.tar.bz2",
    "noise_fullband/datasets_fullband.noise_fullband.audioset_001.tar.bz2",
    "noise_fullband/datasets_fullband.noise_fullband.audioset_002.tar.bz2",
    "noise_fullband/datasets_fullband.noise_fullband.audioset_003.tar.bz2",
    "noise_fullband/datasets_fullband.noise_fullband.audioset_004.tar.bz2",
    "noise_fullband/datasets_fullband.noise_fullband.audioset_005.tar.bz2",
    "noise_fullband/datasets_fullband.noise_fullband.audioset_006.tar.bz2",
    "noise_fullband/datasets_fullband.noise_fullband.freesound_000.tar.bz2",
    "noise_fullband/datasets_fullband.noise_fullband.freesound_001.tar.bz2",
    "datasets_fullband.dev_testset_000.tar.bz2",
    "datasets_fullband.impulse_responses_000.tar.bz2",
]

AZURE_URL = (
    "https://dns4public.blob.core.windows.net/dns4archive/datasets_fullband"
)
BLIND_TESTSET_URL = "https://dns4public.blob.core.windows.net/dns4archive/blind_testset_bothtracks.zip"

COMPRESSED_PATH = "DNS-compressed"
DECOMPRESSED_PATH = "DNS-dataset"

SPLIT_LIST = [
    "dev_testset",
    "impulse_responses",
    "noise_fullband",
    "emotional_speech",
    "french_speech",
    "german_speech",
    "italian_speech",
    "read_speech",
    "russian_speech",
    "spanish_speech",
    "vctk_wav48_silence_trimmed",
    "VocalSet_48kHz_mono",
]


def prepare_download():
    # Real-time DNS track (train set + dev noisy set)
    for file_url in BLOB_NAMES:

        for split in SPLIT_LIST:
            if split in file_url:
                split_name = split

        split_path = os.path.join(COMPRESSED_PATH, split_name)
        if not os.path.exists(split_path):
            os.makedirs(split_path)
        if not os.path.exists(DECOMPRESSED_PATH):
            os.makedirs(DECOMPRESSED_PATH)

        filename = file_url.split("/")[-1]
        download_path = os.path.join(split_path, filename)
        download_url = AZURE_URL + "/" + file_url

        if not validate_file(download_url, download_path):
            download_file(download_url, download_path, split_name, filename)
            decompress_file(download_path, DECOMPRESSED_PATH)
        else:
            print(", \tDownload complete. Skipping")
            decompress_file(download_path, DECOMPRESSED_PATH)

    # BLIND testset
    download_url = BLIND_TESTSET_URL
    download_path = os.path.join(
        COMPRESSED_PATH, BLIND_TESTSET_URL.split("/")[-1]
    )
    if not validate_file(download_url, download_path):
        download_sucessful = download_file(
            download_url,
            download_path,
            "blind_testset",
            BLIND_TESTSET_URL.split("/")[-1],
        )
        if download_sucessful:
            print("Decompressing...")
            shutil.unpack_archive(download_path, DECOMPRESSED_PATH, "zip")
    else:
        print(", \tDownload complete. Skipping")
        print("Decompressing...")
        shutil.unpack_archive(download_path, DECOMPRESSED_PATH, "zip")


def download_file(download_url, download_path, split_name, filename):
    """
    Download file from given URL

    Arguments
    ---------
    download_url : str
        URL of the being downloaded
    download_path : str
        Full path of the file that is to be downloaded
        (or already downloaded)
    split_name : str
        Split name of the file being downloaded
        e.g. read_speech
    filename : str
        Fielname of the file being downloaded

    Returns
    -------
    bool
        If True, the file need not be downloaded again.
        Else re-download it.
    """
    print("\nDownloading:", split_name, "=>", filename)
    with urllib.request.urlopen(download_url) as response:
        # Get content length
        total_length = int(response.headers.get("Content-Length"))
        with tqdm.wrapattr(
            response, "read", total=total_length, desc=""
        ) as raw:
            with open(download_path, "wb") as tmp_file:
                shutil.copyfileobj(raw, tmp_file)

    # Validate downloaded file
    if validate_file(download_url, download_path):
        return True
    else:
        print("Download failed. Moving on.")
        return False


def parallel_download():
    pass


def decompress_file(file, decompress_path):
    """
    Decompress the downloaded file

    Arguments
    ---------
    file : str
        Path to the compressed donwloaded file
    decompress_path : str
        Path to store the decompressed audio files
    """
    print("Decompressing...")
    tar = tarfile.open(file, "r:bz2")
    tar.extractall(decompress_path)
    tar.close()
    # os.remove(file)


def validate_file(download_url, download_path):
    """
    Write data to CSV file in an appropriate format.

    Arguments
    ---------
    download_url : str
        URL of the being downloaded
    download_path : str
        full path of the file that is to be downloaded
        (or already downloaded)

    Returns
    -------
    bool
        If True, the file need not be downloaded again.
        Else, either the file is not yet downloaded or
        partially downloded, thus re-download it.
    """
    if not os.path.isfile(download_path):
        # File not yet downloaded
        return False
    # Get file size in MB
    actual_size = int(
        urllib.request.urlopen(download_url).length / (1024 * 1024)
    )
    download_size = int(os.path.getsize(download_path) / (1024 * 1024))

    print(
        "File: {}, \t downloaded {} MB out of {} MB".format(
            download_path.split("/")[-1], download_size, actual_size
        ),
        end="",
    )
    # Set a margin of 100 MB. We skip re-downloading the file if downloaded
    # size differs from actual size by max 100 MB. More than this margin,
    # re-download is to attempted.
    if abs(download_size - actual_size) < 100:
        return True
    else:
        print(", \tIncomplete download. Re-trying")
        return False


if __name__ == "__main__":
    prepare_download()
