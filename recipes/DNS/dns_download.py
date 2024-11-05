#!/usr/bin/env/python3
"""
Recipe for downloading DNS-4 dataset- training,
baseline DEV noisyset, blind testset
Source:
https://github.com/microsoft/DNS-Challenge
https://github.com/microsoft/DNS-Challenge/blob/master/download-dns-challenge-4.sh

Disk-space (compressed): 550 GB
Disk-space (decompressed): 1 TB

NOTE:
    1. Some of the azure links provided by Microsoft are not perfect and data
    download may stop mid-way through the download process. Hence we validate
    download size of each of the file.
    2. Instead of using the impulse response files provided in the challenge,
    we opt to download them from OPENSLR. OPENSLR offers both real and synthetic
    RIRs, while the challenge offers only real RIRs.

Authors
    * Sangeet Sagar 2022
"""

import argparse
import fileinput
import os
import shutil
import ssl
import tarfile
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

import certifi
import requests
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
]

AZURE_URL = "https://dns4public.blob.core.windows.net/dns4archive/datasets_fullband"  # noqa ignore-url-check

# Impulse response and Blind testset
OTHER_URLS = {
    "impulse_responses": [
        "https://www.openslr.org/resources/26/sim_rir_16k.zip",
        "https://www.openslr.org/resources/28/rirs_noises.zip",
    ],
    "blind_testset": [
        "https://dns4public.blob.core.windows.net/dns4archive/blind_testset_bothtracks.zip"
    ],
}

RIR_table_simple_URL = "https://raw.githubusercontent.com/microsoft/DNS-Challenge/0443a12f5e6e7bec310f453cf0d9637ca28e0eea/datasets/acoustic_params/RIR_table_simple.csv"

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
    """
    Downloads and prepares various data files and resources. It
    downloads real-time DNS track data files (train set and dev
    noisy set).
    """
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
            if os.path.exists(download_path):
                resume_byte_pos = os.path.getsize(download_path)
            else:
                resume_byte_pos = None

            download_file(
                download_url,
                download_path,
                split_name,
                filename,
                resume_byte_pos=resume_byte_pos,
            )
        else:
            print(", \tDownload complete. Skipping")
        decompress_file(download_path, DECOMPRESSED_PATH, split_name)

    # Download RIR (impulse response) & BLIND testset
    rir_blind_test_download()


def rir_blind_test_download():
    """
    Download the RIRs (room impulse responses), and the blind
    test set.
    """
    # RIR (impulse response) & BLIND testset
    for split_name, download_urls in OTHER_URLS.items():
        for file_url in download_urls:
            split_path = os.path.join(COMPRESSED_PATH, split_name)
            if not os.path.exists(split_path):
                os.makedirs(split_path)

            filename = file_url.split("/")[-1]
            download_path = os.path.join(split_path, filename)

            if not validate_file(file_url, download_path):
                if os.path.exists(download_path):
                    resume_byte_pos = os.path.getsize(download_path)
                else:
                    resume_byte_pos = None

                download_file(
                    file_url,
                    download_path,
                    split_name,
                    filename,
                    resume_byte_pos=resume_byte_pos,
                )
            else:
                print(", \tDownload complete. Skipping")
            decompress_file(
                download_path,
                os.path.join(DECOMPRESSED_PATH, split_name),
                split_name,
            )

    # Download RIRs simple table
    file_path = os.path.join(
        DECOMPRESSED_PATH, "impulse_responses", "RIR_table_simple.csv"
    )
    response = requests.get(RIR_table_simple_URL)
    if response.status_code == 200:
        with open(file_path, "wb") as file:
            file.write(response.content)
        print("\nRIR_simple_table downloaded successfully.")

    else:
        print(
            f"\nFailed to download RIR_simple_table. Status code: {response.status_code}"
        )


def download_file(
    download_url, download_path, split_name, filename, resume_byte_pos=None
):
    """
    Download file from given URL

    Arguments
    ---------
    download_url : str
        URL of file being downloaded
    download_path : str
        Full path of the file that is to be downloaded
        (or already downloaded)
    split_name : str
        Split name of the file being downloaded
        e.g. read_speech
    filename : str
        Filename of the file being downloaded
    resume_byte_pos: (int, optional)
        Starting byte position for resuming the download.
        Default is None, which means a fresh download.

    Returns
    -------
    bool
        If True, the file need not be downloaded again.
        Else the download might have failed or is incomplete.
    """
    print("Downloading:", split_name, "=>", filename)
    resume_header = (
        {"Range": f"bytes={resume_byte_pos}-"} if resume_byte_pos else None
    )
    response = requests.get(download_url, headers=resume_header, stream=True)
    file_size = int(response.headers.get("Content-Length"))

    mode = "ab" if resume_byte_pos else "wb"
    initial_pos = resume_byte_pos if resume_byte_pos else 0

    with open(download_path, mode, encoding="utf-8") as f:
        with tqdm(
            total=file_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            initial=initial_pos,
            miniters=1,
        ) as pbar:
            for chunk in response.iter_content(32 * 1024):
                f.write(chunk)
                pbar.update(len(chunk))

    # Validate downloaded file
    if validate_file(download_url, download_path):
        return True
    else:
        print("Download failed. Moving on.")
        return False


def download_file_parallel(args):
    """
    Downloads a file in parallel using the provided arguments. It
    makes use of `download_file` function to download the required file.

    Arguments
    ---------
    args : tuple
        Tuple containing the download URL, download path, split
        name, filename, and required bytes to be downloaded.
    """
    download_url, download_path, split_name, filename, resume_byte_pos = args
    download_file(
        download_url,
        download_path,
        split_name,
        filename,
        resume_byte_pos=resume_byte_pos,
    )


def parallel_download():
    """
    Perform parallel download of files using `using ThreadPoolExecutor`.
    """
    with ThreadPoolExecutor() as executor:
        futures = []
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
                if os.path.exists(download_path):
                    resume_byte_pos = os.path.getsize(download_path)
                else:
                    resume_byte_pos = None
                args = (
                    download_url,
                    download_path,
                    split_name,
                    filename,
                    resume_byte_pos,
                )
                futures.append(executor.submit(download_file_parallel, args))
                # download_file(download_url, download_path, split_name, filename)
                # decompress_file(download_path, DECOMPRESSED_PATH)
            else:
                print(", \tDownload complete. Skipping")
                decompress_file(download_path, DECOMPRESSED_PATH, split_name)

        for future in futures:
            future.result()

    # Download RIR (impulse response) & BLIND testset
    rir_blind_test_download()


def decompress_file(file, decompress_path, split_name):
    """
    Decompress the downloaded file if the target folder does not exist.

    Arguments
    ---------
    file : str
        Path to the compressed downloaded file
    decompress_path : str
        Path to store the decompressed audio files
    split_name : str
        The portion of the data to decompress

    Returns
    -------
    True if decompression skipped.
    """
    for _, dirs, _ in os.walk(decompress_path):
        if split_name in dirs:
            print("\tDecompression skipped. Folder already exists.")
            return True

    if "sim_rir_16k" in file:
        slr26_dir = os.path.join(decompress_path, "SLR26")
        if os.path.exists(slr26_dir):
            print("\tDecompression skipped. Folder already exists.")
            return True

    if "rirs_noises" in file:
        slr28_dir = os.path.join(decompress_path, "SLR28")
        if os.path.exists(slr28_dir):
            print("\tDecompression skipped. Folder already exists.")
            return True

    print("\tDecompressing...")
    file_extension = os.path.splitext(file)[-1].lower()
    if file_extension == ".zip":
        zip = zipfile.ZipFile(file, "r")
        zip.extractall(decompress_path)
        rename_rirs(decompress_path)

    elif file_extension == ".bz2":
        tar = tarfile.open(file, "r:bz2")
        tar.extractall(decompress_path)
        tar.close()
    else:
        print("Unsupported file format. Only zip and bz2 files are supported.")
    # os.remove(file)


def rename_rirs(decompress_path):
    """
    Rename directories containing simulated room impulse responses
    (RIRs).

    Arguments
    ---------
        decompress_path (str): The path to the directory containing the RIRs

    Returns
    -------
        None
    """
    try:
        os.rename(
            os.path.join(decompress_path, "simulated_rirs_16k"),
            os.path.join(decompress_path, "SLR26"),
        )
    except Exception:
        pass
    try:
        os.rename(
            os.path.join(decompress_path, "RIRS_NOISES"),
            os.path.join(decompress_path, "SLR28"),
        )
    except Exception:
        pass


def validate_file(download_url, download_path):
    """
    Validate the downloaded file and resume the download if needed.

    Arguments
    ---------
    download_url : str
        URL of the file being downloaded
    download_path : str
        Full path of the file that is to be downloaded
        (or already downloaded)

    Returns
    -------
    bool
        If True, the file need not be downloaded again.
        Else, either the file is not yet downloaded or
        partially downloaded, thus resume the download.
    """
    if not os.path.isfile(download_path):
        # File not yet downloaded
        return False

    # Get file size in MB
    actual_size = urllib.request.urlopen(
        download_url,
        context=ssl.create_default_context(cafile=certifi.where()),
    ).length

    download_size = os.path.getsize(download_path)

    print(
        "File: {}, \t downloaded {} MB out of {} MB".format(
            download_path.split("/")[-1],
            download_size // (1024 * 1024),
            actual_size // (1024 * 1024),
        ),
        end="",
    )
    # Set a margin of 100 MB. We skip re-downloading the file if downloaded
    # size differs from actual size by max 100 MB. More than this margin,
    # re-download is to attempted.
    if actual_size - download_size < 100 * 1024 * 1024:
        return True
    else:
        print(", \tIncomplete download. Resuming...")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and extract DNS dataset."
    )
    parser.add_argument(
        "--compressed_path",
        type=str,
        default="DNS-compressed",
        help="Path to store the compressed data.",
    )
    parser.add_argument(
        "--decompressed_path",
        type=str,
        default="DNS-dataset",
        help="Path to store the decompressed data.",
    )

    parser.add_argument(
        "--parallel_download",
        action="store_true",
        help="Use parallel download.",
    )

    args = parser.parse_args()

    COMPRESSED_PATH = args.compressed_path
    DECOMPRESSED_PATH = args.decompressed_path

    if args.parallel_download:
        parallel_download()
    else:
        prepare_download()

    # Modify contents inside RIR_simple_table.csv
    file_path = os.path.join(
        DECOMPRESSED_PATH, "impulse_responses", "RIR_table_simple.csv"
    )
    full_path = os.path.abspath(os.path.dirname(file_path))

    replacements = {
        "datasets/impulse_responses/SLR26/simulated_rirs_16k": os.path.join(
            full_path, "SLR26"
        ),
        "datasets/impulse_responses/SLR28/RIRS_NOISES": os.path.join(
            full_path, "SLR28"
        ),
    }

    # Perform the replacements directly in the file using fileinput module
    with fileinput.FileInput(file_path, inplace=True) as file:
        for line in file:
            for original, replacement in replacements.items():
                line = line.replace(original, replacement)
            print(line, end="")

    if not os.path.exists(
        os.path.join("noisyspeech_synthesizer", "RIR_table_simple.csv")
    ):
        shutil.move(file_path, "noisyspeech_synthesizer")
