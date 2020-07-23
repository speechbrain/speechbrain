"""
Source datasets downloading script for LibriParty.

Author
------
Samuele Cornell, 2020
"""

import argparse
import os
from speechbrain.utils.data_utils import download_file
from local.resample_folder import resample_folder

LIBRISPEECH_URLS = [
    "http://www.openslr.org/resources/12/test-clean.tar.gz",
    "http://www.openslr.org/resources/12/dev-clean.tar.gz",
    "http://www.openslr.org/resources/12/train-clean-100.tar.gz",
]

OPENRIR_URL = "http://www.openslr.org/resources/28/rirs_noises.zip"
QUT_TIMIT_URLS = [
    "https://data.researchdatafinder.qut.edu.au/dataset/a0eed5af-abd8-441b-b14a-8e064bc3d732/resource/9b0f10ed-e3f5-40e7-b503-73c2943abfb1/download/qutnoisecafe.zip",
    "https://data.researchdatafinder.qut.edu.au/dataset/a0eed5af-abd8-441b-b14a-8e064bc3d732/resource/7412452a-92e9-4612-9d9a-6b00f167dc15/download/qutnoisecar.zip",
    "https://data.researchdatafinder.qut.edu.au/dataset/a0eed5af-abd8-441b-b14a-8e064bc3d732/resource/35cd737a-e6ad-4173-9aee-a1768e864532/download/qutnoisehome.zip",
    "https://data.researchdatafinder.qut.edu.au/dataset/a0eed5af-abd8-441b-b14a-8e064bc3d732/resource/164d38a5-c08e-4e20-8272-793534eb10c7/download/qutnoisereverb.zip",
    "https://data.researchdatafinder.qut.edu.au/dataset/a0eed5af-abd8-441b-b14a-8e064bc3d732/resource/10eeceae-9f0c-4556-b33a-dcf35c4f4db9/download/qutnoisestreet.zip",
]

parser = argparse.ArgumentParser(
    "Python script to download required recipe data"
)
parser.add_argument("--output_folder", type=str, required=True)
parser.add_argument("--stage", type=int, default=0)
args = parser.parse_args()

# output folder will be:
# LibriSpeech/
# QUT-NOISE/
# rirs_noises/

os.makedirs(args.output_folder, exist_ok=True)

if args.stage <= 0:
    print("Stage 0: Downloading LibriSpeech")
    for url in LIBRISPEECH_URLS:
        name = url.split("/")[-1]
        download_file(url, os.path.join(args.output_folder, name), unpack=True)

if args.stage <= 1:
    print("Stage 1: Downloading RIRs and Noises")
    name = OPENRIR_URL.split("/")[-1]
    download_file(
        OPENRIR_URL, os.path.join(args.output_folder, name), unpack=True
    )

if args.stage <= 2:
    print("Stage 2: Downloading QUT-TIMIT background noises")
    for url in QUT_TIMIT_URLS:
        name = "QUT_NOISE"
        os.makedirs(os.path.join(args.output_folder, name), exist_ok=True)
        download_file(
            url,
            os.path.join(args.output_folder, name, url.split("/")[-1]),
            unpack=True,
        )

if args.stage <= 3:
    print("Stage 3: Resampling QUT noise background noises")
    if os.path.exists(os.path.join(args.output_folder, "QUT_NOISE_16kHz")):
        print("Output folder already exists, skipping resampling.")
    else:
        resample_folder(
            os.path.join(args.output_folder, "QUT_NOISE"),
            os.path.join(args.output_folder, "QUT_NOISE_16kHz"),
            16000,
            ".wav",
        )
        print(
            "Resampling done. Original QUT noise files can be deleted "
            "as they will not be used."
        )
