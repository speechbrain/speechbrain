import argparse
import os
from speechbrain.utils.data_utils import download_file

LIBRISPEECH_URLS = ["http://www.openslr.org/resources/12/test-clean.tar.gz"]

OPENRIR_URL = "http://www.openslr.org/resources/28/rirs_noises.zip"
QUT_TIMIT_URL = ""

parser = argparse.ArgumentParser(
    "Python script to download required recipe data"
)
parser.add_argument("--output_folder", type=str, required=True)
parser.add_argument("--stage", type=int, default=0)
args = parser.parse_args()

# output folder will be:
# LibriSpeech/
# QUT-TIMIT/
# rirs_noises/

# stage 0 download librispeech
if args.stage <= 0:
    print("Stage 0: Downloading LibriSpeech")
    for url in LIBRISPEECH_URLS:
        name = url.split("/")[-1]
        download_file(url, os.path.join(args.output_folder, name), unpack=True)

# stage 1 download rirs and noises
if args.stage <= 1:
    print("Stage 1: Downloading RIRs and Noises")
    name = url.split("/")[-1]
    download_file(
        OPENRIR_URL, os.path.join(args.output_folder, name), unpack=True
    )


if args.stage <= 2:
    print("Stage 2: Downloading QUT-TIMIT background noises")
    name = url.split("/")[-1]
    download_file(
        QUT_TIMIT_URL, os.path.join(args.output_folder, name), unpack=True
    )

if args.stage <= 3:
    print("Stage 3: Resampling QUT-TIMIT background noises")

    print(
        "Resampling done. Original QUT-TIMIT files can be deleted "
        "as they will not be used."
    )
