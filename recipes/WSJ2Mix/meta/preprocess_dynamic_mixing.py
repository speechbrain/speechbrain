"""
This script allows to resample a folder which contains audio files.
The files are parsed recursively. An exact copy of the folder is created,
with same structure but contained resampled audio files.
Resampling is performed by using sox through torchaudio.
Author
------
Samuele Cornell, 2020
"""

import os
import argparse
from pathlib import Path
import tqdm
import soundfile as sf
import glob
from oct2py import octave
from scipy import signal

parser = argparse.ArgumentParser(
    "utility for resampling all audio files in a folder recursively"
    "It --input_folder to --output_folder and "
    "resamples all audio files with specified format to --fs."
)
parser.add_argument("--input_folder", type=str, required=True)
parser.add_argument("--output_folder", type=str, required=True)
parser.add_argument("--fs", type=str, default=16000)
parser.add_argument("--regex", type=str, default="**/*.wav")


def resample_folder(input_folder, output_folder, fs, regex):
    filedir = os.path.dirname(os.path.realpath(__file__))
    octave.addpath(filedir)
    # add the matlab functions to octave dir here

    files = glob.glob(os.path.join(input_folder, regex), recursive=True)
    for f in tqdm.tqdm(files):

        audio, fs_read = sf.read(f)
        audio = signal.resample(audio, int((fs_read / fs) * len(audio)))

        tmp = octave.activlev(audio.tolist(), fs_read, "n")
        audio, _ = tmp[:-1].squeeze(), tmp[-1]


        os.makedirs(
            Path(
                os.path.join(
                    output_folder, Path(f).relative_to(Path(input_folder))
                )
            ).parent,
            exist_ok=True,
        )

        sf.write(
            os.path.join(
                output_folder, Path(f).relative_to(Path(input_folder))
            ), audio, format="WAV", subtype="FLOAT", samplerate=fs)



if __name__ == "__main__":

    args = parser.parse_args()
    resample_folder(
        args.input_folder, args.output_folder, int(args.fs), args.regex
    )
