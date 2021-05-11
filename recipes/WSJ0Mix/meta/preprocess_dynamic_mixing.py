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
import torchaudio
import glob

# from oct2py import octave
from scipy import signal
import numpy as np
import torch


parser = argparse.ArgumentParser(
    "utility for resampling all audio files in a folder recursively"
    "It --input_folder to --output_folder and "
    "resamples all audio files with specified format to --fs."
)
parser.add_argument("--input_folder", type=str, required=True)
parser.add_argument("--output_folder", type=str, required=True)
parser.add_argument(
    "--fs", type=str, default=8000, help="this is the target sampling frequency"
)
parser.add_argument("--regex", type=str, default="**/*.wav")


def resample_folder(input_folder, output_folder, fs, regex):
    """Resamples the wav files within an input folder.

    Arguments
    ---------
    input_folder : path
        Path of the folder to resample.
    output_folder : path
        Path of the output folder with the resampled data.
    fs : int
        Target sampling frequency.
    reg_exp: str
        Regular expression for search.
    """
    # filedir = os.path.dirname(os.path.realpath(__file__))
    # octave.addpath(filedir)
    # add the matlab functions to octave dir here

    files = glob.glob(os.path.join(input_folder, regex), recursive=True)
    for f in tqdm.tqdm(files):

        audio, fs_read = torchaudio.load(f)
        audio = audio[0].numpy()
        audio = signal.resample_poly(audio, fs, fs_read)

        # tmp = octave.activlev(audio.tolist(), fs, "n")
        # audio, _ = tmp[:-1].squeeze(), tmp[-1]

        peak = np.max(np.abs(audio))
        audio = audio / peak
        audio = torch.from_numpy(audio).float()

        relative_path = os.path.join(
            Path(f).relative_to(Path(input_folder)).parent,
            Path(f).relative_to(Path(input_folder)).stem
            + "_peak_{}.wav".format(peak),
        )

        os.makedirs(
            Path(
                os.path.join(
                    output_folder, Path(f).relative_to(Path(input_folder))
                )
            ).parent,
            exist_ok=True,
        )

        torchaudio.save(
            os.path.join(output_folder, relative_path),
            audio.reshape(1, -1),
            fs,
        )


if __name__ == "__main__":

    args = parser.parse_args()
    resample_folder(
        args.input_folder, args.output_folder, int(args.fs), args.regex
    )
