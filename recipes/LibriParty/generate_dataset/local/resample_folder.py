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
import torch
from speechbrain.utils.data_utils import get_all_files

parser = argparse.ArgumentParser(
    "utility for resampling all audio files in a folder recursively. "
    "It --input_folder to --output_folder and "
    "resamples all audio files with specified format to --fs."
)
parser.add_argument("--input_folder", type=str, required=True)
parser.add_argument("--output_folder", type=str, required=True)
parser.add_argument("--fs", type=str, default=16000)
parser.add_argument("--regex", type=str, default="*.wav")


def resample_folder(input_folder, output_folder, fs, regex):

    files = get_all_files(input_folder, match_and=[regex])
    torchaudio.initialize_sox()
    for f in tqdm.tqdm(files):

        # we use sox because torchaudio.Resample uses too much RAM.
        resample = torchaudio.sox_effects.SoxEffectsChain()
        resample.append_effect_to_chain("rate", [fs])
        resample.set_input_file(f)

        audio, fs = resample.sox_build_flow_effects()

        audio = (
            audio / torch.max(torch.abs(audio), dim=-1, keepdim=True)[0]
        )  # scale back otherwise you get empty .wav file
        os.makedirs(
            Path(
                os.path.join(
                    output_folder, Path(f).relative_to(Path(input_folder))
                )
            ).parent,
            exist_ok=True,
        )
        torchaudio.save(
            os.path.join(
                output_folder, Path(f).relative_to(Path(input_folder))
            ),
            audio,
            fs,
        )
    torchaudio.shutdown_sox()


if __name__ == "__main__":

    args = parser.parse_args()
    resample_folder(args.input_folder, args.output_folder, args.fs, args.regex)
