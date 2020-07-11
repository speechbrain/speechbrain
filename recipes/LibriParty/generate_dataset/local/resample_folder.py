import os
import argparse
import glob
from pathlib import Path
import tqdm
import torchaudio
import torch

parser = argparse.ArgumentParser(
    "utility for resampling all audio files in a folder recursively. "
    "It --input_folder to --output_folder and "
    "resamples all audio files with specified format to --fs."
)
parser.add_argument("--input_folder", type=str, required=True)
parser.add_argument("--output_folder", type=str, required=True)
parser.add_argument("--fs", type=str, default=16000)
parser.add_argument("--regex", type=str, default="*.wav")

args = parser.parse_args()
files = glob.glob(
    os.path.join(args.input_folder, "**/*{}".format(args.regex)), recursive=True
)


for f in tqdm.tqdm(files):
    audio, orig_fs = torchaudio.load(f)
    resampler = torchaudio.transforms.Resample(orig_fs, args.fs)
    audio = resampler(audio)
    audio = (
        audio / torch.max(torch.abs(audio), dim=-1, keepdim=True)[0]
    )  # scale back otherwise you get empty .wav file
    os.makedirs(
        Path(
            os.path.join(
                args.output_folder, Path(f).relative_to(Path(args.input_folder))
            )
        ).parent,
        exist_ok=True,
    )
    torchaudio.save(
        os.path.join(
            args.output_folder, Path(f).relative_to(Path(args.input_folder))
        ),
        audio,
        args.fs,
    )
