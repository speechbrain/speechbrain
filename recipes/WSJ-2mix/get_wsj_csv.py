import pandas as pd
import os
import argparse

"""
Cem Subakan, 2020, Mila,

This script generates the csv files required by speechbrain to train on WSJ-2Mix-8k-min dataset
"""
parser = argparse.ArgumentParser()
parser.add_argument(
    "--path", help="the path for the wsj_data",
)
parser.add_argument("--set_type", help="in [cv, tr, tt]", default="tr")

args = parser.parse_args()

mix_path = args.path + "wsj0-mix/2speakers/wav8k/min/" + args.set_type + "/mix/"
s1_path = args.path + "wsj0-mix/2speakers/wav8k/min/" + args.set_type + "/s1/"
s2_path = args.path + "wsj0-mix/2speakers/wav8k/min/" + args.set_type + "/s2/"

files = os.listdir(mix_path)

mix_file_paths = [mix_path + fl for fl in files]
s1_file_paths = [s1_path + fl for fl in files]
s2_file_paths = [s2_path + fl for fl in files]

df = pd.DataFrame(
    {
        "ID": list(range(len(mix_file_paths))),
        "duration": [1.0] * len(mix_file_paths),
        "mix_wav": mix_file_paths,
        "mix_wav_format": ["wav"] * len(mix_file_paths),
        "mix_wav_opts": [None] * len(mix_file_paths),
        "s1_wav": s1_file_paths,
        "s1_wav_format": ["wav"] * len(mix_file_paths),
        "s1_wav_opts": [None] * len(mix_file_paths),
        "s2_wav": s2_file_paths,
        "s2_wav_format": ["wav"] * len(mix_file_paths),
        "s2_wav_opts": [None] * len(mix_file_paths),
    }
)

df.to_csv(
    "wsj_" + args.set_type + ".csv", index=False, index_label="index",
)
