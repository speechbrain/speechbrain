""" This script is used to generate a new folder containing the audio files
of the dev and test manifest of yodas (the yodas preparation must be done before). It will also append a number before the audio ID to make sure that
audio files appear sorted (aligned with the csv) for faster listening.

e.g.
python yodas_dev_test_converter.py /source/dev.csv /output_folder
"""

import os
import shutil
import sys

source_csv = sys.argv[1]
output_path = sys.argv[2]

os.makedirs(output_path, exist_ok=True)

source_content = open(source_csv, mode="r", encoding="utf-8").readlines()[1:]

total_duration = 0.0
for cpt, line in enumerate(source_content):
    ID, duration, start, wav, spk_id, sex, text = line.split("\n")[0].split(",")
    total_duration += float(duration)
    filename = str(cpt) + "_" + wav.split("/")[-1]
    shutil.copyfile(wav, os.path.join(output_path, filename))

print("Duration: " + str(total_duration))

shutil.copyfile(source_csv, os.path.join(output_path, "yodas_csv.csv"))
