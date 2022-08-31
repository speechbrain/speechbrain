#!/usr/bin/env python3
"""
Tamasheq-French data processing.

Author
------
Marcely Zanon Boito 2022
"""

import json
import os


def write_json(json_file_name, data):
    with open(json_file_name, mode="w", encoding="utf-8") as output_file:
        json.dump(
            data,
            output_file,
            ensure_ascii=False,
            indent=2,
            separators=(",", ": "),
        )


def generate_json(folder_path, split):
    yaml_file = read_file(folder_path + "/" + split + ".yaml")
    translations_file = read_file(folder_path + "/" + split + ".fra")

    assert len(yaml_file) == len(translations_file)

    output_json = dict()
    for i in range(len(yaml_file)):
        content = yaml_file[i]
        utt_id = content.split(", wav: ")[1].split("}")[0]
        output_json[utt_id] = dict()
        output_json[utt_id]["path"] = (
            folder_path.replace("/txt", "/wav") + "/" + utt_id + ".wav"
        )
        output_json[utt_id]["trans"] = translations_file[i]
        output_json[utt_id]["duration"] = content.split("{duration: ")[1].split(
            ","
        )[0]

    return output_json


def read_file(f_path):
    return [line for line in open(f_path)]


def data_proc(dataset_folder, output_folder):
    """

    Prepare .csv files for librimix

    Arguments:
    ----------
        dataset_folder (str) : path for the dataset github folder
        output_folder (str) : path where we save the json files.
    """

    try:
        os.mkdir(output_folder)
    except OSError:
        print(
            "Tried to create " + output_folder + ", but folder already exists."
        )

    for split in ["train", "valid", "test"]:
        split_folder = "/".join([dataset_folder, split, "txt"])

        output_json = generate_json(split_folder, split)

        write_json(output_folder + "/" + split + ".json", output_json)
