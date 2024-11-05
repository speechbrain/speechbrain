"""A script to prepare annotations for tokenizers

"""

import json
import os
import re

import datasets

from speechbrain.lobes.models.g2p.dataio import build_token_char_map

MULTI_SPACE = re.compile(r"\s{2,}")


def phn2txt(phn, phoneme_map):
    """Encodes phonemes using a character map for use with SentencePiece

    Arguments
    ---------
    phn: list
        a list of original phonemes (ARPABET)
    phoneme_map: dict
        the phoneme-to-character map

    Returns
    -------
    value: str
        the mapped string representation
    """
    value = "".join(phoneme_map[phoneme] for phoneme in phn).strip()
    value = MULTI_SPACE.sub("", value)
    return value


def prepare_annotation(src, destination_file_name, phonemes):
    """Prepares the annotation file

    Arguments
    ---------
    src: datasets.arrow_dataset.Dataset
        the source dataset
    destination_file_name: str
        the path to the annotation file to be created
    phonemes: list
        the list of phonemes
    """
    phoneme_map = build_token_char_map(phonemes)
    annotation = {
        item["sample_id"]: {
            "char": item["char"],
            "phn": phn2txt(item["phn"], phoneme_map),
        }
        for item in src
    }
    with open(destination_file_name, "w", encoding="utf-8") as dst_file:
        json.dump(annotation, dst_file, indent=2)


DATA_SPLITS = ["train", "valid", "test"]


def prepare_tokenizer(data_folder, save_folder, phonemes, dataset_name):
    """Prepares annotations for the tokenizer

    Arguments
    ---------
    data_folder: str
        the path to the dataset
    save_folder: str
        the path to the folder where annotations will be saved
    phonemes: list
        the list of phonemes
    dataset_name: str
        the name of the HuggingFace dataset
    """
    dataset = datasets.load_dataset(dataset_name) if dataset_name else None
    for data_split in DATA_SPLITS:
        destination_file_name = os.path.join(
            save_folder, f"tokenizer_annotation_{data_split}.json"
        )
        if os.path.exists(destination_file_name):
            print(f"Annotation file '{destination_file_name} already exists")
        else:
            print(
                f"Creating tokenizer annotation '{destination_file_name}' "
                f"from '{data_split}'"
            )
            prepare_annotation(
                src=dataset[f"sentence_{data_split}"],
                destination_file_name=destination_file_name,
                phonemes=phonemes,
            )
