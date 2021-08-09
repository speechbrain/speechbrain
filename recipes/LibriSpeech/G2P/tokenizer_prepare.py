import json
import os
import re

from speechbrain.lobes.models.g2p.attnrnn.dataio import build_token_char_map

MULTI_SPACE = re.compile(r"\s{2,}")

def phn2txt(phn, phoneme_map):
    """Encodes phonemes using a character map for use with SentencePiece

    Arguments
    ---------
    phn: list
        a list of original phonemes (ARPABET)
    phoneme_map
        the phoneme-to-character map

    Returns
    -------
    value: str
        the mapped string representation
    """
    value = "".join(phoneme_map[phoneme] for phoneme in phn).strip()
    value = MULTI_SPACE.sub("", value)
    return value


def prepare_annotation(source_file_name, destination_file_name, phonemes):
    """Prepares the annotation file

    Arguments
    ---------
    source_file_name: str
        the path to the source file (from librig2p)
    destination_file_name: str
        the path to the annotation file to be created
    phonemes: list
        the list of phonemes
    """
    phoneme_map = build_token_char_map(phonemes)
    with open(source_file_name) as src_file:
        src = json.load(src_file)
        annotation = {
            key: {
                "char": item["char"],
                "phn": phn2txt(item["phn"], phoneme_map)
            }
            for key, item in src.items()
        }
    with open(destination_file_name, "w") as dst_file:
        json.dump(annotation, dst_file, indent=2)


DATA_SPLITS = ["train", "valid", "test"]


def prepare_tokenizer(data_folder, save_folder, phonemes):
    """Prepares annotations for the tokenizer

    Arguments
    ---------
    data_folder: str
        the path to the dataset
    save_folder: str
        the path to the folder where annotations will be saved
    phonemes: list
        the list of phonemes
    """
    for data_split in DATA_SPLITS:
        destination_file_name = os.path.join(
            save_folder, f"tokenizer_annotation_{data_split}.json"
        )
        if os.path.exists(destination_file_name):
            print(f"Annotation file '{destination_file_name} already exists")
        else:
            source_file_name = os.path.join(
                data_folder, f"sentence_{data_split}.json")
            print(f"Creating tokenizer annotation '{destination_file_name}' "
                f"from '{source_file_name}'")
            prepare_annotation(
                source_file_name=source_file_name,
                destination_file_name=destination_file_name,
                phonemes=phonemes)