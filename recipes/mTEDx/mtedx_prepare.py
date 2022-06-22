"""
This script assumes that data was downloaded, and it just prepares the data
for training.

You can download the data from here: http://www.openslr.org/100

Author
------
Mohamed Anwar (Anwarvic) 2022
"""

import os
import yaml
import json
import string
import logging
# from tqdm import tqdm

logger = logging.getLogger(__name__)


def prepare_mtedx(
    data_folder,
    save_folder,
    langs=[],
):
    """
    This class prepares the csv files for the mTEDx dataset.
    Download link: http://www.openslr.org/12

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original mTEDx dataset is stored.
    save_folder: str
        Path to the folder where the resulting files will be stored.
    langs: list(str)
        A list of language-codes to be prepared.

    Example
    -------
    >>> data_folder = 'datasets/LibriSpeech'
    >>> save_folder = 'librispeech_prepared'
    >>> langs = ["fr", "es"]
    >>> prepare_mtedx(data_folder, save_folder, langs)
    """ 
    # Saving folder
    os.makedirs(save_folder, exist_ok=True)

    if skip(save_folder, langs):
        logger.info("Skipping preparation, completed in previous run.")
        return
    else:
        logger.info("Data_preparation...")

    # TODO: Additional checks to make sure the data folder contains mTEDx
    # check_mtedx_folders(data_folder, langs)

    # create json files for each group
    for group in ["test", "valid", "train"]:
        create_json(data_folder, save_folder, langs, group)


def create_json(data_folder, save_folder, langs, group):
    """
    Create the dataset json file given a list of wav files.

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original mTEDx dataset is stored.
    save_folder : str
        Location of the folder for storing the csv.
    wav_lst : list
        The list of wav files of a given data split.
    text_dict : list
        The dictionary containing the text of each sentence.
    split : str
        The name of the current data split.
    """
    # Setting path for the json file
    json_file = os.path.join(save_folder, f"{group}_{'_'.join(langs)}.json")
    
    # skip if the file already exists
    if os.path.exists(json_file):
        logger.info(f"{json_file} already exists. Skipping!!")
        return
    
    logger.info(f"Creating json file in {json_file}")
    for lang in langs:
        logger.info(f"Processing language: {lang}, group: {group}")
        base_dir = \
            os.path.join(data_folder, f"{lang}-{lang}", "data", group, "txt")
        
        # parse YAML file containing audio info
        with open( os.path.join(base_dir, f"{group}.yaml"), "r") as fin:
            audio_samples = yaml.load(fin, Loader=yaml.Loader)
        
        # parse text file containing text info
        with open( os.path.join(base_dir, f"{group}.{lang}"), "r") as fin:
            text_samples = fin.readlines()
        
        # sanity check
        assert len(text_samples) == len(audio_samples), \
            f"Data mismatch with language: {lang}, group: {group}"
        
        # combine text & audio information
        group_data = {}
        for i in range(len(audio_samples)):
            audio, text = audio_samples[i], text_samples[i]
            # ignore audios >= 20 seconds & <= 0.1 in `train`
            if (
                group == "train"
                and (audio["duration"] > 20 or audio["duration"] < 0.1)
            ):
                continue
            audio_filepath = \
                f"{lang}-{lang}/data/{group}/wav/{audio['speaker_id']}.flac"
            group_data[audio["speaker_id"]+f"_{i}"] = {
                "wav": {
                    "file": "{data_root}/" + audio_filepath,
                    "start": audio["offset"],
                    "end": audio["offset"]+audio["duration"],
                },
                "words": text.strip(),
                "duration": audio["duration"],
                "lang": lang,
            }
    # write dict into json file    
    with open(json_file, 'w', encoding='utf8') as fout:
        fout.write(json.dumps(group_data, indent=4, ensure_ascii=False))
    logger.info(f"{json_file} successfully created!")


def skip(save_folder, langs):
    """
    Detect when the mTEDx data preparation can be skipped. The preparation
    can be skipped iff `train.json`, `valid.json` and `test.json` exist.

    Arguments
    ---------
    save_folder : str
        The location of the save directory.
    langs : list(str)
        A list of languages expected in the preparation.

    Returns
    -------
    bool
        If True, the preparation is skipped. Otherwise, it must be done.
    """
    langs = '_'.join(langs)
    return (
        os.path.exists(os.path.join(save_folder, f"train_{langs}.json")) and
        os.path.exists(os.path.join(save_folder, f"valid_{langs}.json")) and
        os.path.exists(os.path.join(save_folder, f"test_{langs}.json"))
    )


def check_mtedx_folders(data_folder, splits):
    """
    Check if the data folder actually contains the mTEDx dataset.
    If it does not, an error is raised.

    Returns
    -------
    None

    Raises
    ------
    OSError
        If LibriSpeech is not found at the specified path.
    """
    pass


#helpful function
def remove_punctuations(text):
    """Removes punctuations from text"""
    PUNCS = string.punctuation+'،؟؛¿¡—÷×»«‹›'
    return text.translate(str.maketrans('', '', PUNCS)).strip()