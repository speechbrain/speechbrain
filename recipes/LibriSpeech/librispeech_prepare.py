"""
Data preparation.

Download: http://www.openslr.org/12

Author
------
Mirco Ravanelli, Ju-Chieh Chou, Samuele Cornell 2020
"""

import os
import logging
from speechbrain.utils.data_utils import get_all_files
import soundfile as sf
import argparse
from ruamel import yaml
from pathlib import Path

logger = logging.getLogger(__name__)
SAMPLERATE = 16000

parser = argparse.ArgumentParser(
    "Librispeech datacollection yaml files preparation script"
)
parser.add_argument("--dataset_root", required=True, type=str)
parser.add_argument("--splits", required=True, type=str)
parser.add_argument("--save_dir", required=True, type=str)
parser.add_argument("--use_lexicon", required=False, type=int, default=1)


def prepare_librispeech(
    data_folder, splits, save_folder, use_lexicon=False,
):
    """
    This function prepares the csv files for the LibriSpeech dataset.
    Download link: http://www.openslr.org/12

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original LibriSpeech dataset is stored.
    splits : list
        List of splits to prepare from ['dev-clean','dev-others','test-clean',
        'test-others','train-clean-100','train-clean-360','train-other-500']
    save_folder : str
        The directory where to store the csv files.
    use_lexicon : bool
        When it is true, using lexicon to generate phonemes columns in the csv file.

    Example
    -------
    >>> data_folder = 'datasets/LibriSpeech'
    >>> splits = ['train-clean-100', 'dev-clean', 'test-clean']
    >>> save_folder = 'librispeech_prepared'
    >>> prepare_librispeech(data_folder, splits, save_folder)
    """
    data_folder = data_folder
    splits = splits
    save_folder = save_folder
    use_lexicon = use_lexicon
    # Other variables
    # Saving folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    if use_lexicon:
        lexicon_dict = read_lexicon(
            os.path.join(data_folder, "librispeech-lexicon.txt")
        )
    else:
        lexicon_dict = {}

    # Additional checks to make sure the data folder contains Librispeech
    check_librispeech_folders(data_folder, splits)

    # create csv files for each split
    for split_index in range(len(splits)):

        split = splits[split_index]

        wav_lst = get_all_files(
            os.path.join(data_folder, split), match_and=[".flac"]
        )

        text_lst = get_all_files(
            os.path.join(data_folder, split), match_and=["trans.txt"]
        )

        text_dict = text_to_dict(text_lst)

        create_yamls(
            data_folder,
            save_folder,
            wav_lst,
            text_dict,
            split,
            use_lexicon,
            lexicon_dict,
        )


def read_lexicon(lexicon_path):
    """
    Read the lexicon into a dictionary.
    Download link: http://www.openslr.org/resources/11/librispeech-lexicon.txt
    Arguments
    ---------
    lexicon_path : string
        The path of the lexicon.
    """
    if not os.path.exists(lexicon_path):
        err_msg = (
            f"Lexicon path {lexicon_path} does not exist."
            "Link: http://www.openslr.org/resources/11/librispeech-lexicon.txt"
        )
        raise OSError(err_msg)

    lexicon_dict = {}

    with open(lexicon_path, "r") as f:
        for line in f:
            line_lst = line.split()
            lexicon_dict[line_lst[0]] = " ".join(line_lst[1:])
    return lexicon_dict


def create_yamls(
    data_folder,
    save_folder,
    wav_lst,
    text_dict,
    split,
    use_lexicon,
    lexicon_dict,
):
    """
    Create the csv file given a list of wav files.

    Arguments
    ---------
    save_folder : str
        Location of the folder for storing the csv.
    wav_lst : list
        The list of wav files of a given data split.
    text_dict : list
        The dictionary containing the text of each sentence.
    split : str
        The name of the current data split.
    use_lexicon : bool
        Whether to use a lexicon or not.
    lexicon_dict : dict
        A dictionary for converting words to phones.

    Returns
    -------
    None
    """
    # Setting path for the yaml file
    yaml_file = os.path.join(save_folder, split + ".yaml")

    # Preliminary prints
    msg = "Creating Data Object entries in  %s..." % (yaml_file)
    logger.info(msg)

    snt_cnt = 0
    to_yaml = {}
    # Processing all the wav files in wav_lst
    for wav_file in wav_lst:

        snt_id = str(Path(wav_file).stem)
        spk_id = "-".join(snt_id.split("-")[0:1])
        wrds = text_dict[snt_id]

        samples = len(sf.SoundFile(wav_file))

        # replace space to <space> token
        chars_lst = [c for c in wrds]

        data_obj = {
            "waveforms": {
                "files": os.path.join(
                    "DATASET_ROOT", str(Path(wav_file).relative_to(data_folder))
                ),
                "channels": [0],
                "samplerate": SAMPLERATE,
                "lengths": [samples],
            },
            "supervisions": [
                {"spk_id": spk_id, "words": wrds.split("_"), "chars": chars_lst}
            ],
        }

        if use_lexicon:
            # skip words not in the lexicon
            data_obj["supervisions"][0]["phns"] = [
                lexicon_dict[wrd]
                for wrd in wrds.split("_")
                if wrd in lexicon_dict
            ]

        #  Appending current file to the csv_lines list
        to_yaml[snt_id] = data_obj
        snt_cnt += 1

    # Writing the csv_lines
    with open(yaml_file, mode="w") as f:
        yaml.dump(to_yaml, f)

    # Final print
    msg = "%s sucessfully created!" % (yaml_file)
    logger.info(msg)


def text_to_dict(text_lst):
    """
    This converts lines of text into a dictionary-

    Arguments
    ---------
    text_lst : str
        Path to the file containing the librispeech text transcription.

    Returns
    -------
    dict
        The dictionary containing the text transcriptions for each sentence.

    """
    # Initialization of the text dictionary
    text_dict = {}
    # Reading all the transcription files is text_lst
    for file in text_lst:
        with open(file, "r") as f:
            # Reading all line of the transcription file
            for line in f:
                line_lst = line.strip().split(" ")
                text_dict[line_lst[0]] = "_".join(line_lst[1:])
    return text_dict


def check_librispeech_folders(data_folder, splits):
    """
    Check if the data folder actually contains the LibriSpeech dataset.

    If it does not, an error is raised.

    Returns
    -------
    None

    Raises
    ------
    OSError
        If LibriSpeech is not found at the specified path.
    """
    # Checking if all the splits exist
    for split in splits:
        split_folder = os.path.join(data_folder, split)
        if not os.path.exists(split_folder):
            err_msg = (
                "the folder %s does not exist (it is expected in the "
                "Librispeech dataset)" % split_folder
            )
            raise OSError(err_msg)


if __name__ == "__main__":
    args = parser.parse_args()
    splits = args.splits.split(" ")
    for x in splits:
        assert x in [
            "train-clean-100",
            "dev-clean",
            "test-clean",
            "train-clean-360",
            "train-other-500",
            "dev-other",
            "test-other",
        ]
    prepare_librispeech(
        args.dataset_root, splits, args.save_dir, bool(args.use_lexicon)
    )
