"""
Data preparation.

Download: http://www.openslr.org/12

Author
------
Mirco Ravanelli, Ju-Chieh Chou 2020
"""

import os
import sys
import csv
import json
import logging
from speechbrain.utils.data_utils import get_all_files
from speechbrain.data_io.data_io import (
    read_wav_soundfile,
    save_pkl,
)

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
from LibriSpeech import librispeech_prepare  # noqa E402

logger = logging.getLogger(__name__)
OPT_FILE = "opt_libriparty_prepare.pkl"
SAMPLERATE = 16000


def prepare_libriparty(
    data_folder,
    librispeech_folder,
    splits,
    save_folder,
    select_n_sentences=None,
    use_lexicon=False,
):
    """
    This class prepares the csv files for the LibriSpeech dataset.
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
    select_n_sentences : int
        Default : None
        If not None, only pick this many sentences.
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
    select_n_sentences = select_n_sentences
    use_lexicon = use_lexicon
    conf = {
        "select_n_sentences": select_n_sentences,
        "use_lexicon": use_lexicon,
    }

    # Other variables
    # Saving folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    save_opt = os.path.join(save_folder, OPT_FILE)

    if use_lexicon:
        lexicon_dict = read_lexicon(
            os.path.join(librispeech_folder, "librispeech-lexicon.txt")
        )
    else:
        lexicon_dict = {}

    # Check if this phase is already done (if so, skip it)
    if librispeech_prepare.skip(splits, save_folder, conf):
        logger.info("Skipping preparation, completed in previous run.")
        return

    # Additional checks to make sure the data folder contains Librispeech
    librispeech_prepare.check_librispeech_folders(data_folder, splits)

    # create csv files for each split
    for split_index in range(len(splits)):

        split = splits[split_index]

        wav_lst = get_all_files(
            os.path.join(data_folder, split), match_and=["_dry.wav"]
        )

        text_lst = get_all_files(
            os.path.join(data_folder, split), match_and=[".json"]
        )

        pairs = pair_wav_text(wav_lst, text_lst)

        create_csv(save_folder, pairs, split, use_lexicon, lexicon_dict)

    # saving options
    save_pkl(conf, save_opt)


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


def create_csv(save_folder, triplets, split, use_lexicon, lexicon_dict):
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
    select_n_sentences : int, optional
        The number of sentences to select.

    Returns
    -------
    None
    """
    # Setting path for the csv file
    csv_file = os.path.join(save_folder, split + ".csv")

    # Preliminary prints
    msg = "Creating csv lists in  %s..." % (csv_file)
    logger.info(msg)

    csv_lines = [
        [
            "ID",
            "duration",
            "wav",
            "wav_format",
            "wav_opts",
            "spk_id",
            "spk_id_format",
            "spk_id_opts",
            "wrd",
            "wrd_format",
            "wrd_opts",
            "char",
            "char_format",
            "char_opts",
        ]
    ]

    if use_lexicon:
        csv_lines[0] += ["phn", "phn_format", "phn_opts"]

    for triplet in triplets:
        data = json.load(open(triplet[0], "r"))
        for key, item in data.items():
            session = triplet[0].split("/")[-2]
            current_wav = 1 if key in triplet[1] else 2
            # snt_id = triplet[current_wav].split("/")[-1].replace(".wav", "")
            spk_id = key
            signal = read_wav_soundfile(triplet[current_wav])
            duration = signal.shape[0] / SAMPLERATE
            for k, speech_instance in enumerate(item):
                start = speech_instance["start"] * SAMPLERATE
                stop = speech_instance["stop"] * SAMPLERATE
                wrds = speech_instance["words"]
                chars_lst = [c for c in wrds]
                chars = " ".join(chars_lst)
                csv_line = [
                    session + "_" + spk_id + "_" + str(k),
                    str(duration),
                    triplet[current_wav],
                    "wav",
                    "start:" + str(int(start)) + " stop:" + str(int(stop)),
                    spk_id,
                    "string",
                    "",
                    str(" ".join(wrds.split("_"))),
                    "string",
                    "",
                    str(chars),
                    "string",
                    "",
                ]
                if use_lexicon:
                    #     skip words not in the lexicon
                    phns = " ".join(
                        [
                            lexicon_dict[wrd]
                            for wrd in wrds.split("_")
                            if wrd in lexicon_dict
                        ]
                    )
                    csv_line += [str(phns), "string", ""]
                csv_lines.append(csv_line)

    # Writing the csv_lines
    with open(csv_file, mode="w") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )

        for line in csv_lines:
            csv_writer.writerow(line)

    # Final print
    msg = "%s sucessfully created!" % (csv_file)
    logger.info(msg)


def pair_wav_text(wav_lst, text_lst):
    triplets = []
    for i in text_lst:
        triplet = []
        for j in range(len(wav_lst)):
            if wav_lst[j].split("/")[-2] == i.split("/")[-2]:
                triplet.append(wav_lst[j])
        triplet.insert(0, i)
        triplets.append(triplet)
    return triplets


prepare_libriparty(
    "/data/vision/oliva/scratch/br/LibriParty/LibriParty_dataset/",
    "/data/vision/oliva/scratch/datasets/librispeech/LibriSpeech/",
    ["train", "eval"],
    "/afs/csail.mit.edu/u/b/br/br/speechbrain/recipes/SpeakerDiarization/libriparty_experiment/save",
)
