"""
Data preparation.

Reference: https://boulderlearning.com/request-the-myst-corpus/

Authors
* Jianyuan Zhong 2021
"""

import os
import json
import logging
import glob

from speechbrain.dataio.dataio import read_audio

logger = logging.getLogger(__name__)
SAMPLERATE = 16000
NON_SPEECH_EVENTS = [
    "<BREATH>",
    "<LAUGH>",
    "<COUGH>",
    "<NOISE>",
    "<SIDE_SPEECH>",
    "<SIDE_SPEECH>",
    "<SILENCE>",
    "<SNIFF>",
    "<ECHO>",
    "<DISCARD>",
]
UNINTELLIGIBLE = "(*)"
TRUNCATED_WORDS = "WH-"

myst_path = "/miniscratch/jian/myst-v0.4.2"
lexicon_file = "word2phn.dict"
word_dict = "word.dict"


def prepare_myst(
    data_folder,
    save_json_train,
    save_json_valid,
    save_json_test,
    save_json_unlabeled,
    word_dict=None,
    skip_prep=False,
    remove_non_speech_event=False,
    remove_unintelligible=False,
    remove_truncated_words=False,
):
    """
    repares the json files for the Myst dataset.

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original Myst dataset is stored.
    save_json_train : str
        The path where to store the training json file.
    save_json_valid : str
        The path where to store the valid json file.
    save_json_test : str
        The path where to store the test json file.
    skip_prep: bool
        Default: False
        If True, the data preparation is skipped.
    """

    # if os.path.exists(save_json_test):
    #     return

    if skip_prep:
        return

    # print(remove_non_speech_event, remove_truncated_words, remove_unintelligible)

    lexicon2phn = _get_phns(data_folder, lexicon_file)

    sub_folders = ["development", "test", "train"]
    json_files = [save_json_valid, save_json_test, save_json_train]
    unlabeled_audio_list = []

    # prepare and create json files
    for i, sub_folder in enumerate(sub_folders):
        audio_list, transcripts, unlabeled_audio_list = prepare_sub_folder(
            data_folder,
            sub_folder,
            lexicon2phn,
            unlabeled_audio_list,
            remove_non_speech_event,
            remove_unintelligible,
            remove_truncated_words,
        )
        create_json(audio_list, transcripts, json_files[i])

    # create json for unlabeled audios as well
    create_json(unlabeled_audio_list, None, save_json_unlabeled)


def _get_unique_words(data_folder, word_dict):
    sub_folder_pth = os.path.join(data_folder, "data")
    print(sub_folder_pth)
    assert os.path.exists(sub_folder_pth)
    trans_files = glob.glob(
        os.path.join(sub_folder_pth, "**/*.trn"), recursive=True
    )

    words = set()
    from tqdm import tqdm

    pbar = tqdm(trans_files)
    for f in pbar:
        with open(f, "r") as tf:
            line = tf.readline().strip()
            wrds = (
                line.replace(">", "> ")
                .replace("<", " <")
                .replace("(", " (")
                .replace(")", ") ")
                .replace("\xa0", " ")
            )
            wrds = wrds.strip().split()
            for w in wrds:
                words.add(w)

    with open(word_dict, "w") as wd:
        for wrd in words:
            wd.write(wrd + "\n")


def prepare_sub_folder(
    data_folder,
    sub_folder,
    lexicon2phn,
    unlabeled_audio_list,
    remove_non_speech_event=False,
    remove_unintelligible=False,
    remove_truncated_words=False,
):
    logger.info("Extracting info from {}...".format(sub_folder))
    audios = gather_files(data_folder, sub_folder)

    audio_list = []
    transcripts = []
    wrds = []
    notfound = set()
    for i, f in enumerate(audios):
        transcript = f.replace("flac", "trn")

        if os.path.exists(transcript):

            with open(transcript, "r") as tf:
                line = tf.readline().strip()
                wrds = (
                    line.replace(">", "> ")
                    .replace("<", " <")
                    .replace("(", " (")
                    .replace(")", ") ")
                    .replace("\xa0", " ")
                )

                skip = False
                if remove_non_speech_event:
                    if "<" in wrds and ">" in wrds:
                        skip = True

                    if "NO_SIGNAL" in wrds:
                        skip = True

                if remove_truncated_words:
                    if TRUNCATED_WORDS in wrds:
                        skip = True

                if remove_unintelligible:
                    if "(" in wrds and ")" in wrds:
                        skip = True

                wrds = wrds.split()
                phns = []
                try:
                    for wrd in wrds:
                        phns += lexicon2phn[wrd]

                    phns = " ".join(phns)

                except KeyError:
                    for wrd in wrds:
                        if wrd not in lexicon2phn:
                            notfound.add(wrd)
                    unlabeled_audio_list.append(f)

                if len(phns.strip().split()) == 0:
                    msg = "found empty transcript in {}. removeing it from the dataset!".format(
                        transcript
                    )
                    logger.info(msg)
                    skip = True

                if skip:
                    msg = "skipped {}".format(wrds)
                    logger.info(msg)
                    unlabeled_audio_list.append(f)
                    continue

                transcripts.append((phns, line))
                audio_list.append(f)

        else:
            unlabeled_audio_list.append(f)

    return audio_list, transcripts, unlabeled_audio_list


def create_json(
    wav_lst, trans_list, json_file,
):
    """
    Creates the json file given a list of wav files.

    Arguments
    ---------
    wav_lst : list
        The list of wav files of a given data split.
    json_file : str
            The path of the output json file.
    """
    # Adding some Prints
    msg = "Creating %s..." % (json_file)
    logger.info(msg)
    json_dict = {}

    for i, wav_file in enumerate(wav_lst):
        # Getting sentence and speaker ids
        spk_id = wav_file.split("/")[-3]
        snt_id = wav_file.split("/")[-1].replace(".flac", "")
        snt_id = spk_id + "_" + snt_id

        # Reading the signal (to retrieve duration in seconds)
        signal = read_audio(wav_file)
        duration = len(signal) / SAMPLERATE

        # discard sentences if they are too long
        if duration > 15:
            continue

        # get words and phoneme
        if trans_list is not None:
            phonemes, words = trans_list[i]
        else:
            phonemes, words = "", ""

        json_dict[snt_id] = {
            "wav": wav_file,
            "spk_id": spk_id,
            "snt_id": snt_id,
            "duration": duration,
            "phn": phonemes,
            "wrd": words,
        }

    # Writing the dictionary to the json file
    with open(json_file, mode="w") as json_f:
        json.dump(json_dict, json_f, indent=2)

    logger.info(f"{json_file} successfully created!")


def gather_files(data_folder, sub_folder):
    sub_folder_pth = os.path.join(data_folder, "data", sub_folder)
    assert os.path.exists(sub_folder_pth)
    audio_files = glob.glob(
        os.path.join(sub_folder_pth, "**/*.flac"), recursive=True
    )
    return audio_files


def _get_phns(myst_path, lexicon_file):
    lexicon2phn = {}
    with open(os.path.join(myst_path, "documents", lexicon_file), "r") as f:
        for i, line in enumerate(f):
            line = line.strip().split("\t")
            if len(line) == 2:
                word = line[0]
                phns = line[1].strip().split()
                lexicon2phn[word] = phns
            else:
                word = line[0]
                lexicon2phn[word] = [""]

    # add special tokens
    for event in NON_SPEECH_EVENTS:
        lexicon2phn[event] = [event]
    return lexicon2phn


# prepare_myst(myst_path, "train.json", "dev.json", "test.json", "unlabeled_json", "word.dict", False, True, True, True)
