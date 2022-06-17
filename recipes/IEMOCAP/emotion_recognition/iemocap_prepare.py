"""
Downloads and creates data manifest files for IEMOCAP
(https://paperswithcode.com/dataset/iemocap).

Authors:
 * Mirco Ravanelli, 2021
 * Modified by Pierre-Yves Yanni, 2021
 * Abdel Heba, 2021
 * Yingzhi Wang, 2022
"""

import os
import sys
import re
import json
import random
import logging
from speechbrain.dataio.dataio import read_audio

logger = logging.getLogger(__name__)
SAMPLERATE = 16000
NUMBER_UTT = 5531


def prepare_data(
    data_original,
    save_json_train,
    save_json_valid,
    save_json_test,
    split_ratio=[80, 10, 10],
    different_speakers=False,
    test_spk_id=1,
    seed=12,
):
    """
    Prepares the json files for the IEMOCAP dataset.

    Arguments
    ---------
    data_original : str
        Path to the folder where the original IEMOCAP dataset is stored.
    save_json_train : str
        Path where the train data specification file will be saved.
    save_json_valid : str
        Path where the validation data specification file will be saved.
    save_json_test : str
        Path where the test data specification file will be saved.
    split_ratio: list
        List composed of three integers that sets split ratios for train,
        valid, and test sets, respecively.
        For instance split_ratio=[80, 10, 10] will assign 80% of the sentences
        to training, 10% for validation, and 10% for test.
    test_spk_id: int
        Id of speaker used for test set, 10 speakers in total.
        Here a leave-two-speaker strategy is used for the split,
        if one test_spk_id is selected for test, the other spk_id in the same
        session is automatically used for validation.
        To perform a 10-fold cross-validation,
        10 experiments with test_spk_id from 1 to 10 should be done.
    seed : int
        Seed for reproducibility

    Example
    -------
    >>> data_original = '/path/to/iemocap/IEMOCAP_full_release'
    >>> prepare_data(data_original, 'train.json', 'valid.json', 'test.json')
    """
    data_original = data_original + "/Session"
    # setting seeds for reproducible code.
    random.seed(seed)

    # Check if this phase is already done (if so, skip it)
    if skip(save_json_train, save_json_valid, save_json_test):
        logger.info("Preparation completed in previous run, skipping.")
        return

    speaker_dict = transform_data(data_original)

    if sum([len(value) for value in speaker_dict.values()]) != NUMBER_UTT:
        logger.error(
            "Error: Number of utterances is not 5531, please check your IEMOCAP folder"
        )
        sys.exit()

    # List files and create manifest from list
    logger.info(
        f"Creating {save_json_train}, {save_json_valid}, and {save_json_test}"
    )

    if different_speakers:
        data_split = split_different_speakers(speaker_dict, test_spk_id)
    else:
        data_split = split_sets(speaker_dict, split_ratio)

    # Creating json files
    create_json(data_split["train"], save_json_train)
    create_json(data_split["valid"], save_json_valid)
    create_json(data_split["test"], save_json_test)


def create_json(wav_list, json_file):
    """
    Creates the json file given a list of wav information.

    Arguments
    ---------
    wav_list : list of list
        The list of wav information (path, label, gender).
    json_file : str
        The path of the output json file
    """

    json_dict = {}
    for obj in wav_list:
        wav_file = obj[0]
        emo = obj[1]
        # Read the signal (to retrieve duration in seconds)
        signal = read_audio(wav_file)
        duration = signal.shape[0] / SAMPLERATE

        uttid = wav_file.split("/")[-1][:-4]

        # Create entry for this utterance
        json_dict[uttid] = {
            "wav": wav_file,
            "length": duration,
            "emo": emo,
        }

    # Writing the dictionary to the json file
    with open(json_file, mode="w") as json_f:
        json.dump(json_dict, json_f, indent=2)

    logger.info(f"{json_file} successfully created!")


def skip(*filenames):
    """
    Detects if the data preparation has been already done.
    If the preparation has been done, we can skip it.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """
    for filename in filenames:
        if not os.path.isfile(filename):
            return False
    return True


def split_different_speakers(speaker_dict, test_spk_id):
    """Constructs train, validation and test sets that do not share common
    speakers. There are two different speakers in each session. Train set is
    constituted of 4 sessions (8 speakers), while validation set and test set
    contain each 1 speaker. If test_spk_id is 1, then speaker 2 is selected
    automatically for validation set, and training set contains other 8 speakers.
    If test_spk_id is 2, then speaker 1 is selected for validation set.

    Arguments
    ---------
    speaker_dict: dict
        a dictionary of speaker id and its corresponding audio information
    test_spk_id: int
        Id of speaker used for test set, 10 speakers in total.
        Session1 contains speaker 1&2, Session2 contains speaker 3&4, ...

    Returns
    ------
    dictionary containing train, valid, and test splits.
    """
    data_split = {k: [] for k in ["train", "valid", "test"]}
    data_split["test"].extend(speaker_dict[str(test_spk_id)])

    # use the speaker in the same session as validation set
    if test_spk_id % 2 == 0:
        valid_spk_num = test_spk_id - 1
    else:
        valid_spk_num = test_spk_id + 1

    data_split["valid"].extend(speaker_dict[str(valid_spk_num)])

    for i in range(1, 11):
        if i != valid_spk_num and i != test_spk_id:
            data_split["train"].extend(speaker_dict[str(i)])

    return data_split


def split_sets(speaker_dict, split_ratio):
    """Randomly splits the wav list into training, validation, and test lists.
    Note that a better approach is to make sure that all the classes have the
    same proportion of samples (e.g, spk01 should have 80% of samples in
    training, 10% validation, 10% test, the same for speaker2 etc.). This
    is the approach followed in some recipes such as the Voxceleb one. For
    simplicity, we here simply split the full list without necessarly
    respecting the split ratio within each class.

    Arguments
    ---------
    speaker_dict : list
        a dictionary of speaker id and its corresponding audio information
    split_ratio: list
        List composed of three integers that sets split ratios for train,
        valid, and test sets, respectively.
        For instance split_ratio=[80, 10, 10] will assign 80% of the sentences
        to training, 10% for validation, and 10% for test.

    Returns
    ------
    dictionary containing train, valid, and test splits.
    """

    wav_list = []
    for key in speaker_dict.keys():
        wav_list.extend(speaker_dict[key])

    # Random shuffle of the list
    random.shuffle(wav_list)
    tot_split = sum(split_ratio)
    tot_snts = len(wav_list)
    data_split = {}
    splits = ["train", "valid"]

    for i, split in enumerate(splits):
        n_snts = int(tot_snts * split_ratio[i] / tot_split)
        data_split[split] = wav_list[0:n_snts]
        del wav_list[0:n_snts]
    data_split["test"] = wav_list

    return data_split


def transform_data(path_loadSession):
    """
    Create a dictionary that maps speaker id and corresponding wavs

    Arguments
    ---------
    path_loadSession : str
        Path to the folder where the original IEMOCAP dataset is stored.

    Example
    -------
    >>> data_original = '/path/to/iemocap/IEMOCAP_full_release/Session'
    >>> data_transformed = '/path/to/iemocap/IEMOCAP_ahsn_leave-two-speaker-out'
    >>> transform_data(data_original, data_transformed)
    """

    speaker_dict = {str(i + 1): [] for i in range(10)}

    speaker_count = 0
    for k in range(5):
        session = load_session("%s%s" % (path_loadSession, k + 1))
        for idx in range(len(session)):
            if session[idx][2] == "F":
                speaker_dict[str(speaker_count + 1)].append(session[idx])
            else:
                speaker_dict[str(speaker_count + 2)].append(session[idx])
        speaker_count += 2

    return speaker_dict


def load_utterInfo(inputFile):
    """
    Load utterInfo from original IEMOCAP database
    """

    # this regx allow to create a list with:
    # [START_TIME - END_TIME] TURN_NAME EMOTION [V, A, D]
    # [V, A, D] means [Valence, Arousal, Dominance]
    pattern = re.compile(
        "[\[]*[0-9]*[.][0-9]*[ -]*[0-9]*[.][0-9]*[\]][\t][a-z0-9_]*[\t][a-z]{3}[\t][\[][0-9]*[.][0-9]*[, ]+[0-9]*[.][0-9]*[, ]+[0-9]*[.][0-9]*[\]]",
        re.IGNORECASE,
    )  # noqa
    with open(inputFile, "r") as myfile:
        data = myfile.read().replace("\n", " ")
    result = pattern.findall(data)
    out = []
    for i in result:
        a = i.replace("[", "")
        b = a.replace(" - ", "\t")
        c = b.replace("]", "")
        x = c.replace(", ", "\t")
        out.append(x.split("\t"))
    return out


def load_session(pathSession):
    """Load wav file from IEMOCAP session
    and keep only the following 4 emotions:
    [neural, happy, sad, anger].

    Arguments
    ---------
        pathSession: str
            Path folder of IEMOCAP session.
    Returns
    -------
        improvisedUtteranceList: list
            List of improvised utterancefor IEMOCAP session.
    """
    pathEmo = pathSession + "/dialog/EmoEvaluation/"
    pathWavFolder = pathSession + "/sentences/wav/"

    improvisedUtteranceList = []
    for emoFile in [
        f
        for f in os.listdir(pathEmo)
        if os.path.isfile(os.path.join(pathEmo, f))
    ]:
        for utterance in load_utterInfo(pathEmo + emoFile):
            if (
                (utterance[3] == "neu")
                or (utterance[3] == "hap")
                or (utterance[3] == "sad")
                or (utterance[3] == "ang")
                or (utterance[3] == "exc")
            ):
                path = (
                    pathWavFolder
                    + utterance[2][:-5]
                    + "/"
                    + utterance[2]
                    + ".wav"
                )

                label = utterance[3]
                if label == "exc":
                    label = "hap"

                if emoFile[7] != "i" and utterance[2][7] == "s":
                    improvisedUtteranceList.append(
                        [path, label, utterance[2][18]]
                    )
                else:
                    improvisedUtteranceList.append(
                        [path, label, utterance[2][15]]
                    )
    return improvisedUtteranceList
