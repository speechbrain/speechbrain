"""
Data preparation for the SpokenWOZ dataset.

Download:
    - Audio files: https://spokenwoz.oss-cn-wulanchabu.aliyuncs.com/audio_5700_train_dev.tar.gz
    - Text annotations: https://spokenwoz.oss-cn-wulanchabu.aliyuncs.com/text_5700_train_dev.tar.gz
    - Audio test files: https://spokenwoz.oss-cn-wulanchabu.aliyuncs.com/audio_5700_test.tar.gz
    - Text test annotations: https://spokenwoz.oss-cn-wulanchabu.aliyuncs.com/text_5700_test.tar.gz
See `SpokenWOZ/dialogue_state_tracking/README.md` for more details.

Author
------
Lucas Druart 2024
"""

import csv
import json
import logging
import os

from tqdm.contrib import tzip

from speechbrain.dataio.dataio import load_pkl, merge_csvs, save_pkl
from speechbrain.utils.data_utils import get_all_files

logger = logging.getLogger(__name__)
OPT_FILE = "opt_spokenwoz_prepare.pkl"
SAMPLERATE = 8000


def prepare_spokenwoz(
    data_folder,
    save_folder,
    version="e2e",
    tr_splits=[],
    dev_splits=[],
    te_splits=[],
    select_n_sentences=None,
    merge_lst=[],
    merge_name=None,
    skip_prep=False,
) -> None:
    """
    This class prepares the csv files for the Spoken-Woz dataset.
    Download:
    - Audio files: https://spokenwoz.oss-cn-wulanchabu.aliyuncs.com/audio_5700_train_dev.tar.gz
    - Text annotations: https://spokenwoz.oss-cn-wulanchabu.aliyuncs.com/text_5700_train_dev.tar.gz
    - Audio test files: https://spokenwoz.oss-cn-wulanchabu.aliyuncs.com/audio_5700_test.tar.gz
    - Text test annotations: https://spokenwoz.oss-cn-wulanchabu.aliyuncs.com/text_5700_test.tar.gz

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original Multi-Woz dataset is stored.
    save_folder : str
        The directory where to store the csv files.
    version : str
        Version of dataset to prepare ("cascade" using previously computed transcriptions,
        "local" using turn level states or "global" using dialogue level states).
    tr_splits : list
        List of train splits to prepare from ['train'].
    dev_splits : list
        List of dev splits to prepare from ['dev'].
    te_splits : list
        List of test splits to prepare from ['test'].
    select_n_sentences : int
        Default : None
        If not None, only pick this many sentences.
    merge_lst : list
        List of splits (e.g, train-human, train-tts,..) to merge in a single csv file.
    merge_name: str
        Name of the merged csv file.
    skip_prep: bool
        If True, data preparation is skipped.

    Example
    -------
    >>> data_folder = 'datasets/SpokenWOZ'
    >>> versions = 'e2e'
    >>> tr_splits = ['train']
    >>> dev_splits = ['dev']
    >>> te_splits = ['test']
    >>> save_folder = 'spokenwoz_prepared'
    >>> prepare_spokenwoz(data_folder, save_folder, tr_splits, dev_splits, te_splits)
    """

    if skip_prep:
        return
    data_folder = data_folder
    splits = tr_splits + dev_splits + te_splits
    save_folder = save_folder
    select_n_sentences = select_n_sentences
    conf = {
        "select_n_sentences": select_n_sentences,
    }

    # Other variables
    # Saving folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    save_opt = os.path.join(save_folder, OPT_FILE)

    # Check if this phase is already done (if so, skip it)
    if skip(splits, save_folder, conf):
        logger.info("Skipping preparation, completed in previous run.")
        return
    else:
        logger.info("Data_preparation...")

    # Additional checks to make sure the data folder has the correct architecture
    check_spokenwoz_folders(data_folder, version, splits)

    # create csv files for each split
    all_texts = {}
    for split_index in range(len(splits)):

        split = splits[split_index]

        text_dict = {}

        wav_lst, annotations = get_wavs_and_annotations(
            data_folder, split, version
        )

        prepare_spokenwoz_split(text_dict, annotations, version)

        all_texts.update(text_dict)

        if select_n_sentences is not None:
            n_sentences = select_n_sentences
        else:
            n_sentences = len(wav_lst)

        create_csv(
            save_folder,
            wav_lst,
            text_dict,
            split,
            n_sentences,
        )

    # Merging csv file if needed
    if merge_lst and merge_name is not None:
        merge_files = [split + ".csv" for split in merge_lst]
        merge_csvs(
            data_folder=save_folder,
            csv_lst=merge_files,
            merged_csv=merge_name,
        )

    # saving options
    save_pkl(conf, save_opt)


def prepare_spokenwoz_split(text_dict, annotations, version) -> None:
    """
    Fills the text dictionary with the adequate annotations according to the required version.

    Arguments
    ---------
    text_dict : dict
        The split dictionary to fill to create the split csv.
    annotations : dict
        The dictionary from which to extract the relevant information.
    version: str
        The version of the processing required (cascade[-MODEL] or e2e).
    """
    for dialog_id, dialog_info in annotations.items():
        if dialog_id not in text_dict:
            text_dict[dialog_id] = {}

        for turn_id, turn_info in enumerate(dialog_info["log"]):
            # Dialogue States are updated at each agent turns considering the turns up to the previous user turn
            if turn_id % 2 == 0:
                # User turn --> get end time
                if f"Turn-{turn_id}" not in text_dict[dialog_id]:
                    text_dict[dialog_id][f"Turn-{turn_id}"] = {}
                current_turn = text_dict[dialog_id][f"Turn-{turn_id}"]
                end_time = turn_info["words"][-1]["EndTime"]
                current_turn["end"] = end_time * (SAMPLERATE // 1000)
                if turn_id == 0:
                    current_turn["start"] = 0
                    current_turn["previous"] = ""
                    current_turn["agent"] = ""

                # Saving the user transcription
                if "cascade" in version:
                    # Considering the model's transcription if cascading with a provided model,
                    # otherwise the dataset's automatic transcriptions
                    asr_model = version.replace("cascade", "").split("_")[-1]
                    if asr_model != "":
                        current_turn["user"] = turn_info[asr_model]
                    else:
                        words = [word["Word"] for word in turn_info["words"]]
                        current_turn["user"] = " ".join(words)
                else:
                    current_turn["user"] = ""

            else:
                # Agent turn --> dialogue state annotations
                state = []
                for domain, info in turn_info["metadata"].items():
                    for slot, value in info["book"].items():
                        if slot != "booked" and value != "":
                            state.append(f"{domain}-{slot}={value}")
                    for slot, value in info["semi"].items():
                        if value != "":
                            # One example in train set has , between numbers
                            state.append(
                                f'{domain}-{slot}={value.replace(",", "")}'
                            )
                text_dict[dialog_id][f"Turn-{turn_id - 1}"]["current"] = (
                    "; ".join(state)
                )

                # Preparing data for next turn prediction: last dialogue turn has no succeeding user turn
                if turn_id != len(dialog_info["log"]) - 1:
                    if f"Turn-{turn_id + 1}" not in text_dict[dialog_id]:
                        text_dict[dialog_id][f"Turn-{turn_id + 1}"] = {}
                    next_turn = text_dict[dialog_id][f"Turn-{turn_id + 1}"]
                    # Agent transcription
                    if "cascade" in version:
                        # Considering the model's transcription if cascading with a provided model,
                        # otherwise the dataset's automatic transcriptions
                        asr_model = version.replace("cascade", "").split("_")[
                            -1
                        ]
                        if asr_model != "":
                            next_turn["agent"] = turn_info[asr_model]
                        else:
                            words = [
                                word["Word"] for word in turn_info["words"]
                            ]
                            next_turn["agent"] = " ".join(words)
                    else:
                        next_turn["agent"] = ""

                    # Previous Dialogue State
                    next_turn["previous"] = "; ".join(state)
                    begin_time = turn_info["words"][0]["BeginTime"]
                    next_turn["start"] = begin_time * (SAMPLERATE // 1000)


def create_csv(
    save_folder,
    wav_lst,
    text_dict,
    split,
    select_n_sentences,
):
    """
    Create the dataset csv file given a list of wav files.

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
    select_n_sentences : int, optional
        The number of sentences to select.

    Returns
    -------
    None
    """
    # Setting path for the csv file
    csv_file = os.path.join(save_folder, split + ".csv")
    if os.path.exists(csv_file):
        logger.info("Csv file %s already exists, not recreating." % csv_file)
        return

    # Preliminary prints
    msg = "Creating csv lists in  %s..." % (csv_file)
    logger.info(msg)

    csv_lines = [
        [
            "ID",
            "turnID",
            "duration",
            "previous_state",
            "wav",
            "start",
            "end",
            "agent",
            "user",
            "current_state",
        ]
    ]

    snt_cnt = 0
    # Processing all the wav files in wav_lst
    for wav_file in tzip(wav_lst):
        wav_file = wav_file[0]
        dialog_id = wav_file.split("/")[-1].replace(".wav", "")

        for turn_id, turn_annotations in text_dict[dialog_id].items():
            snt_id = wav_file.replace(".wav", "_") + turn_id
            previous_state = turn_annotations["previous"]
            start = turn_annotations["start"]
            end = turn_annotations["end"]
            agent = turn_annotations["agent"].replace("$", " dollars ")
            user = turn_annotations["user"].replace("$", " dollars ")
            current_state = turn_annotations["current"]

            duration = (end - start) / SAMPLERATE

            if select_n_sentences == snt_cnt and split == "train":
                break

            else:
                csv_line = [
                    snt_id,
                    int(turn_id.replace("Turn-", "")),
                    str(duration),
                    str(previous_state),
                    wav_file,
                    start,
                    end,
                    str(agent),
                    str(user),
                    str(current_state),
                ]

                #  Appending current file to the csv_lines list
                csv_lines.append(csv_line)
                snt_cnt = snt_cnt + 1

    # Writing the csv_lines
    with open(csv_file, mode="w") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )

        for line in csv_lines:
            csv_writer.writerow(line)

    # Final print
    msg = "%s successfully created!" % (csv_file)
    logger.info(msg)


def get_wavs_and_annotations(data_folder, split, version):
    """
    Fetches the wav files and their corresponding annotations for the provided split and version.

    Arguments
    ---------
    data_folder : str
        Location of the root data folder containing the audio files of each split and their annotations.
    split : str
        The split for which the wav files and annotations should be fetched.
    version : str
        The dataset version required. Should be one of e2e or cascade_model.

    Returns
    -------
    wav_lst : list
        The list of wav files for the required split and version.
    annotations : dict
        The dictionary containing the annotations associated to those wav files.
    """
    wav_lst = []
    if split == "train" or split == "dev":
        dev_ids = []
        with open(
            os.path.join(
                data_folder, "text_5700_train_dev", "valListFile.json"
            ),
            "r",
        ) as val_list_file:
            for line in val_list_file:
                dev_ids.append(line.strip())

    if split == "train":
        wav_lst.extend(
            get_all_files(
                os.path.join(data_folder, "audio_5700_train_dev"),
                match_and=[".wav"],
                exclude_or=dev_ids,
            )
        )

        if "cascade" in version:
            annotation_file = os.path.join(
                data_folder,
                "text_5700_train_dev",
                "data{}.json".format(version.replace("cascade", "")),
            )
        else:
            annotation_file = os.path.join(
                data_folder, "text_5700_train_dev", "data.json"
            )
        with open(annotation_file, "r") as data:
            for line in data:
                annotations = json.loads(line)
        dialogues_to_remove = [
            k for k, _ in annotations.items() if k in dev_ids
        ]
        for dialog_id in dialogues_to_remove:
            del annotations[dialog_id]

    elif split == "dev":
        wav_lst.extend(
            get_all_files(
                os.path.join(data_folder, "audio_5700_train_dev"),
                match_and=[".wav"],
                match_or=dev_ids,
            )
        )

        if "cascade" in version:
            annotation_file = os.path.join(
                data_folder,
                "text_5700_train_dev",
                "data{}.json".format(version.replace("cascade", "")),
            )
        else:
            annotation_file = os.path.join(
                data_folder, "text_5700_train_dev", "data.json"
            )
        with open(annotation_file, "r") as data:
            for line in data:
                annotations = json.loads(line)
        dialogues_to_remove = [
            k for k, _ in annotations.items() if k not in dev_ids
        ]
        for dialog_id in dialogues_to_remove:
            del annotations[dialog_id]

    elif split == "test":
        wav_lst.extend(
            get_all_files(
                os.path.join(data_folder, "audio_5700_test"),
                match_and=[".wav"],
            )
        )

        if "cascade" in version:
            annotation_file = os.path.join(
                data_folder,
                "text_5700_test",
                "data{}.json".format(version.replace("cascade", "")),
            )
        else:
            annotation_file = os.path.join(
                data_folder, "text_5700_test", "data.json"
            )
        with open(annotation_file, "r") as data:
            for line in data:
                annotations = json.loads(line)

    else:
        err_msg = (
            "Asked for %s split which does not exist."
            "Please select one of (train|dev|test)" % split
        )
        raise OSError(err_msg)

    return wav_lst, annotations


def skip(splits, save_folder, conf):
    """
    Detect when the data prep can be skipped.

    Arguments
    ---------
    splits : list
        A list of the splits expected in the preparation.
    save_folder : str
        The location of the save directory
    conf : dict
        The configuration options to ensure they haven't changed.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """

    # Checking csv files
    skip = True

    for split in splits:
        if not os.path.isfile(os.path.join(save_folder, split + ".csv")):
            skip = False

    #  Checking saved options
    save_opt = os.path.join(save_folder, OPT_FILE)
    if skip is True:
        if os.path.isfile(save_opt):
            opts_old = load_pkl(save_opt)
            if opts_old == conf:
                skip = True
            else:
                skip = False
        else:
            skip = False

    return skip


def check_spokenwoz_folders(data_folder, splits) -> None:
    """
    Check if the data folder actually contains the Multi-Woz dataset.

    If it does not, an error is raised.

    Arguments
    ---------
    data_folder : str
        Path to the folder where the extracted turn audio files are stored.
    splits : list
        A list of the splits expected in the preparation

    Raises
    ------
    OSError
        If Spoken-Woz is not found at the specified path.
    """
    for split in splits:
        if split == "train" or split == "dev":
            text_folder = os.path.join(data_folder, "text_5700_train_dev")
            audio_folder = os.path.join(data_folder, "audio_5700_train_dev")
        else:
            text_folder = os.path.join(data_folder, f"text_5700_{split}")
            audio_folder = os.path.join(data_folder, f"audio_5700_{split}")
        if not os.path.exists(text_folder):
            err_msg = (
                "the folder %s does not exist (it is expected in the "
                "Spoken-Woz dataset)" % text_folder
            )
            raise OSError(err_msg)
        if not os.path.exists(audio_folder):
            err_msg = (
                "the folder %s does not exist (it is expected in the "
                "Spoken-Woz dataset)" % audio_folder
            )
            raise OSError(err_msg)
