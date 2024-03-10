from itertools import product
from statistics import mean
from typing import Any, Dict, List, Optional, Set, Tuple
import json
import logging
import os
import re
import shutil
from tqdm import tqdm
from speechbrain.utils.data_utils import download_file

"""
Data preparation.
Download: https://github.com/budzianowski/multiwoz/tree/master/data

The original one can be found at:
https://github.com/jasonwu0731/trade-dst/blob/master/create_data.py
Author
------
 * Pooneh Mousavi 2023
 * Simone Alghisi 2023
"""

logger = logging.getLogger(__name__)
MULTIWOZ_21_DATASET_URL = (
    "https://github.com/budzianowski/multiwoz/raw/master/data/MultiWOZ_2.1.zip"
)


def prepare_mwoz_21(
    data_folder: str, save_folder: str, replacements_path: str, skip_prep=False,
) -> None:

    """
    This class prepares the JSON files for the MultiWOZ dataset.
    Data will be automatically downloaded in the data_folder.
    Download link: https://github.com/budzianowski/multiwoz/tree/master/data

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original MultiWOZ dataset is stored.
    save_folder : str
        The directory where to store the JSON files.
    replacements_path: str
        File containing (from, to) pairs, one per line for preprocessing the text.
    skip_prep: bool
        If True, data preparation is skipped.


    Example
    -------
    >>> data_folder = 'data/MultiWOZ_2.1'
    >>> save_folder = 'MultiWOZ_prepared'
    >>> replacements_path = 'mapping.pair'
    >>> prepare_mwoz_21(data_folder, save_folder, replacements_path)
    """

    if skip_prep:
        return

    # Saving folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Setting ouput files
    save_train = save_folder + "/train.json"
    save_dev = save_folder + "/dev.json"
    save_test = save_folder + "/test.json"

    # If csv already exists, we skip the data preparation
    if skip(save_train, save_dev, save_test):

        msg = "%s already exists, skipping data preparation!" % (save_train)
        logger.info(msg)

        msg = "%s already exists, skipping data preparation!" % (save_dev)
        logger.info(msg)

        msg = "%s already exists, skipping data preparation!" % (save_test)
        logger.info(msg)

        return

    # Download dataset
    download_mwoz_21(data_folder)
    data_folder = os.path.join(data_folder, "MultiWOZ_21")

    # Additional checks to make sure the data folder contains MultiWOZ
    check_multiwoz_folders(data_folder)

    data_path = os.path.join(data_folder, "data.json")
    train_split, dev_split, test_split = get_splits(data_folder)
    # Creating json files for {train, dev, test} data
    file_pairs = zip(
        [train_split, dev_split, test_split], [save_train, save_dev, save_test],
    )

    for split, save_file in file_pairs:
        build_dialogue_dataset(
            data_path, split, save_file, replacements_path,
        )


def check_multiwoz_folders(data_folder):
    """
    Check if the data folder actually contains the MultiWOZ dataset.
    If not, raises an error.
    Returns
    -------
    None
    Raises
    ------
    FileNotFoundError
        If the data folder doesn't contain the MultiWOZ dataset.
    """
    files_str = "/data.json"
    # Checking clips
    if not os.path.exists(data_folder + files_str):
        err_msg = (
            "the folder %s does not exist (it is expected in "
            "the MultiWOZ dataset)" % (data_folder + files_str)
        )
        raise FileNotFoundError(err_msg)


def download_mwoz_21(destination):
    """ Download the dataset repo, unpack it, and remove unnecessary elements.
    Arguments
    ---------
    destination: str
        Place to put dataset.
    """
    mwoz_21_archive = os.path.join(destination, "MultiWOZ_21.zip")
    download_file(MULTIWOZ_21_DATASET_URL, mwoz_21_archive)
    shutil.unpack_archive(mwoz_21_archive, destination)
    shutil.rmtree(os.path.join(destination, "__MACOSX"))

    mwoz_21 = os.path.join(destination, "MultiWOZ_21")
    os.makedirs(mwoz_21, exist_ok=True)

    mwoz_21_repo = os.path.join(destination, "MultiWOZ_2.1")
    for relevant_file in ["data.json", "valListFile.txt", "testListFile.txt"]:
        shutil.move(
            os.path.join(mwoz_21_repo, relevant_file),
            os.path.join(mwoz_21, relevant_file),
        )

    shutil.rmtree(mwoz_21_repo)


def skip(save_train, save_dev, save_test):
    """
    Detects if the MultiWOZ data preparation has been already done.
    If the preparation has been done, we can skip it.
    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """

    # Checking folders and save options
    skip = False

    if (
        os.path.isfile(save_train)
        and os.path.isfile(save_dev)
        and os.path.isfile(save_test)
    ):
        skip = True

    return skip


def get_splits(dataset_folder) -> Tuple[List[str], List[str], List[str]]:
    mwoz_21_dialouges = get_json_object(
        os.path.join(dataset_folder, "data.json")
    )
    dialougues_keys: Set[str] = set(mwoz_21_dialouges.keys())
    tr_split: List[str] = []
    with open(os.path.join(dataset_folder, "valListFile.txt")) as f:
        dev_split: List[str] = [key.strip() for key in f]
    with open(os.path.join(dataset_folder, "testListFile.txt")) as f:
        te_split: List[str] = [key.strip() for key in f]

    for key in dialougues_keys:
        if key not in dev_split and key not in te_split:
            tr_split.append(key)

    return tr_split, dev_split, te_split


def build_dialogue_dataset(
    data_path: str,
    data_split: List[str],
    save_file: str,
    replacements_path: str,
) -> None:
    """
    Returns the dialogue dataset for the corresponding data_path.

    Arguments
    ---------
    data_path: str
     Path to the folder where the original MultiWOZ dataset is stored.
    data_split: list of str
        List of strings containing MultiWOZ 2.1 keys of the dialogues
        associated with a certain split (train, dev, test).
    save_file: str
        Path of the file where the dataset will be saved.
    replacements_path: str
        Path to file containing (from, to) pairs, one per line.

    Returns
    -------
    dataset:
        dataset, keys are str, values are dictionaries containing the
        dialogue history, the system reply, and the mean length.
    """
    logger.info(f"Prepare {save_file}")
    encode_dialogue_dataset(
        save_file, data_path, data_split, replacements_path,
    )


def encode_dialogue_dataset(
    save_file: str,
    data_path: str,
    data_split: List[str],
    replacements_path: str,
) -> None:
    """
    Wrapper function that loads processed data stored at
    dst_folder/file_name. If they are not available, it processes the
    original data and then saves them at dst_folder/file_name.

    Arguments
    ---------
    save_file: str
        Path of the file where the dataset will be saved.
    data_path: str
        Path to the folder where the original MultiWOZ dataset is stored.
    data_split: list of str
        List of strings containing MultiWOZ 2.1 keys of the dialogues
        associated with a certain split (train, dev, test).
    replacements_path: str
        Path to file containing (from, to) pairs, one per line.
    """

    replacements = get_replacements(replacements_path)
    logger.info(f"Extract dialogues from {data_path}")
    # custom loading function to return the important elements of a dialogue
    dialogues = load_dialogues(data_path, data_split, replacements)

    logger.info("Create dataset")
    dataset = create_dialogue_dataset(dialogues)
    logger.info(f"Save dataset in {save_file}")
    save_dialogue_dataset(dataset, save_file)


def get_replacements(
    replacements_path: str = "trade/utils/mapping.pair",
) -> List[Tuple[str, str]]:
    """
    Get the replacements from a given file. Used by trade preprocessing.

    Arguments
    ---------
    replacements_path: str
        File containing from, to pairs, one per line.

    Returns
    -------
    replacements: List of replacements, i.e. pairs of str
        Pairs of elements used to substitute the first element with the second.
    """
    replacements = []
    with open(replacements_path, "r") as fin:
        for line in fin.readlines():
            tok_from, tok_to = line.replace("\n", "").split("\t")
            replacements.append((" " + tok_from + " ", " " + tok_to + " "))
    return replacements


def load_dialogues(
    data_path: str, data_split: List[str], replacements: List[Tuple[str, str]],
) -> List[List[Dict[str, Any]]]:
    """
    Load dialogues from data_path, apply trade pre-processing, revert the
    subtokenization, and create a dictionary containing the dialogue id,
    the turn id, and the corrected sequence.

    Arguments
    ---------
    data_path: str
        Path to the json file containing the data.
    data_split: list of str
        List of string containing MultiWOZ 2.1 keys of the dialogues
        associated to a certain split (train, dev, test).
    replacements_path: str
        File containing (from, to) pairs, one per line.

    Returns
    -------
    dialogues: list of list of dict, keys are str, values could be anything
        List of dialogues. Each dialogue is a list of turns. Each turn is a
        dict containing dialogue_idx, turn_idx, and the corrected sequence.
    """

    def get_preprocessed_seq(
        original_seq: str, replacements: List[Tuple[str, str]]
    ) -> str:
        # apply trade normalization
        trade_seq = normalize(original_seq, replacements)
        # merge back subtokens
        sequence = invert_trade_subtokenization(original_seq, trade_seq)
        return sequence

    dialogues: List[List[Dict[str, Any]]] = []

    data = get_json_object(data_path)

    for dialogue_idx in tqdm(data_split, desc="Load Dialogues"):
        dial: List[Dict[str, Any]] = []
        original_dialogue: dict = data[dialogue_idx]
        turns: dict = original_dialogue["log"]
        for i, turn in enumerate(turns):
            sequence = get_preprocessed_seq(turn["text"], replacements)
            to_save = {
                "sequence": sequence,
                "turn_idx": i,
                "dialogue_idx": dialogue_idx,
            }
            dial.append(to_save)
        dialogues.append(dial)
    return dialogues


def normalize(text, replacements):
    # lower case every word
    text = text.lower()

    # replace white spaces in front and end
    text = re.sub(r"^\s*|\s*$", "", text)

    # hotel domain pfb30
    text = re.sub(r"b&b", "bed and breakfast", text)
    text = re.sub(r"b and b", "bed and breakfast", text)

    # weird unicode bug
    text = re.sub("(\u2018|\u2019)", "'", text)

    # replace st.
    text = text.replace(";", ",")
    text = re.sub(r"$\/", "", text)
    text = text.replace("/", " and ")

    # replace other special characters
    text = text.replace("-", " ")
    text = re.sub(r'["\<>@\(\)]', "", text)  # remove

    # insert white space before and after tokens:
    for token in ["?", ".", ",", "!"]:
        text = insertSpace(token, text)

    # insert white space for 's
    text = insertSpace("'s", text)

    # replace it's, does't, you'd ... etc
    text = re.sub("^'", "", text)
    text = re.sub(r"'$", "", text)
    text = re.sub(r"'\s", " ", text)
    text = re.sub(r"\s'", " ", text)
    for fromx, tox in replacements:
        text = " " + text + " "
        text = text.replace(fromx, tox)[1:-1]

    # remove multiple spaces
    text = re.sub(" +", " ", text)

    # concatenate numbers
    tokens = text.split()
    i = 1
    while i < len(tokens):
        if re.match(r"^\d+$", tokens[i]) and re.match(r"\d+$", tokens[i - 1]):
            tokens[i - 1] += tokens[i]
            del tokens[i]
        else:
            i += 1
    text = " ".join(tokens)
    return text


def insertSpace(token, text):
    sidx = 0
    while True:
        sidx = text.find(token, sidx)
        if sidx == -1:
            break
        if (
            sidx + 1 < len(text)
            and re.match("[0-9]", text[sidx - 1])
            and re.match("[0-9]", text[sidx + 1])
        ):
            sidx += 1
            continue
        if text[sidx - 1] != " ":
            text = text[:sidx] + " " + text[sidx:]
            sidx += 1
        if sidx + len(token) < len(text) and text[sidx + len(token)] != " ":
            text = text[: sidx + 1] + " " + text[sidx + 1 :]
        sidx += 1
    return text


TOKEN_EXCEPTIONS = {
    "childs": "children",
    "businesss": "businesses",
    "inchs": "inches",
}
PATTERN_EXCEPTIONS = {"breakfasts": "b&bs"}


def invert_trade_subtokenization(
    original_seq: str,
    trade_seq: str,
    token_exceptions: Dict[str, str] = TOKEN_EXCEPTIONS,
    pattern_exceptions: Dict[str, str] = PATTERN_EXCEPTIONS,
    subtoken_special_chrs: List[str] = [" -", " _"],
) -> str:
    """
    Invert all trade subtokenizations in a string given the original sequence.

    Arguments
    ---------
    original_seq: str
        The original sequence.
    trade_seq: str
        The sequence that has been pre-processed by trade.
    token_exceptions: dict, keys are str, values are str
        A dictionary to map merged token to their correct counterpart. E.g.
        child -s is merged into childs, but the correct token is children.
    pattern_exceptions: dict, keys are str, values are str
        A dictionary to map patterns to their correct counterpart. E.g.
        after the pre-processing "b&bs" is mapped to "bed and breakfast -s",
        making the search of breakfasts impossible if not handled by such
        exceptions.
    subtoken_special_chrs: list of str
        List containing the special characters that are used for subtokens.

    Returns
    -------
    corrected_seq: str
        The sequence corrected, i.e. subtokens replaced by tokens.
    """
    regex = "|".join(subtoken_special_chrs)
    subtoken_pieces = re.split(regex, trade_seq, maxsplit=1)
    search_after: int = 0
    while len(subtoken_pieces) > 1:
        # example: 'the wind is moderate -ly strong'
        # split: ['the wind is moderate ', 'ly strong']
        # split[0]: 'the wind is moderate' --> split on whitespace ['the', 'wind', 'is', 'moderate']
        left_side = subtoken_pieces[0].split()
        subtoken_left = left_side[-1]
        # split[1]: 'ly strong' --> split on whitespace ['ly', 'strong']
        right_side = subtoken_pieces[1].split()
        subtoken_right = right_side[0]
        # try merging the subtoken parts to form a token, i.e. moderate + ly
        token = "".join([subtoken_left, subtoken_right])

        if token in token_exceptions:
            # if you match an exception, replace the token with the exception
            token = token_exceptions[token]

        # assume there are no tokens on left and right side of the subtokens' pieces
        left_token = None  # if token is at the beginnig
        right_token = None  # if token is at the end
        # try looking for them
        if len(left_side) > 1:
            left_token = left_side[-2]
        if len(right_side) > 1:
            right_token = right_side[1]

        # start from a complete match, and progressively remove left and right
        # tokens to counter TRADE preprocessing of some tokens
        # The order is
        # 1. True, True
        # 2. True, False
        # 3. False, True
        # 4. False, False
        # basically, at the end you try looking only for the merged token
        pattern: str = ""
        idx: int = -1
        for use_left, use_right in product((True, False), (True, False)):
            pattern = token
            if (left_token is not None) and use_left:
                pattern = " ".join([left_token, pattern])
            if right_token is not None and use_right:
                pattern = " ".join([pattern, right_token])

            # check if the pattern is in the exceptions
            if pattern in pattern_exceptions:
                pattern = pattern_exceptions[pattern]
            # Search the pattern
            idx = original_seq[search_after:].lower().find(pattern)
            if idx > -1:
                break

        error: str = f"""
            Pattern search failed in the following case:
            PATTERN =  \t{pattern}
            LEFT SIDE = \t{left_side}
            RIGHT SIDE = \t{right_side}
            ORIG SEQ = \t{original_seq[search_after:]}

            This may be due to further TRADE pre-processing, or not correct merging operation.
            To solve this, add a special rule for the token that breaks the code either as a
            token_exception or a pattern_exception.
        """

        assert idx > -1, error
        # move the index to avoid perfect matches with the same token
        # TODO is probably better to move it of len(left_token + token) or
        # len(token) depending on the match
        search_after += idx + 1
        # reconstruct the sentence with the matched pattern
        trade_seq = " ".join([*left_side[:-1], token, *right_side[1:]])

        # try splitting the sentence again and repeat the process
        subtoken_pieces = re.split(regex, trade_seq, maxsplit=1)
    # Good, no subtokens found: return trade seq
    return trade_seq


def get_json_object(data_path: str) -> dict:
    """
    A function to read a json object and return the python
    dictionary associated to it.

    Arguments
    ---------
    data_path: str
        Path to a json file.

    Returns
    -------
    loaded_json: dict
        A loaded json object.
    """
    with open(data_path, "r") as data_file:
        data = json.load(data_file)

    return data


def create_dialogue_dataset(
    dialogues: List[List[Dict[str, Any]]]
) -> Dict[str, Dict[str, Any]]:
    """
    Creates a dialogue dataset starting from a set of dialogues. Each
    entry of the dataset contains the dialogue history and the system
    reply in response to that.

    Arguments
    ---------
    dialogues: list of list of dict, keys are str, values could be anything
        List of dialogues. Each dialogue is a list of turns. Each turn is a
        dict containing dialogue_idx, turn_idx, and the corrected sequence.
    kwargs: any
        Additional arguments for the current function.

    Returns
    -------
    dataset: Dict[str, Dict[str, Any]]
        Dataset, keys are str, values are dictionaries containing the
        dialogue history and the system reply.
    """

    def create_dialogue_dataset_entry(
        turn: Dict[str, Any], history: List[str]
    ) -> Optional[Dict[str, Any]]:
        """
        Creates an entry if the current turn id is odd. An entry is
        composed of the history, which contains the previous turns
        of the current dialogue, and the reply of the system.

        Arguments
        ---------
        turn: dict, keys are str, values could be anything
            A dict containing, the dialogue id, the turn id, the sequence,
            and the mean length.
        replacements_path: str
            Path to TRADE file containing (from, to) pairs, one per line.
        kwargs: any
            Additional arguments for the current function.

        Returns
        -------
        entry: optional dict, keys are str, values could be anything
            Entry of the dialogue dataset. It is a dict containing the history
            of the dialogue, i.e. a list of turns, the reply of the system,
            i.e. a turn, and the mean length.
        """

        turn_idx = turn["turn_idx"]
        entry: Optional[Dict[str, Any]] = None
        if turn_idx % 2 == 0:
            # user turn, simply append it to the history
            user_seq: str = turn["sequence"]
            history.append(user_seq)
        elif turn_idx % 2 == 1:
            # system turn, create the dataset entry, and the append it to the history
            system_seq: str = turn["sequence"]
            history_mean_length = mean([len(turn) for turn in history])
            entry = {
                "history": history.copy(),
                "reply": system_seq,
                "length": history_mean_length + len(system_seq),
            }
            history.append(system_seq)
        return entry

    dataset: Dict[str, Dict[str, Any]] = {}
    for dialogue in tqdm(dialogues, desc="Creating dataset"):
        history: List[str] = []
        for turn in dialogue:
            # custom function to create a dataset entry
            dataset_entry = create_dialogue_dataset_entry(turn, history)
            # custom function to create a dataset key
            key = create_entry_key(turn)
            if dataset_entry is not None:
                dataset[key] = dataset_entry
    return dataset


def create_entry_key(turn: Dict[str, Any]) -> str:
    """
    Creates the entry key for a given entry by considering dialogue id
    and turn id for the given turn.

    Arguments
    ---------
    turn: dict, keys are str, values could be anything
        A dict containing, the dialogue id, the turn id, the sequence,
        and the mean length.
    kwargs: any
        Additional arguments for the current function.

    Returns
    -------
    key: str
        The key for the given turn.
    """
    dialogue_idx = turn["dialogue_idx"]
    turn_idx = turn["turn_idx"]
    return f"{dialogue_idx}_{turn_idx}"


def save_dialogue_dataset(
    dataset: Dict[str, Dict[str, Any]], save_file: str
) -> None:
    """
    Saves the dialogue dataset at dst_folder/file_name as a json file.

    Arguments
    ---------
    dataset: Dict[str, Dict[str, Any]]
        Dataset, keys are str, values are dictionaries containing the
        dialogue history, the system reply, and the mean length.
    save_file: str
        Path to the folder where the original MultiWOZ dataset is stored.
    """
    with open(save_file, "w") as f:
        json.dump(dataset, f, indent=4)
