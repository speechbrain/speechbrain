"""
LJspeech data preparation.
Download: https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2

Authors
 * Yingzhi WANG 2022
"""

import os
import csv
import logging
import random
from speechbrain.dataio.dataio import (
    load_pkl,
    save_pkl,
)

logger = logging.getLogger(__name__)
OPT_FILE = "opt_ljspeech_prepare.pkl"
METADATA_CSV = "metadata.csv"
TRAIN_CSV = "train.csv"
DEV_CSV = "dev.csv"
TEST_CSV = "test.csv"
WAVS = "wavs"

def prepare_ljspeech(
    data_folder,
    save_folder,
    splits=["train", "dev"],
    split_ratio=[90, 10],
    seed=1234,
    skip_prep=False,
):
    """
    Prepares the csv files for the LJspeech datasets.
    Arguments
    ---------
    data_folder : str
        Path to the folder where the original VoxCeleb dataset is stored.
    save_folder : str
        The directory where to store the csv files.
    splits : list
        List of splits to prepare from ['train', 'dev']
    split_ratio : list
        List if int for train and validation splits
    skip_prep: Bool
        If True, skip preparation.
    seed : int
        Random seed
    Example
    -------
    >>> from recipes.VoxCeleb.voxceleb1_prepare import prepare_voxceleb
    >>> data_folder = 'data/LJspeech/'
    >>> save_folder = 'save/'
    >>> splits = ['train', 'dev']
    >>> split_ratio = [90, 10]
    >>> seed = 1234
    >>> prepare_voxceleb(data_folder, save_folder, splits, split_ratio, seed)
    """
    # setting seeds for reproducible code.
    random.seed(seed)

    if skip_prep:
        return
    # Create configuration for easily skipping data_preparation stage
    conf = {
        "data_folder": data_folder,
        "splits": splits,
        "split_ratio": split_ratio,
        "save_folder": save_folder,
        "seed": seed,
    }

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Setting ouput files
    meta_csv = os.path.join(data_folder, METADATA_CSV)
    wavs_folder = os.path.join(data_folder, WAVS)

    save_opt = os.path.join(save_folder, OPT_FILE)
    save_csv_train = os.path.join(save_folder, TRAIN_CSV)
    save_csv_dev = os.path.join(save_folder, DEV_CSV)
    save_csv_test = os.path.join(save_folder, TEST_CSV)

    # Check if this phase is already done (if so, skip it)
    if skip(splits, save_folder, conf):
        logger.info("Skipping preparation, completed in previous run.")
        return

    # Additional check to make sure metadata.csv and wavs folder exists
    assert os.path.exists(meta_csv), "metadata.csv does not exist"
    assert os.path.exists(wavs_folder), "wavs/ folder does not exist"

    msg = "\tCreating csv file for ljspeech Dataset.."
    logger.info(msg)

    data_split, meta_csv = split_sets(
        data_folder, splits, split_ratio
    )

    # Prepare csv
    if "train" in splits:
        prepare_csv(data_split["train"], save_csv_train, wavs_folder, meta_csv)
    if "dev" in splits:
       prepare_csv(data_split["dev"], save_csv_dev, wavs_folder, meta_csv)
    if "test" in splits:
        prepare_csv(data_split["test"], save_csv_test, wavs_folder, meta_csv)

    save_pkl(conf, save_opt)

def skip(splits, save_folder, conf):
    """
    Detects if the ljspeech data_preparation has been already done.
    If the preparation has been done, we can skip it.
    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """
    # Checking csv files
    skip = True

    split_files = {
        "train": TRAIN_CSV,
        "dev": DEV_CSV,
    }
    for split in splits:
        if not os.path.isfile(os.path.join(save_folder, split_files[split])):
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

def split_sets(data_folder, splits, split_ratio):
    """Randomly splits the wav list into training, validation, and test lists.
    Note that a better approach is to make sure that all the classes have the
    same proportion of samples for each speaker.
    Arguments
    ---------
    wav_list : list
        list of all the signals in the dataset
    split_ratio: list
        List composed of three integers that sets split ratios for train,
        valid, and test sets, respectively.
        For instance split_ratio=[80, 10, 10] will assign 80% of the sentences
        to training, 10% for validation, and 10% for test.
    Returns
    ------
    dictionary containing train, valid, and test splits.
    """
    meta_csv = os.path.join(data_folder, METADATA_CSV)
    csv_reader = csv.reader(open(meta_csv), delimiter='|')

    meta_csv = list(csv_reader)

    index_for_speakers = []
    speaker_id_start = "LJ001"
    index_this_speaker = []
    for i in range(len(meta_csv)) : 
        speaker_id = meta_csv[i][0].split("-")[0]
        if speaker_id == speaker_id_start : 
            index_this_speaker.append(i)
            if i==len(meta_csv)-1:
                index_for_speakers.append(index_this_speaker)
        else : 
            index_for_speakers.append(index_this_speaker)
            speaker_id_start = speaker_id
            index_this_speaker = [i]
    
    data_split = {}
    for i, split in enumerate(splits):
        data_split[split] = []
        for speaker in index_for_speakers:
            if split == "train" : 
                random.shuffle(speaker)
                n_snts = int(len(speaker) * split_ratio[i] / sum(split_ratio))
                data_split[split].extend(speaker[0:n_snts])
                del speaker[0:n_snts]
            else : 
                data_split[split].extend(speaker)
    return data_split, meta_csv

def prepare_csv(seg_lst, csv_file, wavs_folder, csv_reader):
    """
    Creates the csv file given a list of csv indexes.

    Arguments
    ---------
    seg_list : list
        The list of csv indexes of a given data split.
    csv_file : str
        Output csv path
    wavs_folder : 
        LJspeech wavs folder
    csv_reader : 
        LJspeech metadata (csv.reader)
    Returns
    -------
    None
    """
    csv_output_head = [["ID", "wav", "segment"]]

    entry = []
    for index in seg_lst:
        id = list(csv_reader)[index][0]
        wav = os.path.join(wavs_folder, f"{id}.wav")
        csv_line = [
            id,
            wav,
            True if "train" in csv_file else False,
            ]
        entry.append(csv_line)

    csv_output = csv_output_head + entry
    # Writing the csv lines
    with open(csv_file, mode="w") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        for line in csv_output:
            csv_writer.writerow(line)

    # Final prints
    msg = "\t%s Sucessfully created!" % (csv_file)
    logger.info(msg)