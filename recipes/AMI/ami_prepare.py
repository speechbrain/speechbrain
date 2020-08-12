"""
Data preparation.

Download: http://groups.inf.ed.ac.uk/ami/download/

Prepares csv from manual annotations "segments/" using RTTM format (Oracle VAD).
"""

import os
import logging


from ami_split import get_AMI_split


logger = logging.getLogger(__name__)
OPT_FILE = "opt_ami_prepare.pkl"
TRAIN_CSV = "train.csv"
DEV_CSV = "dev.csv"
TEST_CSV = "test.csv"
SAMPLERATE = 16000

## mini
## Prepare GT RTTMs from XMLs

## get splits
## verify data directories
## (just throw warnings if not something is missing) BUT proceed
## Use GT_RTTM to SCP to prepare csv
## utt_ID, duration, wav, wav_format, spk_id, spk_id_format, spk_id_opts, rec_id
## Woah! looks like the most complicated dataset to have in one csv

## remember todo (later)
## skip data prep
## verify data present or not

### STEPS:
# Get Split and verify directory
# Convert all XMLs to RTTM (easier to parse)

# Code Flow:   segments.xml -> NIST.rttm -> CSV
# Code Design: loader_xml(), read_write_rttm(), prune_overlaps(), create_labs()

# split can be either of these: scenario-only, full-corpus, full-corpus-asr
# mic_types : hm, ihm, sdm, array1, array2
# vad_type : system / oracle


def prepare_ami(
    data_folder,
    save_folder,
    split_type="full_corpus",
    mic_type="hm",
    vad_type="oracle",
    subseg_dur=300,
):
    """
    Prepares the csv files for the AMI dataset.

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original VoxCeleb dataset is stored.
    save_folder : str
        The directory where to store the csv files.
    """

    # Create configuration for easily skipping data_preparation stage
    conf = {
        "data_folder": data_folder,
        "save_folder": save_folder,
        "split_type": split_type,
        "mic_type": mic_type,
        "vad": vad_type,
        "subseg_dur": subseg_dur,
    }

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Setting ouput files
    save_opt = os.path.join(save_folder, OPT_FILE)  # noqa F841
    save_csv_train = os.path.join(save_folder, TRAIN_CSV)  # noqa F841
    save_csv_dev = os.path.join(save_folder, DEV_CSV)  # noqa F841
    save_csv_test = os.path.join(save_folder, TEST_CSV)  # noqa F841

    # Check if this phase is already done (if so, skip it)
    splits = ["train", "dev", "test"]
    if skip(splits, save_folder, conf):
        logger.debug("Skipping preparation, completed in previous run.")
        return

    msg = "\tCreating csv file for the VoxCeleb1 Dataset.."
    logger.debug(msg)

    # Split data into 90% train and 10% validation (verification split)
    # wav_lst_train, wav_lst_dev = _get_utt_split_lists(data_folder, split_ratio)

    train_set, dev_set, test_set = get_AMI_split(split_type)

    # Creating csv file for training data
    prepare_csv(train_set, dev_set, test_set)

    # save_pkl(conf, save_opt)


def prepare_csv():
    pass


def skip(splits, save_folder, conf):
    """
    Detects if the timit data_preparation has been already done.
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
        "test": TEST_CSV,
    }
    for split in splits:
        if not os.path.isfile(os.path.join(save_folder, split_files[split])):
            skip = False

    #  Checking saved options
    save_opt = os.path.join(save_folder, OPT_FILE)
    if skip is True:
        if os.path.isfile(save_opt):
            opts_old = load_pkl(save_opt)  # noqa F821
            if opts_old == conf:
                skip = True
            else:
                skip = False
        else:
            skip = False

    return skip
