"""
Data preparation.

Download: http://www.robots.ox.ac.uk/~vgg/data/voxceleb/
"""

import os
import csv
import logging
import glob
import random

from speechbrain.data_io.data_io import (
    read_wav_soundfile,
    load_pkl,
    save_pkl,
)

logger = logging.getLogger(__name__)
OPT_FILE = "opt_voxceleb1_prepare.pkl"
TRAIN_CSV = "train.csv"
DEV_CSV = "dev.csv"
SAMPLERATE = 16000


def prepare_voxceleb1(
    data_folder,
    save_folder,
    splits=["train", "dev"],
    split_ratio=[90, 10],
    seg_dur=300,
    vad=False,
    rand_seed=1234,
):
    """
    Prepares the csv files for the Voxceleb1 dataset.

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
    seg_dur : int
        Segment duration of a chunk in milliseconds
    vad : bool
        To perform VAD or not
    rand_seed : int
        random seed

    Example
    -------
    >>> from recipes.VoxCeleb.voxceleb1_prepare import prepare_voxceleb1
    >>> data_folder = 'data/VoxCeleb1/'
    >>> save_folder = 'VoxData/'
    >>> splits = ['train', 'dev']
    >>> split_ratio = [90, 10]
    >>> prepare_voxceleb1(data_folder, save_folder, splits, split_ratio)
    """

    data_folder = os.path.join(data_folder, "wav/")

    # Create configuration for easily skipping data_preparation stage
    conf = {
        "data_folder": data_folder,
        "splits": splits,
        "split_ratio": split_ratio,
        "save_folder": save_folder,
        "vad": vad,
        "seg_dur": seg_dur,
    }

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Setting ouput files
    save_opt = os.path.join(save_folder, OPT_FILE)
    save_csv_train = os.path.join(save_folder, TRAIN_CSV)
    save_csv_dev = os.path.join(save_folder, DEV_CSV)

    # Check if this phase is already done (if so, skip it)
    if skip(splits, save_folder, conf):
        logger.debug("Skipping preparation, completed in previous run.")
        return

    # Additional checks to make sure the data folder contains VoxCeleb data
    _check_voxceleb1_folders(data_folder)

    msg = "\tCreating csv file for the VoxCeleb1 Dataset.."
    logger.debug(msg)

    # Split data into 90% train and 10% validation (verification split)
    wav_lst_train, wav_lst_dev = _get_utt_split_lists(data_folder, split_ratio)

    # Creating csv file for training data
    if "train" in splits:
        prepare_csv(
            SAMPLERATE, seg_dur, wav_lst_train, save_csv_train,
        )

    if "dev" in splits:
        prepare_csv(
            SAMPLERATE, seg_dur, wav_lst_dev, save_csv_dev,
        )

    # Saving options (useful to skip this phase when already done)
    save_pkl(conf, save_opt)


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


def _check_voxceleb1_folders(data_folder):
    """
    Check if the data folder actually contains the Voxceleb1 dataset.

    If it does not, raise an error.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
    """
    # Checking
    if not os.path.exists(data_folder + "/id10001"):
        err_msg = (
            "the folder %s does not exist (as it is expected in "
            "the Voxceleb dataset)" % (data_folder + "/id*")
        )
        raise FileNotFoundError(err_msg)


# Used for verification split
def _get_utt_split_lists(data_folder, split_ratio):
    """
    Tot. number of speakers = 1211.
    Splits the audio file list into train and dev.
    This function is useful when using verification split
    """
    audio_files_list = [
        f for f in glob.glob(data_folder + "**/*.wav", recursive=True)
    ]  # doesn't take much time

    random.shuffle(audio_files_list)
    train_lst = audio_files_list[
        : int(0.01 * split_ratio[0] * len(audio_files_list))
    ]
    dev_lst = audio_files_list[
        int(0.01 * split_ratio[0] * len(audio_files_list)) :
    ]

    return train_lst, dev_lst


def _get_chunks(seg_dur, audio_id, audio_duration):
    """
    Returns list of chunks
    """
    num_chunks = int(audio_duration * 100 / seg_dur)  # all in milliseconds

    chunk_lst = [
        audio_id + "_" + str(i * seg_dur) + "_" + str(i * seg_dur + seg_dur)
        for i in range(num_chunks)
    ]

    return chunk_lst


def prepare_csv(
    samplerate, seg_dur, wav_lst, csv_file, vad=False,
):
    """
    Creates the csv file given a list of wav files.

    Arguments
    ---------
    wav_lst : list
        The list of wav files of a given data split.
    csv_file : str
        The path of the output csv file
    vad : bool
        Perform VAD. True or False

    Returns
    -------
    None
    """

    msg = '\t"Creating csv lists in  %s..."' % (csv_file)
    logger.debug(msg)

    # Format used: example5, 1.000, $data_folder/example5.wav, wav, start:10000 stop:26000, spk05, string,
    csv_output = [
        [
            "ID",
            "duration",
            "wav",
            "wav_format",
            "wav_opts",
            "spk_id",
            "spk_id_format",
            "spk_id_opts",
        ]
    ]

    # For assiging unique ID to each chunk
    my_sep = "--"
    entry = []
    # Processing all the wav files in the list
    for wav_file in wav_lst:

        # Getting sentence and speaker ids
        [spk_id, sess_id, utt_id] = wav_file.split("/")[-3:]
        audio_id = my_sep.join([spk_id, sess_id, utt_id.split(".")[0]])

        # Reading the signal (to retrieve duration in seconds)
        signal = read_wav_soundfile(wav_file)
        audio_duration = signal.shape[0] / samplerate

        uniq_chunks_list = _get_chunks(seg_dur, audio_id, audio_duration)

        for chunk in uniq_chunks_list:
            s, e = chunk.split("_")[-2:]
            start_sample = int(int(s) / 100 * samplerate)
            end_sample = int(int(e) / 100 * samplerate)

            start_stop = (
                "start:" + str(start_sample) + " stop:" + str(end_sample)
            )

            audio_file_sb = (
                "$data_folder/wav/" + spk_id + "/" + sess_id + "/" + utt_id
            )

            # Composition of the csv_line
            csv_line = [
                chunk,
                str(seg_dur / 100),
                audio_file_sb,
                "wav",
                start_stop,
                spk_id,
                "string",
                " ",
            ]

            entry.append(csv_line)

    # Shuffling at chunk level
    random.shuffle(entry)
    csv_output = csv_output + entry

    # Writing the csv lines
    with open(csv_file, mode="w") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )

        for line in csv_output:
            csv_writer.writerow(line)

    # Final prints
    msg = "\t%s Sucessfully created!" % (csv_file)
    logger.debug(msg)
