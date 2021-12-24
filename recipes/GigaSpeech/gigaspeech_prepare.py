"""
Data preparation.

Download: http://www.openslr.org/12

Author
------
Abdel HEBA 2021
"""

import os
import csv
import json
import random
from collections import Counter
import logging
import torchaudio
from speechbrain.utils.data_utils import download_file, get_all_files
from speechbrain.dataio.dataio import (
    load_pkl,
    save_pkl,
    merge_csvs,
)

logger = logging.getLogger(__name__)
OPT_FILE = "opt_gigaspeech_prepare.pkl"
SAMPLERATE = 16000
GRABAGE_UTTERANCE_TAGS = ["<SIL>", "<MUSIC>", "<NOISE>", "<OTHER>"]
PUNCTUATION_TAGS = ["<COMMA>", "<EXCLAMATIONPOINT>", "<PERIOD>", "<QUESTIONMARK>"]
SPLITS = ["train","dev","test"]
TRAIN_SUBSET = ["XS","S", "M", "L", "XL"]


def prepare_gigaspeech(
    data_folder,
    save_folder,
    train_subset="XL",
    dev_subset="dev",
    test_subset="test",
    json_file="GigaSpeech.json",
    skip_opus2wav_convertion=True,
    skip_prep=False,
):
    """
    This class prepares the csv files for the GigaSpeech dataset.
    Github link: https://github.com/SpeechColab/GigaSpeech

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original GigaSpeech dataset is stored.
    save_folder : str
        The directory where to store the csv files.
    train_subset : str
        Default "XL", one of ["XS","S", "M", "L", "XL"] subset.
    create_lexicon: bool
        If True, it outputs csv files containing mapping between grapheme
        to phonemes. Use it for training a G2P system.
    skip_opus2wav_convertion: bool
        If True, convert opus file to wav is skipped.
    skip_prep: bool
        If True, data preparation is skipped.


    Example
    -------
    >>> data_folder = 'datasets/GigaSpeech'
    >>> train_subset = 'XL'
    >>> save_folder = 'gigaspeech_prepared'
    >>> prepare_librispeech(data_folder, save_folder, train_subset)
    """

    if skip_prep:
        return
    conf = {
        "data_folder": data_folder,
        "train_subset": train_subset,
        "json_file": json_file,
    }

    # Other variables
    # Saving folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    save_opt = os.path.join(save_folder, OPT_FILE)

    # Check if this phase is already done (if so, skip it)
    if skip(SPLITS, save_folder, conf):
        logger.info("Skipping preparation, completed in previous run.")
        return
    else:
        logger.info("Data_preparation...")

    # Additional checks to make sure that data folder contains the full Gigaspeech files
    check_gigaspeech_folders(data_folder, json_file)
    
    # Prepare GigaSpeech
    json_metadata = os.path.join(data_folder, json_file)
    # Setting csv path for the train, dev, test subsets
    csv_lines = [["ID", "audio", "start", "stop", "duration", "wrd"]]
    train_csv_file = os.path.join(save_folder, "train.csv")
    dev_csv_file = os.path.join(save_folder, "dev.csv")
    test_csv_file = os.path.join(save_folder, "test.csv")
    logger.info("Creating csv lists for  trai, dev, and test subsets.")
    train_csv = list(csv_lines)
    dev_csv = list(csv_lines)
    test_csv = list(csv_lines)
    # Create csv lists for all subsets
    with open(json_metadata) as json_file:
        logger.info("Loading GigaSpeech Meta-Data, this may take few time..")
        data = json.load(json_file)
        logger.info("Loading complete, preparing CSV files..")
        for meta_data in data["audios"]:
            #try:
            audio_path = os.path.join(data_folder, meta_data['path'])
            audio_path = os.path.realpath(audio_path)
            if not skip_opus2wav_convertion:
                audio_path = convert_opus2wav(audio_path)
            aid = meta_data["aid"]
            segments_list = meta_data["segments"]
            audio_subsets = meta_data["subsets"]
            duration = meta_data["duration"]
            #assert(check_file(audio_path) == True)
            #except AssertionError:
            #logger.info("Warning: " + aid + " something is wrong, maybe AssertionError, skipped")
            #continue
            for segment_file in segments_list:
                try:
                    sid = segment_file["sid"]
                    start_time = segment_file["begin_time"]
                    end_time = segment_file["end_time"]
                    duration = end_time - start_time
                    text = segment_file["text_tn"]
                    text = filter_text(text)
                    if text == "":
                        logger.info("Warning: " + sid + " has empty entry, skipping it..")
                        continue
                    segment_subsets = segment_file["subsets"]
                    # Long audio can have different subsets
                    if "{DEV}" in segment_file["subsets"]:
                        dev_csv.append([sid, audio_path, start_time, end_time, duration, text])
                    elif "{TEST}" in segment_file["subsets"]:
                        test_csv.append([sid, audio_path, start_time, end_time, duration, text])
                    else:
                        if not "{%s}" % (train_subset) in segment_file["subsets"]:
                            continue
                        train_csv.append([sid, audio_path, start_time, end_time, duration, text])
                except:
                    logger.info("Warning: " + aid + "something is wrong.")
                    continue
    # Write CSVs...
    write_csv(train_csv, train_csv_file)
    write_csv(dev_csv, dev_csv_file)
    write_csv(test_csv, test_csv_file)

    # saving options
    save_pkl(conf, save_opt)

def convert_opus2wav(audio_path):
    wav_path = str(audio_path).replace(".opus", ".wav")
    if not os.path.exists(wav_path):
        logger.info("Convert " + audio_path + " to wav file.")
        command = ['ffmpeg', '-i', f'{name}.opus', '-ar', '16000', f'{name}.wav']
        try:
            subprocess.run(command, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            raise e
    else:
        logger.info(wav_path + " already exist, skipping convertion.")
        

def filter_text(text):
    """
    This function Filter text from grabage and punctuations

    Arguments
    ---------
    text: str

    Returns
    ---------
    text: str
    """ 
    for tag in GRABAGE_UTTERANCE_TAGS + PUNCTUATION_TAGS:
        text = text.replace(tag,"")
    # delete spaces
    text = " ".join(text.split())
    return text

def write_csv(
    list_data, csv_file,
):
    """
    Create the dataset csv file given a list of wav files.

    Arguments
    ---------
    list_data : str
        The list of utterrances of a given data split.
    csv_file : str
        Location of the folder for storing the csv.

    Returns
    -------
    None
    """

    # Preliminary prints
    msg = "Creating csv lists in  %s..." % (csv_file)
    logger.info(msg)

    # Writing the csv_lines
    with open(csv_file, mode="w") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )

        for line in list_data:
            csv_writer.writerow(line)

    # Final print
    msg = "%s successfully created!" % (csv_file)
    logger.info(msg)

def skip(splits, save_folder, conf):
    """
    Detect when the librispeech data prep can be skipped.

    Arguments
    ---------
    splits : list
        A list of the splits expected in the preparation.
    save_folder : str
        The location of the seave directory
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

def check_gigaspeech_folders(data_folder, json_file="GigaSpeech.json", audio_folder="audio"):
    """
    Check if the data folder actually contains the GigaSpeech dataset.

    If it does not, an error is raised.

    Returns
    -------
    None

    Raises
    ------
    OSError
        If GigaSpeech is not found at the specified path.
    """
    # Checking if "GigaSpeech.json" exist
    json_gigaspeech = os.path.join(data_folder, json_file)
    check_file(json_gigaspeech)
    
    # Check if audio folders exist
    for folder_subset in ["audiobook", "podcast", "youtube"]:
        audio_subset = os.path.join(data_folder, audio_folder, folder_subset)
        if not os.path.exists(audio_subset):
            err_msg = (
                "the file %s does not exist (it is expected in the "
                "Gigaspeech dataset)" % audio_subset
            )
            raise OSError(err_msg)

def check_file(path):
    # Check if file exist
    if not os.path.exists(path):
        err_msg = (
            "the opus file %s does not exist (it is expected in the "
            "Gigaspeech dataset)" % path
        )
        raise OSError(err_msg)