"""
Data preparation.

Download: https://github.com/SpeechColab/GigaSpeech

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
from tqdm.contrib import tqdm
import subprocess
import io
import multiprocessing as mp
from itertools import islice
import webdataset as wds

logger = logging.getLogger(__name__)
OPT_FILE = "opt_gigaspeech_prepare.pkl"
SAMPLERATE = 16000
GRABAGE_UTTERANCE_TAGS = ["<SIL>", "<MUSIC>", "<NOISE>", "<OTHER>"]
PUNCTUATION_TAGS = ["<COMMA>", "<EXCLAMATIONPOINT>", "<PERIOD>", "<QUESTIONMARK>", "<blank>"]
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
    remove_original= False,
    maxsize = 3e9,
    maxcount = 10000,
    num_proc=40,
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
    logger.info("Creating train, dev, and test subsets.")
    # Create csv lists for all subsets
    #with open(json_metadata) as json_file:
    with open("GigaSpeech.json") as json_file:
        logger.info("Loading GigaSpeech Meta-Data, this may take few time..")
        data = json.load(json_file)
        dict_train_aid = dict()
        dict_valid_aid = dict()
        dict_test_aid = dict()
        for meta_data in tqdm(data["audios"], dynamic_ncols=True, disable=False):
            audio_path = os.path.join(data_folder, meta_data['path'])
            audio_path = os.path.realpath(audio_path)
            aid = meta_data["aid"]
            list_train_aid = []
            list_valid_aid = []
            list_test_aid = []
            sample_rate = int(meta_data["sample_rate"])
            segments_list = meta_data["segments"]
            audio_subsets = meta_data["subsets"]
            for segment_file in segments_list:
                sid = segment_file["sid"]
                start_time = float(segment_file["begin_time"])
                end_time = float(segment_file["end_time"])
                duration = (end_time - start_time)
                text = segment_file["text_tn"]
                text = filter_text(text)
                if text == "":
                    logger.warning("Warning: " + sid + " has empty entry, skipping it..")
                    continue
                segment_subsets = segment_file["subsets"]
                # Long audio can have different subsets
                if "{DEV}" in segment_file["subsets"]:
                    list_valid_aid.append([sid, start_time, duration, text, "valid"])
                elif "{TEST}" in segment_file["subsets"]:
                    list_test_aid.append([sid, start_time, duration, text, "test"])
                else:
                    if not "{%s}" % (train_subset) in segment_file["subsets"]:
                        continue
                    if "{L}" in segment_file["subsets"]:
                        continue
                    list_train_aid.append([sid, start_time, duration, text, "train"])
            if len(list_train_aid) > 0:
                dict_train_aid[audio_path] = list_train_aid
            if len(list_valid_aid) > 0:
                dict_valid_aid[audio_path] = list_valid_aid
            if len(list_test_aid) > 0:
                dict_test_aid[audio_path] = list_test_aid
    logger.info("Loading complete, preparing Shards files..")
    # Write shards with MultiProcessing..
    max_proc = mp.cpu_count() * 2
    if num_proc > max_proc: 
        logger.info("num_proc (%s) is higher than max proc (%s) of your instance " % (str(num_proc), str(max_proc)))
        logger.info("Set num_proc to %s" % (str(max_proc)))
        num_proc = max_proc
    processes = []
    # Save shards for training
    logger.info("Preparing Shards files for Train set..")
    split_folder = os.path.join(data_folder,"train")
    if not os.path.exists(split_folder):
        os.makedirs(split_folder)
    for id_proc, dict_proc in split_dict(dict_train_aid, num_proc):
        proc = mp.Process(
            target=write_shards,
            kwargs={
                "dict_per_proc" : dict_proc,
                "id_proc": id_proc,
                "save_folder": split_folder,
                "maxsize": maxsize,
                "maxcount": maxcount,
                "remove_original" : remove_original,
            }
        )
        proc.start()
        processes.append(proc)
    for proc in processes:
        proc.join()

    logger.info("Preparing Shards files for Valid/Test sets..")    
    processes = []
    valid_folder = os.path.join(data_folder,"valid")
    test_folder = os.path.join(data_folder,"test")
    if not os.path.exists(valid_folder):
        os.makedirs(valid_folder)
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)
    #for dict_subset, folder_subset, in zip([dict_valid_aid,dict_test_aid],[valid_folder, test_folder]):
    #    for id_proc, dict_proc in split_dict(dict_subset, num_proc):
    #        proc = mp.Process(
    #            target=write_shards,
    #            kwargs={
    #                "dict_per_proc" : dict_proc,
    #                "id_proc": id_proc,
    #                "save_folder": folder_subset,
    #                "maxsize": maxsize,
    #                "maxcount": maxcount,
    #                "remove_original" : remove_original,
    #            }
    #        )
    #        proc.start()
    #        processes.append(proc)
    #for proc in processes:
    #    proc.join()

    # saving options
    save_pkl(conf, save_opt)
    

def split_dict(dict_aid, num_proc=1):
    """
        split dict into a (num_proc * sub_dict)
    """
    it = iter(dict_aid)
    num_segs = round(len(dict_aid)/num_proc)
    for i, _ in enumerate(range(0, len(dict_aid), num_segs)):
        i = '%03d' % (i)
        d = {k:dict_aid[k] for k in islice(it, num_segs)}
        yield i, d

def write_shards(
    dict_per_proc, id_proc, save_folder, maxsize=1e9, maxcount=5000, remove_original=False
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
    msg = "Creating shards for process %s in  %s..." % (id_proc, save_folder)
    pattern = os.path.join(save_folder, f"GigaSpeech-{id_proc}-%06d.tar")
    logger.info(msg)

    # Writing the csv_lines
    duration=0
    with wds.ShardWriter(pattern, maxsize=int(maxsize), maxcount=int(maxcount)) as sink:
        for audio_path, list_seg in dict_per_proc.items():
            b_stdout = convert_opus2wav(audio_path, resampling="16000")
            wav_buffer = io.BytesIO(b_stdout)
            if b_stdout == -1:
                logger.info("Error converting opus file %s, skipping it." % (audio_path))
                continue
            sig, sr = torchaudio.load(wav_buffer)
            for seg_info in list_seg:
                # GigaSpeech propose to resample sig to 16khz
                seg_buffer = io.BytesIO()
                duration += float(seg_info[2])/3600
                begin = int(float(seg_info[1]) * sr)
                end = begin + int(float(seg_info[2]) * sr)
                sig_seg = sig[0, begin:end]
                torchaudio.save(seg_buffer, sig_seg.unsqueeze(0), sr, format="wav")
                seg_buffer.seek(0)
                dict_seg = {
                        "__key__": seg_info[0], 
                        "wav": seg_buffer.read(), 
                        "text": seg_info[3],
                    }
                # Write the sample to the sharded tar archives.
                sink.write(dict_seg)
            if remove_original:
                os.remove(audio_path)
    # Final print
    msg = "Process %s successfully executed: %5.2f H was processed" % (id_proc, duration)
    logger.info(msg)

def convert_opus2wav(audio_path, resampling="16000", silence_stderr=True):
    if not os.path.exists(audio_path):
        print("problem")
        print(audio_path)
        exit()
    command = ['ffmpeg', '-i', audio_path, '-ar', resampling, "-f", "wav", "pipe:1"]
    try:
        result = subprocess.run(command,
                        stdout=subprocess.PIPE,
                        stdin=subprocess.PIPE,
                        stderr=subprocess.DEVNULL if silence_stderr else None)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return -1
        

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