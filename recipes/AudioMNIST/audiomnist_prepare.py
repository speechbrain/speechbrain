"""
Data preparation for the AudioMNIST dataset

Download: https://github.com/soerenab/AudioMNIST.git

Author
------
Artem Ploujnikov 2020
"""

import logging
import csv
import math
import json
import os
import speechbrain as sb

from glob import glob
from functools import partial
from tqdm.auto import tqdm
from torchaudio import functional as F
from subprocess import list2cmdline
from speechbrain.utils.superpowers import run_shell
from speechbrain.dataio.dataio import (
    load_pkl,
    save_pkl,
)


DEFAULT_SPLITS = ["train", "valid", "test"]
DEFAULT_AUDIOMNIST_REPO = "https://github.com/soerenab/AudioMNIST.git"
DEFAULT_SRC_SAMPLE_RATE = 48000
DEFAULT_TGT_SAMPLE_RATE = 48000
DB_BASE = 10.0
DB_MULTIPLIER = 0.05
OPT_FILE = "opt_audiomnist_prepare.pkl"


logger = logging.getLogger(__name__)


def prepare_audiomnist(
    data_folder,
    save_folder,
    prepare_data_folder="./audiomnist-prepare-data",
    splits=DEFAULT_SPLITS,
    download=True,
    audiomnist_repo=None,
    src_sample_rate=DEFAULT_SRC_SAMPLE_RATE,
    tgt_sample_rate=DEFAULT_TGT_SAMPLE_RATE,
    trim=True,
    trim_threshold=-30.0,
    norm=True,
    process_audio=None,
):
    """Auto-downloads and prepares the AudioMNIST dataset

    Arguments
    ---------
    data_folder: str
        the folder where the original dataset exists

    save_folder: str
        the destination folder

    download: bool
        whether the dataset should be auto-downloaded (enabled by default)

    audiomnist_repo: str
        the URL of the AudioMNIST repository

    src_sample_rate: int
        the source sampling rate

    tgt_sample_rate: int
        the target sampling rate

    trim: bool
        whether to trim silence from the beginning and the end.
        Ignored if process_audio is provided

    trim_threshold: bool
        the trimming threshold, in decibels.
        Ignored if process_audio is provided

    norm: bool
        whether to normalize the amplitude between -1. and 1.
        Ignored if process_audio is provided

    process_audio: callable
        a custom function used to process audio files - instead of
        the standard transform (resample + normalize + trim)

    """
    # Check if the target folder exists. Create it if it does not.
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    conf = {
        "trim_threshold": trim_threshold,
        "norm": norm,
        "tgt_sample_rate": tgt_sample_rate,
    }

    save_opt = os.path.join(save_folder, OPT_FILE)

    if skip(splits, save_folder, conf):
        logger.info("Skipping preparation, completed in previous run.")
        return
    else:
        logger.info("Data_preparation...")

    # Download AudioMNIST if not present
    if not os.path.exists(data_folder) or not os.listdir(data_folder):
        if download:
            if not audiomnist_repo:
                audiomnist_repo = DEFAULT_AUDIOMNIST_REPO
            download_dataset(data_folder, audiomnist_repo)
        else:
            raise ValueError(f"AudioMNIST not found in {data_folder}")

    # Set up the audio preprocessing function
    if not process_audio:
        process_audio = partial(
            process_audio_default,
            src_sample_rate=src_sample_rate,
            tgt_sample_rate=tgt_sample_rate,
            trim=trim,
            trim_threshold=trim_threshold,
            norm=norm,
        )

    # Get file lists for train/valid/test splits
    splits = get_splits(prepare_data_folder, splits)
    digit_lookup_file_name = os.path.join(prepare_data_folder, "digits.csv")

    # Read the digit look-up file providing annotations text and
    # phonemes
    lookup = read_digit_lookup(digit_lookup_file_name)

    # Convert the dataset
    convert_dataset(
        src=data_folder,
        tgt=save_folder,
        splits=splits,
        lookup=lookup,
        process_audio=process_audio,
        sample_rate=tgt_sample_rate,
    )

    # saving options
    save_pkl(conf, save_opt)


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
        if not os.path.isfile(os.path.join(save_folder, split + ".json")):
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


def get_splits(prepare_data_folder, splits):
    """Retrieves the train/valid/test file splits

    Arguments
    ---------
    prepare_data_folder: str
        the path to auxiliary data

    splits: list
        the list of splits to prepare

    Returns
    -------
    result: dict
        a dictionary of file splits
    """
    split_files = {
        split: os.path.join(prepare_data_folder, f"{split}.txt")
        for split in splits
    }
    return {
        split: read_file_list(file_path)
        for split, file_path in split_files.items()
    }


def read_file_list(file_name):
    """Reads a file list with files being listed one per line

    Arguments
    ---------
    file_name: str
        the file_name

    Returns
    -------
    result: lists
        the file list
    """
    with open(file_name) as list_file:
        return [line.strip() for line in list_file]


def download_dataset(data_folder, repo_url):
    """Downloads the dataset from a GIT repository

    Arguments
    ---------
    data_folder: str
        the destination folder
    repo_url: str
        the repository URL
    """
    cmd = list2cmdline(["git", "clone", repo_url, data_folder])
    output, err, return_code = run_shell(cmd)
    if return_code != 0:
        raise DownloadError(output, err)


class DownloadError(Exception):
    """Thrown when a download attempt fails

    Arguments
    ---------
    output: str
        the command output
    err: str
        stderr contents
    """

    FORMAT_MSG = "Unable to download the dataset: {output} - {err}"

    def __init__(self, output, err):
        msg = self.FORMAT_MSG.format(output=output, err=err)
        super().__init__(msg)


SPEAKER_META_MAP = {
    "native speaker": "native_speaker",
    "recordingdate": "recording_date",
    "recordingroom": "recording_room",
}

BOOL_MAP = {"yes": True, "no": False}


def to_bool(value):
    """Converts a yes/no value to a Boolean

    Arguments
    ---------
    value: str
        A string: "yes" or "no

    Returns
    -------
    result: bool
        True if the value is "yes"
        False if the value is "no"
    """
    return BOOL_MAP[value]


def convert_date(value):
    """Converts a date as recorded in AudioMNIST to an ISO
    date string


    Arguments
    ---------
    value: str
        a value, as encountered in AudioMNIST
        Example: 17-06-26-17-57-29

    Returns
    -------
    result: str
        an ISO date string corresponding to the date provided
    """
    year, month, day, hour, minute, second = value.split("-")
    return f"20{year}-{month}-{day}T{hour}:{minute}:{second}"


SPEAKER_META_VALUES_MAP = {
    "native_speaker": to_bool,
    "recording_date": convert_date,
}


def read_meta(file_name):
    """Reads a metadata file

    Arguments
    ---------
    file_name: str
        the metadata file name

    Returns
    -------
    result: dict
        raw metadata
    """
    with open(file_name) as meta_file:
        return json.load(meta_file)


def convert_value(key, value, conversion_map):
    """Converts a value using a map

    Arguments
    ---------
    key: str
        the item key
    value: object
        the value
    conversion_map: dict
        a dictinary with keys corresponding to keys in the original
        dataset and conversion function as values

    Returns
    -------
    value: object
        the converted value (or the original value if no conversin
        function is found in the map)
    """
    conv_fn = conversion_map.get(key)
    if conv_fn:
        value = conv_fn(value)
    return value


def convert_speaker_meta_keys(speaker_meta):
    """Converts the speaker metadata keys to the target format

    Arguments
    ---------
    speaker_meta: dict
        raw speaker metadata

    Results
    -------
    result: dict
        Mapped metadata
    """
    return {
        SPEAKER_META_MAP.get(key, key): value
        for key, value in speaker_meta.items()
    }


def convert_speaker_meta_values(speaker_meta):
    """Convert the speaker metadata values to the target format

    Arguments
    ---------
    speaker_meta: dict
        raw speaker metadata

    Results
    -------
    result: dict
        the converted metadata
    """
    return {
        key: convert_value(key, value, SPEAKER_META_VALUES_MAP)
        for key, value in speaker_meta.items()
    }


def convert_speaker_meta(speaker_meta):
    """Converts speaker metdata to the target format

    Arguments
    ---------
    speaker_meta: dict
        the raw speaker metadata

    Results
    -------
    result: dict
        the converted metadata
    """
    speaker_meta = convert_speaker_meta_keys(speaker_meta)
    speaker_meta = convert_speaker_meta_values(speaker_meta)
    return speaker_meta


def get_wav_files(tgt_split_path):
    """Returns all wave files at the specified path

    Arguments
    ---------
    tgt_split_path: str
        the path to the target data split

    Returns
    -------
    result: list
        a list of file names
    """
    wavs_pattern = os.path.join(tgt_split_path, "**", "*.wav")
    return list(sorted(glob(wavs_pattern)))


def process_files(wav_files, process_audio, sample_rate):
    """Applies post-processing to a data split

    Arguments
    ---------
    wav_files: list
        a list of (src_file_name, tgt_file_name) tuples

    sample_rate: int
        the sample rate

    Returns
    -------
    results: generator
        a generator of (file_name, extra metadata) tuples

    """
    folders = set(
        os.path.dirname(tgt_file_name) for _, tgt_file_name in wav_files
    )
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
    for src_file_name, tgt_file_name in tqdm(wav_files):
        result = process_file(
            src_file_name, tgt_file_name, process_audio, sample_rate
        )
        yield tgt_file_name, result


def process_file(src_file_name, tgt_file_name, process_audio, sample_rate):
    """Processes a single audio file

    Arguments
    ---------
    src_file_name: str
        the source file name
    tgt_file_name: str
        the target file name
    process_audio: callable
        the audio processing function
    sample_rate: int
        the sampling rate
    """
    sig = sb.dataio.dataio.read_audio(src_file_name)
    sig = process_audio(sig)

    sb.dataio.dataio.write_audio(tgt_file_name, sig, sample_rate)

    return {"len": len(sig), "len_s": len(sig) / sample_rate}


def get_item_id(file_name):
    """Returns the item ID, which is the file name without the extension

    Arguments
    ---------
    file_name: str
        the file name

    Returns
    -------
    item_id: str
        the item ID corresponding to the file name
    """
    _, file_name = os.path.split(file_name)
    file_base_name = os.path.basename(file_name)
    file_base_name_noext, _ = os.path.splitext(file_base_name)
    return file_base_name_noext


def get_file_metadata(meta, split, file_list, lookup):
    """Returns a generator with metadata for each file

    Arguments
    ---------
    meta: dict
        the speaker metadata dictionary
    split: str
        the split identifier ("train", "valid" or "test")
    file_list: list
        the list of files
    lookup: dict
        the digit metadata lookup (for text/phoneme transcriptions)

    Returns
    -------
    item_id: str
        the ID of the item
    file_meta: dict
        the metadata - to be saved
    """
    for file_path in file_list:
        item_id = get_item_id(file_path)
        file_name = os.path.basename(file_path)
        digit, speaker_id, _ = item_id.split("_")
        speaker_meta = meta[speaker_id]
        file_meta = {
            "file_name": f"dataset/{split}/{speaker_id}/{file_name}",
            "digit": digit,
            "speaker_id": speaker_id,
        }
        file_meta.update(convert_speaker_meta(speaker_meta))
        digit_data = lookup[digit]
        file_meta.update(digit_data)
        yield item_id, file_meta


def convert_split(
    src, tgt, split, file_list, meta, lookup, process_audio, sample_rate
):
    """
    Converts a single split of data

    src: str
        the source path
    tgt: str
        the target path
    split: str
        the split identifier
    file_list: list
        the list of files in the data split
    meta: dict
        the metadata dictionary
    lookup: dict
        the digit look-up file
    process_audio: callable
        the function that will be applied to each audio file for processing
    sample_rate: int
        the target sample rate

    """
    metadata = dict(get_file_metadata(meta, split, file_list, lookup))
    metadata_file_path = os.path.join(tgt, f"{split}.json")

    wav_files = [
        (
            os.path.join(src, file_name),
            os.path.join(tgt, metadata[get_item_id(file_name)]["file_name"]),
        )
        for file_name in file_list
    ]

    for file_path, process_meta in process_files(
        wav_files, process_audio, sample_rate
    ):
        item_id = get_item_id(file_path)
        metadata[item_id].update(process_meta)

    logger.info(f"Saving metadata to {metadata_file_path}")
    with open(metadata_file_path, "w") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)


def convert_dataset(src, tgt, splits, lookup, process_audio, sample_rate):
    """Converts the dataset from the original format to the SpeechBrain-friendly
    format

    Arguments
    ---------
    src: str
        the source path

    tgt: str
        the target path

    splits: dict
        a dictionary with split identifiers as keys
        and the file list for the split corresponding to the
        key as the value

    lookup: dict
        the digit look-up

    process_audio: callable
        the audio processing function

    sample_rate: int
        the target sample rate
    """
    if not os.path.exists(tgt):
        print(f"Creating directory {tgt}")

    meta_file_name = os.path.join(src, "data", "audioMNIST_meta.txt")
    meta = read_meta(file_name=meta_file_name)

    for split, file_list in splits.items():
        logger.info("Converting split %s", split)
        convert_split(
            src=src,
            tgt=tgt,
            split=split,
            file_list=file_list,
            meta=meta,
            lookup=lookup,
            process_audio=process_audio,
            sample_rate=sample_rate,
        )


def trim_sig(sig, threshold):
    """A simple energy threshold implementation to remove silence at the
    beginning and at the end of a file

    Arguments
    ---------
    sig: torch.Tensor
        raw audio
    threshold: float
        the decibel threhold
    """
    threshold_amp = math.pow(DB_BASE, threshold * DB_MULTIPLIER)
    sig = sig / sig.abs().max()
    en_sig = sig ** 2
    sound_pos = (en_sig > threshold_amp).nonzero()
    first, last = sound_pos[0], sound_pos[-1]
    return sig[first:last]


def process_audio_default(
    sig,
    norm=True,
    trim=True,
    src_sample_rate=48000,
    tgt_sample_rate=22050,
    trim_threshold=-30.0,
):
    """Standard audio preprocessing / conversion

    Arguments
    ---------
    norm: True
        whether to normalize
    trim: True
        whether to trim silence at the beginning and at the end
    src_sample_rate: int
        the sample rate at which the files are recorded
    tgt_sample_rate: int
        the target sample rate
    trim_threshold: float
        the decibels threshold for trimming the file
    """
    # Normalize
    if norm:
        sig = sig / sig.abs().max()
    # Resample
    if src_sample_rate != tgt_sample_rate:
        sig = F.resample(sig, src_sample_rate, tgt_sample_rate)
    # VAD
    if trim:
        sig = trim_sig(sig, trim_threshold)
    return sig


def read_digit_lookup(file_name):
    """Reads the digit look-up CSV file

    Arguments
    ---------
    file_name: str
        the file name

    Returns
    -------
    result: dict
        a dictionary simialr the following
        {
            "2": {
                "char": "two",
                "phn": ["T", "UW"]
            }
        }

    """
    with open(file_name) as lookup_file:
        reader = csv.DictReader(lookup_file)
        lookup = {row["digit"]: row for row in reader}
        for value in lookup.values():
            del value["digit"]
            value["phn"] = value["phn"].split(" ")
        return lookup
