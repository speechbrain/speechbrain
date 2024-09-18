"""
LJspeech data preparation.
Download: https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2

Authors
 * Yingzhi WANG 2022
 * Sathvik Udupa 2022
 * Pradnya Kandarkar 2023
"""

import csv
import json
import logging
import os
import random
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import tgt
import torch
import torchaudio
from torchaudio.functional import resample
from tqdm import tqdm
from unidecode import unidecode

import speechbrain as sb
from speechbrain.dataio.batch import PaddedData
from speechbrain.dataio.dataio import load_pkl, save_pkl
from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.dataio.preparation import FeatureExtractor
from speechbrain.inference.text import GraphemeToPhoneme
from speechbrain.utils.data_utils import download_file
from speechbrain.utils.text_to_sequence import _g2p_keep_punctuations

logger = logging.getLogger(__name__)
OPT_FILE = "opt_ljspeech_prepare.pkl"
METADATA_CSV = "metadata.csv"
TRAIN_JSON = "train.json"
VALID_JSON = "valid.json"
TEST_JSON = "test.json"
WAVS = "wavs"
DURATIONS = "durations"

logger = logging.getLogger(__name__)
OPT_FILE = "opt_ljspeech_prepare.pkl"


def prepare_ljspeech(
    data_folder,
    save_folder,
    splits=["train", "valid"],
    split_ratio=[90, 10],
    model_name=None,
    seed=1234,
    pitch_n_fft=1024,
    pitch_hop_length=256,
    pitch_min_f0=65,
    pitch_max_f0=400,
    skip_prep=False,
    use_custom_cleaner=False,
    extract_features=None,
    extract_features_opts=None,
    extract_phonemes=False,
    g2p_src="speechbrain/soundchoice-g2p",
    skip_ignore_folders=False,
    frozen_split_path=None,
    device="cpu",
):
    """
    Prepares the csv files for the LJspeech datasets.

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original LJspeech dataset is stored
    save_folder : str
        The directory where to store the csv/json files
    splits : list
        List of dataset splits to prepare
    split_ratio : list
        Proportion for dataset splits
    model_name : str
        Model name (used to prepare additional model specific data)
    seed : int
        Random seed
    pitch_n_fft : int
        Number of fft points for pitch computation
    pitch_hop_length : int
        Hop length for pitch computation
    pitch_min_f0 : int
        Minimum f0 for pitch computation
    pitch_max_f0 : int
        Max f0 for pitch computation
    skip_prep : bool
        If True, skip preparation
    use_custom_cleaner : bool
        If True, uses custom cleaner defined for this recipe
    extract_features : list
        The list of features to be extracted
    extract_features_opts : dict
        Options for feature extraction
    extract_phonemes : bool
        Whether to extract phonemes using a G2P model
    g2p_src : str
        The name of the HuggingFace Hub to use for the Grapheme-to-Phoneme
        model or the path to it
    skip_ignore_folders : bool
        Whether to ignore differences in data and save folders when
        checking if the dataset has already been prepared. This is
        useful on high-performance compute clusters where such
        folders are not permanent
    frozen_split_path : str | path-like
        The path to the frozen split file (used to standardize multiple
        experiments)
    device : str
        Device for to be used for computation (used as required)

    Returns
    -------
    None

    Example
    -------
    >>> from recipes.LJSpeech.TTS.ljspeech_prepare import prepare_ljspeech
    >>> data_folder = 'data/LJspeech/'
    >>> save_folder = 'save/'
    >>> splits = ['train', 'valid']
    >>> split_ratio = [90, 10]
    >>> seed = 1234
    >>> prepare_ljspeech(data_folder, save_folder, splits, split_ratio, seed)
    """
    # Sets seeds for reproducible code
    random.seed(seed)

    if skip_prep:
        return

    # Creating configuration for easily skipping data_preparation stage
    conf = {
        "data_folder": data_folder,
        "splits": splits,
        "split_ratio": split_ratio,
        "save_folder": save_folder,
        "seed": seed,
    }
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Setting output files
    meta_csv = os.path.join(data_folder, METADATA_CSV)
    wavs_folder = os.path.join(data_folder, WAVS)

    save_opt = os.path.join(save_folder, OPT_FILE)
    save_json_train = os.path.join(save_folder, TRAIN_JSON)
    save_json_valid = os.path.join(save_folder, VALID_JSON)
    save_json_test = os.path.join(save_folder, TEST_JSON)

    phoneme_alignments_folder = None
    duration_folder = None
    pitch_folder = None
    # Setting up additional folders required for FastSpeech2
    if model_name == "FastSpeech2":
        # This step requires phoneme alignments to be present in the data_folder
        # We automatically download the alignments from https://www.dropbox.com/s/v28x5ldqqa288pu/LJSpeech.zip
        # Download and unzip LJSpeech phoneme alignments from here: https://drive.google.com/drive/folders/1DBRkALpPd6FL9gjHMmMEdHODmkgNIIK4
        alignment_URL = (
            "https://www.dropbox.com/s/v28x5ldqqa288pu/LJSpeech.zip?dl=1"
        )
        phoneme_alignments_folder = os.path.join(
            data_folder, "TextGrid", "LJSpeech"
        )
        download_file(
            alignment_URL, data_folder + "/alligments.zip", unpack=True
        )

        duration_folder = os.path.join(data_folder, "durations")
        if not os.path.exists(duration_folder):
            os.makedirs(duration_folder)

    # extract pitch for both Fastspeech2 and FastSpeech2WithAligner models
    if "FastSpeech2" in model_name:
        pitch_folder = os.path.join(data_folder, "pitch")
        if not os.path.exists(pitch_folder):
            os.makedirs(pitch_folder)

    # Check if this phase is already done (if so, skip it)
    if skip(splits, save_folder, conf, ignore_folders=skip_ignore_folders):
        logger.info("Skipping preparation, completed in previous run.")
        return

    # Additional check to make sure metadata.csv and wavs folder exists
    assert os.path.exists(meta_csv), "metadata.csv does not exist"
    assert os.path.exists(wavs_folder), "wavs/ folder does not exist"

    # Prepare data splits
    msg = "Creating json file for ljspeech Dataset.."
    logger.info(msg)
    data_split, meta_csv = split_sets(
        data_folder, splits, split_ratio, frozen_split_path
    )

    extract_features_context = None
    extract_features_folder = None
    if extract_features:
        extract_features_context = get_context(
            extract_features=extract_features,
            extract_features_opts=extract_features_opts or {},
            device=device,
        )
        extract_features_folder = Path(save_folder) / "features"

    if "train" in splits:
        prepare_json(
            model_name,
            data_split["train"],
            save_json_train,
            data_folder,
            wavs_folder,
            meta_csv,
            phoneme_alignments_folder,
            duration_folder,
            pitch_folder,
            pitch_n_fft,
            pitch_hop_length,
            pitch_min_f0,
            pitch_max_f0,
            use_custom_cleaner,
            extract_features,
            extract_features_context,
            extract_features_folder,
            extract_features_opts,
            extract_phonemes,
            g2p_src,
            device,
        )
    if "valid" in splits:
        prepare_json(
            model_name,
            data_split["valid"],
            save_json_valid,
            data_folder,
            wavs_folder,
            meta_csv,
            phoneme_alignments_folder,
            duration_folder,
            pitch_folder,
            pitch_n_fft,
            pitch_hop_length,
            pitch_min_f0,
            pitch_max_f0,
            use_custom_cleaner,
            extract_features,
            extract_features_context,
            extract_features_folder,
            extract_features_opts,
            extract_phonemes,
            g2p_src,
            device,
        )
    if "test" in splits:
        prepare_json(
            model_name,
            data_split["test"],
            save_json_test,
            data_folder,
            wavs_folder,
            meta_csv,
            phoneme_alignments_folder,
            duration_folder,
            pitch_folder,
            pitch_n_fft,
            pitch_hop_length,
            pitch_min_f0,
            pitch_max_f0,
            use_custom_cleaner,
            extract_features,
            extract_features_context,
            extract_features_folder,
            extract_features_opts,
            extract_phonemes,
            g2p_src,
            device,
        )
    save_pkl(conf, save_opt)


def skip(splits, save_folder, conf, ignore_folders=False):
    """
    Detects if the ljspeech data_preparation has been already done.
    If the preparation has been done, we can skip it.

    Arguments
    ---------
    splits : list
        The list of dataset splits to consider
    save_folder : str | path-like
        The folder where results are saved
    conf : dict
        The contents of the configuration file
    ignore_folders : bool
        If set to true, differences in folders (such as save_folder)
        will be ignored. This is useful on high-performance
        compute clusters where such folders will change from one run
        to the next


    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """
    # Checking json files
    skip = True

    split_files = {
        "train": TRAIN_JSON,
        "valid": VALID_JSON,
        "test": TEST_JSON,
    }

    for split in splits:
        if not os.path.isfile(os.path.join(save_folder, split_files[split])):
            skip = False

    #  Checking saved options
    save_opt = os.path.join(save_folder, OPT_FILE)
    if skip is True:
        if os.path.isfile(save_opt):
            opts_old = load_pkl(save_opt)
            if ignore_folders:
                opts_old = remove_folder_opts(opts_old)
                conf = remove_folder_opts(opts_old)
            if opts_old == conf:
                skip = True
            else:
                skip = False
        else:
            skip = False
    return skip


def remove_folder_opts(conf):
    """Removes all folder options from  the configuration dict"""
    return {k: v for k, v in conf.items() if not k.endswith("_folder")}


def split_sets(data_folder, splits, split_ratio, frozen_split_path):
    """Randomly splits the wav list into training, validation, and test lists.
    Note that a better approach is to make sure that all the classes have the
    same proportion of samples for each session.

    Arguments
    ---------
    data_folder : str | path-like
        the path to the data folder
    splits : list
        the list of dataset splits to be used (among "train", "valid", "test")
    split_ratio: list
        List composed of three integers that sets split ratios for train,
        valid, and test sets, respectively.
        For instance split_ratio=[80, 10, 10] will assign 80% of the sentences
        to training, 10% for validation, and 10% for test.
    frozen_split_path : str | path-like
        the path to the frozen split file

    Returns
    -------
    dictionary containing train, valid, and test splits.
    """
    meta_csv = os.path.join(data_folder, METADATA_CSV)
    csv_reader = csv.reader(
        open(meta_csv), delimiter="|", quoting=csv.QUOTE_NONE
    )

    meta_csv = list(csv_reader)
    if frozen_split_path is not None:
        frozen_split_path = Path(frozen_split_path)
        if frozen_split_path.exists():
            logger.info("Using frozen split at %s", str(frozen_split_path))
            with open(frozen_split_path, "r") as frozen_split_file:
                data_split = json.load(frozen_split_file)
            return data_split, meta_csv
        else:
            logger.info(
                "Frozen split %s does not exist", str(frozen_split_path)
            )

    index_for_sessions = []
    session_id_start = "LJ001"
    index_this_session = []
    for i in range(len(meta_csv)):
        session_id = meta_csv[i][0].split("-")[0]
        if session_id == session_id_start:
            index_this_session.append(i)
            if i == len(meta_csv) - 1:
                index_for_sessions.append(index_this_session)
        else:
            index_for_sessions.append(index_this_session)
            session_id_start = session_id
            index_this_session = [i]

    session_len = [len(session) for session in index_for_sessions]

    data_split = {}
    for i, split in enumerate(splits):
        data_split[split] = []
        for j in range(len(index_for_sessions)):
            if split == "train":
                random.shuffle(index_for_sessions[j])
                n_snts = int(session_len[j] * split_ratio[i] / sum(split_ratio))
                data_split[split].extend(index_for_sessions[j][0:n_snts])
                del index_for_sessions[j][0:n_snts]
            if split == "valid":
                if "test" in splits:
                    random.shuffle(index_for_sessions[j])
                    n_snts = int(
                        session_len[j] * split_ratio[i] / sum(split_ratio)
                    )
                    data_split[split].extend(index_for_sessions[j][0:n_snts])
                    del index_for_sessions[j][0:n_snts]
                else:
                    data_split[split].extend(index_for_sessions[j])
            if split == "test":
                data_split[split].extend(index_for_sessions[j])

    if frozen_split_path is not None:
        with open(frozen_split_path, "w") as frozen_split_file:
            json.dump(data_split, frozen_split_file, indent=0)

    return data_split, meta_csv


def prepare_json(
    model_name,
    seg_lst,
    json_file,
    data_folder,
    wavs_folder,
    csv_reader,
    phoneme_alignments_folder,
    durations_folder,
    pitch_folder,
    pitch_n_fft,
    pitch_hop_length,
    pitch_min_f0,
    pitch_max_f0,
    use_custom_cleaner=False,
    extract_features=None,
    extract_features_context=None,
    extract_features_folder=None,
    extract_features_opts=None,
    extract_phonemes=False,
    g2p_src="speechbrain/soundchoice-g2p",
    device="cpu",
):
    """
    Creates json file given a list of indexes.

    Arguments
    ---------
    model_name : str
        Model name (used to prepare additional model specific data)
    seg_lst : list
        The list of json indexes of a given data split
    json_file : str
        Output json path
    data_folder : str
        The folder where the dataset is located
    wavs_folder : str
        LJspeech wavs folder
    csv_reader : _csv.reader
        LJspeech metadata
    phoneme_alignments_folder : path
        Path where the phoneme alignments are stored
    durations_folder : path
        Folder where to store the duration values of each audio
    pitch_folder : path
        Folder where to store the pitch of each audio
    pitch_n_fft : int
        Number of fft points for pitch computation
    pitch_hop_length : int
        Hop length for pitch computation
    pitch_min_f0 : int
        Minimum f0 for pitch computation
    pitch_max_f0 : int
        Max f0 for pitch computation
    use_custom_cleaner : bool
        If True, uses custom cleaner defined for this recipe
    extract_features : list, optional
        If specified, feature extraction will be performed
    extract_features_context : types.SimpleNamespace, optional
        Context for feature extraction (pretrained models, etc)
    extract_features_folder : path-like, optional
        The folder where extracted features will be saved
    extract_features_opts : dict, optional
        Options for feature extraction
    extract_phonemes : bool
        Whether to extract phonemes via SoundChocie G2P
    g2p_src : str
        The name of the HuggingFace Hub to use for the Grapheme-to-Phoneme
        model or the path to it
    device : str
        Device for to be used for computation (used as required)
    """

    logger.info(f"preparing {json_file}.")
    if model_name in ["Tacotron2", "FastSpeech2WithAlignment"]:
        extract_phonemes = True
    if extract_phonemes:
        logger.info(
            "Computing phonemes for LJSpeech labels using SpeechBrain G2P. This may take a while."
        )
        g2p = GraphemeToPhoneme.from_hparams(
            g2p_src, run_opts={"device": device}
        )
    if "FastSpeech2" in model_name:
        logger.info(
            "Computing pitch as required for FastSpeech2. This may take a while."
        )

    json_dict = {}
    for index in tqdm(seg_lst):

        # Common data preparation
        id = list(csv_reader)[index][0]
        wav = os.path.join("{data_root}", WAVS, f"{id}.wav")
        label = list(csv_reader)[index][2]
        if use_custom_cleaner:
            label = custom_clean(label, model_name)

        json_dict[id] = {
            "uttid": id,
            "wav": wav,
            "label": label,
            "segment": True if "train" in json_file else False,
        }

        # FastSpeech2 specific data preparation
        if model_name == "FastSpeech2":

            audio, fs = torchaudio.load(wav)

            # Parses phoneme alignments
            textgrid_path = os.path.join(
                phoneme_alignments_folder, f"{id}.TextGrid"
            )
            textgrid = tgt.io.read_textgrid(
                textgrid_path, include_empty_intervals=True
            )

            last_phoneme_flags = get_last_phoneme_info(
                textgrid.get_tier_by_name("words"),
                textgrid.get_tier_by_name("phones"),
            )
            (
                phonemes,
                duration,
                start,
                end,
                trimmed_last_phoneme_flags,
            ) = get_alignment(
                textgrid.get_tier_by_name("phones"),
                fs,
                pitch_hop_length,
                last_phoneme_flags,
            )

            # Gets label phonemes
            label_phoneme = " ".join(phonemes)
            spn_labels = [0] * len(phonemes)
            for i in range(1, len(phonemes)):
                if phonemes[i] == "spn":
                    spn_labels[i - 1] = 1
            if start >= end:
                print(f"Skipping {id}")
                continue

            # Saves durations
            duration_file_path = os.path.join(durations_folder, f"{id}.npy")
            np.save(duration_file_path, duration)

            # Computes pitch
            audio = audio[:, int(fs * start) : int(fs * end)]
            pitch_file = wav.replace(".wav", ".npy").replace(
                wavs_folder, pitch_folder
            )
            if not os.path.isfile(pitch_file):
                pitch = torchaudio.functional.detect_pitch_frequency(
                    waveform=audio,
                    sample_rate=fs,
                    frame_time=(pitch_hop_length / fs),
                    win_length=3,
                    freq_low=pitch_min_f0,
                    freq_high=pitch_max_f0,
                ).squeeze(0)

                # Concatenate last element to match duration.
                pitch = torch.cat([pitch, pitch[-1].unsqueeze(0)])

                # Mean and Variance Normalization
                mean = 256.1732939688805
                std = 328.319759158607

                pitch = (pitch - mean) / std

                pitch = pitch[: sum(duration)]
                np.save(pitch_file, pitch)

            # Updates data for the utterance
            json_dict[id].update({"label_phoneme": label_phoneme})
            json_dict[id].update({"spn_labels": spn_labels})
            json_dict[id].update({"start": start})
            json_dict[id].update({"end": end})
            json_dict[id].update({"durations": duration_file_path})
            json_dict[id].update({"pitch": pitch_file})
            json_dict[id].update(
                {"last_phoneme_flags": trimmed_last_phoneme_flags}
            )

        # FastSpeech2WithAlignment specific data preparation
        if model_name == "FastSpeech2WithAlignment":
            audio, fs = torchaudio.load(wav)
            # Computes pitch
            pitch_file = wav.replace(".wav", ".npy").replace(
                wavs_folder, pitch_folder
            )
            if not os.path.isfile(pitch_file):

                if torchaudio.__version__ < "2.1":
                    pitch = torchaudio.functional.compute_kaldi_pitch(
                        waveform=audio,
                        sample_rate=fs,
                        frame_length=(pitch_n_fft / fs * 1000),
                        frame_shift=(pitch_hop_length / fs * 1000),
                        min_f0=pitch_min_f0,
                        max_f0=pitch_max_f0,
                    )[0, :, 0]
                else:
                    pitch = torchaudio.functional.detect_pitch_frequency(
                        waveform=audio,
                        sample_rate=fs,
                        frame_time=(pitch_hop_length / fs),
                        win_length=3,
                        freq_low=pitch_min_f0,
                        freq_high=pitch_max_f0,
                    ).squeeze(0)

                    # Concatenate last element to match duration.
                    pitch = torch.cat([pitch, pitch[-1].unsqueeze(0)])

                    # Mean and Variance Normalization
                    mean = 256.1732939688805
                    std = 328.319759158607

                    pitch = (pitch - mean) / std

                np.save(pitch_file, pitch)

            json_dict[id].update({"pitch": pitch_file})
        if extract_phonemes:
            phonemes = _g2p_keep_punctuations(g2p, label)
            # Updates data for the utterance
            json_dict[id].update({"phonemes": phonemes})

    # Feature Extraction
    if extract_features:
        extract_features_folder.mkdir(exist_ok=True)
        prepare_features(
            data=json_dict,
            data_folder=data_folder,
            save_folder=extract_features_folder,
            features=extract_features,
            context=extract_features_context,
            options=extract_features_opts,
            device=device,
        )

    # Writing the dictionary to the json file
    with open(json_file, mode="w") as json_f:
        json.dump(json_dict, json_f, indent=2)

    logger.info(f"{json_file} successfully created!")


def get_alignment(tier, sampling_rate, hop_length, last_phoneme_flags):
    """
    Returns phonemes, phoneme durations (in frames), start time (in seconds), end time (in seconds).
    This function is adopted from https://github.com/ming024/FastSpeech2/blob/master/preprocessor/preprocessor.py

    Arguments
    ---------
    tier : tgt.core.IntervalTier
        For an utterance, contains Interval objects for phonemes and their start time and end time in seconds
    sampling_rate : int
        Sample rate if audio signal
    hop_length : int
        Hop length for duration computation
    last_phoneme_flags : list
        List of (phoneme, flag) tuples with flag=1 if the phoneme is the last phoneme else flag=0


    Returns
    -------
    (phones, durations, start_time, end_time) : tuple
        The phonemes, durations, start time, and end time for an utterance
    """

    sil_phones = ["sil", "sp", "spn", ""]

    phonemes = []
    durations = []
    start_time = 0
    end_time = 0
    end_idx = 0
    trimmed_last_phoneme_flags = []

    flag_iter = iter(last_phoneme_flags)

    for t in tier._objects:
        s, e, p = t.start_time, t.end_time, t.text
        current_flag = next(flag_iter)

        # Trims leading silences
        if phonemes == []:
            if p in sil_phones:
                continue
            else:
                start_time = s

        if p not in sil_phones:
            # For ordinary phones
            # Removes stress indicators
            if p[-1].isdigit():
                phonemes.append(p[:-1])
            else:
                phonemes.append(p)
            trimmed_last_phoneme_flags.append(current_flag[1])
            end_time = e
            end_idx = len(phonemes)
        else:
            # Uses a unique token for all silent phones
            phonemes.append("spn")
            trimmed_last_phoneme_flags.append(current_flag[1])

        durations.append(
            int(
                np.round(e * sampling_rate / hop_length)
                - np.round(s * sampling_rate / hop_length)
            )
        )

    # Trims tailing silences
    phonemes = phonemes[:end_idx]
    durations = durations[:end_idx]

    return phonemes, durations, start_time, end_time, trimmed_last_phoneme_flags


def get_last_phoneme_info(words_seq, phones_seq):
    """This function takes word and phoneme tiers from a TextGrid file as input
    and provides a list of tuples for the phoneme sequence indicating whether
    each of the phonemes is the last phoneme of a word or not.

    Each tuple of the returned list has this format: (phoneme, flag)


    Arguments
    ---------
    words_seq : enumerable
        word tier from a TextGrid file
    phones_seq : enumerable
        phoneme tier from a TextGrid file

    Returns
    -------
    last_phoneme_flags : list
        each tuple of the returned list has this format: (phoneme, flag)
    """

    # Gets all phoneme objects for the entire sequence
    phoneme_objects = phones_seq._objects
    phoneme_iter = iter(phoneme_objects)

    # Stores flags to show if an element (phoneme) is a the last phoneme of a word
    last_phoneme_flags = list()

    # Matches the end times of the phoneme and word objects to get the last phoneme information
    for word_obj in words_seq._objects:
        word_end_time = word_obj.end_time

        current_phoneme = next(phoneme_iter, None)
        while current_phoneme:
            phoneme_end_time = current_phoneme.end_time
            if phoneme_end_time == word_end_time:
                last_phoneme_flags.append((current_phoneme.text, 1))
                break
            else:
                last_phoneme_flags.append((current_phoneme.text, 0))
            current_phoneme = next(phoneme_iter, None)

    return last_phoneme_flags


def custom_clean(text, model_name):
    """
    Uses custom criteria to clean text.

    Arguments
    ---------
    text : str
        Input text to be cleaned
    model_name : str
        whether to treat punctuations

    Returns
    -------
    text : str
        Cleaned text
    """

    _abbreviations = [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
        for x in [
            ("mrs", "misess"),
            ("mr", "mister"),
            ("dr", "doctor"),
            ("st", "saint"),
            ("co", "company"),
            ("jr", "junior"),
            ("maj", "major"),
            ("gen", "general"),
            ("drs", "doctors"),
            ("rev", "reverend"),
            ("lt", "lieutenant"),
            ("hon", "honorable"),
            ("sgt", "sergeant"),
            ("capt", "captain"),
            ("esq", "esquire"),
            ("ltd", "limited"),
            ("col", "colonel"),
            ("ft", "fort"),
        ]
    ]
    text = unidecode(text.lower())
    if model_name != "FastSpeech2WithAlignment":
        text = re.sub("[:;]", " - ", text)
        text = re.sub(r'[)(\[\]"]', " ", text)
        text = text.strip().strip().strip("-")

    text = re.sub(" +", " ", text)
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


INLINE_FEATURES = ["audio_ssl_len"]


def prepare_features(
    data,
    data_folder,
    save_folder,
    features,
    context,
    options=None,
    device="cpu",
):
    """Performs feature extraction

    Arguments
    ---------
    data: dict
        a preprocessed dataset
    data_folder : str
        the data folder
    save_folder : str
        the folder where features will be saved
    features : list
        the list of feature extractions to be performed
    context : dict
        context data
    options: dict
        Feature extraction options
    device : str|torch.device
        the device identifier
    """
    dataset = DynamicItemDataset(data)
    feature_extractor = FeatureExtractor(
        save_path=save_folder,
        src_keys=["sig"],
        id_key="uttid",
        dataloader_opts=options.get("dataloader_opts", {}),
        device=device,
    )
    token_model_kwargs = options.get("token_model_kwargs", {})
    ssl_layers = options.get("ssl_model_layers") or options.get(
        "token_model_layers"
    )

    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        """Load the audio signal."""
        wav = wav.replace("{data_root}", data_folder)
        sig = sb.dataio.dataio.read_audio(wav)

        yield sig

    dataset.add_dynamic_item(audio_pipeline)

    @sb.utils.data_pipeline.takes("sig")
    @sb.utils.data_pipeline.provides("sig_resampled")
    def resample_pipeline(sig):
        sig_data = resample(
            waveform=sig.data,
            orig_freq=options["sample_rate"],
            new_freq=options["model_sample_rate"],
        )
        return PaddedData(sig_data, sig.lengths)

    @sb.utils.data_pipeline.takes("sig_resampled")
    @sb.utils.data_pipeline.provides("audio_tokens", "audio_emb")
    def token_pipeline(sig):
        with torch.no_grad():
            result = context.token_model(
                sig.data, sig.lengths, **token_model_kwargs
            )
            # TODO: Clean this up
            if torch.is_tensor(result):
                tokens = result
                # Note: Dummy embedding - meaning embeddings are not available
                emb = torch.zeros((len(sig.data), 1, 1), device=sig.data.device)
            else:
                tokens, emb = result[:2]
                tokens = tokens.int()
            if tokens.dim() < 3:
                tokens = tokens.unsqueeze(-1)
            yield PaddedData(tokens, sig.lengths)
            yield PaddedData(emb, sig.lengths)

    @sb.utils.data_pipeline.takes("sig_resampled")
    @sb.utils.data_pipeline.provides("spk_emb")
    def spk_emb_pipeline(sig):
        mel_spec = context.spk_emb_model.mel_spectogram(audio=sig.data)
        return context.spk_emb_model.encode_mel_spectrogram_batch(
            mel_spec, sig.lengths
        )

    @sb.utils.data_pipeline.takes("sig_resampled")
    @sb.utils.data_pipeline.provides("audio_ssl", "audio_ssl_len")
    def ssl_pipeline(sig):
        ssl_raw = context.ssl_model(sig.data, sig.lengths)
        ssl = ssl_raw[ssl_layers].permute(1, 2, 0, 3)
        yield PaddedData(ssl, sig.lengths)
        yield (sig.lengths * ssl.size(1)).tolist()

    dynamic_items = [
        resample_pipeline,
        token_pipeline,
        ssl_pipeline,
        spk_emb_pipeline,
    ]
    for dynamic_item in dynamic_items:
        feature_extractor.add_dynamic_item(dynamic_item)

    feature_keys = [key for key in features if key not in INLINE_FEATURES]
    inline_keys = [key for key in features if key in INLINE_FEATURES]
    feature_extractor.set_output_features(feature_keys, inline_keys=inline_keys)
    feature_extractor.extract(dataset, data)


def get_context(extract_features, extract_features_opts, device):
    """
    Gets the context (pretrained models, etc) for feature extraction

    Arguments
    ---------
    extract_features : list
        A list of features to extract
        Available features:
        audio_tokens - raw tokens
        audio_emb - embeddings from the model
    extract_features_opts : dict
        Options for feature extraction
    device : str|torch.Device
        The device on which extraction will be run

    Returns
    -------
    context: SimpleNamespace
        The context object
    """
    context = {}
    if any(key in extract_features for key in ["audio_tokens", "audio_emb"]):
        context["token_model"] = extract_features_opts["token_model"].to(device)
    if "audio_ssl" in extract_features:
        context["ssl_model"] = extract_features_opts["ssl_model"].to(device)
    if "spk_emb" in extract_features:
        context["spk_emb_model"] = extract_features_opts["spk_emb_model"](
            run_opts={"device": device}
        )

    return SimpleNamespace(**context)
