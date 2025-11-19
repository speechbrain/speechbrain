#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data preparation.
Download: https://commonvoice.mozilla.org/lang/datasets

Multilingual Common Voice preparation for SENSE from TSV files.
Languages used for training are listed in LANGUAGES.

Author
------
Maryem Bouziane 2025, Avignon Université
Salima Mdhaffar 2025, Avignon Université
Haroun Elleuch 2025, Avignon Université
Yannick Estève 2025, Avignon Université
"""

import os
from os import path
import functools
import importlib.util
import json

import pandas as pd
from tqdm import tqdm

from speechbrain.utils.logger import get_logger

# Languages used for SENSE training
LANGUAGES = [
    "af", "am", "ar", "as", "ast", "az", "ba", "be", "bg", "bn", "br", "ca",
    "ckb", "cs", "cv", "cy", "da", "de", "dv", "el", "en", "eo", "es", "et",
    "fa", "fi", "fr", "fy-NL", "ga-IE", "gl", "gn", "he", "hi", "hsb", "ht",
    "hu", "ia", "id", "is", "it", "ja", "ka", "kab", "kk", "ko", "ky", "lt",
    "lo", "lv", "ml", "mn", "mhr", "mk", "mr", "mt", "ne-NP", "nl", "nn-NO",
    "oc", "or", "os", "pa-IN", "pl", "ps", "pt", "ro", "ru", "sah", "sc",
    "sk", "sl", "sr", "sv-SE", "sw", "ta", "te", "th", "ti", "tk", "tr", "tt",
    "ug", "uk", "ur", "uz", "vi", "yi", "yo", "zh-HK", "zu",
]
logger = get_logger(__name__)

# Import utilities from the official Common Voice recipe
CV_PREP_PATH = (
    "../common_voice_prepare.py"
)

if not os.path.isfile(CV_PREP_PATH):
    raise FileNotFoundError(
        f"common_voice_prepare.py not found at: {CV_PREP_PATH}"
    )

spec = importlib.util.spec_from_file_location("common_voice_prepare", CV_PREP_PATH)
cv_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cv_module)

process_line = cv_module.process_line
CVRow = cv_module.CVRow
check_commonvoice_folders = cv_module.check_commonvoice_folders


def create_directory(dir_path):
    """Creates a directory if it does not exist and returns its path."""
    if dir_path is None or dir_path == "":
        return dir_path
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


def save_json(content, filename):
    """Saves a Python object to JSON (UTF-8, pretty-printed)."""
    if filename is None or filename == "":
        return

    directory = os.path.dirname(filename)
    if directory:
        create_directory(directory)

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(content, f, ensure_ascii=False, indent=2)


def cv_lang_folder(data_folder, lang):
    """Returns the Common Voice folder for a given language."""
    return path.join(data_folder, lang)


def read_split_lang_to_df(
    data_folder,
    lang,
    split,
    convert_to_wav=False,
    duration_max=10.0,
    sample_rate=16000,
):
    """
    Reads one split (train/dev) for a given language from the TSV file
    and returns a DataFrame with columns:
        ID, duration, wav, spk_id, text

    Audio fields are obtained with the original `process_line` function.
    The text is taken from the raw TSV ("sentence" or "text" column).
    """
    data_folder_lang = cv_lang_folder(data_folder, lang)
    orig_tsv_file = path.join(data_folder_lang, f"{split}.tsv")

    if not path.isfile(orig_tsv_file):
        logger.warning(
            f"[{split}] TSV file not found for language {lang}: {orig_tsv_file}"
        )
        return None

    # Check Common Voice folder structure (clips directory, etc.)
    try:
        check_commonvoice_folders(data_folder_lang)
    except FileNotFoundError as e:
        logger.warning(
            f"Invalid Common Voice structure for {data_folder_lang}: {e}"
        )
        return None

    logger.info(f"Reading {split}.tsv for language {lang}: {orig_tsv_file}")
    with open(orig_tsv_file, encoding="utf-8") as f:
        csv_lines = f.readlines()

    if len(csv_lines) <= 1:
        logger.warning(f"No usable data in {orig_tsv_file}")
        return None

    header_line = csv_lines[0]
    data_lines = csv_lines[1:]

    header_map = {
        column_name: index
        for index, column_name in enumerate(header_line.split("\t"))
    }

    # Map sentence id (snt_id) → raw transcription from TSV
    raw_text_by_id = {}
    for line in data_lines:
        cols = line.rstrip("\n").split("\t")
        audio_path_filename = cols[header_map["path"]]
        snt_id = audio_path_filename.split(".")[-2].split("/")[-1]

        if "sentence" in header_map:
            raw_sentence = cols[header_map["sentence"]]
        elif "text" in header_map:
            raw_sentence = cols[header_map["text"]]
        else:
            raw_sentence = ""

        raw_text_by_id[snt_id] = raw_sentence

    # Audio processing and duration filtering (no parallel_map here)
    line_processor = functools.partial(
        process_line,
        convert_to_wav=convert_to_wav,
        data_folder=data_folder_lang,
        language=lang,
        accented_letters=False,
        header_map=header_map,
    )

    rows = []
    for line in tqdm(data_lines, desc=f"{lang}-{split}"):
        row = line_processor(line)
        if row is None:
            continue
        if row.duration == 0.0:
            continue
        if row.duration >= duration_max:
            continue
        rows.append(row)

    if len(rows) == 0:
        logger.warning(f"No valid samples for {lang} / {split}")
        return None

    ids = []
    durations = []
    wavs = []
    spk_ids = []
    texts = []

    for r in rows:
        full_id = r.snt_id
        ids.append(full_id)
        durations.append(r.duration)
        wavs.append(r.audio_path)
        spk_ids.append(r.spk_id)
        raw_sentence = raw_text_by_id.get(r.snt_id, "")
        texts.append(raw_sentence)

    df = pd.DataFrame(
        {
            "ID": ids,
            "duration": durations,
            "wav": wavs,
            "spk_id": spk_ids,
            "text": texts,
        }
    )

    return df


def skip_prepared_splits(preparation_output_folder):
    """
    Returns True if train.csv and dev.csv already exist in
    `preparation_output_folder`. In this case data preparation can be skipped.
    """
    train_csv = path.join(preparation_output_folder, "train.csv")
    dev_csv = path.join(preparation_output_folder, "dev.csv")

    if all(path.isfile(p) for p in [train_csv, dev_csv]):
        logger.info(
            f"Found existing CSV files: {train_csv}, {dev_csv} "
            "-> skipping data preparation."
        )
        return True
    return False


def build_combined_split(
    data_folder,
    save_folder,
    split,
    alpha=0.05,
    language_ratios_file=None,
    convert_to_wav=False,
    sample_rate=16000,
):
    """
    Builds a multilingual split (train or dev) from Common Voice TSV files.

    For each language in LANGUAGES:
        * read the corresponding TSV,
        * use `process_line` for audio and duration,
        * keep utterances with 0 < duration < 10 s,
        * add a `lang` column,
        * concatenate all languages into a single CSV.

    For the `train` split:
        * compute language sampling ratios using the SENSE formula,
        * save them as JSON (language_ratios_file),
        * add a `ratio` column to the combined CSV.
    """
    logger.info(f"=== Building combined split: {split} ===")

    save_folder = create_directory(save_folder)

    dfs = []
    language_counts = {}

    for lang in tqdm(LANGUAGES, desc=f"Split {split}"):
        df_lang = read_split_lang_to_df(
            data_folder=data_folder,
            lang=lang,
            split=split,
            convert_to_wav=convert_to_wav,
            duration_max=10.0,
            sample_rate=sample_rate,
        )
        if df_lang is None or df_lang.empty:
            continue

        df_lang["lang"] = lang
        language_counts[lang] = len(df_lang)
        dfs.append(df_lang)

    if len(dfs) == 0:
        raise RuntimeError(
            f"No valid samples for split {split} across all languages."
        )

    combined = pd.concat(dfs, ignore_index=True)

    if split == "train":
        total = len(combined)
        if total == 0:
            raise RuntimeError("Combined train split is empty after filtering.")

        ratio_map = {}
        ps = {}
        p_alphas = {}
        p_alpha_sum = 0.0

        for lang, count in language_counts.items():
            p = count / total
            ps[lang] = p
            p_alpha = p ** alpha
            p_alphas[lang] = p_alpha
            p_alpha_sum += p_alpha

        for lang in language_counts.keys():
            p = ps[lang]
            if p > 0:
                ratio_map[lang] = (1.0 / p) * (p_alphas[lang] / p_alpha_sum)
            else:
                logger.warning(
                    f"Language {lang} has no valid samples in the train split."
                )

        if language_ratios_file is not None:
            save_json(content=ratio_map, filename=language_ratios_file)
            logger.info(f"Language ratios saved to {language_ratios_file}")
            logger.info(f"Language ratios: {ratio_map}")

        combined["ratio"] = combined["lang"].map(ratio_map)

    out_csv = path.join(save_folder, f"{split}.csv")
    combined.to_csv(out_csv, index=False)
    logger.info(f"Multilingual {split}.csv written to: {out_csv}")
    return out_csv


def prepare_sense(
    data_folder,
    preparation_output_folder,
    language_ratios_file,
    alpha=0.05,
    skip_prep=False,
    convert_to_wav=False,
    sample_rate=16000,
):
    """
    Prepares multilingual train/dev CSV files for SENSE directly from
    Common Voice TSV files.

    Arguments
    ---------
    data_folder : str
        Root Common Voice folder containing the per-language subfolders:
            <data_folder>/br/train.tsv, dev.tsv, ...
            <data_folder>/sv-SE/train.tsv, dev.tsv, ...
    preparation_output_folder : str
        Directory where the combined CSV files will be stored. This folder
        will contain:
            train.csv  (multilingual, with 'lang' and 'ratio' columns)
            dev.csv    (multilingual, with 'lang' column)
    language_ratios_file : str
        Path to the JSON file where language sampling ratios will be saved
        for the training split.
    alpha : float, optional
        Exponent used in the language sampling ratio formula. Default is 0.05.
    skip_prep : bool, optional
        If True, data preparation is skipped and no CSV files are written.
    convert_to_wav : bool, optional
        If True, `.mp3` files are converted to `.wav` in `process_line`.
    sample_rate : int, optional
        Target sample rate for potential resampling in future extensions.
    """
    if skip_prep:
        return

    if skip_prepared_splits(preparation_output_folder):
        return

    build_combined_split(
        data_folder=data_folder,
        save_folder=preparation_output_folder,
        split="train",
        alpha=alpha,
        language_ratios_file=language_ratios_file,
        convert_to_wav=convert_to_wav,
        sample_rate=sample_rate,
    )

    build_combined_split(
        data_folder=data_folder,
        save_folder=preparation_output_folder,
        split="dev",
        alpha=alpha,
        language_ratios_file=None,
        convert_to_wav=convert_to_wav,
        sample_rate=sample_rate,
    )
