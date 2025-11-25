"""
Data preparation for SENSE using multilingual Common Voice TSV files.

Download: https://commonvoice.mozilla.org/lang/datasets

Authors
-------
 * Maryem Bouziane 2025
 * Salima Mdhaffar 2025
 * Haroun Elleuch 2025
 * Yannick Estève 2025
"""

import os
from os import path
import functools
import json

import pandas as pd
from tqdm import tqdm

from speechbrain.utils.logger import get_logger
from speechbrain.utils.parallel import parallel_map
from recipes.CommonVoice.common_voice_prepare import (
    process_line,
    check_commonvoice_folders,
)

logger = get_logger(__name__)

# Maximum duration (in seconds)
DURATION_MAX = 10.0


def prepare_sense(
    data_folder,
    output_folder,
    languages,
    sampling_alpha,
    language_ratios_file,
    train_csv,
    valid_csv,
    convert_to_wav: bool = False,
) -> None:
    """
    Prepares multilingual train/dev CSV files for SENSE from Common Voice TSVs.

    This function iterates over the selected languages and splits, builds
    multilingual CSV manifests for the train and dev sets, and optionally
    computes and saves language sampling ratios used by the training sampler.

    Arguments
    ---------
    data_folder : str
        Root Common Voice folder containing the per-language subfolders.
    output_folder : str
        Directory where the combined CSV files will be stored.
    languages : list of str
        List of Common Voice language codes to include in the multilingual
        split (e.g., ["fr", "de", "ar"]).
    sampling_alpha : float
        Exponent used in the language sampling ratio formula for the train
        split. Values close to 0 yield more uniform sampling, values closer
        to 1 follow the empirical language distribution.
    language_ratios_file : str
        Path to the JSON file where language sampling ratios for the train
        split will be saved. If the path is empty or None, ratios are not
        written to disk.
    train_csv : str
        Output path for the combined train CSV file.
    valid_csv : str
        Output path for the combined dev/validation CSV file.
    convert_to_wav : bool, optional
        If True, `.mp3` files are converted to `.wav` in ``process_line``.

    Returns
    -------
    None
    """
    create_directory(output_folder)

    if skip_prepared_splits(train_csv, valid_csv):
        return

    # Train split
    build_combined_split(
        data_folder=data_folder,
        languages=languages,
        split="train",
        out_csv=train_csv,
        alpha=sampling_alpha,
        language_ratios_file=language_ratios_file,
        convert_to_wav=convert_to_wav,
    )

    # Dev split
    build_combined_split(
        data_folder=data_folder,
        languages=languages,
        split="dev",
        out_csv=valid_csv,
        alpha=sampling_alpha,
        language_ratios_file=None,
        convert_to_wav=convert_to_wav,
    )


def create_directory(dir_path: str) -> None:
    """
    Creates a directory if it does not already exist.

    Arguments
    ---------
    dir_path : str
        Path of the directory to create. If empty or None, nothing is done.

    Returns
    -------
    None
    """
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)


def save_json(content, filename: str) -> None:
    """
    Saves a Python object to a JSON file.

    The parent directory is created if it does not exist.

    Arguments
    ---------
    content : object
        Serializable Python object to be written as JSON.
    filename : str
        Path of the JSON file to create. If empty or None, the function
        returns without writing anything.

    Returns
    -------
    None
    """
    if not filename:
        return

    directory = os.path.dirname(filename)
    if directory:
        create_directory(directory)

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(content, f, ensure_ascii=False, indent=2)


def skip_prepared_splits(train_csv: str, dev_csv: str) -> bool:
    """
    Detects if the train and dev CSV files already exist.

    If both files are present, the data preparation step can be safely
    skipped.

    Arguments
    ---------
    train_csv : str
        Path to the train CSV file.
    dev_csv : str
        Path to the dev/validation CSV file.

    Returns
    -------
    bool
        True if both CSV files already exist and preparation can be skipped,
        False otherwise.
    """
    if all(path.isfile(p) for p in [train_csv, dev_csv]):
        msg = (
            "%s and %s already exist, skipping data preparation!"
            % (train_csv, dev_csv)
        )
        logger.info(msg)
        return True
    return False


def read_split_lang_to_df(
    data_folder: str,
    lang: str,
    split: str,
    convert_to_wav: bool = False,
    duration_max: float = DURATION_MAX,
):
    """
    Reads one split (train/dev) for a given language and returns a DataFrame.

    The function:
      * loads the corresponding Common Voice TSV file,
      * processes audio lines with ``process_line``,
      * filters out utterances with invalid or too long durations,
      * attaches the raw transcription from the TSV file.

    The resulting DataFrame has the following columns:

        ID, duration, wav, spk_id, text

    Arguments
    ---------
    data_folder : str
        Root Common Voice directory containing the per-language subfolders.
    lang : str
        Language code to process (e.g., "fr", "br", "sv-SE").
    split : str
        Name of the split to load ("train" or "dev").
    convert_to_wav : bool, optional
        If True, audio files are converted to WAV by ``process_line``.
    duration_max : float, optional
        Maximum allowed duration in seconds. Utterances with
        ``duration >= duration_max`` are discarded.

    Returns
    -------
    pandas.DataFrame or None
        DataFrame containing the selected utterances for this language and
        split, or None if no valid samples are found.

    Raises
    ------
    FileNotFoundError
        If the expected TSV file for this language and split does not exist.
    """
    data_folder_lang = path.join(data_folder, lang)
    orig_tsv_file = path.join(data_folder_lang, f"{split}.tsv")

    # If a language is listed in `languages`, its TSV file is expected to exist.
    if not path.isfile(orig_tsv_file):
        msg = "%s doesn't exist, verify your dataset!" % (orig_tsv_file)
        logger.info(msg)
        raise FileNotFoundError(msg)

    check_commonvoice_folders(data_folder_lang)

    msg = "Reading %s.tsv for language %s: %s" % (
        split,
        lang,
        orig_tsv_file,
    )
    logger.info(msg)
    with open(orig_tsv_file, encoding="utf-8") as f:
        csv_lines = f.readlines()

    if len(csv_lines) <= 1:
        msg = "No usable data in %s" % (orig_tsv_file)
        logger.warning(msg)
        return None

    header_line = csv_lines[0]
    data_lines = csv_lines[1:]

    header_map = {
        column_name: index
        for index, column_name in enumerate(header_line.split("\t"))
    }

    if "sentence" not in header_map:
        raise KeyError(
            "Expected 'sentence' column in Common Voice TSV header, "
            "got: %s" % (list(header_map.keys()))
        )

    # Map sentence id (snt_id) → raw transcription from TSV
    raw_text_by_id = {}
    for line in data_lines:
        cols = line.rstrip("\n").split("\t")
        audio_path_filename = cols[header_map["path"]]
        snt_id = audio_path_filename.split(".")[-2].split("/")[-1]
        raw_sentence = cols[header_map["sentence"]]
        raw_text_by_id[snt_id] = raw_sentence

    # Audio processing and duration filtering
    line_processor = functools.partial(
        process_line,
        convert_to_wav=convert_to_wav,
        data_folder=data_folder_lang,
        language=lang,
        accented_letters=False,
        header_map=header_map,
    )

    processed_rows = parallel_map(line_processor, data_lines)

    ids = []
    durations = []
    wavs = []
    spk_ids = []
    texts = []
    total_duration = 0.0

    for row in processed_rows:
        if row is None:
            continue
        if row.duration == 0.0:
            continue
        if row.duration >= duration_max:
            continue

        full_id = row.snt_id
        ids.append(full_id)
        durations.append(row.duration)
        wavs.append(row.audio_path)
        spk_ids.append(row.spk_id)
        raw_sentence = raw_text_by_id.get(row.snt_id, "")
        texts.append(raw_sentence)
        total_duration += row.duration

    if len(ids) == 0:
        msg = "No valid samples for %s / %s" % (lang, split)
        logger.warning(msg)
        return None

    msg = "%s / %s: kept %s utterances for a total of %.2f hours" % (
        lang,
        split,
        len(ids),
        total_duration / 3600.0,
    )
    logger.info(msg)

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


def build_combined_split(
    data_folder: str,
    languages,
    split: str,
    out_csv: str,
    alpha: float = 0.05,
    language_ratios_file=None,
    convert_to_wav: bool = False,
) -> str:
    """
    Builds a multilingual split (train or dev) from Common Voice TSV files.

    For each language in ``languages``, this function:
      * reads the corresponding TSV file,
      * processes and filters utterances using ``read_split_lang_to_df``,
      * concatenates all languages into a single CSV manifest.

    For the ``train`` split, it also computes per-language sampling ratios
    according to:

        p_l = N_l / N_total
        r_l = (1 / p_l) * (p_l ** alpha / sum_k p_k ** alpha)

    where ``p_l`` is the empirical probability of language ``l`` and ``r_l``
    is the sampling ratio used by the sampler. In practice, this makes
    low-resource languages appear more often (oversampling) and
    high-resource languages less often (undersampling) during training,
    while still taking the original data distribution into account.

    Arguments
    ---------
    data_folder : str
        Root Common Voice directory containing the per-language subfolders.
    languages : list of str
        List of language codes to include in the multilingual split.
    split : str
        Name of the split to build ("train" or "dev").
    out_csv : str
        Path to the output CSV file that will contain all selected utterances
        for this split.
    alpha : float, optional
        Smoothing exponent used in the language sampling ratio formula for
        the train split. Defaults to 0.05.
    language_ratios_file : str or None, optional
        Path to the JSON file where the language ratios will be saved for the
        train split. If None, ratios are not written to disk.
    convert_to_wav : bool, optional
        If True, audio files are converted to WAV inside
        ``read_split_lang_to_df`` via ``process_line``.

    Returns
    -------
    str
        Path to the output CSV file that has been written.

    Raises
    ------
    RuntimeError
        If no valid samples are found for the given split across all
        selected languages.
    """
    msg = "Building multilingual %s split..." % (split)
    logger.info(msg)
    create_directory(os.path.dirname(out_csv))

    dfs = []
    language_counts = {}

    for lang in tqdm(languages, desc="Split %s" % (split)):
        df_lang = read_split_lang_to_df(
            data_folder=data_folder,
            lang=lang,
            split=split,
            convert_to_wav=convert_to_wav,
            duration_max=DURATION_MAX,
        )
        if df_lang is None or df_lang.empty:
            continue

        df_lang["lang"] = lang
        language_counts[lang] = len(df_lang)
        dfs.append(df_lang)

    if len(dfs) == 0:
        raise RuntimeError(
            "No valid samples for split %s across the selected languages."
            % (split)
        )

    combined = pd.concat(dfs, ignore_index=True)

    if split == "train":
        total = len(combined)
        if total == 0:
            raise RuntimeError("Combined train split is empty after filtering.")

        ratio_map = {}
        ps = {}
        p_alphas = {}

        # p_l = N_l / N_total, then p_l**alpha
        for lang, count in language_counts.items():
            p = count / total
            ps[lang] = p
            p_alphas[lang] = p**alpha

        # Sum over all languages once p_l**alpha is known
        p_alpha_sum = sum(p_alphas.values())

        # r_l = (1 / p_l) * (p_l**alpha / sum_k p_k**alpha)
        for lang, p in ps.items():
            if p > 0:
                ratio_map[lang] = (1.0 / p) * (p_alphas[lang] / p_alpha_sum)
            else:
                msg = (
                    "Language %s has no valid samples in the train split."
                    % (lang)
                )
                logger.warning(msg)

        if language_ratios_file is not None:
            save_json(content=ratio_map, filename=language_ratios_file)
            msg = "Language ratios saved to %s" % (language_ratios_file)
            logger.info(msg)
            msg = "Language ratios: %s" % (ratio_map)
            logger.info(msg)

        combined["ratio"] = combined["lang"].map(ratio_map)

    combined.to_csv(out_csv, index=False)
    msg = "Multilingual %s.csv written to: %s" % (split, out_csv)
    logger.info(msg)
    return out_csv
