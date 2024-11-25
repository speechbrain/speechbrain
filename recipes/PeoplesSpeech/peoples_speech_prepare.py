"""
This is a preparation script for the People Speech dataset.

Data preparation for people speech is slightly different than usual for
SpeechBrain as it relies exclusively on HuggingFace Datasets. This means that
audio files will NOT be extracted but instead read from shard directly. Instead
we propose to still generate .csv files to have transcriptions and durations
readable to the user. This means that the csv file generation part can totally
be skipped and the training recipe would still work.

TL;DR: .csv files are only generated for debugging/monitoring purpose and are
not necessary to start the recipe.

Download instructions:
    1. https://huggingface.co/datasets/MLCommons/peoples_speech
Reference: https://arxiv.org/abs/2111.09344


Author
-------
 * Titouan Parcollet, 2024
"""

import csv
import functools
import logging
import os
import re
from dataclasses import dataclass

from speechbrain.utils.parallel import parallel_map

logger = logging.getLogger(__name__)

HF_HUB = "MLCommons/peoples_speech"


@dataclass
class PeoplesSpeechRow:
    """Dataclass for handling People's Speech rows.

    Attributes
    ----------
    audio_id : str
        The audio ID.
    duration : float
        The duration in seconds.
    text : str
        The text of the segment.
    """

    audio_id: str  # audio[aid]
    duration: float
    text: str


def prepare_peoples_speech(
    hf_download_folder: str,
    save_folder: str,
    subsets: list,
    skip_prep: bool = False,
) -> None:
    """Download the dataset and csv for Peoples Speech.

    Data preparation for people speech is slightly different than usual for
    SpeechBrain as it relies exclusively on HuggingFace Datasets. This means that
    audio files will NOT be extracted but instead read from shard directly. Instead
    we propose to still generate .csv files to have transcriptions and durations
    readable to the user.

    Download: https://huggingface.co/datasets/MLCommons/peoples_speech
    Reference: https://arxiv.org/abs/2111.09344

    The `train.csv` file is created by combining the sets given in the `subsets`
    variable.

    The `dev.csv` and `test.csv` files are created based on the `DEV` and `TEST` splits
    specified in the `splits` list.

    Parameters
    ----------
    hf_download_folder : str
        The path where HF stored the dataset. Important, you must set the global
        env variable HF_HUB_CACHE to the same path as HuggingFace is primilarily
        using this to know where to store datasets.
    save_folder : str
        The path to the folder where the CSV files will be saved.
    subsets : list
        Target subset. People's speech contains multiple subsets, which must be
        loaded invidividually and then concatenated. E.g. 'clean', 'clean_sac',
        'dirty' or 'dirty_sa'. E.g. to combine  ['clean', 'dirty'].
    skip_prep : bool, optional
        If True, the data preparation will be skipped, and the function will return immediately.

    Returns
    -------
    None
    """

    if not os.path.isdir(hf_download_folder):
        msg = "You must download the dataset with HuggingFace before starting "
        msg += (
            "this recipe. Please check the HuggingFace hub of people's speech."
        )
        raise ValueError(msg)

    if skip_prep:
        logger.info("Skipping data preparation as `skip_prep` is set to `True`")
        return

    if len(subsets) == 0:
        raise ImportError(
            "At least one People's speech subset must be specified."
        )

    # Setting output paths
    save_output = {}
    splits = ["train", "validation", "test"]
    for split in splits:
        save_output[split] = os.path.join(save_folder, str(split) + ".csv")

    # check if the data is already prepared
    if skip_csv(save_output):
        logger.info("Skipping preparation, completed in previous run.")
        return
    else:
        logger.info("Starting data preparation...")

    hf_dataset = load_and_concatenate_datasets(subsets, hf_download_folder)

    logger.info(
        f"Preparing CSV of the Peoples Speech dataset in {save_folder}..."
    )

    os.makedirs(save_folder, exist_ok=True)

    for i, split, output in enumerate(save_output.items()):
        logger.info(f"Starting creating {output} using {split} split.")
        HF_create_csv(output, hf_dataset[i], split)

    logger.info("Data preparation completed!")


def load_and_concatenate_datasets(subsets, hf_download_folder):
    """Load/download and concatenate all the specified subsets from People's
    speech. The people's speech dataset have 4 subset "clean", "clean_sa",
    "dirty" and "dirty_sa". Multiple subsets cannot be loaded all at once with
    HuggingFace so this function makes it possible.

    Parameters
    ----------
    subsets : list
        Target subset. People's speech contains multiple subsets, which must be
        loaded invidividually and then concatenated. E.g. 'clean', 'clean_sac',
        'dirty' or 'dirty_sa'. E.g. to combine  ['clean', 'dirty'].
    hf_download_folder : str
        The path where HF stored the dataset. Important, you must set the global
        env variable HF_HUB_CACHE to the same path as HuggingFace is primilarily
        using this to know where to store datasets.

    Returns
    -------
    List of HuggingFace dataset.
    """

    try:
        import datasets
        from datasets import concatenate_datasets, load_dataset
    except ImportError:
        raise ImportError("HuggingFace datasets must be installed.")

    # Managing the download dir as HF can be capricious with this.
    if "HF_HUB_CACHE" in os.environ:
        hf_caching_dir = os.environ["HF_HUB_CACHE"]
    elif "HF_HOME" in os.environ:
        hf_caching_dir = os.environ["HF_HOME"]
    else:
        hf_caching_dir = os.environ["XDG_CACHE_HOME"]

    if hf_caching_dir != hf_download_folder:
        msg = "HuggingFace HF_HUB_CACHE or HF_HOME is not equal to the given"
        msg += " hf_download_folder. Make sure to set these variables properly."
        raise Exception(msg)

    logger.info("Loading dataset from: " + str(hf_caching_dir))

    import multiprocessing

    nproc = (
        multiprocessing.cpu_count() // 2 + 1
    )  # we don't want to use all cores

    # Setting no download mode for HuggingFace. Only cache.
    # We remove progress bars as they repeat for each DDP process.
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    datasets.disable_progress_bars()
    datasets_list = []
    for subset in subsets:
        hf_data = load_dataset(
            HF_HUB,
            name=subset,
            split=["train"],
            num_proc=nproc,
            cache_dir=hf_caching_dir,
        )
        datasets_list.append(hf_data[0])

    os.environ["HF_DATASETS_OFFLINE"] = "0"

    # Datasets need to be concatenated back.
    final_dataset = []
    if len(datasets_list) > 1:
        final_dataset.append(concatenate_datasets(datasets_list, split="train"))
    else:
        final_dataset.append(datasets_list[0])

    # Now get validation and test
    # Setting no download mode for HuggingFace. Only cache.
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    hf_data = load_dataset(
        HF_HUB,
        name=subset,
        split=["validation", "test"],
        num_proc=nproc,
        cache_dir=hf_caching_dir,
    )
    os.environ["HF_DATASETS_OFFLINE"] = "0"
    datasets.enable_progress_bars()

    final_dataset.append(hf_data[0])
    final_dataset.append(hf_data[1])

    return final_dataset


def HF_create_csv(
    csv_file: str,
    hf_dataset,
    split: str,
) -> None:
    """
    Create a CSV file based on a HuggingFace dataset.

    Parameters
    ----------
    csv_file : str
        The path to the CSV file to be created.
    hf_dataset : huggingface dataset,
        The huggingface dataset.
    split : str
        The split to be used for filtering the data.

    Returns
    -------
    None
    """

    # We don't need to open the audio file. This will speed up drastically.
    hf_dataset = hf_dataset.select_columns(["id", "duration_ms", "text"])

    total_duration = 0.0
    nb_samples = 0

    line_processor = functools.partial(HF_process_line)

    csv_file_tmp = csv_file + ".tmp"
    with open(csv_file_tmp, mode="w", encoding="utf-8") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        header = [
            "audio_id",
            "duration",
            "text",
        ]
        csv_writer.writerow(header)

        for row in parallel_map(line_processor, hf_dataset):
            if row is None:
                continue

            csv_writer.writerow(
                [
                    row.audio_id,
                    str(row.duration),
                    row.text,
                ]
            )

            total_duration += row.duration
            nb_samples += 1

    os.replace(csv_file_tmp, csv_file)

    logger.info(f"{csv_file} successfully created!")
    logger.info(f"Number of samples in {split} split: {nb_samples}")
    logger.info(
        f"Total duration of {split} split: {round(total_duration / 3600, 2)} Hours"
    )


def HF_process_line(row: dict) -> list:
    """
    Process the audio line and return the utterances for the given split.

    Parameters
    ----------
    row: dict
        The audio line to be processed.

    Returns
    -------
    list
        The list of utterances for the given split.
    """
    text = english_specific_preprocess(row["text"])

    if text:
        audio_id = row["id"]
        duration = row["duration_ms"] / 1000  # HF dataset column is in ms.

        row = PeoplesSpeechRow(
            audio_id=audio_id,
            duration=duration,
            text=text,
        )

        return row
    else:
        return None


def skip_csv(save_csv_files: dict) -> bool:
    """Check if the CSV files already exist.

    Parameters
    ----------
    save_csv_files : dict
        The dictionary containing the paths to the CSV files.

    Returns
    -------
    bool
        True if all the CSV files already exist, False otherwise.
    """
    return all(os.path.isfile(path) for path in save_csv_files.values())


def english_specific_preprocess(sentence):
    """
    Preprocess English text from the People's Speech dataset into space-separated
    words. This removes various punctuation and treats it as word boundaries.
    It normalises and retains various apostrophes (’‘´) between letters, but not
    other ones, which are probably quotation marks. It capitalises all text.
    This function may error out if new characters show up in the training, dev,
    or test sets.

    Parameters
    ----------
    sentence : str
        The string to modify.

    Returns
    -------
    str
        The normalised sentence.
    """

    # These characters mark word boundaries.
    split_character_regex = '[ ",:;!?¡\\.…()\\-—–‑_“”„/«»]'

    # These could all be used as apostrophes in the middle of words.
    # If at the start or end of a word, they will be removed.
    apostrophes_or_quotes = "['`´ʻ‘’]"

    sentence_level_mapping = {"&": " and ", "+": " plus ", "ﬂ": "fl"}

    # If it contains anything numerical, we remove it as it is only on val and
    # test. Unfortunately, we can't make sure of what is actually being uttered.
    # Hence, we must throw it away from the evaluation (roughly 1 hours each)
    # if bool(re.search(r'\d', sentence)):
    #    return None

    final_characters = set(" ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890'")

    sentence_mapped = sentence
    if any((source in sentence) for source in sentence_level_mapping):
        for source, target in sentence_level_mapping.items():
            sentence_mapped = sentence_mapped.replace(source, target)

    # Some punctuation that indicates a word boundary.
    words_split = re.split(split_character_regex, sentence_mapped)
    words_quotes = [
        # Use ' as apostrophe.
        # Remove apostrophes at the start and end of words (probably quotes).
        # Word-internal apostrophes, even where rotated, are retained.
        re.sub(apostrophes_or_quotes, "'", word).strip("'")
        for word in words_split
    ]

    # Processing that does not change the length.
    words_upper = [word.upper() for word in words_quotes]

    words_mapped = [
        # word.translate(character_mapping)
        word
        for word in words_upper
        # Previous processing may have reduced words to nothing.
        # Remove them.
        if word != ""
    ]

    result = " ".join(words_mapped)
    character_set = set(result)
    assert character_set <= final_characters, (
        "Unprocessed characters",
        sentence,
        result,
        character_set - final_characters,
    )
    return result
