"""
This is a preparation script for the People Speech dataset for the LargeScaleASR Set.

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
from dataclasses import dataclass

import soundfile as sf
from datasets import Audio
from nemo_text_processing.text_normalization.normalize import Normalizer

from speechbrain.utils.parallel import parallel_map
from speechbrain.utils.text_normalisation import TextNormaliser

normaliser = Normalizer(input_case="cased", lang="en")

logger = logging.getLogger(__name__)

HF_HUB = "MLCommons/peoples_speech"
LOWER_DURATION_THRESHOLD_IN_S = 1.0
UPPER_DURATION_THRESHOLD_IN_S = 40
LOWER_WORDS_THRESHOLD = 3


@dataclass
class TheLoquaciousRow:
    ID: str
    duration: float
    start: float
    wav: str
    spk_id: str
    sex: str
    text: str


def prepare_peoples_speech(
    hf_download_folder,
    huggingface_folder,
    subsets,
    audio_decoding,
):
    """Download the dataset and csv for Peoples Speech.

    Download: https://huggingface.co/datasets/MLCommons/peoples_speech
    Reference: https://arxiv.org/abs/2111.09344

    The `peoples_peech_train.csv` file is created by combining the sets given in the `subsets` variable -- only the clean for The LargeScaleASR Set.

    No dev and test creation from this dataset for The LargeScaleASR Set.

    Parameters
    ----------
    hf_download_folder : str
        The path where HF stored the dataset. Important, you must set the global
        env variable HF_HUB_CACHE to the same path as HuggingFace is primilarily
        using this to know where to store datasets.
    huggingface_folder : str
        The path to the folder where the CSV files will be saved.
    subsets : list
        Target subset. People's speech contains multiple subsets, which must be
        loaded invidividually and then concatenated. E.g. 'clean', 'clean_sac',
        'dirty' or 'dirty_sa'. E.g. to combine  ['clean', 'dirty'].
    audio_decoding: boolean
        Whether the audio file should be decoded. This is necessary if wav files have not been generated yet -- it will be much slower. False otherwise (like for text processing only.)

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

    if len(subsets) == 0:
        raise ImportError(
            "At least one People's speech subset must be specified."
        )

    # Setting the save folder
    manifest_folder = os.path.join(huggingface_folder, "manifests")
    wav_folder = os.path.join(
        huggingface_folder, os.path.join("data", "peoples_speech")
    )
    os.makedirs(wav_folder, exist_ok=True)

    save_csv_train = manifest_folder + "/people_speech_train.csv"

    # check if the data is already prepared
    if os.path.isfile(save_csv_train):
        logger.info("Skipping preparation, completed in previous run.")
        return
    else:
        logger.info("Starting data preparation...")

    hf_dataset = load_and_concatenate_datasets(
        subsets, hf_download_folder, audio_decoding
    )

    logger.info(
        f"Preparing CSV of the Peoples Speech dataset in {save_csv_train}..."
    )

    HF_create_csv(save_csv_train, hf_dataset[0], wav_folder)

    logger.info("Data preparation completed!")


def load_and_concatenate_datasets(subsets, hf_download_folder, audio_decoding):
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
    audio_decoding: boolean
        Whether the audio file should be decoded. This is necessary if wav files have not been generated yet -- it will be much slower. False otherwise (like for text processing only.)

    Returns
    -------
    List of HuggingFace dataset.
    """

    try:
        import datasets
        from datasets import concatenate_datasets, load_dataset
    except ImportError as error:
        raise ImportError(
            f"{str(error)}\n" f"HuggingFace datasets must be installed."
        )

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
        datasets_list.append(
            hf_data[0].cast_column("audio", Audio(decode=audio_decoding))
        )

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
    csv_file,
    hf_dataset,
    save_folder,
):
    """
    Create a CSV file based on a HuggingFace dataset.

    Parameters
    ----------
    csv_file : str
        The path to the CSV file to be created.
    hf_dataset : huggingface dataset,
        The huggingface dataset.
    save_folder : str
        Where the wav files will be stored.

    """

    hf_dataset = hf_dataset.select_columns(
        ["id", "duration_ms", "text", "audio"]
    )

    total_duration = 0.0
    nb_samples = 0

    text_norm = TextNormaliser()
    line_processor = functools.partial(
        HF_process_line, save_folder=save_folder, text_normaliser=text_norm
    )

    csv_file_tmp = csv_file + ".tmp"
    with open(csv_file_tmp, mode="w", encoding="utf-8") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        header = ["ID", "duration", "start", "wav", "spk_id", "sex", "text"]
        csv_writer.writerow(header)

        for row in parallel_map(line_processor, hf_dataset):
            if row is None:
                continue

            csv_writer.writerow(
                [
                    row.ID,
                    str(row.duration),
                    str(row.start),
                    row.wav,
                    row.spk_id,
                    row.sex,
                    row.text,
                ]
            )

            total_duration += row.duration
            nb_samples += 1

    os.replace(csv_file_tmp, csv_file)

    logger.info(f"{csv_file} successfully created!")
    logger.info(f"Number of samples in: {nb_samples}")
    logger.info(f"Total duration: {round(total_duration / 3600, 2)} Hours")


def HF_process_line(row, save_folder, text_normaliser):
    """
    Process the audio line and return the utterances for the given split.

    Parameters
    ----------
    row: dict
        The audio line to be processed.
    save_folder: str
        Path to where the wav files will be stored.
    text_normaliser: speechbrain.utils.text_normalisation.TextNormaliser

    Returns
    -------
    TheLoquaciousRow
        A dataclass containing the information about the line.
    text_norm : speechbrain.utils.text_normalisation.TextNormaliser
    """

    text = normaliser.normalize(row["text"])
    text = text_normaliser.english_specific_preprocess(text)

    if text:
        audio_id = row["id"]
        duration = row["duration_ms"] / 1000  # HF dataset column is in ms.

        if duration < 1.0 or duration > UPPER_DURATION_THRESHOLD_IN_S:
            return None

        if text is None or len(text) < LOWER_WORDS_THRESHOLD:
            return None

        wav_path = os.path.join(save_folder, audio_id)
        if "sampling_rate" in row["audio"]:
            if row["audio"]["sampling_rate"] == 16000:
                if not os.path.isfile(wav_path):
                    sf.write(wav_path, row["audio"]["array"], 16000)
            else:
                return None

        row = TheLoquaciousRow(
            ID=audio_id,
            duration=duration,
            start=-1,
            wav=wav_path,
            spk_id=None,
            sex=None,
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
