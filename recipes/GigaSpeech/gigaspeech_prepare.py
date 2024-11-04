"""
Data preparation script for the GigaSpeech dataset.

Download instructions:
    1. https://github.com/SpeechColab/GigaSpeech
    2. https://huggingface.co/datasets/speechcolab/gigaspeech
Reference: https://arxiv.org/abs/2106.06909

Author
-------
 * Adel Moumen, 2024
"""

import csv
import functools
import json
import logging
import os
from dataclasses import dataclass

import torchaudio

from speechbrain.utils.parallel import parallel_map

logger = logging.getLogger(__name__)
FILLERS = [
    "UH",
    "UHH",
    "UM",
    "EH",
    "MM",
    "HM",
    "AH",
    "HUH",
    "HA",
    "ER",
    "OOF",
    "HEE",
    "ACH",
    "EEE",
    "EW",
]
GARBAGE_UTTERANCE_TAGS = ["<SIL>", "<MUSIC>", "<NOISE>", "<OTHER>"]
PUNCTUATION_TAGS = {
    "<COMMA>": ",",
    "<EXCLAMATIONPOINT>": "!",
    "<PERIOD>": ".",
    "<QUESTIONMARK>": "?",
}
SPLITS = ["DEV", "TEST"]
TRAIN_SUBSET = ["XS", "S", "M", "L", "XL"]
SAMPLING_RATE = 16000


@dataclass
class GigaSpeechRow:
    """Dataclass for handling GigaSpeech rows.

    Attributes
    ----------
    utt_id : str
        The segment ID.
    audio_id : str
        The audio ID.
    audio_path : str
        The path to the audio file.
    speaker : str
        The speaker ID.
    begin_time : float
        The start time of the segment.
    end_time : float
        The end time of the segment.
    duration : float
        The duration of the segment.
    text : str
        The text of the segment.
    """

    utt_id: str  # segment[sid]
    audio_id: str  # audio[aid]
    audio_path: str  # by default this is opus files
    speaker: str  # audio["speaker"]
    begin_time: float
    end_time: float
    duration: float
    text: str


def prepare_gigaspeech(
    data_folder: str,
    save_folder: str,
    splits: list,
    output_train: str,
    output_dev: str,
    output_test: str,
    json_file: str = "GigaSpeech.json",
    skip_prep: bool = False,
    convert_opus_to_wav: bool = True,
    download_with_HF: bool = False,
    punctuation: bool = False,
    filler: bool = False,
    hf_multiprocess_load: bool = True,
) -> None:
    """Prepare the csv files for GigaSpeech dataset.

    Download instructions: https://github.com/SpeechColab/GigaSpeech
    Reference: https://arxiv.org/abs/2106.06909

    The `train.csv` file is created by following the train subset specified in the `splits` list.
    It must be part of the `TRAIN_SUBSET` list. You cannot use multiple train subsets.

    The `dev.csv` and `test.csv` files are created based on the `DEV` and `TEST` splits
    specified in the `splits` list.

    Parameters
    ----------
    data_folder : str
        The path to the GigaSpeech dataset.
    save_folder : str
        The path to the folder where the CSV files will be saved.
    splits : list
        The list of splits to be used for creating the CSV files.
    output_train : str
        The path in which the train CSV or shards will be saved.
    output_dev : str
        The path in which the dev CSV or shards will be saved.
    output_test : str
        The path in which the test CSV or shards will be saved.
    json_file : str, optional
        The name of the JSON file containing the metadata of the GigaSpeech dataset.
    skip_prep : bool, optional
        If True, the data preparation will be skipped, and the function will return immediately.
    convert_opus_to_wav : bool, optional
        If True, the opus files will be converted to wav files.
    download_with_HF : bool, optional
        If True, the dataset will be downloaded using the Hugging Face datasets library.
        We highly recommend using this option if you are based in the EU or US as it will
        be faster and more reliable than the official host. Make sure to read the
        instructions on how to get the dataset from Hugging Face here:
        https://huggingface.co/datasets/speechcolab/gigaspeech
        The dataset will be downloaded in the default folder specified in the
        environment variable HF_HUB_CACHE. Please change it if necessary.
    punctuation : bool, optional
        Keeping the punctuation, or not.
    filler : bool, optional
        Keeping filler words (hum), or not.
    hf_multiprocess_load: bool, optional
        If True, all the CPU threads will be used for data prepration. If set to
        False, only one will be. Note that the data prepration of the larger sets
        on a single core car take more than 24 hours (from downloading to done).

    Returns
    -------
    None
    """
    logger.info(f"Preparing GigaSpeech dataset in {save_folder}...")

    if skip_prep:
        logger.info("Skipping data preparation as `skip_prep` is set to `True`")
        return

    # check that `splits` input is valid
    for split in splits:
        assert (
            split in SPLITS + TRAIN_SUBSET
        ), f"Split {split} not recognized. Valid splits are {SPLITS + TRAIN_SUBSET}."

    # check that we are not using multiple train subsets
    if len(set(splits).intersection(TRAIN_SUBSET)) > 1:
        raise ValueError(
            "You cannot use multiple train subsets. Please select only one train subset."
        )

    os.makedirs(save_folder, exist_ok=True)

    # Setting output paths
    save_output = {}
    split_map = {}
    train_split = ""
    for split in splits:
        if split in TRAIN_SUBSET:
            save_output["train"] = output_train
            split_map["train"] = split
            train_split = split
        else:
            if split == "DEV":
                save_output["validation"] = output_dev
                split_map["validation"] = split
            elif split == "TEST":
                save_output["test"] = output_test
                split_map["test"] = split

    # check if the data is already prepared
    if skip_csv(save_output):
        logger.info("Skipping preparation, completed in previous run.")
        return
    else:
        logger.info("Starting data preparation...")

    if download_with_HF:
        from datasets import load_dataset

        if os.path.exists("dataset.py"):
            logger.info("HuggingFace dataset.py found.")
        else:
            raise FileNotFoundError(
                "HuggingFace dataset.py not found. Please run this recipe from the correct recipe folder or copy the dataset.py file."
            )

        if "HF_HUB_CACHE" in os.environ:
            hf_caching_dir = os.environ["HF_HUB_CACHE"]
        elif "HF_HOME" in os.environ:
            hf_caching_dir = os.environ["HF_HOME"]
        else:
            hf_caching_dir = os.environ["XDG_CACHE_HOME"]

        logger.info(
            "Downloading dataset from HuggingFace to: " + str(hf_caching_dir)
        )
        logger.info(
            "To change this directory modify the HF_HUB_CACHE env. variable."
        )

        nproc = 1
        if hf_multiprocess_load:
            import multiprocessing

            nproc = multiprocessing.cpu_count()

        hf_dataset = load_dataset(
            "dataset.py",
            train_split.lower(),
            trust_remote_code=True,
            data_dir=data_folder,
            cache_dir=data_folder,
            num_proc=nproc,
        )
        for split, output in save_output.items():
            logger.info(f"Starting creating {output} using {split} split.")
            HF_create_csv(output, hf_dataset[split], split, punctuation, filler)
    else:
        # check that the data folder contains the GigaSpeech dataset
        check_gigaspeech_folders(data_folder, json_file)

        logger.info(f"Starting reading {json_file}.")
        with open(json_file, "r", encoding="utf-8") as f:
            info = json.load(f)
        logger.info(f"Reading {json_file} done.")

        for split, output in save_output.items():
            logger.info(f"Starting creating {output} using {split} split.")
            create_csv(
                output,
                info,
                data_folder,
                split_map[split],
                convert_opus_to_wav,
                punctuation,
                filler,
            )
    logger.info("Data preparation completed!")


def process_line(
    audio: json,
    data_folder: str,
    split: str,
    convert_opus_to_wav: bool,
    punctuation: bool,
    stopwords: list,
) -> list:
    """
    Process the audio line and return the utterances for the given split.

    Parameters
    ----------
    audio : dict
        The audio line to be processed.
    data_folder : str
        The path to the GigaSpeech dataset.
    split : str
        The split to be used for filtering the data.
    convert_opus_to_wav : bool
        If True, the opus files will be converted to wav files.
    punctuation : bool
        Keeping punctuation or not. Default is no.
    stopwords: list
        List of stopwords to remove from the text of the labels.

    Returns
    -------
    list
        The list of utterances for the given split.
    """
    if ("{" + split + "}") in audio["subsets"]:

        audio_path = os.path.join(data_folder, audio["path"])
        assert os.path.isfile(audio_path), f"File not found: {audio_path}"

        if convert_opus_to_wav and audio_path.endswith(".opus"):
            audio_path = convert_opus2wav(audio_path)

        # 2. iterate over the utterances
        utterances = []
        for segment in audio["segments"]:
            text = preprocess_text(segment["text_tn"], punctuation, stopwords)
            if text:
                begin_time = float(segment["begin_time"])
                end_time = float(segment["end_time"])
                duration = end_time - begin_time
                utterance = GigaSpeechRow(
                    utt_id=segment["sid"],
                    audio_id=audio["aid"],
                    audio_path=str(audio_path),
                    speaker=audio["speaker"],
                    begin_time=begin_time,
                    end_time=end_time,
                    duration=duration,
                    text=text,
                )
                utterances.append(utterance)
        return utterances


def create_csv(
    csv_file: str,
    info: json,
    data_folder: str,
    split: str,
    convert_opus_to_wav: bool,
    punctuation: bool = False,
    filler: bool = False,
) -> None:
    """
    Create a CSV file based on the info in the GigaSpeech JSON file and filter the data based on the split.

    Parameters
    ----------
    csv_file : str
        The path to the CSV file to be created.
    info : dict
        The GigaSpeech JSON file content.
    data_folder : str
        The path to the GigaSpeech dataset.
    split : str
        The split to be used for filtering the data.
    convert_opus_to_wav : bool
        If True, the opus files will be converted to wav files.
    punctuation : bool
        Keeping punctuation or not. Default is no.
    filler : bool
        Keeping filler words or not (hum, er). Default is no.

    Returns
    -------
    None
    """
    total_duration = 0.0
    nb_samples = 0

    to_remove = GARBAGE_UTTERANCE_TAGS
    if not filler:
        to_remove += FILLERS

    line_processor = functools.partial(
        process_line,
        data_folder=data_folder,
        split=split,
        convert_opus_to_wav=convert_opus_to_wav,
        stopwords=to_remove,
        punctuation=punctuation,
    )

    csv_file_tmp = csv_file + ".tmp"
    with open(csv_file_tmp, mode="w", encoding="utf-8") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        header = [
            "ID",
            "audio_id",
            "audio_path",
            "speaker",
            "begin_time",
            "end_time",
            "duration",
            "text",
        ]
        csv_writer.writerow(header)
        for row in parallel_map(line_processor, info["audios"]):
            if row is None:
                continue

            for item in row:
                csv_writer.writerow(
                    [
                        item.utt_id,
                        item.audio_id,
                        item.audio_path,
                        item.speaker,
                        str(item.begin_time),
                        str(item.end_time),
                        str(item.duration),
                        item.text,
                    ]
                )

                total_duration += item.duration
                nb_samples += 1

    os.replace(csv_file_tmp, csv_file)

    logger.info(f"{csv_file} successfully created!")
    logger.info(f"Number of samples in {split} split: {nb_samples}")
    logger.info(
        f"Total duration of {split} split: {round(total_duration / 3600, 2)} Hours"
    )


def HF_create_csv(
    csv_file: str,
    hf_dataset,
    split: str,
    punctuation: bool = False,
    filler: bool = False,
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
    punctuation : bool
        Keeping punctuation or not. Default is no.
    filler : bool
        Keeping filler words or not (hum, er). Default is no.


    Returns
    -------
    None
    """
    total_duration = 0.0
    nb_samples = 0

    to_remove = GARBAGE_UTTERANCE_TAGS
    if not filler:
        to_remove += FILLERS

    line_processor = functools.partial(
        HF_process_line,
        stopwords=to_remove,
        punctuation=punctuation,
    )

    csv_file_tmp = csv_file + ".tmp"
    with open(csv_file_tmp, mode="w", encoding="utf-8") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        header = [
            "ID",
            "audio_id",
            "audio_path",
            "speaker",
            "begin_time",
            "end_time",
            "duration",
            "text",
        ]
        csv_writer.writerow(header)

        for row in parallel_map(line_processor, hf_dataset, chunk_size=1024):
            if row is None:
                continue

            csv_writer.writerow(
                [
                    row.utt_id,
                    row.audio_id,
                    row.audio_path,
                    row.speaker,
                    str(row.begin_time),
                    str(row.end_time),
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


def HF_process_line(row: dict, punctuation: bool, stopwords: list) -> list:
    """
    Process the audio line and return the utterances for the given split.

    Parameters
    ----------
    row: dict
        The audio line to be processed.
    punctuation : bool
        Keeping punctuation or not. Default is no.
    stopwords: list
        List of stopwords to remove from the text of the labels.

    Returns
    -------
    list
        The list of utterances for the given split.
    """
    audio_path = os.path.join(row["audio"]["path"])

    if not os.path.isfile(audio_path):
        return None

    # check reading the audio file ; HF may have some corrupted files
    try:
        _ = torchaudio.info(audio_path)
    except Exception as e:
        logger.error(f"Failed reading {audio_path}: {e}")
        return None

    text = preprocess_text(row["text"], punctuation, stopwords)

    if text:
        utt_id = row["segment_id"]
        audio_id = row["audio_id"]
        audio_path = row["audio"]["path"]
        speaker = row["speaker"]
        begin_time = float(row["begin_time"])
        end_time = float(row["end_time"])
        duration = end_time - begin_time

        row = GigaSpeechRow(
            utt_id=utt_id,
            audio_id=audio_id,
            audio_path=audio_path,
            speaker=speaker,
            begin_time=begin_time,
            end_time=end_time,
            duration=duration,
            text=text,
        )

        return row
    else:
        return None


def convert_opus2wav(audio_opus_path):
    """Convert an opus file to a wav file.

    Parameters
    ----------
    audio_opus_path : str
        The path to the opus file to be converted.

    Returns
    -------
    str
        The path to the converted wav file.

    Raises
    ------
    subprocess.CalledProcessError
        If the conversion process fails.
    """
    audio_wav_path = audio_opus_path.replace(".opus", ".wav")
    os.system(
        f"ffmpeg -y -i {audio_opus_path} -ac 1 -ar {SAMPLING_RATE} {audio_wav_path} > /dev/null 2>&1"
    )
    return audio_wav_path


def preprocess_text(text: str, punctuation: bool, stopwords) -> str:
    """
    Preprocesses the input text by removing garbage tags and removing punctuation
    and filler words if specified.

    Parameters
    ----------
    text : str
        The input text to be preprocessed.
    punctuation : bool
        Keeping punctuation or not. Default is no.
    stopwords : list
        List of words to remove from the input test string.

    Returns
    -------
    str
        The preprocessed text with removed garbage tags and replaced punctuation tags.

    Raises
    ------
    AssertionError
        If '<' or '>' tags are found in the text after preprocessing.

    Notes
    -----
    The function iterates over predefined garbage utterance tags (GARBAGE_UTTERANCE_TAGS)
    and removes them from the input text. It then iterates over predefined punctuation tags
    (PUNCTUATION_TAGS) and replaces them with the corresponding punctuation.

    Examples
    --------
    >>> text = " DOUGLAS MCGRAY IS GOING TO BE OUR GUIDE YOU WALK THROUGH THE DOOR <COMMA> YOU SEE THE RED CARPETING <COMMA> YOU SEE SOMEONE IN A SUIT <PERIOD> THEY MAY BE GREETING YOU <PERIOD>"
    >>> preprocess_text(text, punctuation=True, stopwords=GARBAGE_UTTERANCE_TAGS)
    "DOUGLAS MCGRAY IS GOING TO BE OUR GUIDE YOU WALK THROUGH THE DOOR, YOU SEE THE RED CARPETING, YOU SEE SOMEONE IN A SUIT. THEY MAY BE GREETING YOU."
    """

    text = text.upper()
    text = text.replace("-", " ")

    sentence = " ".join(
        [word for word in text.split() if word not in stopwords]
    )

    if punctuation:
        for tag, punctuation in PUNCTUATION_TAGS.items():
            sentence = sentence.replace(" " + tag, punctuation)

    return sentence


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


def check_gigaspeech_folders(
    data_folder: str,
    json_file: str = "GigaSpeech.json",
    audio_folder: str = "audio",
) -> None:
    """Check if the data folder actually contains the GigaSpeech dataset.

    If it does not, an error is raised.

    Parameters
    ----------
    data_folder : str
        The path to the GigaSpeech dataset.
    json_file : str, optional
        The name of the JSON file containing the metadata of the GigaSpeech dataset.
    audio_folder : str, optional
        The name of the folder containing the audio files of the GigaSpeech dataset.

    Returns
    -------
    None

    Raises
    ------
    OSError
        If GigaSpeech is not found at the specified path.
    """
    # Checking if "GigaSpeech.json" exist
    if not os.path.exists(json_file):
        err_msg = (
            "the opus file %s does not exist (it is expected in the "
            "Gigaspeech dataset)" % json_file
        )
        raise OSError(err_msg)

    # Check if audio folders exist
    for folder_subset in ["audiobook", "podcast", "youtube"]:
        audio_subset = os.path.join(data_folder, audio_folder, folder_subset)
        if not os.path.exists(audio_subset):
            err_msg = (
                "the file %s does not exist (it is expected in the "
                "Gigaspeech dataset)" % audio_subset
            )
            raise OSError(err_msg)
