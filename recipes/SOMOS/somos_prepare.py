"""
SOMOS data preparation

Download: https://zenodo.org/records/7378801
Paper: https://paperswithcode.com/paper/somos-the-samsung-open-mos-dataset-for-the

Authors
 * Artem Ploujnikov 2023
"""
from pathlib import Path
from zipfile import ZipFile
from speechbrain.dataio.dataio import merge_csvs
from speechbrain.utils.superpowers import run_shell
from subprocess import list2cmdline
import os
import re
import csv
import logging

logger = logging.getLogger(__name__)

FILE_AUDIO_ZIP = "audios.zip"
FILE_DATA = "data.csv"
PATH_AUDIOS = "audios"
PATH_METADATA = "training_files/split1/{subset}/{split}_mos_list.txt"
RE_EXT_WAV = re.compile(".wav$")
COLUMN_ID = "ID"
COLUMN_WAV = "wav"
COLUMN_CHAR = "char"
COLUMN_SCORE = "score"
TOOLS_PATH = Path(__file__).parent.parent.parent.parent / "tools"
TOOLS_PATH_VOICEMOS = TOOLS_PATH / "voicemos"
VOICEMOS_NORM_SCRIPT = TOOLS_PATH_VOICEMOS / "sub_normRMSE.sh"


def prepare_somos(
    data_folder,
    save_folder,
    splits=["train", "valid", "test"],
    subset="full",
    use_transcripts=False,
    char_list_file=None,
    audio_preprocessing=None,
):
    """Prepares the csv files for the Somos dataset

    Arguments
    ---------
    data_folder : str | path-like
        Path to the folder where the original LJspeech dataset is stored
    save_folder : str | path-like
        The directory where to store the csv/json files
    splits : list
        List of dataset splits to prepare
    subset : str
        the subset to use:
        "full" for the full dataset
        "clean" for clean data only
    transcripts : bool
        Whether to include transcripts (requires a version of SOMOS where
        transcript/gather_transcripts.py has been run)
    char_list_file : str|path-like
        The list of characters
    audio_preprocessing : str
        The type of audio preprocessing to be applied
        Supported:

        None: no preprocessing will be applied
        voicemos: preprocessing will be applied as specified in the VoiceMOS challenge
    """
    data_folder = Path(data_folder)
    save_folder = Path(save_folder)

    if not data_folder.exists():
        raise ValueError(f"{data_folder} does not exist")
    save_folder.mkdir(parents=True, exist_ok=True)
    extract_audio(data_folder, save_folder)
    # Note: This can be overridden from the command line
    if isinstance(splits, str):
        splits = splits.split(",")
    transcripts = None
    char_set = None
    if use_transcripts:
        if char_list_file is not None:
            char_list = read_list_file(char_list_file)
            char_set = set(char_list)
        transcripts_file_name = data_folder / "transcript/all_transcripts.txt"
        if not transcripts_file_name.exists():
            raise ValueError(
                f"{transcripts_file_name} does not exist, please run "
                "gather_transcripts.py in {data_folder}/transcript"
            )
        transcripts = read_transcripts(transcripts_file_name, char_set)

    if audio_preprocessing:
        audio_preprocessing_cfg = AUDIO_PREPROCESSING.get(audio_preprocessing)
        if audio_preprocessing_cfg is None:
            raise ValueError(
                f"Unsupported preprocessing: {audio_preprocessing}"
            )
        audio_prepare = audio_preprocessing_cfg["prepare"]
        audio_process = audio_preprocessing_cfg["process"]
    else:
        audio_prepare, audio_process = None, None

    if audio_prepare is not None:
        audio_prepare(save_folder)
    for split in splits:
        process_split(
            data_folder, save_folder, split, subset, transcripts, audio_process
        )
    merge_splits(save_folder, splits)


def read_list_file(file_name):
    """Reads a file with a simple list of items - used for
    filtering characters in transcripts

    Arguments
    ---------
    file_name : str|path-like
        The path to the file

    Returns
    -------
    items : list
        The lines from the file
    """
    with open(file_name) as list_file:
        return [line.replace("\r", "").replace("\n", "") for line in list_file]


def extract_audio(data_folder, save_folder):
    """Extracts audio files

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original LJspeech dataset is stored
    save_folder : str
        The directory where to store the csv/json files
    """
    audios_path = Path(data_folder) / PATH_AUDIOS
    if audios_path.exists():
        logging.info(
            "Skipping audio extraction - %s already exists", audios_path
        )
    else:
        audio_archive_path = Path(data_folder) / FILE_AUDIO_ZIP
        logger.info("Extracting audio to %s", save_folder)
        with ZipFile(audio_archive_path) as audio_archive:
            audio_archive.extractall(save_folder)


def get_metadata_columns(use_transcripts=False):
    """Gets the list of columns to be included

    Arguments
    ---------
    use_transcripts : bool
        Whether to include transcripts (requires a version of SOMOS where
        transcript/gather_transcripts.py has been run)

    Returns
    -------
    columns : list
        A list of column names
    """
    columns = [COLUMN_ID, COLUMN_WAV]
    if use_transcripts:
        columns.append(COLUMN_CHAR)
    columns.append(COLUMN_SCORE)
    return columns


def read_transcripts(file_name, char_set):
    """Reads a transcript file

    Arguments
    ---------
    file_name : str|path-like
        The path to the file containing transcripts

    Returns
    -------
    result : dict
        The transcript dictionary
    char_set : set
        The whitelist of allwoed characters
    """

    with open(file_name) as transcript_file:
        records = (
            parse_transcript_line(line.strip(), char_set)
            for line in transcript_file
        )
        return {item_id: transcript for item_id, transcript in records}


def parse_transcript_line(line, char_set):
    """Parses a single line of the transcript

    Arguments
    ---------
    line : str
        A raw line from the file
    char_set : set
        The whitelist of allwoed characters

    Results
    -------
    item_id : str
        The item ID

    transcript : str
        The normalized transcript"""
    item_id, transcript = line.split("\t")
    transcript = transcript.upper()
    if char_set is not None:
        transcript = "".join(char for char in transcript if char in char_set)
    return item_id, transcript


def process_split(
    data_folder, save_folder, split, subset, transcripts=None, process=None
):
    """Processes metadata for the specified split

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original LJspeech dataset is stored
    save_folder : str
        The directory where to store the csv/json files
    split : str
        the split identifier ("train", "valid" or "test")
    subset : str
        the subset to use:
        "full" for the full dataset
        "clean" for clean data only
    transcripts : dict, optional
        The parsed transcripts
    process : callable, optional
        If provided, this function will be applied to the source audio
        file
    """
    src_metadata_file_path = data_folder / PATH_METADATA.format(
        split=split, subset=subset
    )
    tgt_metadata_file_path = save_folder / f"{split}.csv"
    if process:
        processed_folder = save_folder / "processed"
        processed_folder.mkdir(exist_ok=True)
    logger.info(
        "Processing %s - from %s to %s",
        split,
        src_metadata_file_path,
        tgt_metadata_file_path,
    )

    if not src_metadata_file_path.exists():
        raise ValueError(f"{src_metadata_file_path} does not exist")

    metadata_columns = get_metadata_columns(transcripts is not None)

    with open(src_metadata_file_path) as src_file:
        with open(tgt_metadata_file_path, "w") as tgt_file:
            reader = csv.DictReader(src_file)
            writer = csv.DictWriter(tgt_file, metadata_columns)
            writer.writeheader()
            for src_item in reader:
                src_audio_path = (
                    Path(data_folder) / PATH_AUDIOS / src_item["utteranceId"]
                )
                if src_audio_path.exists():
                    tgt_item = process_item(
                        src_item, transcripts, is_processed=process is not None
                    )
                    if process is not None:
                        process(src_audio_path, processed_folder)
                    writer.writerow(tgt_item)
                else:
                    logger.warn("%s does not exist", src_audio_path)


def process_item(item, transcripts, is_processed):
    """Converts a single metadata record to the SpeechBrain
    convention

    Arguments
    ---------
    item : dict
        a single record from the source file
    transcripts : dict
        The parsed transcripts

    Returns
    -------
    result: dict
        the processed item"""
    src_utterance_id = item["utteranceId"]
    tgt_utterance_id = RE_EXT_WAV.sub("", src_utterance_id)
    wav_path = (
        Path("$processed_folder") / src_utterance_id
        if is_processed
        else Path("$data_root") / PATH_AUDIOS / src_utterance_id
    )
    result = {
        "ID": tgt_utterance_id,
        "wav": wav_path,
        "score": item["mean"],
    }
    if transcripts is not None:
        result["char"] = transcripts[tgt_utterance_id]

    return result


def merge_splits(
    save_folder, splits,
):
    """Merges data files into a single file

    Arguments
    ---------
    save_folder : str | path-like
        The directory where to store the csv/json files
    splits : list
        List of dataset splits to prepare
    """
    tgt_file_path = save_folder / FILE_DATA
    csvs = [save_folder / f"{split}.csv" for split in splits]
    merge_csvs(save_folder, csvs, tgt_file_path)


def prepare_voicemos(save_folder):
    """Prepares for the installation of VoiceMOS

    Arguments
    ---------
    save_folder : str
        The path where results and tools will be saved"""
    install_sv56_path = TOOLS_PATH_VOICEMOS / "install_sv56.sh"
    logger.info("Running the sv56 installation script")
    cmd = list2cmdline([str(install_sv56_path), str(save_folder)])
    output, err, return_code = run_shell(cmd)
    if return_code != 0:
        raise PreparationException(
            "\n".join(
                "Unable to install sv56",
                "Command:",
                cmd,
                "Output:",
                output,
                "Errors:",
                err,
            )
        )
    logger.info("sv56 installation was completed")
    sv56_path = save_folder / "src" / "sv56"
    logger.info("Adding %s to the system path", sv56_path)
    current_path = os.environ["PATH"]
    os.environ["PATH"] = current_path + os.pathsep + str(sv56_path)


def process_file_voicemos(file_path, processed_folder):
    """Processes a single file using VoiceMOS challenge scripts

    Arguments
    ---------
    file_path : path-like
        The path to the file
    processed_folder : path-like
        The destination folder
    """
    cmd = list2cmdline(
        [
            str(VOICEMOS_NORM_SCRIPT),
            str(file_path),
            str(TOOLS_PATH),
            str(processed_folder),
        ]
    )
    try:
        run_shell(cmd)
    except OSError as e:
        raise PreparationException(
            "\n".join(
                f"Unable to process {file_path}",
                "Command:",
                cmd,
                "Errors:",
                str(e),
            )
        )


class PreparationException(Exception):
    pass


AUDIO_PREPROCESSING = {
    "voicemos": {"prepare": prepare_voicemos, "process": process_file_voicemos}
}
