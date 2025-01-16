"""
Utilities to merge csv together. This is done with the LargeScaleASR Set in mind.
"""

import csv
import functools
import os

import numpy as np
import soundfile as sf
from tqdm import tqdm

from speechbrain.utils.logger import get_logger
from speechbrain.utils.parallel import parallel_map

MAX_DURATION_LIMIT = 40
MAX_WORD_COUNT = 150
SEED = 666

logger = get_logger(__name__)


def merge_csv_files(
    source_csvs: list[str],
    dest_csv: str,
    hours_per_csv=-1,
    duration_column_indice=1,
    split_and_clean_wavs=True,
):
    """This function takes a list of speechbrain csv files and concatenate them. The lines are shuffled. This function can also pick only a subset of
    a given duration from each file. CSVS MUST HAVE THE SAME HEADER.

    Arguments
    ---------
    source_csvs : list
        List of paths pointing to individual csv files (must have the same header).
    dest_csv : str
        Path and name of the csv to save containing the concatenated csvs.
    hours_per_csv : float
        Duration to pick from every csv in csvs. If not specified, everything
        is taken. The duration column must be in seconds.
    duration_column_indice : float
        If hours_per_csv is specified, then the indice of the column (starting at zero) must be specified to retrieve the value from each line.
    split_and_clean_wavs : bool
        If True, wav that must be accessed with start and duration (slicing)
        will be converted to single audio files using soundfile. False will just keep that. Start is set to -1 once done.
    Returns
    -------
    None

    """

    if len(source_csvs) == 1:
        return

    header = []
    # Check headers and number of fields.
    for csv_file in source_csvs:
        with open(csv_file, "r", encoding="utf-8") as file:
            header.append(file.readline())

    if header[0].count(header[0]) != len(header):
        logger.error("The headers of the provided csv are different!")

    all_lines = []
    logger.info("Now loading all the csvs in memory with correct durations.")

    np.random.seed(SEED)
    total_duration = 0.0
    for csv_file in source_csvs:
        with open(csv_file, "r", encoding="utf-8") as file:
            tmp_lines = file.readlines()[1:]
            random_indices = np.arange(len(tmp_lines))
            np.random.shuffle(random_indices)
            current_duration = 0.0
            for indice in tqdm(random_indices):
                duration = float(
                    tmp_lines[indice].split(",")[duration_column_indice]
                )
                if duration > MAX_DURATION_LIMIT:
                    continue
                if hours_per_csv != -1:
                    if current_duration + duration > (hours_per_csv * 3600):
                        break
                all_lines.append(tmp_lines[indice].split("\n")[0])
                current_duration += duration

            total_duration += current_duration

    logger.info("New total duration is: " + str(total_duration / 3600))
    logger.info("Writing back to: " + str(dest_csv))

    if split_and_clean_wavs:
        line_processor = functools.partial(process_line_split_and_clean_wav)
        wavs_to_delete = []
        with open(dest_csv, mode="w", newline="", encoding="utf-8") as csv_f:
            csv_writer = csv.writer(
                csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            csv_writer.writerow(header[0].split("\n")[0].split(","))
            for row, row2 in parallel_map(line_processor, all_lines):
                if row is None:
                    continue

                if row2 is not None:
                    wavs_to_delete.append(row2)
                csv_writer.writerow(row)

        logger.info("Deleting old wav files.")
        for line in wavs_to_delete:
            if os.path.isfile(line):
                os.remove(line)
    else:
        line_processor = functools.partial(process_line)
        with open(dest_csv, mode="w", newline="", encoding="utf-8") as csv_f:
            csv_writer = csv.writer(
                csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            csv_writer.writerow(header[0].split("\n")[0].split(","))
            for row in parallel_map(line_processor, all_lines):
                if row is None:
                    continue

                csv_writer.writerow(row)


def copy_and_remove_path_csvs(orig_csv, output_path):
    """
    Takes a csv file and copy it somewhere else with the path in the "wav" column removed for better security when sharing the dataset.

    Parameters
    ----------
    orig_csv: str
        Path to the original csv.
    output_path: str
        Path to the folder where to put the new csv.
    """

    output_path = os.path.join(output_path, orig_csv.split("/")[-1])

    logger.info("Generating CSV for parquet: " + str(output_path))
    line_processor = functools.partial(process_line_remove_wav_path)
    all_lines = open(orig_csv, "r", encoding="utf-8").readlines()
    with open(output_path, mode="w", newline="", encoding="utf-8") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        csv_writer.writerow(all_lines[0].split("\n")[0].split(","))
        for row in parallel_map(line_processor, all_lines[1:]):
            csv_writer.writerow(row)


def process_line_remove_wav_path(row: str) -> list:
    """
    Takes a csv line and returns it without path in wav and split.

    Parameters
    ----------
    row: str
        The csv line to be processed.

    Returns
    -------
    TheLoquaciousRow
        A dataclass containing the information about the line.
    """
    ID, duration, start, wav, spk_id, sex, text = row.split("\n")[0].split(",")

    wav = wav.split("/")[-1]

    return [ID, duration, start, wav, spk_id, sex, text]


def process_line_split_and_clean_wav(row: str) -> list:
    """
    Takes a csv line and returns it split.

    Parameters
    ----------
    row: str
        The csv line to be processed.

    Returns
    -------
    TheLoquaciousRow
        A dataclass containing the information about the line.
    """

    ID, duration, start, wav, spk_id, sex, text = row.split("\n")[0].split(",")
    start = float(start)
    duration = float(duration)

    if len(text.split(" ")) > MAX_WORD_COUNT:
        return None, None

    if start >= 0:
        start = int(start * 16000)
        frames = int(duration * 16000)
        old_wav_path = "/".join(wav.split("/")[:-1])
        filename = ID.split("/")[-1]
        ext = wav.split(".")[-1]
        new_path = os.path.join(old_wav_path, filename + "." + ext)
        if not os.path.isfile(new_path):
            audio_data = sf.read(wav, start=start, frames=frames)
            sf.write(new_path, audio_data[0], 16000)
        else:
            wav = None

        return [ID, duration, str(-1.0), new_path, spk_id, sex, text], wav
    else:
        return row.split(","), None


def process_line(row: str) -> list:
    """
    Takes a csv line and returns it split.

    Parameters
    ----------
    row: str
        The csv line to be processed.

    Returns
    -------
    TheLoquaciousRow
        A dataclass containing the information about the line.
    """

    ID, duration, start, wav, spk_id, sex, text = row.split("\n")[0].split(",")

    if len(text.split(" ")) > MAX_WORD_COUNT:
        return None

    return row.split(",")
