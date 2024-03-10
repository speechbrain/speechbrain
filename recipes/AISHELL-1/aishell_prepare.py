"""
Data preparation.

Download: https://www.openslr.org/33/

Authors
-------
 * Adel Moumen 2023
"""

import os
import shutil
import logging
import glob
import csv
from speechbrain.dataio.dataio import read_audio_info
from speechbrain.utils.parallel import parallel_map
import functools

logger = logging.getLogger(__name__)


def extract_and_cleanup_wav_files(
    tgz_list, wav_dir, splits, remove_compressed_wavs
):
    """This function extracts the wav files in the AISHELL-1 dataset.

    Arguments
    ---------
    tgz_list: list
        list of paths to the tar.gz files.
    wav_dir: str
        path to the wav directory.
    splits: list
        list of splits.
    remove_compressed_wavs: bool
        If True, remove compressed wav files after extraction.
    """
    if len(tgz_list) > 0:
        logger.info(f"Extracting wav files in {wav_dir}...")

        decompress_processor = functools.partial(
            shutil.unpack_archive, extract_dir=wav_dir,
        )

        for split in splits:
            os.makedirs(os.path.join(wav_dir, split), exist_ok=True)

        for _ in parallel_map(decompress_processor, tgz_list, chunk_size=64):
            pass

        if remove_compressed_wavs:
            for tgz in tgz_list:
                os.remove(tgz)


def process_line(wav, filename2transcript):
    """This function processes a line of the csv file.

    This function is being used in the context of multi-processing.

    Arguments
    ---------
    wav: str
        path to the wav file.
    filename2transcript: dict
        dictionary mapping filenames to transcripts.

    Returns
    -------
    list
        list containing the duration, the path to the wav file and the transcript.
    """
    filename = wav.split("/")[-1].split(".wav")[0]

    info = read_audio_info(wav)
    duration = info.num_frames / info.sample_rate

    transcript_ = filename2transcript[filename]

    return [str(duration), wav, transcript_]


def skip(splits, save_folder):
    """ Detect when the AiSHELL-1 data preparation can be skipped.

    Arguments
    ---------
    splits : list
        A list of the splits expected in the preparation.
    save_folder : str
        The location of the save directory

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

    return skip


def prepare_aishell(
    data_folder, save_folder, skip_prep=False, remove_compressed_wavs=True
):
    """This function prepares the AISHELL-1 dataset.

    Arguments
    ---------
    data_folder: str
        path to AISHELL-1 dataset.
    save_folder: str
        path where to store the manifest csv files.
    skip_prep: bool
        If True, skip data preparation.
    remove_compressed_wavs: bool
        If True, remove compressed wav files after extraction.
    """

    if skip_prep:
        return

    wav_dir = os.path.join(data_folder, "wav")
    tgz_list = glob.glob(wav_dir + "/*.tar.gz")

    splits = [
        "train",
        "dev",
        "test",
    ]

    if skip(splits, save_folder):
        return

    extract_and_cleanup_wav_files(
        tgz_list, wav_dir, splits, remove_compressed_wavs=remove_compressed_wavs
    )

    # Create filename-to-transcript dictionary
    filename2transcript = {}
    path_to_transcript = os.path.join(
        data_folder, "transcript/aishell_transcript_v0.8.txt"
    )

    with open(path_to_transcript, "r",) as f:
        lines = f.readlines()
        for line in lines:
            key = line.split()[0]
            value = " ".join(line.split()[1:])
            filename2transcript[key] = value

    line_processor = functools.partial(
        process_line, filename2transcript=filename2transcript,
    )

    for split in splits:

        final_csv = os.path.join(save_folder, split) + ".csv"
        tmp_csv = os.path.join(save_folder, split) + ".tmp"

        logger.info("Preparing %s..." % final_csv)

        all_wavs = glob.glob(
            os.path.join(data_folder, "wav") + "/" + split + "/*/*.wav"
        )
        # only keep the files that are in the transcript
        transcript_wavs = [
            wav
            for wav in all_wavs
            if wav.split("/")[-1].split(".wav")[0] in filename2transcript
        ]

        total_line = 0
        total_duration = 0
        id = 0
        with open(tmp_csv, mode="w", encoding="utf-8") as csv_f:
            csv_writer = csv.writer(
                csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            csv_writer.writerow(["ID", "duration", "wav", "transcript"])
            for row in parallel_map(
                line_processor, transcript_wavs, chunk_size=4092
            ):

                if row is None:
                    continue

                row = [str(id)] + row
                csv_writer.writerow(row)

                total_line += 1
                total_duration += float(row[1])
                id += 1

        msg = f"Number of samples: {total_line} "
        logger.info(msg)
        msg = "Total duration: %s Hours" % (
            str(round(total_duration / 3600, 2))
        )

        logger.info(msg)

        os.replace(tmp_csv, final_csv)

        msg = "\t%s successfully created!" % (final_csv)
        logger.info(msg)

        msg = f"Number of samples: {total_line} "
        logger.info(msg)
        msg = "Total duration: %s Hours" % (
            str(round(total_duration / 3600, 2))
        )
        logger.info(msg)
