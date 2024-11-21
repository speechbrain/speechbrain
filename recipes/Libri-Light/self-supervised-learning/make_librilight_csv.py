""" Create speebrain csv files for the Libri-Light dataset for self-supervised-learning

1. Download the Libri-Light dataset through the toolkit in the Libri-Light github repo
    "https://github.com/facebookresearch/libri-light/"

2. Use the data_preparation/cut_by_vad.py script of the Libri-Light repo to do the vad. For example,
    "python cut_by_vad.py --input_dir path_to_Libri-Light_small --output_dir Libri-Light_small_vad --target_len_sec 20"

3. Use the vad output_dir in step 2 as the input_dir in this step to generate the csv file. For example,
    "python make_librilight_csv.py --input_dir Libri-Light_small_vad --output_dir results --max_length 20 --n_processes 128" to generate the train.csv for the SSL pretraining

4. Now, you can use the generated train.csv in the output_dir in step 3 as the "train_csv" in any SpeechBrain SSL pre-training yaml file

Authors
 * Shucong Zhang 2024
 * Titouan Parcollet 2024
"""

import argparse
import csv
import multiprocessing
import os
import pathlib
from pathlib import Path

import torchaudio
import tqdm


def make_csv_for_each(
    subpath_1_csv_file_folder_max_length: tuple,
) -> None:
    """Prepare the csv files for each subfolder of the Libri-Light splits

    Arguments
    ---------
    subpath_1_csv_file_folder_max_length : tuple
        A (subfolder_path, csv_output_path, max_utt_length) tuple
    """
    subpath_1, csv_file_folder, max_length = (
        subpath_1_csv_file_folder_max_length
    )
    for i, flac_file in enumerate(subpath_1.glob("**/*.flac")):
        flac_file_name = flac_file.stem
        waveform, sample_rate = torchaudio.load(str(flac_file))
        num_frames = waveform.size(1)
        duration_seconds = num_frames / sample_rate
        if duration_seconds > max_length:
            continue
        audio_length_seconds = waveform.shape[1] / sample_rate
        csv_file = f"{csv_file_folder}/{flac_file.parent.stem}.csv"
        with open(csv_file, mode="a", newline="", encoding="utf-8") as csvfile:
            csv_writer = csv.writer(
                csvfile, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            csv_writer.writerow(
                [flac_file_name, audio_length_seconds, str(flac_file)]
            )


def processes_folder(
    data_path: str,
    csv_file_folder: str,
    max_length: float,
    n_processes: int,
) -> None:
    """Prepare the csv files for one of the Libri-Light splits

    Arguments
    ---------
    data_path : str
        Path to the Libri-Light split after vad
    csv_file_folder: str
        Path to the output folder of generated csv files
    max_length: float
        The max length (seconds) for each prepared audio clip
    n_processes: int
        Number of parallel processes
    """
    print("Processing each subfolder of this split")
    os.makedirs(csv_file_folder, exist_ok=True)
    os.makedirs(f"{csv_file_folder}/tmp", exist_ok=True)
    list_dir = pathlib.Path(data_path)
    tasks = []
    for x in list_dir.iterdir():
        tasks.append((x, f"{csv_file_folder}/tmp", max_length))
    with multiprocessing.Pool(processes=n_processes) as pool:
        for _ in tqdm.tqdm(
            pool.imap_unordered(make_csv_for_each, tasks), total=len(tasks)
        ):
            pass


def merge_csv_files(
    csv_file_folder: str,
) -> None:
    """Merges the csv files for each subfolder into one single file

    Arguments
    ---------
    csv_file_folder: str
        Path to the output folder of generated csv files
    """
    print("Merging the csvs of each subfolder into one csv")
    file_list = [str(x) for x in Path(f"{csv_file_folder}/tmp").glob("*.csv")]
    output_file = f"{csv_file_folder}/train.csv"
    fieldnames = ["ID", "duration", "wav"]

    with open(output_file, mode="a", newline="", encoding="utf-8") as outfile:
        csv_writer = csv.writer(
            outfile, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        csv_writer.writerow(fieldnames)
        for file_path in tqdm.tqdm(file_list):
            with open(file_path, mode="r", encoding="utf-8") as infile:
                reader = csv.reader(infile)

                # filter out bad rows
                for row in reader:
                    if len(row) == 3 and os.path.exists(row[-1]):
                        new_row = [row[-1], row[1], row[2]]
                        csv_writer.writerow(new_row)
                    else:
                        print(f"bad row {row}")

    import shutil

    shutil.rmtree(f"{csv_file_folder}/tmp")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Path to the Libri-Light split after vad",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Path to the output folder of generated csv files",
        required=True,
    )
    parser.add_argument(
        "--max_length",
        type=float,
        default=20.2,
        help="The max length for each prepared audio clip" "(default is 20.2)",
    )
    parser.add_argument(
        "--n_processes",
        type=int,
        default=32,
        help="Number of parallel processes",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    processes_folder(
        args.input_dir, args.output_dir, args.max_length, args.n_processes
    )
    merge_csv_files(args.output_dir)
