"""
Download link: https://lium.univ-lemans.fr/ted-lium2/
Shucong Zhang 2023
"""

import os
import csv
import logging
import torchaudio
import multiprocessing
from tqdm import tqdm
from speechbrain.dataio.dataio import merge_csvs

logger = logging.getLogger(__name__)


def clip_and_make_csv(
    sph_file,
    stm_file,
    utt_save_folder,
    csv_save_folder,
    avoid_if_shorter_than=1,
):
    """
    This function splits the .sph Ted-talk recording into utterences based on the .stm annotation

    Arguments
    ---------
    sph_file : str
        Path to the sph file containing Ted-talk recording.
    stm_file : str
        Path to the stm file containing Ted-talk annotation.
    utt_save_folder: str
        The folder stores the clipped individual utterences.
    csv_save_folder: str
        The folder stores the generated csv files.
    avoid_if_shorter_than: int
        Any utterance shorter than this will be discarded.
    """
    # the annotation for JillSobuleMANHATTANINJANUARY_2006.sph is not useful
    if "JillSobuleMANHATTANINJANUARY_2006" in sph_file:
        logger.info("JillSobuleMANHATTANINJANUARY_2006.sph is skipped")
        return

    # load the annotation of the entire speech recording
    annotation_file = open(stm_file, "r")
    annotations = annotation_file.readlines()

    # load the original speech recording
    original_speech, sample_rate = torchaudio.load(sph_file)

    entry = []

    # process the annotation utterence by utterance
    for i, line in enumerate(annotations):
        line = line.strip("\n")
        line = line.split(" ")
        # parse the annotation
        talk_id = line[0]
        spk_id = line[2]

        # start and end point of the utterences in the recording
        start = float(line[3])
        end = float(line[4])
        duration = -start + end
        # we skip short utterences in case of CNN padding issues
        if duration < avoid_if_shorter_than:
            continue

        # transcriptions
        wrd_list = line[6:]
        if wrd_list[-1] == "":
            wrd_list = wrd_list[:-1]
        transcript = " ".join(wrd_list)
        if not transcript[-1].isalpha():
            transcript = transcript[:-1]
        transcript = transcript.replace(" 've", "'ve")
        transcript = transcript.replace(" 't", "'t")
        transcript = transcript.replace(" 'll", "'ll")
        transcript = transcript.replace(" 'd", "'d")
        transcript = transcript.replace(" 'm", "'m")
        transcript = transcript.replace(" 're", "'re")
        transcript = transcript.replace(" 's", "'s")
        # skip invalid transcriptions
        if len(wrd_list) <= 1 or transcript == "ignore_time_segment_in_scoring":
            continue

        # clip and save the current utterance
        clipped_save_path = f"{utt_save_folder}/{talk_id}-{str(i)}.wav"
        # we avoid duplicated clip and save
        if not os.path.exists(clipped_save_path):
            start = float(line[3]) * sample_rate
            end = float(line[4]) * sample_rate
            curr_utt = original_speech[:, int(start) : int(end)]
            torchaudio.save(clipped_save_path, curr_utt, sample_rate)
        # append to the csv entry list
        csv_line = [
            f"{talk_id}-{str(i)}",
            str(duration),
            clipped_save_path,
            spk_id,
            transcript,
        ]
        entry.append(csv_line)

    # create the csv file for the utterences in the talk recording.
    # the csv files for all the talks will be merged
    csv_flie = f"{csv_save_folder}/{talk_id}.csv"
    csv_output = [["ID", "duration", "wav", "spk_id", "wrd"]]
    csv_output = csv_output + entry
    with open(csv_flie, mode="w") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        for line in csv_output:
            csv_writer.writerow(line)


def prepare_tedlium2(
    data_folder, utt_save_folder, csv_save_folder, skip_prep=False
):
    """
    This function prepares the Tedlium2 dataset.
    Download link: https://lium.univ-lemans.fr/ted-lium2/

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original Tedlium2 dataset is stored.
    utt_save_folder : list
        Path where to save the clipped utterence-leve recordings.
    csv_save_folder: str
        Path where to save the generated .csv files.
    skip_prep: bool
        If True, data preparation is skipped.

    Example
    -------
    >>> data_folder = 'datasets/TEDLIUM_release2'
    >>> utt_save_folder = 'datasets/TEDLIUM_release2_processed'
    >>> csv_save_folder = 'TEDLIUM2'
    >>> prepare_tedlium2(data_folder, utt_save_folder, csv_save_folder)

    """
    if skip_prep:
        return

    splits = [
        "train",
        "dev",
        "test",
    ]
    for split in splits:
        utt_save_folder_split = f"{utt_save_folder}/{split}"
        csv_save_folder_split = f"{csv_save_folder}/{split}"
        os.makedirs(utt_save_folder_split, exist_ok=True)
        os.makedirs(csv_save_folder_split, exist_ok=True)
        new_filename = os.path.join(csv_save_folder_split) + ".csv"
        if os.path.exists(new_filename):
            continue
        logger.info("Preparing %s..." % new_filename)
        talk_sphs = os.listdir(f"{data_folder}/{split}/sph")
        processes = []

        for talk_sph in talk_sphs:
            talk_name = talk_sph[:-4]
            talk_sph_path = f"{data_folder}/{split}/sph/{talk_sph}"
            talk_stm_path = f"{data_folder}/{split}/stm/{talk_name}.stm"
            p = multiprocessing.Process(
                target=clip_and_make_csv,
                args=(
                    talk_sph_path,
                    talk_stm_path,
                    utt_save_folder_split,
                    csv_save_folder_split,
                ),
            )
            processes.append(p)

        for p in tqdm(processes):
            p.start()
        for p in processes:
            p.join()

        all_file_list = os.listdir(csv_save_folder_split)
        csv_list = []

        for f_name in all_file_list:
            if ".csv" in f_name:
                csv_list.append(f_name)

        merge_csvs(
            data_folder=csv_save_folder_split,
            csv_lst=csv_list,
            merged_csv=new_filename,
        )
        for talk_csv in csv_list:
            os.remove(f"{csv_save_folder_split}/{talk_csv}")
        msg = "\t%s successfully created!" % (new_filename)
        logger.info(msg)
