"""
This script prepares the data of the switchboard-1 release 2 corpus (LDC97S62).
Optionally, the Fisher corpus transcripts (LDC2004T19 and LDC2005T19) can be added to
the CSVs for Tokenizer and LM training.
The test set is based on the eval2000/Hub 5 data (LDC2002S09/LDC2002T43).

The datasets can be obtained from:
- Switchboard: https://catalog.ldc.upenn.edu/LDC97S62
- Fisher part 1: https://catalog.ldc.upenn.edu/LDC2004T19
- Fisher part 2: https://catalog.ldc.upenn.edu/LDC2005T19

The test data is available at:
- Speech data: https://catalog.ldc.upenn.edu/LDC2002S09
- Transcripts: https://catalog.ldc.upenn.edu/LDC2002T43

Author
------
Dominik Wagner 2022
"""

import re
import csv
import logging
import os
import sys
from collections import defaultdict

from speechbrain.dataio.dataio import merge_csvs
from speechbrain.utils.data_utils import download_file, get_all_files

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)
SAMPLERATE = 8000


def prepare_switchboard(
        data_folder,
        save_folder,
        splits=["train", "dev"],
        split_ratio=[90, 10],
        merge_lst=[],
        merge_name=None,
        skip_prep=False,
        add_fisher_corpus=False,
        max_utt=300,
):
    """
    Main function for Switchboard data preparation.

    Parameters
    ----------
    data_folder : str
        Path to the folder where the Switchboard (and Fisher) datasets are stored.
        Note that the Fisher data must be stored (or at least symlinked)
        to the same location.
    save_folder : str
        The directory to store the csv files.
    splits : list
        A list of data splits you want to obtain from the Switchboard dataset.
        This would be usually ["train", "dev"] since the "test" set is generated
        separately using the Hub5/eval2000 portion of the Switchboard corpus.
    split_ratio : list
        List containing the portions you want to allocate to
        each of your data splits e.g. [90, 10]
    merge_lst : list
        This allows you to merge some (or all) of the data splits you specified
        (e.g. ["train", "dev"]) into a single file.
    merge_name : str
        Name of the merged csv file.
    skip_prep : bool
        If True, data preparation is skipped.
    add_fisher_corpus : bool
        If True, a separate csv file called "train_lm.csv" will be created containing
        the Switchboard training data and the Fisher corpus transcripts.
        The "train_lm.csv" file can be used instead of the regular "train.csv" file
        for LM and Tokenizer training.
        Note that this requires the Fisher corpus (part 1 and part 2)
        to be downloaded in your data_folder location.
    max_utt : int
        Remove excess utterances once they appear  more than a specified
        number of times with the same transcription, in a data set.
        This is useful for removing utterances like "uh-huh" from training.

    Example
    -------
    >>> data_folder = "/nfs/data/ldc"
    >>> save_folder = "swbd_data"
    >>> splits = ["train", "dev"]
    >>> split_ratio = [90, 10]
    >>> prepare_switchboard(data_folder, save_folder, splits, split_ratio, add_fisher_corpus=True)
    """
    if skip_prep:
        logger.info("Data preparation skipped manually via hparams")
        return

    filenames = []
    for split in splits:
        filenames.append(os.path.join(save_folder, str(split + ".csv")))
    if add_fisher_corpus:
        filenames.append(os.path.join(save_folder, "train_lm.csv"))
    filenames.append(os.path.join(save_folder, "test.csv"))

    if skip(*filenames):
        logger.info("Preparation completed in previous run, skipping.")
        return

    train_data_folder = os.path.join(data_folder, "LDC97S62")
    for d in ["docs", "swb1_d1", "swb1_d2", "swb1_d3", "swb1_d4"]:
        swbd_folder = os.path.join(train_data_folder, d)
        if not os.path.exists(swbd_folder):
            err_msg = f"The folder {swbd_folder} does not exist (it is expected in the Switchboard dataset)"
            raise OSError(err_msg)

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    transcription_dir = os.path.join(save_folder, "swb_ms98_transcriptions")
    if not os.path.exists(transcription_dir):
        logger.info(f"Download transcriptions and store them in {save_folder}")

        download_source = "http://www.openslr.org/resources/5/switchboard_word_alignments.tar.gz"
        download_target = os.path.join(
            save_folder, "switchboard_word_alignments.tar.gz"
        )
        download_file(download_source, download_target, unpack=True)
    else:
        logger.info(
            f"Skipping download of transcriptions because {transcription_dir} already exists."
        )

    assert len(splits) == len(split_ratio)
    if sum(split_ratio) != 100 and sum(split_ratio) != 1:
        logger.error(
            "Implausible split ratios! Make sure they equal to 1 (or 100)."
        )
        sys.exit(1)
    if sum(split_ratio) == 100:
        split_ratio = [i / 100 for i in split_ratio]

    # collect all files containing transcriptions
    transcript_files = get_all_files(
        os.path.join(save_folder, "swb_ms98_transcriptions"),
        match_and=["trans.text"],
    )
    split_lens = [int(i * len(transcript_files)) for i in split_ratio]

    name2disk = make_name_to_disk_dict(
        os.path.join(train_data_folder, "docs/swb1_all.dvd.tbl")
    )
    logger.info(
        f"Made name2disk mapping dict containing {len(name2disk)} conversations."
    )

    start = 0
    stop = 0
    # We save all lines from the swbd train split, in case we want to combine it
    # with the Fisher corpus for LM and Tokenizer training later
    swbd_train_lines = []
    for i, split in enumerate(splits):
        stop += split_lens[i]
        transcript_files_split = transcript_files[start:stop]
        logger.info(
            f"Preparing data for {split} split. "
            f"Split will contain {len(transcript_files_split)} "
            f"conversations separated by channel."
        )

        start += split_lens[i]

        # Keep track of the number of times each utterance appears
        utt2count = defaultdict(int)

        csv_lines = [
            [
                "ID",
                "length",
                "start",
                "stop",
                "channel",
                "wav",
                "words",
                "spk_id",
            ]
        ]
        # Open each transcription file and extract information
        for filename in transcript_files_split:
            with open(filename) as file:
                for line in file:
                    str_split = line.split()
                    id = str_split[0].strip()
                    channel = id.split("-")[0][-1]
                    wav_name = id.split("-")[0][:6] + ".sph"
                    spk_id = wav_name.replace(".sph", channel)
                    wav_name = wav_name.replace(wav_name[0:2], "sw0")
                    disk = name2disk[wav_name]

                    wav_path = os.path.join(
                        train_data_folder, disk, "data", wav_name
                    )
                    # We want the segment start and end times in samples,
                    # so we can slice the segment from the tensor
                    seg_start = int(float(str_split[1].strip()) * SAMPLERATE)
                    seg_end = int(float(str_split[2].strip()) * SAMPLERATE)
                    audio_duration = (seg_end - seg_start) / SAMPLERATE

                    transcription = " ".join(str_split[3:])
                    cleaned_transcription = filter_text(
                        transcription, dataset="train"
                    )

                    # Skip empty transcriptions
                    if len(cleaned_transcription) > 0:

                        csv_lines.append(
                            [
                                id,
                                audio_duration,
                                seg_start,
                                seg_end,
                                channel,
                                wav_path,
                                cleaned_transcription,
                                spk_id,
                            ]
                        )

                        # We store the lines from the first split
                        # (assuming this is the training data) in a separate list
                        # so we can easily merge it with the Fisher data later
                        if add_fisher_corpus and i == 0:
                            swbd_train_lines.append([id, cleaned_transcription])

        # Setting path for the csv file
        csv_file = os.path.join(save_folder, str(split + ".csv"))
        logger.info(f"Creating csv file {csv_file}")

        with open(csv_file, mode="w") as csv_f:
            csv_writer = csv.writer(
                csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )

            for line in csv_lines:
                current_utt = line[6]
                # Avoid that the same utterance becomes part of the dataset too often
                if utt2count[current_utt] < max_utt:
                    csv_writer.writerow(line)

                utt2count[current_utt] += 1

    # Merging csv file if needed
    if merge_lst and merge_name is not None:
        merge_files = [split_swbd + ".csv" for split_swbd in merge_lst]
        merge_csvs(
            data_folder=save_folder,
            csv_lst=merge_files,
            merged_csv=merge_name,
        )

    eval2000_data_prep(data_folder, save_folder)

    if add_fisher_corpus:
        # Keep track of the number of times each utterance appears
        utt2count = defaultdict(int)

        fisher_lines = fisher_data_prep(data_folder, save_folder)
        # fisher_lines already contains a header, so we don't need to add one here
        combined_lines = fisher_lines + swbd_train_lines

        csv_file = os.path.join(save_folder, "train_lm.csv")
        with open(csv_file, mode="w") as csv_f:
            csv_writer = csv.writer(
                csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )

            for line in combined_lines:
                current_utt = line[1]
                # Avoid that the same utterance becomes part of the dataset too often
                if utt2count[current_utt] < max_utt:
                    csv_writer.writerow(line)

                utt2count[current_utt] += 1

    logger.info("Switchboard data preparation finished.")


def skip(*filenames):
    """
    Detects if the Switchboard data preparation has already been done.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, preparation must be done.
    """
    for filename in filenames:
        if not os.path.isfile(filename):
            return False
    return True


def filter_text(transcription: str, dataset="train"):
    """
    This function takes a string representing a sentence in one
    of the datasets and cleans it using various regular expressions.
    The types of regular expressions applied depend on the dataset.

    Switchboard Training Data:
    When applied to the training data, the function removes tags like:
    [laughter], [noise], okay_1,  m[ost]-, because_1,
    {alrighty}, <b_aside> <e_aside>

    Eval2000 Test Data:
    Applied on the eval2000 data, it removes things like:
    <B_ASIDE> JUST (JU-) ALL THINGS (WE-) (DO-) (%HESITATION)

    Fisher Data:
    When applied to the Fisher transcripts, the function handles things like:
    (( how )) cou-  [laughter] f._d._a.
    Here we keep things in brackets like "(( how ))" because
    it might be beneficial for LM training when there is more
    information on the sentence. We also keep incomplete words such as "cou-".

    Parameters
    ----------
    transcription : str
        A transcribed sentence
    dataset : str
        Either "train", "eval2000", or "fisher" depending on the type
        of data you want to clean.
        (see the description above)

    Returns
    -------
    A string containing the cleaned sentence

    """
    dataset = dataset.strip().lower()

    if dataset == "train":
        transcription = transcription.upper().strip()
        transcription = re.sub(r"\[.*?\]", "", transcription)
        transcription = re.sub(r"\{.*?\}", "", transcription)
        transcription = re.sub(r"\(.*?\)", "", transcription)
        transcription = re.sub(r"<.*?>", "", transcription)
        transcription = re.sub(r"_[0-9]+", "", transcription)
        transcription = transcription.replace("-", "")
    elif dataset in ["eval2000", "hub5", "test"]:
        transcription = re.sub(r"<.*?>", "", transcription)
        transcription = re.sub(r"\(%HESITATION\)", "", transcription)
        # Remove everything within (), except when the - character occurs
        # We do this, so we can still extract partially uttered words
        transcription = re.sub(r"[^-]\(.*?\)", "", transcription)
        # Only remove ( and ) around partially uttered word
        transcription = re.sub(r"\)", "", transcription)
        transcription = re.sub(r"\(", "", transcription)
        transcription = re.sub(r"-", "", transcription)
        transcription = transcription.upper()
    elif dataset == "fisher":
        transcription = transcription.upper().strip()
        transcription = re.sub(r"\)", "", transcription)
        transcription = re.sub(r"\(", "", transcription)
        transcription = re.sub(r"\[.*?\]", "", transcription)
        transcription = re.sub(r"\{.*?\}", "", transcription)
        transcription = re.sub(r"<.*?>", "", transcription)
        transcription = re.sub(r"\._", "", transcription)
        transcription = re.sub(r"-", "", transcription)
        transcription = re.sub(r"\.", "", transcription)
    else:
        raise NameError(f"Invalid dataset descriptor '{dataset}' supplied.")

    # Remove redundant whitespaces
    transcription = re.sub(r"\s\s+", " ", transcription)
    return transcription.strip()


def make_name_to_disk_dict(mapping_table: str):
    """
    The Switchboard data is spread across 4 DVDs
    represented by directories ("swb1_d1", "swb1_d2" and so on).
    This function creates a lookup dictionary to map a given filename to the
    disk it was stored on.
    This information is useful to assemble the absolute path to the sph audio
    files.

    Parameters
    ----------
    mapping_table : str
        String representing the path to the mapping table file "swb1_all.dvd.tbl"
        provided along with the rest of the Switchboard data.

    Returns
    -------
    name2disk : dict
        A dictionary that maps from sph filename (key) to disk-id (value)
    """
    name2disk = {}
    with open(mapping_table) as mt:
        for line in mt:
            split = line.split()
            name = split[1].strip()
            name2disk[name] = split[0].strip()
    return name2disk


def eval2000_data_prep(data_folder: str, save_folder: str):
    """
    This function prepares the eval2000/Hub5 English
    data (LDC2002S09 and LDC2002T43).
    The data serves as the test set and is separated into
    the full dataset (test.csv), the Switchboard portion
    of the dataset (test_swbd.csv) and the Callhome portion
    of the dataset (test_callhome.csv).

    Parameters
    ----------
    data_folder : str
        Path to the folder where the eval2000/Hub5 English data is located.
    save_folder : str
        The directory to store the csv files at.
    """

    logger.info(
        "Begin preparing the eval2000 Hub5 English test set and transcripts (LDC2002S09 and LDC2002T43)"
    )

    audio_folder = os.path.join(data_folder, "LDC2002S09/hub5e_00/english")
    transcription_file = os.path.join(
        data_folder,
        "LDC2002T43/2000_hub5_eng_eval_tr/reference/hub5e00.english.000405.stm",
    )

    for d in [audio_folder, transcription_file]:
        if not os.path.exists(d):
            err_msg = f"The folder {d} does not exist (it is expected to prepare the eval2000/hub5 test set)"
            raise OSError(err_msg)

    csv_lines_callhome = [
        ["ID", "length", "start", "stop", "channel", "wav", "words", "spk_id"]
    ]
    csv_lines_swbd = [
        ["ID", "length", "start", "stop", "channel", "wav", "words", "spk_id"]
    ]

    with open(transcription_file) as file:
        utt_count = 0
        for line in file:
            # Skip header
            if line.startswith(";;"):
                continue

            str_split = line.split()
            # Sometimes the end time of a segment is shifted to the right
            # So we remove all empty strings from the split
            str_split = [i for i in str_split if len(i) > 0]

            # Make ID unique
            id = str_split[2].strip() + "_" + str(utt_count)
            channel = str_split[1].strip()

            wav_name = str_split[0].strip() + ".sph"
            wav_path = os.path.join(audio_folder, wav_name)

            spk_id = str_split[2].strip()

            # The label "en" stands for "Callhome conversations"
            # The label "sw" stands for "Switchboard conversations"
            is_swbd = str_split[0].strip().startswith("sw_")

            # We want the segment start and end times in samples,
            # so we can slice the segment from the tensor
            try:
                seg_start = int(float(str_split[3].strip()) * SAMPLERATE)
                seg_end = int(float(str_split[4].strip()) * SAMPLERATE)
            except ValueError:
                logger.error(
                    f"Unable to determine start and end time of segment. "
                    f"This should not happen! Split in "
                    f"question was: {str_split}"
                )

            audio_duration = (seg_end - seg_start) / SAMPLERATE

            transcription = " ".join(str_split[6:])
            cleaned_transcription = filter_text(
                transcription, dataset="eval2000"
            )

            # Skip empty transcriptions
            if len(cleaned_transcription) > 0:
                utt_line = [
                    id,
                    audio_duration,
                    seg_start,
                    seg_end,
                    channel,
                    wav_path,
                    cleaned_transcription,
                    spk_id,
                ]
                if is_swbd:
                    csv_lines_swbd.append(utt_line)
                else:
                    csv_lines_callhome.append(utt_line)
            utt_count += 1

    merge_files = []
    for name, lines in [
        ("swbd", csv_lines_swbd),
        ("callhome", csv_lines_callhome),
    ]:
        filename = f"test_{name}.csv"
        csv_file = os.path.join(save_folder, filename)
        logger.info(f"Creating csv file {csv_file}")

        with open(csv_file, mode="w") as csv_f:
            csv_writer = csv.writer(
                csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )

            for line in lines:
                csv_writer.writerow(line)

        merge_files.append(filename)
    merge_csvs(
        data_folder=save_folder, csv_lst=merge_files, merged_csv="test.csv"
    )


def fisher_data_prep(data_folder: str, save_folder: str):
    """
    Prepare Fisher data for Tokenizer and LM Training.
    The Fisher transcripts are located at
    LDC2004T19/fe_03_p1_tran/ and LDC2005T19/fe_03_p2_tran/.

    Parameters
    ----------
    data_folder : str
        Path to the folder where the Fisher data is located.
    """

    logger.info(
        "Begin preparing the Fisher corpus transcripts (LDC2002S09 and LDC2002T43)"
    )

    fisher_dirs = [
        "LDC2004T19/fe_03_p1_tran/data/trans",
        "LDC2005T19/fe_03_p2_tran/data/trans",
    ]

    for fisher_dir in fisher_dirs:
        joined_path = os.path.join(data_folder, fisher_dir)
        if not os.path.exists(joined_path):
            err_msg = f"The folder {joined_path} does not exist (it is expected to prepare the Fisher corpus)"
            raise OSError(err_msg)

    csv_lines = [["ID", "words"]]
    num_files_processed = 0
    num_dirs_processed = 0
    utt_count = 0

    for fisher_dir in fisher_dirs:
        joined_path = os.path.join(data_folder, fisher_dir)
        transcript_files = get_all_files(joined_path, match_and=[".txt"])

        for transcript_files in transcript_files:
            with open(transcript_files) as file:
                for line in file:
                    # skip header and empty lines
                    if line.startswith("#") or len(line.strip()) < 1:
                        continue

                    # Create unique id
                    id = "fisher-" + str(utt_count)
                    transcription = line.split()[3:]
                    transcription = " ".join(transcription)
                    transcription_clean = filter_text(
                        transcription, dataset="fisher"
                    )

                    # Skip empty transcriptions
                    if len(transcription_clean) > 0:
                        csv_lines.append([id, transcription_clean])
                        utt_count += 1
            # Just for accounting
            num_files_processed += 1
        num_dirs_processed += 1

    logger.info(
        f"Fisher corpus: Processed {num_files_processed} files in "
        f"{num_dirs_processed} directories."
    )

    csv_file = os.path.join(save_folder, "fisher.csv")
    logger.info(f"Creating csv file {csv_file}")

    with open(csv_file, mode="w") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )

        for line in csv_lines:
            csv_writer.writerow(line)
    return csv_lines


if __name__ == "__main__":
    data_folder = "/nfs/data/ldc"
    save_folder = "swbd_data"
    prepare_switchboard(
        data_folder,
        save_folder,
        splits=["train", "dev"],
        split_ratio=[90, 10],
        merge_lst=[],
        merge_name=None,
        skip_prep=False,
        add_fisher_corpus=True,
    )
