"""
Data preparation for ASR with VoxPopuli for the LargeScaleASR Set.
This is different from the standard VoxPopuli as the folder structure must be
destroyed and turned into a HuggingFace compatible one.

Download: https://github.com/facebookresearch/voxpopuli

Author
------
Titouan Parcollet 2024
"""

import csv
import functools
import os
from dataclasses import dataclass

from nemo_text_processing.text_normalization.normalize import Normalizer

from speechbrain.dataio.dataio import read_audio_info
from speechbrain.utils.logger import get_logger
from speechbrain.utils.parallel import parallel_map
from speechbrain.utils.text_normalisation import TextNormaliser

normaliser = Normalizer(input_case="cased", lang="en")

logger = get_logger(__name__)

SAMPLING_RATE = 16000
LOWER_DURATION_THRESHOLD_IN_S = 1.0
UPPER_DURATION_THRESHOLD_IN_S = 100
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


def prepare_voxpopuli(
    data_folder,
    huggingface_folder,
    train_tsv_file=None,
    dev_tsv_file=None,
    test_tsv_file=None,
    remove_if_longer_than=100,
):
    """
    Prepares the csv files for the Vox Populi dataset.
    Download: https://github.com/facebookresearch/voxpopuli

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original Vox Populi dataset is stored.
        This path should include the transcribed_data folder.
    huggingface_folder : str
        The directory of the HuggingFace LargeScaleASR Set.
    train_tsv_file : str, optional
        Path to the Train Vox Populi .tsv file (cs)
    dev_tsv_file : str, optional
        Path to the Dev Vox Populi .tsv file (cs)
    test_tsv_file : str, optional
        Path to the Test Vox Populi .tsv file (cs)
    remove_if_longer_than: int, optional
        Some audio files in VoxPopuli can be very long (200+ seconds). This option
        removes them from the train set.

    """

    # If not specified point toward standard location w.r.t VoxPopuli tree
    if train_tsv_file is None:
        train_tsv_file = data_folder + "/asr_train.tsv"
    else:
        train_tsv_file = train_tsv_file

    if dev_tsv_file is None:
        dev_tsv_file = data_folder + "/asr_dev.tsv"
    else:
        dev_tsv_file = dev_tsv_file

    if test_tsv_file is None:
        test_tsv_file = data_folder + "/asr_test.tsv"
    else:
        test_tsv_file = test_tsv_file

    # Setting the save folder
    manifest_folder = os.path.join(huggingface_folder, "manifests")
    wav_folder = os.path.join(
        huggingface_folder, os.path.join("data", "voxpopuli")
    )
    os.makedirs(wav_folder, exist_ok=True)

    # Setting output files
    save_csv_train = manifest_folder + "/vox_train.csv"
    save_csv_dev = manifest_folder + "/vox_dev.csv"
    save_csv_test = manifest_folder + "/vox_test.csv"

    # Additional checks to make sure the data folder contains Common Voice
    check_voxpopuli_folders(data_folder)

    if not os.path.isfile(save_csv_train):
        create_csv_and_copy_wav(
            train_tsv_file,
            save_csv_train,
            data_folder,
            wav_folder,
            remove_if_longer_than,
        )

    if not os.path.isfile(save_csv_dev):
        create_csv_and_copy_wav(
            dev_tsv_file,
            save_csv_dev,
            data_folder,
            wav_folder,
            remove_if_longer_than,
        )

    if not os.path.isfile(save_csv_test):
        create_csv_and_copy_wav(
            test_tsv_file,
            save_csv_test,
            data_folder,
            wav_folder,
            remove_if_longer_than,
        )


def skip(save_csv_train, save_csv_dev, save_csv_test):
    """
    Detects if the VoxPopuli data preparation has been already done.
    If the preparation has been done, we can skip it.

    Arguments
    ---------
    save_csv_train : str
        Path to train manifest file
    save_csv_dev : str
        Path to dev manifest file
    save_csv_test : str
        Path to test manifest file

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """

    # Checking folders and save options
    skip = False

    if (
        os.path.isfile(save_csv_train)
        and os.path.isfile(save_csv_dev)
        and os.path.isfile(save_csv_test)
    ):
        skip = True

    return skip


def process_line(line, data_folder, save_folder, text_normaliser):
    """
    Processes each line of the CSV (most likely happening with multiple threads)

    Arguments
    ---------
    line : str
        Line of the csv file.
    data_folder : str
        Path of the Vox Populi dataset.
    save_folder : str
        Where the wav files will be stored
    text_normaliser : speechbrain.utils.text_normalisation.TextNormaliser

    Returns
    -------
    TheLoquaciousRow
        A dataclass containing the information about the line.
    """
    year_path = os.path.join(line[0:4], line.split("\t")[0])
    ogg_path = os.path.join(data_folder, year_path) + ".ogg"
    file_name = line.split("\t")[0]
    spk_id = str(line.split("\t")[3])
    sex = str(line.split("\t")[5])
    snt_id = file_name
    start = -1

    # Reading the signal (to retrieve duration in seconds)
    if os.path.isfile(ogg_path):
        info = read_audio_info(ogg_path)
    else:
        msg = "\tError loading: %s" % (ogg_path)
        logger.info(msg)
        return None

    duration = info.num_frames / info.sample_rate

    if duration < LOWER_DURATION_THRESHOLD_IN_S:
        return None
    elif duration > UPPER_DURATION_THRESHOLD_IN_S:
        return None

    # Getting transcript
    words = line.split("\t")[2]

    # Unicode Normalization TO BE CHANGED WITH UNIFORM NORMALISATION
    words = normaliser.normalize(words)
    words = text_normaliser.english_specific_preprocess(words)

    if (
        len(words.split(" ")) < LOWER_WORDS_THRESHOLD
    ):  # ARBITRARY ! We may want to change this.
        return None

    wav_path = convert_to_wav_and_copy(ogg_path, save_folder)

    # Composition of the csv_line
    return TheLoquaciousRow(
        snt_id, duration, start, wav_path, spk_id, sex, words
    )


def create_csv_and_copy_wav(
    orig_tsv_file, csv_file, data_folder, save_folder, remove_if_longer_than
):
    """
    Creates the csv file given a list of ogg files.

    Arguments
    ---------
    orig_tsv_file : str
        Path to the Vox Populi tsv file (standard file).
    csv_file : str
        Path to the csv file where data will be dumped.
    data_folder : str
        Path of the Vox Populi dataset.
    save_folder : str
        Where the wav files will be stored
    remove_if_longer_than: int, optional
        Some audio files in VoxPopuli can be very long (200+ seconds). This option
        removes them from the train set. Information about the discarded data is given.
    """

    # Check if the given files exists
    if not os.path.isfile(orig_tsv_file):
        msg = "\t%s doesn't exist, verify your dataset!" % (orig_tsv_file)
        logger.info(msg)
        raise FileNotFoundError(msg)

    # We load and skip the header
    loaded_csv = open(orig_tsv_file, "r", encoding="utf-8").readlines()[1:]
    nb_samples = len(loaded_csv)

    msg = "Preparing CSV files for %s samples ..." % (str(nb_samples))
    logger.info(msg)

    # Adding some Prints
    msg = "Creating csv lists in %s ..." % (csv_file)
    logger.info(msg)

    # Process and write lines
    total_duration = 0.0
    skipped_duration = 0.0

    text_norm = TextNormaliser()
    line_processor = functools.partial(
        process_line,
        save_folder=save_folder,
        data_folder=data_folder,
        text_normaliser=text_norm,
    )

    # Stream into a .tmp file, and rename it to the real path at the end.
    csv_file_tmp = csv_file + ".tmp"

    with open(csv_file_tmp, mode="w", encoding="utf-8") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )

        csv_writer.writerow(
            ["ID", "duration", "start", "wav", "spk_id", "sex", "text"]
        )

        for row in parallel_map(line_processor, loaded_csv):
            if row is None:
                continue

            if row.duration < remove_if_longer_than:
                total_duration += row.duration
            else:
                skipped_duration += row.duration
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

    os.replace(csv_file_tmp, csv_file)

    # Final prints
    msg = "%s successfully created!" % (csv_file)
    logger.info(msg)
    msg = "Number of samples: %s " % (str(len(loaded_csv)))
    logger.info(msg)
    msg = "Total duration: %s Hours" % (str(round(total_duration / 3600, 2)))
    logger.info(msg)
    msg = "Total skipped duration (too long segments): %s Hours" % (
        str(round(skipped_duration / 3600, 2))
    )
    logger.info(msg)


def check_voxpopuli_folders(data_folder):
    """
    Check if the data folder actually contains the voxpopuli dataset.
    If not, raises an error.

    Arguments
    ---------
    data_folder : str
        Path to data folder to check

    Raises
    ------
    FileNotFoundError
        If data folder doesn't contain Common Voice dataset.
    """
    files_str = "/2020"
    # Checking clips
    if not os.path.exists(data_folder + files_str):
        err_msg = (
            "the folder %s does not exist (it is expected in "
            "the Common Voice dataset)" % (data_folder + files_str)
        )
        raise FileNotFoundError(err_msg)


def convert_to_wav_and_copy(source_audio_file, dest_audio_path):
    """Convert an audio file to a wav file.

    Parameters
    ----------
    source_audio_file : str
        The path to the opus file to be converted.
    dest_audio_path : str
        The path of the folder where to store the converted audio file.

    Returns
    -------
    str
        The path to the converted wav file.

    Raises
    ------
    subprocess.CalledProcessError
        If the conversion process fails.
    """
    audio_wav_path = source_audio_file.replace(".ogg", ".wav")
    audio_wav_path = os.path.join(
        dest_audio_path, audio_wav_path.split("/")[-1]
    )

    if not os.path.isfile(audio_wav_path):
        os.system(
            f"ffmpeg -y -i {source_audio_file} -ac 1 -ar {SAMPLING_RATE} {audio_wav_path} > /dev/null 2>&1"
        )

    return audio_wav_path
