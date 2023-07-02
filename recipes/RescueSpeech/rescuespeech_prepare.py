"""
Data preparation script for RescueSpeech dataset. This
script prepares CSV files for ASR and Speech Enhancement.
In the generated CSV files the column-

`clean_noisy_mix` : alternates between the paths to the clean and
noisy speech recordings in the same order as they appear in the dataset.

By using this script, you can easily prepare the necessary CSV files
for training and evaluating ASR models on the RescueSpeech dataset.


Author
------
Sangeet Sagar 2023
(while some functions have been
adapted from the CommonVoice recipe)
"""

import os
import re
import csv
import glob
import logging
import torchaudio
import unicodedata
from tqdm import tqdm
from tqdm.contrib import tzip
from speechbrain.dataio.dataio import read_audio

logger = logging.getLogger(__name__)


def prepare_RescueSpeech(
    data_folder,
    save_folder,
    train_tsv_file=None,
    dev_tsv_file=None,
    test_tsv_file=None,
    accented_letters=False,
    skip_prep=False,
    sample_rate=16000,
    task="asr",
):
    """
    Prepares the csv files for RescueSpeech audio data.

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original dataset is stored.
    save_folder : str
        The directory where to store the csv files.
    train_tsv_file : str, optional
        Path to the Train RescueSpeech .tsv file (cs)
    dev_tsv_file : str, optional
        Path to the Dev RescueSpeech .tsv file (cs)
    test_tsv_file : str, optional
        Path to the Test RescueSpeech .tsv file (cs)
    accented_letters : bool, optional
        Defines if accented letters will be kept as individual letters or
        transformed to the closest non-accented letters.
    skip_prep: bool
        If True, skip data preparation.
    sample_rate: int, optional
        Sample rate of the wav files.
    task: str, optional
        States the task for which data prepration is being done.
        It can either be 'asr' or 'enhance'

    """

    if skip_prep:
        return

    # If not specified point toward standard location
    if train_tsv_file is None:
        train_tsv_file = data_folder + "/train.tsv"
    else:
        train_tsv_file = train_tsv_file

    if dev_tsv_file is None:
        dev_tsv_file = data_folder + "/dev.tsv"
    else:
        dev_tsv_file = dev_tsv_file

    if test_tsv_file is None:
        test_tsv_file = data_folder + "/test.tsv"
    else:
        test_tsv_file = test_tsv_file

    # Setting the save folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Setting ouput files
    save_csv_train = save_folder + "/train.csv"
    save_csv_dev = save_folder + "/dev.csv"
    save_csv_test = save_folder + "/test.csv"

    # If csv already exists, we skip the data preparation
    if skip(save_csv_train, save_csv_dev, save_csv_test):

        msg = "%s already exists, skipping data preparation!" % (save_csv_train)
        logger.info(msg)

        msg = "%s already exists, skipping data preparation!" % (save_csv_dev)
        logger.info(msg)

        msg = "%s already exists, skipping data preparation!" % (save_csv_test)
        logger.info(msg)

        return

    # Additional checks to make sure the data folder contains RescueSpeech data.
    check_RescueSpeech_data_folders(data_folder)

    # Creating csv files for {train, dev, test} data
    file_pairs = zip(
        [train_tsv_file, dev_tsv_file, test_tsv_file],
        [save_csv_train, save_csv_dev, save_csv_test],
    )
    if task == "asr":
        for tsv_file, save_csv in file_pairs:
            # Prepare CSV files
            create_asr_csv(tsv_file, save_csv, data_folder, accented_letters)
    elif task == "enhance":
        create_enhance_csv(data_folder, save_csv_train, "train", sample_rate)
        create_enhance_csv(data_folder, save_csv_dev, "valid", sample_rate)
        create_enhance_csv(data_folder, save_csv_test, "test", sample_rate)


def skip(save_csv_train, save_csv_dev, save_csv_test):
    """
    Detects if the RescueSpeech data preparation has been already done.

    If the preparation has been done, we can skip it.

    Returnsw
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


def create_asr_csv(
    orig_tsv_file, csv_file, data_folder, accented_letters=False
):
    """
    Creates the csv file given a list of wav files.

    Arguments
    ---------
    orig_tsv_file : str
        Path to the RescueSpeech tsv file (standard file).
    csv_file: str
        Path to csv file that will be saved.
    data_folder : str
        Path of the RescueSpeech domain dataset (clean, noisy, noise).
    accented_letters : bool, optional
        Defines if accented letters will be kept as individual letters or
        transformed to the closest non-accented letters.

    Returns
    -------
    None
    """

    # Check if the given files exists
    if not os.path.isfile(orig_tsv_file):
        msg = "\t%s doesn't exist, verify your dataset!" % (orig_tsv_file)
        logger.info(msg)
        raise FileNotFoundError(msg)

    # We load and skip the header
    loaded_csv = open(orig_tsv_file, "r").readlines()[1:]
    nb_samples = str(len(loaded_csv))

    msg = "Preparing CSV files for %s samples ..." % (str(nb_samples))
    logger.info(msg)

    # Adding some Prints
    msg = "Creating csv lists in %s ..." % (csv_file)
    logger.info(msg)

    csv_lines = [
        [
            "ID",
            "duration",
            "clean_wav",
            "noisy_wav",
            "clean_noisy_mix",
            "noise_wav",
            "noise_type",
            "snr_level",
            "spk_id",
            "wrd",
        ]
    ]

    # Noise types
    noise_types = [
        "Breathing-noise",
        "Emergency-vehicle-and-siren-noise",
        "Engine-noise",
        "Chopper-noise",
        "Static-radio-noise",
    ]

    idx = 0

    # Start processing lines
    total_duration = 0.0
    for line in tzip(loaded_csv):

        line = line[0]

        clean_data_fp = os.path.join(data_folder, "audio_files/clean")
        noisy_data_fp = os.path.join(data_folder, "audio_files/noisy")
        noise_data_fp = os.path.join(data_folder, "audio_files/noise")

        clean_fp = os.path.join(clean_data_fp, line.split("\t")[1])
        file_name = ".".join(clean_fp.split(".")).split("/")[-1]
        spk_id = line.split("\t")[0]
        snt_id = file_name

        # Retrieive the corresponding noisy file from noisy data file path
        clean_wav_bname = os.path.splitext(file_name)[0] + "_"
        noisy_file = [
            filename
            for filename in os.listdir(noisy_data_fp)
            if filename.startswith(clean_wav_bname)
        ]

        noisy_file = noisy_file[0]
        noisy_fp = os.path.join(noisy_data_fp, noisy_file)

        # alternate between clean and noisy wav
        idx += 1
        if idx % 2 == 0:
            clean_noisy_mix = clean_fp
        else:
            clean_noisy_mix = noisy_fp

        # Get corresponding noise file
        fields = os.path.splitext(noisy_file)[0].split("_")
        fileid = fields[fields.index("fileid") + 1]
        noise_file = "noise_fileid_" + str(fileid) + ".wav"
        # clean_file = "clean_fileid_" + str(fileid) + ".wav"
        noise_fp = os.path.join(noise_data_fp, noise_file)

        # Get noise type
        for item in noise_types:
            if item in noisy_file:
                noise_type = item
                break

        # Get SNR level
        for item in fields:
            if "snr" in item:
                snr_level = item.replace("snr", "")
                break

        # Setting torchaudio backend to sox-io (needed to read mp3 files)
        if torchaudio.get_audio_backend() != "sox_io":
            logger.warning("This recipe needs the sox-io backend of torchaudio")
            logger.warning("The torchaudio backend is changed to sox_io")
            torchaudio.set_audio_backend("sox_io")

        # Reading the signal (to retrieve duration in seconds)
        if os.path.isfile(clean_fp):
            info = torchaudio.info(clean_fp)
            info_noisy = torchaudio.info(noisy_fp)
        else:
            msg = "\tError loading: %s" % (str(len(file_name)))
            logger.info(msg)
            idx += 1
            continue

        duration = info.num_frames / info.sample_rate

        # Do some sanity check duration of clean, and noisy must be same
        duration_noisy = info_noisy.num_frames / info_noisy.sample_rate
        if round(duration, 3) != round(duration_noisy, 3):
            print("Length mismatch detected")

        total_duration += duration

        # Getting transcript
        words = line.split("\t")[2]

        # Unicode Normalization
        words = unicode_normalisation(words)

        # Perform data cleaning
        words = data_cleaning(words)

        # Remove accents if specified
        if not accented_letters:
            words = strip_accents(words)
            words = words.replace("'", " ")
            words = words.replace("’", " ")

        # Remove multiple spaces
        words = re.sub(" +", " ", words)

        # Remove spaces at the beginning and the end of the sentence
        words = words.lstrip().rstrip()

        # Getting chars
        chars = words.replace(" ", "_")
        chars = " ".join([char for char in chars][:])

        # Remove too short sentences (or empty):
        if len(words.split(" ")) < 3:
            idx += 1
            continue

        # Composition of the csv_line
        csv_line = [
            snt_id,
            str(duration),
            clean_fp,
            noisy_fp,
            clean_noisy_mix,
            noise_fp,
            noise_type,
            str(snr_level),
            spk_id,
            str(words),
        ]

        # Adding this line to the csv_lines list
        csv_lines.append(csv_line)

    # Writing the csv lines
    with open(csv_file, mode="w", encoding="utf-8") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )

        for line in csv_lines:
            csv_writer.writerow(line)

    # Final prints
    msg = "%s successfully created!" % (csv_file)
    logger.info(msg)
    msg = "Number of samples: %s " % (str(len(loaded_csv)))
    logger.info(msg)
    msg = "Total duration: %s Hours" % (str(round(total_duration / 3600, 2)))
    logger.info(msg)


def create_enhance_csv(data_folder, csv_file, split, fs=16000):
    """
    Create CSV files for train, valid and test set.

    Arguments:
    ----------
    data_folder :str
        Path to synthesized RescuSpeech data for task enhancement
    csv_file : str
        Save csv_file path for prepared data.
    split : str
        CSV prepration for train/valid/test
    fs : int
        Sampling rate. Defaults to 16000.
    """

    clean_fullpaths = []
    noise_fullpaths = []
    noisy_fullpaths = []
    language = []
    lang = "de"

    clean_f1_path = extract_files(
        os.path.join(data_folder, split), type="clean"
    )
    noise_f1_path = extract_files(
        os.path.join(data_folder, split), type="noise"
    )
    noisy_f1_path = extract_files(
        os.path.join(data_folder, split), type="noisy"
    )

    language.extend([lang] * len(clean_f1_path))
    clean_fullpaths.extend(clean_f1_path)
    noise_fullpaths.extend(noise_f1_path)
    noisy_fullpaths.extend(noisy_f1_path)

    # Write CSV for train and dev
    msg = "Writing " + split + " csv files"
    logger.info(msg)
    write2csv(
        language,
        clean_fullpaths,
        noise_fullpaths,
        noisy_fullpaths,
        csv_file,
        fs,
    )


def write2csv(
    language,
    clean_fullpaths,
    noise_fullpaths,
    noisy_fullpaths,
    csv_file,
    fs=16000,
):
    """
    Write data to CSV file in an appropriate format.

    Arguments
    ---------
    language : str
        Language of audio file
    clean_fullpaths : str
        Path to clean audio files of a split in the train/valid-set
    noise_fullpaths : str
        Path to noise audio files of a split in the train/valid-set
    noisy_fullpaths : str
        Path to noisy audio files of a split in the train/valid-set
    csv_file : str
        Save csv_file path for prepared data.
    fs : int
        Sampling rate. Defaults to 16000.
    """
    csv_columns = [
        "ID",
        "language",
        "duration",
        "clean_wav",
        "clean_wav_format",
        "clean_wav_opts",
        "noise_wav",
        "noise_wav_format",
        "noise_wav_opts",
        "noisy_wav",
        "noisy_wav_format",
        "noisy_wav_opts",
    ]

    # Retreive duration of just one signal. It is assumed
    # that all files have the same duration in MS-DNS dataset.

    total_duration = 0
    with open(csv_file, "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()

        for (i, (lang, clean_fp, noise_fp, noisy_fp),) in enumerate(
            tqdm(
                zip(language, clean_fullpaths, noise_fullpaths, noisy_fullpaths)
            )
        ):
            signal = read_audio(clean_fp)
            duration = signal.shape[0] / fs
            total_duration += duration

            row = {
                "ID": i,
                "language": lang,
                "duration": duration,
                "clean_wav": clean_fp,
                "clean_wav_format": "wav",
                "clean_wav_opts": None,
                "noise_wav": noise_fp,
                "noise_wav_format": "wav",
                "noise_wav_opts": None,
                "noisy_wav": noisy_fp,
                "noisy_wav_format": "wav",
                "noisy_wav_opts": None,
            }
            writer.writerow(row)

    # Final prints
    msg = "%s successfully created!" % (csv_file)
    logger.info(msg)
    msg = "Number of samples: %s " % (str(len(clean_fullpaths)))
    logger.info(msg)
    msg = "Total duration: %s Hours" % (str(round(total_duration / 3600, 2)))
    logger.info(msg)


def check_RescueSpeech_data_folders(data_folder):
    """
    Check if the data folder actually contains the RescueSpeech dataset.

    If not, raises an error.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If data folder doesn't contain RescueSpeech dataset.
    """

    # Checking clips
    if not os.path.exists(data_folder):

        err_msg = (
            "the folder %s does not exist (it is expected in "
            "the RescueSpeech dataset)" % (data_folder)
        )
        raise FileNotFoundError(err_msg)


def unicode_normalisation(text):
    """
    Normalizes the Unicode representation of a given text.

    Arguments
    ---------
    text : str
        The text to be normalized.

    Returns
    -------
    str
        The normalized text.
    """
    try:
        text = unicode(text, "utf-8")
    except NameError:  # unicode is a default on python 3
        pass
    return str(text)


def data_cleaning(words):
    """
    Perform data cleaning

    Arguments
    ---------
        word : str
            Text that needs to be cleaned

    Returns
    -------
    str
        Cleaned data

    """

    # this replacement helps preserve the case of ß
    # (and helps retain solitary occurrences of SS)
    # since python's upper() converts ß to SS.
    words = words.replace("ß", "0000ß0000")
    words = re.sub("[^’'A-Za-z0-9öÖäÄüÜß]+", " ", words).upper()
    words = words.replace("'", " ")
    words = words.replace("’", " ")
    words = words.replace(
        "0000SS0000", "ß"
    )  # replace 0000SS0000 back to ß as its initial presence in the corpus
    return words


def strip_accents(text):
    """
    Strips accents from a given text string.

    Arguments:
    ----------
    text : str
        The text from which accents are to be stripped.

    Returns
    -------
    str
        The text with accents stripped.
    """

    text = (
        unicodedata.normalize("NFD", text)
        .encode("ascii", "ignore")
        .decode("utf-8")
    )

    return str(text)


def extract_files(datapath, type=None):
    """
    Given a dir-path, it extracts full path of all wav files
    and sorts them.

    Arguments:
    ----------
    datapath :str
        Path to synthesized SAR data
    type : str
        Type of split: clean, noisy, noise.

    Returns
    -------
    list
        Sorted list of all wav files found in the given path.
    """
    if type:
        path = os.path.join(datapath, type)
        files = glob.glob(path + "/*.wav")

        # Sort all files based on the suffixed file_id (ascending order)
        files.sort(key=lambda f: int(f.split("fileid_")[-1].strip(".wav")))
    else:
        # Sort all files by name
        files = sorted(glob.glob(datapath + "/*.wav"))

    return files
