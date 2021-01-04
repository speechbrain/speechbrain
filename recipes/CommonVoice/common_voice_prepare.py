"""
Data preparation.

Download: https://voice.mozilla.org/en/datasets

Author
------
Titouan Parcollet
"""

import os
import csv
import re
import logging
import torch
import torchaudio
import unicodedata
from tqdm.contrib import tzip
from speechbrain.data_io.data_io import read_audio

logger = logging.getLogger(__name__)
SAMPLERATE = 16000


def prepare_common_voice(
    data_folder,
    save_folder,
    path_to_wav,
    train_tsv_file=None,
    dev_tsv_file=None,
    test_tsv_file=None,
    accented_letters=False,
    duration_threshold=10,
    language="en",
):
    """
    Prepares the csv files for the Mozilla Common Voice dataset.
    Download: https://voice.mozilla.org/en/datasets

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original Common Voice dataset is stored.
        This path should include the lang: /datasets/CommonVoice/en/
    save_folder : str
        The directory where to store the csv files.
    path_to_wav : str
        Path to store the converted wav files.
        By default Common Voice uses .mp3 audio files. SoundFile cannot process
        .mp3 files. If not already done, and if path_to_wav doesn't exist, then
        audiofiles are converted to .wav 16Khz.
        Please note that more than 100GB may be required for the FULL
        english dataset. Finally, converting the entire dataset may be long.
        An alternative is to convert the files before using this preparer.
        Indeed, if path_to_wav exists, the data_preparer will just assume that
        wav files already exist.
    train_tsv_file : str, optional
        Path to the Train Common Voice .tsv file (cs)
    dev_tsv_file : str, optional
        Path to the Dev Common Voice .tsv file (cs)
    test_tsv_file : str, optional
        Path to the Test Common Voice .tsv file (cs)
    accented_letters : bool, optional
        Defines if accented letters will be kept as individual letters or
        transformed to the closest non-accented letters.
    duration_threshold : int, optional
        Max duration (in seconds) to use as a threshold to filter sentences.
        The CommonVoice dataset contains very long utterance mostly containing
        noise due to open microphones.
    language: str
        Specify the language for text normalization.

    Example
    -------
    >>> from recipes.CommonVoice.common_voice_prepare import prepare_common_voice
    >>> data_folder = '/datasets/CommonVoice/en'
    >>> save_folder = 'exp/CommonVoice_exp'
    >>> path_to_wav = '/datasets/CommonVoice/en/clips_wav_100h'
    >>> train_tsv_file = '/datasets/CommonVoice/en/dev.tsv'
    >>> dev_tsv_file = '/datasets/CommonVoice/en/dev.tsv'
    >>> test_tsv_file = '/datasets/CommonVoice/en/test.tsv'
    >>> accented_letters = False
    >>> duration_threshold = 10
    >>> prepare_common_voice( \
                 data_folder, \
                 save_folder, \
                 path_to_wav, \
                 train_tsv_file, \
                 dev_tsv_file, \
                 test_tsv_file, \
                 accented_letters, \
                 duration_threshold, \
                 language="en" \
                 )
    """

    # If not specified point toward standard location w.r.t CommonVoice tree
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

    # Test if files have already been converted to wav by checking if
    # path_to_wav exists.
    if os.path.exists(path_to_wav):
        msg = "%s already exists, we will check for missing .wav files." % (
            path_to_wav
        )
        print(msg)
    else:
        msg = "%s doesn't exist, we will convert .mp3 files." % (path_to_wav)
        print(msg)
        os.makedirs(path_to_wav)

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
        print(msg)

        msg = "%s already exists, skipping data preparation!" % (save_csv_dev)
        print(msg)

        msg = "%s already exists, skipping data preparation!" % (save_csv_test)
        print(msg)

        return

    # Additional checks to make sure the data folder contains Common Voice
    check_commonvoice_folders(data_folder)

    # Creating csv file for training data
    if train_tsv_file is not None:

        # Convert .mp3 files if required
        msg = "Converting train audio files to .wav if needed ..."
        print(msg)
        convert_mp3_wav(
            data_folder, train_tsv_file, path_to_wav, SAMPLERATE,
        )

        create_csv(
            train_tsv_file,
            path_to_wav,
            save_csv_train,
            data_folder,
            accented_letters,
            duration_threshold,
            language,
        )

    # Creating csv file for dev data
    if dev_tsv_file is not None:

        # Convert .mp3 files if required
        msg = "Converting validation audio files to .wav if needed ..."
        print(msg)
        convert_mp3_wav(
            data_folder, dev_tsv_file, path_to_wav, SAMPLERATE,
        )

        create_csv(
            dev_tsv_file,
            path_to_wav,
            save_csv_dev,
            data_folder,
            accented_letters,
            duration_threshold,
            language,
        )

    # Creating csv file for test data
    if test_tsv_file is not None:

        # Convert .mp3 files if required
        msg = "Converting test audio files to .wav if needed ..."
        print(msg)
        convert_mp3_wav(
            data_folder, test_tsv_file, path_to_wav, SAMPLERATE,
        )

        create_csv(
            test_tsv_file,
            path_to_wav,
            save_csv_test,
            data_folder,
            accented_letters,
            duration_threshold,
            language,
        )


def convert_mp3_wav(data_folder, tsv_file, path_to_wav, samplerate):
    """
    Convert the list of audio files given in the tsv_file that follows
    the standard Common Voice format (for instance see train.tsv).
    From .mp3 to .wav with the given samplerate.

    Returns
    -------
    None
    """

    # Check if the given tsv exists
    if not os.path.isfile(tsv_file):
        msg = "%s doesn't exist, verify your dataset!" % (tsv_file)
        print(msg)
        raise FileNotFoundError(msg)

    # We load and skip the header
    loaded_csv = open(tsv_file, "r").readlines()[1:]

    nb_samples = str(len(loaded_csv))
    msg = "%s files to convert ..." % (str(nb_samples))
    print(msg)

    for line in tzip(loaded_csv):

        # Get the first element of tzip (correspong to the line)
        line = line[0]

        # Path is at indice 1 in Common Voice tsv files. And .mp3 files
        # are located in datasets/lang/clips/
        mp3_path = data_folder + "/clips/" + line.split("\t")[1]
        file_name = mp3_path.split(".")[-2].split("/")[-1]
        new_wav_path = path_to_wav + "/" + file_name + ".wav"

        # If corresponding wav file already exists, continue to the next one
        if os.path.isfile(new_wav_path):
            continue

        # Convert to wav with torchaudio
        if os.path.isfile(mp3_path):
            try:
                sig, orig_rate = torchaudio.load(mp3_path)

                # !! STEREO DETECTED, ME MUST GO TO MONO !!
                if sig.shape[0] == 2:
                    sig = torch.mean(sig, dim=0).unsqueeze(0)
            except RuntimeError:
                msg = "Error loading: %s" % (str(len(file_name)))
                print(msg)
                continue
            res = torchaudio.transforms.Resample(orig_rate, samplerate)
            sig = res(sig)
            torchaudio.save(new_wav_path, sig, samplerate)
        else:
            msg = "%s doesn't exist! Skipping it ..." % (str(len(file_name)))
            print(msg)


def skip(save_csv_train, save_csv_dev, save_csv_test):
    """
    Detects if the Common Voice data preparation has been already done.

    If the preparation has been done, we can skip it.

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


def create_csv(
    orig_tsv_file,
    path_to_wav,
    csv_file,
    data_folder,
    accented_letters=False,
    duration_threshold=10,
    language="en",
):
    """
    Creates the csv file given a list of wav files.

    Arguments
    ---------
    orig_tsv_file : str
        Path to the Common Voice tsv file (standard file).
    path_to_wav : str
        Path of the audio wav files.
    data_folder : str
        Path of the CommonVoice dataset.
    accented_letters : bool, optional
        Defines if accented letters will be kept as individual letters or
        transformed to the closest non-accented letters.
    duration_threshold : int
        Max duration (in seconds) to use as a threshold to filter sentences.
        The CommonVoice dataset contains very long utterance mostly containing
        noise due to open microphones.

    Returns
    -------
    None
    """

    # Check if the given files exists
    if not os.path.isfile(orig_tsv_file):
        msg = "\t%s doesn't exist, verify your dataset!" % (orig_tsv_file)
        print(msg)
        raise FileNotFoundError(msg)

    if not os.path.exists(path_to_wav):
        msg = "\t%s doesn't exists, we need wav files !" % (path_to_wav)
        print(msg)
        raise FileNotFoundError(msg)

    # We load and skip the header
    loaded_csv = open(orig_tsv_file, "r").readlines()[1:]
    nb_samples = str(len(loaded_csv))

    msg = "Preparing CSV files for %s samples ..." % (str(nb_samples))
    print(msg)

    # Adding some Prints
    msg = "Creating csv lists in %s ..." % (csv_file)
    print(msg)

    csv_lines = [
        [
            "ID",
            "duration",
            "wav",
            "wav_format",
            "wav_opts",
            "spk_id",
            "spk_id_format",
            "spk_id_opts",
            "wrd",
            "wrd_format",
            "wrd_opts",
            "char",
            "char_format",
            "char_opts",
        ]
    ]

    # Start processing lines
    total_duration = 0.0
    for line in tzip(loaded_csv):

        line = line[0]

        # Path is at indice 1 in Common Voice tsv files. And .mp3 files
        # are located in datasets/lang/clips/
        mp3_path = data_folder + "/clips/" + line.split("\t")[1]
        file_name = mp3_path.split(".")[-2].split("/")[-1]
        wav_path = path_to_wav + "/" + file_name + ".wav"
        spk_id = line.split("\t")[0]
        snt_id = file_name

        # Reading the signal (to retrieve duration in seconds)
        if os.path.isfile(wav_path):
            signal = read_audio(wav_path)
        else:
            msg = "\tError loading: %s" % (str(len(file_name)))
            print(msg)
            continue

        duration = signal.shape[0] / SAMPLERATE
        total_duration += duration

        # Filter long and noisy samples
        if duration > duration_threshold:
            continue

        # Getting transcript
        words = line.split("\t")[2]

        # !! Language specific cleaning !!
        # Important: feel free to specify the text normalization
        # corresponding to your alphabet.

        if language in ["en", "fr", "it", "rw"]:
            words = re.sub("[^'A-Za-z0-9À-ÖØ-öø-ÿЀ-ӿ]+", " ", words).upper()
        elif language == "ar":
            HAMZA = "\u0621"
            ALEF_MADDA = "\u0622"
            ALEF_HAMZA_ABOVE = "\u0623"
            letters = (
                "ابتةثجحخدذرزسشصضطظعغفقكلمنهويءآأؤإئ"
                + HAMZA
                + ALEF_MADDA
                + ALEF_HAMZA_ABOVE
            )
            words = re.sub("[^" + letters + "]+", " ", words).upper()

        # Remove accents if specified
        if not accented_letters:
            nfkd_form = unicodedata.normalize("NFKD", words)
            words = "".join(
                [c for c in nfkd_form if not unicodedata.combining(c)]
            )
            words = words.replace("'", " ")

        # Remove multiple spaces
        words = re.sub(" +", " ", words)

        # Remove spaces at the beginning and the end of the sentence
        words = words.lstrip().rstrip()

        # Getting chars
        chars = words.replace(" ", "_")
        chars = " ".join([char for char in chars][:])

        # Remove too short sentences (or empty):
        if len(words) < 3:
            continue

        # Composition of the csv_line
        csv_line = [
            snt_id,
            str(duration),
            wav_path,
            "wav",
            "",
            spk_id,
            "string",
            "",
            str(words),
            "string",
            "",
            str(chars),
            "string",
            "",
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
    msg = "%s sucessfully created!" % (csv_file)
    print(msg)
    msg = "Number of samples: %s " % (str(len(loaded_csv)))
    print(msg)
    msg = "Total duration: %s Hours" % (str(round(total_duration / 3600, 2)))
    print(msg)


def check_commonvoice_folders(data_folder):
    """
    Check if the data folder actually contains the Common Voice dataset.

    If not, raises an error.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If data folder doesn't contain Common Voice dataset.
    """

    files_str = "/clips"

    # Checking clips
    if not os.path.exists(data_folder + files_str):

        err_msg = (
            "the folder %s does not exist (it is expected in "
            "the Common Voice dataset)" % (data_folder + files_str)
        )
        raise FileNotFoundError(err_msg)
