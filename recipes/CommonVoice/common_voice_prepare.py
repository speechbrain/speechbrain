"""
Data preparation.

Download: https://voice.mozilla.org/en/datasets

Author
------
Titouan Parcollet
"""

import os
import csv
import torch
import re
import logging
import torchaudio
import unicodedata
from tqdm.contrib import tzip
from speechbrain.data_io.data_io import read_wav_soundfile

logger = logging.getLogger(__name__)


class CommonVoicePreparer(torch.nn.Module):
    """
    repares the csv files for the Mozilla Common Voice dataset.

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

    Example
    -------
    This example requires the Common Voice dataset.
    ```
    >>> from recipes.CommonVoice.common_voice_prepare import \
    >>> CommonVoicePreparer
    >>>
    >>> local_folder='/datasets/CommonVoice/en'
    >>> save_folder='exp/CommonVoice_exp'
    >>> path_to_wav='/datasets/CommonVoice/en/clips_wav'
    >>> train_tsv_file='/datasets/CommonVoice/en/train.tsv'
    >>> dev_tsv_file='/datasets/CommonVoice/en/dev.tsv'
    >>> test_tsv_file='/datasets/CommonVoice/en/test.tsv'
    >>> accented_letters=False
    >>> prepare = CommonVoicePreparer(
    >>>             local_folder,
    >>>             save_folder,
    >>>             path_to_wav,
    >>>             train_tsv_file,
    >>>             dev_tsv_file,
    >>>             test_tsv_file
    >>>             )
    >>> prepare()

    ```
    """

    def __init__(
        self,
        data_folder,
        save_folder,
        path_to_wav,
        train_tsv_file=None,
        dev_tsv_file=None,
        test_tsv_file=None,
        accented_letters=False,
    ):
        # Expected inputs when calling the class (no inputs in this case)
        super().__init__()
        self.data_folder = data_folder
        self.save_folder = save_folder
        self.path_to_wav = path_to_wav
        self.train_tsv_file = train_tsv_file
        self.dev_tsv_file = dev_tsv_file
        self.test_tsv_file = test_tsv_file
        self.accented_letters = accented_letters

        # If not specified point toward standard location
        if train_tsv_file is None:
            self.train_tsv_file = self.data_folder + "/train.tsv"
        else:
            self.train_tsv_file = train_tsv_file

        if dev_tsv_file is None:
            self.dev_tsv_file = self.data_folder + "/dev.tsv"
        else:
            self.dev_tsv_file = dev_tsv_file

        if test_tsv_file is None:
            self.test_tsv_file = self.data_folder + "/test.tsv"
        else:
            self.test_tsv_file = test_tsv_file

        # Test if files have already been converted to wav by checking if
        # path_to_wav exists.
        if os.path.exists(self.path_to_wav):
            msg = "%s already exists, skipping conversion of .mp3 files." % (
                self.path_to_wav
            )
            logger.info(msg)
            self.convert_to_wav = False
        else:
            msg = "%s doesn't exist, we will convert .mp3 files." % (
                self.path_to_wav
            )
            logger.info(msg)
            self.convert_to_wav = True
            os.makedirs(self.path_to_wav)

        self.samplerate = 16000

        # Setting file extension
        self.extension = [".wav"]

        # Setting the save folder
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        # Setting ouput files
        self.save_csv_train = self.save_folder + "/train.csv"
        self.save_csv_dev = self.save_folder + "/dev.csv"
        self.save_csv_test = self.save_folder + "/test.csv"

    def __call__(self):

        if self.skip():

            msg = "%s already exists, skipping data preparation!" % (
                self.save_csv_train
            )
            logger.info(msg)

            msg = "%s already exists, skipping data preparation!" % (
                self.save_csv_dev
            )
            logger.info(msg)

            msg = "%s already exists, skipping data preparation!" % (
                self.save_csv_test
            )
            logger.info(msg)

            return

        # Additional checks to make sure the data folder contains Common Voice
        self.check_commonvoice_folders()

        # Creating csv file for training data
        if self.train_tsv_file is not None:

            # Convert .mp3 files if required
            if self.convert_to_wav:
                msg = "Converting train audio files to .wav ..."
                logger.info(msg)
                self.convert_mp3_wav(
                    self.data_folder,
                    self.train_tsv_file,
                    self.path_to_wav,
                    self.samplerate,
                )

            self.create_csv(
                self.train_tsv_file,
                self.path_to_wav,
                self.save_csv_train,
                self.data_folder,
            )

        # Creating csv file for dev data
        if self.dev_tsv_file is not None:

            # Convert .mp3 files if required
            if self.convert_to_wav:
                msg = "Converting validation audio files to .wav ..."
                logger.info(msg)
                self.convert_mp3_wav(
                    self.data_folder,
                    self.dev_tsv_file,
                    self.path_to_wav,
                    self.samplerate,
                )

            self.create_csv(
                self.dev_tsv_file,
                self.path_to_wav,
                self.save_csv_dev,
                self.data_folder,
            )

        # Creating csv file for test data
        if self.test_tsv_file is not None:

            # Convert .mp3 files if required
            if self.convert_to_wav:
                msg = "Converting test audio files to .wav ..."
                logger.info(msg)
                self.convert_mp3_wav(
                    self.data_folder,
                    self.test_tsv_file,
                    self.path_to_wav,
                    self.samplerate,
                )

            self.create_csv(
                self.test_tsv_file,
                self.path_to_wav,
                self.save_csv_test,
                self.data_folder,
            )

        return

    def convert_mp3_wav(self, data_folder, tsv_file, path_to_wav, samplerate):
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
            logger.error(msg)
            raise FileNotFoundError(msg)

        # We load and skip the header
        loaded_csv = open(tsv_file, "r").readlines()[1:]

        nb_samples = str(len(loaded_csv))
        msg = "%s files to convert ..." % (str(nb_samples))
        logger.info(msg)

        for line in tzip(loaded_csv):
            line = line[0]
            # Path is at indice 1 in Common Voice tsv files. And .mp3 files
            # are located in datasets/lang/clips/
            mp3_path = data_folder + "/clips/" + line.split("\t")[1]
            file_name = mp3_path.split(".")[0].split("/")[-1]
            new_wav_path = path_to_wav + "/" + file_name + ".wav"

            # Convert to wav
            if os.path.isfile(mp3_path):
                sig, orig_rate = torchaudio.load(mp3_path)
                res = torchaudio.transforms.Resample(orig_rate, samplerate)
                sig = res(sig)
                torchaudio.save(new_wav_path, sig, samplerate)
            else:
                msg = "\tError loading: %s" % (str(len(file_name)))
                logger.debug(msg)

    def skip(self):
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
            os.path.isfile(self.save_csv_train)
            and os.path.isfile(self.save_csv_dev)
            and os.path.isfile(self.save_csv_test)
        ):
            skip = True

        return skip

    def create_csv(self, orig_tsv_file, path_to_wav, csv_file, data_folder):
        """
        Creates the csv file given a list of wav files.

        Arguments
        ---------
        orig_tsv_file : str
            Path to the Common Voice tsv file (standard file).
        wav_path : str
            Path of the audio wav files.
        csv_file : str
            Path for the output csv file.

        Returns
        -------
        None
        """

        # Check if the given files exists
        if not os.path.isfile(orig_tsv_file):
            msg = "\t%s doesn't exist, verify your dataset!" % (orig_tsv_file)
            logger.error(msg)
            raise FileNotFoundError(msg)

        if not os.path.exists(path_to_wav):
            msg = "\t%s doesn't exists, we need wav files !" % (path_to_wav)
            logger.error(msg)
            raise FileNotFoundError(msg)

        # We load and skip the header
        loaded_csv = open(orig_tsv_file, "r").readlines()[1:]
        nb_samples = str(len(loaded_csv))

        msg = "Preparing CSV files for %s samples ..." % (str(nb_samples))
        logger.debug(msg)

        # Adding some Prints
        msg = "Creating csv lists in %s ..." % (csv_file)
        logger.debug(msg)

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
        for line in loaded_csv:

            # Path is at indice 1 in Common Voice tsv files. And .mp3 files
            # are located in datasets/lang/clips/
            mp3_path = data_folder + "/clips/" + line.split("\t")[1]
            file_name = mp3_path.split(".")[0].split("/")[-1]
            wav_path = path_to_wav + "/" + file_name + ".wav"
            spk_id = line.split("\t")[0]
            snt_id = file_name

            # Reading the signal (to retrieve duration in seconds)
            if os.path.isfile(wav_path):
                signal = read_wav_soundfile(wav_path)
            else:
                msg = "\tError loading: %s" % (str(len(file_name)))
                logger.debug(msg)
                continue

            duration = signal.shape[0] / self.samplerate
            total_duration += duration

            # Getting transcript
            words = line.split("\t")[2]

            # Do a bit of cleaning on the transcript ...
            words = re.sub("[^'A-Za-z0-9 ]+", " ", words).upper()

            # Remove accents if specified
            if not self.accented_letters:
                nfkd_form = unicodedata.normalize("NFKD", words)
                words = "".join(
                    [c for c in nfkd_form if not unicodedata.combining(c)]
                )

            # Getting chars
            chars = words.replace(" ", "_")
            chars = " ".join([char for char in chars][:-1])

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
        with open(csv_file, mode="w") as csv_f:
            csv_writer = csv.writer(
                csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )

            for line in csv_lines:
                csv_writer.writerow(line)

        # Final prints
        msg = "%s sucessfully created!" % (csv_file)
        logger.info(msg)
        msg = "Number of samples: %s " % (str(len(loaded_csv)))
        logger.info(msg)
        msg = "Total duration: %s Hours" % (
            str(round(total_duration / 3600, 2))
        )
        logger.info(msg)

    def check_commonvoice_folders(self):
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
        if not os.path.exists(self.data_folder + files_str):

            err_msg = (
                "the folder %s does not exist (it is expected in "
                "the Common Voice dataset)" % (self.data_folder + files_str)
            )
            raise FileNotFoundError(err_msg)
