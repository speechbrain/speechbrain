"""
Data preparation.
"""

import os
import csv
import torch
import re
import logging
import torchaudio

from speechbrain.data_io.data_io import read_wav_soundfile

logger = logging.getLogger(__name__)


class CommonVoicePreparer(torch.nn.Module):
    """
    repares the csv files for the Mozilla Common Voice dataset.

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original Common Voice dataset is stored.
        This path should include the lang: /home/test/commonvoice/en/
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

    Example
    -------
    This example requires the Common Voice dataset.
    ```
    local_folder='/home/mirco/datasets/TIMIT'
    save_folder='exp/TIMIT_exp'
    # Definition of the config dictionary
    data_folder = local_folder
    splits = ['train', 'test', 'dev']
    # Initialization of the class
    TIMITPreparer(data_folder, splits)
    ```

    Author
    ------
    Titouan Parcollet
    """

    def __init__(
        self,
        data_folder,
        save_folder,
        path_to_wav,
        train_tsv_file=None,
        dev_tsv_file=None,
        test_tsv_file=None,
    ):
        # Expected inputs when calling the class (no inputs in this case)
        super().__init__()
        self.data_folder = data_folder
        self.save_folder = save_folder
        self.path_to_wav = path_to_wav
        self.train_tsv_file = train_tsv_file
        self.dev_tsv_file = dev_tsv_file
        self.test_tsv_file = test_tsv_file

        # Test if files have already been converted to wav by checking if
        # path_to_wav exists.
        if os.path.exists(self.path_to_wav):
            msg = "\t%s already exists, skipping conversion of .mp3 files." % (
                self.path_to_wav
            )
            logger.debug(msg)
            self.convert_to_wav = False
        else:
            msg = "\t%s doesn't exist, we will convert .mp3 files." % (
                self.path_to_wav
            )
            logger.debug(msg)
            self.convert_to_wav = True
            os.makedirs(self.path_to_wav)

        self.samplerate = 16000

        # Setting file extension
        self.extension = [".wav"]

        # Setting the save folder
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        # Setting ouput files
        self.save_opt = self.save_folder + "/opt_commonvoice_prepare.pkl"
        self.save_csv_train = self.save_folder + "/train.csv"
        self.save_csv_dev = self.save_folder + "/dev.csv"
        self.save_csv_test = self.save_folder + "/test.csv"

        # Check if this phase is already done (if so, skip it)
        if self.skip():

            msg = "\t%s sucessfully created!" % (self.save_csv_train)
            logger.debug(msg)

            msg = "\t%s sucessfully created!" % (self.save_csv_dev)
            logger.debug(msg)

            msg = "\t%s sucessfully created!" % (self.save_csv_test)
            logger.debug(msg)

            return

    def __call__(self):

        # Additional checks to make sure the data folder contains Common Voice
        self.check_commonvoice_folders()

        msg = "\tCreating csv files for the Common Voice Dataset.."
        logger.debug(msg)

        # Creating csv file for training data
        if self.train_tsv_file is not None:

            # Convert .mp3 files if required
            if self.convert_to_wav:
                msg = "\tConverting train audio files to .wav ..."
                logger.debug(msg)
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
                msg = "\tConverting validation audio files to .wav ..."
                logger.debug(msg)
                self.convert_mp3_wav(
                    self.data_folder,
                    self.dev_tsv_file,
                    self.path_to_wav,
                    self.samplerate,
                )

            self.create_csv(
                self.dev_tsv_file,
                self.path_to_wav,
                self.save_csv_train,
                self.data_folder,
            )

        # Creating csv file for test data
        if self.test_tsv_file is not None:

            # Convert .mp3 files if required
            if self.convert_to_wav:
                msg = "\tConverting test audio files to .wav ..."
                logger.debug(msg)
                self.convert_mp3_wav(
                    self.data_folder,
                    self.test_tsv_file,
                    self.path_to_wav,
                    self.samplerate,
                )

            self.create_csv(
                self.test_tsv_file,
                self.path_to_wav,
                self.save_csv_train,
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
            msg = "\t%s doesn't exist, verify your dataset!" % (tsv_file)
            logger.debug(msg)
            raise FileNotFoundError(msg)

        if os.path.exists(path_to_wav):
            msg = "\t%s exists, we might override some files!" % (path_to_wav)
            logger.debug(msg)

        # We load and skip the header
        loaded_csv = open(tsv_file, "r").readlines()[1:]

        nb_samples = str(len(loaded_csv))
        msg = "\t%s files to convert ..." % (str(len(nb_samples)))
        logger.debug(msg)

        for line in loaded_csv:

            # Path is at indice 1 in Common Voice tsv files. And .mp3 files
            # are located in datasets/lang/clips/
            mp3_path = data_folder + "/clips/" + line.split("\t")[1]
            file_name = mp3_path.split(".")[0].split("/")[-1]
            new_wav_path = path_to_wav + "/" + file_name + ".wav"

            # Convert to wav
            try:
                sig, orig_rate = torchaudio.load(mp3_path)
                res = torchaudio.transforms.Resample(orig_rate, samplerate)
                sig = res(sig)
                torchaudio.save(new_wav_path, sig, samplerate)
            except ValueError:
                print("erro")
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
            and os.path.isfile(self.save_opt)
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
            logger.debug(msg)
            raise FileNotFoundError(msg)

        if not os.path.exists(path_to_wav):
            msg = "\t%s doesn't exists, we need wav files !" % (path_to_wav)
            logger.debug(msg)
            raise FileNotFoundError(msg)

        # We load and skip the header
        loaded_csv = open(orig_tsv_file, "r").readlines()[1:]
        nb_samples = str(len(loaded_csv))

        msg = "\t Preparing CSV files for %s samples ..." % (str(nb_samples))
        logger.debug(msg)

        # Adding some Prints
        msg = '\t"Creating csv lists in  %s..."' % (csv_file)
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

        for line in loaded_csv:

            # Path is at indice 1 in Common Voice tsv files. And .mp3 files
            # are located in datasets/lang/clips/
            mp3_path = data_folder + "/clips/" + line.split("\t")[1]
            file_name = mp3_path.split(".")[0].split("/")[-1]
            wav_path = path_to_wav + "/" + file_name + ".wav"
            spk_id = line.split("\t")[0]
            snt_id = file_name

            # Reading the signal (to retrieve duration in seconds)
            signal = read_wav_soundfile(wav_path)
            duration = signal.shape[0] / self.samplerate

            # Getting transcript
            words = line.split("\t")[2]

            # Do a bit of cleaning on the transcript ...
            reg = r"\W+"
            words = re.sub(reg, " ", words).upper()

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
        msg = "\t%s sucessfully created!" % (csv_file)
        logger.debug(msg)

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
