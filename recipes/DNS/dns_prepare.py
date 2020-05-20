"""
Data preparation.
"""

import os
import csv
import torch
import logging
from speechbrain.utils.data_utils import get_all_files

from speechbrain.data_io.data_io import read_wav_soundfile

logger = logging.getLogger(__name__)


class DNSPreparer(torch.nn.Module):
    """
    repares the csv files for the TIMIT dataset.

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original TIMIT dataset is stored.
    splits : list
        List of splits to prepare from ['train', 'dev', 'test']
    save_folder : str
        The directory where to store the csv files.

    Example
    -------
    This example requires the actual TIMIT dataset.
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
    Mirco Ravanelli
    """

    def __init__(
        self, data_folder, save_folder,
    ):
        # Expected inputs when calling the class (no inputs in this case)
        super().__init__()
        self.data_folder = data_folder
        self.save_folder = save_folder

        self.train_clean_folder = os.path.join(
            self.data_folder, "training/clean/"
        )
        self.train_noise_folder = os.path.join(
            self.data_folder, "training/noise/"
        )
        self.train_noisy_folder = os.path.join(
            self.data_folder, "training/noisy/"
        )
        self.test_folder = os.path.join(
            self.data_folder, "datasets/test_set/synthetic/"
        )

        # Other variables
        self.samplerate = 16000

        # Setting file extension.
        self.extension = [".wav"]

        # Setting the save folder
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        # Setting ouput files
        self.save_csv_train = self.save_folder + "/train.csv"
        self.save_csv_test = self.save_folder + "/test.csv"

        # Check if this phase is already done (if so, skip it)
        if self.skip():

            msg = "\t%s sucessfully created!" % (self.save_csv_train)
            logger.debug(msg)

            msg = "\t%s sucessfully created!" % (self.save_csv_test)
            logger.debug(msg)

            return

    def __call__(self):
        # Additional checks to make sure the data folder contains TIMIT
        self.check_DNS_folders()

        msg = "\tCreating csv file for the ms_DNS Dataset.."
        logger.debug(msg)

        # Creating csv file for training data
        wav_lst_train = get_all_files(
            self.train_noisy_folder, match_and=self.extension,
        )

        self.create_csv(
            wav_lst_train, self.save_csv_train, is_noise_folder=True,
        )

        # Creating csv file for test data
        wav_lst_test = get_all_files(
            self.test_folder, match_and=self.extension, exclude_or=["/clean/"],
        )

        self.create_csv(
            wav_lst_test, self.save_csv_test,
        )

        return

    def skip(self):
        """
        Detects if the timit data_preparation has been already done.

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
            and os.path.isfile(self.save_csv_test)
            # and os.path.isfile(self.save_opt)
        ):
            skip = True

        return skip

    # TODO: Consider making this less complex
    def create_csv(  # noqa: C901
        self, wav_lst, csv_file, is_noise_folder=False
    ):
        """
        Creates the csv file given a list of wav files.

        Arguments
        ---------
        wav_lst : list
            The list of wav files of a given data split.
        csv_file : str
            The path of the output csv file

        Returns
        -------
        None
        """

        # Adding some Prints
        msg = '\t"Creating csv lists in  %s..."' % (csv_file)
        logger.debug(msg)

        csv_lines = [
            [
                "ID",
                "duration",
                "noisy_wav",
                "noisy_wav_format",
                "noisy_wav_opts",
                "clean_wav",
                "clean_wav_format",
                "clean_wav_opts",
                "noise_wav",
                "noise_wav_format",
                "noise_wav_opts",
                # "SNR",
            ]
        ]

        # Processing all the wav files in the list
        for wav_file in wav_lst:

            # Example wav_file: /path/training/noisy/book_00000_chp_0009_reader_06709_10_MTzjwt0Sgo0-C3KP2eKC7l0-gcZAba9W5R0_snr38_fileid_35203.wav
            # Getting fileids
            full_file_name = wav_file.split("/")[-1]
            fileid = full_file_name.split("_")[-1]  # 35203.wav

            clean_folder = os.path.join(
                os.path.split(os.path.split(wav_file)[0])[0], "clean"
            )
            clean_wav = clean_folder + "/clean_fileid_" + fileid

            if is_noise_folder:
                noise_folder = os.path.join(
                    os.path.split(os.path.split(wav_file)[0])[0], "noise"
                )
                noise_wav = noise_folder + "/noise_fileid_" + fileid
            else:
                noise_wav = ""

            # Reading the signal (to retrieve duration in seconds)
            signal = read_wav_soundfile(wav_file)
            duration = signal.shape[0] / self.samplerate

            # Composition of the csv_line
            csv_line = [
                "fileid_" + fileid,
                str(duration),
                wav_file,
                "wav",
                "",
                clean_wav,
                "wav",
                "",
                noise_wav,
                "wav",
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

    def check_DNS_folders(self):
        """
        Check if the data folder actually contains the DNS training dataset.

        If not, raises an error.

        Returns
        -------
        None

        Raises
        ------
        FileNotFoundError
            If data folder doesn't contain DNS dataset (training and testset included).
        """
        # Checking clean folder
        if not os.path.exists(self.train_clean_folder):

            err_msg = (
                "the folder %s does not exist (it is expected in "
                "the DNS dataset)" % (self.train_clean_folder)
            )
            raise FileNotFoundError(err_msg)

        # Checking noise folder
        if not os.path.exists(self.train_noise_folder):

            err_msg = (
                "the folder %s does not exist (it is expected in "
                "the DNS dataset)" % (self.train_noise_folder)
            )
            raise FileNotFoundError(err_msg)

        # Checking noisy folder
        if not os.path.exists(self.train_noisy_folder):

            err_msg = (
                "the folder %s does not exist (it is expected in "
                "the DNS dataset)" % (self.train_noisy_folder)
            )
            raise FileNotFoundError(err_msg)

        # Checking testset folder
        if not os.path.exists(self.test_folder):

            err_msg = (
                "the folder %s does not exist (it is expected in "
                "the DNS dataset)" % (self.test_folder)
            )
            raise FileNotFoundError(err_msg)
