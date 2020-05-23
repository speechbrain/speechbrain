# -*- coding: utf-8 -*-
"""
Data preparation.

Download: https://datashare.is.ed.ac.uk/handle/10283/1942

Author
------
Szu-Wei Fu 2020
"""

import os
import csv
import torch
import logging
from speechbrain.utils.data_utils import get_all_files
from speechbrain.data_io.data_io import read_wav_soundfile
    
logger = logging.getLogger(__name__)


class VoicebankPreparer(torch.nn.Module):
    """
    Prepares the csv files for the Voicebank dataset.
    Arguments
    ---------
    data_folder : str
        Path to the folder where the original Voicebank dataset is stored.
    splits : list
        List of splits to prepare from ['train', 'dev', 'test']
    save_folder : str
        The directory where to store the csv files.
    
    Example
    -------
    This example requires the actual Voicebank dataset.
    The noisy_vctk_prepare.py can be used to download the dataset.
    ```
    local_folder='/path/to/datasets/Voicebank'
    save_folder='exp/Voicebank_exp'
    # Definition of the config dictionary
    data_folder = local_folder
    # Initialization of the class
    VoicebankPreparer(data_folder, save_folder)
    ```
    Author
    ------
    Szu-Wei Fu
    """

    def __init__(
        self,
        data_folder,
        save_folder,
    ):
        # Expected inputs when calling the class (no inputs in this case)
        super().__init__()
        self.data_folder = data_folder
        self.save_folder = save_folder

        self.train_clean_folder = os.path.join(
            self.data_folder, "clean_trainset_28spk_wav_16k/"
        )
        self.train_noisy_folder = os.path.join(
            self.data_folder, "noisy_trainset_28spk_wav_16k/"
        )
        self.test_clean_folder = os.path.join(
            self.data_folder, "clean_testset_wav_16k/"
        )
        self.test_noisy_folder = os.path.join(
            self.data_folder, "noisy_testset_wav_16k/"
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
        # Additional checks to make sure the data folder contains Voicebank
        self.check_Voicebank_folders()

        msg = "\tCreating csv file for the Voicebank Dataset.."
        logger.debug(msg)

        # Creating csv file for training data
        wav_lst_train = get_all_files(
            self.train_noisy_folder, match_and=self.extension,
        )

        self.create_csv(
            wav_lst_train, self.save_csv_train, is_train_folder=True,
        )

        # Creating csv file for testing data
        wav_lst_test = get_all_files(
            self.test_noisy_folder, match_and=self.extension,
        )

        self.create_csv(
            wav_lst_test, self.save_csv_test, is_train_folder=False,
        )

        return

    def skip(self):
        """
        Detects if the Voicebank data_preparation has been already done.
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
        ):
            skip = True

        return skip

    # TODO: Consider making this less complex
    def create_csv(  # noqa: C901
        self, wav_lst, csv_file, is_train_folder
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
            ]
        ]


        # Processing all the wav files in the list
        for wav_file in wav_lst:  # ex:p203_122.wav

            # Example wav_file: p232_001.wav
            # Getting fileids
            snt_id = wav_file.split("/")[-1]
            
            if is_train_folder:
                clean_folder=self.train_clean_folder
            else:
                clean_folder=self.test_clean_folder            
            clean_wav = clean_folder + snt_id


            # Reading the signal (to retrieve duration in seconds)
            signal = read_wav_soundfile(wav_file)
            duration = signal.shape[0] / self.samplerate


            # Composition of the csv_line
            csv_line = [
                snt_id.replace(".wav", ""),
                str(duration),
                wav_file,
                "wav",
                "",
                clean_wav,
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

    def check_Voicebank_folders(self):
        """
        Check if the data folder actually contains the Voicebank dataset.
        If not, raises an error.
        Returns
        -------
        None
        Raises
        ------
        FileNotFoundError
            If data folder doesn't contain Voicebank dataset.
        """

        # Checking train_clean folder
        if not os.path.exists(self.train_clean_folder):

            err_msg = (
                "the folder %s does not exist (it is expected in "
                "the DNS dataset)" % (self.train_clean_folder)
            )
            raise FileNotFoundError(err_msg)

        # Checking train_noisy folder
        if not os.path.exists(self.train_noisy_folder):

            err_msg = (
                "the folder %s does not exist (it is expected in "
                "the DNS dataset)" % (self.train_noisy_folder)
            )
            raise FileNotFoundError(err_msg)

        # Checking test_clean folder
        if not os.path.exists(self.test_clean_folder):

            err_msg = (
                "the folder %s does not exist (it is expected in "
                "the DNS dataset)" % (self.test_clean_folder)
            )
            raise FileNotFoundError(err_msg)

        # Checking test_noisy folder
        if not os.path.exists(self.test_noisy_folder):

            err_msg = (
                "the folder %s does not exist (it is expected in "
                "the DNS dataset)" % (self.test_noisy_folder)
            )
            raise FileNotFoundError(err_msg)