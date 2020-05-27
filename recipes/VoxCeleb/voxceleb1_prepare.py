"""
Data preparation.

Download: http://www.robots.ox.ac.uk/~vgg/data/voxceleb/

Author
------
Nauman Dawalatabad 2020
"""

import os
import csv
import logging
import glob
import random
from speechbrain.data_io.data_io import (
    read_wav_soundfile,
    load_pkl,
    save_pkl,
)

logger = logging.getLogger(__name__)


class VoxCelebPreparer:
    """
    Prepares the csv files for the Voxceleb1 dataset.

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original VoxCeleb dataset is stored.
    splits : list
        List of splits to prepare from ['train', 'dev', 'test']
    save_folder : str
        The directory where to store the csv files.

    Example
    -------
    >>> from recipes.VoxCeleb.voxceleb1_prepare import VoxCelebPreparer
    >>> data_folder='/Users/nauman/Desktop/Mila/nauman/data_folder/'
    >>> splits = ['train', 'dev']
    >>> save_folder = 'results/'
    >>> seg_dur = 300
    >>> vad = False
    >>> VoxCelebPreparer(data_folder, splits, save_folder, seg_dur, vad)
    """

    def __init__(
        self,
        data_folder,
        splits,
        save_folder,
        seg_dur=300,
        vad=False,
        rand_seed=1234,
    ):

        self.data_folder = os.path.join(data_folder, "wav/")
        self.iden_meta_file = os.path.join(data_folder, "meta/iden_split.txt")
        self.splits = splits
        self.vad = vad
        self.seg_dur = seg_dur
        self.rand_seed = rand_seed
        self.save_folder = save_folder
        self.samplerate = 16000
        random.seed(self.rand_seed)

        self.conf = {
            "data_folder": self.data_folder,
            "splits": self.splits,
            "save_folder": self.save_folder,
            "vad": self.vad,
            "seg_dur": self.seg_dur,
        }

        # Split data into 90% train and 10% validation (verification split)
        wav_lst_train, wav_lst_dev = self._get_data_split()

        # Split data according to identification split
        # wav_lst_train, wav_lst_dev = self._get_data_iden_split()

        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        # Setting ouput files
        self.save_opt = self.save_folder + "/opt_voxceleb_prepare.pkl"
        self.save_csv_train = self.save_folder + "/train.csv"
        self.save_csv_dev = self.save_folder + "/dev.csv"

        # Check if this phase is already done (if so, skip it)
        if self.skip():
            msg = "\t%s already existed, skipping" % (self.save_csv_train)
            logger.debug(msg)
            msg = "\t%s already existed, skipping" % (self.save_csv_dev)
            logger.debug(msg)
            return

        # Additional checks to make sure the data folder contains TIMIT
        self.check_voxceleb1_folders()

        msg = "\tCreating csv file for the Voxceleb Dataset.."
        logger.debug(msg)

        # Creating csv file for training data
        if "train" in self.splits:
            self.prepare_csv(
                wav_lst_train, self.save_csv_train,
            )

        if "dev" in self.splits:
            self.prepare_csv(
                wav_lst_dev, self.save_csv_dev,
            )

        # Saving options (useful to skip this phase when already done)
        save_pkl(self.conf, self.save_opt)

    # VOX
    def __call__(self):
        return

    def _get_data_iden_split(self):
        """
        Uses standard voxceleb1 identification split.
        Tot. number of speakers = 1251
        Here, we use following split
        For training = 145,265 utterances (mentioned as dev split)
        For validation = 8,251 utterances (mentioned as test split)

        Arguments
        ---------
        self.data_folder : string
            The path of the audio and iden_split.txt files
        """

        (
            train_wav_lst,
            dev_wav_lst,
            test_wav_lst,
        ) = self._prepare_wav_list_from_iden()

        # Get complete training list (train+dev)
        train_wav_lst = train_wav_lst + dev_wav_lst

        # Utterance level shuffling
        random.shuffle(train_wav_lst)
        valid_wav_lst = test_wav_lst
        random.shuffle(valid_wav_lst)

        return train_wav_lst, test_wav_lst

    # Used for verification split
    def _get_data_split(self):
        """
        Splits the audio file list into train (90%) and dev(10%).
        This function is useful when using verification split
        """
        audio_files_list = [
            f for f in glob.glob(self.data_folder + "**/*.wav", recursive=True)
        ]
        random.shuffle(audio_files_list)
        # print ('self.data_folder . .', self.data_folder)
        # print ('asdasdsadsa ...', audio_files_list)
        train_lst = audio_files_list[: int(0.9 * len(audio_files_list))]
        dev_lst = audio_files_list[int(0.9 * len(audio_files_list)) :]

        return train_lst, dev_lst

    # When using identification splits
    def _prepare_wav_list_from_iden(self):
        """
        Prepares list of audio files given data_folder
        """

        # iden_split_file = os.path.join(self.data_folder, "meta/iden_split.txt")
        iden_split_file = self.iden_meta_file
        train_wav_lst = []
        dev_wav_lst = []
        test_wav_lst = []

        with open(iden_split_file, "r") as f:
            for line in f:
                [spkr_split, audio_path] = line.split(" ")
                [spkr_id, session_id, utt_id] = audio_path.split("/")
                if spkr_split == "1":
                    train_wav_lst.append(
                        os.path.join(self.data_folder, audio_path.strip())
                    )
                if spkr_split == "2":
                    dev_wav_lst.append(
                        os.path.join(self.data_folder, audio_path.strip())
                    )
                if spkr_split == "3":
                    test_wav_lst.append(
                        os.path.join(self.data_folder, audio_path.strip())
                    )
        f.close()
        return train_wav_lst, dev_wav_lst, test_wav_lst

    def skip(self):
        """
        Detect when the VoxCeleb data preparation can be skipped.

        Arguments
        --------
        self :
            if True, the preparation phase can be skipped.
            if False, it must be done.
        """

        # Checking folders and save options
        skip = False
        if (
            os.path.isfile(self.save_csv_train)
            and os.path.isfile(self.save_csv_dev)
            and os.path.isfile(self.save_opt)
        ):
            opts_old = load_pkl(self.save_opt)
            if opts_old == self.conf:
                skip = True
        return skip

    def _get_chunks(self, audio_id, audio_duration):
        num_chunks = int(
            audio_duration * 100 / self.seg_dur
        )  # all in milliseconds

        chunk_lst = [
            audio_id
            + "_"
            + str(i * self.seg_dur)
            + "_"
            + str(i * self.seg_dur + self.seg_dur)
            for i in range(num_chunks)
        ]
        # print (chunk_lst)
        return chunk_lst

    def prepare_csv(
        self, wav_lst, csv_file, vad=False,
    ):
        """
        Creates the csv file given a list of wav files.

        Arguments
        ---------
        wav_lst : list
            The list of wav files of a given data split.
        csv_file : str
            The path of the output csv file
        vad : bool
            Perform VAD. True or False

        Returns
        -------
        None

        """

        # Adding some Prints
        msg = '\t"Creating csv lists in  %s..."' % (csv_file)
        logger.debug(msg)

        # Important format: example5, 1.000, $data_folder/example5.wav, wav, start:10000 stop:26000, spk05, string,
        csv_output = [
            [
                "ID",
                "duration",
                "wav",
                "wav_format",
                "wav_opts",
                "spk_id",
                "spk_id_format",
                "spk_id_opts",
            ]
        ]

        # For assiging unique ID to each chunk
        my_sep = "--"
        entry = []
        # Processing all the wav files in the list
        for wav_file in wav_lst:

            # Getting sentence and speaker ids
            [spk_id, sess_id, utt_id] = wav_file.split("/")[-3:]
            audio_id = my_sep.join([spk_id, sess_id, utt_id.split(".")[0]])

            # Reading the signal (to retrieve duration in seconds)
            signal = read_wav_soundfile(wav_file)
            audio_duration = signal.shape[0] / self.samplerate

            uniq_chunks_list = self._get_chunks(audio_id, audio_duration)

            for chunk in uniq_chunks_list:
                s, e = chunk.split("_")[-2:]
                start_sample = int(int(s) / 100 * self.samplerate)
                end_sample = int(int(e) / 100 * self.samplerate)

                start_stop = (
                    "start:" + str(start_sample) + " stop:" + str(end_sample)
                )

                audio_file_sb = (
                    "$data_folder/wav/" + spk_id + "/" + sess_id + "/" + utt_id
                )

                # Composition of the csv_line
                csv_line = [
                    chunk,
                    str(self.seg_dur / 100),
                    audio_file_sb,
                    "wav",
                    start_stop,
                    spk_id,
                    "string",
                    " ",
                ]

                entry.append(csv_line)

        # Shuffling at chunk level
        random.shuffle(entry)
        csv_output = csv_output + entry

        # Writing the csv lines
        with open(csv_file, mode="w") as csv_f:
            csv_writer = csv.writer(
                csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )

            for line in csv_output:
                csv_writer.writerow(line)

        # Final prints
        msg = "\t%s sucessfully created!" % (csv_file)
        logger.debug(msg)

    def check_voxceleb1_folders(self):
        """
        Check if the data folder actually contains the Voxceleb1 dataset.

        If it does not, raise an error.

        Returns
        -------
        None

        Raises
        ------
        FileNotFoundError
        """
        # Checking
        if not os.path.exists(self.data_folder + "/id10001"):
            err_msg = (
                "the folder %s does not exist (as it is expected in "
                "the Voxceleb dataset)" % (self.data_folder + "/id*")
            )
            raise FileNotFoundError(err_msg)
