"""
Data preparation (VoxCeleb1).
"""

import os
import csv
import logging
import glob
import random
from speechbrain.data_io.data_io import (
    read_wav_soundfile,
    load_pkl,
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
    kaldi_ali_tr : dict, optional
        Default: 'None'
        When set, this is the directiory where the kaldi
        training alignments are stored.  They will be automatically converted
        into pkl for an easier use within speechbrain.
    kaldi_ali_dev : str, optional
        Default: 'None'
        When set, this is the path to directory where the
        kaldi dev alignments are stored.
    kaldi_ali_test : str, optional
        Default: 'None'
        When set, this is the path to the directory where the
        kaldi test alignments are stored.
    kaldi_lab_opts : str, optional
        it a string containing the options use to compute
        the labels.

    Author
    ------
    Nauman Dawalatabad 2020
    """

    def __init__(
        self,
        data_folder,
        splits,
        save_folder,
        seg_dur=300,
        rand_seed=1234,
        vad=False,
        kaldi_ali_tr=None,
        kaldi_ali_dev=None,
        kaldi_ali_test=None,
        kaldi_lab_opts=None,
    ):

        self.data_folder = data_folder
        self.splits = splits
        self.vad = vad
        self.seg_dur = seg_dur
        self.rand_seed = rand_seed
        self.save_folder = save_folder
        self.kaldi_ali_tr = kaldi_ali_tr
        self.kaldi_ali_dev = kaldi_ali_dev
        self.kaldi_ali_test = kaldi_ali_test
        self.kaldi_lab_opts = kaldi_lab_opts
        self.samplerate = 16000
        random.seed(self.rand_seed)

        """
        wav_lst_train, wav_lst_dev, wav_lst_test = self.prepare_wav_list(
            self.data_folder
        )
        """
        # Split data into 90% train and 10% validation
        wav_lst_train, wav_lst_dev = self._get_data_split()

        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        # Setting ouput files
        self.save_opt = self.save_folder + "/opt_voxceleb_prepare.pkl"
        self.save_csv_train = self.save_folder + "/train.csv"
        self.save_csv_dev = self.save_folder + "/dev.csv"
        self.save_csv_test = (
            self.save_folder + "/test.csv"
        )  # use in case of iden

        # Check if this phase is already done (if so, skip it)
        if self.skip():
            msg = "\t%s already existed, skipping" % (self.save_csv_train)
            logger.debug(msg)
            msg = "\t%s already existed, skipping" % (self.save_csv_dev)
            logger.debug(msg)
            msg = "\t%s already existed, skipping" % (self.save_csv_test)
            logger.debug(msg)
            return

        # Additional checks to make sure the data folder contains TIMIT
        self.check_voxceleb1_folders()

        msg = "\tCreating csv file for the Voxceleb Dataset.."
        logger.debug(msg)

        # Creating csv file for training data
        if "train" in self.splits:
            self.prepare_csv(
                wav_lst_train,
                self.save_csv_train,
                kaldi_lab=self.kaldi_ali_tr,
                kaldi_lab_opts=self.kaldi_lab_opts,
            )

        if "dev" in self.splits:
            self.prepare_csv(
                wav_lst_dev,
                self.save_csv_dev,
                kaldi_lab=self.kaldi_ali_dev,
                kaldi_lab_opts=self.kaldi_lab_opts,
            )
        """
        if "test" in self.splits:
            self.prepare_csv(
                wav_lst_test,
                self.save_csv_test,
                kaldi_lab=self.kaldi_ali_tr,
                kaldi_lab_opts=self.kaldi_lab_opts,
            )
        """
        # Saving options (useful to skip this phase when already done)
        # save_pkl(self.conf, self.save_opt)

    # VOX
    def __call__(self):
        return

    # def _get_all_audio_file(self, data_folder):
    def _get_data_split(self):
        """
        Splits the audio file list into train (90%) and dev(10%)
        """
        audio_files_list = [
            f for f in glob.glob(self.data_folder + "**/*.wav", recursive=True)
        ]
        random.shuffle(audio_files_list)

        train_lst = audio_files_list[: int(0.9 * len(audio_files_list))]
        dev_lst = audio_files_list[int(0.9 * len(audio_files_list)) :]

        return train_lst, dev_lst

    # Future
    def _prepare_wav_list_from_iden(self, data_folder):
        """
        For future to be used for identification splits
        """
        iden_split_file = os.path.join(data_folder, "iden_split_sample.txt")
        train_wav_lst = []
        dev_wav_lst = []
        test_wav_lst = []
        with open(iden_split_file, "r") as f:
            for line in f:
                [spkr_split, audio_path] = line.split(" ")
                [spkr_id, session_id, utt_id] = audio_path.split("/")
                if spkr_split not in ["1", "2", "3"]:
                    # todo: raise an error here!
                    break
                if spkr_split == "1":
                    train_wav_lst.append(
                        os.path.join(data_folder, audio_path.strip())
                    )
                if spkr_split == "2":
                    dev_wav_lst.append(
                        os.path.join(data_folder, audio_path.strip())
                    )
                if spkr_split == "3":
                    test_wav_lst.append(
                        os.path.join(data_folder, audio_path.strip())
                    )
        f.close()
        return train_wav_lst, dev_wav_lst, test_wav_lst

    def skip(self):
        """
        Detect when the VoxCeleb data preparation can be skipped.

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
            opts_old = load_pkl(self.save_opt)
            if opts_old == self.conf:
                skip = True
        return skip

    def _get_chunks(self, audio_id, audio_duration):
        num_chunks = int(
            audio_duration * 100 / self.seg_dur
        )  # all in milliseconds
        # print(num_chunks)
        # print (audio_duration, self.seg_dur, num_chunks)
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
        self, wav_lst, csv_file, vad=False, kaldi_lab=None, kaldi_lab_opts=None,
    ):
        """
        Creates the csv file given a list of wav files.

        Arguments
        ---------
        wav_lst : list
            The list of wav files of a given data split.
        csv_file : str
            The path of the output csv file
        kaldi_lab : str, optional
            Default: None
            The path of the kaldi labels (optional).
        kaldi_lab_opts : str, optional
            Default: None
            A string containing the options used to compute the labels.

        Returns
        -------
        None

        Example
        -------
        # Sample output csv list
        id10001---1zcIwhmdeo4---00001,8.1200625, \
              /home/nauman/datasets/VoxCeleb1/id10001/ \
              1zcIwhmdeo4/00001.wav,wav, ,id10001,string,
        id10002---xTV-jFAUKcw---00001,5.4400625, \
              /home/nauman/datasets/VoxCeleb1/id10002/ \
              xTV-jFAUKcw/00001.wav,wav, ,id10002,string,
        """

        # Adding some Prints
        msg = '\t"Creating csv lists in  %s..."' % (csv_file)
        logger.debug(msg)

        # Reading kaldi labels if needed:
        # FIX: These statements were unused, should they be deleted?
        # snt_no_lab = 0
        # missing_lab = False
        """
        # Kaldi labs will be added in future
        if kaldi_lab is not None:

            lab = read_kaldi_lab(
                kaldi_lab,
                kaldi_lab_opts,
                logfile=self.global_config["output_folder"] + "/log.log",
            )

            lab_out_dir = self.save_folder + "/kaldi_labels"

            if not os.path.exists(lab_out_dir):
                os.makedirs(lab_out_dir)
        """
        # example5, 1.000, $data_folder/example5.wav, wav, start:10000 stop:26000, spk05, string,
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
            ]
        ]

        """
        if kaldi_lab is not None:
            csv_lines[0].append("kaldi_lab")
            csv_lines[0].append("kaldi_lab_format")
            csv_lines[0].append("kaldi_lab_opts")
        """

        # ND: update separator
        my_sep = "---"
        # Processing all the wav files in the list
        for wav_file in wav_lst:

            # Getting sentence and speaker ids
            [spk_id, sess_id, utt_id] = wav_file.split("/")[-3:]  # 3 from last
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

                # Composition of the csv_line
                csv_line = [
                    chunk,
                    str(self.seg_dur / 100),
                    wav_file,
                    "wav",
                    start_stop,
                    spk_id,
                    "string",
                    " ",
                ]

                csv_lines.append(csv_line)

        # -Writing the csv lines
        with open(csv_file, mode="w") as csv_f:
            csv_writer = csv.writer(
                csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )

            for line in csv_lines:
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
                "the folder %s does not exist (it is expected in "
                "the Voxceleb dataset)" % (self.data_folder + "/id*")
            )
            raise FileNotFoundError(err_msg)
