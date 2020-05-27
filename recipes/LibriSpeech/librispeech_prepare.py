"""
Data preparation.

Download: http://www.openslr.org/12

Author
------
Mirco Ravanelli, Ju-Chieh Chou 2020
"""

import os
import csv
import logging
from speechbrain.utils.data_utils import get_all_files
from speechbrain.data_io.data_io import (
    read_wav_soundfile,
    load_pkl,
    save_pkl,
)

logger = logging.getLogger(__name__)


class LibriSpeechPreparer:
    """
    This class prepares the csv files for the LibriSpeech dataset

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original LibriSpeech dataset is stored.
    splits : list
        List of splits to prepare from ['dev-clean','dev-others','test-clean',
        'test-others','train-clean-100','train-clean-360','train-other-500']
    save_folder : str
        The directory where to store the csv files.
    select_n_sentences : int
        Default : None
        If not None, only pick this many sentences.
    use_lexicon : bool
        When it is true, using lexicon to generate phonemes columns in the csv file.

    Example
    -------
    >>> from recipes.LibriSpeech.librispeech_prepare import LibriSpeechPreparer
    >>> data_folder='datasets/LibriSpeech'
    >>> splits=['train-clean-100', 'dev-clean', 'test-clean']
    >>> save_folder='librispeech_prepared'
    >>> LibriSpeechPreparer(data_folder,splits,save_folder)
    """

    def __init__(
        self,
        data_folder,
        splits,
        save_folder,
        select_n_sentences=None,
        use_lexicon=False,
    ):
        super().__init__()
        self.data_folder = data_folder
        self.splits = splits
        self.save_folder = save_folder
        self.select_n_sentences = select_n_sentences
        self.use_lexicon = use_lexicon
        self.conf = {
            "select_n_sentences": select_n_sentences,
            "use_lexicon": use_lexicon,
        }

        # Other variables
        self.samplerate = 16000
        # Saving folder
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        self.save_opt = self.save_folder + "/opt_librispeech_prepare.pkl"

        if use_lexicon:
            self.lexicon_dict = self.read_lexicon(
                self.data_folder + "/librispeech-lexicon.txt"
            )

        # Check if this phase is already done (if so, skip it)
        if self.skip():
            for split in self.splits:
                text = self.save_folder + "/" + split + ".csv"
                msg = "\t" + text + " created!"
                logger.debug(msg)
            return

        # Additional checks to make sure the data folder contains Librispeech
        self.check_librispeech_folders()

        # create csv files for each split
        for split_index in range(len(self.splits)):

            split = self.splits[split_index]

            wav_lst = get_all_files(
                self.data_folder + "/" + split, match_and=[".flac"]
            )

            text_lst = get_all_files(
                self.data_folder + "/" + split, match_and=["trans.txt"]
            )

            text_dict = self.text_to_dict(text_lst)

            if self.select_n_sentences is not None:
                select_n_sentences = self.select_n_sentences[split_index]
            else:
                select_n_sentences = len(wav_lst)

            self.create_csv(wav_lst, text_dict, split, select_n_sentences)

        # saving options
        save_pkl(self.conf, self.save_opt)

    def read_lexicon(self, lexicon_path):
        """
        Read the lexicon into a dictionary.
        Download link: http://www.openslr.org/resources/11/librispeech-lexicon.txt
        Arguments
        ---------
        lexicon_path : string
            The path of the lexicon.
        """
        if not os.path.exists(lexicon_path):
            err_msg = (
                f"Lexicon path {lexicon_path} does not exist."
                "Link: http://www.openslr.org/resources/11/librispeech-lexicon.txt"
            )
            raise OSError(err_msg)

        lexicon_dict = {}

        with open(lexicon_path, "r") as f:
            for line in f:
                line_lst = line.split()
                lexicon_dict[line_lst[0]] = " ".join(line_lst[1:])
        return lexicon_dict

    def __call__(self):
        return

    def create_csv(self, wav_lst, text_dict, split, select_n_sentences):
        """
        Create the csv file given a list of wav files.

        Arguments
        ---------
        wav_lst : list
            The list of wav files of a given data split.
        text_dict : list
            The dictionary containing the text of each sentence.
        split : str
            The name of the current data split.
        select_n_sentences : int, optional
            The number of sentences to select.

        Returns
        -------
        None
        """
        # Setting path for the csv file
        csv_file = self.save_folder + "/" + split + ".csv"

        # Preliminary prints
        msg = "\tCreating csv lists in  %s..." % (csv_file)
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

        # add phn column when there is a lexicon.
        if self.use_lexicon:
            csv_lines[0] += ["phn", "phn_format", "phn_opts"]

        snt_cnt = 0
        # Processing all the wav files in wav_lst
        for wav_file in wav_lst:

            snt_id = wav_file.split("/")[-1].replace(".flac", "")
            spk_id = "-".join(snt_id.split("-")[0:2])
            wrds = text_dict[snt_id]

            signal = read_wav_soundfile(wav_file)
            duration = signal.shape[0] / self.samplerate

            # replace space to <space> token
            chars_lst = [c for c in wrds]
            chars = " ".join(chars_lst)

            csv_line = [
                snt_id,
                str(duration),
                wav_file,
                "flac",
                "",
                spk_id,
                "string",
                "",
                str(" ".join(wrds.split("_"))),
                "string",
                "",
                str(chars),
                "string",
                "",
            ]

            if self.use_lexicon:
                # skip words not in the lexicon
                phns = " ".join(
                    [
                        self.lexicon_dict[wrd]
                        for wrd in wrds.split("_")
                        if wrd in self.lexicon_dict
                    ]
                )
                csv_line += [str(phns), "string", ""]

            #  Appending current file to the csv_lines list
            csv_lines.append(csv_line)
            snt_cnt = snt_cnt + 1

            if snt_cnt == select_n_sentences:
                break

        # Writing the csv_lines
        with open(csv_file, mode="w") as csv_f:
            csv_writer = csv.writer(
                csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )

            for line in csv_lines:
                csv_writer.writerow(line)

        # Final print
        msg = "\t%s sucessfully created!" % (csv_file)
        logger.debug(msg)

    def skip(self):
        """
        Detect when the librispeech data prep can be skipped.

        Returns
        -------
        bool
            if True, the preparation phase can be skipped.
            if False, it must be done.
        """

        # Checking csv files
        skip = True

        for split in self.splits:
            if not os.path.isfile(self.save_folder + "/" + split + ".csv"):
                skip = False

        #  Checking saved options
        if skip is True:
            if os.path.isfile(self.save_opt):
                opts_old = load_pkl(self.save_opt)
                if opts_old == self.conf:
                    skip = True
                else:
                    skip = False
            else:
                skip = False

        return skip

    @staticmethod
    def text_to_dict(text_lst):
        """
        This converts lines of text into a dictionary-

        Arguments
        ---------
        text_lst : str
            Path to the file containing the librispeech text transcription.

        Returns
        -------
        dict
            The dictionary containing the text transcriptions for each sentence.

        """
        # Initialization of the text dictionary
        text_dict = {}
        # Reading all the transcription files is text_lst
        for file in text_lst:
            with open(file, "r") as f:
                # Reading all line of the transcription file
                for line in f:
                    line_lst = line.strip().split(" ")
                    text_dict[line_lst[0]] = "_".join(line_lst[1:])
        return text_dict

    def check_librispeech_folders(self):
        """
        Check if the data folder actually contains the LibriSpeech dataset.

        If it does not, an error is raised.

        Returns
        -------
        None

        Raises
        ------
        OSError
            If LibriSpeech is not found at the specified path.
        """
        # Checking if all the splits exist
        for split in self.splits:
            if not os.path.exists(self.data_folder + "/" + split):
                err_msg = (
                    "the folder %s does not exist (it is expected in the "
                    "Librispeech dataset)" % (self.data_folder + "/" + split)
                )
                raise OSError(err_msg)
