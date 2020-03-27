"""
-----------------------------------------------------------------------------
 data_preparation.py

 Description: This library gathers classes form data preparation.
 -----------------------------------------------------------------------------
 """

import os
import sys
import csv
import errno
import torch
import logging
from speechbrain.utils.data_utils import get_all_files
from speechbrain.utils.superpowers import run_shell

from speechbrain.data_io.data_io import (
    read_wav_soundfile,
    load_pkl,
    save_pkl,
    read_kaldi_lab,
    write_txt_file,
)
logger = logging.getLogger(__name__)


class copy_data_locally():
    """
     -------------------------------------------------------------------------
     Description:
        This class copies a compressed dataset into another folder.
        It can be used to store the data locally when the original
        dataset it is stored in a shared filesystem.

     Input:- data_file (type: file_list, mandatory):
               it is a list containing the files to copy.

           - local_folder (type: directory, mandatory):
               it the local directory where to store the
               dataset. The dataset will be uncompressed in
               this folder.

           - copy_cmd (type: str, default: 'rsync'):
               it is command to run for copying the dataset.

           - copy_opts (type: str, default: ''):
               it is a string containing the flags to be used
               for the copy command copy_cmd.

           - uncompress_cmd (type: str, default: 'tar'):
               it is command to uncompress the dataset.

           - uncompress_opts (type: str, default: '-zxf'):
               it is a string containing the flags to be used
               for the uncompress command.

     Example:    from speechbrain.data_io.data_preparation import (
                    copy_data_locally)

                 data_file='/home/mirco/datasets/TIMIT.tar.gz'
                 local_folder='/home/mirco/datasets/local_folder/TIMIT'

                # Initialization of the class
                copy_data_locally(data_file=data_file,
                                  local_folder=local_folder)
     -------------------------------------------------------------------------
     """

    def __init__(
        self,
        data_file,
        local_folder,
        copy_cmd='rsync',
        copy_opts='',
        uncompress_cmd='tar',
        uncompress_opts='-zxf',
    ):
        self.data_file = data_file
        self.local_folder = local_folder
        self.copy_cmd = copy_cmd
        self.copy_opts = copy_opts
        self.uncompress_cmd = uncompress_cmd
        self.uncompress_opts = uncompress_opts

    def __call__(self):

        # Try to make the local folder
        try:
            os.makedirs(self.local_folder)
        except OSError as e:
            if e.errno != errno.EEXIST:

                err_msg = "Cannot create the data local folder %s!" % (
                    self.local_folder
                )

                logger.error(err_msg, exc_info=True)

        self.local_folder = self.local_folder + "/"
        upper_folder = os.path.dirname(os.path.dirname(self.local_folder))

        # Copying all the files in the data_file list
        for data_file in self.data_file:

            # Destination file
            self.dest_file = upper_folder + "/" + os.path.basename(data_file)

            if not os.path.exists(self.dest_file):

                # Copy data file in the local_folder
                msg = "\tcopying file %s into %s !" % (
                    data_file,
                    self.dest_file,
                )

                logger.debug(msg)

                cmd = (
                    self.copy_cmd
                    + " "
                    + self.copy_opts
                    + data_file
                    + " "
                    + self.dest_file
                )

                run_shell(cmd)

                # Uncompress the data_file in the local_folder
                msg = "\tuncompressing file %s into %s !" % (
                    self.dest_file,
                    self.local_folder,
                )

                logger.debug(msg)

                cmd = (
                    self.uncompress_cmd
                    + " "
                    + self.uncompress_opts
                    + self.dest_file
                    + " -C "
                    + " "
                    + self.local_folder
                    + " --strip-components=1"
                )

                run_shell(cmd)


class timit_prepare(torch.nn.Module):
    """
     -------------------------------------------------------------------------
     speechbrain.data_io.data_preparation.timit_prepare
     (author: Mirco Ravanelli)

     Description: This class prepares the csv files for the TIMIT dataset.

     Input: - data_folder (type: directory, mandatory):
               it the folder where the original TIMIT dataset
               is stored.

           - splits ('train','dev','test',mandatory):
               it the local directory where to store the
               dataset. The dataset will be uncompressed in
               this folder.

           - kaldi_ali_tr (type: direcory, optional,
               default: 'None'):
               When set, this is the directiory where the
               kaldi training alignments are stored.
               They will be automatically converted into pkl
               for an easier use within speechbrain.

           - kaldi_ali_dev (type: direcory, optional,
               default: 'None'):
               When set, this is the directiory where the
               kaldi dev alignments are stored.

           - kaldi_ali_te (type: direcory, optional,
               default: 'None'):
               When set, this is the directiory where the
               kaldi test alignments are stored.

           - save_folder (type: str,default: None):
               it the folder where to store the csv files.
               If None, the results will be saved in
               $output_folder/prepare_timit/*.csv.

     Example:    from speechbrain.data_io.data_preparation import timit_prepare

                 local_folder='/home/mirco/datasets/TIMIT'
                 save_folder='exp/TIMIT_exp'

                 # Definition of the config dictionary
                 config={'class_name':'data_processing.copy_data_locally', \
                              'data_folder': local_folder, \
                              'splits':'train,test,dev',
                               'save_folder': save_folder}

                # Initialization of the class
                timit_prepare(config)

     -------------------------------------------------------------------------
     """

    def __init__(
        self,
        data_folder,
        splits,
        kaldi_ali_tr=None,
        kaldi_ali_dev=None,
        kaldi_ali_test=None,
        kaldi_lab_opts=None,
        save_folder=None,
    ):
        # Expected inputs when calling the class (no inputs in this case)
        super().__init__()

        self.data_folder = data_folder
        self.splits = splits
        self.kaldi_ali_tr = kaldi_ali_tr
        self.kaldi_ali_dev = kaldi_ali_dev
        self.kaldi_ali_test = kaldi_ali_test
        self.kaldi_lab_opts = kaldi_lab_opts
        self.save_folder = save_folder

        # Other variables
        self.samplerate = 16000

        # List of test speakers
        self.test_spk = [
            "fdhc0",
            "felc0",
            "fjlm0",
            "fmgd0",
            "fmld0",
            "fnlp0",
            "fpas0",
            "fpkt0",
            "mbpm0",
            "mcmj0",
            "mdab0",
            "mgrt0",
            "mjdh0",
            "mjln0",
            "mjmp0",
            "mklt0",
            "mlll0",
            "mlnt0",
            "mnjm0",
            "mpam0",
            "mtas1",
            "mtls0",
            "mwbt0",
            "mwew0",
        ]

        # List of dev speakers
        self.dev_spk = [
            "fadg0",
            "faks0",
            "fcal1",
            "fcmh0",
            "fdac1",
            "fdms0",
            "fdrw0",
            "fedw0",
            "fgjd0",
            "fjem0",
            "fjmg0",
            "fjsj0",
            "fkms0",
            "fmah0",
            "fmml0",
            "fnmr0",
            "frew0",
            "fsem0",
            "majc0",
            "mbdg0",
            "mbns0",
            "mbwm0",
            "mcsh0",
            "mdlf0",
            "mdls0",
            "mdvc0",
            "mers0",
            "mgjf0",
            "mglb0",
            "mgwt0",
            "mjar0",
            "mjfc0",
            "mjsw0",
            "mmdb1",
            "mmdm2",
            "mmjr0",
            "mmwh0",
            "mpdf0",
            "mrcs0",
            "mreb0",
            "mrjm4",
            "mrjr0",
            "mroa0",
            "mrtk0",
            "mrws1",
            "mtaa0",
            "mtdt0",
            "mteb0",
            "mthc0",
            "mwjg0",
        ]

        # This dictionary is used to conver the 60 phoneme set
        # into the 48 one
        from_60_to_48_phn = {}
        from_60_to_48_phn["sil"] = "sil"
        from_60_to_48_phn["aa"] = "aa"
        from_60_to_48_phn["ae"] = "ae"
        from_60_to_48_phn["ah"] = "ah"
        from_60_to_48_phn["ao"] = "ao"
        from_60_to_48_phn["aw"] = "aw"
        from_60_to_48_phn["ax"] = "ax"
        from_60_to_48_phn["ax-h"] = "ax"
        from_60_to_48_phn["axr"] = "er"
        from_60_to_48_phn["ay"] = "ay"
        from_60_to_48_phn["b"] = "b"
        from_60_to_48_phn["bcl"] = "vcl"
        from_60_to_48_phn["ch"] = "ch"
        from_60_to_48_phn["d"] = "d"
        from_60_to_48_phn["dcl"] = "vcl"
        from_60_to_48_phn["dh"] = "dh"
        from_60_to_48_phn["dx"] = "dx"
        from_60_to_48_phn["eh"] = "eh"
        from_60_to_48_phn["el"] = "el"
        from_60_to_48_phn["em"] = "m"
        from_60_to_48_phn["en"] = "en"
        from_60_to_48_phn["eng"] = "ng"
        from_60_to_48_phn["epi"] = "epi"
        from_60_to_48_phn["er"] = "er"
        from_60_to_48_phn["ey"] = "ey"
        from_60_to_48_phn["f"] = "f"
        from_60_to_48_phn["g"] = "g"
        from_60_to_48_phn["gcl"] = "vcl"
        from_60_to_48_phn["h#"] = "sil"
        from_60_to_48_phn["hh"] = "hh"
        from_60_to_48_phn["hv"] = "hh"
        from_60_to_48_phn["ih"] = "ih"
        from_60_to_48_phn["ix"] = "ix"
        from_60_to_48_phn["iy"] = "iy"
        from_60_to_48_phn["jh"] = "jh"
        from_60_to_48_phn["k"] = "k"
        from_60_to_48_phn["kcl"] = "cl"
        from_60_to_48_phn["l"] = "l"
        from_60_to_48_phn["m"] = "m"
        from_60_to_48_phn["n"] = "n"
        from_60_to_48_phn["ng"] = "ng"
        from_60_to_48_phn["nx"] = "n"
        from_60_to_48_phn["ow"] = "ow"
        from_60_to_48_phn["oy"] = "oy"
        from_60_to_48_phn["p"] = "p"
        from_60_to_48_phn["pau"] = "sil"
        from_60_to_48_phn["pcl"] = "cl"
        from_60_to_48_phn["q"] = "k"
        from_60_to_48_phn["r"] = "r"
        from_60_to_48_phn["s"] = "s"
        from_60_to_48_phn["sh"] = "sh"
        from_60_to_48_phn["t"] = "t"
        from_60_to_48_phn["tcl"] = "cl"
        from_60_to_48_phn["th"] = "th"
        from_60_to_48_phn["uh"] = "uh"
        from_60_to_48_phn["uw"] = "uw"
        from_60_to_48_phn["ux"] = "uw"
        from_60_to_48_phn["v"] = "v"
        from_60_to_48_phn["w"] = "w"
        from_60_to_48_phn["y"] = "y"
        from_60_to_48_phn["z"] = "z"
        from_60_to_48_phn["zh"] = "zh"

        self.from_60_to_48_phn = from_60_to_48_phn

        # Avoid calibration sentences
        self.avoid_sentences = ["sa1", "sa2"]

        # Setting file extension.
        self.extension = [".wav"]

        # Setting the save folder
        if self.save_folder is None:
            self.output_folder = self.global_config["output_folder"]
            self.save_folder = self.output_folder + "/" + funct_name

        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        # Setting ouput files
        self.save_opt = self.save_folder + "/opt_timit_prepare.pkl"
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

        # Additional checks to make sure the data folder contains TIMIT
        self.check_timit_folders()

        msg = "\tCreating csv file for the TIMIT Dataset.."
        logger.debug(msg)

        # Creating csv file for training data
        if "train" in self.splits:
            match_lst = self.extension + ["train"]

            wav_lst_train = get_all_files(
                self.data_folder,
                match_and=match_lst,
                exclude_or=self.avoid_sentences,
            )

            self.create_csv(
                wav_lst_train,
                self.save_csv_train,
                kaldi_lab=self.kaldi_ali_tr,
                kaldi_lab_opts=self.kaldi_lab_opts,
            )

        # Creating csv file for dev data
        if "dev" in self.splits:
            match_lst = self.extension + ["test"]

            wav_lst_dev = get_all_files(
                self.data_folder,
                match_and=match_lst,
                match_or=self.dev_spk,
                exclude_or=self.avoid_sentences,
            )

            self.create_csv(
                wav_lst_dev,
                self.save_csv_dev,
                kaldi_lab=self.kaldi_ali_dev,
                kaldi_lab_opts=self.kaldi_lab_opts,
            )

        # Creating csv file for test data
        if "test" in self.splits:
            match_lst = self.extension + ["test"]

            wav_lst_test = get_all_files(
                self.data_folder,
                match_and=match_lst,
                match_or=self.test_spk,
                exclude_or=self.avoid_sentences,
            )

            self.create_csv(
                wav_lst_test,
                self.save_csv_test,
                kaldi_lab=self.kaldi_ali_test,
                kaldi_lab_opts=self.kaldi_lab_opts,
            )

        # Saving options (useful to skip this phase when already done)
        # save_pkl(self.conf, self.save_opt)

        return

    def skip(self):
        """
         ---------------------------------------------------------------------
          speechbrain.data_io.data_preparation.prepare_timit.skip
          (auth: M. Ravanelli)

         Description: This function detects when the timit data_preparation
                      has been already done and can be skipped.

         Input:        - self (type, prepare_timit class, mandatory)


         Output:      - skip (type: boolean):
                           if True, the preparation phase can be skipped.
                           if False, it must be done.

         Example:    from speechbrain.data_io.data_preparation import (
                        timit_prepare)

                     local_folder='/home/mirco/datasets/TIMIT'
                     save_folder='exp/TIMIT_exp'

                     # Definition of the config dictionary
                     config={'class_name':\
                            'data_processing.copy_data_locally',\
                            'data_folder': local_folder, \
                            'splits':'train,test,dev',
                            'save_folder': save_folder}

                    # Initialization of the class
                    data_prep=timit_prepare(config)

                    # Skip function is True because data_pre has already
                    been done:
                    print(data_prep.skip())

         ---------------------------------------------------------------------
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

    def create_csv(
        self,
        wav_lst,
        csv_file,
        kaldi_lab=None,
        kaldi_lab_opts=None,
        logfile=None,
    ):
        """
         ---------------------------------------------------------------------
          speechbrain.data_io.data_preparation.prepare_timit.create_csv
          (M. Ravanelli)

         Description: This function creates the csv file given a list of wav
                       files.

         Input:        - self (type, prepare_timit class, mandatory)

                       - wav_lst (type: list, mandatory):
                           it is the list of wav files of a given data split.

                       - csv_file (type:file, mandatory):
                           it is the path of the output csv file

                       - kaldi_lab (type:file, default:None):
                           it is the path of the kaldi labels (optional).

                       - kaldi_lab_opts (type:str, default:None):
                           it a string containing the options use to compute
                           the labels.

                       - logfile(type, logger, default: None):
                           it the logger used to write debug and error msgs.


         Output:      None


         Example:   from speechbrain.data_io.data_preparation import (
                        timit_prepare)

                    local_folder='/home/mirco/datasets/TIMIT'
                    save_folder='exp/TIMIT_exp'

                    # Definition of the config dictionary
                    config={'class_name':'data_processing.copy_data_locally',\
                                  'data_folder': local_folder, \
                                  'splits':'train,test,dev',
                                   'save_folder': save_folder}

                   # Initialization of the class
                   data_prep=timit_prepare(config)

                   # Get csv list
                   wav_lst=['/home/mirco/datasets/TIMIT\
                           /train/dr3/mtpg0/sx213.wav',
                           '/home/mirco/datasets/TIMIT\
                           /train/dr3/mtpg0/si2013.wav']

                   csv_file='exp/ex_csv.csv'
                   data_prep.create_csv(wav_lst,csv_file)

         ---------------------------------------------------------------------
         """

        # Adding some Prints
        msg = '\t"Creating csv lists in  %s..."' % (csv_file)
        logger.debug(msg)

        # Reading kaldi labels if needed:
        snt_no_lab = 0
        missing_lab = False

        if kaldi_lab is not None:

            lab = read_kaldi_lab(
                kaldi_lab,
                kaldi_lab_opts,
                logfile=self.global_config["output_folder"] + "/log.log",
            )

            lab_out_dir = self.save_folder + "/kaldi_labels"

            if not os.path.exists(lab_out_dir):
                os.makedirs(lab_out_dir)

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
                "phn",
                "phn_format",
                "phn_opts",
                "wrd",
                "wrd_format",
                "wrd_opts",
            ]
        ]

        if kaldi_lab is not None:
            csv_lines[0].append("kaldi_lab")
            csv_lines[0].append("kaldi_lab_format")
            csv_lines[0].append("kaldi_lab_opts")

        # Processing all the wav files in the list
        for wav_file in wav_lst:

            # Getting sentence and speaker ids
            spk_id = wav_file.split("/")[-2]
            snt_id = wav_file.split("/")[-1].replace(".wav", "")

            snt_id = spk_id + "_" + snt_id

            if kaldi_lab is not None:
                if snt_id not in lab.keys():
                    missing_lab = False

                    msg = (
                        "\tThe sentence %s does not have a corresponding "
                        "kaldi label" % (snt_id)
                    )

                    logger.debug(msg)

                    snt_no_lab = snt_no_lab + 1
                else:
                    snt_lab_path = lab_out_dir + "/" + snt_id + ".pkl"
                    save_pkl(lab[snt_id], snt_lab_path)

                # If too many kaldi labels are missing rise an error
                if snt_no_lab / len(wav_lst) > 0.05:

                    err_msg = (
                        "Too many sentences do not have the "
                        "corresponding kaldi label. Please check data and "
                        "kaldi labels (check %s and %s)."
                        % (self.data_folder, self.kaldi_ali_test)
                    )

                    logger.error(err_msg, exc_info=True)

            if missing_lab:
                continue

            # Reading the signal (to retrieve duration in seconds)
            signal = read_wav_soundfile(wav_file)
            duration = signal.shape[0] / self.samplerate

            # Retrieving words
            wrd_file = wav_file.replace(".wav", ".wrd")
            if not os.path.exists(os.path.dirname(wrd_file)):

                err_msg = "the wrd file %s does not exists!" % (wrd_file)

                logger.error(err_msg, exc_info=True)

            words = [
                line.rstrip("\n").split(" ")[2] for line in open(wrd_file)
            ]

            words = " ".join(words)

            # Retrieving phonemes
            phn_file = wav_file.replace(".wav", ".phn")

            if not os.path.exists(os.path.dirname(phn_file)):

                err_msg = "the wrd file %s does not exists!" % (phn_file)

                logger.error(err_msg, exc_info=True)

            # Phoneme list
            phonemes = []

            for line in open(phn_file):

                phoneme = line.rstrip("\n").replace("h#", "sil").split(" ")[2]

                # From 60 to 48 phonemes
                phoneme = self.from_60_to_48_phn[phoneme]

                # Apping phoneme in the phoneme list
                phonemes.append(phoneme)

            phonemes = " ".join(phonemes)

            # Composition of the csv_line
            csv_line = [
                snt_id,
                str(duration),
                wav_file,
                "wav",
                "",
                spk_id,
                "string",
                "",
                str(phonemes),
                "string",
                "",
                str(words),
                "string",
                "",
            ]

            if kaldi_lab is not None:
                csv_line.append(snt_lab_path)
                csv_line.append("pkl")
                csv_line.append("")

            # Adding this line to the csv_lines list
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

    def check_timit_folders(self):
        """
         ---------------------------------------------------------------------
         speechbrain.data_io.check_timit_folders (author: Mirco Ravanelli)

         Description: This function cheks if the dat folder actually contains
                      the TIMIT dataset. If not, it raises an error.

         Input:        - self (type, prepare_timit class, mandatory)


         Output:      None


         Example:   from speechbrain.data_io.data_preparation import (
                        timit_prepare)

                    local_folder='/home/mirco/datasets/TIMIT'
                    save_folder='exp/TIMIT_exp'

                    # Definition of the config dictionary
                    config={'class_name':'data_processing.copy_data_locally',\
                                  'data_folder': local_folder, \
                                  'splits':'train,test,dev',
                                   'save_folder': save_folder}

                   # Initialization of the class
                   data_prep=timit_prepare(config)

                   # Check folder
                   data_prep.check_timit_folders()

         ---------------------------------------------------------------------
         """

        # Checking test/dr1
        if not os.path.exists(self.data_folder + "/test/dr1"):

            err_msg = (
                "the folder %s does not exist (it is expected in "
                "the TIMIT dataset)" % (self.data_folder + "/test/dr*")
            )

            logger.error(err_msg, exc_info=True)

        # Checking train/dr1
        if not os.path.exists(self.data_folder + "/train/dr1"):

            err_msg = (
                "the folder %s does not exist (it is expected in "
                "the TIMIT dataset)" % (self.data_folder + "/train/dr*")
            )

            logger.error(err_msg, exc_info=True)


class librispeech_prepare:
    """
     -------------------------------------------------------------------------
     speechbrain.data_io.librispeech_prepare (author: Mirco Ravanelli)

     Description: This class prepares the csv files for the LibriSpeech
                  dataset.

     Input (init):  - config (type, dict, mandatory):
                       it is a dictionary containing the keys described below.

                           - data_folder (type: directory, mandatory):
                               it the folder where the original TIMIT dataset
                               is stored.

                           - splits ('dev-clean','dev-others','test-clean','
                             'test-others','train-clean-100',
                             'train-clean-360', 'train-other-500',mandatory):
                               it contains the list of splits for which we are
                               going to create the corresponding csv file.

                           - select_n_sentences (type:int,opt,
                             default:'None'0):
                               it is the number of sentences to select.
                               It might be useful for debugging when I want to
                               test the script on a reduced number of
                               sentences only.

                           - save_folder (type: str,default: None):
                               it the folder where to store the csv files.
                               If None, the results will be saved in
                               $output_folder/prepare_librispeech/*.csv.

                   - funct_name (type, str, default: None):
                       it is a string containing the name of the parent
                       function that has called this method.

                   - global_config (type, dict, default: None):
                       it a dictionary containing the global variables of the
                       parent config file.

                   - logger (type, logger, default: None):
                       it the logger used to write debug and error messages.
                       If logger=None and root_cfg=True, the file is created
                       from scratch.

                   - first_input (type, list, default: None)
                      this variable allows users to analyze the first input
                      given when calling the class for the first time.


     Input (call): - inp_lst(type, list, mandatory):
                       by default the input arguments are passed with a list.
                       In this case, the list is empty. The call function is
                       just a dummy function here because all the meaningful
                       computation must be executed only once and they are
                       thus done in the initialization method only.


     Output (call):  - stop_at_lst (type: list):
                       when stop_at is set, it returns the stop_at in a list.
                       Otherwise it returns None. It this case it returns
                       always None.

     Example:    from speechbrain.data_io.data_preparation import (
                    librispeech_prepare)

                 local_folder='/home/mirco/datasets/LibriSpeech'
                 save_folder='exp/LibriSpeech_exp'

                 # Definition of the config dictionary
                 config={'class_name':'data_processing.copy_data_locally', \
                              'data_folder': local_folder, \
                              'splits':'dev-clean,test-clean',
                               'save_folder': save_folder}

                # Running the data preparation
                librispeech_prepare(config)

     -------------------------------------------------------------------------
     """

    def __init__(
        self,
        config,
        funct_name=None,
        global_config=None,
        logger=None,
        first_input=None,
        functions=None,
    ):

        # Logger setup
        self.logger = logger

        # Here are summarized the expected options for this class
        self.expected_options = {
            "data_folder": ("directory", "mandatory"),
            "splits": (
                "one_of_list(dev-clean,dev-others,test-clean,test-others,"
                + "train-clean-100,train-clean-360,train-other-500)",
                "mandatory",
            ),
            "save_folder": ("str", "optional", "None"),
            "select_n_sentences": ("int_list(1,inf)", "optional", "None"),
        }

        # Check, cast , and expand the options
        self.conf = check_opts(
            self, self.expected_options, config, logger=self.logger
        )

        # Expected input
        self.expected_inputs = []

        # Check the first input
        check_inputs(
            self.conf, self.expected_inputs, first_input, logger=self.logger
        )

        # Other variables
        self.samplerate = 16000
        self.funct_name = funct_name

        # Saving folder
        if self.save_folder is None:
            self.output_folder = global_config["output_folder"]
            self.save_folder = self.output_folder + "/" + funct_name

        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        self.save_opt = self.save_folder + "/opt_librispeech_prepare.pkl"

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

    def __call__(self, inp):
        return

    def create_csv(self, wav_lst, text_dict, split, select_n_sentences):
        """
         ---------------------------------------------------------------------
         speechbrain.data_io.prepare_librispeech.create_csv
         (author: Mirco Ravanelli)

         Description: This function creates the csv file given a list of wav
                      files.

         Input:        - self (type, prepare_librispeecg class, mandatory)

                       - wav_lst (type: list, mandatory):
                           it is the list of wav files of a given data split.

                       - text_dict (type: list, mandatory):
                           it is the a dictionary containing the text of each
                           sentence.

                       - split (type: str, mandatory):
                           it is the name of the current data split.

                       - select_n_sentences (type:int,opt, default:'None'0):
                           it is the number of sentences to select.

         Output:      None


         Example:   from speechbrain.data_io.data_preparation import \
             librispeech_prepare

                    local_folder='/home/mirco/datasets/LibriSpeech'
                    save_folder='exp/LibriSpeech_exp'

                    # Definition of the config dictionary
                    config={'class_name':'data_processing.copy_data_locally',\
                                  'data_folder': local_folder, \
                                  'splits':'train-clean-100',
                                   'save_folder': save_folder}

                   # Initialization of the class
                   data_prep=librispeech_prepare(config)

                   # Get csv list
                   wav_lst=['/home/mirco/datasets/LibriSpeech/dev-clean/84\
                   /121123/84-121123-0000.flac']

                   text_dict={'84-121123-0000':'Hello world'}

                   split='debug_split'

                   select_n_sentences=1

                   data_prep.create_csv(wav_lst,text_dict,split,
                   select_n_sentences)

                   # take a look into exp/LibriSpeech_exp

         ---------------------------------------------------------------------
         """

        # Setting path for the csv file
        csv_file = self.save_folder + "/" + split + ".csv"

        # Preliminary prints
        msg = "\tCreating csv lists in  %s..." % (csv_file)
        logger.debug(msg)

        csv_lines = []
        snt_cnt = 0

        # Processing all the wav files in wav_lst
        for wav_file in wav_lst:

            snt_id = wav_file.split("/")[-1].replace(".flac", "")
            spk_id = "-".join(snt_id.split("-")[0:2])
            wrd = text_dict[snt_id]

            signal = read_wav_soundfile(wav_file)
            duration = signal.shape[0] / self.samplerate

            # Composing the csv file
            csv_line = (
                "ID="
                + snt_id
                + " duration="
                + str(duration)
                + " wav=("
                + wav_file
                + ",flac)"
                + " spk_id=("
                + spk_id
                + ",string)"
                + " wrd=("
                + wrd
                + ",string)"
            )

            #  Appending current file to the csv_lines list
            csv_lines.append(csv_line)
            snt_cnt = snt_cnt + 1

            if snt_cnt == select_n_sentences:
                break

        # Writing the csv_lines
        write_txt_file(csv_lines, csv_file)

        # Final print
        msg = "\t%s sucessfully created!" % (csv_file)
        logger.debug(msg)

    def skip(self):
        """
         ---------------------------------------------------------------------
         speechbrain.data_io.prepare_librispeech.skip (author: Mirco Ravanelli)

         Description: This function detects when the librispeeh data_prep
                       has been already done and can be skipped.

         Input:        - self (type, prepare_timit class, mandatory)


         Output:      - skip (type: boolean):
                           if True, the preparation phase can be skipped.
                           if False, it must be done.

         Example:    from speechbrain.data_io.data_preparation import \
             librispeech_prepare

                     local_folder='/home/mirco/datasets/LibriSpeech'
                     save_folder='exp/LibriSpeech_exp'

                     # Definition of the config dictionary
                     config={'class_name':\
                             'data_processing.copy_data_locally', \
                             'data_folder': local_folder, \
                             'splits':'dev-clean,test-clean',
                             'save_folder': save_folder}

                    # Running the data preparation
                    data_prep=librispeech_prepare(config)

                    # Skip function is True because data_pre has already been
                    # done:
                    print(data_prep.skip())

         ---------------------------------------------------------------------
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
        return skip

    @staticmethod
    def text_to_dict(text_lst):
        """
         ---------------------------------------------------------------------
         speechbrain.data_io.data_preparation.prepare_librispeech.text_to_dict
         (author: Mirco Ravanelli)

         Description: This converts lines of text into a dictionary-

         Input:        - self (type: prepare_timit class, mandatory)

                       text_lst (type: file, mandatory):
                        it is the file containing  the librispeech text
                        transcription.

         Output:      text_dict (type: dictionary)
                           it is the dictionary containing the text
                           transcriptions for each sentence.

         Example:    from speechbrain.data_io.data_preparation import \
             librispeech_prepare

                     local_folder='/home/mirco/datasets/LibriSpeech'
                     save_folder='exp/LibriSpeech_exp'

                     # Definition of the config dictionary
                     config={'class_name':
                             'data_processing.copy_data_locally', \
                             'data_folder': local_folder, \
                             'splits':'dev-clean,test-clean',
                             'save_folder': save_folder}

                    # Running the data preparation
                    data_prep=librispeech_prepare(config)

                    # Text dictionary creation
                    text_lst=['/home/mirco/datasets/LibriSpeech/dev-clean\
                    /84/121550/84-121550.trans.txt']

                    print(data_prep.text_to_dict(text_lst))

         ---------------------------------------------------------------------
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
         ---------------------------------------------------------------------
         speechbrain.data_io.data_preparation.check_librispeech_folders
         (M. Ravanelli)

         Description: This function cheks if the dat folder actually contains
                      the LibriSpeech dataset. If not, it raises an error.

         Input:        - self (type, prepare_librispeech class, mandatory)


         Output:      None


         Example:    from speechbrain.data_io.data_preparation import \
             librispeech_prepare

                     local_folder='/home/mirco/datasets/LibriSpeech'
                     save_folder='exp/LibriSpeech_exp'

                     # Definition of the config dictionary
                     config={'class_name':
                             'data_processing.copy_data_locally', \
                             'data_folder': local_folder, \
                             'splits':'dev-clean,test-clean',
                              'save_folder': save_folder}

                    # Running the data preparation
                    data_prep=librispeech_prepare(config)

                    # Check folder
                    data_prep.check_librispeech_folders()

         ---------------------------------------------------------------------
         """

        # Checking if all the splits exist
        for split in self.splits:
            if not os.path.exists(self.data_folder + "/" + split):

                err_msg = (
                    "the folder %s does not exist (it is expected in the "
                    "Librispeech dataset)" % (self.data_folder + "/" + split)
                )

                logger.error(err_msg, exc_info=True)
