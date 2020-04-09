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
from speechbrain.utils.input_validation import check_opts, check_inputs
from speechbrain.utils.data_utils import get_all_files
from speechbrain.utils.logger import logger_write
from speechbrain.utils.superpowers import run_shell

from speechbrain.data_io.data_io import (
    read_wav_soundfile,
    load_pkl,
    save_pkl,
    read_kaldi_lab,
    write_txt_file,
)


from speechbrain.data_io.data_preparation import copy_data_locally



## remove kaldi, and copy_data_locally can be imported...


class Voxceleb_prepare:
    """
     -------------------------------------------------------------------------
     speechbrain.data_io.data_preparation_voxceleb.Voxceleb_prepare
     (author: Nauman Dawalatabad)

     Description: This class prepares the csv files for the Voxceleb1 dataset.

     Input (init):  - config (type, dict, mandatory):
                       it is a dictionary containing the keys described below.

                           - data_folder (type: directory, mandatory):
                               it the folder where the original TIMIT dataset
                               is stored.

                           - splits ('train','dev','test',mandatory):
                               it the local directory where to store the
                               dataset. The dataset will be uncompressed in
                               this folder.

                           - save_folder (type: str,optional, default: None):
                               it the folder where to store the csv files.
                               If None, the results will be saved in
                               $output_folder/prepare_timit/*.csv.

                   - funct_name (type, str, optional, default: None):
                       it is a string containing the name of the parent
                       function that has called this method.

                   - global_config (type, dict, optional, default: None):
                       it a dictionary containing the global variables of the
                       parent config file.

                   - logger (type, logger, optional, default: None):
                       it the logger used to write debug and error messages.
                       If logger=None and root_cfg=True, the file is created
                       from scratch.

                   - first_input (type, list, optional, default: None)
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
        config,
        funct_name=None,
        global_config=None,
        functions=None,
        logger=None,
        first_input=None,
    ):

        self.logger = logger

        # Here are summarized the expected options for this class
        # Note: Kaldi ali not applicable here (None)
        self.expected_options = {
            "class_name": ("str", "mandatory"),
            "data_folder": ("directory", "mandatory"),
            "splits": ("one_of_list(train,dev,test)", "mandatory"),
            "kaldi_ali_tr": ("directory", "optional", "None"),
            "kaldi_ali_dev": ("directory", "optional", "None"),
            "kaldi_ali_test": ("directory", "optional", "None"),
            "kaldi_lab_opts": ("str", "optional", "None"),
            "save_folder": ("str", "optional", "None"),
        }

        # Check, cast , and expand the options
        self.conf = check_opts(
            self, self.expected_options, config, logger=self.logger
        )

        # Expected inputs when calling the class (no inputs in this case)
        self.expected_inputs = []

        # Check the first input
        check_inputs(
            self.conf, self.expected_inputs, first_input, logger=self.logger
        )

        # Other variables
        self.global_config = global_config
        self.samplerate = 16000

        # ND
        wav_lst_train, wav_lst_dev, wav_lst_test  = self.prepare_wav_list(self.data_folder)


        # Setting the save folder
        if self.save_folder is None:
            self.output_folder = self.global_config["output_folder"]
            self.save_folder = self.output_folder + "/" + funct_name

        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        # Setting ouput files
        self.save_opt = self.save_folder + "/opt_voxceleb_prepare.pkl"
        self.save_csv_train = self.save_folder + "/train.csv"
        self.save_csv_dev = self.save_folder + "/dev.csv"
        self.save_csv_test = self.save_folder + "/test.csv"

        # Check if this phase is already done (if so, skip it)
        if self.skip():   

            msg = "\t%s sucessfully created!" % (self.save_csv_train)
            logger_write(msg, logfile=self.logger, level="debug")

            msg = "\t%s sucessfully created!" % (self.save_csv_dev)
            logger_write(msg, logfile=self.logger, level="debug")

            msg = "\t%s sucessfully created!" % (self.save_csv_test)
            logger_write(msg, logfile=self.logger, level="debug")

            return

        # Additional checks to make sure the data folder contains TIMIT
        self.check_voxceleb1_folders()

        msg = "\tCreating csv file for the Voxceleb Dataset.."
        logger_write(msg, logfile=self.logger, level="debug")

        # Creating csv file for training data
        if "train" in self.splits:
            self.prepare_csv(
                wav_lst_train,
                self.save_csv_train,
                kaldi_lab=self.kaldi_ali_tr, 
                kaldi_lab_opts=self.kaldi_lab_opts,
                logfile=self.logger,
            )

        if "dev" in self.splits:
            self.prepare_csv(
                wav_lst_dev,
                self.save_csv_dev,
                kaldi_lab=self.kaldi_ali_dev, 
                kaldi_lab_opts=self.kaldi_lab_opts,
                logfile=self.logger,
            )

        if "test" in self.splits:
            self.prepare_csv(
                wav_lst_test,
                self.save_csv_test,
                kaldi_lab=self.kaldi_ali_tr, 
                kaldi_lab_opts=self.kaldi_lab_opts,
                logfile=self.logger,
            )

        # Saving options (useful to skip this phase when already done)
        save_pkl(self.conf, self.save_opt)

    def __call__(self, inp):
        return

    def prepare_wav_list(self, data_folder):
        iden_split_file = os.path.join(data_folder, 'iden_split_sample.txt')
        train_wav_lst = [] 
        dev_wav_lst = [] 
        test_wav_lst = [] 
        with open(iden_split_file, 'r') as f:
            for line in f:
                [spkr_split, audio_path] = line.split(' ')
                [spkr_id, session_id, utt_id] = audio_path.split('/')
                if spkr_split not in ['1', '2', '3']:
                    # todo: raise an error here!
                    break
                if spkr_split == '1':
                    train_wav_lst.append(os.path.join(data_folder,audio_path.strip()))
                if spkr_split == '2':
                    dev_wav_lst.append(os.path.join(data_folder,audio_path.strip()))
                if spkr_split == '3':
                    test_wav_lst.append(os.path.join(data_folder,audio_path.strip()))
            #print ("TRAIN_LST below....: ")    
            #print (train_wav_lst)
        f.close()
        return train_wav_lst, dev_wav_lst, test_wav_lst



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
            opts_old = load_pkl(self.save_opt)
            if opts_old == self.conf:
                skip = True

        return skip

    def prepare_csv(
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
          (Nauman Dawalatabad)

         Description: This function creates the csv file given a list of wav
                       files.

         Input:        - self (type, prepare_timit class, mandatory)

                       - wav_lst (type: list, mandatory):
                           it is the list of wav files of a given data split.

                       - csv_file (type:file, mandatory):
                           it is the path of the output csv file

                       - kaldi_lab (type:file, optional, default:None):
                           it is the path of the kaldi labels (optional).

                       - kaldi_lab_opts (type:str, optional, default:None):
                           it a string containing the options use to compute
                           the labels.

                       - logfile(type, logger, optional, default: None):
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
        logger_write(msg, logfile=self.logger, level="debug")

        # Reading kaldi labels if needed:
        snt_no_lab = 0
        missing_lab = False
        '''
        if kaldi_lab is not None:

            lab = read_kaldi_lab(
                kaldi_lab,
                kaldi_lab_opts,
                logfile=self.global_config["output_folder"] + "/log.log",
            )

            lab_out_dir = self.save_folder + "/kaldi_labels"

            if not os.path.exists(lab_out_dir):
                os.makedirs(lab_out_dir)
        '''
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

        '''
        if kaldi_lab is not None:
            csv_lines[0].append("kaldi_lab")
            csv_lines[0].append("kaldi_lab_format")
            csv_lines[0].append("kaldi_lab_opts")
        '''

        my_sep = '---'
        # Processing all the wav files in the list
        for wav_file in wav_lst:

            # Getting sentence and speaker ids
            [spk_id, sess_id, utt_id] = wav_file.split('/')[-3:]
            uniq_utt_id = my_sep.join([spk_id, sess_id, utt_id.split('.')[0]])
            #spk_id = wav_file.split("/")[-2]
            #snt_id = wav_file.split("/")[-1].replace(".wav", "")

            # Reading the signal (to retrieve duration in seconds)
            signal = read_wav_soundfile(wav_file, logger=logfile)
            duration = signal.shape[0] / self.samplerate

            # Composition of the csv_line
            csv_line = [
                uniq_utt_id,
                str(duration),
                wav_file,
                "wav",
                " ",
                spk_id,
                "string",
                " ",

            ]

            #if kaldi_lab is not None:
            #    csv_line.append(snt_lab_path)
            #    csv_line.append("pkl")
            #    csv_line.append("")

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
        logger_write(msg, logfile=self.logger, level="debug")




    def check_voxceleb1_folders(self):
        """
         ---------------------------------------------------------------------
         speechbrain.data_io.check_voxceleb1_folders (author: Nauman Dawalatabad)

         Description: This function cheks if the dat folder actually contains
                      the Voxceleb1 dataset. If not, it raises an error.

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
        if not os.path.exists(self.data_folder + "/id10001"):

            err_msg = (
                "the folder %s does not exist (it is expected in "
                "the Voxceleb dataset)" % (self.data_folder + "/id*")
            )

            logger_write(err_msg, logfile=self.logger)





