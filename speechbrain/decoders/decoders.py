import os
import threading
from shutil import copyfile
import torch
from speechbrain.utils.superpowers import run_shell
from speechbrain.utils.data_utils import get_all_files, split_list


def undo_padding(batch, lengths):
    # Produces Python lists
    batch_max_len = batch.shape[1]
    as_list = []
    for seq, seq_length in zip(batch, lengths):
        actual_size = int(torch.round(seq_length * batch_max_len))
        seq_true = seq.narrow(0, 0, actual_size)
        as_list.append(seq_true.tolist())
    return as_list


class kaldi_decoder:
    """
     -------------------------------------------------------------------------
     speechbrain.decoders.decoder.kaldi_decoder (author: Mirco Ravanelli)

     Description: This class manages decoding using the kaldi decoder.

     Input (init):  - config (type, dict, mandatory):
                       it is a dictionary containing the keys described below.


                           - decoding_script_folder (type: directory,\
                               mandatory):
                               it the folder where the kaldi decoding script
                               are saved (e.g, tools/kaldi_decoder)

                           - decoding_script (type: str, mandatory):
                               it the folder decoding script (within the
                               decoding_script_folder) to use for decoding
                               (e.g, decode_dnn.sh)

                           - graphdir (type: directory, mandatory):
                               it the folder containing the kaldi graph
                               to use for decoding (e.g.,$kaldi_folder/graph )

                            - alidir (type: directory, mandatory):
                               it the folder containing the alignments.
                               (e.g., $kaldi_folder/\
                                   dnn4_pretrain-dbn_dnn_ali_test)

                            - datadir (type: directory, mandatory):
                               it the folder containing data transcriptions
                               (e.g,$kaldi_folder/data/test )

                            - posterior_folder (type: directory, mandatory):
                               it the folder where the neural network
                               posterior probabilities are saved.

                            - save_folder (type: directory, mandatory):
                               it the folder where to save the decoding
                               results.

                            - min_active (type: int(1,inf), optional,\
                                Def: 200):
                                Decoder minimum #active states.

                            - max_active (type: int(1,inf), optional, \
                                Def: 7000):
                                Decoder maxmium #active states.

                            - beam (type: float(0,inf), optional, Def: 13.0):
                                Decoding beam.  Larger->slower, more accurate.

                            - lat_beam (type: float(0,inf), optional, \
                                Def: 13.0):
                                Lattice generation beam.
                                Larger->slower, and deeper lattices

                            - acwt (type: float(0,inf), optional, Def: 0.2):
                                Scaling factor for acoustic likelihoods

                            - max_arcs (type: int, optional, Def: -1):

                            - scoring (type: bool, optional, Def: True):
                                If True, the scoring script based on sclite
                                is used.

                           - scoring_script (type: path, optional, Def:None):
                               it the scoring script used to compute the
                               final score (e.g.
                               tools/kaldi_decoder/local/score.sh)

                          - scoring_opt (type: str, optional, Def: None):
                              are the options give to the scoring script
                              (e.g, --min_lmwt 1 --max_lmwt 10)

                          -norm_vars (type: bool, optional, Def: False):
                                If True, variace are normalized.

                           - num_jobs (type: int(1,inf), optional, Def: 8):
                                it is the number of decoding jobs to run
                                in parallel.


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

        # Setting logger and exec_config
        self.logger = logger

        self.output_folder = global_config["output_folder"]

        # Definition of the expected options
        self.expected_options = {
            "class_name": ("str", "mandatory"),
            "decoding_script_folder": ("directory", "mandatory"),
            "decoding_script": ("file", "mandatory"),
            "graphdir": ("directory", "mandatory"),
            "alidir": ("directory", "mandatory"),
            "datadir": ("directory", "mandatory"),
            "posterior_folder": ("directory", "mandatory"),
            "save_folder": ("str", "optional", "None"),
            "min_active": ("int(1,inf)", "optional", "200"),
            "max_active": ("int(1,inf)", "optional", "7000"),
            "max_mem": ("int(1,inf)", "optional", "50000000"),
            "beam": ("float(0,inf)", "optional", "13.0"),
            "lat_beam": ("float(0,inf)", "optional", "8.0"),
            "acwt": ("float(0,inf)", "optional", "0.2"),
            "max_arcs": ("int", "optional", "-1"),
            "scoring": ("bool", "optional", "True"),
            "scoring_script": ("file", "optional", "None"),
            "scoring_opts": ("str", "optional", "--min-lmwt 1 --max-lmwt 10"),
            "norm_vars": ("bool", "optional", "False"),
            "num_job": ("int(1,inf)", "optional", "8"),
        }

        # FIX: Old style
        # Check, cast , and expand the options
        # self.conf = check_opts(self, self.expected_options, config, self.logger)

        # Expected inputs when calling the class
        self.expected_inputs = []

        # Check the first input
        # check_inputs(
        #     self.conf, self.expected_inputs, first_input, logger=self.logger
        # )

        # Setting the save folder
        if self.save_folder is None:
            self.save_folder = self.output_folder + "/" + funct_name
            if not os.path.exists(self.save_folder):
                os.makedirs(self.save_folder)
                os.makedirs(self.save_folder + "/log")

        # getting absolute paths
        self.save_folder = os.path.abspath(self.save_folder)
        self.graphdir = os.path.abspath(self.graphdir)
        self.alidir = os.path.abspath(self.alidir)

        if self.scoring:
            self.scoring_script = os.path.abspath(self.scoring_script)
        # Reading all the ark files in the posterior_folder
        ark_files = get_all_files(self.posterior_folder, match_and=[".ark"],)

        # Sorting files (processing sentences of comparable size can make
        # multithreading much more efficient)
        ark_files.sort(key=lambda f: os.stat(f).st_size, reverse=False)

        # Deciding which sentences decoding in parallel
        N_chunks = int(len(ark_files) / self.num_job)
        ark_lists = split_list(ark_files, N_chunks)

        cnt = 1

        # Manage multi-thread decoding
        for ark_list in ark_lists:

            threads = []

            for ark_file in ark_list:

                t = threading.Thread(
                    target=self.decode_sentence, args=(ark_file, cnt)
                )
                threads.append(t)
                t.start()

                # Updating the sentence counter
                cnt = cnt + 1

            for t in threads:
                t.join()

        if self.scoring:

            # copy final model as expected by the kaldi scoring algorithm
            copyfile(
                self.alidir + "/final.mdl", self.output_folder + "/final.mdl"
            )

            scoring_cmd = "cd tools/kaldi_decoder/; %s %s %s %s %s" % (
                self.scoring_script,
                self.scoring_opts,
                self.datadir,
                self.graphdir,
                self.save_folder,
            )

            # Running the scoring command
            run_shell(scoring_cmd, logger=self.logger)

            # Print scoring results()
            self.print_results()

    def __call__(self, inp_lst):
        return

    def decode_sentence(self, ark_file, cnt):
        """
         ---------------------------------------------------------------------
         speechbrain.decoders.decoders.decode_sentence
         (author: Mirco Ravanelli)

         Description: This function runs a decoding job.

         ---------------------------------------------------------------------
        """
        # Getting the absolute path
        ark_file = os.path.abspath(ark_file)

        # Composing the decoding command
        dec_cmd = (
            "latgen-faster-mapped --min-active=%i --max-active=%i "
            "--max-mem=%i --beam=%f --lattice-beam=%f "
            "--acoustic-scale=%f --allow-partial=true "
            "--word-symbol-table=%s/words.txt %s/final.mdl %s/HCLG.fst "
            '"ark,s,cs: cat %s |" '
            '"ark:|gzip -c > %s/lat.%i.gz"'
            % (
                self.min_active,
                self.max_active,
                self.max_mem,
                self.beam,
                self.lat_beam,
                self.acwt,
                self.graphdir,
                self.alidir,
                self.graphdir,
                ark_file,
                self.save_folder,
                cnt,
            )
        )

        # Running the command
        run_shell(dec_cmd, logger=self.logger)

    def print_results(self):
        """
         ---------------------------------------------------------------------
         speechbrain.decoders.decoders.decode_sentence
         (author: Mirco Ravanelli)

         Description: This print the final performance on the logger.

         ---------------------------------------------------------------------
        """

        # Print the results (change it for librispeech scoring)
        subfolders = [
            f.path for f in os.scandir(self.save_folder) if f.is_dir()
        ]

        errors = []

        for subfolder in subfolders:
            if "score_" in subfolder:
                files = os.listdir(subfolder)

                for file in files:
                    if ".sys" in file:
                        with open(subfolder + "/" + file) as fp:
                            line = fp.readline()
                            cnt = 1
                            while line:
                                # FIX: logger_write; re-enable if
                                # if "SPKR" in line and cnt == 1:
                                #     logger_write(
                                #         line, logfile=self.logger, level="info"
                                #     )
                                if "Mean" in line:
                                    line = line.replace(
                                        " Mean ", subfolder.split("/")[-1]
                                    )
                                    # FIX: logger_write
                                    # logger_write(
                                    #     line, logfile=self.logger, level="info"
                                    # )

                                    line = (
                                        line.replace("  ", " ")
                                        .replace("  ", " ")
                                        .replace("  ", " ")
                                        .replace("  ", " ")
                                    )
                                    errors.append(
                                        float(
                                            line.split("|")[-3].split(" ")[-3]
                                        )
                                    )

                                line = fp.readline()
                                cnt += 1

        # FIX: logger_write
        # logger_write(
        #     "\nBEST ERROR RATE: %f\n" % (min(errors)),
        #     logfile=self.logger,
        #     level="info",
        # )
