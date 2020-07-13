"""
Miscellaneous decoders and decoder utility

Authors
 * Mirco Ravanelli 2020
"""
import os
import torch
import logging
import threading
from shutil import copyfile
from speechbrain.utils.superpowers import run_shell
from speechbrain.utils.data_utils import get_all_files, split_list

logger = logging.getLogger(__name__)


def undo_padding(batch, lengths):
    """Produces Python lists with ragged edges
    """
    batch_max_len = batch.shape[1]
    as_list = []
    for seq, seq_length in zip(batch, lengths):
        actual_size = int(torch.round(seq_length * batch_max_len))
        seq_true = seq.narrow(0, 0, actual_size)
        as_list.append(seq_true.tolist())
    return as_list


class kaldi_decoder:
    """Manages decoding using the kaldi decoder.

    Arguments
    ---------
    decoding_script_folder : str
        folder where the kaldi decoding script are saved.
    decoding_script : str
        decoding script (within the decoding_script_folder) to use for
        decoding (e.g, decode_dnn.sh)
    graphdir : str
        folder containing the kaldi graph to use for decoding
        (e.g. $kaldi_folder/graph)
    alidir : str
        folder containing the alignments.
        (e.g. $kaldi_folder/dnn4_pretrain-dbn_dnn_ali_test)
    datadir : str
        folder containing data transcriptions (e.g. $kaldi_folder/data/test)
    posterior_folder : str
        folder where the neural network posterior probabilities are saved.
    save_folder : str
        it the folder where to save the decoding results.
    min_active : int
        Decoder minimum #active states.
    max_active : int
        Decoder maxmium #active states.
    beam : float
        Decoding beam. Larger->slower, more accurate.
    lat_beam : float
        Lattice generation beam. Larger->slower, and deeper lattices
    acwt : float
        Scaling factor for acoustic likelihoods
    max_arcs : int
        Maximum number of arcs
    scoring : bool
        If True, the scoring script based on sclite is used.
    scoring_script : str
        scoring script used to compute the final score (e.g.
        tools/kaldi_decoder/local/score.sh)
    scoring_opt : str
        options give to the scoring script (e.g, --min_lmwt 1 --max_lmwt 10)
    norm_vars : bool
        If True, variace are normalized.
    num_jobs : int
        number of decoding jobs to run in parallel.
    """

    def __init__(
        self,
        decoding_script_folder,
        decoding_script,
        graphdir,
        alidir,
        datadir,
        posterior_folder,
        save_folder,
        min_active=200,
        max_active=7000,
        max_mem=50000000,
        beam=13.0,
        lat_beam=8.0,
        acwt=0.2,
        max_arcs=-1,
        scoring=True,
        scoring_script=None,
        scoring_opts="--min-lmwt 1 --maxlmwt 10",
        norm_vars=False,
        num_job=8,
    ):
        self.decoding_script = decoding_script
        self.min_active = min_active
        self.max_active = max_active
        self.max_mem = max_mem
        self.beam = beam
        self.lat_beam = lat_beam
        self.acwt = acwt
        self.max_arcs = max_arcs
        self.scoring = scoring
        self.scoring_opts = scoring_opts
        self.norm_vars = norm_vars
        self.num_job = num_job

        # getting absolute paths
        self.decoding_script_folder = os.path.abspath(decoding_script_folder)
        self.graphdir = os.path.abspath(graphdir)
        self.alidir = os.path.abspath(alidir)
        self.datadir = os.path.abspath(datadir)
        self.posterior_folder = os.path.abspath(posterior_folder)
        self.save_folder = os.path.abspath(save_folder)

        if self.scoring:
            self.scoring_script = os.path.abspath(scoring_script)
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
                    target=self._decode_sentence, args=(ark_file, cnt)
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
            run_shell(scoring_cmd)

            # Print scoring results()
            self.print_results()

    def __call__(self, inp_lst):
        return

    def _decode_sentence(self, ark_file, cnt):
        """Run a decoding job
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
        run_shell(dec_cmd)

    def print_results(self):
        """Print the final performance on the logger.
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
                                if "SPKR" in line and cnt == 1:
                                    logger.info(line)
                                if "Mean" in line:
                                    line = line.replace(
                                        " Mean ", subfolder.split("/")[-1]
                                    )
                                    logger.info(line)

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

        logger.info("\nBEST ERROR RATE: %f\n" % (min(errors)))
