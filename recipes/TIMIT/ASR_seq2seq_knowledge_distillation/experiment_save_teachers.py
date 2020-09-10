#!/usr/bin/env python3
import os
import sys
import torch
import speechbrain as sb
from tqdm.contrib import tqdm
import h5py
import numpy as np

from speechbrain.data_io.data_io import prepend_bos_token
import speechbrain.utils.edit_distance as edit_distance
from speechbrain.data_io.data_io import convert_index_to_lab
from speechbrain.decoders.ctc import ctc_greedy_decode
from speechbrain.decoders.decoders import undo_padding
from speechbrain.decoders.seq2seq import batch_filter_seq2seq_output

# This hack needed to import data preparation script from ..
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
from timit_prepare import prepare_timit  # noqa E402

# Load hyperparameters file with command-line overrides
params_file, overrides = sb.core.parse_arguments(sys.argv[1:])
with open(params_file) as fin:
    params = sb.yaml.load_extended_yaml(fin, overrides)

# Create experiment directory
sb.core.create_experiment_directory(
    experiment_directory=params.output_folder, overrides=overrides,
)

# initialise teacher model variables
tea_enc_list = [
    params.tea0_enc,
    params.tea1_enc,
    params.tea2_enc,
    params.tea3_enc,
    params.tea4_enc,
    params.tea5_enc,
    params.tea6_enc,
    params.tea7_enc,
    params.tea8_enc,
    params.tea9_enc,
]
tea_emb_list = [
    params.tea0_emb,
    params.tea1_emb,
    params.tea2_emb,
    params.tea3_emb,
    params.tea4_emb,
    params.tea5_emb,
    params.tea6_emb,
    params.tea7_emb,
    params.tea8_emb,
    params.tea9_emb,
]
tea_dec_list = [
    params.tea0_dec,
    params.tea1_dec,
    params.tea2_dec,
    params.tea3_dec,
    params.tea4_dec,
    params.tea5_dec,
    params.tea6_dec,
    params.tea7_dec,
    params.tea8_dec,
    params.tea9_dec,
]
tea_ctc_lin_list = [
    params.tea0_ctc_lin,
    params.tea1_ctc_lin,
    params.tea2_ctc_lin,
    params.tea3_ctc_lin,
    params.tea4_ctc_lin,
    params.tea5_ctc_lin,
    params.tea6_ctc_lin,
    params.tea7_ctc_lin,
    params.tea8_ctc_lin,
    params.tea9_ctc_lin,
]
tea_seq_lin_list = [
    params.tea0_seq_lin,
    params.tea1_seq_lin,
    params.tea2_seq_lin,
    params.tea3_seq_lin,
    params.tea4_seq_lin,
    params.tea5_seq_lin,
    params.tea6_seq_lin,
    params.tea7_seq_lin,
    params.tea8_seq_lin,
    params.tea9_seq_lin,
]


for i in range(params.num_tea):
    exec(
        "tea{}_modules = torch.nn.ModuleList([tea_enc_list[i], tea_emb_list[i], tea_dec_list[i], tea_ctc_lin_list[i], tea_seq_lin_list[i]])".format(
            i
        )
    )  # i denotes the index of teacher models

tea_modules_list = []
for i in range(params.num_tea):
    exec("tea_modules_list.append(tea{}_modules)".format(i))


def exclude_wer(wer):
    """
    This function is used to exclude the
    wer values which is more than 100.
    """
    wer_list = []
    for item in wer:
        if item > 100:
            item = 100
        wer_list.append(item)
    return np.array(wer_list)


# Define training procedure
class ASR(sb.core.Brain):
    def __init__(
        self, tea_modules_list=None, first_inputs=None,
    ):

        # Initialize teacher parameters
        tea_modules_list_ = []
        for tea_modules in tea_modules_list:
            tea_modules_ = torch.nn.ModuleList(tea_modules)
            tea_modules_list_.append(tea_modules_)
        self.tea_modules_list = tea_modules_list_

        if first_inputs is not None:
            self.compute_forward_tea(*first_inputs, init_params=True)

    def compute_forward_tea(self, x, y, init_params=False):
        ids, wavs, wav_lens = x
        ids, phns, phn_lens = y

        wavs, wav_lens = wavs.to(params.device), wav_lens.to(params.device)
        phns, phn_lens = phns.to(params.device), phn_lens.to(params.device)

        if hasattr(params, "augmentation"):
            wavs = params.augmentation(wavs, wav_lens, init_params)
        feats = params.compute_features(wavs, init_params)
        feats = params.normalize(feats, wav_lens)
        apply_softmax = torch.nn.Softmax(dim=-1)

        ind2lab = params.train_loader.label_dict["phn"]["index2lab"]
        phns_decode = undo_padding(phns, phn_lens)
        phns_decode = convert_index_to_lab(phns_decode, ind2lab)

        # run inference to each teacher model
        tea_dict_list = []
        for num in range(params.num_tea):
            tea_dict = {}
            self.tea_modules_list[num].eval()
            with torch.no_grad():
                x_tea = tea_enc_list[num](feats, init_params=init_params)
                ctc_logits_tea = tea_ctc_lin_list[num](x_tea, init_params)

                # output layer for ctc log-probabilities
                p_ctc_tea = params.log_softmax(ctc_logits_tea / params.T)

                # Prepend bos token at the beginning
                y_in_tea = prepend_bos_token(phns, bos_index=params.bos_index)
                e_in_tea = tea_emb_list[num](y_in_tea, init_params=init_params)
                h_tea, _ = tea_dec_list[num](
                    e_in_tea, x_tea, wav_lens, init_params
                )

                # output layer for seq2seq log-probabilities
                seq_logits_tea = tea_seq_lin_list[num](h_tea, init_params)
                p_seq_tea = apply_softmax(seq_logits_tea / params.T)

                # WER from output layer of CTC
                sequence_ctc = ctc_greedy_decode(
                    p_ctc_tea, wav_lens, blank_id=params.blank_index
                )
                sequence_ctc = convert_index_to_lab(sequence_ctc, ind2lab)
                per_stats_ctc = edit_distance.wer_details_for_batch(
                    ids, phns_decode, sequence_ctc, compute_alignments=False
                )

                wer_ctc_tea = []
                for item in per_stats_ctc:
                    wer_ctc_tea.append(item["WER"])

                wer_ctc_tea = exclude_wer(wer_ctc_tea)
                wer_ctc_tea = np.expand_dims(wer_ctc_tea, axis=0)

                # WER from output layer of CE
                _, predictions = p_seq_tea.max(dim=-1)
                hyps = batch_filter_seq2seq_output(
                    predictions, eos_id=params.eos_index
                )
                sequence_ce = convert_index_to_lab(hyps, ind2lab)
                per_stats_ce = edit_distance.wer_details_for_batch(
                    ids, phns_decode, sequence_ce, compute_alignments=False
                )

                wer_tea = []
                for item in per_stats_ce:
                    wer_tea.append(item["WER"])

                wer_tea = exclude_wer(wer_tea)
                wer_tea = np.expand_dims(wer_tea, axis=0)

            # save the variables into dict
            tea_dict["p_ctc_tea"] = p_ctc_tea.cpu().numpy()
            tea_dict["p_seq_tea"] = p_seq_tea.cpu().numpy()
            tea_dict["wer_ctc_tea"] = wer_ctc_tea
            tea_dict["wer_tea"] = wer_tea
            tea_dict_list.append(tea_dict)

        return tea_dict_list

    def fit_save(self, train_set, valid_set=None, test_set=None):
        data_sets = [train_set, valid_set, test_set]
        stage = ["train", "valid", "test"]
        tea_name = ["t0", "t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9"]

        if hasattr(params, "augmentation"):
            f_name = "/tea_infer_{}batch.hdf5".format(params.batch_size)
        else:
            f_name = "/tea_infer_noAug_{}batch.hdf5".format(params.batch_size)

        f = h5py.File(params.output_folder + f_name, "w")
        for num in range(3):
            # create group for each set (train, valid, test).
            g_sets = f.create_group(stage[num])

            with tqdm(data_sets[num], dynamic_ncols=True) as t:
                for i, batch in enumerate(t):
                    # create group for each batch
                    g_batch = g_sets.create_group(str(i))
                    inputs, targets = batch

                    # run inference to each teacher
                    tea_dict_list = self.compute_forward_tea(inputs, targets)

                    for tea_num in range(params.num_tea):
                        # create group for each teacher
                        g_tea = g_batch.create_group(tea_name[tea_num])
                        g_tea.create_dataset(
                            "p_ctc_tea",
                            data=tea_dict_list[tea_num]["p_ctc_tea"],
                        )
                        g_tea.create_dataset(
                            "p_seq_tea",
                            data=tea_dict_list[tea_num]["p_seq_tea"],
                        )
                        g_tea.create_dataset(
                            "wer_ctc_tea",
                            data=tea_dict_list[tea_num]["wer_ctc_tea"][0],
                        )
                        g_tea.create_dataset(
                            "wer_tea", data=tea_dict_list[tea_num]["wer_tea"][0]
                        )
        f.close()


# Prepare data
prepare_timit(
    data_folder=params.data_folder,
    splits=["train", "dev", "test"],
    save_folder=params.data_folder,
    uppercase=True,
)
train_set = params.train_loader()
valid_set = params.valid_loader()
test_set = params.test_loader()
first_x, first_y = next(iter(train_set))

asr_brain = ASR(
    tea_modules_list=tea_modules_list, first_inputs=[first_x, first_y],
)

# load teacher models
with open(params.tea_models_dir, "r") as f:
    enter_token = "\n"
    for i, path in enumerate(f.readlines()):
        exec(
            "tea{}_modules.load_state_dict(torch.load(path.strip(enter_token)))".format(
                i
            )
        )

# run inference and save results
asr_brain.fit_save(train_set, valid_set, test_set)
