#!/usr/bin/env python3
import os
import sys
import torch
import speechbrain as sb
from tqdm.contrib import tqdm
import numpy as np

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
    experiment_directory=params.output_folder,
    params_to_save=params_file,
    overrides=overrides,
)

for i in range(params.num_tea):
    exec(
        "tea{}_modules = torch.nn.ModuleList([params.tea{}_enc, params.tea{}_emb, params.tea{}_dec, params.tea{}_ctc_lin, params.tea{}_seq_lin])".format(
            i, i, i, i, i, i
        )
    )  # i denotes the index of teacher models

tea_modules_list = []
for i in range(params.num_tea):
    exec("tea_modules_list.append(tea{}_modules)".format(i))


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
        m = torch.nn.Softmax(dim=-1)
        print(m)

        # run inference to each teacher model
        tea_dict_list = []
        for num in range(params.num_tea):
            tea_dict = {}
            self.tea_modules_list[num].eval()
            with torch.no_grad():
                exec(
                    "x_tea{} = params.tea{}_enc(feats, init_params=init_params)".format(
                        num, num
                    )
                )
                exec(
                    "ctc_logits_tea{} = params.tea{}_ctc_lin(x_tea{}, init_params)".format(
                        num, num, num
                    )
                )
                # output layer for ctc log-probabilities
                exec(
                    "p_ctc_tea{} = params.log_softmax(ctc_logits_tea{} / params.T)".format(
                        num, num
                    )
                )
                # Prepend bos token at the beginning
                exec(
                    "y_in_tea{} = prepend_bos_token(phns, bos_index=params.bos_index)".format(
                        num
                    )
                )
                exec(
                    "e_in_tea{} = params.tea{}_emb(y_in_tea{}, init_params=init_params)".format(
                        num, num, num
                    )
                )
                exec(
                    "h_tea{}, _ = params.tea{}_dec(e_in_tea{}, x_tea{}, wav_lens, init_params)".format(
                        num, num, num, num
                    )
                )
                # output layer for seq2seq log-probabilities
                exec(
                    "seq_logits_tea{} = params.tea{}_seq_lin(h_tea{}, init_params)".format(
                        num, num, num
                    )
                )
                exec(
                    "p_seq_tea{} = m(seq_logits_tea{} / params.T)".format(
                        num, num
                    )
                )
                # WER from output layer of CTC
                exec(
                    "wer_ctc_tea{} = params.wer_ctc(p_ctc_tea{}, phns, wav_lens, phn_lens)".format(
                        num, num
                    )
                )
                # WER from output layer of CE
                exec(
                    "wer_tea{} = params.wer_ce(p_seq_tea{}, phns, phn_lens)".format(
                        num, num
                    )
                )

                exec(
                    "wer_ctc_tea{} = wer_ctc_tea{}.to(params.device)".format(
                        num, num
                    )
                )
                exec("wer_tea{} = wer_tea{}.to(params.device)".format(num, num))

            # save the variables into dict
            exec(
                "tea_dict['p_ctc_tea'] = p_ctc_tea{}.cpu().numpy()".format(num)
            )
            exec(
                "tea_dict['p_seq_tea'] = p_seq_tea{}.cpu().numpy()".format(num)
            )
            exec(
                "tea_dict['wer_ctc_tea'] = wer_ctc_tea{}.cpu().numpy()".format(
                    num
                )
            )
            exec("tea_dict['wer_tea'] = wer_tea{}.cpu().numpy()".format(num))
            tea_dict_list.append(tea_dict)

        return tea_dict_list

    def fit_save(self, epoch_counter, train_set, valid_set=None, test_set=None):
        data_sets = [train_set, valid_set, test_set]
        save_dict_list = []
        for epoch in epoch_counter:
            # loop for each data set [train_set, valid_set, test_set]
            for data_set in data_sets:
                save_dict = {}
                with tqdm(data_set, dynamic_ncols=True) as t:
                    # loop for each mini-batch
                    for i, batch in enumerate(t):
                        inputs, targets = batch
                        tea_dict_list = self.compute_forward_tea(
                            inputs, targets
                        )
                        ids, _, _ = inputs
                        batch_len = len(ids)
                        # loop for each sentence
                        for b_num in range(batch_len):
                            temp_dict_list = []
                            # loop for each teacher
                            for tea_num in range(params.num_tea):
                                temp_dict = {}
                                temp_dict["p_ctc_tea"] = tea_dict_list[tea_num][
                                    "p_ctc_tea"
                                ][b_num]
                                temp_dict["p_seq_tea"] = tea_dict_list[tea_num][
                                    "p_seq_tea"
                                ][b_num]
                                temp_dict["wer_ctc_tea"] = tea_dict_list[
                                    tea_num
                                ]["wer_ctc_tea"][0][b_num]
                                temp_dict["wer_tea"] = tea_dict_list[tea_num][
                                    "wer_tea"
                                ][0][b_num]
                                temp_dict_list.append(temp_dict)

                            save_dict[ids[b_num]] = temp_dict_list
                save_dict_list.append(save_dict)

        if hasattr(params, "augmentation"):
            f_name = "/tea_infer_{}batch.npz".format(params.batch_size)
        else:
            f_name = "/tea_infer_noAug_{}batch.npz".format(params.batch_size)
        # save on disk
        np.savez(
            current_dir + f_name,
            train_dict=save_dict_list[0],
            valid_dict=save_dict_list[1],
            test_dict=save_dict_list[2],
        )


# Prepare data
prepare_timit(
    data_folder=params.data_folder,
    splits=["train", "dev", "test"],
    save_folder=params.data_folder,
)
train_set = params.train_loader()
valid_set = params.valid_loader()
test_set = params.test_loader()
first_x, first_y = next(iter(train_set))

asr_brain = ASR(
    tea_modules_list=tea_modules_list, first_inputs=[first_x, first_y],
)

# load teacher models
for i in range(params.num_tea):
    save_dir = (
        current_dir
        + "/teacher_models_training/results/tea{}/{}/save/".format(
            i, params.seed
        )
    )
    ckpt_dir_list = os.listdir(save_dir)
    chpt_path = save_dir + ckpt_dir_list[0] + "/model.ckpt"
    exec("tea{}_modules.load_state_dict(torch.load(chpt_path))".format(i))

# run inference and save results
asr_brain.fit_save(params.epoch_counter, train_set, valid_set, test_set)
