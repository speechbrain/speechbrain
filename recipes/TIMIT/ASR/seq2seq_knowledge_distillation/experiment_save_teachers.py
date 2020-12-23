#!/usr/bin/env python3
import os
import sys
import torch
import speechbrain as sb

from tqdm.contrib import tqdm
import h5py
import numpy as np


# Define training procedure
class ASR(sb.Brain):
    def __init__(self, tea_modules_list=None, hparams=None):
        super(ASR, self).__init__(
            modules=None, opt_class=None, hparams=hparams, checkpointer=None
        )

        # Initialize teacher parameters
        tea_modules_list_ = []
        for tea_modules in tea_modules_list:
            tea_modules_ = torch.nn.ModuleList(tea_modules)
            tea_modules_list_.append(tea_modules_)
        self.tea_modules_list = tea_modules_list_

    def compute_forward_tea(self, x, y):
        ids, wavs, wav_lens = x
        ids, phns, phn_lens = y

        wavs, wav_lens = (
            wavs.to(self.hparams.device),
            wav_lens.to(self.hparams.device),
        )
        phns, phn_lens = (
            phns.to(self.hparams.device),
            phn_lens.to(self.hparams.device),
        )

        if hasattr(self.hparams, "augmentation"):
            wavs = self.hparams.augmentation(wavs, wav_lens)

        feats = self.hparams.compute_features(wavs)
        feats = self.hparams.normalize(feats, wav_lens)
        apply_softmax = torch.nn.Softmax(dim=-1)

        ind2lab = self.hparams.train_loader.label_dict["phn"]["index2lab"]
        phns_decode = sb.utils.data_utils.undo_padding(phns, phn_lens)
        phns_decode = sb.data_io.data_io.convert_index_to_lab(
            phns_decode, ind2lab
        )

        # run inference to each teacher model
        tea_dict_list = []
        for num in range(self.hparams.num_tea):
            tea_dict = {}
            self.tea_modules_list[num].eval()
            with torch.no_grad():
                x_tea = tea_enc_list[num](feats)
                ctc_logits_tea = tea_ctc_lin_list[num](x_tea)

                # output layer for ctc log-probabilities
                p_ctc_tea = self.hparams.log_softmax(
                    ctc_logits_tea / self.hparams.temperature
                )

                # Prepend bos token at the beginning
                y_in_tea = sb.data_io.data_io.prepend_bos_token(
                    phns, bos_index=self.hparams.bos_index
                )
                e_in_tea = tea_emb_list[num](y_in_tea)
                h_tea, _ = tea_dec_list[num](e_in_tea, x_tea, wav_lens)

                # output layer for seq2seq log-probabilities
                seq_logits_tea = tea_seq_lin_list[num](h_tea)
                p_seq_tea = apply_softmax(
                    seq_logits_tea / self.hparams.temperature
                )

                # WER from output layer of CTC
                sequence_ctc = sb.decoders.ctc.ctc_greedy_decode(
                    p_ctc_tea, wav_lens, blank_id=self.hparams.blank_index
                )
                sequence_ctc = sb.data_io.data_io.convert_index_to_lab(
                    sequence_ctc, ind2lab
                )
                per_stats_ctc = sb.utils.edit_distance.wer_details_for_batch(
                    ids, phns_decode, sequence_ctc, compute_alignments=False
                )

                wer_ctc_tea = []
                for item in per_stats_ctc:
                    wer_ctc_tea.append(item["WER"])

                wer_ctc_tea = exclude_wer(wer_ctc_tea)
                wer_ctc_tea = np.expand_dims(wer_ctc_tea, axis=0)

                # WER from output layer of CE
                _, predictions = p_seq_tea.max(dim=-1)
                hyps = sb.decoders.seq2seq.batch_filter_seq2seq_output(
                    predictions, eos_id=self.hparams.eos_index
                )
                sequence_ce = sb.data_io.data_io.convert_index_to_lab(
                    hyps, ind2lab
                )
                per_stats_ce = sb.utils.edit_distance.wer_details_for_batch(
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

    def def_tea_name(self):
        # define teacher variable name
        tea_name = []
        for tea_num in range(self.hparams.num_tea):
            tea = "t{}".format(tea_num)
            tea_name.append(tea)
        return tea_name

    def fit_save(self, train_set, valid_set=None, test_set=None):
        data_sets = [train_set, valid_set, test_set]
        stage = self.hparams.stage
        tea_name = self.def_tea_name()

        # define output file name
        f_name = "/tea_infer_{}batch.hdf5".format(self.hparams.batch_size)
        f = h5py.File(self.hparams.output_folder + f_name, "w")
        for num in range(len(stage)):
            # create group for each set (train, valid, test).
            g_sets = f.create_group(stage[num])

            data_sets[num] = data_sets[num].get_dataloader()
            with tqdm(data_sets[num], dynamic_ncols=True) as t:
                for i, batch in enumerate(t):
                    # create group for each batch
                    g_batch = g_sets.create_group(str(i))
                    inputs, targets = batch

                    # run inference to each teacher
                    tea_dict_list = self.compute_forward_tea(inputs, targets)

                    for tea_num in range(self.hparams.num_tea):
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


if __name__ == "__main__":
    # This hack needed to import data preparation script from ../..
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(os.path.dirname(current_dir)))
    from timit_prepare import prepare_timit  # noqa E402

    # Load hyperparameters file with command-line overrides
    hparams_file, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = sb.load_extended_yaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Prepare data
    prepare_timit(
        data_folder=hparams["data_folder"],
        splits=["train", "dev", "test"],
        save_folder=hparams["data_folder"],
    )

    # initialise teacher model variables
    tea_enc_list = []
    tea_emb_list = []
    tea_dec_list = []
    tea_ctc_lin_list = []
    tea_seq_lin_list = []
    for i in range(hparams["num_tea"]):
        exec("tea_enc_list.append(hparams['tea{}_enc'])".format(i))
        exec("tea_emb_list.append(hparams['tea{}_emb'])".format(i))
        exec("tea_dec_list.append(hparams['tea{}_dec'])".format(i))
        exec("tea_ctc_lin_list.append(hparams['tea{}_ctc_lin'])".format(i))
        exec("tea_seq_lin_list.append(hparams['tea{}_seq_lin'])".format(i))

    # create ModuleList
    for i in range(hparams["num_tea"]):
        exec(
            "tea{}_modules = torch.nn.ModuleList([tea_enc_list[i], tea_emb_list[i], tea_dec_list[i], tea_ctc_lin_list[i], tea_seq_lin_list[i]])".format(
                i
            )
        )  # i denotes the index of teacher models

        exec("tea{}_modules = tea{}_modules.to(hparams['device'])".format(i, i))

    tea_modules_list = []
    for i in range(hparams["num_tea"]):
        exec("tea_modules_list.append(tea{}_modules)".format(i))

    # Collect index to label conversion dict for decoding
    train_set = hparams["train_loader"]()
    valid_set = hparams["valid_loader"]()
    test_set = hparams["test_loader"]()
    hparams["ind2lab"] = hparams["train_loader"].label_dict["phn"]["index2lab"]

    asr_brain = ASR(tea_modules_list=tea_modules_list, hparams=hparams)

    # load pre-trained weights of teacher models
    with open(hparams["tea_models_dir"], "r") as f:
        enter_token = "\n"
        for i, path in enumerate(f.readlines()):
            exec(
                "tea{}_modules.load_state_dict(torch.load(path.strip(enter_token)))".format(
                    i
                )
            )

    # run inference and save results
    asr_brain.fit_save(train_set, valid_set, test_set)
