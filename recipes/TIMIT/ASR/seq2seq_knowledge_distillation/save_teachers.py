#!/usr/bin/env python3

"""Recipe for doing ASR with phoneme targets and joint seq2seq
and CTC loss on the TIMIT dataset following a knowledge distillation scheme as
reported in " Distilling Knowledge from Ensembles of Acoustic Models for Joint
CTC-Attention End-to-End Speech Recognition", Yan Gao et al.

To run this recipe, do the following:
> python experiment.py hyperparams.yaml --data_folder /path/to/TIMIT

Authors
 * Yan Gao 2021
 * Titouan Parcollet 2021
"""

import sys
import torch
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main
from hyperpyyaml import load_hyperpyyaml

from tqdm.contrib import tqdm
import h5py
import numpy as np


# Define training procedure
class ASR(sb.Brain):
    def __init__(self, tea_modules_list=None, hparams=None, run_opts=None):
        super(ASR, self).__init__(
            modules=None,
            opt_class=None,
            hparams=hparams,
            run_opts=run_opts,
            checkpointer=None,
        )

        # Initialize teacher parameters
        tea_modules_list_ = []
        for tea_modules in tea_modules_list:
            tea_modules_ = torch.nn.ModuleList(tea_modules)
            tea_modules_ = tea_modules_.to(self.device)
            tea_modules_list_.append(tea_modules_)
        self.tea_modules_list = tea_modules_list_

    def compute_forward_tea(self, batch):
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        phns_bos, _ = batch.phn_encoded_bos
        phns, phn_lens = batch.phn_encoded

        feats = self.hparams.compute_features(wavs)
        feats = self.hparams.normalize(feats, wav_lens)
        apply_softmax = torch.nn.Softmax(dim=-1)

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

                e_in_tea = tea_emb_list[num](phns_bos)
                h_tea, _ = tea_dec_list[num](e_in_tea, x_tea, wav_lens)

                # output layer for seq2seq log-probabilities
                seq_logits_tea = tea_seq_lin_list[num](h_tea)
                p_seq_tea = apply_softmax(
                    seq_logits_tea / self.hparams.temperature
                )

                # WER from output layer of CTC
                sequence_ctc = sb.decoders.ctc_greedy_decode(
                    p_ctc_tea, wav_lens, blank_id=self.hparams.blank_index
                )

                phns_decode = sb.utils.data_utils.undo_padding(phns, phn_lens)
                phns_decode = self.label_encoder.decode_ndim(phns_decode)
                sequence_decode = self.label_encoder.decode_ndim(sequence_ctc)

                per_stats_ctc = sb.utils.edit_distance.wer_details_for_batch(
                    batch.id,
                    phns_decode,
                    sequence_decode,
                    compute_alignments=False,
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
                sequence_ce = self.label_encoder.decode_ndim(hyps)
                per_stats_ce = sb.utils.edit_distance.wer_details_for_batch(
                    batch.id, phns_decode, sequence_ce, compute_alignments=False
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

            with tqdm(
                data_sets[num], initial=self.step, dynamic_ncols=True,
            ) as t:
                for batch in t:
                    self.step += 1
                    # create group for each batch
                    g_batch = g_sets.create_group(str(self.step))

                    # run inference to each teacher
                    tea_dict_list = self.compute_forward_tea(batch)

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
            self.step = 0
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


def data_io_prep(hparams):
    "Creates the datasets and their data processing pipelines."
    data_folder = hparams["data_folder"]
    # 1. Declarations:
    train_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["train_annotation"],
        replacements={"data_root": data_folder},
    )
    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["valid_annotation"],
        replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    test_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["test_annotation"],
        replacements={"data_root": data_folder},
    )
    test_data = test_data.filtered_sorted(sort_key="duration")

    datasets = [train_data, valid_data, test_data]
    label_encoder = sb.dataio.encoder.CTCTextEncoder()

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("phn")
    @sb.utils.data_pipeline.provides(
        "phn_list",
        "phn_encoded_list",
        "phn_encoded",
        "phn_encoded_eos",
        "phn_encoded_bos",
    )
    def text_pipeline(phn):
        phn_list = phn.strip().split()
        yield phn_list
        phn_encoded_list = label_encoder.encode_sequence(phn_list)
        yield phn_encoded_list
        phn_encoded = torch.LongTensor(phn_encoded_list)
        yield phn_encoded
        phn_encoded_eos = torch.LongTensor(
            label_encoder.append_eos_index(phn_encoded_list)
        )
        yield phn_encoded_eos
        phn_encoded_bos = torch.LongTensor(
            label_encoder.prepend_bos_index(phn_encoded_list)
        )
        yield phn_encoded_bos

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 3. Fit encoder:
    # NOTE: In this minimal example, also update from valid data

    label_encoder.update_from_didataset(train_data, output_key="phn_list")
    if (
        hparams["blank_index"] != hparams["bos_index"]
        or hparams["blank_index"] != hparams["eos_index"]
    ):
        label_encoder.insert_blank(index=hparams["blank_index"])

    if hparams["bos_index"] == hparams["eos_index"]:
        label_encoder.insert_bos_eos(
            bos_label="<eos-bos>",
            eos_label="<eos-bos>",
            bos_index=hparams["bos_index"],
        )
    else:
        label_encoder.insert_bos_eos(
            bos_label="<bos>",
            eos_label="<eos>",
            bos_index=hparams["bos_index"],
            eos_index=hparams["eos_index"],
        )

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "sig", "phn_encoded", "phn_encoded_eos", "phn_encoded_bos"],
    )

    return train_data, valid_data, test_data, label_encoder


if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Dataset prep (parsing TIMIT and annotation into csv files)
    from timit_prepare import prepare_timit  # noqa

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # multi-gpu (ddp) save data preparation
    run_on_main(
        prepare_timit,
        kwargs={
            "data_folder": hparams["data_folder"],
            "splits": ["train", "dev", "test"],
            "save_folder": hparams["data_folder"],
        },
    )

    # Dataset IO prep: creating Dataset objects and proper encodings for phones
    train_data, valid_data, test_data, label_encoder = data_io_prep(hparams)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
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

    tea_modules_list = []
    for i in range(hparams["num_tea"]):
        exec("tea_modules_list.append(tea{}_modules)".format(i))

    # Trainer initialization
    asr_brain = ASR(
        tea_modules_list=tea_modules_list, hparams=hparams, run_opts=run_opts
    )
    asr_brain.label_encoder = label_encoder

    # load pre-trained weights of teacher models
    with open(hparams["tea_models_dir"], "r") as f:
        enter_token = "\n"
        for i, path in enumerate(f.readlines()):
            exec(
                "tea{}_modules.load_state_dict(torch.load(path.strip(enter_token)))".format(
                    i
                )
            )

    # make dataloaders
    train_set = sb.dataio.dataloader.make_dataloader(
        train_data, **hparams["train_dataloader_opts"]
    )
    valid_set = sb.dataio.dataloader.make_dataloader(
        valid_data, **hparams["valid_dataloader_opts"]
    )
    test_set = sb.dataio.dataloader.make_dataloader(
        test_data, **hparams["test_dataloader_opts"]
    )

    # run inference and save results
    asr_brain.fit_save(train_set, valid_set, test_set)
