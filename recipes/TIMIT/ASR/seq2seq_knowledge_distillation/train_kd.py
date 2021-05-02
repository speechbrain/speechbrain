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
import h5py
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main
from hyperpyyaml import load_hyperpyyaml


# Define training procedure
class ASR(sb.Brain):
    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        phns_bos, _ = batch.phn_encoded_bos

        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "env_corrupt"):
                wavs_noise = self.hparams.env_corrupt(wavs, wav_lens)
                wavs = torch.cat([wavs, wavs_noise], dim=0)
                wav_lens = torch.cat([wav_lens, wav_lens])
                phns_bos = torch.cat([phns_bos, phns_bos])
            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs, wav_lens)

        feats = self.hparams.compute_features(wavs)
        feats = self.modules.normalize(feats, wav_lens)
        x = self.modules.enc(feats)

        # output layer for ctc log-probabilities
        logits = self.modules.ctc_lin(x)
        p_ctc = self.hparams.log_softmax(logits)

        e_in = self.modules.emb(phns_bos)
        h, _ = self.modules.dec(e_in, x, wav_lens)

        # output layer for seq2seq log-probabilities
        logits = self.modules.seq_lin(h)
        p_seq = self.hparams.log_softmax(logits)

        if stage == sb.Stage.VALID:
            hyps, scores = self.hparams.greedy_searcher(x, wav_lens)
            return p_ctc, p_seq, wav_lens, hyps

        elif stage == sb.Stage.TEST:
            hyps, scores = self.hparams.beam_searcher(x, wav_lens)
            return p_ctc, p_seq, wav_lens, hyps

        return p_ctc, p_seq, wav_lens

    def def_tea_name(self):
        # define teacher variable name
        tea_name = []
        for tea_num in range(self.hparams.num_tea):
            tea = "t{}".format(tea_num)
            tea_name.append(tea)
        return tea_name

    def re_format(self, data_dict):
        item_tea_list = [None, None, None, None]
        tea_name = self.def_tea_name()
        for tea_num in range(self.hparams.num_tea):
            for i in range(4):
                item_tea = data_dict[str(self.step)][tea_name[tea_num]][
                    self.hparams.tea_keys[i]
                ][()]

                if self.hparams.tea_keys[i].startswith("wer"):
                    item_tea = torch.tensor(item_tea)
                else:
                    item_tea = torch.from_numpy(item_tea)

                item_tea = item_tea.to(self.device)
                item_tea = torch.unsqueeze(item_tea, 0)
                if tea_num == 0:
                    item_tea_list[i] = item_tea
                else:
                    item_tea_list[i] = torch.cat(
                        [item_tea_list[i], item_tea], 0
                    )
        return item_tea_list

    def compute_objectives(self, predictions, batch, stage):
        if stage == sb.Stage.TRAIN:
            p_ctc, p_seq, wav_lens = predictions
        else:
            p_ctc, p_seq, wav_lens, hyps = predictions

        ids = batch.id
        phns_eos, phn_lens_eos = batch.phn_encoded_eos
        phns, phn_lens = batch.phn_encoded

        if hasattr(self.modules, "env_corrupt") and stage == sb.Stage.TRAIN:
            phns_eos = torch.cat([phns_eos, phns_eos], dim=0)
            phn_lens_eos = torch.cat([phn_lens_eos, phn_lens_eos], dim=0)

        # normal supervised training
        loss_ctc_nor = self.hparams.ctc_cost(p_ctc, phns, wav_lens, phn_lens)
        loss_seq_nor = self.hparams.seq_cost(p_seq, phns_eos, phn_lens_eos)

        # load teacher inference results
        data_dict = (
            self.train_dict
            if stage == sb.Stage.TRAIN
            else self.valid_dict
            if stage == sb.Stage.VALID
            else self.test_dict
        )

        item_tea_list = self.re_format(data_dict)
        p_ctc_tea, p_seq_tea, wer_ctc_tea, wer_tea = [
            item for item in item_tea_list
        ]

        # Strategy "average": average losses of teachers when doing distillation.
        # Strategy "best": choosing the best teacher based on WER.
        # Strategy "weighted": assigning weights to teachers based on WER.
        if self.hparams.strategy == "best":
            # tea_ce for kd
            wer_scores, indx = torch.min(wer_tea, dim=0)
            indx = list(indx.cpu().numpy())

            # select the best teacher for each sentence
            tea_seq2seq_pout = None
            for stn_indx, tea_indx in enumerate(indx):
                s2s_one = p_seq_tea[tea_indx][stn_indx]
                s2s_one = torch.unsqueeze(s2s_one, 0)
                if stn_indx == 0:
                    tea_seq2seq_pout = s2s_one
                else:
                    tea_seq2seq_pout = torch.cat([tea_seq2seq_pout, s2s_one], 0)

        apply_softmax = torch.nn.Softmax(dim=0)

        if (
            self.hparams.strategy == "best"
            or self.hparams.strategy == "weighted"
        ):
            # mean wer for ctc
            tea_wer_ctc_mean = wer_ctc_tea.mean(1)
            tea_acc_main = 100 - tea_wer_ctc_mean

            # normalise weights via Softmax function
            tea_acc_softmax = apply_softmax(tea_acc_main)

        if self.hparams.strategy == "weighted":
            # mean wer for ce
            tea_wer_mean = wer_tea.mean(1)
            tea_acc_ce_main = 100 - tea_wer_mean

            # normalise weights via Softmax function
            tea_acc_ce_softmax = apply_softmax(tea_acc_ce_main)

        # kd loss
        ctc_loss_list = None
        ce_loss_list = None
        for tea_num in range(self.hparams.num_tea):
            # ctc
            p_ctc_tea_one = p_ctc_tea[tea_num]
            # calculate CTC distillation loss of one teacher
            loss_ctc_one = self.hparams.ctc_cost_kd(
                p_ctc, p_ctc_tea_one, wav_lens, device=self.device
            )
            loss_ctc_one = torch.unsqueeze(loss_ctc_one, 0)
            if tea_num == 0:
                ctc_loss_list = loss_ctc_one
            else:
                ctc_loss_list = torch.cat([ctc_loss_list, loss_ctc_one])

            # ce
            p_seq_tea_one = p_seq_tea[tea_num]
            # calculate CE distillation loss of one teacher
            loss_seq_one = self.hparams.seq_cost_kd(
                p_seq, p_seq_tea_one, phn_lens_eos
            )
            loss_seq_one = torch.unsqueeze(loss_seq_one, 0)
            if tea_num == 0:
                ce_loss_list = loss_seq_one
            else:
                ce_loss_list = torch.cat([ce_loss_list, loss_seq_one])

        # kd loss
        if self.hparams.strategy == "average":
            # get average value of losses from all teachers (CTC and CE loss)
            ctc_loss_kd = ctc_loss_list.mean(0)
            seq2seq_loss_kd = ce_loss_list.mean(0)
        else:
            # assign weights to different teachers (CTC loss)
            ctc_loss_kd = (tea_acc_softmax * ctc_loss_list).sum(0)
            if self.hparams.strategy == "best":
                # only use the best teacher to compute CE loss
                seq2seq_loss_kd = self.hparams.seq_cost_kd(
                    p_seq, tea_seq2seq_pout, phn_lens_eos
                )
            if self.hparams.strategy == "weighted":
                # assign weights to different teachers (CE loss)
                seq2seq_loss_kd = (tea_acc_ce_softmax * ce_loss_list).sum(0)

        # total loss
        # combine normal supervised training
        loss_ctc = (
            self.hparams.temperature
            * self.hparams.temperature
            * self.hparams.alpha
            * ctc_loss_kd
            + (1 - self.hparams.alpha) * loss_ctc_nor
        )
        loss_seq = (
            self.hparams.temperature
            * self.hparams.temperature
            * self.hparams.alpha
            * seq2seq_loss_kd
            + (1 - self.hparams.alpha) * loss_seq_nor
        )

        loss = (
            self.hparams.ctc_weight * loss_ctc
            + (1 - self.hparams.ctc_weight) * loss_seq
        )

        # Record losses for posterity
        if stage != sb.Stage.TRAIN:
            self.ctc_metrics.append(ids, p_ctc, phns, wav_lens, phn_lens)
            self.seq_metrics.append(ids, p_seq, phns_eos, phn_lens_eos)
            self.per_metrics.append(
                ids, hyps, phns, None, phn_lens, self.label_encoder.decode_ndim,
            )

        return loss

    def fit_batch(self, batch):
        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)
        loss.backward()
        if self.check_gradients(loss):
            self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.detach()

    def evaluate_batch(self, batch, stage):
        predictions = self.compute_forward(batch, stage=stage)
        loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()

    def on_stage_start(self, stage, epoch):
        self.ctc_metrics = self.hparams.ctc_stats()
        self.seq_metrics = self.hparams.seq_stats()

        if stage != sb.Stage.TRAIN:
            self.per_metrics = self.hparams.per_stats()

    def on_stage_end(self, stage, stage_loss, epoch):
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        else:
            per = self.per_metrics.summarize("error_rate")

        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(per)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats={"loss": self.train_loss},
                valid_stats={
                    "loss": stage_loss,
                    "ctc_loss": self.ctc_metrics.summarize("average"),
                    "seq_loss": self.seq_metrics.summarize("average"),
                    "PER": per,
                },
            )
            self.checkpointer.save_and_keep_only(
                meta={"PER": per}, min_keys=["PER"]
            )

        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats={"loss": stage_loss, "PER": per},
            )
            with open(self.hparams.wer_file, "w") as w:
                w.write("CTC loss stats:\n")
                self.ctc_metrics.write_stats(w)
                w.write("\nseq2seq loss stats:\n")
                self.seq_metrics.write_stats(w)
                w.write("\nPER stats:\n")
                self.per_metrics.write_stats(w)
                print(
                    "CTC, seq2seq, and PER stats written to file",
                    self.hparams.wer_file,
                )


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


def load_teachers(hparams):
    """
    Load results of inference of teacher models stored on disk.
    Note: Run experiment_save_teachers.py beforehand to generate .hdf5 files.
    """
    path = hparams["tea_infer_dir"] + "/tea_infer_{}batch.hdf5".format(
        hparams["batch_size"]
    )
    f = h5py.File(path, "r")
    train_dict = f["train"]
    valid_dict = f["valid"]
    test_dict = f["test"]

    return train_dict, valid_dict, test_dict


def st_load(hparams, asr_brain):
    """
    load pre-trained student model and remove decoder layer.
    """
    print("loading pre-trained student model...")
    chpt_path = hparams["pretrain_st_dir"] + "/model.ckpt"
    weight_dict = torch.load(chpt_path)
    # del the decoder layer
    key_list = []
    for k in weight_dict.keys():
        key_list.append(k)
    for k in key_list:
        if not k.startswith("0"):
            del weight_dict[k]

    # loading weights
    asr_brain.hparams.model.load_state_dict(weight_dict, strict=False)


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

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    asr_brain.label_encoder = label_encoder

    # load teacher models
    train_dict, valid_dict, test_dict = load_teachers(hparams)
    asr_brain.train_dict = train_dict
    asr_brain.valid_dict = valid_dict
    asr_brain.test_dict = test_dict

    if hparams["pretrain"]:
        # load pre-trained student model except last layer
        if hparams["epoch_counter"].current == 0:
            st_load(hparams, asr_brain)

    # Training/validation loop
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    # Test
    asr_brain.evaluate(
        test_data,
        min_key="PER",
        test_loader_kwargs=hparams["test_dataloader_opts"],
    )
