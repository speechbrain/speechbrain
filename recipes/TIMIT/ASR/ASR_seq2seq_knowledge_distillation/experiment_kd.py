#!/usr/bin/env python3
import os
import sys
import torch
import speechbrain as sb
from tqdm.contrib import tqdm
import h5py

from torch.utils.data import DistributedSampler
from speechbrain.data_io.data_io import DataLoaderFactory


TEA_KEYS = ["p_ctc_tea", "p_seq_tea", "wer_ctc_tea", "wer_tea"]
TEA_NAME = ["t0", "t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9"]


# Define training procedure
class ASR(sb.Brain):
    def compute_forward(self, x, y, stage):
        ids, wavs, wav_lens = x
        ids, phns, phn_lens = y

        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        phns, phn_lens = phns.to(self.device), phn_lens.to(self.device)

        if stage == sb.Stage.TRAIN:
            if hasattr(self.modules, "env_corrupt"):
                wavs_noise = self.modules.env_corrupt(wavs, wav_lens)
                wavs = torch.cat([wavs, wavs_noise], dim=0)
                wav_lens = torch.cat([wav_lens, wav_lens])
                phns = torch.cat([phns, phns])
            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs, wav_lens)

        feats = self.hparams.compute_features(wavs)
        feats = self.modules.normalize(feats, wav_lens)
        x = self.modules.enc(feats)

        # output layer for ctc log-probabilities
        logits = self.modules.ctc_lin(x)
        p_ctc = self.hparams.log_softmax(logits)

        # Prepend bos token at the beginning
        y_in = sb.data_io.data_io.prepend_bos_token(
            phns, self.hparams.bos_index
        )
        e_in = self.modules.emb(y_in)
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

    def compute_objectives(
        self, predictions, targets, data_dict, batch_id, stage
    ):
        if stage == sb.Stage.TRAIN:
            p_ctc, p_seq, wav_lens = predictions
        else:
            p_ctc, p_seq, wav_lens, hyps = predictions

        ids, phns, phn_lens = targets
        phns, phn_lens = phns.to(self.device), phn_lens.to(self.device)

        # Add phn_lens by one for eos token
        abs_length = torch.round(phn_lens * phns.shape[1])

        # Append eos token at the end of the label sequences
        phns_with_eos = sb.data_io.data_io.append_eos_token(
            phns, length=abs_length, eos_index=self.hparams.eos_index
        )

        # convert to speechbrain-style relative length
        rel_length = (abs_length + 1) / phns.shape[1]

        # normal supervised training
        loss_ctc_nor = self.hparams.ctc_cost(p_ctc, phns, wav_lens, phn_lens)
        loss_seq_nor = self.hparams.seq_cost(p_seq, phns_with_eos, length=rel_length)

        # load teacher inference results
        item_tea_list = [None, None, None, None]
        for tea_num in range(self.hparams.num_tea):
            for i in range(4):
                item_tea = data_dict[str(batch_id)][TEA_NAME[tea_num]][
                    TEA_KEYS[i]
                ][()]

                if TEA_KEYS[i].startswith("wer"):
                    item_tea = torch.tensor(item_tea)
                else:
                    item_tea = torch.from_numpy(item_tea)

                item_tea = item_tea.to(self.hparams.device)
                item_tea = torch.unsqueeze(item_tea, 0)
                if tea_num == 0:
                    item_tea_list[i] = item_tea
                else:
                    item_tea_list[i] = torch.cat(
                        [item_tea_list[i], item_tea], 0
                    )

        p_ctc_tea = item_tea_list[0]
        p_seq_tea = item_tea_list[1]
        wer_ctc_tea = item_tea_list[2]
        wer_tea = item_tea_list[3]

        # Stategy "average": average losses of teachers when doing distillation.
        # Stategy "best": choosing the best teacher based on WER.
        # Stategy "weighted": assigning weights to teachers based on WER.
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

        if self.hparams.strategy == "best" or self.hparams.strategy == "weighted":
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
            loss_ctc_one = self.hparams.ctc_cost_kd(p_ctc, p_ctc_tea_one, wav_lens)
            loss_ctc_one = torch.unsqueeze(loss_ctc_one, 0)
            if tea_num == 0:
                ctc_loss_list = loss_ctc_one
            else:
                ctc_loss_list = torch.cat([ctc_loss_list, loss_ctc_one])

            # ce
            p_seq_tea_one = p_seq_tea[tea_num]
            # calculate CE distillation loss of one teacher
            loss_seq_one = self.hparams.seq_cost_kd(p_seq, p_seq_tea_one, rel_length)
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
                    p_seq, tea_seq2seq_pout, rel_length
                )
            if self.hparams.strategy == "weighted":
                # assign weights to different teachers (CE loss)
                seq2seq_loss_kd = (tea_acc_ce_softmax * ce_loss_list).sum(0)

        # total loss
        # combine normal supervised training
        loss_ctc = (
            self.hparams.Temperature * self.hparams.Temperature * self.hparams.alpha * ctc_loss_kd
            + (1 - self.hparams.alpha) * loss_ctc_nor
        )
        loss_seq = (
            self.hparams.Temperature
            * self.hparams.Temperature
            * self.hparams.alpha
            * seq2seq_loss_kd
            + (1 - self.hparams.alpha) * loss_seq_nor
        )

        loss = self.hparams.ctc_weight * loss_ctc + (1 - self.hparams.ctc_weight) * loss_seq

        # Record losses for posterity
        if stage != sb.Stage.TRAIN:
            self.ctc_metrics.append(ids, p_ctc, phns, wav_lens, phn_lens)
            self.seq_metrics.append(ids, p_seq, phns_with_eos, rel_length)
            self.per_metrics.append(
                ids, hyps, phns, None, phn_lens, self.hparams.ind2lab
            )

        return loss

    def fit_batch(self, batch, train_dict, batch_id):
        inputs, targets = batch
        predictions = self.compute_forward(inputs, targets, sb.Stage.TRAIN)
        loss = self.compute_objectives(
            predictions, targets, train_dict, batch_id, sb.Stage.TRAIN
        )
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.detach()

    def evaluate_batch(self, batch, batch_id, data_dict, stage):
        inputs, targets = batch
        predictions = self.compute_forward(inputs, targets, stage=stage)
        loss = self.compute_objectives(
            predictions, targets, data_dict, batch_id, stage=stage
        )
        return loss.detach()

    def fit(
        self, epoch_counter, save_dict, train_set, valid_set=None, progressbar=None,
    ):
        train_dict, valid_dict = save_dict
        self.on_fit_start()

        if progressbar is None:
            progressbar = self.progressbar

        # Use factories to get loaders
        self.train_sampler = None
        if isinstance(train_set, DataLoaderFactory):
            if self.rank is not None:
                self.train_sampler = DistributedSampler(
                    dataset=train_set.dataset,
                    num_replicas=self.multigpu_count,
                    rank=self.rank,
                    shuffle=train_set.shuffle,
                )
            train_set = train_set.get_dataloader(self.train_sampler)
        if isinstance(valid_set, DataLoaderFactory):
            valid_set = valid_set.get_dataloader()

        # Iterate epochs
        for epoch in epoch_counter:

            # Training stage
            self.on_stage_start(sb.Stage.TRAIN, epoch)
            self.modules.train()
            avg_train_loss = 0.0

            # Reset nonfinite count to 0 each epoch
            self.nonfinite_count = 0

            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            # Only show progressbar if requested and root_process
            disable = not (progressbar and self.root_process)
            with tqdm(train_set, dynamic_ncols=True, disable=disable) as t:
                for self.step, batch in enumerate(t):
                    loss = self.fit_batch(batch, train_dict, self.step)
                    avg_train_loss = self.update_average(loss, avg_train_loss)
                    t.set_postfix(train_loss=avg_train_loss)
            self.on_stage_end(sb.Stage.TRAIN, avg_train_loss, epoch)

            # Validation stage
            avg_valid_loss = None
            if valid_set is not None:
                self.on_stage_start(sb.Stage.VALID, epoch)
                self.modules.eval()
                avg_valid_loss = 0.0
                with torch.no_grad():
                    for self.step, batch in enumerate(
                        tqdm(valid_set, dynamic_ncols=True, disable=disable)
                    ):
                        loss = self.evaluate_batch(batch, self.step, valid_dict, stage=sb.Stage.VALID)
                        avg_valid_loss = self.update_average(
                            loss, avg_valid_loss
                        )
                self.on_stage_end(sb.Stage.VALID, avg_valid_loss, epoch)

    def evaluate(self, test_set, test_dict, max_key=None, min_key=None, progressbar=None):
        if progressbar is None:
            progressbar = self.progressbar

        # Get test loader from factory
        if isinstance(test_set, DataLoaderFactory):
            test_set = test_set.get_dataloader()

        self.on_evaluate_start(max_key=max_key, min_key=min_key)
        self.on_stage_start(sb.Stage.TEST, epoch=None)
        self.modules.eval()
        avg_test_loss = 0.0
        disable = not progressbar
        with torch.no_grad():
            for self.step, batch in enumerate(
                tqdm(test_set, dynamic_ncols=True, disable=disable)
            ):
                loss = self.evaluate_batch(batch, self.step, test_dict, stage=sb.Stage.TEST)
                avg_test_loss = self.update_average(loss, avg_test_loss)
        self.on_stage_end(sb.Stage.TEST, avg_test_loss, epoch=None)

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

            if self.root_process:
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

    def on_fit_start(self):
        """Gets called at the beginning of ``fit()``, on multiple processes
        if multigpu_count is more than 0 and backend is ddp.

        Default implementation compiles the jit modules, initializes
        optimizers, and loads the latest checkpoint to resume training.
        """
        # Run this *after* mp.spawn since jit modules cannot be pickled.
        self._compile_jit()

        # Wrap modules with parallel backend after jit
        self._wrap_multigpu()

        # Initialize optimizers after parameters are configured
        self.init_optimizers()

        # initialization strategy
        if self.hparams.pretrain:
            # load pre-trained student model except last layer
            if self.hparams.epoch_counter.current == 0:
                chpt_path = self.hparams.pretrain_tea_dir + "/model.ckpt"
                weight_dict = torch.load(chpt_path)
                # del the last layer
                key_list = []
                for k in weight_dict.keys():
                    key_list.append(k)
                for k in key_list:
                    if k.startswith("1") or k.startswith("2"):
                        del weight_dict[k]

                self.modules.load_state_dict(weight_dict, strict=False)
            else:
                # Load latest checkpoint to resume training
                self.checkpointer.recover_if_possible(device=torch.device(self.device))
        else:
            self.checkpointer.recover_if_possible(device=torch.device(self.device))


def load_teachers(hparams):
    """
    Load results of inference of teacher models stored on disk.
    Note: Run experiment_save_teachers.py beforehand to generate .hdf5 files.
    """
    if "augmentation" in hparams.keys():
        path = hparams["tea_infer_dir"] + "/tea_infer_{}batch.hdf5".format(
            hparams["batch_size"]
        )
    else:
        path = hparams["tea_infer_dir"] + "/tea_infer_noAug_{}batch.hdf5".format(
            hparams["batch_size"]
        )

    f = h5py.File(path, "r")
    train_dict = f["train"]
    valid_dict = f["valid"]
    test_dict = f["test"]

    return [train_dict, valid_dict], test_dict


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

    # Collect index to label conversion dict for decoding
    train_set = hparams["train_loader"]()
    valid_set = hparams["valid_loader"]()
    hparams["ind2lab"] = hparams["train_loader"].label_dict["phn"]["index2lab"]

    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        checkpointer=hparams["checkpointer"],
    )

    # load teacher models
    save_dict, test_dict = load_teachers(hparams)

    asr_brain.fit(asr_brain.hparams.epoch_counter, save_dict, train_set, valid_set)
    asr_brain.evaluate(hparams["test_loader"](), test_dict, min_key="PER")
