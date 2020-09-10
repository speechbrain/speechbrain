#!/usr/bin/env python3
import os
import sys
import torch
import speechbrain as sb
from speechbrain.data_io.data_io import prepend_bos_token
from speechbrain.data_io.data_io import append_eos_token


# Define training procedure
class ASR(sb.Brain):
    def compute_forward(self, x, y, stage):
        ids, wavs, wav_lens = x
        ids, phns, phn_lens = y

        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        phns, phn_lens = phns.to(self.device), phn_lens.to(self.device)

        if hasattr(self, "env_corrupt") and stage == sb.Stage.TRAIN:
            wavs_noise = self.env_corrupt(wavs, wav_lens)
            wavs = torch.cat([wavs, wavs_noise], dim=0)
            wav_lens = torch.cat([wav_lens, wav_lens])
            phns = torch.cat([phns, phns])

        if hasattr(self, "augmentation"):
            wavs = self.augmentation(wavs, wav_lens)
        feats = self.compute_features(wavs)
        feats = self.normalize(feats, wav_lens)
        x = self.enc(feats)

        # output layer for ctc log-probabilities
        logits = self.ctc_lin(x)
        p_ctc = self.log_softmax(logits)

        # Prepend bos token at the beginning
        y_in = prepend_bos_token(phns, bos_index=self.bos_index)
        e_in = self.emb(y_in)
        h, _ = self.dec(e_in, x, wav_lens)

        # output layer for seq2seq log-probabilities
        logits = self.seq_lin(h)
        p_seq = self.log_softmax(logits)

        if stage == sb.Stage.VALID:
            hyps, scores = self.greedy_searcher(x, wav_lens)
            return p_ctc, p_seq, wav_lens, hyps

        elif stage == sb.Stage.TEST:
            hyps, scores = self.beam_searcher(x, wav_lens)
            return p_ctc, p_seq, wav_lens, hyps

        return p_ctc, p_seq, wav_lens

    def compute_objectives(self, predictions, targets, stage):
        if stage == sb.Stage.TRAIN:
            p_ctc, p_seq, wav_lens = predictions
        else:
            p_ctc, p_seq, wav_lens, hyps = predictions

        ids, phns, phn_lens = targets
        phns, phn_lens = phns.to(self.device), phn_lens.to(self.device)

        if hasattr(self, "env_corrupt") and stage == sb.Stage.TRAIN:
            phns = torch.cat([phns, phns], dim=0)
            phn_lens = torch.cat([phn_lens, phn_lens], dim=0)

        # Add phn_lens by one for eos token
        abs_length = torch.round(phn_lens * phns.shape[1])

        # Append eos token at the end of the label sequences
        phns_with_eos = append_eos_token(
            phns, length=abs_length, eos_index=self.eos_index
        )

        # convert to speechbrain-style relative length
        rel_length = (abs_length + 1) / phns_with_eos.shape[1]

        loss_ctc = self.ctc_cost(p_ctc, phns, wav_lens, phn_lens)
        loss_seq = self.seq_cost(p_seq, phns_with_eos, length=rel_length)
        loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_seq

        # Record losses for posterity
        if stage != sb.Stage.TRAIN:
            self.ctc_metrics.append(ids, p_ctc, phns, wav_lens, phn_lens)
            self.seq_metrics.append(ids, p_seq, phns_with_eos, rel_length)
            self.per_metrics.append(ids, hyps, phns, phn_lens, self.ind2lab)

        return loss

    def fit_batch(self, batch):
        inputs, targets = batch
        predictions = self.compute_forward(inputs, targets, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, targets, sb.Stage.TRAIN)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.detach()

    def evaluate_batch(self, batch, stage):
        inputs, targets = batch
        predictions = self.compute_forward(inputs, targets, stage=stage)
        loss = self.compute_objectives(predictions, targets, stage=stage)
        return loss.detach()

    def on_stage_start(self, stage, epoch=None):
        self.ctc_metrics = self.ctc_stats()
        self.seq_metrics = self.seq_stats()

        if stage != sb.Stage.TRAIN:
            self.per_metrics = self.per_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        else:
            per = self.per_metrics.summarize("error_rate")

        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.lr_annealing(per)
            sb.nnet.update_learning_rate(self.optimizer, new_lr)
            self.train_logger.log_stats(
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
            self.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.epoch_counter.current},
                test_stats={"loss": stage_loss, "PER": per},
            )
            with open(self.wer_file, "w") as w:
                w.write("CTC loss stats:\n")
                self.ctc_metrics.write_stats(w)
                w.write("\nseq2seq loss stats:\n")
                self.seq_metrics.write_stats(w)
                w.write("\nPER stats:\n")
                self.per_metrics.write_stats(w)
                print(
                    "CTC, seq2seq, and PER stats written to file", self.wer_file
                )


if __name__ == "__main__":
    # This hack needed to import data preparation script from ../..
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(os.path.dirname(current_dir)))
    from timit_prepare import prepare_timit  # noqa E402

    # Load hyperparameters file with command-line overrides
    params_file, overrides = sb.parse_arguments(sys.argv[1:])
    with open(params_file) as fin:
        params = sb.load_extended_yaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=params.output_folder,
        hyperparams_to_save=params_file,
        overrides=overrides,
    )

    # Prepare data
    prepare_timit(
        data_folder=params.data_folder,
        splits=["train", "dev", "test"],
        save_folder=params.data_folder,
    )

    train_set = params.train_loader()
    valid_set = params.valid_loader()
    ind2lab = params.train_loader.label_dict["phn"]["index2lab"]
    asr_brain = ASR(
        modules=dict(params.modules, ind2lab=ind2lab),
        optimizers=["optimizer"],
        jit_modules=["enc"],
        device=params.device,
    )

    # Load latest checkpoint to resume training
    asr_brain.checkpointer.recover_if_possible()
    asr_brain.fit(params.epoch_counter, train_set, valid_set)

    # Load best checkpoint for evaluation
    asr_brain.checkpointer.recover_if_possible(min_key="PER")
    test_stats = asr_brain.evaluate(params.test_loader())
