#!/usr/bin/env python3
import os
import sys
import torch
import speechbrain as sb
from speechbrain.utils.data_utils import download_file


# Define training procedure
class ASR(sb.Brain):
    def compute_forward(self, x, y, stage):
        ids, wavs, wav_lens = x
        ids, words, word_lens = y
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)

        # Add augmentation if specified
        if stage == sb.Stage.TRAIN:
            if hasattr(self, "env_corrupt"):
                wavs_noise = self.env_corrupt(wavs, wav_lens)
                wavs = torch.cat([wavs, wavs_noise], dim=0)
                wav_lens = torch.cat([wav_lens, wav_lens])
                words = torch.cat([words, words], dim=0)
                word_lens = torch.cat([word_lens, word_lens])
            if hasattr(self, "augmentation"):
                wavs = self.augmentation(wavs, wav_lens)

        # Prepare labels
        bpe, _ = self.bpe_tokenizer(
            words, word_lens, self.index2lab, task="encode"
        )
        bpe = bpe.to(self.device)
        y_in = sb.data_io.data_io.prepend_bos_token(bpe, self.bos_index)

        # Forward pass
        feats = self.compute_features(wavs)
        feats = self.normalize(feats, wav_lens)
        x = self.enc(feats)
        e_in = self.emb(y_in)
        h, _ = self.dec(e_in, x, wav_lens)

        # output layer for seq2seq log-probabilities
        logits = self.seq_lin(h)
        p_seq = self.log_softmax(logits)

        # Compute outputs
        if stage == sb.Stage.TRAIN:
            if self.epoch_counter.current <= self.number_of_ctc_epochs:
                # output layer for ctc log-probabilities
                logits = self.ctc_lin(x)
                p_ctc = self.log_softmax(logits)
                return p_ctc, p_seq, wav_lens
            else:
                return p_seq, wav_lens
        elif stage == sb.Stage.VALID:
            hyps, scores = self.greedy_searcher(x, wav_lens)
            return p_seq, wav_lens, hyps
        elif stage == sb.Stage.TEST:
            hyps, scores = self.beam_searcher(x, wav_lens)
            return p_seq, wav_lens, hyps

    def compute_objectives(self, predictions, targets, stage):
        if stage == sb.Stage.TRAIN:
            if self.epoch_counter.current <= self.number_of_ctc_epochs:
                p_ctc, p_seq, wav_lens = predictions
            else:
                p_seq, wav_lens = predictions
        else:
            p_seq, wav_lens, hyps = predictions

        ids, words, word_lens = targets
        bpe, bpe_lens = self.bpe_tokenizer(
            words, word_lens, self.index2lab, task="encode"
        )
        bpe, bpe_lens = bpe.to(self.device), bpe_lens.to(self.device)
        if hasattr(self, "env_corrupt") and stage == sb.Stage.TRAIN:
            bpe = torch.cat([bpe, bpe], dim=0)
            bpe_lens = torch.cat([bpe_lens, bpe_lens], dim=0)

        # Add char_lens by one for eos token
        abs_length = torch.round(bpe_lens * bpe.shape[1])

        # Append eos token at the end of the label sequences
        bpe_with_eos = sb.data_io.data_io.append_eos_token(
            bpe, length=abs_length, eos_index=self.eos_index
        )

        # convert to speechbrain-style relative length
        rel_length = (abs_length + 1) / bpe_with_eos.shape[1]
        loss_seq = self.seq_cost(p_seq, bpe_with_eos, length=rel_length)

        # Add ctc loss if necessary
        if (
            stage == sb.Stage.TRAIN
            and self.epoch_counter.current <= self.number_of_ctc_epochs
        ):
            loss_ctc = self.ctc_cost(p_ctc, bpe, wav_lens, bpe_lens)
            loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_seq
        else:
            loss = loss_seq

        if stage != sb.Stage.TRAIN:
            # Decode BPE terms to words
            target = self.bpe_tokenizer(hyps, task="decode_from_list")

            # Convert indices to words
            words = sb.decoders.undo_padding(words, word_lens)
            words = sb.data_io.data_io.convert_index_to_lab(
                words, self.index2lab
            )

            if self.ter_eval:
                self.ter_metric.append(ids, bpe, hyps, predict_len=bpe_lens)
            self.wer_metric.append(ids, words, target)
            self.cer_metric.append(ids, words, target)

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
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.cer_computer()
            self.wer_metric = self.error_rate_computer()
            if self.ter_eval:
                self.ter_metric = self.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch=None):

        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")
            if self.ter_eval:
                stage_stats["TER"] = self.ter_metric.summarize("error_rate")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.lr_annealing(stage_stats["WER"])
            sb.nnet.update_learning_rate(self.optimizer, new_lr)
            self.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"WER": stage_stats["WER"]}, min_keys=["WER"],
            )
        elif stage == sb.Stage.TEST:
            self.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.wer_file, "w") as w:
                self.wer_metric.write_results(w)

    def load_tokenizer(self):
        save_model_path = self.save_folder + "/tok_unigram.model"
        save_vocab_path = self.save_folder + "/tok_unigram.vocab"

        if hasattr(self, "tok_mdl_file"):
            download_file(
                self.tok_mdl_file, save_model_path, replace_existing=True
            )
            self.bpe_tokenizer.sp.load(save_model_path)
        if hasattr(self, "tok_voc_file"):
            download_file(
                self.tok_voc_file, save_vocab_path, replace_existing=True
            )

    def load_lm(self):
        save_model_path = self.output_folder + "/save/lm_model.ckpt"
        download_file(self.lm_ckpt_file, save_model_path)
        state_dict = torch.load(save_model_path)
        # Removing prefix
        state_dict = {k.split(".", 1)[1]: v for k, v in state_dict.items()}
        self.lm_model.load_state_dict(state_dict, strict=True)


if __name__ == "__main__":
    # This hack needed to import data preparation script from ../..
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(os.path.dirname(current_dir)))
    from librispeech_prepare import prepare_librispeech  # noqa E402

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
    prepare_librispeech(
        data_folder=params.data_folder,
        splits=params.train_splits + [params.dev_split, params.test_split],
        merge_lst=params.train_splits,
        merge_name=params.csv_train,
        save_folder=params.data_folder,
    )

    train_set = params.train_loader()
    valid_set = params.valid_loader()
    test_set = params.test_loader()
    ind2lab = params.test_loader.label_dict["wrd"]["index2lab"]
    asr_brain = ASR(
        modules=dict(params.modules, index2lab=ind2lab),
        optimizers=["optimizer"],
        device=params.device,
    )

    # Load latest checkpoint to resume training
    asr_brain.load_tokenizer()
    asr_brain.checkpointer.recover_if_possible()
    asr_brain.fit(asr_brain.epoch_counter, train_set, valid_set)

    # Load best checkpoint for evaluation
    asr_brain.checkpointer.recover_if_possible(min_key="WER")
    if hasattr(params, "lm_ckpt_file"):
        asr_brain.load_lm()
    test_stats = asr_brain.evaluate(test_set)
