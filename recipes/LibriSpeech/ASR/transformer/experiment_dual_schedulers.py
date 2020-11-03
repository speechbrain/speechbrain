#!/usr/bin/env python3
"""Recipe for training a Transformer ASR system with librispeech.
The system employs an encoder, a decoder, and an attention mechanism
between them. Decoding is performed with (CTC/Att joint) beamsearch coupled with a neural
language model.

To run this recipe, do the following:
> python experiment.py hyperparams.yaml

With the default hyperparameters, the system employs a convolutional frontend (ContextNet) and a transformer.
The decoder is based on a Transformer decoder. Beamsearch coupled with a Transformer
language model is used  on the top of decoder probabilities.

The neural network is trained on both CTC and negative-log likelihood
targets and sub-word units estimated with Byte Pairwise Encoding (BPE)
are used as basic recognition tokens. Training is performed on the full
LibriSpeech dataset (960 h).

The experiment file is flexible enough to support a large variety of
different systems. By properly changing the parameter files, you can try
different encoders, decoders, tokens (e.g, characters instead of BPE),
training split (e.g, train-clean 100 rather than the full one), and many
other possible variations.


Authors
 * Jianyuan Zhong 2020
 * Mirco Ravanelli 2020
 * Peter Plantinga 2020
"""
import os
import sys
import torch

import speechbrain as sb
from speechbrain.utils.data_utils import download_file
from speechbrain.utils.data_utils import undo_padding

import logging

logger = logging.getLogger(__name__)


# Define training procedure
class ASR(sb.core.Brain):
    def compute_forward(self, x, y, stage):
        ids, wavs, wav_lens = x
        ids, target_words, target_word_lens = y
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)

        # Add augmentation if specified
        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "env_corrupt"):
                wavs_noise = self.hparams.env_corrupt(wavs, wav_lens)
                wavs = torch.cat([wavs, wavs_noise], dim=0)
                wav_lens = torch.cat([wav_lens, wav_lens])
                target_words = torch.cat([target_words, target_words], dim=0)
                target_word_lens = torch.cat(
                    [target_word_lens, target_word_lens]
                )
            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs, wav_lens)

        # Prepare labels
        target_tokens, target_tokens_len = self.hparams.tokenizer(
            target_words, target_word_lens, self.hparams.ind2lab, task="encode"
        )
        y_in = sb.data_io.prepend_bos_token(
            target_tokens, self.hparams.bos_index
        ).to(self.device)
        target_tokens = target_tokens.to(self.device)
        target_tokens_len = target_tokens_len.to(self.device)

        # forward pass
        feats = self.hparams.compute_features(wavs)
        # feats = self.hparams.normalize(feats, wav_lens)
        feats = self.hparams.normalize(feats)
        src = self.hparams.CNN(feats)
        enc_out, pred = self.hparams.Transformer(
            src, y_in, wav_lens, pad_idx=self.hparams.pad_index
        )

        # output layer for ctc log-probabilities
        logits = self.hparams.ctc_lin(enc_out)
        p_ctc = self.hparams.log_softmax(logits)

        # output layer for seq2seq log-probabilities
        pred = self.hparams.seq_lin(pred)
        p_seq = self.hparams.log_softmax(pred)

        # Compute outputs
        if stage == sb.Stage.TRAIN:
            hyps = None
        elif stage == sb.Stage.VALID:
            hyps = None
            current_epoch = self.hparams.epoch_counter.current
            if current_epoch % self.hparams.valid_search_interval == 0:
                hyps, _ = self.hparams.valid_search(enc_out.detach(), wav_lens)
        else:
            hyps, _ = self.hparams.test_search(enc_out.detach(), wav_lens)

        return p_ctc, p_seq, wav_lens, hyps, target_tokens, target_tokens_len

    def compute_objectives(self, predictions, targets, stage):
        (
            p_ctc,
            p_seq,
            wav_lens,
            hyps,
            target_tokens,
            target_tokens_len,
        ) = predictions
        ids, target_words, target_word_lens = targets

        # Add char_lens by one for eos token
        abs_length = torch.round(target_tokens_len * target_tokens.shape[1])

        # Append eos token at the end of the label sequences
        target_tokens_with_eos = sb.data_io.append_eos_token(
            target_tokens, length=abs_length, eos_index=self.hparams.eos_index
        )
        # convert to speechbrain-style relative length
        rel_length = (abs_length + 1) / target_tokens_with_eos.shape[1]

        loss_seq = self.hparams.seq_cost(
            p_seq, target_tokens_with_eos, rel_length
        )
        loss_ctc = self.hparams.ctc_cost(
            p_ctc, target_tokens, wav_lens, target_tokens_len
        )
        loss = (
            self.hparams.ctc_weight * loss_ctc
            + (1 - self.hparams.ctc_weight) * loss_seq
        )

        if stage != sb.Stage.TRAIN:
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval
            if (
                current_epoch % valid_search_interval == 0
                or stage == sb.Stage.TEST
            ):
                # Decode token terms to words
                predicted_words = self.hparams.tokenizer(
                    hyps, task="decode_from_list"
                )

                # Convert indices to words
                target_words = undo_padding(target_words, target_word_lens)
                target_words = sb.data_io.convert_index_to_lab(
                    target_words, self.hparams.ind2lab
                )

                self.wer_metric.append(ids, predicted_words, target_words)
            # compute the accuracy of the one-step-forward prediction
            self.acc_metric.append(p_seq, target_tokens_with_eos, rel_length)
        return loss

    def fit_batch(self, batch):
        inputs, targets = batch
        predictions = self.compute_forward(inputs, targets, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, targets, sb.Stage.TRAIN)

        # normalize the loss by gradient_accumulation step
        (loss / self.hparams.gradient_accumulation).backward()

        if self.step % self.hparams.gradient_accumulation == 0:
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(self.modules.parameters(), 5.0)

            self.optimizer.step()
            self.optimizer.zero_grad()

            # anneal lr every update
            old_lr, new_lr = self.hparams.noam_annealing(self.optimizer)
            old_lr, new_lr = self.hparams.cosine_annealing(self.optimizer)

        return loss.detach()

    def evaluate_batch(self, batch, stage="valid"):
        """Computations needed for validation/test batches"""
        inputs, targets = batch
        predictions = self.compute_forward(inputs, targets, stage=stage)
        loss = self.compute_objectives(predictions, targets, stage=stage)
        return loss.detach()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.acc_metric = self.hparams.acc_computer()
            self.wer_metric = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["ACC"] = self.acc_metric.summarize()
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval
            if (
                current_epoch % valid_search_interval == 0
                or stage == sb.Stage.TEST
            ):
                stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        # log stats and save checkpoint at end-of-epoch
        if stage == sb.Stage.VALID and self.root_process:
            epoch_stats = {
                "epoch": epoch,
                "lr": self.hparams.cosine_annealing.current_lr,
                "steps": self.hparams.cosine_annealing.n_steps,
            }
            self.hparams.train_logger.log_stats(
                stats_meta=epoch_stats,
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"ACC": stage_stats["ACC"], "epoch": epoch},
                max_keys=["ACC"],
            )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.wer_file, "w") as w:
                self.wer_metric.write_stats(w)

    def load_tokenizer(self):
        """Loads the sentence piece tokinizer specified in the yaml file"""
        save_model_path = self.hparams.save_folder + "/{}_unigram.model".format(
            self.hparams.vocab_size
        )
        save_vocab_path = self.hparams.save_folder + "/{}_unigram.vocab".format(
            self.hparams.vocab_size
        )

        if hasattr(self.hparams, "tok_mdl_file"):
            download_file(
                self.hparams.tok_mdl_file,
                save_model_path,
                replace_existing=True,
            )
            self.hparams.tokenizer.sp.load(save_model_path)
        if hasattr(self.hparams, "tok_voc_file"):
            download_file(
                self.hparams.tok_voc_file,
                save_vocab_path,
                replace_existing=True,
            )

    def load_lm(self):
        """Loads the LM specified in the yaml file"""
        save_model_path = os.path.join(
            self.hparams.output_folder, "save", "lm_model.ckpt"
        )
        download_file(self.hparams.lm_ckpt_file, save_model_path)

        # Load downloaded model, removing prefix
        state_dict = torch.load(save_model_path)
        state_dict = {k.split(".", 1)[1]: v for k, v in state_dict.items()}
        self.hparams.lm_model.load_state_dict(state_dict, strict=True)
        self.hparams.lm_model.eval()
        logger.info("loaded LM from {}".format(save_model_path))

    def on_fit_start(self):
        torch.cuda.set_device(self.device)
        self.checkpointer.add_recoverable("model", self.modules)
        super().on_fit_start()

    def on_evaluate_start(self, max_key=None, min_key=None):
        self.checkpointer.add_recoverable("model", self.modules)
        super().on_evaluate_start(max_key=max_key, min_key=min_key)


if __name__ == "__main__":
    # This hack needed to import data preparation script from ../..
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(os.path.dirname(current_dir)))
    from librispeech_prepare import prepare_librispeech  # noqa E402

    while True:
        try:
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
            prepare_librispeech(
                data_folder=hparams["data_folder"],
                splits=hparams["train_splits"]
                + [hparams["dev_split"], "test-clean", "test-other"],
                merge_lst=hparams["train_splits"],
                merge_name=hparams["csv_train"],
                save_folder=hparams["data_folder"],
            )

            # Creating tokenizer must be done after preparation
            tokenizer = hparams["tokenizer"]()

            # Load index2label dict for decoding
            train_set = hparams["train_loader"]()
            valid_set = hparams["valid_loader"]()
            test_clean_set = hparams["test_clean_loader"]()
            test_other_set = hparams["test_other_loader"]()
            ind2lab = hparams["test_other_loader"].label_dict["wrd"]["index2lab"]
            hparams["ind2lab"] = ind2lab
            hparams["tokenizer"] = tokenizer
            

            # Brain class initialization
            asr_brain = ASR(
                modules=hparams["model"],
                opt_class=hparams["optimizer"],
                hparams=hparams,
                checkpointer=hparams["checkpointer"],
            )

            asr_brain.load_tokenizer()
            if hasattr(asr_brain.hparams, "lm_ckpt_file"):
                asr_brain.load_lm()

            # Training
            asr_brain.fit(asr_brain.hparams.epoch_counter, train_set, valid_set)

            # Test
            asr_brain.hparams.wer_file = (
                hparams["output_folder"] + "/wer_test_clean.txt"
            )
            asr_brain.evaluate(test_clean_set, max_key="ACC")
            asr_brain.hparams.wer_file = (
                hparams["output_folder"] + "/wer_test_other.txt"
            )
            asr_brain.evaluate(test_other_set, max_key="ACC")

        except KeyboardInterrupt:
            sys.exit()

        except:
            msg = "rand {} failed with pkl I/O, retrying...."
            print(msg)

        
            
