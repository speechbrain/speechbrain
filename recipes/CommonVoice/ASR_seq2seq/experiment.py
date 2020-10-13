#!/usr/bin/env python3
import os
import sys
import torch
import speechbrain as sb
from speechbrain.utils.data_utils import download_file
from speechbrain.tokenizers.SentencePiece import SentencePiece
from speechbrain.utils.data_utils import undo_padding
"""Recipe for training a sequence-to-sequence ASR system with CommonVoice.
The system employs an encoder, a decoder, and an attention mechanism
between them. Decoding is performed with beamsearch and can be coupled with
a neural language model.

To run this recipe, do the following:
> python experiment.py hyperparams.yaml

With the default hyperparameters, the system employs a CRDNN encoder.
The decoder is based on a standard GRU. Beamsearch coupled with a RNN
language model is used  on the top of decoder probabilities.

The neural network is trained on both CTC and negative-log likelihood
targets and sub-word units estimated with Byte Pairwise Encoding (BPE)
are used as basic recognition tokens. Training is performed on the full
LibriSpeech dataset (960 h).

The experiment file is flexible enough to support a large variety of
different systems. By properly changing the parameter files, you can try
different encoders, decoders, tokens (e.g, characters instead of BPE),
training languages (all CommonVoice languages), and many
other possible variations.


Authors
 * Titouan Parcollet 2020
"""


# Define training procedure
class ASR(sb.core.Brain):
    def compute_forward(self, x, y, stage):
        """
            Forward computations from the waveform batches to the output
            probabilities.
        """
        ids, wavs, wav_lens = x
        ids, target_words, target_word_lens = y
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)

        # Add augmentation if specified
        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs, wav_lens)

        # Prepare labels
        target_tokens, _ = self.hparams.tokenizer(
            target_words, target_word_lens, self.hparams.ind2lab, task="encode"
        )
        target_tokens = target_tokens.to(self.device)
        y_in = sb.data_io.prepend_bos_token(
            target_tokens, self.hparams.bos_index
        )

        # Forward pass
        feats = self.hparams.compute_features(wavs)
        feats = self.hparams.normalize(feats, wav_lens)

        # We detach as we don't need the features to be on the backward graph
        x = self.hparams.enc(feats.detach())
        e_in = self.hparams.emb(y_in)
        h, _ = self.hparams.dec(e_in, x, wav_lens)

        # Output layer for seq2seq log-probabilities
        logits = self.hparams.seq_lin(h)
        p_seq = self.hparams.log_softmax(logits)

        # Compute outputs
        if stage == sb.Stage.TRAIN:
            current_epoch = self.hparams.epoch_counter.current
            if current_epoch <= self.hparams.number_of_ctc_epochs:
                # Output layer for ctc log-probabilities
                logits = self.hparams.ctc_lin(x)
                p_ctc = self.hparams.log_softmax(logits)
                return p_ctc, p_seq, wav_lens
            else:
                return p_seq, wav_lens

        else:
            p_tokens, scores = self.hparams.beam_searcher(x, wav_lens)
            return p_seq, wav_lens, p_tokens


    def compute_objectives(self, predictions, targets, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""

        current_epoch = self.hparams.epoch_counter.current
        if stage == sb.Stage.TRAIN:
            if current_epoch <= self.hparams.number_of_ctc_epochs:
                p_ctc, p_seq, wav_lens = predictions
            else:
                p_seq, wav_lens = predictions
        else:
            p_seq, wav_lens, predicted_tokens = predictions

        ids, target_words, target_word_lens = targets
        target_tokens, target_token_lens = self.hparams.tokenizer(
            target_words, target_word_lens, self.hparams.ind2lab, task="encode"
        )
        target_tokens = target_tokens.to(self.device)
        target_token_lens = target_token_lens.to(self.device)

        # Add char_lens by one for eos token
        abs_length = torch.round(target_token_lens * target_tokens.shape[1])

        # Append eos token at the end of the label sequences
        target_tokens_with_eos = sb.data_io.append_eos_token(
            target_tokens, length=abs_length, eos_index=self.hparams.eos_index
        )

        # convert to speechbrain-style relative length
        rel_length = (abs_length + 1) / target_tokens_with_eos.shape[1]
        loss_seq = self.hparams.seq_cost(
            p_seq, target_tokens_with_eos, length=rel_length
        )

        # Add ctc loss if necessary
        if (
            stage == sb.Stage.TRAIN
            and current_epoch <= self.hparams.number_of_ctc_epochs
        ):
            loss_ctc = self.hparams.ctc_cost(
                p_ctc, target_tokens, wav_lens, target_token_lens
            )

            # Lctc * w_ctc + (1 - w_ctc) * Lseq
            loss = self.hparams.ctc_weight * loss_ctc
            loss += (1 - self.hparams.ctc_weight) * loss_seq
        else:
            loss = loss_seq

        if stage != sb.Stage.TRAIN:
            # Decode token terms to words
            predicted_words = self.hparams.tokenizer(
                predicted_tokens, task="decode_from_list"
            )

            # Convert indices to words
            target_words = undo_padding(target_words, target_word_lens)
            target_words = sb.data_io.convert_index_to_lab(
                target_words, self.hparams.ind2lab
            )

            self.wer_metric.append(ids, predicted_words, target_words)
            self.cer_metric.append(ids, predicted_words, target_words)

        return loss

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""
        inputs, targets = batch
        predictions = self.compute_forward(inputs, targets, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, targets, sb.Stage.TRAIN)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.detach()

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        inputs, targets = batch
        predictions = self.compute_forward(inputs, targets, stage=stage)
        loss = self.compute_objectives(predictions, targets, stage=stage)
        return loss.detach()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(stage_stats["WER"])
            sb.nnet.update_learning_rate(self.optimizer, new_lr)
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"WER": stage_stats["WER"]}, min_keys=["WER"],
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.wer_file, "w") as w:
                self.wer_metric.write_stats(w)

    def load_tokenizer(self):
        """Loads the sentence piece tokenizer specified in the yaml file"""
        save_model_path = self.hparams.save_folder + "/tok_unigram.model"
        save_vocab_path = self.hparams.save_folder + "/tok_unigram.vocab"

        if hasattr(self.hparams, "tok_mdl_file"):
            download_file(
                source=self.hparams.tok_mdl_file,
                dest=save_model_path,
                replace_existing=True,
            )
            self.hparams.tokenizer.sp.load(save_model_path)

        if hasattr(self.hparams, "tok_voc_file"):
            download_file(
                source=self.hparams.tok_voc_file,
                dest=save_vocab_path,
                replace_existing=True,
            )

    def init_optimizers(self):
        """Initializes the optmizers (needed to support DDP)"""
        self.optimizer = self.opt_class(self.hparams.model.parameters())
        self.checkpointer.add_recoverable("optimizer", self.optimizer)

if __name__ == "__main__":

    # This hack needed to import data preparation script from ..
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(current_dir))
    from common_voice_prepare import prepare_common_voice  # noqa E402

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
    prepare_common_voice(
        data_folder=hparams["data_folder"],
        save_folder=hparams["save_folder"],
        path_to_wav=hparams["wav_folder"],
        train_tsv_file=hparams["train_tsv_file"],
        dev_tsv_file=hparams["dev_tsv_file"],
        test_tsv_file=hparams["test_tsv_file"],
        accented_letters=hparams["accented_letters"],
        language=hparams["language"],
    )

    # Creating tokenizer must be done after preparation
    # Specify the bos_id/eos_id if different from blank_id
    tokenizer = SentencePiece(
        model_dir=hparams["save_folder"],
        vocab_size=hparams["output_neurons"],
        csv_train=hparams["csv_train"],
        csv_read="wrd",
        model_type=hparams["token_type"],
        character_coverage=1.0,
    )

    # Train the tokenizer
    tokenizer.train()

    # Load DataLoaders :-)
    # Load ind2label dict for decoding
    train_set = hparams["train_loader"]()
    valid_set = hparams["valid_loader"]()
    test_set = hparams["test_loader"]()
    ind2lab = hparams["test_loader"].label_dict["wrd"]["index2lab"]
    hparams["hparams"]["ind2lab"] = ind2lab
    hparams["hparams"]["tokenizer"] = tokenizer

    # Brain class initialization
    asr_brain = ASR(
        hparams=hparams["hparams"],
        opt_class=hparams["opt_class"],
        checkpointer=hparams["checkpointer"],
        device=hparams["device"],
        ddp_procs=hparams["ddp_procs"],
    )

    # Training
    asr_brain.fit(asr_brain.hparams.epoch_counter, train_set, valid_set)

    # TO MODIFY
    #asr_brain.load_tokenizer()

    # Test
    asr_brain.hparams.wer_file = (
        hparams["output_folder"] + "/wer_test.txt"
    )
    asr_brain.evaluate(test_clean_set)
