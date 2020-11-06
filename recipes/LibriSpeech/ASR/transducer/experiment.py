#!/usr/bin/env/python3
"""Recipe for training a Transducer ASR system with librispeech.
The system employs an encoder, a decoder, and an joint network
between them. Decoding is performed with beamsearch coupled with a neural
language model.

To run this recipe, do the following:
> python experiment.py hyperparams.yaml

With the default hyperparameters, the system employs a CRDNN encoder.
The decoder is based on a standard  GRU. Beamsearch coupled with a RNN
language model is used on the top of decoder probabilities.

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
 * Abdel Heba 2020
 * Mirco Ravanelli 2020
 * Ju-Chieh Chou 2020
 * Peter Plantinga 2020
"""

import os
import sys
import torch
import speechbrain as sb
from speechbrain.utils.data_utils import download_file
from speechbrain.tokenizers.SentencePiece import SentencePiece
from speechbrain.utils.data_utils import undo_padding
from speechbrain.decoders.transducer import TransducerBeamSearcher


# Define training procedure
class ASR(sb.Brain):
    def compute_forward(self, x, y, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        ids, wavs, wav_lens = x
        ids, target_words, target_word_lens = y
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)

        # Add augmentation if specified
        if stage == sb.Stage.TRAIN:
            if hasattr(self.modules, "env_corrupt"):
                wavs_noise = self.modules.env_corrupt(wavs, wav_lens)
                wavs = torch.cat([wavs, wavs_noise], dim=0)
                wav_lens = torch.cat([wav_lens, wav_lens])
                target_words = torch.cat([target_words, target_words], dim=0)
                target_word_lens = torch.cat(
                    [target_word_lens, target_word_lens]
                )
            if hasattr(self.modules, "augmentation"):
                wavs = self.modules.augmentation(wavs, wav_lens)

        # Prepare labels
        target_tokens, _ = self.hparams.tokenizer(
            target_words, target_word_lens, self.hparams.ind2lab, task="encode"
        )
        target_tokens = target_tokens.to(self.device)
        y_in = sb.data_io.data_io.prepend_bos_token(
            target_tokens, self.hparams.blank_index
        )

        # Forward pass
        feats = self.hparams.compute_features(wavs)
        feats = self.modules.normalize(feats, wav_lens)
        x = self.modules.enc(feats.detach())
        e_in = self.modules.emb(y_in)
        h, _ = self.modules.dec(e_in)
        # Joint network
        # add labelseq_dim to the encoder tensor: [B,T,H_enc] => [B,T,1,H_enc]
        # add timeseq_dim to the decoder tensor: [B,U,H_dec] => [B,1,U,H_dec]
        joint = self.modules.Tjoint(x.unsqueeze(2), h.unsqueeze(1))

        # Output layer for transducer log-probabilities
        logits = self.modules.transducer_lin(joint)
        p_transducer = self.hparams.log_softmax(logits)

        # Compute outputs
        if stage == sb.Stage.TRAIN:
            return_CTC = False
            return_CE = False
            current_epoch = self.hparams.epoch_counter.current
            if (
                hasattr(self.hparams, "ctc_cost")
                and current_epoch <= self.hparams.number_of_ctc_epochs
            ):
                return_CTC = True
                # Output layer for ctc log-probabilities
                out_ctc = self.modules.enc_lin(x)
                p_ctc = self.hparams.log_softmax(out_ctc)
            if (
                hasattr(self.hparams, "ce_cost")
                and current_epoch <= self.hparams.number_of_ce_epochs
            ):
                return_CE = True
                # Output layer for ctc log-probabilities
                p_ce = self.modules.dec_lin(h)
                p_ce = self.hparams.log_softmax(p_ce)
            if return_CE and return_CTC:
                return p_ctc, p_ce, p_transducer, wav_lens
            elif return_CTC:
                return p_ctc, p_transducer, wav_lens
            elif return_CE:
                return p_ce, p_transducer, wav_lens
            else:
                return p_transducer, wav_lens

        elif stage == sb.Stage.VALID:
            predicted_tokens, scores, _, _ = searcher.forward(x)
            return p_transducer, wav_lens, predicted_tokens
        else:
            best_hyps, best_scores, nbest_hyps, nbest_scores = searcher.forward(
                x
            )
            return p_transducer, wav_lens, best_hyps

    def compute_objectives(self, predictions, targets, stage):
        """Computes the loss (CTC+Transducer) given predictions and targets."""
        ids, target_words, target_word_lens = targets
        target_tokens, target_token_lens = self.hparams.tokenizer(
            target_words, target_word_lens, self.hparams.ind2lab, task="encode"
        )
        target_tokens = target_tokens.to(self.device)
        target_token_lens = target_token_lens.to(self.device)
        if hasattr(self.modules, "env_corrupt") and stage == sb.Stage.TRAIN:
            target_tokens = torch.cat([target_tokens, target_tokens], dim=0)
            target_token_lens = torch.cat(
                [target_token_lens, target_token_lens], dim=0
            )

        current_epoch = self.hparams.epoch_counter.current
        if stage == sb.Stage.TRAIN:
            if len(predictions) == 4:
                p_ctc, p_ce, p_transducer, wav_lens = predictions
                CTC_loss = self.hparams.ctc_cost(
                    p_ctc, target_tokens, wav_lens, target_token_lens
                )
                # generate output sequence for decoder + CE loss
                abs_length = torch.round(
                    target_token_lens * target_tokens.shape[1]
                )
                target_tokens_with_eos = sb.data_io.data_io.append_eos_token(
                    target_tokens,
                    length=abs_length,
                    eos_index=self.hparams.blank_index,
                )
                rel_length = (abs_length + 1) / target_tokens_with_eos.shape[1]
                CE_loss = self.hparams.ce_cost(
                    p_ce, target_tokens_with_eos, length=rel_length
                )
                target_tokens = target_tokens.long()
                loss_transducer = self.hparams.transducer_cost(
                    p_transducer, target_tokens, wav_lens, target_token_lens
                )
                loss = (
                    self.hparams.ctc_weight * CTC_loss
                    + self.hparams.ce_weight * CE_loss
                    + (1 - (self.hparams.ctc_weight + self.hparams.ce_weight))
                    * loss_transducer
                )
            elif len(predictions) == 3:
                # one of the 2 heads (CTC or CE) is still computed
                # CTC alive
                if current_epoch <= self.hparams.number_of_ctc_epochs:
                    p_ctc, p_transducer, wav_lens = predictions
                    CTC_loss = self.hparams.ctc_cost(
                        p_ctc, target_tokens, wav_lens, target_token_lens
                    )
                    target_tokens = target_tokens.long()
                    loss_transducer = self.hparams.transducer_cost(
                        p_transducer, target_tokens, wav_lens, target_token_lens
                    )
                    loss = (
                        self.hparams.ctc_weight * CTC_loss
                        + (1 - self.hparams.ctc_weight) * loss_transducer
                    )
                # CE for decoder alive
                else:
                    p_ce, p_transducer, wav_lens = predictions
                    # generate output sequence for decoder + CE loss
                    abs_length = torch.round(
                        target_token_lens * target_tokens.shape[1]
                    )
                    target_tokens_with_eos = sb.data_io.append_eos_token(
                        target_tokens,
                        length=abs_length,
                        eos_index=self.hparams.blank_index,
                    )
                    rel_length = (
                        abs_length + 1
                    ) / target_tokens_with_eos.shape[1]
                    CE_loss = self.hparams.ce_cost(
                        p_ce, target_tokens_with_eos, length=rel_length
                    )
                    target_tokens = target_tokens.long()
                    loss_transducer = self.hparams.transducer_cost(
                        p_transducer, target_tokens, wav_lens, target_token_lens
                    )
                    loss = (
                        self.hparams.ce_weight * CE_loss
                        + (1 - self.hparams.ctc_weight) * loss_transducer
                    )
            else:
                p_transducer, wav_lens = predictions
                target_tokens = target_tokens.long()
                loss = self.hparams.transducer_cost(
                    p_transducer, target_tokens, wav_lens, target_token_lens
                )
        else:
            p_transducer, wav_lens, predicted_tokens = predictions
            target_tokens = target_tokens.long()
            loss = self.hparams.transducer_cost(
                p_transducer, target_tokens, wav_lens, target_token_lens
            )

        if stage != sb.Stage.TRAIN:
            # Decode token terms to words
            predicted_words = self.hparams.tokenizer(
                predicted_tokens, task="decode_from_list"
            )

            # Convert indices to words
            target_words = undo_padding(target_words, target_word_lens)
            target_words = sb.data_io.data_io.convert_index_to_lab(
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
        """Gets called at the end of a epoch."""
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
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
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
        """Loads the sentence piece tokinizer specified in the yaml file"""
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


if __name__ == "__main__":
    # This hack needed to import data preparation script from ../..
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(os.path.dirname(current_dir)))
    from librispeech_prepare import prepare_librispeech  # noqa E402

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
    # Specify the bos_id/eos_id if different from blank_id
    hparams["tokenizer"] = SentencePiece(
        model_dir=hparams["save_folder"],
        vocab_size=hparams["output_neurons"],
        csv_train=hparams["csv_train"],
        csv_read="wrd",
        model_type=hparams["token_type"],
        character_coverage=1.0,
    )

    # Load index2label dict for decoding
    train_set = hparams["train_loader"]()
    valid_set = hparams["valid_loader"]()
    test_clean_set = hparams["test_clean_loader"]()
    test_other_set = hparams["test_other_loader"]()
    hparams["ind2lab"] = hparams["test_other_loader"].label_dict["wrd"][
        "index2lab"
    ]

    # Brain class initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        checkpointer=hparams["checkpointer"],
    )

    asr_brain.load_tokenizer()
    if hasattr(asr_brain.hparams, "lm_ckpt_file"):
        asr_brain.load_lm()

    # Searcher
    # TODO (Brian) move this part to yaml file
    searcher = TransducerBeamSearcher(
        decode_network_lst=[hparams["emb"], hparams["dec"]],
        tjoint=hparams["Tjoint"],
        classifier_network=[hparams["transducer_lin"]],
        blank_id=0,
        beam_size=hparams["beam_size"],
        nbest=hparams["nbest"],
        lm_module=hparams["lm_model"],
        lm_weight=hparams["lm_weight"],
        state_beam=2.3,
        expand_beam=2.3,
    )

    # Training
    asr_brain.fit(asr_brain.hparams.epoch_counter, train_set, valid_set)

    # Test
    asr_brain.hparams.wer_file = (
        hparams["output_folder"] + "/wer_test_clean.txt"
    )
    asr_brain.evaluate(test_clean_set)
    asr_brain.hparams.wer_file = (
        hparams["output_folder"] + "/wer_test_other.txt"
    )
    asr_brain.evaluate(test_other_set)
