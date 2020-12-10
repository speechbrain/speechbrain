#!/usr/bin/env/python3
"""
todo
"""

import os
import sys
import torch
import speechbrain as sb
import numpy as np
from speechbrain.utils.data_utils import download_file
from speechbrain.tokenizers.SentencePiece import SentencePiece
from speechbrain.utils.data_utils import undo_padding


# Define training procedure
class ASR(sb.Brain):
    def random_align(self, p_ctc, y, T, U):
        """
        p_ctc: (B, Tmax, #labels)
        y: (B, Umax)
        T: (B)
        U: (B)
        returns: list of alignments [ [0,1,0,0] , ... ]
        """
        B = p_ctc.shape[0]
        alignments = []
        for b in range(B):
            T_b = T[b].item() - 1
            U_b = U[b].item()
            alignment = list(np.random.permutation([0] * T_b + [1] * U_b))
            alignments.append(alignment)
        return alignments

    def ctc_align(self, p_ctc, y, T, U):
        """
        p_ctc: (B, Tmax, #labels)
        y: (B, Umax)
        T: (B)
        U: (B)
        returns: list of alignments [ [0,1,0,0] , ... ]
        """
        B = p_ctc.shape[0]
        alignments = []
        for b in range(B):
            T_b = T[b].item() - 1
            U_b = U[b].item()

            # Step 1: Get CTC alignment
            S = 2 * U_b + 1
            y_prime = []  # [_, y1, _, y2, _, y3, _]
            for i in range(S):
                label = (
                    self.hparams.blank_index
                    if (i + 1) % 2
                    else y[b, int(i / 2)].item()
                )
                y_prime.append(label)
            log_alpha = torch.log(torch.zeros(T_b, S))
            psi = torch.zeros(T_b, S).long()
            for t in range(0, T_b):
                if t == 0:
                    log_alpha[t, 0] = p_ctc[b, 0, self.hparams.blank_index]
                    log_alpha[t, 1] = p_ctc[b, 0, y_prime[1]]
                else:
                    for s in range(S):
                        if s == 0:
                            log_alpha[t, s] = (
                                log_alpha[t - 1, s] + p_ctc[b, t, y_prime[s]]
                            )
                            psi[t, s] = 0
                        if s == 1:
                            log_alpha[t, s] = (
                                torch.max(
                                    log_alpha[t - 1, s - 1 : s + 1], dim=0
                                )[0]
                                + p_ctc[b, t, y_prime[s]]
                            )
                            psi[t, s] = (
                                torch.max(
                                    log_alpha[t - 1, s - 1 : s + 1], dim=0
                                )[1]
                                + s
                                - 1
                            )
                        if s > 1:
                            if (
                                y_prime[s] == self.hparams.blank_index
                                or y_prime[s - 2] == y_prime[s]
                            ):
                                log_alpha[t, s] = (
                                    torch.max(
                                        log_alpha[t - 1, s - 1 : s + 1], dim=0
                                    )[0]
                                    + p_ctc[b, t, y_prime[s]]
                                )
                                psi[t, s] = (
                                    torch.max(
                                        log_alpha[t - 1, s - 1 : s + 1], dim=0
                                    )[1]
                                    + s
                                    - 1
                                )
                            else:
                                log_alpha[t, s] = (
                                    torch.max(
                                        log_alpha[t - 1, s - 2 : s + 1], dim=0
                                    )[0]
                                    + p_ctc[b, t, y_prime[s]]
                                )
                                psi[t, s] = (
                                    torch.max(
                                        log_alpha[t - 1, s - 2 : s + 1], dim=0
                                    )[1]
                                    + s
                                    - 2
                                )
            ctc_alignment = []
            s = (
                torch.max(log_alpha[T_b - 1, S - 2 : S], dim=0)[1].item()
                + (S - 1)
                - 1
            )
            ctc_alignment.append(y_prime[s])
            for t in range(T_b - 1, 0, -1):
                s = psi[t, s].item()
                ctc_alignment.append(y_prime[s])
            ctc_alignment.reverse()

            # Step 2: Convert CTC alignment to Transducer alignment
            # Step 2.1: Convert repeated labels to label + blanks
            current_label = ctc_alignment[0]
            for t in range(1, T_b):
                if ctc_alignment[t] == current_label:
                    ctc_alignment[t] = self.hparams.blank_index
                else:
                    current_label = ctc_alignment[t]
            # Step 2.2: Add a blank after each label
            transducer_alignment = []
            for a in ctc_alignment:
                if a != self.hparams.blank_index:
                    transducer_alignment.append(1)
                transducer_alignment.append(0)
            alignments.append(transducer_alignment)
        return alignments

    def gather_outputs_and_labels(
        self, encoder_out, predictor_out, y, alignments, T, U
    ):
        """
        encoder_out: (B, Tmax, d)
        predictor_out: (B, Umax+1, d)
        alignments: list of alignments [ [0,1,0,0] , ... ]
        T: (B)
        U: (U)
        returns: joiner_output (B, max(T+U), #labels)
        """
        B = encoder_out.shape[0]
        output_list = []
        labels_list = []
        T_U_max = (T + U).max().item()
        for b in range(B):
            t = 0
            u = 0
            t_u_indices = []
            y_expanded = []
            for step in alignments[b]:
                t_u_indices.append((t, u))
                if step == 0:  # right (null)
                    y_expanded.append(self.hparams.blank_index)
                    t += 1
                if step == 1:  # down (label)
                    y_expanded.append(y[b, u].item())
                    u += 1
            # t_u_indices.append((T[b].item() - 1, U[b].item()))
            # y_expanded.append(self.hparams.blank_index)
            t_indices = [t for (t, u) in t_u_indices]
            u_indices = [u for (t, u) in t_u_indices]
            combined = encoder_out[b, t_indices] + predictor_out[b, u_indices]
            combined = torch.nn.functional.pad(
                combined, (0, 0, 0, T_U_max - len(t_u_indices))
            )
            output_list.append(combined)
            y_expanded = torch.nn.functional.pad(
                torch.tensor(y_expanded), (0, T_U_max - len(t_u_indices))
            )
            labels_list.append(y_expanded)
        joiner_output = torch.stack(output_list)
        joiner_output = self.modules.transducer_lin(
            self.modules.Tjoint.nonlinearity(joiner_output)
        ).log_softmax(2)
        joiner_labels = torch.stack(labels_list).long().to(joiner_output.device)
        joiner_labels_lengths = (T + U).float() / T_U_max
        return joiner_output, joiner_labels, joiner_labels_lengths

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
        encoder_out = self.modules.enc(feats.detach())
        e_in = self.modules.emb(y_in)
        predictor_out, _ = self.modules.dec(e_in)

        # Output layer for ctc log-probabilities
        out_ctc = self.modules.ctc_head(encoder_out)
        p_ctc = self.hparams.log_softmax(out_ctc)

        # Compute outputs
        if stage == sb.Stage.TRAIN:
            # Output layer for LM log-probabilities
            p_ce = self.modules.lm_head(predictor_out)
            p_ce = self.hparams.log_softmax(p_ce)
            return p_ctc, p_ce, encoder_out, predictor_out, wav_lens

        elif stage == sb.Stage.VALID:
            predicted_tokens, scores, _, _ = self.hparams.Greedysearcher(
                encoder_out
            )
            return p_ctc, encoder_out, predictor_out, wav_lens, predicted_tokens
        else:
            (
                best_hyps,
                best_scores,
                nbest_hyps,
                nbest_scores,
            ) = self.hparams.Beamsearcher(encoder_out)
            return p_ctc, encoder_out, predictor_out, wav_lens, best_hyps

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

        if stage == sb.Stage.TRAIN:
            p_ctc, p_ce, encoder_out, predictor_out, wav_lens = predictions
        else:
            (
                p_ctc,
                encoder_out,
                predictor_out,
                wav_lens,
                predicted_tokens,
            ) = predictions

        target_tokens = target_tokens.long()
        abs_length = torch.round(target_token_lens * target_tokens.shape[1])
        y = target_tokens
        T = torch.round(wav_lens * encoder_out.shape[1]).long()
        U = abs_length.long()
        with torch.no_grad():
            alignments = self.ctc_align(p_ctc, y, T, U)
        (
            joiner_out,
            joiner_labels,
            joiner_labels_lengths,
        ) = self.gather_outputs_and_labels(
            encoder_out, predictor_out, y, alignments, T, U
        )
        loss_transducer = self.hparams.transducer_ce_cost(
            joiner_out, joiner_labels, length=joiner_labels_lengths
        )

        if stage == sb.Stage.TRAIN:
            CTC_loss = self.hparams.ctc_cost(
                p_ctc, target_tokens, wav_lens, target_token_lens
            )
            # generate output sequence for decoder + CE loss
            # abs_length = torch.round(
            #    target_token_lens * target_tokens.shape[1]
            # )
            target_tokens_with_eos = sb.data_io.data_io.append_eos_token(
                target_tokens,
                length=abs_length,
                eos_index=self.hparams.blank_index,
            )
            rel_length = (abs_length + 1) / target_tokens_with_eos.shape[1]
            CE_loss = self.hparams.lm_ce_cost(
                p_ce, target_tokens_with_eos, length=rel_length
            )
            # target_tokens = target_tokens.long()

            loss = (
                self.hparams.ctc_weight * CTC_loss
                + self.hparams.lm_ce_weight * CE_loss
                + (1 - (self.hparams.ctc_weight + self.hparams.lm_ce_weight))
                * loss_transducer
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

            loss = loss_transducer

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
            if self.root_process:
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
        state_dict = torch.load(save_model_path, map_location=self.device)
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
