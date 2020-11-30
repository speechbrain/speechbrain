#!/usr/bin/env python3
"""Recipe for multi-task learning, using CTC and enhancement objectives.

To run this recipe, do the following:
> python experiment.py {hyperparameter file} --data_folder /path/to/noisy-vctk

Use your own hyperparameter file or the provided `hyperparams.yaml`
The different losses can be turned on and off, and pre-trained models
can be used for enhancement or ASR models.

Authors
 * Peter Plantinga 2020
"""
import os
import sys
import torch
import speechbrain as sb
from pesq import pesq
from pystoi import stoi
from speechbrain.utils.data_utils import download_file, undo_padding
from speechbrain.tokenizers.SentencePiece import SentencePiece


def pesq_eval(pred_wav, target_wav):
    return pesq(
        fs=16000, ref=target_wav.numpy(), deg=pred_wav.numpy(), mode="wb",
    )


def estoi_eval(pred_wav, target_wav):
    return stoi(
        x=target_wav.numpy(), y=pred_wav.numpy(), fs_sig=16000, extended=True
    )


# Define training procedure
class ASR_Brain(sb.Brain):
    def compute_forward(self, noisy, clean, targets, stage):
        """The forward pass computes enhanced feats and targets"""

        predictions = {}
        if self.hparams.enhance_type is not None:
            noisy_wavs, noisy_feats, noisy_lens, targets = self.prepare_feats(
                noisy, targets, stage
            )

            # Mask with "signal approximation (SA)"
            if self.hparams.enhance_type == "masking":
                mask = self.modules.enhance_model(noisy_feats)
                predictions["feats"] = torch.mul(mask, noisy_feats)

            # Spectral mapping
            elif self.hparams.enhance_type == "mapping":
                predictions["feats"] = self.modules.enhance_model(noisy_feats)

            # Noisy
            elif self.hparams.enhance_type == "noisy":
                predictions["feats"] = noisy_feats

            elif self.hparams.enhance_type == "clean":
                _, clean_feats, clean_lens, _ = self.prepare_feats(
                    clean, targets, stage,
                )
                predictions["feats"] = clean_feats

            # Resynthesize waveforms
            enhanced_mag = torch.expm1(predictions["feats"])
            predictions["wavs"] = self.hparams.resynth(enhanced_mag, noisy_wavs)

        # Generate clean features for ASR pre-training
        if self.hparams.ctc_type == "clean" or self.hparams.seq_type == "clean":
            clean_feats, clean_lens, targets = self.prepare_feats(
                clean, targets, stage,
            )

        # Compute seq outputs
        if self.hparams.seq_type is not None:

            # Prepare target inputs
            tokens, token_lens = self.prepare_target(
                targets, tokenize=True, bos=True
            )
            tokens = self.modules.tgt_embedding(tokens)

            # Compute forward
            if self.hparams.seq_type == "clean":
                embed = self.modules.src_embedding(clean_feats)
                feat_lens = clean_lens
            if self.hparams.seq_type == "joint":
                asr_feats = predictions["wavs"]
                # if stage == sb.Stage.TRAIN:
                #    asr_feats = self.hparams.augment(asr_feats, noisy_lens)
                asr_feats = self.hparams.fbank(asr_feats)
                asr_feats = self.hparams.normalizer(asr_feats, noisy_lens)
                embed = self.modules.src_embedding(asr_feats)
                feat_lens = noisy_lens
            dec_out = self.modules.recognizer(tokens, embed, feat_lens)
            out = self.modules.seq_output(dec_out[0])
            predictions["seq_pout"] = self.hparams.log_softmax(out)

            if stage != sb.Stage.TRAIN:
                predictions["hyps"], _ = self.hparams.beam_searcher(
                    embed.detach(), feat_lens
                )

        # Compute ctc outputs
        if self.hparams.ctc_type is not None:
            if self.hparams.seq_type is not None:
                out = embed
            elif self.hparams.ctc_type == "clean":
                out = self.modules.recognizer(clean_feats)
            elif self.hparams.ctc_type == "joint":
                asr_feats = predictions["wavs"]
                if stage == sb.Stage.TRAIN:
                    asr_feats = self.hparams.augment(asr_feats, noisy_lens)
                asr_feats = self.hparams.fbank(asr_feats)
                asr_feats = self.hparams.normalizer(asr_feats, noisy_lens)
                out = self.modules.recognizer(asr_feats)
            out = self.modules.ctc_output(out)
            predictions["out"] = out
            predictions["ctc_pout"] = self.hparams.log_softmax(out)

        return predictions

    def prepare_feats(self, wavs, targets, stage, augment=False):
        ids, wavs, wav_lens = wavs
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)

        if augment and stage == sb.Stage.TRAIN:
            wavs = self.hparams.augment(wavs, wav_lens)

        if hasattr(self.hparams, "env_corr") and stage == sb.Stage.TRAIN:
            ids, targets, target_lens = targets
            wavs_noise = self.hparams.env_corr(wavs, wav_lens)
            wavs = torch.cat([wavs, wavs_noise], dim=0)
            wav_lens = torch.cat([wav_lens, wav_lens])
            targets = torch.cat([targets, targets], dim=0)
            target_lens = torch.cat([target_lens, target_lens])
            targets = (ids, targets, target_lens)

        # Generate log-magnitude features expected by enhancement model
        feats = self.hparams.compute_stft(wavs)
        feats = self.hparams.spectral_magnitude(feats)
        feats = torch.log1p(feats)

        return wavs, feats, wav_lens, targets

    def prepare_target(self, target, tokenize=False, bos=False, eos=False):
        ids, tokens, token_lens = target

        if tokenize and hasattr(self.hparams, "tokenizer"):
            tokens, _ = self.hparams.tokenizer(
                tokens, token_lens, self.hparams.ind2lab, task="encode"
            )

        tokens = tokens.to(self.device)
        token_lens = token_lens.to(self.device)

        abs_length = torch.round(token_lens * tokens.shape[1])
        if bos:
            tokens = sb.data_io.data_io.prepend_bos_token(
                tokens, bos_index=self.hparams.bos_index
            )
            token_lens = (abs_length + 1) / tokens.shape[1]
        if eos:
            tokens = sb.data_io.data_io.append_eos_token(
                tokens, length=abs_length, eos_index=self.hparams.eos_index
            )
            token_lens = (abs_length + 1) / tokens.shape[1]

        return tokens, token_lens

    def compute_objectives(self, predictions, clean, targets, stage):
        """We have multiple targets here, ``clean`` and ``targets``"""

        clean_wavs, clean_feats, clean_lens, targets = self.prepare_feats(
            clean, targets, stage
        )

        loss = 0

        # Compute enhancement loss
        if self.hparams.enhance_weight > 0:
            enhance_loss = self.hparams.enhance_loss(
                predictions["feats"], clean_feats, clean_lens
            )
            loss += self.hparams.enhance_weight * enhance_loss

            if stage != sb.Stage.TRAIN:
                ids, _, _ = targets
                self.enh_metrics.append(
                    ids, predictions["feats"], clean_feats, clean_lens
                )
                self.stoi_metrics.append(
                    ids,
                    predict=predictions["wavs"],
                    target=clean_wavs,
                    lengths=clean_lens,
                )
                self.pesq_metrics.append(
                    ids,
                    predict=predictions["wavs"],
                    target=clean_wavs,
                    lengths=clean_lens,
                )

        # Compute mimic loss
        if self.hparams.mimic_weight > 0:

            soft_targets = self.modules.src_embedding.CNN(clean_feats)
            enhanced_out = self.modules.src_embedding.CNN(predictions["feats"])

            mimic_loss = self.hparams.mimic_loss(
                enhanced_out, soft_targets, clean_lens
            )
            loss += self.hparams.mimic_weight * mimic_loss

            if stage != sb.Stage.TRAIN:
                self.mimic_metrics.append(
                    ids, enhanced_out, soft_targets, clean_lens
                )

        # Compute hard ASR loss
        if self.hparams.ctc_weight > 0 and (
            not hasattr(self.hparams, "ctc_epochs")
            or self.hparams.epoch_counter.current < self.hparams.ctc_epochs
        ):
            tokens, token_lens = self.prepare_target(targets, tokenize=True)
            ctc_loss = self.hparams.ctc_loss(
                predictions["ctc_pout"], tokens, clean_lens, token_lens
            )
            loss += self.hparams.ctc_weight * ctc_loss

            if stage != sb.Stage.TRAIN:
                self.ctc_metrics.append(
                    ids, predictions["ctc_pout"], tokens, clean_lens, token_lens
                )

                if self.hparams.seq_weight == 0:
                    predict = sb.decoders.ctc_greedy_decode(
                        predictions["ctc_pout"], clean_lens, blank_id=-1
                    )
                    self.err_rate_metrics.append(
                        ids=ids,
                        predict=predict,
                        target=tokens,
                        target_len=token_lens,
                        ind2lab=self.hparams.ind2lab,
                    )

        # Compute nll loss for seq2seq model
        if self.hparams.seq_weight > 0:

            # Append eos token
            tokens, token_lens = self.prepare_target(
                targets, tokenize=True, eos=True
            )

            # Compute loss
            seq_loss = self.hparams.seq_loss(
                predictions["seq_pout"], tokens, token_lens
            )
            loss += self.hparams.seq_weight * seq_loss

            if stage != sb.Stage.TRAIN:
                ids, _, _ = targets
                self.seq_metrics.append(
                    ids, predictions["seq_pout"], tokens, token_lens
                )

                ids, targets, target_lens = targets
                if hasattr(self.hparams, "tokenizer"):

                    pred_words = self.hparams.tokenizer(
                        predictions["hyps"], task="decode_from_list"
                    )

                    target_words = undo_padding(targets, target_lens)
                    target_words = sb.data_io.data_io.convert_index_to_lab(
                        target_words, self.hparams.ind2lab
                    )
                    self.err_rate_metrics.append(ids, pred_words, target_words)

                else:
                    self.err_rate_metrics.append(
                        ids=ids,
                        predict=predictions["hyps"],
                        target=targets,
                        target_len=target_lens,
                        ind2lab=self.hparams.ind2lab,
                    )

        return loss

    def fit_batch(self, batch):
        noisy, clean, targets = batch
        self.optimizer.zero_grad()
        preds = self.compute_forward(noisy, clean, targets, sb.Stage.TRAIN)
        loss = self.compute_objectives(preds, clean, targets, sb.Stage.TRAIN)
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu()

    def evaluate_batch(self, batch, stage):
        noisy, clean, targets = batch
        preds = self.compute_forward(noisy, clean, targets, stage)
        loss = self.compute_objectives(preds, clean, targets, stage)

        return loss.detach().cpu()

    def on_stage_start(self, stage, epoch):
        if stage != sb.Stage.TRAIN:

            if self.hparams.enhance_weight > 0:
                self.enh_metrics = self.hparams.enhance_stats()
                # self.stoi_metrics = self.hparams.stoi_stats()
                self.stoi_metrics = sb.MetricStats(metric=estoi_eval, n_jobs=30)
                self.pesq_metrics = sb.MetricStats(metric=pesq_eval, n_jobs=30)

            if self.hparams.mimic_weight > 0:
                self.mimic_metrics = self.hparams.mimic_stats()

            if self.hparams.ctc_weight > 0:
                self.ctc_metrics = self.hparams.ctc_stats()
                self.err_rate_metrics = self.hparams.err_rate_stats()

            if self.hparams.seq_weight > 0:
                self.seq_metrics = self.hparams.seq_stats()

                if not hasattr(self, "err_rate_metrics"):
                    self.err_rate_metrics = self.hparams.err_rate_stats()

        # Freeze models at beginning of training
        elif epoch <= 1:
            for model in self.hparams.frozen_models:
                for p in self.modules[model].parameters():
                    p.requires_grad = False

    def on_stage_end(self, stage, stage_loss, epoch):
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        else:
            stage_stats = {"loss": stage_loss}
            max_keys = []
            min_keys = []

            if self.hparams.enhance_weight > 0:
                stage_stats["enhance"] = self.enh_metrics.summarize("average")
                stage_stats["stoi"] = self.stoi_metrics.summarize("average")
                stage_stats["pesq"] = self.pesq_metrics.summarize("average")
                max_keys.extend(["pesq", "stoi"])

            if self.hparams.mimic_weight > 0:
                stage_stats["mimic"] = self.mimic_metrics.summarize("average")
                min_keys.append("mimic")

            if self.hparams.ctc_weight > 0 or self.hparams.seq_weight > 0:
                err_rate = self.err_rate_metrics.summarize("error_rate")

                err_rate_type = "PER"
                if self.hparams.target_type == "wrd":
                    err_rate_type = "WER"

                stage_stats[err_rate_type] = err_rate
                min_keys.append(err_rate_type)

        if stage == sb.Stage.VALID:

            if self.hparams.ctc_weight > 0 or self.hparams.seq_weight > 0:
                old_lr, new_lr = self.hparams.lr_annealing(err_rate)
                sb.nnet.update_learning_rate(self.optimizer, new_lr)

            # In distributed setting, only want to save model/stats once
            if self.root_process:
                self.hparams.train_logger.log_stats(
                    stats_meta={"epoch": epoch},
                    train_stats={"loss": self.train_loss},
                    valid_stats=stage_stats,
                )
                self.checkpointer.save_and_keep_only(
                    meta=stage_stats, max_keys=max_keys, min_keys=min_keys,
                )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.stats_file, "w") as w:
                if self.hparams.enhance_weight > 0:
                    w.write("\nstoi stats:\n")
                    self.stoi_metrics.write_stats(w)
                    w.write("\npesq stats:\n")
                    self.pesq_metrics.write_stats(w)
                if self.hparams.mimic_weight > 0:
                    w.write("\nmimic stats:\n")
                    self.mimic_metrics.write_stats(w)
                if self.hparams.ctc_weight > 0:
                    self.err_rate_metrics.write_stats(w)
                print("stats written to ", self.hparams.stats_file)

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


# Begin Recipe!
if __name__ == "__main__":

    # This hack needed to import data preparation script from ..
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(os.path.dirname(current_dir)))
    from voicebank_prepare import prepare_voicebank  # noqa E402

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
    prepare_voicebank(
        data_folder=hparams["data_folder"], save_folder=hparams["data_folder"],
    )

    # Create tokenizer after preparation
    if hparams["target_type"] == "wrd":
        hparams["tokenizer"] = SentencePiece(
            model_dir=hparams["save_folder"],
            vocab_size=hparams["output_neurons"],
            csv_train=hparams["csv_train"],
            csv_read="wrd",
            model_type=hparams["token_type"],
            character_coverage=1.0,
        )

    # Collect index to label dictionary for decoding
    train_set = hparams["train_loader"]()
    valid_set = hparams["valid_loader"]()
    test_set = hparams["test_loader"]()
    tgt = hparams["target_type"]
    hparams["ind2lab"] = hparams["test_loader"].label_dict[tgt]["index2lab"]
    # hparams["ind2lab"][42] = "blank"

    # Load pretrained models
    if "pretrained_path" in hparams:
        for model_name, path in hparams["pretrained_path"].items():
            hparams[model_name].load_state_dict(torch.load(path))

    if "pretrain_checkpointer" in hparams:
        hparams["pretrain_checkpointer"].recover_if_possible()

    asr_brain = ASR_Brain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        checkpointer=hparams["checkpointer"],
    )

    asr_brain.load_tokenizer()
    if hasattr(asr_brain.hparams, "lm_ckpt_file"):
        asr_brain.load_lm()

    asr_brain.fit(asr_brain.hparams.epoch_counter, train_set, valid_set)
    # asr_brain.evaluate(hparams["test_loader"](), max_key="stoi")
    # asr_brain.evaluate(test_set, min_key="WER")
    asr_brain.evaluate(hparams["test_loader"]())
