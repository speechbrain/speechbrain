#!/usr/bin/env/python3
"""Recipe for training a sequence-to-sequence ASR system with librispeech.
The system employs an encoder, a decoder, and an attention mechanism
between them. Decoding is performed with beamsearch coupled with a neural
language model.

To run this recipe, do the following:
> python experiment.py hyperparams.yaml

With the default hyperparameters, the system employs a CRDNN encoder.
The decoder is based on a standard  GRU. Beamsearch coupled with a RNN
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
 * Ju-Chieh Chou 2020
 * Mirco Ravanelli 2020
 * Abdel Heba 2020
 * Peter Plantinga 2020
"""

import os
import sys
import torch
import speechbrain as sb
from speechbrain.utils.data_utils import download_file, undo_padding
from speechbrain.tokenizers.SentencePiece import SentencePiece
from speechbrain.utils.distributed import run_on_main


# Define training procedure
class ASR(sb.Brain):
    def compute_forward(self, batch, stage):
        """Computation of output probabilities based on input signals"""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        tokens_bos, tokens_bos_lens = batch.tokens_bos

        # Add augmentation if specified
        if stage == sb.Stage.TRAIN:
            if hasattr(self.modules, "env_corrupt"):
                wavs_noise = self.modules.env_corrupt(wavs, wav_lens)
                wavs = torch.cat([wavs, wavs_noise], dim=0)
                wav_lens = torch.cat([wav_lens, wav_lens])
                tokens_bos = torch.cat([tokens_bos, tokens_bos], dim=0)
                tokens_bos_lens = torch.cat([tokens_bos_lens, tokens_bos_lens])
            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs, wav_lens)

        # Forward pass
        feats = self.hparams.compute_features(wavs)
        feats = self.modules.normalize(feats, wav_lens)
        encoded_sig = self.modules.enc(feats.detach())
        embedded_tokens = self.modules.emb(tokens_bos)
        h, _ = self.modules.dec(embedded_tokens, encoded_sig, wav_lens)

        # Output layer for seq2seq log-probabilities
        outputs = {"wav_lens": wav_lens}
        logits = self.modules.seq_lin(h)
        outputs["p_seq"] = self.hparams.log_softmax(logits)

        if stage == sb.Stage.TRAIN:
            current_epoch = self.hparams.epoch_counter.current
            if current_epoch <= self.hparams.number_of_ctc_epochs:
                # Output layer for ctc log-probabilities
                logits = self.modules.ctc_lin(encoded_sig)
                outputs["p_ctc"] = self.hparams.log_softmax(logits)
        else:
            outputs["p_tokens"], scores = self.hparams.beam_searcher(
                encoded_sig, wav_lens
            )

        return outputs

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""

        current_epoch = self.hparams.epoch_counter.current
        wav_lens = predictions["wav_lens"]

        tokens, tokens_lens = batch.tokens
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        if hasattr(self.modules, "env_corrupt") and stage == sb.Stage.TRAIN:
            tokens = torch.cat([tokens, tokens], dim=0)
            tokens_lens = torch.cat([tokens_lens, tokens_lens])
            tokens_eos = torch.cat([tokens_eos, tokens_eos], dim=0)
            tokens_eos_lens = torch.cat([tokens_eos_lens, tokens_eos_lens])

        loss = self.hparams.seq_cost(
            predictions["p_seq"], tokens_eos, length=tokens_eos_lens
        )

        # Add ctc loss if necessary
        if (
            stage == sb.Stage.TRAIN
            and current_epoch <= self.hparams.number_of_ctc_epochs
        ):
            loss_ctc = self.hparams.ctc_cost(
                predictions["p_ctc"], tokens, wav_lens, tokens_lens
            )
            loss *= 1 - self.hparams.ctc_weight
            loss += self.hparams.ctc_weight * loss_ctc

        if stage != sb.Stage.TRAIN:
            # Decode token terms to words
            predicted_words = self.tokenizer(
                predictions["p_tokens"], task="decode_from_list"
            )

            # Convert indices to words
            target_words = undo_padding(tokens, tokens_lens)
            target_words = self.tokenizer(target_words, task="decode_from_list")

            self.wer_metric.append(batch.id, predicted_words, target_words)
            self.cer_metric.append(batch.id, predicted_words, target_words)

        return loss

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

    def load_lm(self):
        """Loads the LM specified in the yaml file"""
        save_model_path = os.path.join(
            self.hparams.output_folder, "save", "lm_model.ckpt"
        )
        download_file(self.hparams.lm_ckpt_file, save_model_path)
        state_dict = torch.load(save_model_path)
        self.hparams.lm_model.load_state_dict(state_dict, strict=True)
        self.hparams.lm_model.eval()


def data_io_prep(hparams):
    """Creates the datasets and their data processing pipelines"""

    # 1. define tokenizer and load it
    tokenizer = SentencePiece(
        model_dir=hparams["save_folder"],
        vocab_size=hparams["output_neurons"],
        csv_train=hparams["train_annotation"],
        csv_read="wrd",
        model_type=hparams["token_type"],
        character_coverage=hparams["character_coverage"],
    )

    """Loads the sentence piece tokenizer specified in the yaml file"""
    save_model_path = os.path.join(hparams["save_folder"], "tok_unigram.model")
    save_vocab_path = os.path.join(hparams["save_folder"], "tok_unigram.vocab")

    if "tok_mdl_file" in hparams:
        download_file(
            source=hparams["tok_mdl_file"],
            dest=save_model_path,
            replace_existing=True,
        )
        tokenizer.sp.load(save_model_path)

    if "tok_voc_file" in hparams:
        download_file(
            source=hparams["tok_voc_file"],
            dest=save_vocab_path,
            replace_existing=True,
        )
        tokenizer.sp.load(save_model_path)

    if (tokenizer.sp.eos_id() + 1) == (tokenizer.sp.bos_id() + 1) == 0 and not (
        hparams["eos_index"]
        == hparams["bos_index"]
        == hparams["blank_index"]
        == hparams["unk_index"]
        == 0
    ):
        raise ValueError(
            "Desired indexes for special tokens do not agree "
            "with loaded tokenizer special tokens !"
        )

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes(hparams["input_type"])
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.data_io.data_io.read_audio(wav)
        return sig

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides("tokens_bos", "tokens_eos", "tokens")
    def text_pipeline(wrd):
        tokens_list = tokenizer.sp.encode_as_ids(wrd)
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    # 4. Create datasets
    data = {}
    for dataset in ["train", "valid", "test"]:
        data[dataset] = sb.data_io.dataset.DynamicItemDataset.from_csv(
            csv_path=hparams[f"{dataset}_annotation"],
            replacements={"data_root", hparams["data_folder"]},
            dynamic_items=[audio_pipeline, text_pipeline],
            output_keys=["id", "sig", "tokens_bos", "tokens_eos", "tokens"],
        )
        if dataset != "train":
            data[dataset] = data[dataset].filtered_sorted(sort_key="duration")

    # Sort train dataset and ensure it doesn't get un-sorted
    if hparams["sorting"] == "ascending" or hparams["sorting"] == "descending":
        data["train"] = data["train"].filtered_sorted(
            sort_key="duration", reverse=hparams["sorting"] == "descending",
        )
        hparams["dataloader_options"]["shuffle"] = False
    elif hparams["sorting"] != "random":
        raise NotImplementedError(
            "Sorting must be random, ascending, or descending"
        )

    return data, tokenizer


if __name__ == "__main__":

    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = sb.load_extended_yaml(fin, overrides)

    # Needed to run experiment with distributed system
    sb.ddp_init_group(run_opts)

    # Prepare data
    from voicebank_prepare import prepare_voicebank  # noqa E402

    run_on_main(
        prepare_voicebank,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["data_folder"],
        },
    )

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Create dataset objects and tokenizer
    datasets, tokenizer = data_io_prep(hparams)

    # Load pretrained models if provided
    if "pretrain_checkpointer" in hparams:
        hparams["pretrain_checkpointer"].recover_if_possible()

    # Brain class initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        run_opts=run_opts,
        hparams=hparams,
        checkpointer=hparams["checkpointer"],
    )
    asr_brain.tokenizer = tokenizer
    if hasattr(asr_brain.hparams, "lm_ckpt_file"):
        asr_brain.load_lm()

    # Training
    asr_brain.fit(
        epoch_counter=asr_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )

    # Test checkpoint that performs best on validation data (lowest WER)
    asr_brain.evaluate(
        datasets["test"],
        min_key="WER",
        test_loader_kwargs=hparams["dataloader_options"],
    )
