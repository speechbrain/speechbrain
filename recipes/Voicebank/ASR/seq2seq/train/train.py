#!/usr/bin/env/python3
"""Recipe for training a sequence-to-sequence ASR system with librispeech.
The system employs an encoder, a decoder, and an attention mechanism
between them. Decoding is performed with beamsearch coupled with a neural
language model.

To run this recipe, do the following:
> python train.py hparams/train.yaml

With the default hyperparameters, the system downloads a pre-trained
librispeech model and fine-tunes it on Voicebank.

Authors
 * Peter Plantinga 2020
"""

import os
import sys
import torch
import urllib.parse
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.data_utils import download_file, undo_padding
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
        feats = self.modules.normalizer(feats, wav_lens)
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
            predicted_words = [
                self.hparams.tokenizer.decode_ids(token_seq)
                for token_seq in predictions["p_tokens"]
            ]

            # Convert indices to words
            target_words = [
                self.hparams.tokenizer.decode_ids(token_seq)
                for token_seq in undo_padding(tokens, tokens_lens)
            ]

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


def dataio_prep(hparams):
    """Creates the datasets and their data processing pipelines"""

    # Define pipelines
    @sb.utils.data_pipeline.takes(hparams["input_type"])
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        return sb.dataio.dataio.read_audio(wav)

    @sb.utils.data_pipeline.takes("words")
    @sb.utils.data_pipeline.provides("tokens_bos", "tokens_eos", "tokens")
    def text_pipeline(words):
        tokens_list = hparams["tokenizer"].encode_as_ids(words)
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    # 4. Create datasets
    data = {}
    for dataset in ["train", "valid", "test"]:
        data[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=hparams[f"{dataset}_annotation"],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline, text_pipeline],
            output_keys=["id", "sig", "tokens_bos", "tokens_eos", "tokens"],
        )
        if dataset != "train":
            data[dataset] = data[dataset].filtered_sorted(sort_key="length")

    # Sort train dataset and ensure it doesn't get un-sorted
    if hparams["sorting"] == "ascending" or hparams["sorting"] == "descending":
        data["train"] = data["train"].filtered_sorted(
            sort_key="length", reverse=hparams["sorting"] == "descending",
        )
        hparams["dataloader_options"]["shuffle"] = False
    elif hparams["sorting"] != "random":
        raise NotImplementedError(
            "Sorting must be random, ascending, or descending"
        )

    return data


def download_to_dir(url, directory):
    """Parse filename from url and download to directory."""
    print("called download_to_dir")
    os.makedirs(directory, exist_ok=True)
    filename = os.path.basename(urllib.parse.urlparse(url).path)
    download_file(url, os.path.join(directory, filename))
    return os.path.join(directory, filename)


if __name__ == "__main__":

    # Download the asr model yaml file so we can "!include" it
    download_to_dir(
        url="https://www.dropbox.com/s/wbu3i82urhxe3in/asr_model.yaml?dl=1",
        directory=os.path.join("hparams", "models"),
    )

    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Prepare data
    from voicebank_prepare import prepare_voicebank  # noqa E402

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    run_on_main(
        prepare_voicebank,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["data_folder"],
            "skip_prep": hparams["skip_prep"],
        },
    )

    # Load pre-trained models
    run_on_main(hparams["asr_pretrained"].collect_files)
    hparams["asr_pretrained"].load_collected()

    # Create dataset objects and tokenizer
    datasets = dataio_prep(hparams)

    # Brain class initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        run_opts=run_opts,
        hparams=hparams,
        checkpointer=hparams["checkpointer"],
    )

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
