#!/usr/bin/env python3

"""Recipe for fine-tuning a Whisper-based ASR system on Common Voice in a continual
learning fashion via Averaged Gradient Episodic Memory (https://arxiv.org/abs/1812.00420).

The following optimization tricks were used to improve performance:
- use custom decoding implementation (faster than built-in searchers
  and supports decoding with predicted batch of languages)
- apply the correct padding tokens directly in the dataloader
- use cross-entropy loss (with `ignore_index` correctly set) instead of log softmax + NLL
- remove unnecessary `undo_padding` since padding tokens are now set correctly
- improve memory usage during model recovery (see https://github.com/speechbrain/speechbrain/pull/1743)
- optionally use gradient checkpointing

To run this recipe, do the following:
> python train_agem.py hparams/train_agem.yaml

Authors
 * Luca Della Libera 2023
"""

import copy
import logging
import os
import pathlib
import random
import sys
import time

import ptflops
import torch
import torchaudio
import torchinfo
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.dataio.batch import PaddedBatch
from speechbrain.utils.distributed import run_on_main

from common_voice_prepare import prepare_common_voice


class ASR(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        bos_tokens, _ = batch.tokens_bos

        # Forward encoder + decoder
        if self.hparams.gradient_checkpointing:
            bos_tokens.requires_grad_()
            enc_out, logits, _ = torch.utils.checkpoint.checkpoint(
                self.modules.whisper, wavs, bos_tokens,
            )
        else:
            enc_out, logits, _ = self.modules.whisper(wavs, bos_tokens)

        hyps = None
        if stage != sb.Stage.TRAIN:
            hyps, _ = self.modules.whisper.generate(
                audio_features=enc_out,
                forced_decoder_locale=self.hparams.forced_decoder_locale,
                max_gen_tokens=self.hparams.max_gen_tokens,
            )

        return logits, hyps

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss given predictions and targets."""
        logits, hyps = predictions
        ids = batch.id
        tokens_eos, _ = batch.tokens_eos

        loss = self.hparams.ce_loss(
            logits.flatten(end_dim=-2), tokens_eos.flatten()
        )

        if stage != sb.Stage.TRAIN:
            target_words = batch.target_wrd

            # Decode predicted tokens to words
            predicted_words = self.tokenizer.batch_decode(
                hyps, skip_special_tokens=True
            )

            if self.hparams.normalize_transcripts:
                predicted_words = [
                    self.tokenizer._normalize(text).split(" ")
                    for text in predicted_words
                ]
            else:
                predicted_words = [text.split(" ") for text in predicted_words]

            self.wer_metric.append(ids, predicted_words, target_words)
            self.cer_metric.append(ids, predicted_words, target_words)

        return loss

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.wer_computer()
        else:
            self.replay_data_iter = iter(self.replay_data)

    def on_stage_end(self, stage, stage_loss, epoch=None):
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
            old_lr, new_lr = self.hparams.lr_annealing(stage_stats["loss"])
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            stats_meta_data = {
                "epoch": epoch,
                "lr": old_lr,
            }
            self.hparams.train_logger.log_stats(
                stats_meta=stats_meta_data,
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
            with open(self.hparams.wer_file, "w", encoding="utf-8") as w:
                self.wer_metric.write_stats(w)

    def fit_batch(self, batch):
        # Managing automatic mixed precision
        if self.auto_mix_prec:
            # Compute gradient
            with torch.cuda.amp.autocast(self.auto_mix_prec):
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            with self.no_sync(False):
                self.scaler.scale(loss).backward()
            with torch.no_grad():
                grad = torch.cat(
                    [
                        p.grad.flatten()
                        for p in self.modules.whisper.parameters()
                        if p.grad is not None
                    ]
                )
                self.modules.whisper.zero_grad()

            # Draw data from replay buffer
            try:
                batch = next(self.replay_data_iter)
            except StopIteration:
                self.replay_data_iter = iter(self.replay_data)
                batch = next(self.replay_data_iter)

            # Compute reference gradient
            with torch.cuda.amp.autocast(self.auto_mix_prec):
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            with self.no_sync(False):
                self.scaler.scale(loss).backward()
            with torch.no_grad():
                grad_ref = torch.cat(
                    [
                        p.grad.flatten()
                        for p in self.modules.whisper.parameters()
                        if p.grad is not None
                    ]
                )

            # Compute and inject modified gradient
            with torch.no_grad():
                grad_T_grad_ref = grad.dot(grad_ref)
                if grad_T_grad_ref < 0:
                    grad -= (grad_T_grad_ref / grad_ref.norm() ** 2) * grad_ref
                start = 0
                for param in self.modules.whisper.parameters():
                    if param.grad is not None:
                        end = start + param.numel()
                        param.grad = grad[start:end].reshape_as(param)
                        start = end

            self.scaler.unscale_(self.optimizer)
            if self.check_gradients(loss):
                self.scaler.step(self.optimizer)
            self.scaler.update()
            self.zero_grad()
            self.optimizer_step += 1
        else:
            # Compute gradient
            outputs = self.compute_forward(batch, sb.Stage.TRAIN)
            loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            with self.no_sync(False):
                loss.backward()
            with torch.no_grad():
                grad = torch.cat(
                    [
                        p.grad.flatten()
                        for p in self.modules.whisper.parameters()
                        if p.grad is not None
                    ]
                )
                self.modules.whisper.zero_grad()

            # Draw data from replay buffer
            try:
                batch = next(self.replay_data_iter)
            except StopIteration:
                self.replay_data_iter = iter(self.replay_data)
                batch = next(self.replay_data_iter)

            # Compute reference gradient
            outputs = self.compute_forward(batch, sb.Stage.TRAIN)
            loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            with self.no_sync(False):
                loss.backward()
            with torch.no_grad():
                grad_ref = torch.cat(
                    [
                        p.grad.flatten()
                        for p in self.modules.whisper.parameters()
                        if p.grad is not None
                    ]
                )

            # Compute and inject modified gradient
            with torch.no_grad():
                grad_T_grad_ref = grad.dot(grad_ref)
                if grad_T_grad_ref < 0:
                    grad -= (grad_T_grad_ref / grad_ref.norm() ** 2) * grad_ref
                start = 0
                for param in self.modules.whisper.parameters():
                    if param.grad is not None:
                        end = start + param.numel()
                        param.grad = grad[start:end].reshape_as(param)
                        start = end

            if self.check_gradients(loss):
                self.optimizer.step()
            self.zero_grad()
            self.optimizer_step += 1

        self.on_fit_batch_end(batch, outputs, loss, True)
        return loss.detach().cpu()


def dataio_prepare(hparams, tokenizer):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=os.path.join(hparams["data_dir"], "train.csv"),
        replacements={"data_root": hparams["data_dir"]},
    )

    if hparams["sorting"] in ["descending", "ascending"]:
        # We sort training data to speed up training and get better results
        train_data = train_data.filtered_sorted(
            sort_key="duration",
            reverse=hparams["sorting"] == "descending",
            key_max_value={"duration": hparams["avoid_if_longer_than"]},
        )
        # When sorting do not shuffle in dataloader otherwise it is pointless
        hparams["train_dataloader_kwargs"]["shuffle"] = False

    elif hparams["sorting"] != "random":
        raise ValueError(
            f"`sorting` ({hparams['sorting']}) must be random, ascending or descending"
        )

    # reverse=True to fail fast in case of out-of-memory error
    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=os.path.join(hparams["data_dir"], "dev.csv"),
        replacements={"data_root": hparams["data_dir"]},
    ).filtered_sorted(
        sort_key="duration",
        reverse=True,
        key_max_value={"duration": hparams["avoid_if_longer_than"]},
    )

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=os.path.join(hparams["data_dir"], "test.csv"),
        replacements={"data_root": hparams["data_dir"]},
    ).filtered_sorted(
        sort_key="duration",
        reverse=True,
        key_max_value={"duration": hparams["avoid_if_longer_than"]},
    )

    datasets = [train_data, valid_data, test_data]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("mp3")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(mp3):
        info = torchaudio.info(mp3)
        sig = sb.dataio.dataio.read_audio(mp3)
        resampled = torchaudio.transforms.Resample(
            info.sample_rate, hparams["sample_rate"],
        )(sig)
        return resampled

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("wrd", "locale")
    @sb.utils.data_pipeline.provides("tokens_bos", "tokens_eos", "target_wrd")
    def text_pipeline(wrd, locale):
        language = tokenizer.supported_languages.get(
            locale, "english"
        )  # Use English if unknown
        tokenizer.set_prefix_tokens(language=language)
        tokens_list = tokenizer.encode(wrd)
        assert sum(i == tokenizer.unk_token_id for i in tokens_list) == 1
        # Remove BOS and EOS tokens from tokens_list
        bos_index, tokens_list, eos_index = (
            tokens_list[0],
            tokens_list[1:-1],
            tokens_list[-1],
        )
        tokens_list = tokens_list[: hparams["max_target_length"] - 1]
        tokens_bos = torch.LongTensor([bos_index] + tokens_list)
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [eos_index])
        yield tokens_eos
        if hparams["normalize_transcripts"]:
            wrd = tokenizer._normalize(wrd)
        wrd = wrd.split(" ")
        # When `ref_tokens` is an empty string add dummy space
        # to avoid division by 0 when computing WER/CER
        for i, char in enumerate(wrd):
            if len(char) == 0:
                wrd[i] = " "
        yield wrd

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "sig", "tokens_bos", "tokens_eos", "target_wrd"],
    )

    return train_data, valid_data, test_data


def test(hparams, run_opts, locales, wer_file="wer_test.txt"):
    # Test on old + new locales
    for locale in locales:
        # Multi-gpu (ddp) save data preparation
        run_on_main(
            prepare_common_voice,
            kwargs={
                "locales": [locale],
                "data_dir": hparams["data_dir"],
                "max_durations": hparams["max_durations"],
            },
        )

        if locale in ["zh-CN", "ja"]:
            # Use CER instead of WER (spaces are not used)
            hparams[
                "wer_computer"
            ] = lambda *args, **kwargs: sb.utils.metric_stats.ErrorRateStats(
                split_tokens=True
            )
        else:
            hparams["wer_computer"] = sb.utils.metric_stats.ErrorRateStats

        # Set forced decoder locale
        hparams["forced_decoder_locale"] = locale

        # Define tokenizer
        tokenizer = hparams["whisper"].tokenizer

        # Create datasets, tokenization and encoding
        _, _, test_data = dataio_prepare(hparams, tokenizer)

        # Trainer initialization
        asr_brain = ASR(
            modules=hparams["modules"], hparams=hparams, run_opts=run_opts,
        )

        # We dynamically add the tokenizer to our brain class
        # NB: This tokenizer corresponds to the one used for Whisper
        asr_brain.tokenizer = tokenizer

        # Testing
        locale_dir = os.path.join(hparams["output_dir"], locale)
        os.makedirs(locale_dir, exist_ok=True)
        asr_brain.hparams.wer_file = os.path.join(locale_dir, wer_file)
        asr_brain.evaluate(
            test_data,
            min_key="WER",
            test_loader_kwargs=hparams["valid_dataloader_kwargs"],
        )

    # MACs not 100% accurate but still useful for comparisons
    profile(hparams)


def train(hparams, run_opts):
    test(
        hparams, run_opts, hparams["old_locales"], f"wer_test_before.txt",
    )

    # Train on new locales
    for i, locale in enumerate(hparams["new_locales"]):
        # Multi-gpu (ddp) save data preparation
        run_on_main(
            prepare_common_voice,
            kwargs={
                "locales": [locale],
                "data_dir": hparams["data_dir"],
                "max_durations": hparams["max_durations"],
            },
        )

        # Add new language token
        new_tokens = [f"<|{locale.lower()}|>"]
        tokenizer = hparams["whisper"].tokenizer
        tokenizer._additional_special_tokens += new_tokens
        tokenizer.supported_languages.update({locale.lower(): locale.lower()})
        tokenizer.to_language_codes.update({locale.lower(): locale.lower()})

        # Check if already in Whisper tokenizer's vocabulary
        new_tokens = set(new_tokens) - set(tokenizer.get_vocab().keys())

        # Add to Whisper tokenizer's vocabulary
        tokenizer.add_tokens(list(new_tokens))

        # Log total number of tokens
        logging.info(
            f"Total number of tokens: {hparams['whisper'].model.decoder.embed_tokens.num_embeddings}"
        )

        # Add a new random embedding for the new language token
        hparams["whisper"].model.resize_token_embeddings(len(tokenizer))

        # Log total number of tokens
        logging.info(
            f"Total number of tokens: {hparams['whisper'].model.decoder.embed_tokens.num_embeddings}"
        )

        # Set forced decoder locale
        hparams["forced_decoder_locale"] = locale

        # Create datasets, tokenization and encoding
        train_data, valid_data, test_data = dataio_prepare(hparams, tokenizer)
        length = len(train_data)

        # Get train data from previous tasks
        replay_data = copy.deepcopy(train_data)
        replay_data.data = {}
        for old_locale in hparams["old_locales"] + hparams["new_locales"][:i]:
            run_on_main(
                prepare_common_voice,
                kwargs={
                    "locales": [old_locale],
                    "data_dir": hparams["data_dir"],
                    "max_durations": hparams["max_durations"],
                },
            )
            old_train_data, _, _ = dataio_prepare(hparams, tokenizer)
            selected_keys = random.sample(
                list(old_train_data.data.keys()),
                round(
                    min(length * hparams["replay_ratio"], len(old_train_data))
                ),
            )
            old_train_data.data = {
                k: old_train_data.data[k] for k in selected_keys
            }
            replay_data.data.update(old_train_data.data)

        # Shuffle replay data
        all_keys = list(replay_data.data.keys())
        random.shuffle(all_keys)
        replay_data.data = {k: replay_data.data[k] for k in all_keys}
        replay_data.data_ids = list(replay_data.data.keys())

        # Trainer initialization
        checkpoint_dir = os.path.join(hparams["save_dir"], locale)
        os.makedirs(checkpoint_dir, exist_ok=True)
        hparams["checkpointer"].checkpoints_dir = pathlib.Path(checkpoint_dir)
        hparams["lr_annealing"].hyperparam_value = hparams["lr"]
        hparams["lr_annealing"].metric_values.clear()
        hparams["lr_annealing"].current_patient = 0
        asr_brain = ASR(
            modules=hparams["modules"],
            hparams=hparams,
            run_opts=run_opts,
            opt_class=hparams["opt_class"],
            checkpointer=hparams["checkpointer"],
        )

        # We dynamically add the tokenizer to our brain class
        # NB: This tokenizer corresponds to the one used for Whisper
        asr_brain.tokenizer = tokenizer

        # Training
        hparams["valid_dataloader_kwargs"].pop("ckpt_prefix", None)
        hparams["epoch_counter"].current = 0
        replay_brain = ASR(
            modules=hparams["modules"], hparams=hparams, run_opts=run_opts,
        )
        replay_data = replay_brain.make_dataloader(
            replay_data,
            stage=sb.Stage.TRAIN,
            **hparams["train_dataloader_kwargs"],
        )
        del replay_brain
        asr_brain.replay_data = replay_data
        asr_brain.fit(
            hparams["epoch_counter"],
            train_data,
            valid_data,
            train_loader_kwargs=hparams["train_dataloader_kwargs"],
            valid_loader_kwargs=hparams["valid_dataloader_kwargs"],
        )

        # Testing
        test(
            hparams,
            run_opts,
            hparams["old_locales"] + hparams["new_locales"][: i + 1],
            f"wer_test_after_{locale}.txt",
        )


def profile(hparams):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.whisper = hparams["whisper"]
            self.wavs = torch.randn(
                1, hparams["sample_rate"], device=run_opts["device"],
            )
            self.bos_tokens = torch.ones(
                1,
                self.whisper.model.config.max_length,
                dtype=torch.int,
                device=run_opts["device"],
            )

        @torch.no_grad()
        def forward(self, _=None):
            enc_out, logits, _ = self.whisper(self.wavs, self.bos_tokens)
            return logits

    model = Model().eval().to(run_opts["device"])
    macs, params = ptflops.get_model_complexity_info(
        model, (1,), as_strings=True, print_per_layer_stat=False,
    )
    time_start = time.time()
    model()
    torch.cuda.synchronize()
    time_stop = time.time() - time_start
    max_mem = torch.cuda.max_memory_allocated("cuda") / 10 ** 9
    result = {
        "MACs": macs,
        "memory": max_mem,
        "time": time_stop,
    }
    logging.info(torchinfo.summary(model, verbose=0))
    logging.info(result)


if __name__ == "__main__":
    # Command-line interface
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)
    random.seed(hparams["seed"])
    grad_accumulation_factor = hparams.get("grad_accumulation_factor")
    if grad_accumulation_factor is not None and grad_accumulation_factor > 1:
        logging.info(
            "`grad_accumulation_factor` > 1 not supported, setting to 1"
        )
        hparams["grad_accumulation_factor"] = 1

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_dir"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    class CustomPaddedBatch(PaddedBatch):
        def __init__(self, examples, *args, **kwargs):
            for k in ["tokens_bos", "tokens_eos"]:
                max_len = max([len(x[k]) for x in examples])
                pad_value = 0.0
                if k in ["tokens_bos"]:
                    pad_value = hparams["whisper"].tokenizer.pad_token_id
                elif k == "tokens_eos":
                    pad_value = hparams["ignore_index"]
                for example in examples:
                    x = example[k]
                    example[k] = torch.nn.functional.pad(
                        x, [0, max_len - len(x)], value=pad_value
                    )
            super().__init__(examples, *args, **kwargs)

    hparams["train_dataloader_kwargs"]["collate_fn"] = CustomPaddedBatch
    hparams["valid_dataloader_kwargs"]["collate_fn"] = CustomPaddedBatch

    # Train
    start_time = time.time()
    train(hparams, run_opts)
    duration = time.time() - start_time
    logging.info(f"Time elapsed: {duration} seconds")
