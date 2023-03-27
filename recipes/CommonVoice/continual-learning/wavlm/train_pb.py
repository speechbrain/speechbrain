#!/usr/bin/env python3

"""Recipe for fine-tuning an OpenAI Whisper-based ASR system on Common Voice in a continual
learning fashion via Progressive Neural Networks (https://arxiv.org/abs/1612.00796).

The following technical tricks were implemented to improve performance:
- use custom greedy decoding implementation (several times faster than built-in
  greedy searchers and supports decoding with predicted batch of languages)
- apply the correct padding tokens directly in the dataloader
- use cross-entropy loss (with `ignore_index` correctly set) instead of log softmax + NLL
- remove unnecessary `undo_padding` since padding tokens are now set correctly
- improve memory usage during model recovery (see https://github.com/speechbrain/speechbrain/pull/1743)
- optionally use gradient checkpointing
- minor optimizations (e.g. remove leading special tokens from `tokens` during data loading)

To run this recipe, do the following:
> python train_pnn.py hparams/train_pnn.yaml

NOTE: although checkpoints are saved regularly, automatic experiment resumption is not supported.
      To manually resume an experiment, you have to modify the script to load the correct checkpoint
      and set the corresponding state variables (e.g. current locale).

Authors
 * Luca Della Libera 2022
"""

import copy
import logging
import os
import pathlib
import sys
import time
from speechbrain.lobes.models.VanillaNN import VanillaNN
from speechbrain.nnet.linear import Linear
import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml
from speechbrain.nnet.RNN import LSTM
import speechbrain as sb
from speechbrain.dataio.batch import PaddedBatch
from speechbrain.tokenizers.SentencePiece import SentencePiece
from speechbrain.utils.distributed import run_on_main

from common_voice_prepare import prepare_common_voice

class ASR(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        bos_tokens, _ = batch.tokens_bos
        # Forward encoder + decoder
        
        
        feats = self.modules.wavlm(wavs)

        if self.hparams.forced_decoder_locale in self.hparams.new_locales and  stage != sb.Stage.TEST:
            self.modules.decoder.load_state_dict(
                self.hparams.decoder_state_backup, strict=False
            )
            decoder_mask = self.hparams.decoder_mask.get(
                self.hparams.forced_decoder_locale
            )
            if decoder_mask is not None:
                for (
                    k,
                    v,
                ) in self.modules.decoder.named_parameters():
                    if k not in decoder_mask:
                        continue
                    v.detach_()
                    thresholded_mask = Threshold.apply(
                        decoder_mask[k].to(v.device),
                        self.hparams.mask_threshold,
                    )
                    v *= thresholded_mask

        x = self.modules.decoder(feats)
        x=x[0]
        # Compute outputs
        p_tokens = None
        logits = hparams["ctc_decoders"][self.hparams.forced_decoder_locale](x)
        p_ctc = self.hparams.log_softmax(logits)
        if stage != sb.Stage.TRAIN:
            p_tokens = sb.decoders.ctc_greedy_decode(
                p_ctc, wav_lens, blank_id=self.hparams.blank_index
            )
        return p_ctc, wav_lens, p_tokens

    def compute_objectives(self, predictions, batch, stage):
        p_ctc, wav_lens, predicted_tokens = predictions

        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        tokens, tokens_lens = batch.tokens

        if hasattr(self.modules, "env_corrupt") and stage == sb.Stage.TRAIN:
            tokens_eos = torch.cat([tokens_eos, tokens_eos], dim=0)
            tokens_eos_lens = torch.cat(
                [tokens_eos_lens, tokens_eos_lens], dim=0
            )
            tokens = torch.cat([tokens, tokens], dim=0)
            tokens_lens = torch.cat([tokens_lens, tokens_lens], dim=0)
        loss_ctc = self.hparams.ctc_cost(p_ctc, tokens, wav_lens, tokens_lens)
        loss = loss_ctc
        if stage !=sb.Stage.TRAIN:
            predicted_words = [
                "".join(self.tokenizer.decode_ndim(utt_seq)).split(" ")
                for utt_seq in predicted_tokens
            ]
            target_words = [wrd.split(" ") for wrd in batch.wrd]
            self.wer_metric.append(ids, predicted_words, target_words)
            self.cer_metric.append(ids, predicted_words, target_words)

        return loss

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""
        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)
        loss.backward()
        if self.check_gradients(loss):
            self.model_optimizer.step()
        self.model_optimizer.zero_grad()

        return loss.detach()

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.wer_computer()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")
        if stage == sb.Stage.VALID:
            old_lr_model, new_lr_model = self.hparams.lr_annealing_model(
                stage_stats["loss"]
            )
            sb.nnet.schedulers.update_learning_rate(
                self.model_optimizer, new_lr_model
            )
            self.hparams.train_logger.log_stats(
                stats_meta={
                    "epoch": epoch,
                    "lr_model": old_lr_model,
                },
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



    def init_optimizers(self):
        parameters = [
            p for p in self.hparams.model.parameters() if p.requires_grad
        ]
        decoder_mask = self.hparams.decoder_mask.get(
            self.hparams.forced_decoder_locale
        )
        if decoder_mask is not None:
            parameters += list(decoder_mask.values())

        self.model_optimizer = self.hparams.model_opt_class(parameters)

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("modelopt", self.model_optimizer)


class Threshold(torch.autograd.Function):
    """Pseudo-differentiable thresholding function."""

    @staticmethod
    def forward(ctx, input, threshold=0.005):
        return torch.where(input >= threshold, 1.0, 0.0)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def dataio_prepare(hparams, label_encoder):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=os.path.join(hparams["download_dir"], "train.csv"),
        replacements={"data_root": hparams["download_dir"]},
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
        csv_path=os.path.join(hparams["download_dir"], "dev.csv"),
        replacements={"data_root": hparams["download_dir"]},
    ).filtered_sorted(
        sort_key="duration",
        reverse=True,
        key_max_value={"duration": hparams["avoid_if_longer_than"]},
    )

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=os.path.join(hparams["download_dir"], "test.csv"),
        replacements={"data_root": hparams["download_dir"]},
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
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "wrd", "char_list", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(wrd):
        yield wrd
        char_list = list(wrd)
        yield char_list
        tokens_list = label_encoder.encode_sequence(char_list)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "sig", "wrd", "char_list", "tokens_bos", "tokens_eos", "tokens"],
    )


    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    special_labels = {
        "bos_label": hparams["bos_index"],
        "eos_label": hparams["eos_index"],
        "blank_label": hparams["blank_index"],
    }
    return train_data,valid_data, test_data



def create_label_encoder(hparams, locale):
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=os.path.join(hparams["download_dir"], "train.csv"),
        replacements={"data_root": hparams["download_dir"]},
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

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=os.path.join(hparams["download_dir"], "dev.csv"),
        replacements={"data_root": hparams["download_dir"]},
    ).filtered_sorted(
        sort_key="duration",
        reverse=True,
        key_max_value={"duration": hparams["avoid_if_longer_than"]},
    )

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=os.path.join(hparams["download_dir"], "test.csv"),
        replacements={"data_root": hparams["download_dir"]},
    ).filtered_sorted(
        sort_key="duration",
        reverse=True,
        key_max_value={"duration": hparams["avoid_if_longer_than"]},
    )



    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "wrd", "char_list", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(wrd):
        yield wrd
        char_list = list(wrd)
        yield char_list
        tokens_list = label_encoder.encode_sequence(char_list)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens


    label_encoder = sb.dataio.encoder.CTCTextEncoder()
    sb.dataio.dataset.add_dynamic_item([train_data, valid_data, test_data], text_pipeline)
    lab_enc_file = os.path.join(hparams["save_folder"], f"label_encoder_{locale}.txt")
    special_labels = {
        "bos_label": hparams["bos_index"],
        "eos_label": hparams["eos_index"],
        "blank_label": hparams["blank_index"],
    }

    label_encoder.load_or_create(
            path=lab_enc_file,
            from_didatasets=[train_data, test_data, valid_data],
            output_key="char_list",
            special_labels=special_labels,
            sequence_input=True,
        )



    return label_encoder



def test(hparams, run_opts, locales, wer_file="wer_test.txt"):
    # Test on old + new locales
    for i, locale in enumerate(locales):
        # Multi-gpu (ddp) save data preparation
        run_on_main(
            prepare_common_voice,
            kwargs={
                "locales": [locale],
                "download_dir": hparams["download_dir"],
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

        hparams["ctc_decoders"][locale].eval()
        # Retrieve corresponding tokenizer
        # Here we create the datasets objects as well as tokenization and encoding

        tokenizer = hparams["tokenizer_backup"][locale]
        _, _, test_data = dataio_prepare(hparams, tokenizer)

        hparams["decoder"].load_state_dict(
            hparams["decoder_state_backup"], strict=False
        )
        decoder_mask = hparams["decoder_mask"].get(locale)
        if decoder_mask is not None:
            for k, v in hparams["decoder"].named_parameters():
                if k not in decoder_mask:
                    continue
                v.detach_()
                with torch.no_grad():
                    thresholded_mask = Threshold.apply(
                        decoder_mask[k].to(v.device), hparams["mask_threshold"]
                    )
                    v *= thresholded_mask

        
        # Retrieve corresponding embedding layer and decoder layers
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


def train(hparams, run_opts):
    # Store embedding layer, decoder layers and tokenizer for each locale
    hparams["ctc_decoders"]={}
    hparams["tokenizer_backup"] = {}
    hparams["decoder_mask"] = {}
    hparams["decoder_state_backup"] = hparams[
        "wavlm"
    ].model.state_dict()
    for i, locale in enumerate(hparams["old_locales"]):

        # Train on (old) locales
        # Multi-gpu (ddp) save data preparation
        run_on_main(
            prepare_common_voice,
            kwargs={
                "locales": [locale],
                "download_dir": hparams["download_dir"],
                "max_durations": hparams["max_durations"],
            },
        )
        label_encoder = create_label_encoder(hparams, locale)
        hparams["tokenizer_backup"][locale] = label_encoder
        hparams["ctc_decoders"][locale] = Linear(input_size = 2048 ,n_neurons=len(label_encoder.ind2lab))

        hparams["forced_decoder_locale"] = locale
        hparams["new_locale"]=False
        # Here we create the datasets objects as well as tokenization and encoding
        label_encoder = hparams["tokenizer_backup"][locale]
        train_data, valid_data, test_data = dataio_prepare(hparams, label_encoder)

        # Trainer initialization
        checkpoint_dir = os.path.join(hparams["save_dir"], locale)
        os.makedirs(checkpoint_dir, exist_ok=True)
        hparams["checkpointer"].checkpoints_dir = pathlib.Path(checkpoint_dir)
        hparams["lr_annealing_model"].hyperparam_value = hparams["lr"]
        hparams["lr_annealing_model"].metric_values.clear()
        hparams["lr_annealing_model"].current_patient = 0

        hparams["model"] =torch.nn.ModuleList([hparams["decoder"], hparams["ctc_decoders"][locale]])
        asr_brain = ASR(
            modules=hparams["modules"],
            hparams=hparams,
            run_opts=run_opts,
            checkpointer=hparams["checkpointer"],
        )
        hparams["ctc_decoders"][locale].to(asr_brain.device)

        # We dynamically add the tokenizer to our brain class
        # NB: This tokenizer corresponds to the one used for Whisper
        asr_brain.tokenizer = label_encoder

        # Training
        hparams["valid_dataloader_kwargs"].pop("ckpt_prefix", None)
        hparams["epoch_counter"].current = 0
        asr_brain.fit(
            hparams["epoch_counter"],
            train_data,
            valid_data,
            train_loader_kwargs=hparams["train_dataloader_kwargs"],
            valid_loader_kwargs=hparams["valid_dataloader_kwargs"],
        )

    hparams["decoder_state_backup"] = asr_brain.modules.decoder.state_dict()
    for k in list(hparams["decoder_state_backup"]):
           hparams["decoder_state_backup"][k] = (
            hparams["decoder_state_backup"][k].clone().cpu()
        )



    for i, locale in enumerate(hparams["new_locales"]):

        # Train on new locales
        # Multi-gpu (ddp) save data preparation
        run_on_main(
            prepare_common_voice,
            kwargs={
                "locales": [locale],
                "download_dir": hparams["download_dir"],
                "max_durations": hparams["max_durations"],
            },
        )
        label_encoder = create_label_encoder(hparams, locale)
        hparams["tokenizer_backup"][locale] = label_encoder
        hparams["ctc_decoders"][locale] = Linear(input_size = 2048 ,n_neurons=len(label_encoder.ind2lab))


        hparams["decoder_mask"][locale] = {
            k: torch.full_like(v, hparams["mask_init"], requires_grad=True)
            for k, v in hparams["decoder"].named_parameters()
        }

 

        #hparams["model"] =torch.nn.ModuleList([hparams["rnn_decoders"][locale], hparams["ctc_decoders"][locale], hparams["linear_encs"][i]])
        # Set forced decoder locale
        hparams["forced_decoder_locale"] = locale

        # Here we create the datasets objects as well as tokenization and encoding
        label_encoder = hparams["tokenizer_backup"][locale]
        train_data, valid_data, test_data = dataio_prepare(hparams, label_encoder)

        # Trainer initialization
        checkpoint_dir = os.path.join(hparams["save_dir"], locale)
        os.makedirs(checkpoint_dir, exist_ok=True)
        hparams["checkpointer"].checkpoints_dir = pathlib.Path(checkpoint_dir)
        hparams["lr_annealing_model"].hyperparam_value = hparams["lr"]
        hparams["lr_annealing_model"].metric_values.clear()
        hparams["lr_annealing_model"].current_patient = 0

        hparams["model"] =torch.nn.ModuleList([ hparams["ctc_decoders"][locale]])
        asr_brain = ASR(
            modules=hparams["modules"],
            hparams=hparams,
            run_opts=run_opts,
            checkpointer=hparams["checkpointer"],
        )
        hparams["ctc_decoders"][locale].to(asr_brain.device)

        # We dynamically add the tokenizer to our brain class
        # NB: This tokenizer corresponds to the one used for Whisper
        asr_brain.tokenizer = label_encoder

        # Training
        hparams["valid_dataloader_kwargs"].pop("ckpt_prefix", None)
        hparams["epoch_counter"].current = 0
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
            hparams["new_locales"][: i + 1], #Test only on new locales
            f"wer_test_after_{locale}.txt",
        )


if __name__ == "__main__":
    # Command-line interface
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_dir"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Train
    start_time = time.time()
    train(hparams, run_opts)
    duration = time.time() - start_time
    logging.info(f"Time elapsed: {duration} seconds")
