#!/usr/bin/env python3

"""Recipe for fine-tuning an OpenAI Whisper-based ASR system on Common Voice.

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
> python train_ft.py hparams/train_ft.yaml

NOTE: although checkpoints are saved regularly, automatic experiment resumption is not supported.
      To manually resume an experiment, you have to modify the script to load the correct checkpoint
      and set the corresponding state variables (e.g. current locale).

Authors
 * Luca Della Libera 2022
"""

import logging
import os
import pathlib
import sys
import time

import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml
from speechbrain.nnet.linear import Linear

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
        x = self.modules.decoder(feats)
        x=x[0]
        # Compute outputs
        p_tokens = None
        logits = self.modules.ctc_lin(x)
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
            self.wav2vec_optimizer.step()
            self.model_optimizer.step()
        self.wav2vec_optimizer.zero_grad()
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
            old_lr_wav2vec, new_lr_wav2vec = self.hparams.lr_annealing_wav2vec(
                stage_stats["loss"]
            )
            sb.nnet.schedulers.update_learning_rate(
                self.model_optimizer, new_lr_model
            )
            sb.nnet.schedulers.update_learning_rate(
                self.wav2vec_optimizer, new_lr_wav2vec
            )
            self.hparams.train_logger.log_stats(
                stats_meta={
                    "epoch": epoch,
                    "lr_model": old_lr_model,
                    "lr_wav2vec": old_lr_wav2vec,
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
        "Initializes the wav2vec2 optimizer and model optimizer"
        self.wav2vec_optimizer = self.hparams.wav2vec_opt_class(
            self.modules.wavlm.parameters()
        )
        self.model_optimizer = self.hparams.model_opt_class(
            self.hparams.model.parameters()
        )

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable(
                "wav2vec_opt", self.wav2vec_optimizer
            )
            self.checkpointer.add_recoverable("modelopt", self.model_optimizer)
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



def create_label_encoder(hparams, label_encoder, first):
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


    sb.dataio.dataset.add_dynamic_item([train_data, valid_data, test_data], text_pipeline)
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    special_labels = {
        "bos_label": hparams["bos_index"],
        "eos_label": hparams["eos_index"],
        "blank_label": hparams["blank_index"],
    }
    if first : 
        label_encoder.load_or_create(
            path=lab_enc_file,
            from_didatasets=[train_data, test_data, valid_data],
            output_key="char_list",
            special_labels=special_labels,
            sequence_input=True,
        )


    if not first : 
        label_encoder.update_from_didataset(
            didataset=train_data,
            output_key="char_list",
        )
        label_encoder.update_from_didataset(
            didataset=valid_data,
            output_key="char_list",
        )
        label_encoder.update_from_didataset(
            didataset=test_data,
            output_key="char_list",
        )



    return train_data, label_encoder



def test(hparams, run_opts, locales,label_encoder, wer_file="wer_test.txt"):
    # Test on old + new locales
    for locale in locales:
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

        # Define tokenizer
        tokenizer = label_encoder

        # Here we create the datasets objects as well as tokenization and encoding
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

#def merge_label_encoders(label_encoders):

def train(hparams, run_opts):
    #Create label encoder

    label_encoder = sb.dataio.encoder.CTCTextEncoder()
    first = True
    labels_locales = hparams["new_locales"] 
    if hparams["train_with_old_locales"]:
        labels_locales += hparams["old_locales"]
    for i, locale in enumerate(labels_locales):
        run_on_main(
            prepare_common_voice,
            kwargs={
                "locales": [locale],
                "download_dir": hparams["download_dir"],
                "max_durations": hparams["max_durations"],
            },
        )
        train_data,label_encoder = create_label_encoder(hparams, label_encoder, first)
        first=False # Very bad way to do this 
    
    print(label_encoder.ind2lab)
    #Redefine the output size of the CTC decoder
    hparams["ctc_lin"] = Linear(input_size = 2048 ,n_neurons=len(label_encoder.ind2lab))
    hparams["model"].append(hparams["ctc_lin"])
    hparams["modules"]["ctc_lin"] = hparams["ctc_lin"]
    if hparams["train_with_old_locales"] : 
        #Initialize the models with training on old locales
        for i, locale in enumerate(hparams["old_locales"]):
            print(f" training with locale {locale}")
            # Here we create the datasets objects as well as tokenization and encoding
            run_on_main(
                prepare_common_voice,
                kwargs={
                    "locales": [locale],
                    "download_dir": hparams["download_dir"],
                    "max_durations": hparams["max_durations"],
                },
            )
     
            train_data, valid_data, test_data = dataio_prepare(hparams, label_encoder)

            # Trainer initialization
            checkpoint_dir = os.path.join(hparams["save_dir"], locale)
            os.makedirs(checkpoint_dir, exist_ok=True)
            hparams["checkpointer"].checkpoints_dir = pathlib.Path(checkpoint_dir)
            #Reinitialize learning rrates 
            hparams["lr_annealing_model"].hyperparam_value = hparams["lr"]
            hparams["lr_annealing_model"].metric_values.clear()
            hparams["lr_annealing_model"].current_patient = 0
            hparams["lr_annealing_wav2vec"].hyperparam_value = hparams["lr_wav2vec"]
            hparams["lr_annealing_wav2vec"].metric_values.clear()
            hparams["lr_annealing_wav2vec"].current_patient = 0

            asr_brain = ASR(
                modules=hparams["modules"],
                hparams=hparams,
                run_opts=run_opts,
                checkpointer=hparams["checkpointer"],
            )



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

    # Train on new locales
    for i, locale in enumerate(hparams["new_locales"]):
        print(f" training with locale {locale}")
        # Here we create the datasets objects as well as tokenization and encoding
        run_on_main(
            prepare_common_voice,
            kwargs={
                "locales": [locale],
                "download_dir": hparams["download_dir"],
                "max_durations": hparams["max_durations"],
            },
        )
 
        train_data, valid_data, test_data = dataio_prepare(hparams, label_encoder)

        # Trainer initialization
        checkpoint_dir = os.path.join(hparams["save_dir"], locale)
        os.makedirs(checkpoint_dir, exist_ok=True)
        hparams["checkpointer"].checkpoints_dir = pathlib.Path(checkpoint_dir)
        #Reinitialize learning rrates 
        hparams["lr_annealing_model"].hyperparam_value = hparams["lr"]
        hparams["lr_annealing_model"].metric_values.clear()
        hparams["lr_annealing_model"].current_patient = 0
        hparams["lr_annealing_wav2vec"].hyperparam_value = hparams["lr_wav2vec"]
        hparams["lr_annealing_wav2vec"].metric_values.clear()
        hparams["lr_annealing_wav2vec"].current_patient = 0

        asr_brain = ASR(
            modules=hparams["modules"],
            hparams=hparams,
            run_opts=run_opts,
            checkpointer=hparams["checkpointer"],
        )



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
            hparams["new_locales"][: i + 1], #Only test on new locales
            label_encoder,
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
