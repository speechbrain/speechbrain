#!/usr/bin/env python3
"""Recipe for training a whisper-based ASR system with CommonVoice.
The system employs whisper from OpenAI (https://cdn.openai.com/papers/whisper.pdf).
This recipe take the whisper encoder-decoder to fine-tune on.

To run this recipe, do the following:
> python train_with_whisper.py hparams/train_<locale>_hf_whisper.yaml

 * Pooneh Mousavi 2022
"""

import sys
import torch
import logging
import torchaudio
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main, if_main_process
from speechbrain.utils.data_utils import undo_padding
from hyperpyyaml import load_hyperpyyaml
from transformers.models.whisper.tokenization_whisper import LANGUAGES

logger = logging.getLogger(__name__)


# Define training procedure
class ASR(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        bos_tokens, bos_tokens_lens = batch.tokens_bos

        # Add augmentation if specified
        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs, wav_lens)

        # We compute the padding mask and replace the values with the pad_token_id
        # that the Whisper decoder expect to see.
        abs_tokens_lens = (bos_tokens_lens * bos_tokens.shape[1]).long()
        pad_mask = (
            torch.arange(abs_tokens_lens.max(), device=self.device)[None, :]
            < abs_tokens_lens[:, None]
        )
        bos_tokens[~pad_mask] = self.tokenizer.pad_token_id

        # Forward encoder + decoder
        enc_out, logits, _ = self.modules.whisper(wavs, bos_tokens)

        hyps = None
        if stage == sb.Stage.VALID:
            hyps, _ = self.hparams.valid_greedy_searcher(enc_out, wav_lens)
        elif stage == sb.Stage.TEST:
            hyps, _ = self.hparams.valid_greedy_searcher(enc_out, wav_lens)

        return logits, hyps, wav_lens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss NLL given predictions and targets."""

        logits, hyps, wav_lens, = predictions
        batch = batch.to(self.device)
        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos

        log_probs = self.hparams.log_softmax(logits)
        loss = self.hparams.nll_loss(
            log_probs, tokens_eos, length=tokens_eos_lens,
        )

        if stage != sb.Stage.TRAIN:
            tokens, tokens_lens = batch.tokens

            # Decode token terms to words
            predicted_words = self.tokenizer.batch_decode(
                hyps, skip_special_tokens=True
            )

            # Convert indices to words
            target_words = undo_padding(tokens, tokens_lens)
            target_words = self.tokenizer.batch_decode(
                target_words, skip_special_tokens=True
            )

            if hasattr(self.hparams, "normalized_transcripts"):
                predicted_words = [
                    self.tokenizer._normalize(text).split(" ")
                    for text in predicted_words
                ]

                target_words = [
                    self.tokenizer._normalize(text).split(" ")
                    for text in target_words
                ]
            else:
                predicted_words = [text.split(" ") for text in predicted_words]

                target_words = [text.split(" ") for text in target_words]
            self.wer_metric.append(ids, predicted_words, target_words)
            self.cer_metric.append(ids, predicted_words, target_words)

        return loss

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

            old_lr_whisper, new_lr_whisper = self.hparams.lr_annealing_whisper(
                stage_stats["loss"]
            )

            sb.nnet.schedulers.update_learning_rate(
                self.optimizer, new_lr_whisper
            )
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr_whisper": old_lr_whisper},
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
            if if_main_process():
                with open(self.hparams.test_wer_file, "w") as w:
                    self.wer_metric.write_stats(w)


def dataio_prepare(hparams, tokenizer):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"], replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(
            sort_key="duration",
            key_max_value={"duration": hparams["avoid_if_longer_than"]},
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_loader_kwargs"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration",
            reverse=True,
            key_max_value={"duration": hparams["avoid_if_longer_than"]},
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_loader_kwargs"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"], replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    # test is separate
    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_csv"], replacements={"data_root": data_folder},
    )

    datasets = [train_data, valid_data, test_data]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        info = torchaudio.info(wav)
        sig = sb.dataio.dataio.read_audio(wav)
        resampled = torchaudio.transforms.Resample(
            info.sample_rate, hparams["sample_rate"],
        )(sig)
        return resampled

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "wrd", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(wrd):
        yield wrd
        tokens_list = tokenizer.encode(wrd)
        # avoid bos and eos tokens.
        tokens_list = tokens_list[1:-1]
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + tokens_list)
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "sig", "tokens_list", "tokens_bos", "tokens_eos", "tokens"],
    )

    return train_data, valid_data, test_data


if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Dataset prep (parsing Librispeech)
    from common_voice_prepare import prepare_common_voice  # noqa

    # multi-gpu (ddp) save data preparation
    run_on_main(
        prepare_common_voice,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["save_folder"],
            "train_tsv_file": hparams["train_tsv_file"],
            "dev_tsv_file": hparams["dev_tsv_file"],
            "test_tsv_file": hparams["test_tsv_file"],
            "accented_letters": hparams["accented_letters"],
            "language": hparams["locale"],
            "skip_prep": hparams["skip_prep"],
        },
    )
    # Defining tokenizer and loading it
    tokenizer = hparams["whisper"].tokenizer
    language = LANGUAGES[hparams["locale"]]

    tokenizer.set_prefix_tokens(language, "transcribe", False)

    # we need to prepare the tokens for searchers
    hparams["valid_greedy_searcher"].set_decoder_input_tokens(
        tokenizer.prefix_tokens
    )
    hparams["valid_greedy_searcher"].set_language_token(
        tokenizer.prefix_tokens[1]
    )

    hparams["test_beam_searcher"].set_decoder_input_tokens(
        tokenizer.prefix_tokens
    )
    hparams["test_beam_searcher"].set_language_token(tokenizer.prefix_tokens[1])

    # here we create the datasets objects as well as tokenization and encoding
    train_data, valid_data, test_data = dataio_prepare(hparams, tokenizer)

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
        opt_class=hparams["whisper_opt_class"],
    )

    # We load the pretrained whisper model
    if "pretrainer" in hparams.keys():
        run_on_main(hparams["pretrainer"].collect_files)
        hparams["pretrainer"].load_collected(asr_brain.device)

    # We dynamicaly add the tokenizer to our brain class.
    # NB: This tokenizer corresponds to the one used for Whisper.
    asr_brain.tokenizer = tokenizer

    # Training
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_loader_kwargs"],
        valid_loader_kwargs=hparams["valid_loader_kwargs"],
    )

    # Testing
    asr_brain.hparams.test_wer_file = hparams["test_wer_file"]
    asr_brain.evaluate(
        test_data,
        min_key="WER",
        test_loader_kwargs=hparams["test_loader_kwargs"],
    )

    asr_brain.hparams.test_wer_file = hparams["valid_wer_file"]
    asr_brain.evaluate(
        valid_data,
        min_key="WER",
        test_loader_kwargs=hparams["test_loader_kwargs"],
    )
