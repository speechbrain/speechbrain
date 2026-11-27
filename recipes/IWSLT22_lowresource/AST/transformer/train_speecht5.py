#!/usr/bin/env python3
"""Recipe for training a SpeechT5-based AST system with IWSLT 2022 Tamasheq-French dataset.
The system uses SpeechT5 (https://arxiv.org/abs/2110.07205) and fine-tunes it on the data
with NLL loss.

This recipe uses the SpeechT5 for Speech-to-Text integration in SpeechBrain.
For more details about it, you can check speechbrain/lobes/models/huggingface_transformers/speecht5.py
Beam Search is used for the decoding step.

To run this recipe, do the following:
> python train_speechT5.py hparams/train_speecht5_st.yaml

Authors
 * Haroun Elleuch, 2024
"""

import logging
import sys

import torch
from hyperpyyaml import load_hyperpyyaml
from sacremoses import MosesDetokenizer

import speechbrain as sb
from speechbrain.utils.distributed import run_on_main

logger = logging.getLogger(__name__)


# Define training procedure
class ST(sb.core.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""

        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig  # audio
        bos_tokens, _ = batch.tokens_bos  # translation

        encoder_output, logits, _ = self.modules.speecht5(
            wav=wavs, decoder_input_ids=bos_tokens
        )

        log_probs = self.hparams.log_softmax(logits)

        # compute outputs
        hyps = None
        if stage == sb.Stage.VALID:
            # the output of the encoder is used for valid search
            hyps, _, _, _ = self.hparams.valid_search(
                encoder_output.detach(), wav_lens
            )

        elif stage == sb.Stage.TEST:
            hyps, _, _, _ = self.hparams.test_search(
                encoder_output.detach(), wav_lens
            )

        return log_probs, wav_lens, hyps

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss given predictions and targets."""
        (log_probs, _, hyps) = predictions
        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos

        # st loss
        loss = self.hparams.nll_loss(
            log_probs, tokens_eos, length=tokens_eos_lens
        )

        if stage != sb.Stage.TRAIN:
            fr_detokenizer = MosesDetokenizer(lang=self.hparams.lang)

            # Decode token terms to words
            predicted_words = self.tokenizer.batch_decode(
                hyps, skip_special_tokens=True
            )
            predicted_words = [text.split(" ") for text in predicted_words]
            predictions = [
                fr_detokenizer.detokenize(utt_seq)
                for utt_seq in predicted_words
            ]

            detokenized_translation = [
                fr_detokenizer.detokenize(translation.split(" "))
                for translation in batch.trans
            ]
            targets = [detokenized_translation]

            # compute scores for a one-step-forward prediction
            self.bleu_metric.append(ids, predictions, targets)
            self.acc_metric.append(log_probs, tokens_eos, tokens_eos_lens)

        return loss

    def on_stage_start(self, stage, epoch):
        """Gets called when a stage (either training, validation, test) starts."""

        if stage != sb.Stage.TRAIN:
            self.acc_metric = self.hparams.acc_computer()
            self.bleu_metric = self.hparams.bleu_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_loss

        else:  # valid or test
            stage_stats = {"loss": stage_loss}
            stage_stats["ACC"] = self.acc_metric.summarize()
            stage_stats["BLEU"] = self.bleu_metric.summarize(field="BLEU")
            stage_stats["BLEU_extensive"] = self.bleu_metric.summarize()

        # log stats and save checkpoint at end-of-epoch
        if stage == sb.Stage.VALID:
            current_epoch = self.hparams.epoch_counter.current
            old_lr_adam, new_lr_adam = self.hparams.lr_annealing_adam(
                stage_stats["BLEU"]
            )
            sb.nnet.schedulers.update_learning_rate(
                self.adam_optimizer, new_lr_adam
            )

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": current_epoch, "lr_adam": old_lr_adam},
                train_stats={"loss": self.train_stats},
                valid_stats=stage_stats,
            )

            # create checkpoint
            meta = {"BLEU": stage_stats["BLEU"], "epoch": current_epoch}
            name = "checkpoint_epoch" + str(current_epoch)

            self.checkpointer.save_and_keep_only(
                meta=meta, name=name, num_to_keep=1, max_keys=["BLEU"]
            )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )


# Define custom data procedure
def dataio_prepare(hparams, tokenizer):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    """

    # Define audio pipeline. In this case, we simply read the path contained
    # in the variable wav with the audio reader.
    @sb.utils.data_pipeline.takes("path")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        """Load the audio signal. This is done on the CPU in the `collate_fn`."""
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    # Define text processing pipeline. We start from the raw text and then
    # encode it using the tokenizer. The tokens with BOS are used for feeding
    # decoder during training, the tokens with EOS for computing the cost function.
    @sb.utils.data_pipeline.takes("trans")
    @sb.utils.data_pipeline.provides(
        "trans", "tokens_list", "tokens_bos", "tokens_eos"
    )
    def reference_text_pipeline(translation):
        """Processes the transcriptions to generate proper labels"""
        yield translation
        tokens_list = tokenizer.encode(translation, truncation=True)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos

    data_folder = hparams["data_folder"]

    # Load data and tokenize with tokenizer
    datasets = {}
    for dataset in ["train", "valid", "test"]:
        json_path = hparams[f"annotation_{dataset}"]

        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=json_path,
            replacements={"data_root": data_folder},
            dynamic_items=[audio_pipeline, reference_text_pipeline],
            output_keys=[
                "id",
                "sig",
                "duration",
                "trans",
                "tokens_list",
                "tokens_bos",
                "tokens_eos",
            ],
        )

    # Sorting training data with ascending order makes the code  much
    # faster  because we minimize zero-padding. In most of the cases, this
    # does not harm the performance.
    if hparams["sorting"] == "ascending":
        if hparams["debug"]:
            datasets["train"] = datasets["train"].filtered_sorted(
                key_min_value={"duration": hparams["sorting_min_duration"]},
                key_max_value={"duration": hparams["sorting_max_duration"]},
                sort_key="duration",
                reverse=True,
            )
            datasets["valid"] = datasets["valid"].filtered_sorted(
                key_min_value={"duration": hparams["sorting_min_duration"]},
                key_max_value={"duration": hparams["sorting_max_duration"]},
                sort_key="duration",
                reverse=True,
            )
        else:
            datasets["train"] = datasets["train"].filtered_sorted(
                sort_key="duration"
            )
            datasets["valid"] = datasets["valid"].filtered_sorted(
                sort_key="duration"
            )

        hparams["dataloader_options"]["shuffle"] = False
    elif hparams["sorting"] == "descending":
        # use smaller dataset to debug the model
        if hparams["debug"]:
            datasets["train"] = datasets["train"].filtered_sorted(
                key_min_value={"duration": hparams["sorting_min_duration"]},
                key_max_value={"duration": hparams["sorting_max_duration"]},
                sort_key="duration",
                reverse=True,
            )
            datasets["valid"] = datasets["valid"].filtered_sorted(
                key_min_value={"duration": hparams["sorting_min_duration"]},
                key_max_value={"duration": hparams["sorting_max_duration"]},
                sort_key="duration",
                reverse=True,
            )
        else:
            datasets["train"] = datasets["train"].filtered_sorted(
                sort_key="duration", reverse=True
            )
            datasets["valid"] = datasets["valid"].filtered_sorted(
                sort_key="duration", reverse=True
            )

        hparams["dataloader_options"]["shuffle"] = False
    elif hparams["sorting"] == "random":
        # use smaller dataset to debug the model
        if hparams["debug"]:
            datasets["train"] = datasets["train"].filtered_sorted(
                key_min_value={"duration": hparams["sorting_debug_duration"]},
                key_max_value={"duration": hparams["sorting_max_duration"]},
                sort_key="duration",
            )
            datasets["valid"] = datasets["valid"].filtered_sorted(
                key_min_value={"duration": hparams["sorting_min_duration"]},
                key_max_value={"duration": hparams["sorting_max_duration"]},
                sort_key="duration",
            )

        hparams["dataloader_options"]["shuffle"] = True
    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    return datasets


if __name__ == "__main__":

    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Create main experiment class
    st_brain = ST(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
        opt_class=hparams["adam_opt_class"],
    )

    # Data preparation
    from prepare_iwslt22 import data_proc

    run_on_main(
        data_proc,
        kwargs={
            "dataset_folder": hparams["root_data_folder"],
            "output_folder": hparams["data_folder"],
            "skip_prep": hparams["skip_prep"],
        },
    )

    # Load datasets for training, valid, and test, trains and applies tokenizer
    datasets = dataio_prepare(hparams, hparams["speecht5"].tokenizer)

    # Training
    st_brain.fit(
        st_brain.hparams.epoch_counter,
        datasets["train"],
        datasets["valid"],
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["test_dataloader_options"],
    )
    # Adding a tokenizer:
    st_brain.tokenizer = hparams["speecht5"].tokenizer

    # Test
    for dataset in ["valid", "test"]:
        st_brain.evaluate(
            datasets[dataset],
            test_loader_kwargs=hparams["test_dataloader_options"],
        )
