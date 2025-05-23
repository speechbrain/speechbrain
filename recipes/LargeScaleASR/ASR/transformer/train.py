#!/usr/bin/env python3
"""Recipe for training a Transformer ASR system with The LargeScaleASR Set.

Authors
-------
 * Titouan Parcollet 2024
"""

import os
import sys

import torch
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.dataio.dataio import read_audio
from speechbrain.dataio.sampler import DynamicBatchSampler  # noqa
from speechbrain.tokenizers.SentencePiece import SentencePiece
from speechbrain.utils.data_utils import undo_padding
from speechbrain.utils.distributed import if_main_process
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


# Define training procedure
class ASR(sb.core.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        tokens_bos, _ = batch.tokens_bos
        # Add waveform augmentation if specified.
        if (
            stage == sb.Stage.TRAIN
            and hasattr(self.hparams, "wav_augment")
            and self.optimizer_step > self.hparams.augment_warmup
        ):
            wavs, wav_lens = self.hparams.wav_augment(wavs, wav_lens)
            tokens_bos = self.hparams.wav_augment.replicate_labels(tokens_bos)

        # compute features
        feats = self.hparams.compute_features(wavs)
        current_epoch = self.hparams.epoch_counter.current
        feats = self.hparams.normalize(feats, wav_lens, epoch=current_epoch)

        # Add feature augmentation if specified.
        if (
            stage == sb.Stage.TRAIN
            and hasattr(self.hparams, "fea_augment")
            and self.optimizer_step > self.hparams.augment_warmup
        ):
            feats, fea_lens = self.hparams.fea_augment(feats, wav_lens)
            tokens_bos = self.hparams.fea_augment.replicate_labels(tokens_bos)

        # forward modules
        src = self.modules.CNN(feats)
        enc_out, pred = self.modules.Transformer(
            src, tokens_bos, wav_lens, pad_idx=self.hparams.pad_index
        )

        # output layer for ctc log-probabilities
        logits = self.modules.ctc_lin(enc_out)
        p_ctc = self.hparams.log_softmax(logits)

        # output layer for seq2seq log-probabilities
        pred = self.modules.seq_lin(pred)
        p_seq = self.hparams.log_softmax(pred)

        # Compute outputs
        hyps = None
        current_epoch = self.hparams.epoch_counter.current
        is_valid_search = (
            stage == sb.Stage.VALID
            and current_epoch % self.hparams.valid_search_interval == 0
        )
        is_test_search = stage == sb.Stage.TEST

        if is_valid_search:
            hyps, _, _, _ = self.hparams.valid_search(
                enc_out.detach(), wav_lens
            )

        elif is_test_search:
            hyps, _, _, _ = self.hparams.test_search(enc_out.detach(), wav_lens)

        return p_ctc, p_seq, wav_lens, hyps

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""

        (p_ctc, p_seq, wav_lens, predicted_tokens) = predictions

        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        tokens, tokens_lens = batch.tokens

        # Augment Labels
        if stage == sb.Stage.TRAIN:
            # Labels must be extended if parallel augmentation or concatenated
            # augmentation was performed on the input (increasing the time dimension)
            if (
                hasattr(self.hparams, "wav_augment")
                and self.optimizer_step > self.hparams.augment_warmup
            ):
                (
                    tokens,
                    tokens_lens,
                    tokens_eos,
                    tokens_eos_lens,
                ) = self.hparams.wav_augment.replicate_multiple_labels(
                    tokens, tokens_lens, tokens_eos, tokens_eos_lens
                )
            if (
                hasattr(self.hparams, "fea_augment")
                and self.optimizer_step > self.hparams.augment_warmup
            ):
                (
                    tokens,
                    tokens_lens,
                    tokens_eos,
                    tokens_eos_lens,
                ) = self.hparams.fea_augment.replicate_multiple_labels(
                    tokens, tokens_lens, tokens_eos, tokens_eos_lens
                )

        loss_seq = self.hparams.seq_cost(
            p_seq, tokens_eos, length=tokens_eos_lens
        )
        loss_ctc = self.hparams.ctc_cost(p_ctc, tokens, wav_lens, tokens_lens)
        loss = (
            self.hparams.ctc_weight * loss_ctc
            + (1 - self.hparams.ctc_weight) * loss_seq
        )

        if stage != sb.Stage.TRAIN:
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval
            if current_epoch % valid_search_interval == 0 or (
                stage == sb.Stage.TEST
            ):
                # Decode token terms to words
                predicted_words = self.tokenizer(
                    predicted_tokens, task="decode_from_list"
                )

                # Convert indices to words
                target_words = undo_padding(tokens, tokens_lens)
                target_words = self.tokenizer(
                    target_words, task="decode_from_list"
                )
                if not isinstance(ids, list):
                    ids = ids.tolist()
                self.wer_metric.append(ids, predicted_words, target_words)
                self.cer_metric.append(ids, predicted_words, target_words)

            # compute the accuracy of the one-step-forward prediction
            self.acc_metric.append(p_seq, tokens_eos, tokens_eos_lens)
        return loss

    def on_fit_batch_end(self, batch, outputs, loss, should_step):
        """At the end of the optimizer step, apply noam annealing."""
        if should_step:
            self.hparams.noam_annealing(self.optimizer)

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.acc_metric = self.hparams.acc_computer()
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["ACC"] = self.acc_metric.summarize()
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval
            if (
                current_epoch % valid_search_interval == 0
                or stage == sb.Stage.TEST
            ):
                stage_stats["WER"] = self.wer_metric.summarize("error_rate")
                stage_stats["CER"] = self.cer_metric.summarize("error_rate")

        # log stats and save checkpoint at end-of-epoch
        if stage == sb.Stage.VALID:
            # report different epoch stages according current stage
            current_epoch = self.hparams.epoch_counter.current
            lr = self.hparams.noam_annealing.current_lr
            steps = self.hparams.noam_annealing.current_step

            epoch_stats = {
                "epoch": epoch,
                "lr": lr,
                "steps": steps,
            }
            self.hparams.train_logger.log_stats(
                stats_meta=epoch_stats,
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"ACC": stage_stats["ACC"], "epoch": epoch},
                max_keys=["ACC"],
            )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            if if_main_process():
                with open(
                    self.hparams.test_wer_file, "w", encoding="utf-8"
                ) as w:
                    self.wer_metric.write_stats(w)


# Define custom data procedure
def dataio_prepare_hf(hparams, tokenizer):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    This is valid for The LargeScaleASR Set prepared with HuggingFace
    """
    from large_scale_ASR_prepare import load_datasets

    hf_data_dict = load_datasets(
        hparams["tls_subset"],
        hparams["hf_hub"],
        hparams["hf_caching_dir"],
    )

    # We must rename the 'id' column because SpeechBrain sampling use this
    # name for the sampler already, also it's not an id, but an audio_path.
    train_data = hf_data_dict["train"].rename_column("ID", "audio_id")
    valid_data = hf_data_dict["dev"].rename_column("ID", "audio_id")
    test_data = hf_data_dict["test"].rename_column("ID", "audio_id")

    # We need to get the full list of durations of all samples to enable
    # bucketing from the dynamic batch sampler. We do it that way instead
    # of the usual iterable because the HF dataset ALWAYS open the file
    # when called, which means that the dynamic sampling needs to read the
    # 1.5M audio samples from disk.... using a list instead is much master.
    train_len_list = list(train_data.select_columns("duration")["duration"])
    val_len_list = list(valid_data.select_columns("duration")["duration"])

    train_data = sb.dataio.dataset.DynamicItemDataset.from_arrow_dataset(
        train_data,
    )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_arrow_dataset(
        valid_data,
    )

    test_data = sb.dataio.dataset.DynamicItemDataset.from_arrow_dataset(
        test_data,
    )

    # we sort testing/val data to speed up decoding and get better results.
    valid_data = valid_data.filtered_sorted(
        sort_key="duration",
    )
    test_data = test_data.filtered_sorted(
        sort_key="duration",
    )

    datasets = [train_data, valid_data, test_data]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = read_audio(wav["bytes"])
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("text")
    @sb.utils.data_pipeline.provides(
        "wrd", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(wrd):
        yield wrd
        tokens_list = tokenizer.sp.encode_as_ids(wrd)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "sig", "tokens_bos", "tokens_eos", "tokens"],
    )

    # 5. we instantiate the needed samplers with dynamic batching
    dynamic_hparams_train = hparams["dynamic_batch_sampler_train"]
    dynamic_hparams_valid = hparams["dynamic_batch_sampler_valid"]

    train_batch_sampler = DynamicBatchSampler(
        train_data,
        length_func=lambda x: x["duration"],
        lengths_list=train_len_list,
        **dynamic_hparams_train,
    )
    valid_batch_sampler = DynamicBatchSampler(
        valid_data,
        length_func=lambda x: x["duration"],
        lengths_list=val_len_list,
        **dynamic_hparams_valid,
    )

    train_loader_kwargs = {
        "batch_sampler": train_batch_sampler,
        "num_workers": hparams["num_workers"],
    }
    valid_loader_kwargs = {"batch_sampler": valid_batch_sampler}

    return (
        train_data,
        valid_data,
        test_data,
        train_loader_kwargs,
        valid_loader_kwargs,
    )


if __name__ == "__main__":
    # CLI:
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

    # Defining tokenizer and loading it
    tokenizer = SentencePiece(
        model_dir=hparams["save_folder"],
        vocab_size=hparams["output_neurons"],
        annotation_train=hparams["train_csv"],
        annotation_read="text",
        model_type=hparams["token_type"],
        character_coverage=hparams["character_coverage"],
        bos_id=hparams["bos_index"],
        eos_id=hparams["eos_index"],
    )

    # here we create the datasets objects as well as tokenization and encoding
    (
        train_data,
        valid_data,
        test_data,
        train_loader_kwargs,
        valid_loader_kwargs,
    ) = dataio_prepare_hf(hparams, tokenizer)

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["Adam"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # adding objects to trainer:
    asr_brain.tokenizer = tokenizer

    # Training
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=train_loader_kwargs,
        valid_loader_kwargs=valid_loader_kwargs,
    )

    # Testing
    os.makedirs(hparams["output_wer_folder"], exist_ok=True)

    asr_brain.hparams.test_wer_file = os.path.join(
        hparams["output_wer_folder"], "wer_valid.txt"
    )
    asr_brain.evaluate(
        valid_data,
        max_key="ACC",
        test_loader_kwargs=hparams["test_dataloader_opts"],
    )

    asr_brain.hparams.test_wer_file = os.path.join(
        hparams["output_wer_folder"], "wer_test.txt"
    )
    asr_brain.evaluate(
        test_data,
        max_key="ACC",
        test_loader_kwargs=hparams["test_dataloader_opts"],
    )
