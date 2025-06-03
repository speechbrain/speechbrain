#!/usr/bin/env/python3
"""Recipe for training a Transducer ASR system with CommonVoice.
The system employs an encoder, a decoder, and an joint network
between them following Dynamic Chunk Training for streaming speech recognition.

To run this recipe, do the following:
> python train.py hparams/conformer_transducer_large.yaml

The neural network is trained on both CTC and negative-log likelihood
targets and sub-word units estimated with Byte Pairwise Encoding (BPE)
are used as basic recognition tokens.

The experiment file is flexible enough to support a large variety of
different systems. By properly changing the parameter files, you can try
different encoders, decoders, tokens (e.g, characters instead of BPE),
training CommonVoice language, different streaming, parameters,
and many other possible variations.


Authors
 * Titouan Parcollet 2024
 * Abdel Heba 2020
 * Mirco Ravanelli 2020
 * Ju-Chieh Chou 2020
 * Peter Plantinga 2020
"""

import sys

import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.tokenizers.SentencePiece import SentencePiece
from speechbrain.utils.data_utils import undo_padding
from speechbrain.utils.distributed import if_main_process, run_on_main
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)

# Define training procedure


class ASR(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        tokens_with_bos, token_with_bos_lens = batch.tokens_bos

        feats = self.hparams.compute_features(wavs)

        # Add feature augmentation if specified.
        if (
            stage == sb.Stage.TRAIN
            and hasattr(self.hparams, "fea_augment")
            and self.optimizer_step > self.hparams.augment_warmup
        ):
            feats, fea_lens = self.hparams.fea_augment(feats, wav_lens)
            tokens_with_bos = self.hparams.fea_augment.replicate_labels(
                tokens_with_bos
            )

        current_epoch = self.hparams.epoch_counter.current

        # Old models may not have the streaming hparam, we don't break them in
        # any other way so just check for its presence
        if hasattr(self.hparams, "streaming") and self.hparams.streaming:
            dynchunktrain_config = self.hparams.dynchunktrain_config_sampler(
                stage
            )
        else:
            dynchunktrain_config = None

        feats = self.modules.normalize(feats, wav_lens, epoch=current_epoch)

        src = self.modules.CNN(feats)
        x = self.modules.enc(
            src,
            wav_lens,
            pad_idx=self.hparams.pad_index,
            dynchunktrain_config=dynchunktrain_config,
        )
        x = self.modules.proj_enc(x)

        e_in = self.modules.emb(tokens_with_bos)
        e_in = torch.nn.functional.dropout(
            e_in,
            self.hparams.dec_emb_dropout,
            training=(stage == sb.Stage.TRAIN),
        )
        h, _ = self.modules.dec(e_in)
        h = torch.nn.functional.dropout(
            h, self.hparams.dec_dropout, training=(stage == sb.Stage.TRAIN)
        )
        h = self.modules.proj_dec(h)

        # Joint network
        # add labelseq_dim to the encoder tensor: [B,T,H_enc] => [B,T,1,H_enc]
        # add timeseq_dim to the decoder tensor: [B,U,H_dec] => [B,1,U,H_dec]
        joint = self.modules.Tjoint(x.unsqueeze(2), h.unsqueeze(1))

        # Output layer for transducer log-probabilities
        logits_transducer = self.modules.transducer_lin(joint)

        # Compute outputs
        if stage == sb.Stage.TRAIN:
            p_ctc = None
            p_ce = None

            if (
                self.hparams.ctc_weight > 0.0
                and current_epoch <= self.hparams.number_of_ctc_epochs
            ):
                # Output layer for ctc log-probabilities
                out_ctc = self.modules.proj_ctc(x)
                p_ctc = self.hparams.log_softmax(out_ctc)

            if self.hparams.ce_weight > 0.0:
                # Output layer for ctc log-probabilities
                p_ce = self.modules.dec_lin(h)
                p_ce = self.hparams.log_softmax(p_ce)

            return p_ctc, p_ce, logits_transducer, wav_lens

        else:
            best_hyps, scores, _, _ = self.hparams.Greedysearcher(x)
            return logits_transducer, wav_lens, best_hyps

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (Transducer+(CTC+NLL)) given predictions and targets."""

        ids = batch.id
        tokens, token_lens = batch.tokens
        tokens_eos, token_eos_lens = batch.tokens_eos

        # Train returns 4 elements vs 3 for val and test
        if len(predictions) == 4:
            p_ctc, p_ce, logits_transducer, wav_lens = predictions
        else:
            logits_transducer, wav_lens, predicted_tokens = predictions

        if stage == sb.Stage.TRAIN:
            if (
                hasattr(self.hparams, "fea_augment")
                and self.optimizer_step > self.hparams.augment_warmup
            ):
                tokens = self.hparams.fea_augment.replicate_labels(tokens)
                token_lens = self.hparams.fea_augment.replicate_labels(
                    token_lens
                )
                tokens_eos = self.hparams.fea_augment.replicate_labels(
                    tokens_eos
                )
                token_eos_lens = self.hparams.fea_augment.replicate_labels(
                    token_eos_lens
                )

        if stage == sb.Stage.TRAIN:
            CTC_loss = 0.0
            CE_loss = 0.0
            if p_ctc is not None:
                CTC_loss = self.hparams.ctc_cost(
                    p_ctc, tokens, wav_lens, token_lens
                )
            if p_ce is not None:
                CE_loss = self.hparams.ce_cost(
                    p_ce, tokens_eos, length=token_eos_lens
                )
            loss_transducer = self.hparams.transducer_cost(
                logits_transducer, tokens, wav_lens, token_lens
            )
            loss = (
                self.hparams.ctc_weight * CTC_loss
                + self.hparams.ce_weight * CE_loss
                + (1 - (self.hparams.ctc_weight + self.hparams.ce_weight))
                * loss_transducer
            )
        else:
            loss = self.hparams.transducer_cost(
                logits_transducer, tokens, wav_lens, token_lens
            )

        if stage != sb.Stage.TRAIN:
            # Decode token terms to words
            predicted_words = self.tokenizer(
                predicted_tokens, task="decode_from_list"
            )

            # Convert indices to words
            target_words = undo_padding(tokens, token_lens)
            target_words = self.tokenizer(target_words, task="decode_from_list")

            self.wer_metric.append(ids, predicted_words, target_words)
            self.cer_metric.append(ids, predicted_words, target_words)

        return loss

    def on_fit_batch_end(self, batch, outputs, loss, should_step):
        """At the end of the optimizer step, apply noam annealing."""
        if should_step:
            self.hparams.noam_annealing(self.optimizer)

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

            lr = self.hparams.noam_annealing.current_lr
            steps = self.optimizer_step
            optimizer = self.optimizer.__class__.__name__

            epoch_stats = {
                "epoch": epoch,
                "lr": lr,
                "steps": steps,
                "optimizer": optimizer,
            }

            self.hparams.train_logger.log_stats(
                stats_meta=epoch_stats,
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"WER": stage_stats["WER"], "epoch": epoch},
                min_keys=["WER"],
                num_to_keep=self.hparams.avg_checkpoints,
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

            # save the averaged checkpoint at the end of the evaluation stage
            # delete the rest of the intermediate checkpoints
            # WER is set to -0.1 so checkpointer only keeps the averaged checkpoint
            self.checkpointer.save_and_keep_only(
                meta={"WER": -0.1, "epoch": epoch},
                min_keys=["WER"],
                num_to_keep=1,
            )


def dataio_prepare(hparams, tokenizer):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    """

    # 1. Define datasets
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"],
        replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(
            sort_key="duration",
            key_max_value={"duration": hparams["avoid_if_longer_than"]},
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False
    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass
    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )
    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"],
        replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    # test is separate
    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_csv"],
        replacements={"data_root": data_folder},
    )

    # We also sort the test data so it is faster to test
    test_data = test_data.filtered_sorted(sort_key="duration")

    datasets = [train_data, valid_data, test_data]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        info = torchaudio.info(wav)
        sig = sb.dataio.dataio.read_audio(wav)
        resampled = torchaudio.transforms.Resample(
            info.sample_rate,
            hparams["sample_rate"],
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
        ["id", "sig", "wrd", "tokens_bos", "tokens_eos", "tokens"],
    )

    # 5. If Dynamic Batching is used, we instantiate the needed samplers.
    train_batch_sampler = None
    valid_batch_sampler = None
    if hparams["dynamic_batching"]:
        from speechbrain.dataio.sampler import DynamicBatchSampler  # noqa

        dynamic_hparams = hparams["dynamic_batch_sampler"]
        num_buckets = dynamic_hparams["num_buckets"]

        train_batch_sampler = DynamicBatchSampler(
            train_data,
            dynamic_hparams["max_batch_len"],
            num_buckets=num_buckets,
            length_func=lambda x: x["duration"],
            shuffle=dynamic_hparams["shuffle_ex"],
            batch_ordering=dynamic_hparams["batch_ordering"],
        )

        valid_batch_sampler = DynamicBatchSampler(
            valid_data,
            dynamic_hparams["max_batch_len_val"],
            num_buckets=num_buckets,
            length_func=lambda x: x["duration"],
            shuffle=dynamic_hparams["shuffle_ex"],
            batch_ordering=dynamic_hparams["batch_ordering"],
        )

    return (
        train_data,
        valid_data,
        test_data,
        tokenizer,
        train_batch_sampler,
        valid_batch_sampler,
    )


if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # Dataset preparation (parsing CommonVoice)
    from common_voice_prepare import prepare_common_voice  # noqa

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Due to DDP, we do the preparation ONLY on the main python process
    run_on_main(
        prepare_common_voice,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["save_folder"],
            "train_tsv_file": hparams["train_tsv_file"],
            "dev_tsv_file": hparams["dev_tsv_file"],
            "test_tsv_file": hparams["test_tsv_file"],
            "accented_letters": hparams["accented_letters"],
            "language": hparams["language"],
            "skip_prep": hparams["skip_prep"],
        },
    )

    # Defining tokenizer and loading it
    tokenizer = SentencePiece(
        model_dir=hparams["save_folder"],
        vocab_size=hparams["output_neurons"],
        annotation_train=hparams["train_csv"],
        annotation_read="wrd",
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
        tokenizer,
        train_bsampler,
        valid_bsampler,
    ) = dataio_prepare(hparams, tokenizer)

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # We dynamically add the tokenizer to our brain class.
    # NB: This tokenizer corresponds to the one used for the LM!!
    asr_brain.tokenizer = tokenizer
    train_dataloader_opts = hparams["train_dataloader_opts"]
    valid_dataloader_opts = hparams["valid_dataloader_opts"]

    if train_bsampler is not None:
        train_dataloader_opts = {
            "batch_sampler": train_bsampler,
            "num_workers": hparams["num_workers"],
        }

    if valid_bsampler is not None:
        valid_dataloader_opts = {"batch_sampler": valid_bsampler}

    # Training
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=train_dataloader_opts,
        valid_loader_kwargs=valid_dataloader_opts,
    )

    # Testing
    asr_brain.evaluate(
        test_data,
        min_key="WER",
        test_loader_kwargs=hparams["test_dataloader_opts"],
    )
