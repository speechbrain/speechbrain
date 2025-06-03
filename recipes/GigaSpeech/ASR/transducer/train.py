#!/usr/bin/env/python3
"""Recipe for training a Transducer ASR system with GigaSpeech.
The system employs an encoder, a decoder, and an joint network
between them. Decoding is performed with beamsearch coupled with a neural
language model.

To run this recipe, do the following:
> python train.py hparams/conformer_transducer.yaml

With the default hyperparameters, the system employs a conformer encoder.
The decoder is based on a standard LSTM. Beamsearch coupled with a RNN
language model is used on the top of decoder probabilities.

The neural network is trained on both CTC and negative-log likelihood
targets and sub-word units estimated with Byte Pairwise Encoding (BPE)
are used as basic recognition tokens.

The experiment file is flexible enough to support a large variety of
different systems. By properly changing the parameter files, you can try
different encoders, decoders, tokens (e.g, characters instead of BPE),
training split, and many
other possible variations.


Authors
 * Sylvain de Langen 2024
 * Titouan Parcollet 2024
 * Abdel Heba 2020
 * Mirco Ravanelli 2020
 * Ju-Chieh Chou 2020
 * Peter Plantinga 2020
"""

import os
import sys

import torch
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.tokenizers.SentencePiece import SentencePiece
from speechbrain.utils.data_utils import undo_padding
from speechbrain.utils.distributed import if_main_process, run_on_main
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


class ASR(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        tokens_with_bos, token_with_bos_lens = batch.tokens_bos

        # Add waveform augmentation if specified.
        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "wav_augment"):
                wavs, wav_lens = self.hparams.wav_augment(wavs, wav_lens)
                tokens_with_bos = self.hparams.wav_augment.replicate_labels(
                    tokens_with_bos
                )

        feats = self.hparams.compute_features(wavs)

        # Add feature augmentation if specified.
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "fea_augment"):
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

        elif stage == sb.Stage.VALID:
            best_hyps, scores, _, _ = self.hparams.Greedysearcher(x)
            return logits_transducer, wav_lens, best_hyps
        else:
            (
                best_hyps,
                best_scores,
                nbest_hyps,
                nbest_scores,
            ) = self.hparams.Beamsearcher(x)
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
            # Labels must be extended if parallel augmentation or concatenated
            # augmentation was performed on the input (increasing the time dimension)
            if hasattr(self.hparams, "fea_augment"):
                (
                    tokens,
                    token_lens,
                    tokens_eos,
                    token_eos_lens,
                ) = self.hparams.fea_augment.replicate_multiple_labels(
                    tokens, token_lens, tokens_eos, token_eos_lens
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

    def on_evaluate_start(self, max_key=None, min_key=None):
        """perform checkpoint average if needed"""
        super().on_evaluate_start()

        ckpts = self.checkpointer.find_checkpoints(
            max_key=max_key,
            min_key=min_key,
        )
        ckpt = sb.utils.checkpoints.average_checkpoints(
            ckpts, recoverable_name="model"
        )

        self.hparams.model.load_state_dict(ckpt, strict=True)
        self.hparams.model.eval()


def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    """
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"],
        replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
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

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_csv"],
        replacements={"data_root": data_folder},
    )

    # We also sort the validation data so it is faster to validate
    test_data = test_data.filtered_sorted(sort_key="duration")

    datasets = [train_data, valid_data, test_data]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("audio_path", "begin_time", "end_time")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(audio_path, begin_time, end_time):
        if hparams["download_with_HF"]:
            sig = sb.dataio.dataio.read_audio(audio_path)
        else:
            start_sample = int(float(begin_time) * hparams["sample_rate"])
            stop_sample = int(float(end_time) * hparams["sample_rate"])
            sig = sb.dataio.dataio.read_audio(
                {"file": audio_path, "start": start_sample, "stop": stop_sample}
            )
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
        ["id", "sig", "wrd", "tokens_bos", "tokens_eos", "tokens"],
    )

    # 5. If Dynamic Batching is used, we instantiate the needed samplers.
    train_batch_sampler = None
    valid_batch_sampler = None
    if hparams["dynamic_batching"]:
        from speechbrain.dataio.sampler import DynamicBatchSampler  # noqa

        dynamic_hparams_train = hparams["dynamic_batch_sampler_train"]
        dynamic_hparams_valid = hparams["dynamic_batch_sampler_valid"]

        train_batch_sampler = DynamicBatchSampler(
            train_data,
            length_func=lambda x: x["duration"],
            **dynamic_hparams_train,
        )
        valid_batch_sampler = DynamicBatchSampler(
            valid_data,
            length_func=lambda x: x["duration"],
            **dynamic_hparams_valid,
        )

    return (
        train_data,
        valid_data,
        test_data,
        train_batch_sampler,
        valid_batch_sampler,
    )


if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Use torchaudio if the device is CPU
    if run_opts.get("device") == "cpu":
        if "use_torchaudio: True" in overrides:
            overrides.replace("use_torchaudio: True", "use_torchaudio: False")
        else:
            overrides += "\nuse_torchaudio: True"

    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Dataset prep (parsing Librispeech)
    from gigaspeech_prepare import prepare_gigaspeech  # noqa

    # multi-gpu (ddp) save data preparation
    run_on_main(
        prepare_gigaspeech,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["save_folder"],
            "splits": hparams["splits"],
            "output_train": hparams["train_csv"],
            "output_dev": hparams["valid_csv"],
            "output_test": hparams["test_csv"],
            "json_file": hparams["json_file"],
            "skip_prep": hparams["skip_prep"],
            "convert_opus_to_wav": hparams["convert_opus_to_wav"],
            "download_with_HF": hparams["download_with_HF"],
            "punctuation": hparams["keep_punctuation"],
            "filler": hparams["keep_filler_words"],
        },
    )

    if hparams["data_prep_only"]:
        logger.info(
            "Data preparation finished. Restart the script with data_prep_only to False. "
        )
        import sys

        sys.exit()

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
        train_bsampler,
        valid_bsampler,
    ) = dataio_prepare(hparams)

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

    # We dynamically add the tokenizer to our brain class.
    # NB: This tokenizer corresponds to the one used for the LM!!
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
    os.makedirs(hparams["output_wer_folder"], exist_ok=True)

    # report WER on valid data
    asr_brain.hparams.test_wer_file = os.path.join(
        hparams["output_wer_folder"], "valid_wer.txt"
    )
    asr_brain.evaluate(
        valid_data,
        min_key="WER",
        test_loader_kwargs=hparams["test_dataloader_opts"],
    )

    # report WER on test data
    asr_brain.hparams.test_wer_file = os.path.join(
        hparams["output_wer_folder"], "test_wer.txt"
    )
    asr_brain.evaluate(
        test_data,
        min_key="WER",
        test_loader_kwargs=hparams["test_dataloader_opts"],
    )
