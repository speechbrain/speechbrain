#!/usr/bin/env/python3
"""
Recipe for "multistage" (speech -> ASR -> text -> NLU -> semantics) SLU.

We transcribe each minibatch using a model trained on LibriSpeech,
then feed the transcriptions into a seq2seq model to map them to semantics.

(The transcriptions could be done offline to make training faster;
the benefit of doing it online is that we can use augmentation
and sample many possible transcriptions.)

(Adapted from the LibriSpeech seq2seq ASR recipe written by Ju-Chieh Chou, Mirco Ravanelli, Abdel Heba, and Peter Pl$

Run using:
> python train.py hparams/train.yaml

Authors
 * Loren Lugosch, Mirco Ravanelli 2020
"""

import sys

import torch
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.utils.distributed import if_main_process, run_on_main


# Define training procedure
class SLU(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        tokens_bos, tokens_bos_lens = batch.tokens_bos

        # Add waveform augmentation if specified.
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            wavs, wav_lens = self.hparams.wav_augment(wavs, wav_lens)
            tokens_bos = self.hparams.wav_augment.replicate_labels(tokens_bos)
            tokens_bos_lens = self.hparams.wav_augment.replicate_labels(
                tokens_bos_lens
            )

        # ASR forward pass
        words, asr_tokens = self.hparams.asr_model.transcribe_batch(
            wavs.detach(), wav_lens
        )

        # Pad examples to have same length.
        asr_tokens_lens = torch.tensor([max(len(t), 1) for t in asr_tokens])
        max_length = asr_tokens_lens.max().item()
        for t in asr_tokens:
            t += [0] * (max_length - len(t))
        asr_tokens = torch.tensor([t for t in asr_tokens])

        asr_tokens_lens = asr_tokens_lens.float()
        asr_tokens_lens = asr_tokens_lens / asr_tokens_lens.max()

        asr_tokens, asr_tokens_lens = (
            asr_tokens.to(self.device),
            asr_tokens_lens.to(self.device),
        )
        embedded_transcripts = self.hparams.input_emb(asr_tokens)

        # SLU forward pass
        encoder_out = self.hparams.slu_enc(embedded_transcripts)
        e_in = self.hparams.output_emb(tokens_bos)
        h, _ = self.hparams.dec(e_in, encoder_out, asr_tokens_lens)

        # Output layer for seq2seq log-probabilities
        logits = self.hparams.seq_lin(h)
        p_seq = self.hparams.log_softmax(logits)

        # Compute outputs
        p_tokens = None
        if stage != sb.Stage.TRAIN:
            p_tokens, _, _, _ = self.hparams.beam_searcher(
                encoder_out, asr_tokens_lens
            )

        return p_seq, asr_tokens_lens, p_tokens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (NLL) given predictions and targets."""

        p_seq, asr_tokens_lens, predicted_tokens = predictions

        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos

        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            tokens_eos = self.hparams.wav_augment.replicate_labels(tokens_eos)
            tokens_eos_lens = self.hparams.wav_augment.replicate_labels(
                tokens_eos_lens
            )

        loss_seq = self.hparams.seq_cost(
            p_seq, tokens_eos, length=tokens_eos_lens
        )

        # (No ctc loss)
        loss = loss_seq

        if (stage != sb.Stage.TRAIN) or (self.step % show_results_every == 0):
            # Decode token terms to words
            predicted_semantics = [
                tokenizer.decode_ids(utt_seq).split(" ")
                for utt_seq in predicted_tokens
            ]

            target_semantics = [wrd.split(" ") for wrd in batch.semantics]

            for i in range(len(target_semantics)):
                print(" ".join(predicted_semantics[i]))
                print(" ".join(target_semantics[i]))
                print("")

            if stage != sb.Stage.TRAIN:
                self.wer_metric.append(
                    ids, predicted_semantics, target_semantics
                )
                self.cer_metric.append(
                    ids, predicted_semantics, target_semantics
                )

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
            stage_stats["SER"] = self.wer_metric.summarize("SER")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(stage_stats["SER"])
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"SER": stage_stats["SER"]},
                min_keys=["SER"],
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


def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    """

    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["csv_train"],
        replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    # If we are testing on all the real data, including dev-real,
    # we shouldn't use dev-real as the validation set.
    if hparams["test_on_all_real"]:
        valid_path = hparams["csv_dev_synth"]
    else:
        valid_path = hparams["csv_dev_real"]
    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=valid_path,
        replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    test_real_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["csv_test_real"],
        replacements={"data_root": data_folder},
    )
    test_real_data = test_real_data.filtered_sorted(sort_key="duration")

    test_synth_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["csv_test_synth"],
        replacements={"data_root": data_folder},
    )
    test_synth_data = test_synth_data.filtered_sorted(sort_key="duration")

    all_real_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["csv_all_real"],
        replacements={"data_root": data_folder},
    )
    all_real_data = all_real_data.filtered_sorted(sort_key="duration")

    datasets = [
        train_data,
        valid_data,
        test_real_data,
        test_synth_data,
        all_real_data,
    ]

    tokenizer = hparams["tokenizer"]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("semantics")
    @sb.utils.data_pipeline.provides(
        "semantics", "token_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(semantics):
        yield semantics
        tokens_list = tokenizer.encode_as_ids(semantics)
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
        ["id", "sig", "semantics", "tokens_bos", "tokens_eos", "tokens"],
    )
    return (
        train_data,
        valid_data,
        test_real_data,
        test_synth_data,
        all_real_data,
        tokenizer,
    )


if __name__ == "__main__":
    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    show_results_every = 100  # plots results every N iterations

    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Dataset prep (parsing TAS)
    from prepare import prepare_TAS  # noqa

    # multi-gpu (ddp) save data preparation
    run_on_main(
        prepare_TAS,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["output_folder"],
            "train_splits": hparams["train_splits"],
            "type": "multistage",
            "skip_prep": hparams["skip_prep"],
        },
    )
    run_on_main(hparams["prepare_noise_data"])

    # here we create the datasets objects as well as tokenization and encoding
    (
        train_set,
        valid_set,
        test_real_set,
        test_synth_set,
        all_real_set,
        tokenizer,
    ) = dataio_prepare(hparams)

    # We download and pretrain the tokenizer
    hparams["pretrainer"].collect_files()
    hparams["pretrainer"].load_collected()

    # Brain class initialization
    slu_brain = SLU(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # adding objects to trainer:
    slu_brain.tokenizer = tokenizer

    # Training
    slu_brain.fit(
        slu_brain.hparams.epoch_counter,
        train_set,
        valid_set,
        train_loader_kwargs=hparams["dataloader_opts"],
        valid_loader_kwargs=hparams["dataloader_opts"],
    )

    # Test (ALL real data)
    if slu_brain.hparams.test_on_all_real:
        slu_brain.hparams.test_wer_file = hparams["all_real_wer_file"]
        slu_brain.evaluate(
            all_real_set,
            test_loader_kwargs=hparams["dataloader_opts"],
            min_key="SER",
        )

    # Test (real data)
    slu_brain.hparams.test_wer_file = hparams["test_real_wer_file"]
    slu_brain.evaluate(
        test_real_set,
        test_loader_kwargs=hparams["dataloader_opts"],
        min_key="SER",
    )

    # Test (synth data)
    slu_brain.hparams.test_wer_file = hparams["test_synth_wer_file"]
    slu_brain.evaluate(
        test_synth_set,
        test_loader_kwargs=hparams["dataloader_opts"],
        min_key="SER",
    )
