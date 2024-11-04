#!/usr/bin/env python3
"""Recipe for training a phoneme recognizer on TIMIT.
The system relies on an encoder, a decoder, and attention mechanisms between them.
Training is done with NLL. CTC loss is also added on the top of the encoder.
Greedy search is using for validation, while beamsearch is used at test time to
improve the system performance.

To run this recipe, do the following:
> python train.py hparams/train.yaml --data_folder /path/to/TIMIT

Authors
 * Mirco Ravanelli 2020
 * Ju-Chieh Chou 2020
 * Abdel Heba 2020
"""

import os
import sys

import torch
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.utils.distributed import if_main_process, run_on_main
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


# Define training procedure
class ASR(sb.Brain):
    def compute_forward(self, batch, stage):
        "Given an input batch it computes the phoneme probabilities."
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        phns_bos, _ = batch.phn_encoded_bos

        # Add waveform augmentation if specified.
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            wavs, wav_lens = self.hparams.wav_augment(wavs, wav_lens)
            phns_bos = self.hparams.wav_augment.replicate_labels(phns_bos)

        feats = self.modules.wav2vec2(wavs, wav_lens)
        x = self.modules.enc(feats)

        # output layer for ctc log-probabilities
        logits = self.modules.ctc_lin(x)
        p_ctc = self.hparams.log_softmax(logits)

        e_in = self.modules.emb(phns_bos)
        h, _ = self.modules.dec(e_in, x, wav_lens)

        # output layer for seq2seq log-probabilities
        logits = self.modules.seq_lin(h)
        p_seq = self.hparams.log_softmax(logits)

        hyps = None
        if stage == sb.Stage.VALID:
            hyps, _, _, _ = self.hparams.valid_searcher(x, wav_lens)
        elif stage == sb.Stage.TEST:
            hyps, _, _, _ = self.hparams.test_searcher(x, wav_lens)

        return p_ctc, p_seq, wav_lens, hyps

    def compute_objectives(self, predictions, batch, stage):
        "Given the network predictions and targets computed the NLL loss."
        p_ctc, p_seq, wav_lens, hyps = predictions

        ids = batch.id
        phns_eos, phn_lens_eos = batch.phn_encoded_eos
        phns, phn_lens = batch.phn_encoded

        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            phns = self.hparams.wav_augment.replicate_labels(phns)
            phn_lens = self.hparams.wav_augment.replicate_labels(phn_lens)
            phns_eos = self.hparams.wav_augment.replicate_labels(phns_eos)
            phn_lens_eos = self.hparams.wav_augment.replicate_labels(
                phn_lens_eos
            )

        loss_ctc = self.hparams.ctc_cost(p_ctc, phns, wav_lens, phn_lens)
        loss_seq = self.hparams.seq_cost(p_seq, phns_eos, phn_lens_eos)
        loss = self.hparams.ctc_weight * loss_ctc
        loss += (1 - self.hparams.ctc_weight) * loss_seq

        # Record losses for posterity
        if stage != sb.Stage.TRAIN:
            self.ctc_metrics.append(ids, p_ctc, phns, wav_lens, phn_lens)
            self.seq_metrics.append(ids, p_seq, phns_eos, phn_lens_eos)
            self.per_metrics.append(
                ids, hyps, phns, None, phn_lens, self.label_encoder.decode_ndim
            )

        return loss

    def on_stage_start(self, stage, epoch):
        "Gets called when a stage (either training, validation, test) starts."
        self.ctc_metrics = self.hparams.ctc_stats()
        self.seq_metrics = self.hparams.seq_stats()

        if stage != sb.Stage.TRAIN:
            self.per_metrics = self.hparams.per_stats()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        else:
            per = self.per_metrics.summarize("error_rate")

        if stage == sb.Stage.VALID:
            old_lr_adam, new_lr_adam = self.hparams.lr_annealing_adam(per)
            old_lr_wav2vec, new_lr_wav2vec = self.hparams.lr_annealing_wav2vec(
                per
            )
            sb.nnet.schedulers.update_learning_rate(
                self.adam_optimizer, new_lr_adam
            )
            sb.nnet.schedulers.update_learning_rate(
                self.wav2vec_optimizer, new_lr_wav2vec
            )

            self.hparams.train_logger.log_stats(
                stats_meta={
                    "epoch": epoch,
                    "lr_adam": old_lr_adam,
                    "lr_wav2vec": old_lr_wav2vec,
                },
                train_stats={"loss": self.train_loss},
                valid_stats={
                    "loss": stage_loss,
                    "ctc_loss": self.ctc_metrics.summarize("average"),
                    "seq_loss": self.seq_metrics.summarize("average"),
                    "PER": per,
                },
            )
            self.checkpointer.save_and_keep_only(
                meta={"PER": per}, min_keys=["PER"]
            )

        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats={"loss": stage_loss, "PER": per},
            )
            if if_main_process():
                with open(
                    self.hparams.test_wer_file, "w", encoding="utf-8"
                ) as w:
                    w.write("CTC loss stats:\n")
                    self.ctc_metrics.write_stats(w)
                    w.write("\nseq2seq loss stats:\n")
                    self.seq_metrics.write_stats(w)
                    w.write("\nPER stats:\n")
                    self.per_metrics.write_stats(w)
                    print(
                        "CTC, seq2seq, and PER stats written to file",
                        self.hparams.test_wer_file,
                    )

    def init_optimizers(self):
        "Initializes the wav2vec2 optimizer and model optimizer"
        self.wav2vec_optimizer = self.hparams.wav2vec_opt_class(
            self.modules.wav2vec2.parameters()
        )
        self.adam_optimizer = self.hparams.adam_opt_class(
            self.hparams.model.parameters()
        )

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable(
                "wav2vec_opt", self.wav2vec_optimizer
            )
            self.checkpointer.add_recoverable("adam_opt", self.adam_optimizer)

        self.optimizers_dict = {
            "wav2vec_opt": self.wav2vec_optimizer,
            "adam_opt": self.adam_optimizer,
        }


def dataio_prep(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    """
    data_folder = hparams["data_folder"]
    # 1. Declarations:
    train_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["train_annotation"],
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

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["valid_annotation"],
        replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    test_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["test_annotation"],
        replacements={"data_root": data_folder},
    )
    test_data = test_data.filtered_sorted(sort_key="duration")

    datasets = [train_data, valid_data, test_data]
    label_encoder = sb.dataio.encoder.CTCTextEncoder()

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("phn")
    @sb.utils.data_pipeline.provides(
        "phn_list",
        "phn_encoded_list",
        "phn_encoded",
        "phn_encoded_eos",
        "phn_encoded_bos",
    )
    def text_pipeline(phn):
        phn_list = phn.strip().split()
        yield phn_list
        phn_encoded_list = label_encoder.encode_sequence(phn_list)
        yield phn_encoded_list
        phn_encoded = torch.LongTensor(phn_encoded_list)
        yield phn_encoded
        phn_encoded_eos = torch.LongTensor(
            label_encoder.append_eos_index(phn_encoded_list)
        )
        yield phn_encoded_eos
        phn_encoded_bos = torch.LongTensor(
            label_encoder.prepend_bos_index(phn_encoded_list)
        )
        yield phn_encoded_bos

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 3. Fit encoder:
    # Load or compute the label encoder
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    special_labels = {
        "bos_label": hparams["bos_index"],
        "eos_label": hparams["eos_index"],
        "blank_label": hparams["blank_index"],
    }
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[train_data],
        output_key="phn_list",
        special_labels=special_labels,
        sequence_input=True,
    )

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "sig", "phn_encoded", "phn_encoded_eos", "phn_encoded_bos"],
    )

    return train_data, valid_data, test_data, label_encoder


if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Load hyperparameters file with command-line overrides
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Dataset prep (parsing TIMIT and annotation into csv files)
    from timit_prepare import prepare_timit  # noqa

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # multi-gpu (ddp) save data preparation
    run_on_main(
        prepare_timit,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_json_train": hparams["train_annotation"],
            "save_json_valid": hparams["valid_annotation"],
            "save_json_test": hparams["test_annotation"],
            "skip_prep": hparams["skip_prep"],
            "uppercase": hparams["uppercase"],
        },
    )

    # Dataset IO prep: creating Dataset objects and proper encodings for phones
    train_data, valid_data, test_data, label_encoder = dataio_prep(hparams)

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    asr_brain.label_encoder = label_encoder

    # Training/validation loop
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    # Test
    asr_brain.evaluate(
        test_data,
        min_key="PER",
        test_loader_kwargs=hparams["test_dataloader_opts"],
    )
