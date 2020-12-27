#!/usr/bin/env python3
"""Recipe for doing ASR with phoneme targets and CTC loss on the TIMIT dataset

To run this recipe, do the following:
> python experiment.py hyperparams.yaml --data_folder /path/to/TIMIT

Authors
 * Mirco Ravanelli 2020
 * Peter Plantinga 2020
"""
import sys
import torch
import speechbrain as sb


# Define training procedure
class ASR_Brain(sb.Brain):
    def compute_forward(self, batch, stage):
        "Given an input batch it computes the phoneme probabilities."
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        # Adding optional augmentation when specified:
        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "env_corrupt"):
                wavs_noise = self.hparams.env_corrupt(wavs, wav_lens)
                wavs = torch.cat([wavs, wavs_noise], dim=0)
                wav_lens = torch.cat([wav_lens, wav_lens])
            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs, wav_lens)

        feats = self.hparams.compute_features(wavs)
        feats = self.modules.normalize(feats, wav_lens)
        out = self.modules.model(feats)
        out = self.modules.output(out)
        pout = self.hparams.log_softmax(out)

        return pout, wav_lens

    def compute_objectives(self, predictions, batch, stage):
        "Given the network predictions and targets computed the CTC loss."
        pout, pout_lens = predictions
        ids = batch.id
        phns, phn_lens = batch.phn_encoded

        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "env_corrupt"):
            phns = torch.cat([phns, phns], dim=0)
            phn_lens = torch.cat([phn_lens, phn_lens], dim=0)

        loss = self.hparams.compute_cost(pout, phns, pout_lens, phn_lens)
        self.ctc_metrics.append(ids, pout, phns, pout_lens, phn_lens)

        if stage != sb.Stage.TRAIN:
            sequence = sb.decoders.ctc_greedy_decode(
                pout, pout_lens, blank_id=self.hparams.blank_index
            )
            self.per_metrics.append(
                ids,
                sequence,
                phns,
                None,
                phn_lens,
                self.hparams.label_encoder.decode_ndim,
            )

        return loss

    def on_stage_start(self, stage, epoch):
        "Gets called when a stage (either training, validation, test) starts."
        self.ctc_metrics = self.hparams.ctc_stats()

        if stage != sb.Stage.TRAIN:
            self.per_metrics = self.hparams.per_stats()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a stage."""
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        else:
            per = self.per_metrics.summarize("error_rate")

        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(per)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats={"loss": self.train_loss},
                valid_stats={"loss": stage_loss, "PER": per},
            )
            self.checkpointer.save_and_keep_only(
                meta={"PER": per}, min_keys=["PER"],
            )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats={"loss": stage_loss, "PER": per},
            )
            with open(self.hparams.wer_file, "w") as w:
                w.write("CTC loss stats:\n")
                self.ctc_metrics.write_stats(w)
                w.write("\nPER stats:\n")
                self.per_metrics.write_stats(w)
                print("CTC and PER stats written to ", self.hparams.wer_file)


def data_io_prep(hparams):
    "Creates the datasets and their data processing pipelines."

    data_folder = hparams["data_folder"]

    # 1. Declarations:
    train_data = sb.data_io.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_annotation"],
        replacements={"data_root": data_folder},
    )

    if hparams["sorting"] is not None:
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["dataloader_options"]["train_shuffle"] = False

    valid_data = sb.data_io.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_annotation"],
        replacements={"data_root": data_folder},
    )

    test_data = sb.data_io.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_annotation"],
        replacements={"data_root": data_folder},
    )

    datasets = [train_data, valid_data, test_data]
    label_encoder = sb.data_io.encoder.CTCTextEncoder()

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.data_io.data_io.read_audio(wav)
        return sig

    sb.data_io.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("phn")
    @sb.utils.data_pipeline.provides("phn_list", "phn_encoded")
    def text_pipeline(phn):
        phn_list = phn.strip().split()
        yield phn_list
        phn_encoded = label_encoder.encode_sequence_torch(phn_list)
        yield phn_encoded

    sb.data_io.dataset.add_dynamic_item(datasets, text_pipeline)

    # 3. Fit encoder:
    label_encoder.insert_blank(hparams["blank_index"])
    label_encoder.update_from_didataset(train_data, output_key="phn_list")

    # 4. Set output:
    sb.data_io.dataset.set_output_keys(datasets, ["id", "sig", "phn_encoded"])

    return train_data, valid_data, test_data, label_encoder


# Begin Recipe!
if __name__ == "__main__":

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = sb.load_extended_yaml(fin, overrides)

    # Dataset prep (parsing TIMIT and annotation into csv files)
    from timit_prepare import prepare_timit  # noqa

    prepare_timit(
        hparams["data_folder"],
        splits=["train", "dev", "test"],
        save_folder=hparams["data_folder"],
    )

    # Dataset IO prep: creating Dataset objects and proper encodings for phones
    train_data, valid_data, test_data, label_encoder = data_io_prep(hparams)
    hparams["label_encoder"] = label_encoder

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Trainer initialization
    asr_brain = ASR_Brain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Training/validation loop
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        **hparams["dataloader_options"],
    )

    # Test
    asr_brain.evaluate(
        test_data, min_key="PER", **hparams["dataloader_options"]
    )
