#!/usr/bin/env python3
"""Recipe for training a HMM-DNN alignment system on the TIMIT dataset.
The system is trained can be trained with Viterbi, forward, or CTC loss.

To run this recipe, do the following:
> python train.py hparams/train.yaml --data_folder /path/to/TIMIT

Authors
 * Elena Rastorgueva 2020
 * Mirco Ravanelli 2020
 * Peter Plantinga 2020
"""
import os
import sys
import torch
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main

# Define training procedure


class AlignBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        """Computations from the waveform to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig

        # Adding augmentation when specified:
        if stage == sb.Stage.TRAIN:
            if hasattr(self.modules, "env_corrupt"):
                wavs_noise = self.modules.env_corrupt(wavs, wav_lens)
                wavs = torch.cat([wavs, wavs_noise], dim=0)
                wav_lens = torch.cat([wav_lens, wav_lens])
            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs, wav_lens)

        feats = self.hparams.compute_features(wavs)
        if hasattr(self.hparams, "normalize"):
            feats = self.modules.normalize(feats, wav_lens)
        out = self.modules.model(feats)
        out = self.modules.output(out)
        out = out - out.mean(1).unsqueeze(1)
        pout = self.hparams.log_softmax(out)
        return pout, wav_lens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss with the specified alignment algorithm"""
        pout, pout_lens = predictions
        ids = batch.id
        phns, phn_lens = batch.phn_encoded
        phn_ends, _ = batch.phn_ends

        if stage == sb.Stage.TRAIN and hasattr(self.modules, "env_corrupt"):
            phns = torch.cat([phns, phns], dim=0)
            phn_lens = torch.cat([phn_lens, phn_lens], dim=0)

        phns, phn_lens = phns.to(self.device), phn_lens.to(self.device)
        phns_orig = sb.utils.data_utils.undo_padding(phns, phn_lens)
        phns = self.hparams.aligner.expand_phns_by_states_per_phoneme(
            phns, phn_lens
        )

        phns = phns.int()

        if self.training_type == "forward":
            forward_scores = self.hparams.aligner(
                pout, pout_lens, phns, phn_lens, "forward"
            )
            loss = -forward_scores

        elif self.training_type == "ctc":
            loss = self.hparams.compute_cost_ctc(
                pout, phns, pout_lens, phn_lens
            )
        elif self.training_type == "viterbi":
            prev_alignments = self.hparams.aligner.get_prev_alignments(
                ids, pout, pout_lens, phns, phn_lens
            )
            prev_alignments = prev_alignments.to(self.hparams.device)
            loss = self.hparams.compute_cost_nll(pout, prev_alignments)

        viterbi_scores, alignments = self.hparams.aligner(
            pout, pout_lens, phns, phn_lens, "viterbi"
        )

        if self.training_type in ["viterbi", "forward"]:
            self.hparams.aligner.store_alignments(ids, alignments)

        if stage != sb.Stage.TRAIN:
            self.accuracy_metrics.append(ids, alignments, phn_ends, phns_orig)

        return loss

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        self.training_type = self.hparams.init_training_type
        if stage != sb.Stage.TRAIN:
            self.accuracy_metrics = self.hparams.accuracy_stats()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        if hasattr(self.hparams, "switch_training_type"):
            if not hasattr(self.hparams, "switch_training_epoch"):
                raise ValueError(
                    "Please specify `switch_training_epoch` in `params`"
                )
            if (
                self.hparams.epoch_counter.current
                == self.hparams.switch_training_epoch
            ) and stage == sb.Stage.VALID:
                self.training_type = self.hparams.switch_training_type
                print("Switching to training type", self.training_type)

        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        else:
            acc = self.accuracy_metrics.summarize("average")

        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(acc)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats={"loss": self.train_loss},
                valid_stats={"loss": stage_loss, "accuracy": acc},
            )
            self.checkpointer.save_and_keep_only(
                meta={"accuracy": acc}, max_keys=["accuracy"],
            )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats={"loss": stage_loss, "accuracy": acc},
            )


def dataio_prep(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""

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
        hparams["dataloader_options"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["dataloader_options"]["shuffle"] = False

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

    test_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["test_annotation"],
        replacements={"data_root": data_folder},
    )

    datasets = [train_data, valid_data, test_data]
    label_encoder = sb.dataio.encoder.TextEncoder()

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("phn")
    @sb.utils.data_pipeline.provides("phn_list", "phn_encoded")
    def text_pipeline(phn):
        phn_list = phn.strip().split()
        yield phn_list
        phn_encoded = label_encoder.encode_sequence_torch(phn_list)
        yield phn_encoded

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Define end sample pipeline
    # (end sample is used to retrieve the golden alignment )

    @sb.utils.data_pipeline.takes("ground_truth_phn_ends")
    @sb.utils.data_pipeline.provides("phn_ends")
    def phn_ends_pipeline(ground_truth_phn_ends):
        phn_ends = ground_truth_phn_ends.strip().split()
        phn_ends = [int(i) for i in phn_ends]
        phn_ends = torch.Tensor(phn_ends)
        return phn_ends

    sb.dataio.dataset.add_dynamic_item(datasets, phn_ends_pipeline)

    # 3. Fit encoder:
    # Load or compute the label encoder
    label_encoder_file = os.path.join(
        hparams["save_folder"], "label_encoder.txt"
    )
    if os.path.exists(label_encoder_file):
        label_encoder.load(label_encoder_file)
    else:
        label_encoder.update_from_didataset(train_data, output_key="phn_list")
        label_encoder.save(
            os.path.join(hparams["save_folder"], "label_encoder.txt")
        )

    # 6. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "sig", "phn_encoded", "phn_ends"]
    )

    return train_data, valid_data, test_data, label_encoder


# Begin Recipe!
if __name__ == "__main__":

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
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
            "phn_set": hparams["phn_set"],
            "skip_prep": hparams["skip_prep"],
        },
    )

    # Dataset IO prep: creating Dataset objects and proper encodings for phones
    train_data, valid_data, test_data, label_encoder = dataio_prep(hparams)

    # Trainer initialization
    align_brain = AlignBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    align_brain.label_encoder = label_encoder

    # Training/validation loop
    print("Starting training type:", hparams["init_training_type"])
    align_brain.fit(
        align_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )

    # Test
    align_brain.evaluate(
        test_data,
        max_key="accuracy",
        test_loader_kwargs=hparams["dataloader_options"],
    )
