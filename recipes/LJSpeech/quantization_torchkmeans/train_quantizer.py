#!/usr/bin/env/python

"""Recipe for training a quantizer to extract semantic discrete audio representations.

Currently, it supports any quantizer that implements the expected interface.

To run this recipe:
> python train_quantizer.py hparams/quantizer/<config>.yaml

Authors
 * Luca Della Libera 2024
"""

import sys

import speechbrain as sb
import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml
from speechbrain.dataio.sampler import DynamicBatchSampler
from speechbrain.utils.distributed import if_main_process, run_on_main


class Quantization(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward pass."""
        batch = batch.to(self.device)
        sig, lens = batch.sig

        # Augment if specified
        if stage == sb.Stage.TRAIN and self.hparams.augment:
            sig, lens = self.hparams.augmentation(sig, lens)

        # Extract features
        with torch.no_grad():
            self.modules.codec.encoder.eval()
            feats = self.modules.codec.encode(sig, lens)

        return feats

    def compute_objectives(self, predictions, batch, stage):
        """Computes the objectives."""
        feats = predictions

        if stage == sb.Stage.TRAIN:
            # Unsupervised training loss + step
            # For example, for K-means the training loss is the drift between current centroids and old centroids
            # If close to 0, it means that the training has converged
            loss = self.modules.codec.quantizer.step(feats)
            # Workaround to keep consistency with standard supervised training
            loss.requires_grad_()
        else:
            # Unsupervised validation/test loss
            # For example, for K-means the validation/test loss is the inertia
            # The lower the inertia, the better the clustering
            loss = self.modules.codec.quantizer.evaluate(feats).mean()

        return loss

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of each epoch."""
        # Compute/store important stats
        current_epoch = self.hparams.epoch_counter.current
        stage_stats = {"loss": stage_loss}

        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats

        # Perform end-of-iteration operations, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            if if_main_process():
                self.checkpointer.save_and_keep_only(
                    meta={"loss": stage_stats["loss"]},
                    min_keys=["loss"],
                    num_to_keep=self.hparams.keep_checkpoints,
                )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": current_epoch},
                test_stats=stage_stats,
            )


def dataio_prepare(
    data_folder,
    train_json,
    valid_json,
    test_json,
    sample_rate=16000,
    train_remove_if_longer=60.0,
    valid_remove_if_longer=60.0,
    test_remove_if_longer=60.0,
    sorting="ascending",
    debug=False,
    **_,
):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    """
    train_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=train_json, replacements={"DATA_ROOT": data_folder},
    )
    # Sort training data to speed up training
    train_data = train_data.filtered_sorted(
        sort_key="duration",
        reverse=sorting == "descending",
        key_max_value={"duration": train_remove_if_longer},
    )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=valid_json, replacements={"DATA_ROOT": data_folder},
    )
    # Sort validation data to speed up validation
    valid_data = valid_data.filtered_sorted(
        sort_key="duration",
        reverse=not debug,
        key_max_value={"duration": valid_remove_if_longer},
    )

    test_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=test_json, replacements={"DATA_ROOT": data_folder},
    )
    # Sort the test data to speed up testing
    test_data = test_data.filtered_sorted(
        sort_key="duration",
        reverse=not debug,
        key_max_value={"duration": test_remove_if_longer},
    )

    datasets = [train_data, valid_data, test_data]

    # Define audio pipeline
    takes = ["wav"]
    provides = ["sig"]

    def audio_pipeline(wav):
        original_sample_rate = sb.dataio.dataio.read_audio_info(wav).sample_rate
        sig = sb.dataio.dataio.read_audio(wav)

        sig = torchaudio.functional.resample(
            sig, original_sample_rate, sample_rate,
        )
        yield sig

    sb.dataio.dataset.add_dynamic_item(
        [train_data, valid_data, test_data], audio_pipeline, takes, provides
    )

    # Set output
    sb.dataio.dataset.set_output_keys(datasets, ["id"] + provides)

    return train_data, valid_data, test_data


if __name__ == "__main__":
    # Command-line interface
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # If --distributed_launch then create ddp_init_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Dataset preparation
    from ljspeech_prepare import prepare_ljspeech as prepare_data

    # Due to DDP, do the preparation ONLY on the main Python process
    run_on_main(
        prepare_data,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["save_folder"],
            "splits": hparams["splits"],
        },
    )

    # Create the datasets objects and tokenization
    train_data, valid_data, test_data = dataio_prepare(
        debug=run_opts.get("debug", False), **hparams
    )

    # Trainer initialization
    brain = Quantization(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Dynamic batching
    hparams["train_dataloader_kwargs"] = {
        "num_workers": hparams["dataloader_workers"]
    }
    if hparams["dynamic_batching"]:
        hparams["train_dataloader_kwargs"][
            "batch_sampler"
        ] = DynamicBatchSampler(
            train_data,
            hparams["train_max_batch_length"],
            num_buckets=hparams["num_buckets"],
            length_func=lambda x: x["duration"],
            shuffle=False,
            batch_ordering=hparams["sorting"],
            max_batch_ex=hparams["max_batch_size"],
        )
    else:
        hparams["train_dataloader_kwargs"]["batch_size"] = hparams[
            "train_batch_size"
        ]
        hparams["train_dataloader_kwargs"]["shuffle"] = (
            hparams["sorting"] == "random"
        )

    hparams["valid_dataloader_kwargs"] = {
        "num_workers": hparams["dataloader_workers"]
    }
    if hparams["dynamic_batching"]:
        hparams["valid_dataloader_kwargs"][
            "batch_sampler"
        ] = DynamicBatchSampler(
            valid_data,
            hparams["valid_max_batch_length"],
            num_buckets=hparams["num_buckets"],
            length_func=lambda x: x["duration"],
            shuffle=False,
            batch_ordering="descending",
            max_batch_ex=hparams["max_batch_size"],
        )
    else:
        hparams["valid_dataloader_kwargs"]["batch_size"] = hparams[
            "valid_batch_size"
        ]

    hparams["test_dataloader_kwargs"] = {
        "num_workers": hparams["dataloader_workers"]
    }
    if hparams["dynamic_batching"]:
        hparams["test_dataloader_kwargs"][
            "batch_sampler"
        ] = DynamicBatchSampler(
            test_data,
            hparams["test_max_batch_length"],
            num_buckets=hparams["num_buckets"],
            length_func=lambda x: x["duration"],
            shuffle=False,
            batch_ordering="descending",
            max_batch_ex=hparams["max_batch_size"],
        )
    else:
        hparams["test_dataloader_kwargs"]["batch_size"] = hparams[
            "test_batch_size"
        ]

    # Train
    brain.fit(
        brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_dataloader_kwargs"],
        valid_loader_kwargs=hparams["valid_dataloader_kwargs"],
    )

    # Test
    brain.evaluate(
        test_data,
        min_key="loss",
        test_loader_kwargs=hparams["test_dataloader_kwargs"],
    )
