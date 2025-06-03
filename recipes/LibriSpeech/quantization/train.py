#!/usr/bin/env/python

"""Recipe for training a K-means quantizer on features from an SSL model.

To run this recipe:
> python train.py hparams/train_discrete_ssl.yaml

Authors
 * Luca Della Libera 2024
"""

# Adapted from:
# https://github.com/speechbrain/speechbrain/blob/v1.0.2/recipes/LJSpeech/quantization/train.py

import sys

import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.utils.distributed import if_main_process


class Quantization(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward pass."""
        batch = batch.to(self.device)
        sig, lens = batch.sig  # [B, T]

        # Extract features
        with torch.no_grad():
            self.modules.ssl_model.eval()
            feats = self.modules.ssl_model(sig, lens)  # [K, B, N, H]
            feats = feats[self.hparams.layer_id]  # [B, N, H]

        return feats

    def compute_objectives(self, predictions, batch, stage):
        """Computes the objectives."""
        feats = predictions  # [B, N, H]

        if stage != sb.Stage.TRAIN:
            # For K-means the validation/test loss is the inertia
            # The lower the inertia, the better should be the clustering
            # It is useful to monitor progress across epochs
            # However, when saving checkpoints we always keep the last one (i.e. max_keys=["epoch"])
            # to keep backward compatibility
            loss = self.hparams.quantizer.inertia(feats)
            return loss

        # If training, accumulate features (batch size used for K-means training
        # should be much larger than batch size used for feature extraction)
        feats = feats.flatten(end_dim=-2)  # [BN, H]
        self.curr_feats.append(feats)
        self.curr_batch_size += len(feats)
        if self.curr_batch_size < self.hparams.kmeans_batch_size:
            # If not enough features, leave average loss unchanged and go to next batch
            # avg_loss is computed as: (avg_loss - avg_loss / self.step) + float(loss) / self.step
            # If we set loss = avg_loss, avg_loss stays unchanged
            loss = torch.tensor(self.avg_train_loss)
            # Keep compatibility with standard supervised training
            # (SpeechBrain expects a tensor with gradient)
            loss.requires_grad_()
            return loss
        self.curr_feats = torch.cat(self.curr_feats)
        feats = self.curr_feats[: self.hparams.kmeans_batch_size]

        # Keep remaining features for next iteration
        self.curr_feats = [self.curr_feats[self.hparams.kmeans_batch_size :]]
        self.curr_batch_size = len(self.curr_feats[0])

        # Retrieve current centroids
        old_cluster_centers = self.hparams.quantizer.cluster_centers

        # Partial fit on current batch
        self.hparams.quantizer.partial_fit(feats)

        # For K-means the training loss is the drift between current centroids and old centroids
        # If close to 0, it means that the training has converged
        curr_cluster_centers = self.hparams.quantizer.cluster_centers
        loss = (curr_cluster_centers - old_cluster_centers).norm()

        # Keep compatibility with standard supervised training
        # (SpeechBrain expects a tensor with gradient)
        loss.requires_grad_()
        self.optimizer_step += 1
        assert self.optimizer_step == self.modules.quantizer.n_steps, (
            f"optimizer_step: {self.optimizer_step}",
            f"quantizer.n_steps: {self.modules.quantizer.n_steps}",
        )

        return loss

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch."""
        if stage == sb.Stage.TRAIN:
            # NOTE: not included in intra-epoch checkpoints
            self.curr_feats = []
            self.curr_batch_size = 0

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of each epoch."""
        # Compute/store important stats
        current_epoch = self.hparams.epoch_counter.current
        stage_stats = {"loss": stage_loss}

        if stage == sb.Stage.TRAIN:
            self.avg_train_loss = 0.0
            self.train_stats = stage_stats
            self.stats_meta = {"epoch": epoch, "steps": self.optimizer_step}
            if if_main_process():
                self.checkpointer.save_and_keep_only(
                    meta={"loss": stage_stats["loss"], "epoch": epoch},
                    max_keys=["epoch"],
                    num_to_keep=self.hparams.keep_checkpoints,
                )
            self.hparams.train_logger.log_stats(
                stats_meta=self.stats_meta,
                train_stats=self.train_stats,
            )

        # Perform end-of-iteration operations, like annealing, logging, etc.
        elif stage == sb.Stage.VALID:
            self.hparams.train_logger.log_stats(
                stats_meta=self.stats_meta,
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": current_epoch},
                test_stats=stage_stats,
            )


def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.

    """
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"],
        replacements={"DATA_ROOT": hparams["data_folder"]},
    )
    # Sort training data to speed up training
    train_data = train_data.filtered_sorted(
        sort_key="duration",
        reverse=hparams["sorting"] == "descending",
        key_max_value={"duration": hparams["train_remove_if_longer"]},
    )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"],
        replacements={"DATA_ROOT": hparams["data_folder"]},
    )
    # Sort validation data to speed up validation
    valid_data = valid_data.filtered_sorted(
        sort_key="duration",
        reverse=True,
        key_max_value={"duration": hparams["valid_remove_if_longer"]},
    )

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_csv"],
        replacements={"DATA_ROOT": hparams["data_folder"]},
    )
    # Sort the test data to speed up testing
    test_data = test_data.filtered_sorted(
        sort_key="duration",
        reverse=True,
        key_max_value={"duration": hparams["test_remove_if_longer"]},
    )

    datasets = [train_data, valid_data, test_data]

    # Define audio pipeline
    takes = ["wav"]
    provides = ["sig"]

    def audio_pipeline(wav):
        original_sample_rate = sb.dataio.dataio.read_audio_info(wav).sample_rate
        sig = sb.dataio.dataio.read_audio(wav)
        sig = torchaudio.functional.resample(
            sig, original_sample_rate, hparams["sample_rate"]
        )
        yield sig

    sb.dataio.dataset.add_dynamic_item(
        datasets, audio_pipeline, takes, provides
    )

    # Set output
    sb.dataio.dataset.set_output_keys(datasets, ["id"] + provides)

    return datasets


if __name__ == "__main__":
    # Command-line interface
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # If --distributed_launch then create ddp_init_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Prepare data
    from librispeech_prepare import prepare_librispeech

    kwargs = {
        "data_folder": hparams["data_folder"],
        "tr_splits": hparams["train_splits"],
        "dev_splits": hparams["dev_splits"],
        "te_splits": hparams["test_splits"],
        "save_folder": hparams["output_folder"],
        "merge_lst": hparams["train_splits"],
        "merge_name": "train.csv",
        "skip_prep": hparams["skip_prep"],
    }
    prepare_librispeech(**kwargs)

    # Create the datasets objects
    train_data, valid_data, test_data = dataio_prepare(hparams)

    # Trainer initialization
    brain = Quantization(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Train
    brain.fit(
        brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=dict(
            num_workers=hparams["dataloader_workers"],
            batch_size=hparams["train_batch_size"],
            shuffle=hparams["sorting"] == "random",
            pin_memory=run_opts.get("device", "cpu") != "cpu",
        ),
        valid_loader_kwargs=dict(
            num_workers=hparams["dataloader_workers"],
            batch_size=hparams["valid_batch_size"],
            pin_memory=run_opts.get("device", "cpu") != "cpu",
        ),
    )

    # Test
    brain.evaluate(
        test_data,
        max_key="epoch",
        test_loader_kwargs=dict(
            num_workers=hparams["dataloader_workers"],
            batch_size=hparams["test_batch_size"],
            pin_memory=run_opts.get("device", "cpu") != "cpu",
        ),
    )
