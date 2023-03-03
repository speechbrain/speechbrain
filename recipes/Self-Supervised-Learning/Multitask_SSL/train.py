#!/usr/bin/env python3
import pandas as pd
import sys
from torch.nn.utils.rnn import pad_sequence
import torch
import logging
import speechbrain as sb
import torchaudio
from hyperpyyaml import load_hyperpyyaml
from speechbrain.lobes.features import MFCC
from speechbrain.processing.features import spectral_magnitude, Deltas

"""Recipe for training a multi-task self supervised learning model.
Representations are learned in a self-supervised way through learning to
solve a set of pretext tasks, i.e learning to predict a set of pretext labels
like spectrograms or signal features.
More details here: 
https://arxiv.org/abs/2001.09239
https://arxiv.org/abs/2104.07388

To run this recipe, do the following:
> python train.py hparams/train.yaml
With the default hyperparameters, the system employs a CRDNN encoder.

Authors :
    Salah Zaiem
    Titouan Parcollet

"""


logger = logging.getLogger(__name__)

# Define training procedure


class SSL(sb.core.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        self.considered_workers = self.hparams.workers
        # Forward pass
        feats = self.hparams.compute_features(wavs)
        feats = self.modules.normalize(feats, wav_lens)
        online_values = {"melfs": feats}
        for worker in hparams["online_workers"]:
            values_nonnorm = hparams["online_computation"][worker](wavs)
            online_values[worker] = hparams["online_normalization"][worker](
                values_nonnorm, wav_lens
            )
        feats = self.modules.normalize(feats, wav_lens)
        # Forward pass
        z = self.modules.enc(feats.detach())
        workers_predictions = dict()
        for worker in self.considered_workers:
            workers_predictions[worker] = self.workers_regressors[worker](z)
        for worker in self.considered_workers:
            if worker not in signal_workers + hparams["online_workers"]:
                # The mean is used mainly for workers that yield one value per
                # speech sample (e.g. speaker age, sample quality ... )
                workers_predictions[worker] = torch.mean(
                    workers_predictions[worker], dim=1
                )
            else:
                # For other classic workers : spectrograms and signal features
                # As the labels are present per frame, no need to take the mean
                workers_predictions[worker] = torch.squeeze(
                    workers_predictions[worker]
                )
        return workers_predictions, online_values

    def compute_objectives(self, predictions, batch, online_values, stage):
        # Load prediction
        for signal_worker in signal_workers:
            if signal_worker in self.considered_workers:
                self.workers_losses[
                    signal_worker
                ] = self.hparams.regression_loss
        exoworkers = signal_workers
        self.workers_target = online_values
        # exoworkers : workers whose labels come from the csv files, and not
        # from in-loop computation
        if len(exoworkers) > 0:

            workers_targets_values = batch.workers_targets
            batch_size = len(batch.workers_targets)
            for ind, worker in enumerate(exoworkers):
                worker_values = [
                    workers_targets_values[i][worker] for i in range(batch_size)
                ]
                worker_values = pad_sequence(worker_values)
                self.workers_target[worker] = torch.transpose(
                    torch.tensor(worker_values), 0, 1
                )
        workers_loss = dict()
        for worker in self.considered_workers:
            workers_loss[worker] = self.workers_losses[worker]()(
                predictions[worker], self.workers_target[worker].to(self.device)
            )
        self.workers_loss = workers_loss
        workers_weights = self.hparams.workers_weights
        # If you do not include the worker in the weigths, we just weight it 1
        for worker in workers_loss:
            if worker not in workers_weights:
                workers_weights[worker] = 1
        # Weighting the losses of the workers
        losses = [workers_loss[x] * workers_weights[x] for x in workers_loss]
        # Summing the Losses from every worker
        final_loss = losses[0]
        for i in range(1, len(losses)):
            final_loss += losses[i]
        return final_loss, workers_loss

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""
        predictions, online_values = self.compute_forward(batch, sb.Stage.TRAIN)
        loss, workers_loss = self.compute_objectives(
            predictions, batch, online_values, sb.Stage.TRAIN
        )
        loss.backward()
        if self.check_gradients(loss):
            self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.detach()

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        predictions, online_values = self.compute_forward(batch, stage=stage)
        with torch.no_grad():
            final_loss, losses = self.compute_objectives(
                predictions, batch, online_values, stage=stage
            )
        return final_loss.detach()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        self.train_stats = stage_stats

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(stage_stats["loss"])
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only()


# Define custom data procedure
def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    # 1. Define datasets
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"], replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(
            sort_key="duration",
            key_max_value={"duration": hparams["avoid_if_longer_than"]},
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["dataloader_options"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration",
            reverse=True,
            key_max_value={"duration": hparams["avoid_if_longer_than"]},
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["dataloader_options"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"], replacements={"data_root": data_folder},
    )
    # We also sort the validation data so it is faster to validate
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_csv"], replacements={"data_root": data_folder},
    )

    # We also sort the validation data so it is faster to validate
    test_data = valid_data.filtered_sorted(sort_key="duration")

    datasets = [train_data, valid_data, test_data]
    # 2. Define audio pipeline:

    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        info = torchaudio.info(wav)
        sig = sb.dataio.dataio.read_audio(wav)
        resampled = torchaudio.transforms.Resample(
            info.sample_rate, hparams["sample_rate"],
        )(sig)
        return resampled

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    needed_workers = signal_workers
    if len(needed_workers) > 0:
        # Defining the csv reading pipeline
        @sb.utils.data_pipeline.takes("csv_path")
        @sb.utils.data_pipeline.provides("workers_targets")
        def csv_pipeline(csv):
            feats = {}
            csv_tab = pd.read_pickle(csv)
            for worker in needed_workers:
                feats[worker] = torch.tensor(csv_tab[worker])
            return feats

        sb.dataio.dataset.add_dynamic_item(datasets, csv_pipeline)
        # 4. set output:
        sb.dataio.dataset.set_output_keys(
            datasets, ["id", "sig", "workers_targets"],
        )
    else:
        sb.dataio.dataset.set_output_keys(
            datasets, ["id", "sig"],
        )
    return train_data, valid_data, test_data


if __name__ == "__main__":

    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # Dataset preparation (parsing CommonVoice) uncomment if using Commonvoice
    # from common_voice_prepare import prepare_common_voice  # noqa

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )
    signal_workers = list(
        set(hparams["workers"]) - set(hparams["online_workers"])
    )
    for worker in hparams["online_workers"]:
        hparams["online_computation"][worker] = hparams["online_computation"][
            worker
        ].to(hparams["device"])

    # Due to DDP, we do the preparation ONLY on the main python process
    # Create the datasets objects as well as tokenization and encoding :-D
    train_data, valid_data, test_set = dataio_prepare(hparams)

    # Trainer initialization
    ssl_brain = SSL(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        opt_class=hparams["opt_class"],
        checkpointer=hparams["checkpointer"],
    )

    # Defining the workers regressors and losses
    # When adding new pretext tasks, you will have to add the corresponding
    # regression worker
    # an example is provided with a pitch_feature
    ssl_brain.workers_regressors = hparams["workers_regressors"]
    # Same for the losses
    ssl_brain.workers_losses = hparams["workers_losses"]

    ssl_brain.workers_loss_recap = {}
    for worker in ssl_brain.hparams.workers:
        ssl_brain.workers_loss_recap[worker] = []

    # Adding objects to trainer.
    # Training
    ssl_brain.fit(
        ssl_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["test_dataloader_options"],
    )
