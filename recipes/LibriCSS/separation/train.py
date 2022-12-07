#!/usr/bin/env python
"""Recipe for training a neural speech separation system on Libri2/3Mix datasets.
The system employs an encoder, a decoder, and a masking network.

To run this recipe, do the following:
> python train.py hparams/sepformer-libri2mix.yaml
> python train.py hparams/sepformer-libri3mix.yaml


The experiment file is flexible enough to support different neural
networks. By properly changing the parameter files, you can try
different architectures. The script supports both libri2mix and
libri3mix.


Authors
 * Cem Subakan 2020
 * Mirco Ravanelli 2020
 * Samuele Cornell 2020
 * Mirko Bronzi 2020
 * Jianyuan Zhong 2020
"""

import os
import sys
import torch
import torch.nn.functional as F
import torchaudio
import speechbrain as sb
import speechbrain.nnet.schedulers as schedulers
from speechbrain.utils.distributed import run_on_main
from hyperpyyaml import load_hyperpyyaml
import numpy as np
from tqdm import tqdm
import csv
import logging
from speechbrain.dataio.batch import PaddedBatch
from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.processing.dynamic_mixing import DynamicMixingDataset
from speechbrain.processing.signal_processing import reverberate

import wandb


# Define training procedure
class Separation(sb.Brain):
    def fit_batch(self, batch):
        if not isinstance(batch, PaddedBatch):
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].squeeze(0)
            batch = PaddedBatch([batch])
        return super().fit_batch(batch)

    def compute_forward(self, batch, stage=sb.Stage.TRAIN):
        """Forward computations from the mixture to the separated signals."""
        # Unpacking batch list
        mix = batch.mix_sig
        noise = batch.noise_sig
        targets = [batch.s1_sig, batch.s2_sig]

        if self.hparams.num_spks == 3:
            targets.append(batch.s3_sig)

        # Unpack lists and put tensors in the right device
        mix, mix_lens = mix
        mix, mix_lens = mix.to(self.device), mix_lens.to(self.device)

        # Convert targets to tensor
        targets = torch.cat(
            [targets[i][0].unsqueeze(-1) for i in range(self.hparams.num_spks)],
            dim=-1,
        ).to(self.device)

        # Add speech distortions
        if stage == sb.Stage.TRAIN:
            with torch.no_grad():
                if self.hparams.use_speedperturb or self.hparams.use_rand_shift:
                    mix, targets = self.add_speed_perturb(targets, mix_lens)
                    mix = targets.sum(-1)
                    if (
                        not self.hparams.dm_config.reverb_sources
                        and batch.rir is not None
                    ):
                        # targets are clear, we need to manually reverberate the mixture
                        mix = reverberate(mix, batch.rir)
                    mix += noise

                if self.hparams.use_wavedrop:
                    mix = self.hparams.wavedrop(mix, mix_lens)

                if self.hparams.limit_training_signal_len:
                    mix, targets = self.cut_signals(mix, targets)

        # Separation
        mix_w = self.hparams.Encoder(mix)
        est_mask = self.hparams.MaskNet(mix_w)
        mix_w = torch.stack([mix_w] * self.hparams.num_spks)
        sep_h = mix_w * est_mask

        # Decoding
        est_source = torch.cat(
            [
                self.hparams.Decoder(sep_h[i]).unsqueeze(-1)
                for i in range(self.hparams.num_spks)
            ],
            dim=-1,
        )

        # T changed after conv1d in encoder, fix it here
        T_origin = mix.size(1)
        T_est = est_source.size(1)
        if T_origin > T_est:
            est_source = F.pad(est_source, (0, 0, 0, T_origin - T_est))
        else:
            est_source = est_source[:, :T_origin, :]

        if stage == sb.Stage.TEST:
            if hasattr(self.hparams, "n_audio_to_save"):
                if self.hparams.n_audio_to_save > 0:
                    self.save_audio(batch.id[0], mix, targets, est_source)
                    self.hparams.n_audio_to_save += -1
            else:
                self.save_audio(batch.id[0], mix, targets, est_source)

        return mix, est_source, targets

    def compute_objectives(self, outputs, batch, stage=sb.Stage.TRAIN):
        """Computes the si-snr loss"""
        _, est_source, targets = outputs
        loss = self.hparams.loss(targets, est_source)
        # hard threshold the easy dataitems
        if self.hparams.threshold_byloss and stage == sb.Stage.TRAIN:
            th = self.hparams.threshold
            loss_to_keep = loss[loss > th]
            if loss_to_keep.nelement() > 0:
                loss = loss_to_keep

        return loss.mean()

    def on_fit_batch_end(self, batch, outputs, loss, should_step):
        """Called from _fit_train"""
        mix, est_source, targets = outputs

        if self.optimizer_step % self.hparams.audio_logging_interval == 0:
            signals = [x.detach().cpu() for x in [mix, targets[:, :, 0], targets[:, :, 1], est_source[:, :, 0], est_source[:, :, 1]]]
            signals = [x / x.abs().max() for x in signals]
            audios = [wandb.Audio(x.squeeze().float().numpy(), sample_rate=self.hparams.sample_rate) for x in signals]
            if not hasattr(self, "wandb_train_table"):
                self.wandb_train_table = wandb.Table(columns=["epoch", "step", "id", "mix", "target1", "target2", "est_source1", "est_source2"])
            data = [self.hparams.epoch_counter.current, self.optimizer_step, batch.id[0]] + audios
            self.wandb_train_table.add_data(*data)
            wandb.log({"train_samples": self.wandb_train_table})

        if self.optimizer_step % self.hparams.logging_interval == 0:
            wandb.log({"train_sisnr": loss.detach().cpu().numpy()}, commit=True, step=self.optimizer_step)

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {"si-snr": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            # Learning rate annealing
            if isinstance(
                self.hparams.lr_scheduler, schedulers.ReduceLROnPlateau
            ):
                current_lr, next_lr = self.hparams.lr_scheduler(
                    [self.optimizer], epoch, stage_loss
                )
                schedulers.update_learning_rate(self.optimizer, next_lr)
            else:
                # if we do not use the reducelronplateau, we do not change the lr
                current_lr = self.hparams.optimizer.optim.param_groups[0]["lr"]

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": current_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            wandb.log(
                {"lr": current_lr}, step=self.hparams.epoch_counter.current
            )
            wandb.log({**stage_stats})

            self.checkpointer.save_and_keep_only(
                meta={"si-snr": stage_stats["si-snr"]},
                min_keys=["si-snr"],
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            if hasattr(self, "wandb_table") and self.wandb_table is not None:
                wandb.log(
                    {"test_samples": self.wandb_table}, step=epoch, commit=True
                )
                self.wandb_table = None

    def add_speed_perturb(self, targets, targ_lens):
        """Adds speed perturbation and random_shift to the input signals"""

        min_len = -1
        recombine = False

        if self.hparams.use_speedperturb:
            # Performing speed change (independently on each source)
            new_targets = []
            recombine = True

            for i in range(targets.shape[-1]):
                new_target = self.hparams.speedperturb(
                    targets[:, :, i], targ_lens
                )
                new_targets.append(new_target)
                if i == 0:
                    min_len = new_target.shape[-1]
                else:
                    if new_target.shape[-1] < min_len:
                        min_len = new_target.shape[-1]

            if self.hparams.use_rand_shift:
                # Performing random_shift (independently on each source)
                recombine = True
                for i in range(targets.shape[-1]):
                    rand_shift = torch.randint(
                        self.hparams.min_shift, self.hparams.max_shift, (1,)
                    )
                    new_targets[i] = new_targets[i].to(self.device)
                    new_targets[i] = torch.roll(
                        new_targets[i], shifts=(rand_shift[0],), dims=1
                    )

            # Re-combination
            if recombine:
                if self.hparams.use_speedperturb:
                    targets = torch.zeros(
                        targets.shape[0],
                        min_len,
                        targets.shape[-1],
                        device=targets.device,
                        dtype=torch.float,
                    )
                for i, new_target in enumerate(new_targets):
                    targets[:, :, i] = new_targets[i][:, 0:min_len]

        mix = targets.sum(-1)
        return mix, targets

    def cut_signals(self, mixture, targets):
        """This function selects a random segment of a given length withing the mixture.
        The corresponding targets are selected accordingly"""
        length = torch.randint(
            self.hparams.min_training_len, self.hparams.max_training_len, (1,)
        ).item()
        randstart = torch.randint(
            0, max(1, mixture.shape[1] - length), (1,)
        ).item()
        targets = targets[:, randstart : randstart + length, :]
        mixture = mixture[:, randstart : randstart + length]
        return mixture, targets

    def reset_layer_recursively(self, layer):
        """Reinitializes the parameters of the neural networks"""
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
        for child_layer in layer.modules():
            if layer != child_layer:
                self.reset_layer_recursively(child_layer)

    def compute_metrics(self, mix_sig, est_sources, targets):
        from mir_eval.separation import bss_eval_sources

        mix_expanded = torch.stack([mix_sig] * self.hparams.num_spks, dim=-1)
        mix_expanded = mix_expanded.to(targets.device)

        # Compute SI-SNR
        sisnr = self.compute_objectives(est_sources, targets)

        # Compute SI-SNR improvement
        sisnr_baseline = self.compute_objectives(
            mix_expanded,
            targets,
        )
        sisnr_i = sisnr - sisnr_baseline

        # Compute SDR
        try:
            sdr, _, _, _ = bss_eval_sources(
                targets[0].t().cpu().numpy(),
                est_sources[0].detach().t().cpu().numpy(),
            )
            sdr = sdr.mean()

            sdr_baseline, _, _, _ = bss_eval_sources(
                targets[0].t().cpu().numpy(),
                mix_expanded[0].detach().t().cpu().numpy(),
            )
            sdr_baseline = sdr_baseline.mean()

            sdr_i = sdr - sdr_baseline
        except ValueError:
            sdr, sdr_baseline, sdr_i = None, None, None

        return {
            "sdr": sdr,
            "sdr_i": sdr_i,
            "si-snr": -sisnr.item(),
            "si-snr_i": -sisnr_i.item(),
        }

    def save_results(self, test_data):
        """This script computes the SDR and SI-SNR metrics and saves
        them into a csv file"""

        # This package is required for SDR computation
        from mir_eval.separation import bss_eval_sources

        # Create folders where to store audio
        save_file = os.path.join(self.hparams.output_folder, "test_results.csv")

        # Variable init
        all_sdrs = []
        all_sdrs_i = []
        all_sisnrs = []
        all_sisnrs_i = []
        csv_columns = ["snt_id", "sdr", "sdr_i", "si-snr", "si-snr_i"]

        test_loader = sb.dataio.dataloader.make_dataloader(
            test_data, **self.hparams.dataloader_opts
        )

        with open(save_file, "w") as results_csv:
            writer = csv.DictWriter(results_csv, fieldnames=csv_columns)
            writer.writeheader()

            # Loop over all test sentence
            with tqdm(test_loader, dynamic_ncols=True) as t:
                for i, batch in enumerate(t):

                    # Apply Separation
                    with torch.no_grad():
                        mixture, predictions, targets = self.compute_forward(batch, sb.Stage.TEST)

                    metrics = self.compute_metrics(
                        mixture, predictions, targets
                    )
                    # Saving on a csv file
                    row = {
                        "snt_id": snt_id[0],
                        **metrics,
                    }
                    writer.writerow(row)

                    # Metric Accumulation
                    if (
                        metrics["sdr"] is not None
                        and metrics["sdr_i"] is not None
                    ):
                        all_sdrs.append(metrics["sdr"])
                        all_sdrs_i.append(metrics["sdr_i"])
                    all_sisnrs.append(metrics["si-snr"])
                    all_sisnrs_i.append(metrics["si-snr_i"])

                row = {
                    "snt_id": "avg",
                    "sdr": np.array(all_sdrs).mean(),
                    "sdr_i": np.array(all_sdrs_i).mean(),
                    "si-snr": np.array(all_sisnrs).mean(),
                    "si-snr_i": np.array(all_sisnrs_i).mean(),
                }
                writer.writerow(row)

        logger.info("Mean SISNR is {}".format(np.array(all_sisnrs).mean()))
        logger.info("Mean SISNRi is {}".format(np.array(all_sisnrs_i).mean()))
        logger.info("Mean SDR is {}".format(np.array(all_sdrs).mean()))
        logger.info("Mean SDRi is {}".format(np.array(all_sdrs_i).mean()))

    def save_audio(self, snt_id, mixture, targets, predictions):
        "saves the test audio (mixture, targets, and estimated sources) on disk"

        # Create outout folder
        save_path = os.path.join(self.hparams.save_folder, "audio_results")
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        metrics = self.compute_metrics(mixture[0], predictions, targets)
        data = [snt_id] + list(metrics.values())

        if not hasattr(self, "wandb_table") or self.wandb_table is None:
            columns = ["est_source", "target"] * self.hparams.num_spks
            columns = [x + str(i // 2) for i, x in enumerate(columns)]
            columns = ["id"] + list(metrics.keys()) + columns + ["mixture"]
            self.wandb_table = wandb.Table(columns=columns)

        for ns in range(self.hparams.num_spks):
            # Estimated source
            signal = predictions[0, :, ns]
            signal = signal / signal.abs().max()
            save_file = os.path.join(
                save_path, "item{}_source{}hat.wav".format(snt_id, ns + 1)
            )
            torchaudio.save(
                save_file, signal.unsqueeze(0).cpu(), self.hparams.sample_rate
            )
            data.append(
                wandb.Audio(
                    signal.detach().cpu().numpy(),
                    sample_rate=self.hparams.sample_rate,
                )
            )

            # Original source
            signal = targets[0, :, ns]
            signal = signal / signal.abs().max()
            save_file = os.path.join(
                save_path, "item{}_source{}.wav".format(snt_id, ns + 1)
            )
            torchaudio.save(
                save_file, signal.unsqueeze(0).cpu(), self.hparams.sample_rate
            )
            data.append(
                wandb.Audio(
                    signal.detach().cpu().numpy(),
                    sample_rate=self.hparams.sample_rate,
                )
            )

        # Mixture
        signal = mixture[0][0, :]
        signal = signal / signal.abs().max()
        save_file = os.path.join(save_path, "item{}_mix.wav".format(snt_id))
        torchaudio.save(
            save_file, signal.unsqueeze(0).cpu(), self.hparams.sample_rate
        )
        data.append(
            wandb.Audio(
                signal.detach().cpu().numpy(),
                sample_rate=self.hparams.sample_rate,
            )
        )

        self.wandb_table.add_data(*data)


def dataio_prep(hparams):
    """Creates data processing pipeline"""
    dm_config = hparams["dm_config"]
    datasets = [
        DynamicItemDataset.from_csv(
            hparams[key], output_keys=["id", "wav", "spk_id", "duration"]
        )
        for key in ["train_data", "valid_data"]
    ]
    dm_datasets = [
        DynamicMixingDataset.from_didataset(
            data,
            dm_config,
            "wav",
            "spk_id",
            noise_flist=hparams["noise_files"],
            rir_flist=hparams["rir_files"],
            replacements={"RIRS_NOISES": hparams["noises_root"]},
        )
        for data in datasets
    ]
    return dm_datasets


if __name__ == "__main__":

    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Logger info
    logger = logging.getLogger(__name__)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Data preparation
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    from recipes.LibriSpeech.librispeech_prepare import prepare_librispeech

    run_on_main(
        prepare_librispeech,
        kwargs={
            "data_folder": hparams["dm_data_folder"],
            "save_folder": hparams["save_folder"],
            "skip_prep": hparams["skip_prep"],
            "tr_splits": ["train-clean-360"],
            "dev_splits": ["dev-clean"],
        },
    )
    train_data, valid_data = dataio_prep(hparams)

    # Load pretrained model if pretrained_separator is present in the yaml
    if "pretrained_separator" in hparams:
        run_on_main(hparams["pretrained_separator"].collect_files)
        hparams["pretrained_separator"].load_collected(
            device=run_opts["device"]
        )

    wandb.init(
        project="SepFormer",
        entity="mato1102",
        config={},
        resume=True,
        name=hparams["experiment_name"],
    )

    # Brain class initialization
    separator = Separation(
        modules=hparams["modules"],
        opt_class=hparams["optimizer"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # re-initialize the parameters if we don't use a pretrained model
    if "pretrained_separator" not in hparams:
        for module in separator.modules.values():
            separator.reset_layer_recursively(module)

    if not hparams["test_only"]:
        # Training
        separator.fit(
            separator.hparams.epoch_counter,
            train_data,
            valid_data,
            train_loader_kwargs=hparams["dataloader_opts"],
            valid_loader_kwargs=hparams["dataloader_opts"],
        )

    #    # Eval
    #    separator.evaluate(test_data, min_key="si-snr", test_loader_kwargs={"shuffle": True})
    #    separator.save_results(test_data)

    wandb.finish()
