#!/usr/bin/env/python3
"""
Recipe for training a Blind SI-SNR estimator

Authors:
 * Cem Subakan 2021
 * Mirco Ravanelli 2021
 * Samuele Cornell 2021
"""

import os
import sys
import torch
import speechbrain as sb
import speechbrain.nnet.schedulers as schedulers
from speechbrain.utils.distributed import run_on_main
from hyperpyyaml import load_hyperpyyaml
from torch.cuda.amp import autocast
import itertools as it
from tqdm import tqdm
import numpy as np
import logging
import csv


# Define training procedure
class Separation(sb.Brain):
    def compress_snrrange(self, inp):
        """Convert from true snr range to 0-1 range"""
        rnge = self.hparams.snrmax - self.hparams.snrmin

        inp = torch.clip(inp, min=self.hparams.snrmin, max=self.hparams.snrmax)
        inp = inp - self.hparams.snrmin
        inp = inp / rnge
        return inp

    def gettrue_snrrange(self, inp):
        """Convert from 0-1 range to true snr range"""
        rnge = self.hparams.snrmax - self.hparams.snrmin
        inp = inp * rnge
        inp = inp + self.hparams.snrmin
        return inp

    def compute_forward(self, mix, targets, stage, noise=None):
        """Forward computations from the mixture to the separated signals."""

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

                    if self.hparams.use_reverb_augment:
                        targets_rev = [
                            self.hparams.reverb(targets[:, :, i], None)
                            for i in range(self.hparams.num_spks)
                        ]
                        targets_rev = torch.stack(targets_rev, dim=-1)
                        mix = targets_rev.sum(-1)
                    else:
                        mix = targets.sum(-1)

                    if self.hparams.use_wham_noise:
                        noise = noise.to(self.device)
                        len_noise = noise.shape[1]
                        len_mix = mix.shape[1]
                        min_len = min(len_noise, len_mix)

                        # add the noise
                        mix = mix[:, :min_len] + noise[:, :min_len]

                        # fix the length of targets also
                        targets = targets[:, :min_len, :]

                if self.hparams.use_wavedrop:
                    mix = self.hparams.wavedrop(mix, mix_lens)

                if self.hparams.limit_training_signal_len:
                    mix, targets = self.cut_signals(mix, targets)

        # randomly select the separator to use, and separate
        with torch.no_grad():
            separator_model = np.random.choice(self.all_separators)
            predictions = separator_model.separate_batch(mix)

        # normalize the separation results
        if hasattr(self.hparams, "separation_norm_type"):
            if self.hparams.separation_norm_type == "max":
                predictions = (
                    predictions / predictions.max(dim=1, keepdim=True)[0]
                )
                mix = mix / mix.max(dim=1, keepdim=True)[0]

            elif self.hparams.separation_norm_type == "stnorm":
                predictions = (
                    predictions - predictions.mean(dim=1, keepdim=True)
                ) / predictions.std(dim=1, keepdim=True)
                mix = (mix - mix.mean(dim=1, keepdim=True)) / mix.std(
                    dim=1, keepdim=True
                )
            else:
                raise ValueError("Unknown type of normalization")

        # calculate oracle sisnrs
        snr = self.compute_oracle_sisnrs(predictions, targets)
        snr = snr.to(self.device)

        # compress the si-snr values to 0-1 range
        if self.hparams.use_snr_compression:
            snr_compressed = self.compress_snrrange(snr)
        predictions = predictions.permute(0, 2, 1)
        predictions = predictions.reshape(-1, predictions.size(-1))

        # make sure signal lengths do not change
        min_T = min(predictions.shape[1], mix.shape[1])
        assert predictions.shape[1] == mix.shape[1], "lengths change"

        # concat the mixtures to the separation results
        mix_repeat = mix.repeat(2, 1)
        inp_cat = torch.cat(
            [
                predictions[:, :min_T].unsqueeze(1),
                mix_repeat[:, :min_T].unsqueeze(1),
            ],
            dim=1,
        )

        # get the encoder output and then calculate stats pooling
        enc = self.hparams.encoder(inp_cat)
        enc = enc.permute(0, 2, 1)
        enc_stats = self.hparams.stat_pooling(enc)

        # get the si-snr estimate by passing through the output layers
        snrhat = self.hparams.encoder_out(enc_stats).squeeze()
        return predictions, snrhat, snr, snr_compressed

    def compute_oracle_sisnrs(self, predictions, targets):
        """Computes the oracle si-snrs"""

        snr1_1 = self.hparams.loss(targets[:, :, 0:1], predictions[:, :, 0:1])
        snr1_2 = self.hparams.loss(targets[:, :, 1:2], predictions[:, :, 0:1])
        snr1, ind1 = torch.stack([snr1_1, snr1_2]).min(0)

        ind2 = 1 - ind1
        snr2 = self.hparams.loss(targets[:, :, ind2], predictions[:, :, 1:2])
        return torch.stack([-snr1, -snr2], dim=1)

    def fit_batch(self, batch):
        """Trains one batch"""

        if self.hparams.use_whamr_train:
            whamr_prob = torch.rand(1).item()
            if whamr_prob > (1 - self.hparams.whamr_proportion):
                batch = next(self.hparams.train_whamr_loader)

        mixture = batch.mix_sig

        targets = [batch.s1_sig, batch.s2_sig]
        if self.hparams.use_wham_noise:
            noise = batch.noise_sig[0]
        else:
            noise = None

        if self.hparams.num_spks == 3:
            targets.append(batch.s3_sig)

        if self.auto_mix_prec:
            with autocast():
                predictions, snrhat, snr, snr_compressed = self.compute_forward(
                    mixture, targets, sb.Stage.TRAIN, noise
                )

                snr = snr.reshape(-1)
                loss = ((snr_compressed - snrhat).abs()).mean()

                if (
                    loss < self.hparams.loss_upper_lim and loss.nelement() > 0
                ):  # the fix for computational problems

                    self.scaler.scale(loss).backward()
                    if self.hparams.clip_grad_norm >= 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.modules.parameters(),
                            self.hparams.clip_grad_norm,
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.nonfinite_count += 1
                    logger.info(
                        "infinite loss or empty loss! it happened {} times so far - skipping this batch".format(
                            self.nonfinite_count
                        )
                    )
                    loss.data = torch.tensor(0).to(self.device)

        else:
            # get the oracle snrs, estimated snrs, and the source estimates
            predictions, snrhat, snr, snr_compressed = self.compute_forward(
                mixture, targets, sb.Stage.TRAIN, noise
            )

            snr = snr.reshape(-1)
            loss = ((snr_compressed - snrhat).abs()).mean()

            if (
                loss < self.hparams.loss_upper_lim and loss.nelement() > 0
            ):  # the fix for computational problems
                loss.backward()
                if self.hparams.clip_grad_norm >= 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.modules.parameters(), self.hparams.clip_grad_norm
                    )
                self.optimizer.step()
            else:
                self.nonfinite_count += 1
                logger.info(
                    "infinite loss or empty loss! it happened {} times so far - skipping this batch".format(
                        self.nonfinite_count
                    )
                )
                loss.data = torch.tensor(0).to(self.device)

        self.optimizer.zero_grad()

        return loss.detach().cpu()

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        # snt_id = batch.id
        mixture = batch.mix_sig
        targets = [batch.s1_sig, batch.s2_sig]
        if self.hparams.num_spks == 3:
            targets.append(batch.s3_sig)

        if self.hparams.use_wham_noise:
            noise = batch.noise_sig[0]
        else:
            noise = None

        with torch.no_grad():
            predictions, snrhat, snr, snr_compressed = self.compute_forward(
                mixture, targets, sb.Stage.VALID, noise
            )

            if self.hparams.use_snr_compression:
                snrhat = self.gettrue_snrrange(snrhat)

            loss = (snr - snrhat).abs().mean()

        return loss.detach()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {"error": stage_loss}
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
            self.checkpointer.save_and_keep_only(
                meta={"error": stage_stats["error"]}, min_keys=["error"],
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )

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
        randstart = torch.randint(
            0,
            1 + max(0, mixture.shape[1] - self.hparams.training_signal_len),
            (1,),
        ).item()
        targets = targets[
            :, randstart : randstart + self.hparams.training_signal_len, :
        ]
        mixture = mixture[
            :, randstart : randstart + self.hparams.training_signal_len
        ]
        return mixture, targets

    def reset_layer_recursively(self, layer):
        """Reinitializes the parameters of the neural networks"""
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
        for child_layer in layer.modules():
            if layer != child_layer:
                self.reset_layer_recursively(child_layer)

    def save_results(self, test_data):
        """
        This function calculates the oracle si-snrs and the estimated si-snr on the test set of WHAMR! dataset, and writes these results into a csv file
        """

        # Create folders where to store audio
        save_file = os.path.join(
            self.hparams.output_folder, "test_results_wsj.csv"
        )

        # Variable init
        all_sisnr1s = []
        all_sisnr1_hats = []
        all_sisnr2s = []
        all_sisnr2_hats = []

        csv_columns = ["snt_id", "snr1", "snr1-hat", "snr2", "snr2-hat"]

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
                    mixture = batch.mix_sig
                    snt_id = batch.id
                    targets = [batch.s1_sig, batch.s2_sig]
                    if self.hparams.num_spks == 3:
                        targets.append(batch.s3_sig)

                    if self.hparams.use_wham_noise:
                        noise = batch.noise_sig[0]
                    else:
                        noise = None

                    with torch.no_grad():
                        (
                            predictions,
                            snrhat,
                            snr,
                            snr_compressed,
                        ) = self.compute_forward(
                            mixture, targets, sb.Stage.VALID, noise
                        )

                        if self.hparams.use_snr_compression:
                            snrhat = self.gettrue_snrrange(snrhat)

                    # Saving on a csv file
                    row = {
                        "snt_id": snt_id[0],
                        "snr1": snr.squeeze()[0].item(),
                        "snr1-hat": snrhat[0].item(),
                        "snr2": snr.squeeze()[1].item(),
                        "snr2-hat": snrhat[1].item(),
                    }
                    writer.writerow(row)

                    # Metric Accumulation
                    all_sisnr1s.append(snr.squeeze()[0].item())
                    all_sisnr1_hats.append(snrhat[0].item())
                    all_sisnr2s.append(snr.squeeze()[1].item())
                    all_sisnr2_hats.append(snrhat[1].item())

                row = {
                    "snt_id": "avg",
                    "snr1": np.array(all_sisnr1s).mean(),
                    "snr1-hat": np.array(all_sisnr1_hats).mean(),
                    "snr2": np.array(all_sisnr2s).mean(),
                    "snr2-hat": np.array(all_sisnr2_hats).mean(),
                }
                writer.writerow(row)

        logger.info(
            "Mean SISNR for source 1 is {}".format(np.array(all_sisnr1s).mean())
        )
        logger.info(
            "Mean SISNR hat for source 1 is {}".format(
                np.array(all_sisnr1_hats).mean()
            )
        )
        logger.info(
            "Mean SISNR for source 2 is {}".format(np.array(all_sisnr2s).mean())
        )
        logger.info(
            "Mean SISNR hat for source 2 is {}".format(
                np.array(all_sisnr2_hats).mean()
            )
        )


def dataio_prep(hparams):
    """Creates data processing pipeline"""

    # 1. Define datasets
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_data"],
        replacements={"data_root": hparams["data_folder"]},
    )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_data"],
        replacements={"data_root": hparams["data_folder"]},
    )

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_data"],
        replacements={"data_root": hparams["data_folder"]},
    )

    datasets = [train_data, valid_data, test_data]

    # 2. Provide audio pipelines

    @sb.utils.data_pipeline.takes("mix_wav")
    @sb.utils.data_pipeline.provides("mix_sig")
    def audio_pipeline_mix(mix_wav):
        mix_sig = sb.dataio.dataio.read_audio(mix_wav)
        return mix_sig

    @sb.utils.data_pipeline.takes("s1_wav")
    @sb.utils.data_pipeline.provides("s1_sig")
    def audio_pipeline_s1(s1_wav):
        s1_sig = sb.dataio.dataio.read_audio(s1_wav)
        return s1_sig

    @sb.utils.data_pipeline.takes("s2_wav")
    @sb.utils.data_pipeline.provides("s2_sig")
    def audio_pipeline_s2(s2_wav):
        s2_sig = sb.dataio.dataio.read_audio(s2_wav)
        return s2_sig

    if hparams["num_spks"] == 3:

        @sb.utils.data_pipeline.takes("s3_wav")
        @sb.utils.data_pipeline.provides("s3_sig")
        def audio_pipeline_s3(s3_wav):
            s3_sig = sb.dataio.dataio.read_audio(s3_wav)
            return s3_sig

    if hparams["use_wham_noise"]:

        @sb.utils.data_pipeline.takes("noise_wav")
        @sb.utils.data_pipeline.provides("noise_sig")
        def audio_pipeline_noise(noise_wav):
            noise_sig = sb.dataio.dataio.read_audio(noise_wav)
            return noise_sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_mix)
    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_s1)
    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_s2)
    if hparams["num_spks"] == 3:
        sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_s3)

    if hparams["use_wham_noise"]:
        print("Using the WHAM! noise in the data pipeline")
        sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_noise)

    if (hparams["num_spks"] == 2) and hparams["use_wham_noise"]:
        sb.dataio.dataset.set_output_keys(
            datasets, ["id", "mix_sig", "s1_sig", "s2_sig", "noise_sig"]
        )
    elif (hparams["num_spks"] == 3) and hparams["use_wham_noise"]:
        sb.dataio.dataset.set_output_keys(
            datasets,
            ["id", "mix_sig", "s1_sig", "s2_sig", "s3_sig", "noise_sig"],
        )
    elif (hparams["num_spks"] == 2) and not hparams["use_wham_noise"]:
        sb.dataio.dataset.set_output_keys(
            datasets, ["id", "mix_sig", "s1_sig", "s2_sig"]
        )
    else:
        sb.dataio.dataset.set_output_keys(
            datasets, ["id", "mix_sig", "s1_sig", "s2_sig", "s3_sig"]
        )

    return train_data, valid_data, test_data


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

    # Check if wsj0_tr is set with dynamic mixing
    if hparams["dynamic_mixing"] and not os.path.exists(
        hparams["base_folder_dm"]
    ):
        print(
            "Please, specify a valid base_folder_dm folder when using dynamic mixing"
        )
        sys.exit(1)

    # Data preparation for LibriMix
    from prepare_data_librimix import prepare_librimix as prepare_libri

    # create the csv files
    run_on_main(
        prepare_libri,
        kwargs={
            "datapath": hparams["data_folder"],
            "savepath": hparams["save_folder"],
            "n_spks": hparams["num_spks"],
            "skip_prep": hparams["skip_prep"],
            "fs": hparams["sample_rate"],
            "librimix_addnoise": hparams["use_wham_noise"],
        },
    )

    # Data preparation for WHAMR
    from prepare_data_wham import create_wham_whamr_csv
    from train_wham import dataio_prep as dataio_prep_whamr

    # add another skip_prep to distinguish between LibriSpeech & WHAM/R prep
    skip_prep = hparams["skip_prep"]
    if not skip_prep:
        create_wham_whamr_csv(
            datapath=hparams["whamr_data_folder"],
            savepath=hparams["save_folder"],
            fs=hparams["sample_rate"],
            add_reverb=True,
            savename="whamr_",
            set_types=["tr", "cv", "tt"],
        )

    train_data_whamr, valid_data, test_data = dataio_prep_whamr(hparams)

    # if whamr, and we do speedaugment we need to prepare the csv file
    if hparams["use_reverb_augment"]:
        from prepare_data_wham import create_whamr_rir_csv
        from create_whamr_rirs import create_rirs

        # If the Room Impulse Responses do not exist, we create them
        if not os.path.exists(hparams["rir_path"]):
            print("ing Room Impulse Responses...")
            run_on_main(
                create_rirs,
                kwargs={
                    "output_dir": hparams["rir_path"],
                    "sr": hparams["sample_rate"],
                },
            )

        run_on_main(
            create_whamr_rir_csv,
            kwargs={
                "datapath": hparams["rir_path"],
                "savepath": hparams["save_folder"],
            },
        )

        hparams["reverb"] = sb.processing.speech_augmentation.AddReverb(
            os.path.join(hparams["save_folder"], "whamr_rirs.csv")
        )

    if hparams["dynamic_mixing"]:
        from dynamic_mixing_librimix import (
            dynamic_mix_data_prep_librimix as dynamic_mix_data_prep,
        )
        from dynamic_mixing_wham import (
            dynamic_mix_data_prep as dynamic_mix_data_prep_whamr,
        )

        if hparams["use_whamr_train"]:

            if "processed" not in hparams["base_folder_dm_whamr"]:
                # if the processed folder does not exist for whamr dynamic mixing, we do the necessary preprocessing

                if not os.path.exists(
                    os.path.normpath(hparams["base_folder_dm_whamr"])
                    + "_processed"
                ):
                    from preprocess_dynamic_mixing_wham import resample_folder

                    print("Resampling the base folder")
                    run_on_main(
                        resample_folder,
                        kwargs={
                            "input_folder": hparams["base_folder_dm_whamr"],
                            "output_folder": os.path.normpath(
                                hparams["base_folder_dm_whamr"]
                            )
                            + "_processed",
                            "fs": hparams["sample_rate"],
                            "regex": "**/*.wav",
                        },
                    )
                    # adjust the base_folder_dm path
                    hparams["base_folder_dm_whamr"] = (
                        os.path.normpath(hparams["base_folder_dm_whamr"])
                        + "_processed"
                    )
                else:
                    print(
                        "Using the existing processed folder on the same directory as base_folder_dm"
                    )
                    hparams["base_folder_dm_whamr"] = (
                        os.path.normpath(hparams["base_folder_dm_whamr"])
                        + "_processed"
                    )

            train_data_whamr = dynamic_mix_data_prep_whamr(
                tr_csv=hparams["train_whamr_data"],
                data_root_folder=hparams["whamr_data_folder"],
                base_folder_dm=hparams["base_folder_dm_whamr"],
                sample_rate=hparams["sample_rate"],
                num_spks=hparams["num_spks"],
                max_training_signal_len=hparams["training_signal_len"],
                batch_size=hparams["dataloader_opts"]["batch_size"],
                num_workers=hparams["dataloader_opts"]["num_workers"],
            )
            hparams["train_whamr_loader"] = it.cycle(iter(train_data_whamr))

        # if the base_folder for dm is not processed for LibriMix, preprocess it
        if "processed" not in hparams["base_folder_dm"]:
            # if the processed folder already exists we just use it otherwise we do the preprocessing
            if not os.path.exists(
                os.path.normpath(hparams["base_folder_dm"]) + "_processed"
            ):
                from preprocess_dynamic_mixing_librimix import resample_folder

                print("Resampling the base folder")
                run_on_main(
                    resample_folder,
                    kwargs={
                        "input_folder": hparams["base_folder_dm"],
                        "output_folder": os.path.normpath(
                            hparams["base_folder_dm"]
                        )
                        + "_processed",
                        "fs": hparams["sample_rate"],
                        "regex": "**/*.flac",
                    },
                )
                # adjust the base_folder_dm path
                hparams["base_folder_dm"] = (
                    os.path.normpath(hparams["base_folder_dm"]) + "_processed"
                )
            else:
                print(
                    "Using the existing processed folder on the same directory as base_folder_dm"
                )
                hparams["base_folder_dm"] = (
                    os.path.normpath(hparams["base_folder_dm"]) + "_processed"
                )

        train_data = dynamic_mix_data_prep(hparams)
    else:
        train_data, valid_data, test_data = dataio_prep(hparams)

    # Brain class initialization
    snrestimator = Separation(
        modules=hparams["modules"],
        opt_class=hparams["optimizer"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    from speechbrain.pretrained import SepformerSeparation as separator
    from speechbrain.pretrained.interfaces import fetch

    all_separators = []
    for separator_model in hparams["separators_to_use"]:
        fetch(
            separator_model + "_encoder.ckpt",
            source=hparams["separator_repo"],
            savedir=separator_model,
            save_filename="encoder.ckpt",
        )

        fetch(
            separator_model + "_decoder.ckpt",
            source=hparams["separator_repo"],
            savedir=separator_model,
            save_filename="decoder.ckpt",
        )

        fetch(
            separator_model + "_masknet.ckpt",
            source=hparams["separator_repo"],
            savedir=separator_model,
            save_filename="masknet.ckpt",
        )

        fetch(
            separator_model + "_hyperparams.yaml",
            source=hparams["separator_repo"],
            savedir=separator_model,
            save_filename="hyperparams.yaml",
        )

        separator_loaded = separator.from_hparams(
            source=separator_model,
            run_opts={"device": "cuda"},
            savedir=separator_model,
        )

        all_separators.append(separator_loaded)

    snrestimator.all_separators = all_separators

    if not hparams["test_only"]:
        # Training
        snrestimator.fit(
            snrestimator.hparams.epoch_counter,
            train_data,
            valid_data,
            train_loader_kwargs=hparams["dataloader_opts"],
            valid_loader_kwargs=hparams["dataloader_opts"],
        )

    # Eval
    snrestimator.evaluate(test_data, min_key="error")
    snrestimator.save_results(test_data)
