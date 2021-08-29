#!/usr/bin/env/python3
"""Recipe for training a neural speech separation system on wsjmix the
dataset. The system employs an encoder, a decoder, and a masking network.

To run this recipe, do the following:
> python train.py hparams/sepformer.yaml
> python train.py hparams/dualpath_rnn.yaml
> python train.py hparams/convtasnet.yaml

The experiment file is flexible enough to support different neural
networks. By properly changing the parameter files, you can try
different architectures. The script supports both wsj2mix and
wsj3mix.


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
from torch.cuda.amp import autocast
from hyperpyyaml import load_hyperpyyaml
import numpy as np
from tqdm import tqdm
import csv
import logging
import itertools as it


# Define training procedure
class Separation(sb.Brain):
    def compress_snrrange(self, inp):
        rnge = self.hparams.snrmax - self.hparams.snrmin

        inp = torch.clip(inp, min=self.hparams.snrmin, max=self.hparams.snrmax)
        inp = inp - self.hparams.snrmin
        inp = inp / rnge
        return inp

    def gettrue_snrrange(self, inp):
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
        if 1:  # stage == sb.Stage.TRAIN:
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

        # Separation
        with torch.no_grad():
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
            predictions = F.pad(est_source, (0, 0, 0, T_origin - T_est))
        else:
            predictions = est_source[:, :T_origin, :]

        # added part

        snr = self.compute_objectives(predictions, targets)
        snr = snr.to(self.device)
        if self.hparams.use_snr_compression:
            snr_compressed = self.compress_snrrange(snr)
        predictions = predictions.permute(0, 2, 1)
        predictions = predictions.reshape(-1, predictions.size(-1))

        if hasattr(self.hparams, "compute_features"):
            feats_preds = self.hparams.compute_features(predictions)
            feats_mix = self.hparams.compute_features(mix)
            min_T = min(feats_preds.shape[1], feats_mix.shape[1])

            assert feats_preds.shape[1] == feats_mix.shape[1], "lengths change"
            feats_preds = feats_preds[:, :min_T, :]
            feats_mix = feats_mix[:, :min_T, :]

            feats_mix_repeat = feats_mix.repeat(feats_preds.shape[0], 1, 1)
            inp_cat = torch.cat([feats_preds, feats_mix_repeat], dim=-1)

            enc_stats = self.hparams.classifier_enc(inp_cat)
        else:
            mix_repeat = mix.repeat(2, 1)
            min_T = min(predictions.shape[1], mix.shape[1])
            assert predictions.shape[1] == mix.shape[1], "lengths change"

            inp_cat = torch.cat(
                [
                    predictions[:, :min_T].unsqueeze(1),
                    mix_repeat[:, :min_T].unsqueeze(1),
                ],
                dim=1,
            )

            enc = self.hparams.classifier_enc(inp_cat)
            enc = enc.permute(0, 2, 1)
            enc_stats = self.hparams.stat_pooling(enc)

        snrhat = self.hparams.classifier_out(enc_stats).squeeze()
        return predictions, snrhat, snr, snr_compressed

    def compute_objectives(self, predictions, targets):
        """Computes the sinr loss"""
        # self.hparams.loss(targets, predictions)

        # snr1_1 = self.hparams.loss(targets[:, :, 0:1], predictions[:, :, 0:1]).item()
        # snr1_2 = self.hparams.loss(targets[:, :, 1:2], predictions[:, :, 0:1]).item()
        # snr1 = -min(snr1_1, snr1_2)

        # snr2_1 = self.hparams.loss(targets[:, :, 0:1], predictions[:, :, 1:2]).item()
        # snr2_2 = self.hparams.loss(targets[:, :, 1:2], predictions[:, :, 1:2]).item()
        # snr2 = -min(snr2_1, snr2_2)

        snr1_1 = self.hparams.loss(targets[:, :, 0:1], predictions[:, :, 0:1])
        snr1_2 = self.hparams.loss(targets[:, :, 1:2], predictions[:, :, 0:1])
        snr1 = torch.stack([snr1_1, snr1_2]).min(0)

        snr2_1 = self.hparams.loss(targets[:, :, 0:1], predictions[:, :, 1:2])
        snr2_2 = self.hparams.loss(targets[:, :, 1:2], predictions[:, :, 1:2])
        snr2 = torch.stack([snr2_1, snr2_2]).min(0)

        return torch.stack([-snr1[0], -snr2[0]], dim=1)

    def fit_batch(self, batch):
        """Trains one batch"""
        # Unpacking batch list

        if self.hparams.use_whamr_train:
            whamr_prob = torch.rand(1).item()
            if whamr_prob > 0.5:
                batch = next(self.hparams.train_wham_loader)

        mixture = batch.mix_sig
        targets = [batch.s1_sig, batch.s2_sig]
        if self.hparams.use_wham_noise:
            noise = batch.noise_sig[0]
        else:
            noise = None

        if self.hparams.num_spks == 3:
            targets.append(batch.s3_sig)

        if self.hparams.auto_mix_prec:
            pass
        else:
            predictions, snrhat, snr, snr_compressed = self.compute_forward(
                mixture, targets, sb.Stage.TRAIN, noise
            )

            if self.hparams.use_snr_compression:
                snr = snr.reshape(-1)
                loss = ((snr_compressed - snrhat).abs()).mean()
            else:
                loss = ((snr - snrhat) ** 2).mean()

            # total_grad_norm = 0
            # for param in self.modules.parameters():
            #     if param.grad != None:
            #         total_grad_norm = total_grad_norm + param.grad.abs().sum()
            # print(total_grad_norm)

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
        snt_id = batch.id
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

        # Manage audio file saving
        if stage == sb.Stage.TEST and self.hparams.save_audio:
            if hasattr(self.hparams, "n_audio_to_save"):
                if self.hparams.n_audio_to_save > 0:
                    self.save_audio(snt_id[0], mixture, targets, predictions)
                    self.hparams.n_audio_to_save += -1
            else:
                self.save_audio(snt_id[0], mixture, targets, predictions)

        return loss.detach()

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
            self.checkpointer.save_and_keep_only(
                meta={"si-snr": stage_stats["si-snr"]}, min_keys=["si-snr"],
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
        """This script computes the SDR and SI-SNR metrics and saves
        them into a csv file"""

        # This package is required for SDR computation
        # from mir_eval.separation import bss_eval_sources

        # Create folders where to store audio
        if self.hparams.test_onwsj:
            save_file = os.path.join(
                self.hparams.output_folder, "test_results_wsj.csv"
            )
        else:
            save_file = os.path.join(
                self.hparams.output_folder, "test_results_libri.csv"
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

    def save_audio(self, snt_id, mixture, targets, predictions):
        "saves the test audio (mixture, targets, and estimated sources) on disk"

        # Create outout folder
        save_path = os.path.join(self.hparams.save_folder, "audio_results")
        if not os.path.exists(save_path):
            os.mkdir(save_path)

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

            # Original source
            signal = targets[0, :, ns]
            signal = signal / signal.abs().max()
            save_file = os.path.join(
                save_path, "item{}_source{}.wav".format(snt_id, ns + 1)
            )
            torchaudio.save(
                save_file, signal.unsqueeze(0).cpu(), self.hparams.sample_rate
            )

        # Mixture
        signal = mixture[0][0, :]
        signal = signal / signal.abs().max()
        save_file = os.path.join(save_path, "item{}_mix.wav".format(snt_id))
        torchaudio.save(
            save_file, signal.unsqueeze(0).cpu(), self.hparams.sample_rate
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


def dataio_prep_wham_traintest(hparams):
    """Creates data processing pipeline"""

    # 1. Define datasets
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_wsj_data"],
        replacements={"data_root": hparams["wsj_data_folder"]},
    )

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_wsj_data"],
        replacements={"data_root": hparams["wsj_data_folder"]},
    )

    datasets = [train_data, test_data]

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

    if hparams["use_wham_noise"]:

        @sb.utils.data_pipeline.takes("noise_wav")
        @sb.utils.data_pipeline.provides("noise_sig")
        def audio_pipeline_noise(noise_wav):
            noise_sig = sb.dataio.dataio.read_audio(noise_wav)
            return noise_sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_mix)
    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_s1)
    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_s2)

    if hparams["use_wham_noise"]:
        print("Using the WHAM! noise in the data pipeline")
        sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_noise)

    if hparams["use_wham_noise"]:
        sb.dataio.dataset.set_output_keys(
            datasets, ["id", "mix_sig", "s1_sig", "s2_sig", "noise_sig"]
        )
    else:
        sb.dataio.dataset.set_output_keys(
            datasets, ["id", "mix_sig", "s1_sig", "s2_sig"]
        )

    return train_data, test_data


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

    # Data preparation
    if "Libri" in hparams["data_folder"]:
        from recipes.LibriMix.prepare_data import (
            prepare_librimix as prepare_wsjmix,
        )  # noqa
    else:
        from recipes.WSJ0Mix.prepare_data import prepare_wsjmix  # noqa

    # create the csv files
    run_on_main(
        prepare_wsjmix,
        kwargs={
            "datapath": hparams["data_folder"],
            "savepath": hparams["save_folder"],
            "n_spks": hparams["num_spks"],
            "skip_prep": hparams["skip_prep"],
            "fs": hparams["sample_rate"],
            "librimix_addnoise": hparams["use_wham_noise"],
        },
    )

    from recipes.WHAMandWHAMR.prepare_data import create_wham_whamr_csv

    create_wham_whamr_csv(
        datapath=hparams["wsj_data_folder"],
        savepath=hparams["save_folder"],
        fs=hparams["sample_rate"],
        add_reverb=False,
        savename="wham_",
        set_types=["tr", "tt"],
    )

    train_data_wham, test_data_wham = dataio_prep_wham_traintest(hparams)

    # if whamr, and we do speedaugment we need to prepare the csv file
    if hparams["use_reverb_augment"]:
        from recipes.WHAMandWHAMR.prepare_data import create_whamr_rir_csv
        from recipes.WHAMandWHAMR.meta.create_whamr_rirs import create_rirs

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
        from recipes.LibriMix.separation.dynamic_mixing import (
            dynamic_mix_data_prep_librimix as dynamic_mix_data_prep,
        )
        from recipes.WSJ0Mix.separation.whamr_dynamic_mixing import (
            dynamic_mix_data_prep as dynamic_mix_data_prep_whamr,
        )

        if hparams["use_whamr_train"]:
            train_data_wham = dynamic_mix_data_prep_whamr(hparams)
            hparams["train_wham_loader"] = it.cycle(iter(train_data_wham))

        # if the base_folder for dm is not processed, preprocess them
        if "processed" not in hparams["base_folder_dm"]:
            # if the processed folder already exists we just use it otherwise we do the preprocessing
            if not os.path.exists(
                os.path.normpath(hparams["base_folder_dm"]) + "_processed"
            ):
                from recipes.LibriMix.meta.preprocess_dynamic_mixing import (
                    resample_folder,
                )

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
        _, valid_data, test_data = dataio_prep(hparams)
    else:
        train_data, valid_data, test_data = dataio_prep(hparams)

    # Load pretrained model if pretrained_separator is present in the yaml
    if "pretrained_separator" in hparams:
        run_on_main(hparams["pretrained_separator"].collect_files)
        hparams["pretrained_separator"].load_collected()

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

    # Eval

    if hparams["test_onwsj"]:
        test_data = test_data_wham
    separator.evaluate(test_data, min_key="si-snr")
    separator.save_results(test_data)
