#!/usr/bin/env/python3
"""Recipe for training a neural speech separation system on Microsoft DNS
(Deep Noise Suppression) dataset challenge. The system employs an encoder,
a decoder, and a masking network.

To run this recipe, do the following:
> python train.py hparams/sepformer-wham.yaml --data_folder /your_path/dns_dataset

The experiment file is flexible enough to support different neural
networks. By properly changing the parameter files, you can try
different architectures.

Authors
 * Sangeet Sagar 2022
 * Cem Subakan 2020
 * Mirco Ravanelli 2020
 * Samuele Cornell 2020
 * Mirko Bronzi 2020
 * Jianyuan Zhong 2020
"""

import os
import sys
import csv
import logging
import numpy as np
from tqdm import tqdm

import torch
import torchaudio
import torch.nn.functional as F
from torch.cuda.amp import autocast

import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from composite_eval import eval_composite
import speechbrain.nnet.schedulers as schedulers
from speechbrain.utils.distributed import run_on_main
from speechbrain.utils.metric_stats import MetricStats
from speechbrain.processing.features import spectral_magnitude

from pesq import pesq
from pystoi import stoi

# import pdb
# from pprint import pprint


# Define training procedure
class Separation(sb.Brain):
    def compute_forward(self, noisy, clean, stage, noise=None):
        """Forward computations from the noisy to the separated signals."""

        # Unpack lists and put tensors in the right device
        noisy, noisy_lens = noisy
        noisy, noisy_lens = noisy.to(self.device), noisy_lens.to(self.device)
        # Convert clean to tensor
        clean = clean[0].unsqueeze(-1).to(self.device)

        # Add speech distortions
        if stage == sb.Stage.TRAIN:
            with torch.no_grad():
                if self.hparams.use_speedperturb or self.hparams.use_rand_shift:
                    noisy, clean = self.add_speed_perturb(clean, noisy_lens)

                    # Reverb already added, not adding any reverb
                    clean_rev = clean
                    noisy = clean.sum(-1)
                    # if we reverberate, we set the clean to be reverberant
                    if not self.hparams.dereverberate:
                        clean = clean_rev

                    noise = noise.to(self.device)
                    len_noise = noise.shape[1]
                    len_noisy = noisy.shape[1]
                    min_len = min(len_noise, len_noisy)

                    # add the noise
                    noisy = noisy[:, :min_len] + noise[:, :min_len]

                    # fix the length of clean also
                    clean = clean[:, :min_len, :]
                    
                if self.hparams.use_wavedrop:
                    noisy = self.hparams.wavedrop(noisy, noisy_lens)

                if self.hparams.limit_training_signal_len:
                    noisy, clean = self.cut_signals(noisy, clean)

        # Separation
        if self.use_freq_domain:
            noisy_w = self.compute_feats(noisy)
            est_mask = self.modules.masknet(noisy_w)

            sep_h = noisy_w * est_mask
            est_source = self.hparams.resynth(torch.expm1(sep_h), noisy)
        else:
            noisy_w = self.hparams.Encoder(noisy)
            est_mask = self.modules.masknet(noisy_w)

            sep_h = noisy_w * est_mask
            est_source = self.hparams.Decoder(sep_h[0])

        # T changed after conv1d in encoder, fix it here
        T_origin = noisy.size(1)
        T_est = est_source.size(1)
        est_source = est_source.squeeze(-1)
        if T_origin > T_est:
            est_source = F.pad(est_source, (0, T_origin - T_est))
        else:
            est_source = est_source[:, :T_origin]

        return [est_source, sep_h], clean.squeeze(-1)

    def compute_feats(self, wavs):
        """Feature computation pipeline"""
        feats = self.hparams.Encoder(wavs)
        feats = spectral_magnitude(feats, power=0.5)
        feats = torch.log1p(feats)
        return feats

    def compute_objectives(self, predictions, clean):
        """Computes the si-snr loss"""
        predicted_wavs, predicted_specs = predictions

        if self.use_freq_domain:
            target_specs = self.compute_feats(clean)
            return self.hparams.loss(target_specs, predicted_specs)
        else:
            return self.hparams.loss(
                clean.unsqueeze(-1), predicted_wavs.unsqueeze(-1)
            )

    def fit_batch(self, batch):
        """Trains one batch"""
        # Unpacking batch list
        noisy = batch.noisy_sig
        clean = batch.clean_sig
        noise = batch.noise_sig[0]

        if self.auto_mix_prec:
            with autocast():
                predictions, clean = self.compute_forward(
                    noisy, clean, sb.Stage.TRAIN, noise
                )
                loss = self.compute_objectives(predictions, clean)

                # hard threshold the easy dataitems
                if self.hparams.threshold_byloss:
                    th = self.hparams.threshold
                    loss_to_keep = loss[loss > th]
                    if loss_to_keep.nelement() > 0:
                        loss = loss_to_keep.mean()
                else:
                    loss = loss.mean()

            if (
                loss < self.hparams.loss_upper_lim and loss.nelement() > 0
            ):  # the fix for computational problems
                self.scaler.scale(loss).backward()
                if self.hparams.clip_grad_norm >= 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.modules.parameters(), self.hparams.clip_grad_norm,
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
            predictions, clean = self.compute_forward(
                noisy, clean, sb.Stage.TRAIN, noise
            )
            loss = self.compute_objectives(predictions, clean)

            if self.hparams.threshold_byloss:
                th = self.hparams.threshold
                loss_to_keep = loss[loss > th]
                if loss_to_keep.nelement() > 0:
                    loss = loss_to_keep.mean()
            else:
                loss = loss.mean()

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
        noisy = batch.noisy_sig
        clean = batch.clean_sig

        with torch.no_grad():
            predictions, clean = self.compute_forward(noisy, clean, stage)
            loss = self.compute_objectives(predictions, clean)
            loss = torch.mean(loss)

        if stage != sb.Stage.TRAIN:
            self.pesq_metric.append(
                ids=batch.id, predict=predictions[0].cpu(), target=clean.cpu()
            )

        # Manage audio file saving
        if stage == sb.Stage.TEST and self.hparams.save_audio:
            if hasattr(self.hparams, "n_audio_to_save"):
                if self.hparams.n_audio_to_save > 0:
                    self.save_audio(snt_id[0], noisy, clean, predictions[0])
                    self.hparams.n_audio_to_save += -1
            else:
                self.save_audio(snt_id[0], noisy, clean, predictions[0])

        return loss.detach()

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            # Define function taking (prediction, target) for parallel eval
            def pesq_eval(pred_wav, target_wav):
                """Computes the PESQ evaluation metric"""
                psq_mode = "wb" if self.hparams.sample_rate == 16000 else "nb"
                try:
                    return pesq(
                        fs=self.hparams.sample_rate,
                        ref=target_wav.numpy(),
                        deg=pred_wav.numpy(),
                        mode=psq_mode,
                    )
                except Exception:
                    print("pesq encountered an error for this data item")
                    return 0

            self.pesq_metric = MetricStats(
                metric=pesq_eval, n_jobs=1, batch_eval=False
            )

    def on_stage_end(self, stage, stage_loss, epoch, tr_time):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {"si-snr": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stats = {
                "si-snr": stage_loss,
                "pesq": self.pesq_metric.summarize("average"),
            }

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:

            # Save valid logs in TensorBoard
            valid_stats = {
                "Epochs": epoch,
                "Valid SI-SNR": stage_loss,
                "Valid PESQ": self.pesq_metric.summarize("average"),
            }
            if self.hparams.use_tensorboard:
                self.hparams.tensorboard_train_logger.log_stats(valid_stats)

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
                stats_meta={"epoch": epoch, "time": tr_time, "lr": current_lr},
                train_stats=self.train_stats,
                valid_stats=stats,
            )
            if (
                hasattr(self.hparams, "save_all_checkpoints")
                and self.hparams.save_all_checkpoints
            ):
                self.checkpointer.save_checkpoint(meta={"pesq": stats["pesq"]})
            else:
                self.checkpointer.save_and_keep_only(
                    meta={"pesq": stats["pesq"]}, max_keys=["pesq"],
                )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )

    def add_speed_perturb(self, clean, targ_lens):
        """Adds speed perturbation and random_shift to the input signals"""

        min_len = -1
        recombine = False

        if self.hparams.use_speedperturb:
            # Performing speed change (independently on each source)
            new_clean = []
            recombine = True

            for i in range(clean.shape[-1]):
                new_target = self.hparams.speedperturb(
                    clean[:, :, i], targ_lens
                )
                new_clean.append(new_target)
                if i == 0:
                    min_len = new_target.shape[-1]
                else:
                    if new_target.shape[-1] < min_len:
                        min_len = new_target.shape[-1]

            if self.hparams.use_rand_shift:
                # Performing random_shift (independently on each source)
                recombine = True
                for i in range(clean.shape[-1]):
                    rand_shift = torch.randint(
                        self.hparams.min_shift, self.hparams.max_shift, (1,)
                    )
                    new_clean[i] = new_clean[i].to(self.device)
                    new_clean[i] = torch.roll(
                        new_clean[i], shifts=(rand_shift[0],), dims=1
                    )

            # Re-combination
            if recombine:
                if self.hparams.use_speedperturb:
                    clean = torch.zeros(
                        clean.shape[0],
                        min_len,
                        clean.shape[-1],
                        device=clean.device,
                        dtype=torch.float,
                    )
                for i, new_target in enumerate(new_clean):
                    clean[:, :, i] = new_clean[i][:, 0:min_len]

        noisy = clean.sum(-1)
        return noisy, clean

    def cut_signals(self, noisy, clean):
        """This function selects a random segment of a given length withing the noisy.
        The corresponding clean are selected accordingly"""
        randstart = torch.randint(
            0,
            1 + max(0, noisy.shape[1] - self.hparams.training_signal_len),
            (1,),
        ).item()
        clean = clean[
            :, randstart : randstart + self.hparams.training_signal_len, :
        ]
        noisy = noisy[
            :, randstart : randstart + self.hparams.training_signal_len
        ]
        return noisy, clean

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
        from mir_eval.separation import bss_eval_sources

        # Create folders where to store audio
        save_file = os.path.join(self.hparams.output_folder, "test_results.csv")

        # Variable init
        all_sdrs = []
        all_sdrs_i = []
        all_sisnrs = []
        all_sisnrs_i = []
        all_pesqs = []
        all_stois = []
        all_csigs = []
        all_cbaks = []
        all_covls = []
        csv_columns = [
            "snt_id",
            "sdr",
            "sdr_i",
            "si-snr",
            "si-snr_i",
            "pesq",
            "stoi",
            "csig",
            "cbak",
            "covl",
        ]

        test_loader = sb.dataio.dataloader.make_dataloader(
            test_data, **self.hparams.dataloader_opts_test
        )

        with open(save_file, "w") as results_csv:
            writer = csv.DictWriter(results_csv, fieldnames=csv_columns)
            writer.writeheader()

            # Loop over all test sentence
            with tqdm(test_loader, dynamic_ncols=True) as t:
                for i, batch in enumerate(t):
                    # Apply Separation
                    noisy, noisy_len = batch.noisy_sig
                    snt_id = batch.id
                    clean = batch.clean_sig

                    with torch.no_grad():
                        predictions, clean = self.compute_forward(
                            batch.noisy_sig, clean, sb.Stage.TEST
                        )

                    # Compute PESQ
                    psq_mode = (
                        "wb" if self.hparams.sample_rate == 16000 else "nb"
                    )

                    try:
                        # Compute SI-SNR
                        sisnr = self.compute_objectives(predictions, clean)

                        # Compute SI-SNR improvement
                        noisy_signal = noisy

                        noisy_signal = noisy_signal.to(clean.device)
                        sisnr_baseline = self.compute_objectives(
                            [noisy_signal.squeeze(-1), None], clean
                        )
                        sisnr_i = sisnr - sisnr_baseline

                        # Compute SDR
                        sdr, _, _, _ = bss_eval_sources(
                            clean[0].t().cpu().numpy(),
                            predictions[0][0].t().detach().cpu().numpy(),
                        )

                        sdr_baseline, _, _, _ = bss_eval_sources(
                            clean[0].t().cpu().numpy(),
                            noisy_signal[0].t().detach().cpu().numpy(),
                        )

                        sdr_i = sdr.mean() - sdr_baseline.mean()

                        # Compute PESQ
                        psq = pesq(
                            self.hparams.sample_rate,
                            clean.squeeze().cpu().numpy(),
                            predictions[0].squeeze().cpu().numpy(),
                            mode=psq_mode,
                        )
                        # Compute STOI
                        stoi_score = stoi(
                            clean.squeeze().cpu().numpy(),
                            predictions[0].squeeze().cpu().numpy(),
                            fs_sig=self.hparams.sample_rate,
                            extended=False,
                        )
                        # Compute CSIG, CBAK, COVL
                        composite_metrics = eval_composite(
                            clean.squeeze().cpu().numpy(),
                            predictions[0].squeeze().cpu().numpy(),
                            self.hparams.sample_rate,
                        )
                    except Exception:
                        # this handles all sorts of error that may
                        # occur when evaluating an enhanced file.
                        continue

                    # Saving on a csv file
                    row = {
                        "snt_id": snt_id[0],
                        "sdr": sdr.mean(),
                        "sdr_i": sdr_i,
                        "si-snr": -sisnr.item(),
                        "si-snr_i": -sisnr_i.item(),
                        "pesq": psq,
                        "stoi": stoi_score,
                        "csig": composite_metrics["csig"],
                        "cbak": composite_metrics["cbak"],
                        "covl": composite_metrics["covl"],
                    }
                    writer.writerow(row)

                    # Metric Accumulation
                    all_sdrs.append(sdr.mean())
                    all_sdrs_i.append(sdr_i.mean())
                    all_sisnrs.append(-sisnr.item())
                    all_sisnrs_i.append(-sisnr_i.item())
                    all_pesqs.append(psq)
                    all_stois.append(stoi_score)
                    all_csigs.append(composite_metrics["csig"])
                    all_cbaks.append(composite_metrics["cbak"])
                    all_covls.append(composite_metrics["covl"])

                row = {
                    "snt_id": "avg",
                    "sdr": np.array(all_sdrs).mean(),
                    "sdr_i": np.array(all_sdrs_i).mean(),
                    "si-snr": np.array(all_sisnrs).mean(),
                    "si-snr_i": np.array(all_sisnrs_i).mean(),
                    "pesq": np.array(all_pesqs).mean(),
                    "stoi": np.array(all_stois).mean(),
                    "csig": np.array(all_csigs).mean(),
                    "cbak": np.array(all_cbaks).mean(),
                    "covl": np.array(all_covls).mean(),
                }
                writer.writerow(row)

        logger.info("Mean SISNR is {}".format(np.array(all_sisnrs).mean()))
        logger.info("Mean SISNRi is {}".format(np.array(all_sisnrs_i).mean()))
        logger.info("Mean SDR is {}".format(np.array(all_sdrs).mean()))
        logger.info("Mean SDRi is {}".format(np.array(all_sdrs_i).mean()))
        logger.info("Mean PESQ {}".format(np.array(all_pesqs).mean()))
        logger.info("Mean STOI {}".format(np.array(all_stois).mean()))
        logger.info("Mean CSIG {}".format(np.array(all_csigs).mean()))
        logger.info("Mean CBAK {}".format(np.array(all_cbaks).mean()))
        logger.info("Mean COVL {}".format(np.array(all_covls).mean()))

    def save_audio(self, snt_id, noisy, clean, predictions):
        "saves the test audio (noisy, clean, and estimated sources) on disk"
        print("Saving enhanced sources")
        # Create outout folder
        save_path = os.path.join(self.hparams.save_folder, "audio_results")
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        # Estimated source
        signal = predictions[0, :]
        signal = signal / signal.abs().max()
        save_file = os.path.join(
            save_path, "item{}_sourcehat.wav".format(snt_id)
        )
        torchaudio.save(
            save_file, signal.unsqueeze(0).cpu(), self.hparams.sample_rate
        )

        # Original source
        signal = clean[0, :]
        signal = signal / signal.abs().max()
        save_file = os.path.join(save_path, "item{}_source.wav".format(snt_id))
        torchaudio.save(
            save_file, signal.unsqueeze(0).cpu(), self.hparams.sample_rate
        )

        # noisy
        signal = noisy[0][0, :]
        signal = signal / signal.abs().max()
        save_file = os.path.join(save_path, "item{}_noisy.wav".format(snt_id))
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

    # Shuffle training data.
    hparams["dataloader_opts"]["shuffle"] = hparams["shuffle_train_data"]

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

    @sb.utils.data_pipeline.takes("clean_wav")
    @sb.utils.data_pipeline.provides("clean_wav", "clean_sig")
    def audio_pipeline_clean(clean_wav):
        clean_sig = sb.dataio.dataio.read_audio(clean_wav)
        if hparams["downsample"]:
            info = torchaudio.info(clean_wav)
            clean_sig = torchaudio.transforms.Resample(
                info.sample_rate, hparams["sample_rate"],
            )(clean_sig)
        return clean_wav, clean_sig

    @sb.utils.data_pipeline.takes("noise_wav")
    @sb.utils.data_pipeline.provides("noise_wav", "noise_sig")
    def audio_pipeline_noise(noise_wav):
        noise_sig = sb.dataio.dataio.read_audio(noise_wav)
        if hparams["downsample"]:
            info = torchaudio.info(noise_wav)
            noise_sig = torchaudio.transforms.Resample(
                info.sample_rate, hparams["sample_rate"],
            )(noise_sig)
        return noise_wav, noise_sig

    @sb.utils.data_pipeline.takes("noisy_wav")
    @sb.utils.data_pipeline.provides("noisy_wav", "noisy_sig")
    def audio_pipeline_noisy(noisy_wav):
        noisy_sig = sb.dataio.dataio.read_audio(noisy_wav)
        if hparams["downsample"]:
            info = torchaudio.info(noisy_wav)
            noisy_sig = torchaudio.transforms.Resample(
                info.sample_rate, hparams["sample_rate"],
            )(noisy_sig)
        return noisy_wav, noisy_sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_clean)
    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_noise)
    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_noisy)

    sb.dataio.dataset.set_output_keys(
        datasets,
        [
            "id",
            "clean_wav",
            "clean_sig",
            "noise_wav",
            "noise_sig",
            "noisy_wav",
            "noisy_sig",
        ],
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

    if hparams["use_tensorboard"]:
        from speechbrain.utils.train_logger import TensorboardLogger

        hparams["tensorboard_train_logger"] = TensorboardLogger(
            hparams["tensorboard_logs"]
        )

    # Data preparation
    from prepare_dns import prepare_dns_csv

    run_on_main(
        prepare_dns_csv,
        kwargs={
            "datapath": hparams["data_folder"],
            "baseline_noisy_datapath": hparams["baseline_noisy_folder"],
            "baseline_enhanced_datapath": hparams["baseline_enhanced_folder"],
            "savepath": hparams["save_folder"],
            "skip_prep": hparams["skip_prep"],
            "fs": hparams["sample_rate"],
        },
    )

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

    # determine if frequency domain enhancement or not
    use_freq_domain = hparams.get("use_freq_domain", False)
    separator.use_freq_domain = use_freq_domain

    if not hparams["test_only"]:
        # Training
        separator.fit(
            separator.hparams.epoch_counter,
            train_data,
            valid_data,
            train_loader_kwargs=hparams["dataloader_opts"],
            valid_loader_kwargs=hparams["dataloader_opts_valid"],
        )

    # Eval
    separator.evaluate(test_data, max_key="pesq")
    separator.save_results(test_data)
