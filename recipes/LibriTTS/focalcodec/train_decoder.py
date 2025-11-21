#!/usr/bin/env/python

"""Recipe for training a decoder from continuous audio representations to waveform.

To run this recipe:
> python train_decoder.py hparams/<path-to-config>.yaml

Authors
 * Luca Della Libera 2025
"""

import os
import sys

import torch
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.dataio.dataio import write_audio
from speechbrain.utils.distributed import if_main_process


class Generation(sb.Brain):
    def fit_batch(self, batch):
        amp = sb.core.AMPConfig.from_name(self.precision)

        # Train discriminator
        with torch.autocast(
            device_type=torch.device(self.device).type,
            dtype=amp.dtype,
            enabled=self.precision != torch.float32,
        ):
            self.extract_feats(batch, sb.Stage.TRAIN)
            self.compute_forward_generator(batch, sb.Stage.TRAIN)
            outputs = self.compute_forward_discriminator(batch, sb.Stage.TRAIN)
            loss_discriminator = self.compute_objectives_discriminator(
                outputs, batch, sb.Stage.TRAIN
            )

        self.scaler.scale(loss_discriminator).backward()
        self.scaler.step(self.optimizer_discriminator)
        self.optimizer_discriminator.zero_grad(set_to_none=True)
        del loss_discriminator

        # Train generator
        with torch.autocast(
            device_type=torch.device(self.device).type,
            dtype=amp.dtype,
            enabled=self.precision != torch.float32,
        ):
            outputs = self.compute_forward_discriminator(
                batch, sb.Stage.TRAIN, return_discriminator=False
            )
            loss_generator = self.compute_objectives_generator(
                outputs, batch, sb.Stage.TRAIN
            )

        self.scaler.scale(loss_generator).backward()
        self.scaler.step(self.optimizer_generator)
        self.optimizer_generator.zero_grad(set_to_none=True)

        self.scaler.update()
        self.optimizer_step += 1
        self.step = self.optimizer_step

        # Cleanup
        batch.sig = batch.hyp_sig = None

        return loss_generator.detach().cpu()

    def extract_feats(self, batch, stage):
        batch = batch.to(self.device)
        sig, lens = batch.sig

        # Augment if specified
        if stage == sb.Stage.TRAIN and self.hparams.augment:
            sig, lens = self.hparams.augmentation(sig, lens)

        # Extract features
        with torch.no_grad():
            self.hparams.encoder.to(self.device).eval()
            feats, *encoder_state_ = self.hparams.encoder(sig, length=lens)

        # Extract segments
        if (
            stage == sb.Stage.TRAIN
            and self.hparams.segment_size_feats is not None
        ):
            segment_size_feats = self.hparams.segment_size_feats
            abs_lens = (
                (feats.shape[1] * lens)
                .ceil()
                .clamp(min=segment_size_feats, max=feats.shape[1])
                .long()
            )
            max_starts = abs_lens - segment_size_feats  # [B]
            starts = (
                torch.rand(feats.shape[0], device=self.device)
                * (max_starts + 1).float()
            ).to(torch.long)
            offsets = torch.arange(
                segment_size_feats, device=self.device
            )  # [L]
            idx = starts[:, None] + offsets[None, :]  # [B, L]
            idx_expanded = idx[:, :, None].expand(-1, -1, feats.shape[-1])
            feats = feats.gather(1, idx_expanded)  # [B, L, H]

            segment_size_sig = (
                segment_size_feats * self.hparams.generator_hop_length
            )
            starts = starts * self.hparams.generator_hop_length  # [B]
            offsets = torch.arange(
                segment_size_sig, device=self.device
            )  # [L_sig]
            idx = starts[:, None] + offsets[None, :]  # [B, L_sig]
            idx = idx.clamp(max=sig.shape[1] - 1).long()
            sig = sig.gather(1, idx)
            lens = torch.ones_like(lens)

        batch.sig = sig, lens
        batch.feats = feats, lens

    def compute_forward_generator(self, batch, stage):
        sig, lens = batch.sig

        # Forward generator
        feats, _ = batch.feats
        hyp_sig, *generator_state_ = self.modules.generator(feats)  # [B, T]
        hyp_sig = hyp_sig[:, None]  # [B, 1, T]

        # Adjust length if not matching
        sig = sig[:, None]
        if sig.shape[-1] > hyp_sig.shape[-1]:
            pad = [0, sig.shape[-1] - hyp_sig.shape[-1]]
            hyp_sig = torch.nn.functional.pad(hyp_sig, pad, mode="replicate")
        elif sig.shape[-1] < hyp_sig.shape[-1]:
            hyp_sig = hyp_sig.narrow(-1, 0, sig.shape[-1])

        batch.sig = sig, lens
        batch.hyp_sig = hyp_sig, lens  # With gradient

    def compute_forward_discriminator(
        self, batch, stage, return_discriminator=True
    ):
        sig, lens = batch.sig
        hyp_sig, _ = batch.hyp_sig  # With gradient

        if return_discriminator:
            # Return predictions to compute discriminator loss
            scores_fake, _ = self.modules.discriminator(hyp_sig.detach())
            scores_real, _ = self.modules.discriminator(sig)
            return scores_fake, scores_real

        # Return predictions to compute generator loss
        self.modules.discriminator.requires_grad_(False)
        scores_fake, feats_fake = self.modules.discriminator(hyp_sig)
        scores_real, feats_real = self.modules.discriminator(sig)
        self.modules.discriminator.requires_grad_()

        return hyp_sig, sig, scores_fake, feats_fake, feats_real

    def compute_objectives_generator(self, predictions, batch, stage):
        loss = self.hparams.generator_loss(
            stage,
            y_hat=predictions[0],
            y=predictions[1],
            scores_fake=predictions[2],
            feats_fake=predictions[3],
            feats_real=predictions[4],
        )
        return loss["G_loss"]

    def compute_objectives_discriminator(self, predictions, batch, stage):
        loss = self.hparams.discriminator_loss(
            scores_fake=predictions[0],
            scores_real=predictions[1],
        )
        return loss["D_loss"]

    def _fit_valid(self, valid_set, epoch, enable):
        if epoch % self.hparams.valid_freq == 0:
            return super()._fit_valid(valid_set, epoch, enable)

    @torch.no_grad()
    def evaluate_batch(self, batch, stage):
        assert stage in (sb.Stage.VALID, sb.Stage.TEST)
        self.extract_feats(batch, stage)
        self.compute_forward_generator(batch, stage)
        outputs = self.compute_forward_discriminator(
            batch, stage, return_discriminator=False
        )
        loss = self.compute_objectives_generator(outputs, batch, stage)

        IDs = batch.id
        _, lens = batch.sig
        hyp_sig, sig, *_ = outputs
        hyp_sig = hyp_sig[:, 0]
        sig = sig[:, 0]

        if (
            self.hparams.save_audios
            and self.saved_audios < 10
            and if_main_process()
        ):
            save_folder = os.path.join(
                self.hparams.output_folder,
                "audios",
                f"epoch{str(self.hparams.epoch_counter.current).zfill(4)}",
            )
            os.makedirs(save_folder, exist_ok=True)
            for i in range(len(IDs)):
                write_audio(
                    os.path.join(save_folder, f"{IDs[i]}_hyp.wav"),
                    hyp_sig[i].cpu(),
                    self.hparams.sample_rate,
                )
                write_audio(
                    os.path.join(save_folder, f"{IDs[i]}_ref.wav"),
                    sig[i].cpu(),
                    self.hparams.sample_rate,
                )
                self.saved_audios += 1

        if stage == sb.Stage.TEST and self.hparams.compute_metrics:
            self.utmos_metric.append(IDs, hyp_sig, lens)
            self.ref_utmos_metric.append(IDs, sig, lens)
            self.dwer_metric.append(IDs, hyp_sig, sig, lens)
            self.wavlm_sim_metric.append(IDs, hyp_sig, sig, lens)

        # Cleanup
        batch.sig = batch.hyp_sig = None

        return loss.detach().cpu()

    def init_optimizers(self):
        """Called during ``on_fit_start().``"""
        self.optimizer_generator = self.opt_class(
            self.modules.generator.parameters()
        )
        self.optimizer_discriminator = self.opt_class(
            self.modules.discriminator.parameters()
        )
        self.optimizers_dict = {
            "optimizer_generator": self.optimizer_generator,
            "optimizer_discriminator": self.optimizer_discriminator,
        }
        if self.checkpointer is not None:
            self.checkpointer.add_recoverable(
                "optimizer_generator", self.optimizer_generator
            )
            self.checkpointer.add_recoverable(
                "optimizer_discriminator", self.optimizer_discriminator
            )

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch."""
        super().on_stage_start(stage, epoch)
        torch.backends.cudnn.benchmark = (
            stage == sb.Stage.TRAIN
            and self.hparams.segment_size is not None
            and self.hparams.segment_pad
        )
        if stage != sb.Stage.TRAIN:
            self.saved_audios = 0
        if stage == sb.Stage.TEST and self.hparams.compute_metrics:
            self.utmos_metric = self.hparams.utmos_computer()
            self.ref_utmos_metric = self.hparams.utmos_computer(
                model=self.utmos_metric.model
            )
            self.dwer_metric = self.hparams.dwer_computer()
            self.wavlm_sim_metric = self.hparams.wavlm_sim_computer()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of each epoch."""
        # Compute/store important stats
        current_epoch = self.hparams.epoch_counter.current
        stage_stats = {"loss": stage_loss}

        # Save checkpoint and anneal learning rate at the end of each epoch
        if stage == sb.Stage.TRAIN:
            self.avg_train_loss = 0.0
            self.train_stats = stage_stats
            _, lr = self.hparams.scheduler(epoch)
            sb.nnet.schedulers.update_learning_rate(
                self.optimizer_generator, lr
            )
            sb.nnet.schedulers.update_learning_rate(
                self.optimizer_discriminator, lr
            )
            self.stats_meta = {
                "epoch": epoch,
                "steps": self.optimizer_step,
                "lr": lr,
            }
            if if_main_process():
                self.checkpointer.save_and_keep_only(
                    meta={"loss": stage_stats["loss"], "epoch": epoch},
                    max_keys=["epoch"],
                    num_to_keep=self.hparams.keep_checkpoints,
                )
            if epoch % self.hparams.valid_freq != 0:
                self.hparams.train_logger.log_stats(
                    stats_meta=self.stats_meta,
                    train_stats=self.train_stats,
                )

        # Perform end-of-validation operations
        elif stage == sb.Stage.VALID:
            self.hparams.train_logger.log_stats(
                stats_meta=self.stats_meta,
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )

        elif stage == sb.Stage.TEST:
            if self.hparams.compute_metrics:
                stage_stats["UTMOS"] = self.utmos_metric.summarize("average")
                stage_stats["RefUTMOS"] = self.ref_utmos_metric.summarize(
                    "average"
                )
                stage_stats["dWER"] = self.dwer_metric.summarize("error_rate")
                stage_stats["dCER"] = self.dwer_metric.summarize(
                    "error_rate_char"
                )
                stage_stats["WavLMSim"] = self.wavlm_sim_metric.summarize(
                    "average"
                )
                if if_main_process():
                    # Save dWER
                    with open(self.hparams.dwer_file, "w") as w:
                        self.dwer_metric.write_stats(w)
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": current_epoch},
                test_stats=stage_stats,
            )


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

    # Prepare recipe
    from utils import prepare_recipe

    hparams, train_data, valid_data, test_data = prepare_recipe(
        hparams, run_opts
    )

    # Trainer initialization
    brain = Generation(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Train
    brain.fit(
        brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_dataloader_kwargs"],
        valid_loader_kwargs=hparams["valid_dataloader_kwargs"],
    )

    # Test
    brain.hparams.dwer_file = os.path.join(hparams["output_folder"], "dwer.txt")
    brain.evaluate(
        test_data,
        max_key="epoch",
        test_loader_kwargs=hparams["test_dataloader_kwargs"],
    )
