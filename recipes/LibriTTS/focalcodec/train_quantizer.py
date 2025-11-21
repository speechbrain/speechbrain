#!/usr/bin/env/python

"""Recipe for training a quantizer on continuous audio representations.

To run this recipe:
> python train_quantizer.py hparams/<path-to-config>.yaml

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
            self.hparams.encoder.to(self.device).eval()
            feats, *encoder_state_ = self.hparams.encoder(sig, length=lens)

        # Forward model
        lats, *compressor_state_ = self.modules.compressor(feats)
        codes, toks, aux_loss = self.modules.quantizer(lats)
        hyp_feats, *decompressor_state_ = self.modules.decompressor(codes)

        return hyp_feats, feats, aux_loss

    def compute_objectives(self, predictions, batch, stage):
        """Computes the objectives."""
        hyp_feats, feats, aux_loss = predictions

        _, lens = batch.sig

        # Reconstruction loss
        loss = self.hparams.rec_loss(hyp_feats, feats, length=lens)
        loss += aux_loss

        return loss

    def _fit_valid(self, valid_set, epoch, enable):
        if epoch % self.hparams.valid_freq == 0:
            return super()._fit_valid(valid_set, epoch, enable)

    @torch.no_grad()
    def evaluate_batch(self, batch, stage):
        assert stage in (sb.Stage.VALID, sb.Stage.TEST)
        outputs = self.compute_forward(batch, stage=stage)
        loss = self.compute_objectives(outputs, batch, stage=stage)
        hyp_feats, feats, _ = outputs

        IDs = batch.id
        sig, lens = batch.sig

        self.hparams.decoder.to(self.device).eval()
        hyp_sig, *decoder_state_ = self.hparams.decoder(hyp_feats)
        rec_sig, *decoder_state_ = self.hparams.decoder(feats)

        # Adjust length if not matching
        if sig.shape[-1] > hyp_sig.shape[-1]:
            pad = [0, sig.shape[-1] - hyp_sig.shape[-1]]
            hyp_sig = torch.nn.functional.pad(hyp_sig, pad, mode="replicate")
            rec_sig = torch.nn.functional.pad(rec_sig, pad, mode="replicate")
        elif sig.shape[-1] < hyp_sig.shape[-1]:
            hyp_sig = hyp_sig.narrow(-1, 0, sig.shape[-1])
            rec_sig = rec_sig.narrow(-1, 0, sig.shape[-1])

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
                    os.path.join(save_folder, f"{IDs[i]}_rec.wav"),
                    rec_sig[i].cpu(),
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

        return loss.detach().cpu()

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
        current_epoch = self.hparams.epoch_counter.current
        stage_stats = {"loss": stage_loss}

        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
            self.stats_meta = {"epoch": epoch, "steps": self.optimizer_step}
            if epoch % self.hparams.valid_freq != 0:
                self.hparams.train_logger.log_stats(
                    stats_meta=self.stats_meta,
                    train_stats=self.train_stats,
                )

        # Perform end-of-iteration operations, like annealing, logging, etc.
        elif stage == sb.Stage.VALID:
            _, lr = self.hparams.scheduler(stage_stats["loss"])
            sb.nnet.schedulers.update_learning_rate(self.optimizer, lr)
            self.stats_meta["lr"] = lr
            self.hparams.train_logger.log_stats(
                stats_meta=self.stats_meta,
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
    brain = Quantization(
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
        min_key="loss",
        test_loader_kwargs=hparams["test_dataloader_kwargs"],
    )
