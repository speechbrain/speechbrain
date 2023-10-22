#!/usr/bin/env python3
"""Recipe for training a hifi-gan vocoder.
For more details about hifi-gan: https://arxiv.org/pdf/2010.05646.pdf

To run this recipe, do the following:
> python train.py hparams/train.yaml --data_folder /path/to/LJspeech

Authors
 * Duret Jarod 2021
 * Yingzhi WANG 2022
"""

import sys
import torch
import copy
from hyperpyyaml import load_hyperpyyaml
import speechbrain as sb
from speechbrain.utils.data_utils import scalarize
import torchaudio
import os


class HifiGanBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        """The forward function, generates synthesized waveforms,
        calculates the scores and the features of the discriminator
        for synthesized waveforms and real waveforms.

        Arguments
        ---------
        batch: str
            a single batch
        stage: speechbrain.Stage
            the training stage

        """
        batch = batch.to(self.device)
        x, _ = batch.mel
        y, _ = batch.sig

        # generate sythesized waveforms
        y_g_hat = self.modules.generator(x)[:, :, : y.size(2)]

        # get scores and features from discriminator for real and synthesized waveforms
        scores_fake, feats_fake = self.modules.discriminator(y_g_hat.detach())
        scores_real, feats_real = self.modules.discriminator(y)

        return (y_g_hat, scores_fake, feats_fake, scores_real, feats_real)

    def compute_objectives(self, predictions, batch, stage):
        """Computes and combines generator and discriminator losses"""
        batch = batch.to(self.device)
        x, _ = batch.mel
        y, _ = batch.sig

        # Hold on to the batch for the inference sample. This is needed because
        # the infernece sample is run from on_stage_end only, where
        # batch information is not available
        self.last_batch = (x, y)

        # Hold on to a sample (for logging)
        self._remember_sample(self.last_batch, predictions)

        y_hat, scores_fake, feats_fake, scores_real, feats_real = predictions
        loss_g = self.hparams.generator_loss(
            stage, y_hat, y, scores_fake, feats_fake, feats_real
        )
        loss_d = self.hparams.discriminator_loss(scores_fake, scores_real)
        loss = {**loss_g, **loss_d}
        self.last_loss_stats[stage] = scalarize(loss)
        return loss

    def fit_batch(self, batch):
        """Train discriminator and generator adversarially"""

        batch = batch.to(self.device)
        y, _ = batch.sig

        outputs = self.compute_forward(batch, sb.core.Stage.TRAIN)
        (y_g_hat, scores_fake, feats_fake, scores_real, feats_real) = outputs
        # calculate discriminator loss with the latest updated generator
        loss_d = self.compute_objectives(outputs, batch, sb.core.Stage.TRAIN)[
            "D_loss"
        ]
        # First train the discriminator
        self.optimizer_d.zero_grad()
        loss_d.backward()
        self.optimizer_d.step()

        # calculate generator loss with the latest updated discriminator
        scores_fake, feats_fake = self.modules.discriminator(y_g_hat)
        scores_real, feats_real = self.modules.discriminator(y)
        outputs = (y_g_hat, scores_fake, feats_fake, scores_real, feats_real)
        loss_g = self.compute_objectives(outputs, batch, sb.core.Stage.TRAIN)[
            "G_loss"
        ]
        # Then train the generator
        self.optimizer_g.zero_grad()
        loss_g.backward()
        self.optimizer_g.step()

        return loss_g.detach().cpu()

    def evaluate_batch(self, batch, stage):
        """Evaluate one batch"""
        out = self.compute_forward(batch, stage=stage)
        loss = self.compute_objectives(out, batch, stage=stage)
        loss_g = loss["G_loss"]
        return loss_g.detach().cpu()

    def on_fit_start(self):
        """Gets called at the beginning of ``fit()``, on multiple processes
        if ``distributed_count > 0`` and backend is ddp and initializes statistics
        """
        self.last_epoch = 0
        self.last_batch = None
        self.last_loss_stats = {}
        return super().on_fit_start()

    def init_optimizers(self):
        """Called during ``on_fit_start()``, initialize optimizers
        after parameters are fully configured (e.g. DDP, jit).
        """
        if self.opt_class is not None:
            (
                opt_g_class,
                opt_d_class,
                sch_g_class,
                sch_d_class,
            ) = self.opt_class

            self.optimizer_g = opt_g_class(self.modules.generator.parameters())
            self.optimizer_d = opt_d_class(
                self.modules.discriminator.parameters()
            )
            self.scheduler_g = sch_g_class(self.optimizer_g)
            self.scheduler_d = sch_d_class(self.optimizer_d)

            if self.checkpointer is not None:
                self.checkpointer.add_recoverable(
                    "optimizer_g", self.optimizer_g
                )
                self.checkpointer.add_recoverable(
                    "optimizer_d", self.optimizer_d
                )
                self.checkpointer.add_recoverable(
                    "scheduler_g", self.scheduler_d
                )
                self.checkpointer.add_recoverable(
                    "scheduler_d", self.scheduler_d
                )

    def zero_grad(self, set_to_none=False):
        if self.opt_class is not None:
            self.optimizer_g.zero_grad(set_to_none)
            self.optimizer_d.zero_grad(set_to_none)

    def _remember_sample(self, batch, predictions):
        """Remembers samples of spectrograms and the batch for logging purposes

        Arguments
        ---------
        batch: tuple
            a training batch
        predictions: tuple
            predictions (raw output of the Tacotron model)
        """
        mel, sig = batch
        y_hat, scores_fake, feats_fake, scores_real, feats_real = predictions

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a stage (TRAIN, VALID, Or TEST)"""
        if stage == sb.Stage.VALID:
            # Update learning rate
            self.scheduler_g.step()
            self.scheduler_d.step()
            lr_g = self.optimizer_g.param_groups[-1]["lr"]
            lr_d = self.optimizer_d.param_groups[-1]["lr"]

            self.hparams.train_logger.log_stats(  # 1#2#
                stats_meta={"Epoch": epoch, "lr_g": lr_g, "lr_d": lr_d},
                train_stats=self.last_loss_stats[sb.Stage.TRAIN],
                valid_stats=self.last_loss_stats[sb.Stage.VALID],
            )
            # The tensorboard_logger writes a summary to stdout and to the logfile.
            if self.hparams.use_tensorboard:
                self.tensorboard_logger.log_stats(
                    stats_meta={"Epoch": epoch, "lr_g": lr_g, "lr_d": lr_d},
                    train_stats=self.last_loss_stats[sb.Stage.TRAIN],
                    valid_stats=self.last_loss_stats[sb.Stage.VALID],
                )

            # Save the current checkpoint and delete previous checkpoints.
            epoch_metadata = {
                **{"epoch": epoch},
                **self.last_loss_stats[sb.Stage.VALID],
            }
            self.checkpointer.save_and_keep_only(
                meta=epoch_metadata,
                end_of_epoch=True,
                min_keys=["loss"],
                ckpt_predicate=(
                    lambda ckpt: (
                        ckpt.meta["epoch"]
                        % self.hparams.keep_checkpoint_interval
                        != 0
                    )
                )
                if self.hparams.keep_checkpoint_interval is not None
                else None,
            )

            self.run_inference_sample("Valid")

        # We also write statistics about test data to stdout and to the TensorboardLogger.
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(  # 1#2#
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=self.last_loss_stats[sb.Stage.TEST],
            )
            if self.hparams.use_tensorboard:
                self.tensorboard_logger.log_stats(
                    {"Epoch loaded": self.hparams.epoch_counter.current},
                    test_stats=self.last_loss_stats[sb.Stage.TEST],
                )
            self.run_inference_sample("Test")

    def run_inference_sample(self, name):
        """Produces a sample in inference mode. This is called when producing
        samples.
        """
        with torch.no_grad():
            if self.last_batch is None:
                return
            x, y = self.last_batch

            # Preparing model for inference by removing weight norm
            inference_generator = copy.deepcopy(self.hparams.generator)
            inference_generator.remove_weight_norm()
            sig_out = inference_generator.inference(x)
            spec_out = self.hparams.mel_spectogram(
                audio=sig_out.squeeze(0).cpu()
            )
        if self.hparams.use_tensorboard:
            self.tensorboard_logger.log_audio(
                f"{name}/audio_target", y.squeeze(0), self.hparams.sample_rate
            )
            self.tensorboard_logger.log_audio(
                f"{name}/audio_pred",
                sig_out.squeeze(0),
                self.hparams.sample_rate,
            )
            self.tensorboard_logger.log_figure(f"{name}/mel_target", x)
            self.tensorboard_logger.log_figure(f"{name}/mel_pred", spec_out)
        else:
            # folder name is the current epoch for validation and "test" for test
            folder = (
                self.hparams.epoch_counter.current
                if name == "Valid"
                else "test"
            )
            self.save_audio("target", y.squeeze(0), folder)
            self.save_audio("synthesized", sig_out.squeeze(0), folder)

    def save_audio(self, name, data, epoch):
        """Saves a single wav

        Arguments
        ---------
        name: str
            the name of the saved audio
        data: torch.Tensor
            the  wave data to save
        epoch: int or str
            the epoch number (used in file path calculations)
            or "test" for test stage
        """
        target_path = os.path.join(
            self.hparams.progress_sample_path, str(epoch)
        )
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        file_name = f"{name}.wav"
        effective_file_name = os.path.join(target_path, file_name)
        torchaudio.save(effective_file_name, data.cpu(), 22050)


def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    """
    segment_size = hparams["segment_size"]

    # Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav", "segment")
    @sb.utils.data_pipeline.provides("mel", "sig")
    def audio_pipeline(wav, segment):
        audio = sb.dataio.dataio.read_audio(wav)
        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)
        if segment:
            if audio.size(1) >= segment_size:
                max_audio_start = audio.size(1) - segment_size
                audio_start = torch.randint(0, max_audio_start, (1,))
                audio = audio[:, audio_start : audio_start + segment_size]
            else:
                audio = torch.nn.functional.pad(
                    audio, (0, segment_size - audio.size(1)), "constant"
                )

        mel = hparams["mel_spectogram"](audio=audio.squeeze(0))

        return mel, audio

    datasets = {}
    data_info = {
        "train": hparams["train_json"],
        "valid": hparams["valid_json"],
        "test": hparams["test_json"],
    }
    for dataset in hparams["splits"]:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=data_info[dataset],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline],
            output_keys=["id", "mel", "sig"],
        )

    return datasets


if __name__ == "__main__":
    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    from ljspeech_prepare import prepare_ljspeech

    sb.utils.distributed.run_on_main(
        prepare_ljspeech,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["save_folder"],
            "splits": hparams["splits"],
            "split_ratio": hparams["split_ratio"],
            "seed": hparams["seed"],
            "skip_prep": hparams["skip_prep"],
        },
    )

    datasets = dataio_prepare(hparams)

    # Brain class initialization
    hifi_gan_brain = HifiGanBrain(
        modules=hparams["modules"],
        opt_class=[
            hparams["opt_class_generator"],
            hparams["opt_class_discriminator"],
            hparams["sch_class_generator"],
            hparams["sch_class_discriminator"],
        ],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    if hparams["use_tensorboard"]:
        hifi_gan_brain.tensorboard_logger = sb.utils.train_logger.TensorboardLogger(
            save_dir=hparams["output_folder"] + "/tensorboard"
        )

    # Training
    hifi_gan_brain.fit(
        hifi_gan_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    # Test
    if "test" in datasets:
        hifi_gan_brain.evaluate(
            datasets["test"],
            test_loader_kwargs=hparams["test_dataloader_opts"],
        )
