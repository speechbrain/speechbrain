#!/usr/bin/env python3
"""Recipe for training a hifi-gan vocoder on self-supervised representations.
For more details about hifi-gan: https://arxiv.org/pdf/2010.05646.pdf
For more details about speech synthesis using self-supervised representations: https://arxiv.org/pdf/2104.00355.pdf

To run this recipe, do the following:
> python train.py hparams/train.yaml --kmeans_folder=/path/to/Kmeans/ckpt --data_folder=/path/to/LJspeech

Authors
 * Jarod Duret 2023
 * Yingzhi WANG 2022
"""

import sys
import copy
import random
import pathlib as pl

from hyperpyyaml import load_hyperpyyaml
import speechbrain as sb
from speechbrain.utils.data_utils import scalarize
import torch
import torchaudio
import numpy as np


class HifiGanBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        """The forward function, generates synthesized waveforms,
        calculates the scores and the features of the discriminator
        for synthesized waveforms and real waveforms.

        Arguments
        ---------
        batch : torch.Tensor or tensors
            An element from the dataloader, including inputs for processing.
        stage : Stage
            The stage of the experiment: Stage.TRAIN, Stage.VALID, Stage.TEST

        """
        batch = batch.to(self.device)

        x, _ = batch.code
        y, _ = batch.sig

        # generate sythesized waveforms
        y_g_hat, (log_dur_pred, log_dur) = self.modules.generator(x)
        y_g_hat = y_g_hat[:, :, : y.size(2)]

        # get scores and features from discriminator for real and synthesized waveforms
        scores_fake, feats_fake = self.modules.discriminator(y_g_hat.detach())
        scores_real, feats_real = self.modules.discriminator(y)

        return (
            y_g_hat,
            scores_fake,
            feats_fake,
            scores_real,
            feats_real,
            log_dur_pred,
            log_dur,
        )

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss given the predicted and targeted outputs.
        Arguments
        ---------
        predictions : torch.Tensor
            The model generated spectrograms and other metrics from `compute_forward`.
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        Returns
        -------
        loss : torch.Tensor
            A one-element tensor used for backpropagating the gradient.
        """
        batch = batch.to(self.device)

        x, _ = batch.code
        y, _ = batch.sig

        # Hold on to the batch for the inference sample. This is needed because
        # the infernece sample is run from on_stage_end only, where
        # batch information is not available
        self.last_batch = (x, y)

        (
            y_hat,
            scores_fake,
            feats_fake,
            scores_real,
            feats_real,
            log_dur_pred,
            log_dur,
        ) = predictions

        loss_g = self.hparams.generator_loss(
            stage,
            y_hat,
            y,
            scores_fake,
            feats_fake,
            feats_real,
            log_dur_pred,
            log_dur,
        )
        loss_d = self.hparams.discriminator_loss(scores_fake, scores_real)
        loss = {**loss_g, **loss_d}
        self.last_loss_stats[stage] = scalarize(loss)
        return loss

    def fit_batch(self, batch):
        """Fits a single batch.
        Arguments
        ---------
        batch: tuple
            a training batch
        Returns
        -------
        loss: torch.Tensor
            detached loss
        """
        batch = batch.to(self.device)
        y, _ = batch.sig

        outputs = self.compute_forward(batch, sb.core.Stage.TRAIN)
        (
            y_g_hat,
            scores_fake,
            feats_fake,
            scores_real,
            feats_real,
            log_dur_pred,
            log_dur,
        ) = outputs
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
        outputs = (
            y_g_hat,
            scores_fake,
            feats_fake,
            scores_real,
            feats_real,
            log_dur_pred,
            log_dur,
        )
        loss_g = self.compute_objectives(outputs, batch, sb.core.Stage.TRAIN)[
            "G_loss"
        ]
        # Then train the generator
        self.optimizer_g.zero_grad()
        loss_g.backward()
        self.optimizer_g.step()

        return loss_g.detach().cpu()

    def evaluate_batch(self, batch, stage):
        """Evaluate one batch.

        Arguments
        ---------
        batch : list of torch.Tensors
            Batch of data to use for evaluation. Default implementation assumes
            this batch has two elements: inputs and targets.
        stage : Stage
            The stage of the experiment: Stage.VALID, Stage.TEST

        Returns
        -------
        detached loss
        """
        out = self.compute_forward(batch, stage=stage)
        loss = self.compute_objectives(out, batch, stage=stage)
        loss_g = loss["G_loss"]
        return loss_g.detach().cpu()

    def on_fit_start(self):
        """Gets called at the beginning of ``fit()``, on multiple processes
        if ``distributed_count > 0`` and backend is ddp and initializes statistics.
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

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all of the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """
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
            if self.checkpointer is not None:
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

            self.run_inference_sample("Valid", epoch)

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

    def run_inference_sample(self, name, epoch):
        """Produces a sample in inference mode.
        This is called when producing samples.

        Arguments
        ---------
        name: str
            the name of the saved audio folder
        epoch: int or str
            the epoch number (used in file path calculations)
            or "test" for test stage
        """
        with torch.no_grad():
            if self.last_batch is None:
                return
            x, y = self.last_batch

            # Preparing model for inference by removing weight norm
            inference_generator = copy.deepcopy(self.hparams.generator)
            inference_generator.remove_weight_norm()
            if inference_generator.duration_predictor:
                x = torch.unique_consecutive(x, dim=1)
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
        """Saves a single wav file.

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
        target_path = pl.Path(self.hparams.progress_sample_path) / str(epoch)
        target_path.mkdir(parents=True, exist_ok=True)
        file_name = target_path / f"{name}.wav"
        torchaudio.save(file_name, data.cpu(), 16000)


def sample_interval(seqs, segment_size):
    "This function sample an interval of audio and code according to segment size."
    N = max([v.shape[-1] for v in seqs])
    seq_len = segment_size if segment_size > 0 else N
    hops = [N // v.shape[-1] for v in seqs]
    lcm = np.lcm.reduce(hops)
    interval_start = 0
    interval_end = N // lcm - seq_len // lcm
    start_step = random.randint(interval_start, interval_end)

    new_seqs = []
    for i, v in enumerate(seqs):
        start = start_step * (lcm // hops[i])
        end = (start_step + seq_len // lcm) * (lcm // hops[i])
        new_seqs += [v[..., start:end]]

    return new_seqs


def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    """
    segment_size = hparams["segment_size"]
    code_hop_size = hparams["code_hop_size"]
    code_folder = pl.Path(hparams["codes_folder"])

    # Define audio pipeline:
    @sb.utils.data_pipeline.takes("id", "wav", "segment")
    @sb.utils.data_pipeline.provides("code", "sig")
    def audio_pipeline(utt_id, wav, segment):
        info = torchaudio.info(wav)
        audio = sb.dataio.dataio.read_audio(wav)
        audio = torchaudio.transforms.Resample(
            info.sample_rate, hparams["sample_rate"],
        )(audio)

        code = np.load(code_folder / f"{utt_id}.npy")
        code = torch.IntTensor(code)

        # Maps indices from the range [0, k] to [1, k+1]
        code = code + 1

        # Trim end of audio
        code_length = min(audio.shape[0] // code_hop_size, code.shape[0])
        code = code[:code_length]
        audio = audio[: code_length * code_hop_size]

        while audio.shape[0] < segment_size:
            audio = torch.hstack([audio, audio])
            code = torch.hstack([code, code])

        audio = audio.unsqueeze(0)
        if segment:
            audio, code = sample_interval([audio, code], segment_size)

        return code, audio

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
            output_keys=["id", "code", "sig"],
        )

    return datasets


if __name__ == "__main__":
    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # If --distributed_launch then
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

    from extract_code import extract_ljspeech

    sb.utils.distributed.run_on_main(
        extract_ljspeech,
        kwargs={
            "data_folder": hparams["save_folder"],
            "splits": hparams["splits"],
            "kmeans_folder": hparams["kmeans_folder"],
            "encoder": hparams["encoder_hub"],
            "layer": hparams["layer"],
            "save_folder": hparams["save_folder"],
            "sample_rate": hparams["sample_rate"],
            "skip_extract": hparams["skip_prep"],
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
