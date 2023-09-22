#!/usr/bin/env python3
"""script to train a diffwave vocoder
See https://arxiv.org/pdf/2009.09761.pdf for more details

Authors
 * Yingzhi Wang 2022
"""

import torchaudio
import logging
import sys
import torch
import speechbrain as sb
import os
from hyperpyyaml import load_hyperpyyaml

logger = logging.getLogger(__name__)


class DiffWaveBrain(sb.Brain):
    """Class that manages the training loop. See speechbrain.core.Brain."""

    def compute_forward(self, batch, stage):
        """Runs all the computation of that transforms the input into the
        output probabilities over the N classes.
        Arguments
        ---------
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        Returns
        -------
        predictions : Tensor
            Tensor that contains the posterior probabilities over the N classes.
        """

        # We first move the batch to the appropriate device.
        batch = batch.to(self.device)

        x, _ = batch.mel
        y, _ = batch.sig

        pred, noise, noisy_sample = self.modules.diffusion.train_sample(
            y, timesteps=None, condition=x,
        )

        return pred, noise, noisy_sample, None

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss given the predicted and targeted outputs.
        Arguments
        ---------
        predictions : tensor
            The output tensor from `compute_forward`.
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
        x, _ = batch.mel
        y, _ = batch.sig
        self.last_batch = (x, y)
        self._remember_sample(self.last_batch, predictions)

        preds, noise, noisy_sample, lens = predictions

        loss = self.hparams.compute_cost(
            preds.squeeze(1), noise.squeeze(1), length=lens
        )

        self.last_loss_stats[stage] = {"loss": loss}
        return loss

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""
        loss = super().fit_batch(batch)
        return loss.detach().cpu()

    def evaluate_batch(self, batch, stage):
        """Evaluate one batch
        """
        out = self.compute_forward(batch, stage=stage)
        loss = self.compute_objectives(out, batch, stage=stage)
        return loss.detach().cpu()

    def on_fit_start(self):
        """Gets called at the beginning of ``fit()``, on multiple processes
        if ``distributed_count > 0`` and backend is ddp and initializes statistics
        """
        self.last_batch = None
        self.last_loss_stats = {}
        return super().on_fit_start()

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
        pred, noise, noisy_sample, steps = predictions

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a stage (TRAIN, VALID, Or TEST)
        """
        if stage == sb.Stage.VALID:
            lr = self.optimizer.param_groups[0]["lr"]
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch": epoch, "lr": lr},
                train_stats=self.last_loss_stats[sb.Stage.TRAIN],
                valid_stats=self.last_loss_stats[sb.Stage.VALID],
            )
            # The tensorboard_logger writes a summary to stdout and to the logfile.
            if self.hparams.use_tensorboard:
                self.tensorboard_logger.log_stats(
                    stats_meta={"Epoch": epoch, "lr": lr},
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

            if epoch % self.hparams.progress_samples_interval == 0:
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

            sig_out = self.modules.diffusion.inference(
                unconditional=self.hparams.unconditional,
                scale=self.hparams.spec_hop_length,
                condition=x,
                fast_sampling=self.hparams.fast_sampling,
                fast_sampling_noise_schedule=self.hparams.fast_sampling_noise_schedule,
            )

            spec_out = self.hparams.mel_spectogram(
                audio=sig_out.squeeze(1).cpu()
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
            self.save_audio("target", y.squeeze(1), folder)
            self.save_audio("synthesized", sig_out, folder)

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

        # for diffwave the audio length needs to be hop_length * mel_length
        audio_length = mel.shape[-1] * hparams["spec_hop_length"]
        audio = torch.nn.functional.pad(
            audio, (0, audio_length - audio.size(1)), "constant"
        )
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


def check_tensorboard(hparams):
    """Checks whether Tensorboard is enabled and initializes the logger if it is
    Arguments
    ---------
    hparams: dict
        the hyperparameter dictionary
    """
    if hparams["use_tensorboard"]:
        try:
            from speechbrain.utils.train_logger import TensorboardLogger

            hparams["tensorboard_train_logger"] = TensorboardLogger(
                hparams["tensorboard_logs"]
            )
        except ImportError:
            logger.warning(
                "Could not enable TensorBoard logging - TensorBoard is not available"
            )
            hparams["use_tensorboard"] = False


# Recipe begins!
if __name__ == "__main__":

    # Reading command line arguments.
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training).
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides.
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Check whether Tensorboard is available and enabled
    check_tensorboard(hparams)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Create dataset objects "train", "valid", and "test".
    sys.path.append("../../")
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

    # Initialize the Brain object to prepare for mask training.
    diffusion_brain = DiffWaveBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # The `fit()` method iterates the training loop, calling the methods
    # necessary to update the parameters of the model. Since all objects
    # with changing state are managed by the Checkpointer, training can be
    # stopped at any point, and will be resumed on next call.
    diffusion_brain.fit(
        epoch_counter=diffusion_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    # Load the best checkpoint for evaluation
    if "test" in datasets:
        test_stats = diffusion_brain.evaluate(
            test_set=datasets["test"],
            min_key="error",
            test_loader_kwargs=hparams["test_dataloader_opts"],
        )
