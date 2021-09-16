
import os
from speechbrain.nnet.normalization import BatchNorm2d
import torch
import speechbrain as sb
import sys
import logging
from hyperpyyaml import load_hyperpyyaml
from speechbrain.dataio.dataloader import SaveableDataLoader
from speechbrain.lobes.models.synthesis.dataio import load_datasets, filter_by_id_list
from speechbrain.lobes.models.synthesis.tacotron2.dataio import dynamic_range_compression
from speechbrain.utils.data_utils import scalarize
from torchaudio import transforms


logger = logging.getLogger(__name__)


class HifiGanBrain(sb.Brain):
    """The Brain implementation for HifiGan"""

    def on_fit_start(self):
        """Gets called at the beginning of ``fit()``, on multiple processes
        if ``distributed_count > 0`` and backend is ddp and initializes statistics"""
        self.hparams.progress_sample_logger.reset()
        self.last_epoch = 0
        self.last_batch = None
        self.last_loss_stats = {}
        return super().on_fit_start()

    def init_optimizers(self):
        """Called during ``on_fit_start()``, initialize optimizers
        after parameters are fully configured (e.g. DDP, jit).
        """
        if self.opt_class is not None:
            self.optimizer_g = self.opt_class[0](self.modules.generator.parameters())
            self.optimizer_d = self.opt_class[1](self.modules.discriminator.parameters())

            if self.checkpointer is not None:
                self.checkpointer.add_recoverable("optimizer_g", self.optimizer_g)
                self.checkpointer.add_recoverable("optimizer_d", self.optimizer_d)


    def compute_forward(self, batch, stage):
        """
        Computes the forward pass

        Arguments
        ---------
        batch: str
            a single batch
        stage: speechbrain.Stage
            the training stage

        Returns
        -------
        the model output
        """

        x, y = self.batch_to_device(batch)
        y_g_hat = self.modules.generator(x)[:, :, : y.size(2)]
        scores_fake, feats_fake = self.modules.discriminator(y_g_hat)
        scores_real, feats_real = self.modules.discriminator(y)

        return (y_g_hat, scores_fake, feats_fake, scores_real, feats_real)

    def fit_batch(self, batch):
        """
        Fits a single batch and applies annealing

        Arguments
        ---------
        batch: tuple
            a training batch

        Returns
        -------
        loss: torch.Tensor
            detached loss
        """
        outputs = self.compute_forward(batch, sb.core.Stage.TRAIN)
        loss = self.compute_objectives(outputs, batch, sb.core.Stage.TRAIN)
        loss_g = loss["G_loss"]
        loss_d = loss["D_loss"]

        loss_g.backward(retain_graph=True)
        loss_d.backward()

        for optimizer, loss in zip([self.optimizer_g, self.optimizer_d], [loss_g, loss_d]):
            if self.check_gradients(loss):
                optimizer.step()
            optimizer.zero_grad()

        return loss_g.detach().cpu()

    def evaluate_batch(self, batch, stage):
        """Evaluate one batch, override for different procedure than train.

        The default implementation depends on two methods being defined
        with a particular behavior:

        * ``compute_forward()``
        * ``compute_objectives()``

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
        loss_d = loss["D_loss"]
        return loss_g.detach().cpu() + loss_d.detach().cpu()

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
        x, y = self.batch_to_device(batch)

        # Hold on to the batch for the inference sample. This is needed because
        # the infernece sample is run from on_stage_end only, where
        # batch information is not available
        self.last_batch = (x, y)

        # Hold on to a sample (for logging)
        self._remember_sample(self.last_batch, predictions)

        y_hat, scores_fake, feats_fake, scores_real, feats_real = predictions
        loss_g = self.hparams.generator_loss(y_hat, y, scores_fake, feats_fake, feats_real)
        loss_d = self.hparams.discriminator_loss(scores_fake, scores_real)
        loss = {**loss_g , **loss_d}
        self.last_loss_stats[stage] = scalarize(loss)
        return loss

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

        self.hparams.progress_sample_logger.remember(
            raw_batch=self.hparams.progress_sample_logger.get_batch_sample(
                {
                    "input": self.get_spectrogram_sample(mel),
                    "target": sig,
                    "prediction": y_hat,
                    "scores_fake": scores_fake,
                    "feats_fake": feats_fake,
                    "scores_real": scores_real,
                    "feats_real": feats_real,
                }
            ),
        )

    def batch_to_device(self, batch):
        """
        Transfers the batch to the target device

        Arguments
        ---------
        batch: tuple
            the batch to use
        """
        mel, sig = batch["mel_sig_pair"]
        x = self.to_device(mel)
        y = self.to_device(sig)
        return (x, y)

    def to_device(self, x):
        """
        Transfers a single tensor to the target device

        Arguments
        ---------
        x: torch.Tensor
            the tensor to transfer

        Returns
        -------
        result: tensor
            the same tensor, on the target device
        """
        x = x.contiguous()
        x = x.to(self.device, non_blocking=True)
        return x

    def get_spectrogram_sample(self, raw):
        """Converts a raw spectrogram to one that can be saved as an image
        sample  = sqrt(exp(raw))
        Arguments
        ---------
        raw: torch.Tensor
            the raw spectrogram (as used in the model)
        Returns
        -------
        sample: torch.Tensor
            the spectrogram, for image saving purposes
        """
        sample = raw[0]
        return torch.sqrt(torch.exp(sample))

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

        # Store the train loss until the validation stage.

        # At the end of validation, we can write
        if stage == sb.Stage.VALID:
            # Update learning rate
            lr_g = self.optimizer_g.param_groups[-1]["lr"]
            lr_d = self.optimizer_d.param_groups[-1]["lr"]
            self.last_epoch = epoch

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(  # 1#2#
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
            output_progress_sample = (
                self.hparams.progress_samples
                and epoch % self.hparams.progress_samples_interval == 0
            )
            if output_progress_sample:
                self.run_inference_sample()
                self.hparams.progress_sample_logger.save(epoch)

        # We also write statistics about test data to stdout and to the logfile.
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=self.last_loss_stats[sb.Stage.VALID],
            )

    def run_inference_sample(self):
            """Produces a sample in inference mode. This is called when producing
            samples and can be useful because"""
            if self.last_batch is None:
                return
            inputs, y = self.last_batch
            sig_out = self.modules.generator(inputs)

            spec_out = mel_spectogram(self.hparams, sig_out.cpu())
            self.hparams.progress_sample_logger.remember(
                target_mel=self.get_spectrogram_sample(inputs),
                inference_mel=self.get_spectrogram_sample(spec_out),
                audio_target=sig_out.squeeze(0),
                audio_source=y.squeeze(0)
            )


def load_datasets(hparams, dataset_prep):
    """A convenience function to load multiple datasets, from hparams

    Arguments
    ---------
    hparams: dict
        a hyperparameters file
    dataset_prep: callable
        a function taking two parameters: (dataset, hparams) that

    """
    result = {}
    for name, dataset_params in hparams["datasets"].items():
        loader = dataset_params["loader"]
        dataset = loader(dataset_params["path"])
        filter_file_name = dataset_params.get("filter")
        if filter_file_name:
            dataset = filter_by_id_list(dataset, filter_file_name)
        result[name] = dataset_prep(dataset, name, hparams)
    return result


def dataset_prep(dataset, name, hparams):
    """
    Adds pipeline elements for Tacotron to a dataset and
    wraps it in a saveable data loader

    Arguments
    ---------
    dataset: DynamicItemDataSet
        a raw dataset

    Returns
    -------
    result: SaveableDataLoader
        a data loader
    """
    dataset.add_dynamic_item(audio_pipeline(hparams, name=="train"))
    dataset.set_output_keys(["mel_sig_pair"])
    return SaveableDataLoader(
        dataset,
        batch_size=hparams["batch_size"] if name=="train" else 1,
        drop_last=hparams.get("drop_last", False),
    )


def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.

    Arguments
    ---------
    hparams: dict
        model hyperparameters

    Returns
    -------
    datasets: tuple
        a tuple of data loaders (train_data_loader, valid_data_loader, test_data_loader)
    """

    return load_datasets(hparams, dataset_prep)


def mel_spectogram(hparams, audio):
    audio_to_mel = transforms.MelSpectrogram(
        sample_rate=hparams.sample_rate,
        hop_length=hparams.hop_length,
        win_length=hparams.win_length,
        n_fft=hparams.n_fft,
        n_mels=hparams.n_mel_channels,
        f_min=hparams.mel_fmin,
        f_max=hparams.mel_fmax,
        power=hparams.power,
        normalized=hparams.mel_normalized,
        norm=hparams.norm,
        mel_scale=hparams.mel_scale,
    )
    return audio_to_mel(audio)

# TODO: Modularize and decouple this
def audio_pipeline(hparams, split):
    """
    A pipeline function that provide a mel spectrogram

    Arguments
    ---------
    hparams: dict
        model hyperparameters

    Returns
    -------
    result: DymamicItem
        a pipeline element
    """
    audio_to_mel = transforms.MelSpectrogram(
        sample_rate=hparams["sample_rate"],
        hop_length=hparams["hop_length"],
        win_length=hparams["win_length"],
        n_fft=hparams["n_fft"],
        n_mels=hparams["n_mel_channels"],
        f_min=hparams["mel_fmin"],
        f_max=hparams["mel_fmax"],
        power=hparams["power"],
        normalized=hparams["mel_normalized"],
        norm=hparams["norm"],
        mel_scale=hparams["mel_scale"],
    )

    wav_folder = hparams.get("wav_folder")
    segment_size = hparams.get("segment_size")

    @sb.utils.data_pipeline.takes("wav", "label")
    @sb.utils.data_pipeline.provides("mel_sig_pair")
    def f(file_path, words):
        if wav_folder:
            file_path = os.path.join(wav_folder, file_path)
        audio = sb.dataio.dataio.read_audio(file_path)
        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)

        if split:
            if audio.size(1) >= segment_size:
                max_audio_start = audio.size(1) - segment_size
                audio_start = torch.randint(0, max_audio_start, (1,))
                audio = audio[:, audio_start:audio_start+segment_size]
            else:
                audio = torch.nn.functional.pad(audio, (0, segment_size - audio.size(1)), 'constant')
        mel = audio_to_mel(audio).squeeze(0)

        if hparams["dynamic_range_compression"]:
            mel = dynamic_range_compression(mel)
        yield mel, audio

    return f


if __name__ == "__main__":

    # Load hyperparameters file with command-line overrides
    #########
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    #############
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    #show_results_every = 5  # plots results every N iterations

    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    # sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # # Dataset prep
    # # here we create the datasets objects as well as tokenization and encoding
    datasets = dataio_prepare(hparams)

    # Brain class initialization
    hifi_gan_brain = HifiGanBrain(
        modules=hparams["modules"],
        opt_class=[hparams["opt_class_generator"], hparams["opt_class_discriminator"]],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Training
    hifi_gan_brain.fit(
        hifi_gan_brain.hparams.epoch_counter,
        datasets["train"],
        datasets["valid"],
    )

    # Test
    if "test" in datasets:
        hifi_gan_brain.evaluate(datasets["test"])