#!/usr/bin/env/python3
"""Recipe for training a speech enhancement system with the Voicebank dataset based on the SEGAN model architecture.
(based on the paper: Pascual et al. https://arxiv.org/pdf/1703.09452.pdf).

To run this recipe, do the following:
> python train.py hparams/train.yaml

Authors
 * Francis Carter 2021
 * Mirco Ravanelli 2021
"""

import os
import sys
import torch
import torchaudio
import speechbrain as sb
from pesq import pesq
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.metric_stats import MetricStats
from speechbrain.nnet.loss.stoi_loss import stoi_loss
from speechbrain.utils.distributed import run_on_main


# Brain class for speech enhancement training
class SEBrain(sb.Brain):
    def compute_forward_g(self, noisy_wavs):
        """Forward computations of the generator. Input noisy signal,
        output clean signal"""
        noisy_wavs = noisy_wavs.to(self.device)
        predict_wavs = self.modules["model_g"](noisy_wavs)
        return predict_wavs

    def compute_forward_d(self, noisy_wavs, clean_wavs):
        """Forward computations from discriminator. Input denoised-noisy pair,
        output whether denoising was properly acheived"""
        noisy_wavs = noisy_wavs.to(self.device)
        clean_wavs = clean_wavs.to(self.device)

        inpt = torch.cat((noisy_wavs, clean_wavs), -1)
        out = self.modules["model_d"](inpt)
        return out

    def compute_objectives_d1(self, d_outs, batch):
        """Computes the loss of a discriminator given predicted and
        targeted outputs, with target being clean"""
        loss = self.hparams.compute_cost["d1"](d_outs)
        self.loss_metric_d1.append(batch.id, d_outs, reduction="batch")
        return loss

    def compute_objectives_d2(self, d_outs, batch):
        """Computes the loss of a discriminator given predicted and targeted outputs,
        with target being noisy"""
        loss = self.hparams.compute_cost["d2"](d_outs)
        self.loss_metric_d2.append(batch.id, d_outs, reduction="batch")
        return loss

    def compute_objectives_g3(
        self,
        d_outs,
        predict_wavs,
        clean_wavs,
        batch,
        stage,
        z_mean=None,
        z_logvar=None,
    ):
        """Computes the loss of the generator based on discriminator and generator losses"""
        clean_wavs_orig, lens = batch.clean_sig
        clean_wavs_orig = clean_wavs_orig.to(self.device)
        clean_wavs = clean_wavs.to(self.device)

        loss = self.hparams.compute_cost["g3"](
            d_outs,
            predict_wavs,
            clean_wavs,
            lens,
            l1LossCoeff=self.hparams.l1LossCoeff,
            klLossCoeff=self.hparams.klLossCoeff,
            z_mean=z_mean,
            z_logvar=z_logvar,
        )
        self.loss_metric_g3.append(
            batch.id,
            d_outs,
            predict_wavs,
            clean_wavs,
            lens,
            l1LossCoeff=self.hparams.l1LossCoeff,
            klLossCoeff=self.hparams.klLossCoeff,
            z_mean=z_mean,
            z_logvar=z_logvar,
            reduction="batch",
        )

        if stage != sb.Stage.TRAIN:

            # Evaluate speech quality/intelligibility
            predict_wavs = predict_wavs.reshape(self.batch_current, -1)
            clean_wavs = clean_wavs.reshape(self.batch_current, -1)

            predict_wavs = predict_wavs[:, 0 : self.original_len]
            clean_wavs = clean_wavs[:, 0 : self.original_len]

            self.stoi_metric.append(
                batch.id, predict_wavs, clean_wavs, lens, reduction="batch"
            )
            self.pesq_metric.append(
                batch.id, predict=predict_wavs.cpu(), target=clean_wavs.cpu()
            )

            # Write enhanced test wavs to file
            if stage == sb.Stage.TEST:
                lens = lens * clean_wavs.shape[1]
                for name, pred_wav, length in zip(batch.id, predict_wavs, lens):
                    name += ".wav"
                    enhance_path = os.path.join(
                        self.hparams.enhanced_folder, name
                    )
                    print(enhance_path)

                    pred_wav = pred_wav / torch.max(torch.abs(pred_wav)) * 0.99
                    torchaudio.save(
                        enhance_path,
                        pred_wav[: int(length)].cpu().unsqueeze(0),
                        hparams["sample_rate"],
                    )
        return loss

    def fit_batch(self, batch):
        """Fit one batch, override to do multiple updates.

        The default implementation depends on a few methods being defined
        with a particular behavior:

        * ``compute_forward()``
        * ``compute_objectives()``

        Also depends on having optimizers passed at initialization.

        Arguments
        ---------
        batch : list of torch.Tensors
            Batch of data to use for training. Default implementation assumes
            this batch has two elements: inputs and targets.

        Returns
        -------
        detached loss
        """
        noisy_wavs, lens = batch.noisy_sig
        clean_wavs, lens = batch.clean_sig

        # split sentences in smaller chunks
        noisy_wavs = create_chunks(
            noisy_wavs,
            chunk_size=hparams["chunk_size"],
            chunk_stride=hparams["chunk_stride"],
        )
        clean_wavs = create_chunks(
            clean_wavs,
            chunk_size=hparams["chunk_size"],
            chunk_stride=hparams["chunk_stride"],
        )

        # first of three step training process detailed in SEGAN paper
        out_d1 = self.compute_forward_d(noisy_wavs, clean_wavs)
        loss_d1 = self.compute_objectives_d1(out_d1, batch)
        loss_d1.backward()
        if self.check_gradients(loss_d1):
            self.optimizer_d.step()
        self.optimizer_d.zero_grad()

        # second training step
        z_mean = None
        z_logvar = None
        if self.modules["model_g"].latent_vae:
            out_g2, z_mean, z_logvar = self.compute_forward_g(noisy_wavs)
        else:
            out_g2 = self.compute_forward_g(noisy_wavs)
        out_d2 = self.compute_forward_d(out_g2, clean_wavs)
        loss_d2 = self.compute_objectives_d2(out_d2, batch)
        loss_d2.backward(retain_graph=True)
        if self.check_gradients(loss_d2):
            self.optimizer_d.step()
        self.optimizer_d.zero_grad()

        # third (last) training step
        self.optimizer_g.zero_grad()
        out_d3 = self.compute_forward_d(out_g2, clean_wavs)
        loss_g3 = self.compute_objectives_g3(
            out_d3,
            out_g2,
            clean_wavs,
            batch,
            sb.Stage.TRAIN,
            z_mean=z_mean,
            z_logvar=z_logvar,
        )
        loss_g3.backward()
        if self.check_gradients(loss_g3):
            self.optimizer_g.step()
        self.optimizer_g.zero_grad()
        self.optimizer_d.zero_grad()

        loss_d1.detach().cpu()
        loss_d2.detach().cpu()
        loss_g3.detach().cpu()

        return loss_d1 + loss_d2 + loss_g3

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
        noisy_wavs, lens = batch.noisy_sig
        clean_wavs, lens = batch.clean_sig
        self.batch_current = clean_wavs.shape[0]
        self.original_len = clean_wavs.shape[1]

        # Add padding to make sure all the signal will be processed.
        padding_elements = torch.zeros(
            clean_wavs.shape[0], hparams["chunk_size"], device=clean_wavs.device
        )
        clean_wavs = torch.cat([clean_wavs, padding_elements], dim=1)
        noisy_wavs = torch.cat([noisy_wavs, padding_elements], dim=1)

        # Split sentences in smaller chunks
        noisy_wavs = create_chunks(
            noisy_wavs,
            chunk_size=hparams["chunk_size"],
            chunk_stride=hparams["chunk_size"],
        )
        clean_wavs = create_chunks(
            clean_wavs,
            chunk_size=hparams["chunk_size"],
            chunk_stride=hparams["chunk_size"],
        )

        # Perform speech enhancement with the current model
        out_d1 = self.compute_forward_d(noisy_wavs, clean_wavs)
        loss_d1 = self.compute_objectives_d1(out_d1, batch)

        z_mean = None
        z_logvar = None
        if self.modules["model_g"].latent_vae:
            out_g2, z_mean, z_logvar = self.compute_forward_g(noisy_wavs)
        else:
            out_g2 = self.compute_forward_g(noisy_wavs)
        out_d2 = self.compute_forward_d(out_g2, clean_wavs)
        loss_d2 = self.compute_objectives_d2(out_d2, batch)

        loss_g3 = self.compute_objectives_g3(
            out_d2,
            out_g2,
            clean_wavs,
            batch,
            stage=stage,
            z_mean=z_mean,
            z_logvar=z_logvar,
        )

        loss_d1.detach().cpu()
        loss_d2.detach().cpu()
        loss_g3.detach().cpu()

        return loss_d1 + loss_d2 + loss_g3

    def init_optimizers(self):
        """Called during ``on_fit_start()``, initialize optimizers
        after parameters are fully configured (e.g. DDP, jit).

        The default implementation of this method depends on an optimizer
        class being passed at initialization that takes only a list
        of parameters (e.g., a lambda or a partial function definition).
        This creates a single optimizer that optimizes all trainable params.

        Override this class if there are multiple optimizers.
        """
        if self.opt_class is not None:
            self.optimizer_d = self.opt_class(
                self.modules["model_d"].parameters()
            )
            self.optimizer_g = self.opt_class(
                self.modules["model_g"].parameters()
            )

            if self.checkpointer is not None:
                self.checkpointer.add_recoverable(
                    "optimizer_g", self.optimizer_g
                )
                self.checkpointer.add_recoverable(
                    "optimizer_d", self.optimizer_d
                )

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch"""
        self.loss_metric_d1 = MetricStats(
            metric=self.hparams.compute_cost["d1"]
        )
        self.loss_metric_d2 = MetricStats(
            metric=self.hparams.compute_cost["d2"]
        )
        self.loss_metric_g3 = MetricStats(
            metric=self.hparams.compute_cost["g3"]
        )
        self.stoi_metric = MetricStats(metric=stoi_loss)

        # Define function taking (prediction, target) for parallel eval
        def pesq_eval(pred_wav, target_wav):
            """Computes the PESQ evaluation metric"""
            return pesq(
                fs=hparams["sample_rate"],
                ref=target_wav.numpy().squeeze(),
                deg=pred_wav.numpy().squeeze(),
                mode="wb",
            )

        if stage != sb.Stage.TRAIN:
            self.pesq_metric = MetricStats(
                metric=pesq_eval, batch_eval=False, n_jobs=1
            )

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch."""
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
            self.train_stats = {  # "loss": self.loss_metric.scores,
                "loss_d1": self.loss_metric_d1.scores,
                "loss_d2": self.loss_metric_d2.scores,
                "loss_g3": self.loss_metric_g3.scores,
            }
        else:
            stats = {
                "loss": stage_loss,
                "pesq": self.pesq_metric.summarize("average"),
                "stoi": -self.stoi_metric.summarize("average"),
            }

        if stage == sb.Stage.VALID:
            if self.hparams.use_tensorboard:
                valid_stats = {
                    # "loss": self.loss_metric.scores,
                    "loss_d1": self.loss_metric_d1.scores,
                    "loss_d2": self.loss_metric_d2.scores,
                    "loss_g3": self.loss_metric_g3.scores,
                    "stoi": self.stoi_metric.scores,
                    "pesq": self.pesq_metric.scores,
                }
                self.hparams.tensorboard_train_logger.log_stats(
                    {"Epoch": epoch}, self.train_stats, valid_stats
                )
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )
            self.checkpointer.save_and_keep_only(meta=stats, max_keys=["pesq"])

        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )


def create_chunks(x, chunk_size=16384, chunk_stride=16384):
    """Splits the input into smaller chunks of size chunk_size with
    an overlap chunk_stride. The chunks are concatenated over
    the batch axis."""
    x = x.unfold(1, chunk_size, chunk_stride)
    x = x.reshape(x.shape[0] * x.shape[1], -1, 1)
    return x


def dataio_prep(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""

    # Define audio piplines
    @sb.utils.data_pipeline.takes("noisy_wav")
    @sb.utils.data_pipeline.provides("noisy_sig")
    def noisy_pipeline(noisy_wav):
        noisy_wav = sb.dataio.dataio.read_audio(noisy_wav)
        return noisy_wav

    @sb.utils.data_pipeline.takes("clean_wav")
    @sb.utils.data_pipeline.provides("clean_sig")
    def clean_pipeline(clean_wav):
        clean_wav = sb.dataio.dataio.read_audio(clean_wav)
        return clean_wav

    # Define datasets
    datasets = {}
    for dataset in ["train", "valid", "test"]:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=hparams[f"{dataset}_annotation"],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[noisy_pipeline, clean_pipeline],
            output_keys=["id", "noisy_sig", "clean_sig"],
        )

    # Sort train dataset
    if hparams["sorting"] == "ascending" or hparams["sorting"] == "descending":
        datasets["train"] = datasets["train"].filtered_sorted(
            sort_key="length", reverse=hparams["sorting"] == "descending"
        )
        hparams["train_dataloader_opts"]["shuffle"] = False
    elif hparams["sorting"] != "random":
        raise NotImplementedError(
            "Sorting must be random, ascending, or descending"
        )

    return datasets


def create_folder(folder):
    """Creates a new folder (where to store enhanced wavs)"""
    if not os.path.isdir(folder):
        os.makedirs(folder)


# Recipe begins!
if __name__ == "__main__":

    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Data preparation
    from voicebank_prepare import prepare_voicebank  # noqa

    run_on_main(
        prepare_voicebank,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["data_folder"],
            "skip_prep": hparams["skip_prep"],
        },
    )

    # Create dataset objects
    datasets = dataio_prep(hparams)

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

    # Create the folder to save enhanced files (+ support for DDP)
    run_on_main(create_folder, kwargs={"folder": hparams["enhanced_folder"]})

    se_brain = SEBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Load latest checkpoint to resume training
    se_brain.fit(
        epoch_counter=se_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    # Load best checkpoint for evaluation
    test_stats = se_brain.evaluate(
        test_set=datasets["test"],
        max_key="pesq",
        test_loader_kwargs=hparams["test_dataloader_opts"],
    )
