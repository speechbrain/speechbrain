# -*- coding: utf-8 -*-
"""
 Recipe for training the Tacotron Text-To-Speech model, an end-to-end
 neural text-to-speech (TTS) system

 To run this recipe, do the following:
 # python train.py --device=cuda:0 --max_grad_norm=1.0 hparams.yaml

 to infer simply load saved model and do
 savemodel.infer(text_Sequence,len(textsequence))

 were text_Sequence is the ouput of the text_to_sequence function from
 textToSequence.py (from textToSequence import text_to_sequence)

 Authors
 * Georges Abous-Rjeili 2021
 * Artem Ploujnikov 2021

"""

import torch
import speechbrain as sb
import sys
import logging
from hyperpyyaml import load_hyperpyyaml
from speechbrain.lobes.models.synthesis.tacotron2 import dataio_prepare
from speechbrain.pretrained.training import PretrainedModelMixin
from speechbrain.utils.progress_samples import ProgressSampleImageMixin
from speechbrain.utils.data_utils import scalarize


logger = logging.getLogger(__name__)


class Tacotron2Brain(sb.Brain, PretrainedModelMixin, ProgressSampleImageMixin):
    PROGRESS_SAMPLE_FORMATS = {"raw_batch": "raw"}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_progress_samples()
        self.last_epoch = 0
        self.last_batch = None
        self.last_loss_stats = {}

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
        effective_batch = self.batch_to_device(batch)
        inputs, y, num_items, _, _ = effective_batch
        _, input_lengths, _, _, _ = inputs
        max_input_length = input_lengths.max().item()
        return self.modules.model(
            inputs, alignments_dim=max_input_length
        )  # 1#2#

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
        result = super().fit_batch(batch)
        self.hparams.lr_annealing(self.optimizer)
        return result

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
        effective_batch = self.batch_to_device(batch)
        self.last_batch = effective_batch
        self._remember_sample(effective_batch, predictions)
        loss = self._compute_loss(predictions, effective_batch, stage)
        return loss

    def _compute_loss(self, predictions, batch, stage):
        inputs, targets, num_items, labels, wavs = batch
        text_padded, input_lengths, _, max_len, output_lengths = inputs
        loss_stats = self.hparams.criterion(
            predictions, targets, input_lengths, output_lengths, self.last_epoch
        )
        self.last_loss_stats[stage] = scalarize(loss_stats)
        return loss_stats.loss

    def _remember_sample(self, batch, predictions):
        inputs, targets, num_items, labels, wavs = batch
        text_padded, input_lengths, _, max_len, output_lengths = inputs
        mel_target, _ = targets
        mel_out, mel_out_postnet, gate_out, alignments = predictions
        alignments_max = (
            alignments[0]
            .max(dim=-1)
            .values.max(dim=-1)
            .values.unsqueeze(-1)
            .unsqueeze(-1)
        )
        alignments_output = alignments[0].T.flip(dims=(1,)) / alignments_max
        self.remember_progress_sample(
            target=self._get_sample_data(mel_target),
            output=self._get_sample_data(mel_out),
            output_postnet=self._get_sample_data(mel_out_postnet),
            alignments=alignments_output,
            raw_batch=self.get_batch_sample(
                {
                    "text_padded": text_padded,
                    "input_lengths": input_lengths,
                    "mel_target": mel_target,
                    "mel_out": mel_out,
                    "mel_out_postnet": mel_out_postnet,
                    "max_len": max_len,
                    "output_lengths": output_lengths,
                    "gate_out": gate_out,
                    "alignments": alignments,
                    "labels": labels,
                    "wavs": wavs,
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
        (
            text_padded,
            input_lengths,
            mel_padded,
            gate_padded,
            output_lengths,
            len_x,
            labels,
            wavs,
        ) = batch
        text_padded = self.to_device(text_padded).long()
        input_lengths = self.to_device(input_lengths).long()
        max_len = torch.max(input_lengths.data).item()
        mel_padded = self.to_device(mel_padded).float()
        gate_padded = self.to_device(gate_padded).float()
        output_lengths = self.to_device(output_lengths).long()
        x = (text_padded, input_lengths, mel_padded, max_len, output_lengths)
        y = (mel_padded, gate_padded)
        len_x = torch.sum(output_lengths)
        return (x, y, len_x, labels, wavs)

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

    def _get_sample_data(self, raw):
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
            lr = self.optimizer.param_groups[-1]["lr"]
            self.last_epoch = epoch

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(  # 1#2#
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
                self.save_progress_sample(epoch)

        # We also write statistics about test data to stdout and to the logfile.
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=self.last_loss_stats[sb.Stage.VALID],
            )

    def run_inference_sample(self):
        if self.last_batch is None:
            return
        inputs, _, _, _, _ = self.last_batch
        text_padded, input_lengths, _, _, _ = inputs
        mel_out, _, _ = self.hparams.model.infer(
            text_padded[:1], input_lengths[:1]
        )
        self.remember_progress_sample(
            inference_mel_out=self._get_sample_data(mel_out)
        )


if __name__ == "__main__":

    # Load hyperparameters file with command-line overrides
    #########
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    #############
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    show_results_every = 5  # plots results every N iterations

    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    # sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Dataset prep
    # here we create the datasets objects as well as tokenization and encoding
    datasets = dataio_prepare(hparams)

    # Brain class initialization
    tacotron2_brain = Tacotron2Brain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Training
    tacotron2_brain.fit(
        tacotron2_brain.hparams.epoch_counter,
        datasets["train"],
        datasets["valid"],
    )
    if hparams.get("save_for_pretrained"):
        tacotron2_brain.save_for_pretrained()

    # Test
    if "test" in datasets:
        tacotron2_brain.evaluate(datasets["test"])
