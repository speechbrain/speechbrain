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

"""

import torch
import speechbrain as sb
import sys
import logging
from hyperpyyaml import load_hyperpyyaml
from speechbrain.lobes.models.synthesis.tacotron2 import dataio_prepare


sys.path.append("..")
from common.utils import PretrainedModelMixin  # noqa

logger = logging.getLogger(__name__)


class Tacotron2Brain(sb.Brain, PretrainedModelMixin):
    def compute_forward(self, batch, stage):
        inputs, y, num_items = batch_to_gpu(batch)
        return self.hparams.model(inputs)  # 1#2#

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

        inputs, y, num_items = batch_to_gpu(batch)
        return criterion(predictions, y)

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
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        # Summarize the statistics from the stage for record-keeping.
        else:
            stats = {
                "loss": stage_loss,
            }

        # At the end of validation, we can write
        if stage == sb.Stage.VALID:
            # Update learning rate
            old_lr, new_lr = self.hparams.lr_annealing(stage_loss)  # 1#2#
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(  # 1#2#
                {"Epoch": epoch},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )

            # Save the current checkpoint and delete previous checkpoints.
            self.checkpointer.save_and_keep_only(meta=stats, min_keys=["loss"])

        # We also write statistics about test data to stdout and to the logfile.
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )


# some helper functoions
def batch_to_gpu(batch):
    (
        text_padded,
        input_lengths,
        mel_padded,
        gate_padded,
        output_lengths,
        len_x,
    ) = batch
    text_padded = to_gpu(text_padded).long()
    input_lengths = to_gpu(input_lengths).long()
    max_len = torch.max(input_lengths.data).item()
    mel_padded = to_gpu(mel_padded).float()
    gate_padded = to_gpu(gate_padded).float()
    output_lengths = to_gpu(output_lengths).long()
    x = (text_padded, input_lengths, mel_padded, max_len, output_lengths)
    y = (mel_padded, gate_padded)
    len_x = torch.sum(output_lengths)
    return (x, y, len_x)


def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return x


def criterion(model_output, targets):
    mel_target, gate_target = targets[0], targets[1]
    mel_target.requires_grad = False
    gate_target.requires_grad = False
    gate_target = gate_target.view(-1, 1)

    mel_out, mel_out_postnet, gate_out, _ = model_output
    gate_out = gate_out.view(-1, 1)
    mel_loss = torch.nn.MSELoss()(mel_out, mel_target) + torch.nn.MSELoss()(
        mel_out_postnet, mel_target
    )
    gate_loss = torch.nn.BCEWithLogitsLoss()(gate_out, gate_target)
    return mel_loss + gate_loss


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
