#!/usr/bin/env python3
"""Recipe for training a language model with a phoneme model.

> python train.py RNNLM.yaml

To run this recipe, you need to first install the Huggingface dataset:
> pip install datasets


Authors
 * Ju-Chieh Chou 2020
 * Jianyuan Zhong 2021
 * Mirco Ravanelli 2021
 * Artem Ploujnikov 2021
"""
import sys
import logging
import os
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main
from train import dataio_prep

# The following hyperparameters are used in dataio_prep, shared with the
# main G2P training script:
# - hparams["phoneme_encoder"]
# - hparams["origins"]
# - hparams["phn_token_wordwise"]
# - hparams["graphemes"]
# - hparams["sorting"]
# - hparams["test_data"]
# - hparams["data_load"]
# - hparams["token_space_index"]
# - hparams["valid_data"]
# - hparams["tokenizer_valid_data"]
# - hparams["grapheme_encoder"]
# - hparams["train_data"]
# - hparams["eos_index"]
# - hparams["bos_index"]
# - hparams["tokenizer_file"]
# - hparams["output_neurons"]
# - hparams["tokenizer"]
# - hparams["blank_index"]


logger = logging.getLogger(__name__)


# Brain class for language model training
class LM(sb.core.Brain):
    def compute_forward(self, batch, stage):
        """Predicts the next word given the previous ones.

        Arguments
        ---------
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns
        -------
        predictions : torch.Tensor
            A tensor containing the posterior probabilities (predictions).
        """
        batch = batch.to(self.device)
        tokens_bos, _ = batch.phn_encoded_bos
        pred = self.hparams.model(tokens_bos)
        return self.hparams.log_softmax(pred)

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss given the predicted and targeted outputs.

        Arguments
        ---------
        predictions : torch.Tensor
            The posterior probabilities from `compute_forward`.
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
        tokens_eos, tokens_len = batch.phn_encoded_eos
        loss = self.hparams.compute_cost(
            predictions, tokens_eos, length=tokens_len
        )
        return loss

    def fit_batch(self, batch):
        """Runs all the steps needed to train the model on a single batch.

        Arguments
        ---------
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.

        Returns
        -------
        Loss : torch.Tensor
            A tensor containing the loss (single real number).
        """
        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)

        # Loss backpropagation (gradient computation)
        (loss / self.hparams.accu_steps).backward()

        # Manage gradient accumulation
        if self.step % self.hparams.accu_steps == 0:

            # Gradient clipping & early stop if loss is not fini
            self.check_gradients(loss)

            # Update the parameters
            self.optimizer.step()

            # Reset the gradient
            self.optimizer.zero_grad()

            if isinstance(
                self.hparams.lr_annealing, sb.nnet.schedulers.NoamScheduler
            ) or isinstance(
                self.hparams.lr_annealing,
                sb.nnet.schedulers.CyclicCosineScheduler,
            ):
                self.hparams.lr_annealing(self.optimizer)

        return loss

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

        # At the end of validation, we can wrote
        if stage == sb.Stage.VALID:

            # Update learning rate
            old_lr, new_lr = self.hparams.lr_annealing(stage_loss)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
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


# Recipe begins!
if __name__ == "__main__":

    # Reading command line arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    from tokenizer_prepare import prepare_tokenizer  # noqa

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    if hparams.get("phn_tokenize"):
        path = hparams["phoneme_tokenizer_output_folder"]
        if not os.path.exists(path):
            os.makedirs(path)
        run_on_main(
            prepare_tokenizer,
            kwargs={
                "dataset_name": hparams["dataset"],
                "data_folder": hparams["data_folder"],
                "save_folder": hparams["save_folder"],
                "phonemes": hparams["phonemes"],
            },
        )

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )
    # Create dataset objects "train", "valid", and "test"
    train_data, valid_data, test_data, _ = dataio_prep(hparams)

    # Initialize the Brain object to prepare for LM training.
    lm_brain = LM(
        modules=hparams["modules"],
        opt_class=hparams["optimizer"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # The `fit()` method iterates the training loop, calling the methods
    # necessary to update the parameters of the model. Since all objects
    # with changing state are managed by the Checkpointer, training can be
    # stopped at any point, and will be resumed on next call.
    lm_brain.fit(
        lm_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    # Load best checkpoint for evaluation
    test_stats = lm_brain.evaluate(
        test_data,
        min_key="loss",
        test_loader_kwargs=hparams["test_dataloader_opts"],
    )
