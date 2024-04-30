#!/usr/bin/env python3
"""Recipe for training a language model with a given text corpus.

> python train.py RNNLM.yaml

To run this recipe, you need to first install the Huggingface dataset:
> pip install datasets


Authors
 * Ju-Chieh Chou 2020
 * Jianyuan Zhong 2021
 * Mirco Ravanelli 2021
"""
import logging
import sys

import torch
from datasets import load_dataset
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.utils.distributed import run_on_main

logger = logging.getLogger(__name__)


# Brain class for language model training
class LM(sb.core.Brain):
    """Class that manages the training loop. See speechbrain.core.Brain."""

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
        tokens_bos, _ = batch.tokens_bos
        pred = self.hparams.model(tokens_bos)
        return pred

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
        tokens_eos, tokens_len = batch.tokens_eos
        loss = self.hparams.compute_cost(
            predictions, tokens_eos, length=tokens_len
        )
        return loss

    def on_fit_batch_end(self, batch, outputs, loss, should_step):
        """At the end of the optimizer step, apply noam annealing."""
        if should_step:
            if isinstance(
                self.hparams.lr_annealing, sb.nnet.schedulers.NoamScheduler
            ) or isinstance(
                self.hparams.lr_annealing,
                sb.nnet.schedulers.CyclicCosineScheduler,
            ):
                self.hparams.lr_annealing(self.optimizer)

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


def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.

    The language model is trained with the text files specified by the user in
    the hyperparameter file.

    Arguments
    ---------
    hparams : dict
        This dictionary is loaded from the `train.yaml` file, and it includes
        all the hyperparameters needed for dataset construction and loading.

    Returns
    -------
    datasets : list
        List containing "train", "valid", and "test" sets that correspond
        to the appropriate DynamicItemDataset object.
    """
    logging.info("generating datasets...")

    # Prepare datasets
    datasets = load_dataset(
        "text",
        data_files={
            "train": hparams["lm_train_data"],
            "valid": hparams["lm_valid_data"],
            "test": hparams["lm_test_data"],
        },
    )

    # Convert huggingface's dataset to DynamicItemDataset via a magical function
    train_data = sb.dataio.dataset.DynamicItemDataset.from_arrow_dataset(
        datasets["train"]
    )
    valid_data = sb.dataio.dataset.DynamicItemDataset.from_arrow_dataset(
        datasets["valid"]
    )
    test_data = sb.dataio.dataset.DynamicItemDataset.from_arrow_dataset(
        datasets["test"]
    )

    datasets = [train_data, valid_data, test_data]
    tokenizer = hparams["tokenizer"]

    # Define text processing pipeline. We start from the raw text and then
    # encode it using the tokenizer. The tokens with bos are used for feeding
    # the neural network, the tokens with eos for computing the cost function.
    @sb.utils.data_pipeline.takes("text")
    @sb.utils.data_pipeline.provides("text", "tokens_bos", "tokens_eos")
    def text_pipeline(text):
        """Defines the pipeline that processes the input text."""
        yield text
        tokens_list = tokenizer.encode_as_ids(text)
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set outputs to add into the batch. The batch variable will contain
    # all these fields (e.g, batch.id, batch.text, batch.tokens.bos,..)
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "text", "tokens_bos", "tokens_eos"],
    )
    return train_data, valid_data, test_data


# Recipe begins!
if __name__ == "__main__":

    # Reading command line arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # We download the tokenizer from HuggingFace (or elsewhere depending on
    # the path given in the YAML file).
    run_on_main(hparams["pretrainer"].collect_files)
    hparams["pretrainer"].load_collected()

    # Create dataset objects "train", "valid", and "test"
    train_data, valid_data, test_data = dataio_prepare(hparams)

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
