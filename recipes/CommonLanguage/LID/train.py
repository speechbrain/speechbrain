#!/usr/bin/env python3
import sys
import torch
import logging
from torch import nn
import speechbrain as sb
from tqdm.contrib import tqdm
import torch.nn.functional as F
from hyperpyyaml import load_hyperpyyaml
from common_language_prepare import prepare_common_language, dataio_prep # noqa

"""Recipe for training a LID system with CommonLanguage.

To run this recipe, do the following:
> python train.py hparams/train_ecapa.yaml
> python train.py hparams/train_xvectors.yaml

Author
------
 * Mirco Ravanelli 2021
 * Pavlo Ruban 2021
"""

logger = logging.getLogger(__name__)


class CosClassifier(torch.nn.Module):
    def __init__(
        self,
        input_size,
        out_neurons,
        dropout=None,
        device="cpu"
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.FloatTensor(out_neurons, input_size, device=device)
        )
        nn.init.xavier_uniform_(self.weight)
        if dropout is not None:
            self.dropout = torch.nn.Dropout(p=dropout)
        else:
            self.dropout = dropout

    def forward(self, x):
        """Returns the output probabilities over languages.

        Arguments
        ---------
        x : torch.Tensor
            Torch tensor.
        """
        if self.dropout is not None:
            x = self.dropout(x)
        # Needs to be normalized to discard magnitudes
        x = F.linear(
            F.normalize(x.squeeze(1)), F.normalize(self.weight)
        )

        return x.unsqueeze(1)


# Brain class for Language ID training
class LID(sb.Brain):
    def prepare_features(self, wavs, stage):
        """Prepare the features for computation, including augmentation.

        Arguments
        ---------
        wavs : tuple
            Input signals (tensor) and their relative lengths (tensor).
        stage : sb.Stage
            The current stage of training.
        """
        wavs, lens = wavs

        # Add augmentation if specified. In this version of augmentation, we
        # concatenate the original and the augment batches in a single bigger
        # batch. This is more memory-demanding, but helps to improve the
        # performance. Change it if you run OOM.
        if stage == sb.Stage.TRAIN:
            wavs_noise = self.modules.env_corrupt(wavs, lens)
            wavs = torch.cat([wavs, wavs_noise], dim=0)
            lens = torch.cat([lens, lens], dim=0)
            wavs = self.hparams.augmentation(wavs, lens)

        # Feature extraction and normalization
        feats = self.modules.compute_features(wavs)
        feats = self.modules.mean_var_norm_input(feats, lens)

        return feats, lens

    def compute_embeddings(self, feats):
        # Compute embeddings
        embeddings = self.modules.embedding_model(feats)
        # Normalize embeddings
        embeddings = self.modules.mean_var_norm_emb(
            x=embeddings,
            lengths=torch.ones(embeddings.shape[0], device=embeddings.device)
        )

        return embeddings

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

        # Compute features, embeddings and output
        feats, lens = self.prepare_features(batch.sig, stage)

        embeddings = self.compute_embeddings(feats)

        outputs = self.modules.classifier(embeddings)

        return outputs, lens

    def compute_objectives(self, inputs, batch, stage):
        """Computes the loss given the predicted and targeted outputs.

        Arguments
        ---------
        inputs : tensors
            The output tensors from `compute_forward`.
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns
        -------
        loss : torch.Tensor
            A one-element tensor used for backpropagating the gradient.
        """

        predictions, lens = inputs

        targets = batch.language_name_encoded.data

        # Concatenate labels (due to data augmentation)
        if stage == sb.Stage.TRAIN:
            targets = torch.cat([targets, targets], dim=0)
            lens = torch.cat([lens, lens], dim=0)

        loss = self.hparams.compute_cost(
            embeddings=embeddings,
            outputs=predictions,
            targets=targets,
            weights=self.modules.classifier.weight
        )

        if hasattr(self.hparams.lr_annealing, "on_batch_end"):
            self.hparams.lr_annealing.on_batch_end(self.optimizer)

        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(batch.id, predictions, targets, lens)

        return loss

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Set up evaluation-only statistics trackers
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
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
                "error": self.error_metrics.summarize("average"),
            }

        # At the end of validation...
        if stage == sb.Stage.VALID:

            old_lr, new_lr = self.hparams.lr_annealing(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch, "lr": old_lr},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )

            # Save the current checkpoint and delete previous checkpoints,
            self.checkpointer.save_and_keep_only(
                meta=stats,
                min_keys=["error"]
            )

        # We also write statistics about test data to stdout and to the logfile.
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )


# Recipe begins!
if __name__ == "__main__":

    # Reading command line arguments.
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training).
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides.
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Data preparation, to be run on only one process.
    sb.utils.distributed.run_on_main(
        prepare_common_language,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["save_folder"],
            "skip_prep": hparams["skip_prep"]
        },
    )

    # Create dataset objects "train", "dev", and "test" and language_encoder
    datasets, language_encoder = dataio_prep(hparams)

    # Fetch and laod pretrained modules
    sb.utils.distributed.run_on_main(hparams["pretrainer"].collect_files)
    hparams["pretrainer"].load_collected(device=run_opts["device"])

    # Initialize the Brain object to prepare for mask training.
    lid_brain = LID(
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
    lid_brain.fit(
        epoch_counter=lid_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["dev"],
        train_loader_kwargs=hparams["train_dataloader_options"],
        valid_loader_kwargs=hparams["test_dataloader_options"],
    )

    # Load the best checkpoint for evaluation
    test_stats = lid_brain.evaluate(
        test_set=datasets["test"],
        min_key="error",
        test_loader_kwargs=hparams["test_dataloader_options"],
    )