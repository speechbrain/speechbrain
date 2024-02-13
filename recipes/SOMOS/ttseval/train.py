#!/usr/bin/env python3
"""Recipe for training a TTS evaluation system

Authors
 * Artem Ploujnikov 2024
 * Yingzi Wang 2024
"""
import sys
import speechbrain as sb
import torchaudio
import logging
from hyperpyyaml import load_hyperpyyaml
from pathlib import Path
from contrastive_sampling import RegressionContrastiveEnhancement


logger = logging.getLogger(__name__)

LABEL_MODEL_SCORE = "Model Score"
LABEL_HUMAN_SCORE = "Human Score"
KEY_MODEL_SCORE = "model_score"
KEY_HUMAN_SCORE = "human_score"


# Brain class for TTS evaluation training
class TTSEvalBrain(sb.Brain):
    """Class that manages the training loop for TTS evaluation.
    See speechbrain.core.Brain."""

    def compute_forward(self, batch, stage):
        """Computes the forward pass

        Arguments
        ---------
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns
        -------
        predictions : Tensor | tuple
            predictions
        """
        if self.hparams.contrastive:
            return self.compute_forward_contrastive(batch)

        # We first move the batch to the appropriate device.
        batch = batch.to(self.device)

        # Compute predictions
        predictions = self.modules.model(batch.sig.data, batch.sig.lengths)

        return predictions

    def compute_forward_contrastive(self, batch):
        """Computes the forward pass (contrastive mode)

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
        batch = batch.to(self.device)

        # Compute predictions
        predictions_anchor = self.modules.model(
            batch.sig.data, batch.sig.lengths
        )
        predictions_contrast = self.modules.model(
            batch.contrast_sig.data, batch.contrast_sig.lengths
        )

        return predictions_anchor, predictions_contrast

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
        if self.hparams.contrastive:
            return self.compute_objectives_contrastive(
                predictions, batch, stage
            )
        scores = batch.score_num[:, None, None]

        loss = self.hparams.compute_cost(predictions, scores)

        # Append this batch of losses to the loss metric for easy
        self.loss_metric.append(
            batch.id, predictions, scores, batch.sig.lengths, reduction="batch"
        )
        if self.reg_metric is not None:
            self.reg_metric.append(batch.id, predictions, scores)
            self.reg_metric.append(
                batch.id, predictions, scores, groups=batch.system
            )

        return loss

    def compute_objectives_contrastive(self, predictions, batch, stage):
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
        predictions_anchor, predictions_contrast = predictions
        diff_predictions = predictions_anchor - predictions_contrast
        scores_anchor = batch.score_num[:, None, None].to(
            predictions_anchor.dtype
        )
        scores_contrast = batch.contrast_score_num[:, None, None].to(
            predictions_contrast.dtype
        )
        diff_targets = scores_anchor - scores_contrast

        loss_predictive = 0.5 * (
            self.hparams.compute_cost(scores_anchor, predictions_anchor)
            + self.hparams.compute_cost(scores_contrast, predictions_contrast)
        )
        loss_contrastive = self.hparams.compute_cost_contrastive(
            diff_predictions, diff_targets
        )

        predictive_loss_weight = 1.0 - self.hparams.contrastive_loss_weight
        loss = (
            loss_predictive * predictive_loss_weight
            + loss_contrastive * self.hparams.contrastive_loss_weight
        )

        # Append this batch of losses to the loss metric for easy
        self.loss_metric.append(
            batch.id,
            diff_predictions,
            diff_targets,
            batch.sig.lengths,
            reduction="batch",
        )
        self.loss_metric_contrastive.append(
            batch.id,
            diff_predictions,
            diff_targets,
            batch.sig.lengths,
            reduction="batch",
        )
        if self.reg_metric is not None:
            self.reg_metric.append(batch.id, predictions_anchor, scores_anchor)
            self.reg_metric.append(
                batch.contrast_id, predictions_contrast, scores_contrast
            )

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

        # Set up statistics trackers for this stage
        self.loss_metric = sb.utils.metric_stats.MetricStats(
            metric=self.hparams.compute_cost
        )
        self.loss_metric_contrastive = sb.utils.metric_stats.MetricStats(
            metric=self.hparams.compute_cost_contrastive
        )
        if stage != sb.Stage.TRAIN or self.hparams.train_regression_metric:
            self.reg_metric = sb.utils.metric_stats.LinearRegressionStats(
                scores_label=LABEL_MODEL_SCORE,
                targets_label=LABEL_HUMAN_SCORE,
                scores_key=KEY_MODEL_SCORE,
                targets_key=KEY_HUMAN_SCORE,
            )
            self.reg_system_metric = sb.utils.metric_stats.LinearRegressionStats(
                scores_label=LABEL_MODEL_SCORE,
                targets_label=LABEL_HUMAN_SCORE,
                scores_key=KEY_MODEL_SCORE,
                targets_key=KEY_HUMAN_SCORE,
                grouped=True,
            )
        else:
            self.reg_metric = None
            self.reg_system_metric = None

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
            self.train_stats = {
                "loss": stage_loss,
                "predictive_loss": self.loss_metric.summarize("average"),
            }
            if self.hparams.contrastive:
                self.train_stats[
                    "contrastive_loss"
                ] = self.loss_metric_contrastive.summarize("average")
            if self.reg_metric is not None:
                self.train_stats.update(
                    self.get_prefixed_metric_stats(self.reg_metric, "utt")
                )
                self.train_stats.update(
                    self.get_prefixed_metric_stats(self.reg_system_metric, "sys")
                )

        # Summarize the statistics from the stage for record-keeping.
        else:
            stats = {
                "loss": stage_loss,
                "predictive_loss": self.loss_metric.summarize("average"),
            }
            if self.hparams.contrastive:
                stats[
                    "contrastive_loss"
                ] = self.loss_metric_contrastive.summarize("average")
            stats.update(
                self.get_prefixed_metric_stats(self.reg_metric, "utt")
            )
            stats.update(
                self.get_prefixed_metric_stats(self.reg_system_metric, "sys")
            )

        # At the end of validation...
        if stage == sb.Stage.VALID:

            old_lr, new_lr = self.hparams.lr_annealing(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stats,
            )

            # Save the current checkpoint and delete previous checkpoints,
            self.checkpointer.save_and_keep_only(meta=stats, min_keys=["loss"])

        # We also write statistics about test data to stdout and to the logfile.
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )

        self.output_details(stage, epoch)
        if self.hparams.contrastive:
            self.shuffle(stage)

    def get_prefixed_metric_stats(self, metric, prefix):
        """Gets statistics from a MetricStats instance and applies
        a prfix to them
        
        Arguments
        ---------
        metric : speechbrain.utils.metric_stats.MetricStats
            A metric instance
        prefix : str
            The prefix to use
        
        Returns
        -------
        stats : dict
            prefixed statistics
        """
        stats = metric.summarize()
        return {
            f"{prefix}_{key}": value
            for key, value in stats.items()
        }

    def shuffle(self, stage):
        """Shuffles contrastive pairings

        Arguments
        ---------
        stage : speechbrain.Stage
            The experiment stage"""
        stage_key = stage.name.lower()
        self.contrastive_enhancements[stage_key].shuffle()

    def output_details(self, stage, epoch=None):
        """Outputs raw CSV stats and regression plots

        Arguments
        ---------
        stage : speechbrain.Stage
            The experiment stage
        epoch : int, optional
            The epoch number"""
        if self.reg_metric is None:
            return None
        target_path = Path(self.hparams.save_folder) / "details"
        if epoch is not None:
            target_path = target_path / str(epoch)
        target_path.mkdir(exist_ok=True, parents=True)
        stage_label = str(stage.name).lower()
        csv_file_name = f"raw_{stage_label}.csv"
        self.reg_metric.write_csv(target_path / csv_file_name)
        try:
            plot_file_name = f"regression_{stage_label}.png"
            self.reg_metric.plot(target_path / plot_file_name)
            plot_file_name = f"regression_{stage_label}_system.png"
            self.reg_system_metric.plot(target_path / plot_file_name)
        except ImportError:
            logger.warn("Unable to output plots, seaborn is not installed")


def dataio_prepare(hparams):
    """Prepares the dataset

    Arguments
    ---------
    hparams : dict
        Raw hyperparameters"""

    @sb.utils.data_pipeline.takes("score")
    @sb.utils.data_pipeline.provides("score_num")
    def score_pipeline(score):
        return float(score)

    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        """Load the signal, and pass it and its length to the corruption class.
        This is done on the CPU in the `collate_fn`."""
        sig = sb.dataio.dataio.read_audio(wav)
        sig = torchaudio.functional.resample(
            sig,
            orig_freq=hparams["src_sample_rate"],
            new_freq=hparams["tgt_sample_rate"],
        )
        return sig

    datasets = {}
    for key in ["train", "valid", "test"]:
        dataset = sb.dataio.dataset.DynamicItemDataset.from_csv(
            hparams[f"{key}_annotation"],
            dynamic_items=[score_pipeline, audio_pipeline],
            replacements={
                "data_root": hparams["data_folder"],
                "processed_folder": str(
                    Path(hparams["data_folder"]) / "processed"
                ),
            },
        )
        output_keys = ["id", "sig", "score_num", "system"]
        if hparams.get("use_transcripts", False):
            output_keys.append("char")
        dataset.set_output_keys(output_keys)
        datasets[key] = dataset
    return datasets


def add_contrastive(datasets, hparams):
    """Adds contrastive enhancement to the dataset

    Arguments
    ---------
    datasets : dict
        a dictionary of datasets with "train", "valid"
        and "test" keys

    hparams : dict
        Hyperparameters

    Returns
    -------
    contrastive_enhancements : dict
        A dictionary with the same keys as the dataset
        and corresponding RegressionContrastiveEnhancement
        objects as values
    """
    contrastive_enhancements = {}
    for key, dataset in datasets.items():
        contrastive_enhancement = RegressionContrastiveEnhancement(
            metric_key="score_num", min_delta=hparams["contrastive_min_delta"]
        )
        contrastive_enhancement.bind(dataset)
        contrastive_enhancements[key] = contrastive_enhancement
    return contrastive_enhancements


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
    from somos_prepare import prepare_somos

    if not hparams["skip_prep"]:

        sb.utils.distributed.run_on_main(
            prepare_somos,
            kwargs={
                "data_folder": hparams["data_folder"],
                "save_folder": hparams["data_folder"],
                "splits": hparams["splits"],
                "subset": hparams["subset"],
                "use_transcripts": hparams.get("use_transcripts", False),
                "char_list_file": hparams.get("char_list_file"),
            },
        )

    # Create dataset objects "train", "valid", and "test".
    datasets = dataio_prepare(hparams)

    # Initialize the Brain object to prepare for mask training.
    ttseval_brain = TTSEvalBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    if hparams["contrastive"]:
        contrastive_enhancements = add_contrastive(datasets, hparams)
        ttseval_brain.contrastive_enhancements = contrastive_enhancements

    # The `fit()` method iterates the training loop, calling the methods
    # necessary to update the parameters of the model. Since all objects
    # with changing state are managed by the Checkpointer, training can be
    # stopped at any point, and will be resumed on next call.
    ttseval_brain.fit(
        epoch_counter=ttseval_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )

    # Load the best checkpoint for evaluation
    test_stats = ttseval_brain.evaluate(
        test_set=datasets["test"],
        min_key="error",
        test_loader_kwargs=hparams["dataloader_options"],
    )
