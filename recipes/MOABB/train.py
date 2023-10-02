#!/usr/bin/python
"""
This script implements raining neural networks to decode single EEG trials using various paradigms on MOABB datasets.
For a list of supported datasets and paradigms, please refer to the official documentation at http://moabb.neurotechx.com/docs/api.html.

To run training (e.g., architecture: EEGNet; dataset: BNCI2014001) for a specific subject, recording session and training strategy:
    > python train.py hparams/MotorImagery/BNCI2014001/EEGNet.yaml --data_folder=eeg_data --cached_data_folder=eeg_pickled_data --target_subject_idx=0 --target_session_idx=0 --data_iterator_name=leave-one-session-out

Author
------
Davide Borra, 2022
Mirco Ravanelli, 2023
"""

import pickle
import os
import torch
from hyperpyyaml import load_hyperpyyaml
from torch.nn import init
import numpy as np
import logging
import sys
from utils.dataio_iterators import LeaveOneSessionOut, LeaveOneSubjectOut
from torchinfo import summary
import speechbrain as sb


class MOABBBrain(sb.Brain):
    def init_model(self, model):
        """Function to initialize neural network modules"""
        for mod in model.modules():
            if hasattr(mod, "weight"):
                if not ("Norm" in mod.__class__.__name__):
                    init.xavier_uniform_(mod.weight, gain=1)
                else:
                    init.constant_(mod.weight, 1)
            if hasattr(mod, "bias"):
                if mod.bias is not None:
                    init.constant_(mod.bias, 0)

    def compute_forward(self, batch, stage):
        "Given an input batch it computes the model output."
        inputs = batch[0].to(self.device)

        # Perform data augmentation
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "augment"):
            inputs, _ = self.hparams.augment(
                inputs.squeeze(3),
                lengths=torch.ones(inputs.shape[0], device=self.device),
            )
            inputs = inputs.unsqueeze(3)

        # Normalization
        if hasattr(self.hparams, "normalize"):
            inputs = self.hparams.normalize(inputs)
        return self.modules.model(inputs)

    def compute_objectives(self, predictions, batch, stage):
        "Given the network predictions and targets computes the loss."
        targets = batch[1].to(self.device)

        # Target augmentation
        N_augments = int(predictions.shape[0] / targets.shape[0])
        targets = torch.cat(N_augments * [targets], dim=0)

        loss = self.hparams.loss(
            predictions,
            targets,
            weight=torch.FloatTensor(self.hparams.class_weights).to(
                self.device
            ),
        )
        if stage != sb.Stage.TRAIN:
            # From log to linear predictions
            tmp_preds = torch.exp(predictions)
            self.preds.extend(tmp_preds.detach().cpu().numpy())
            self.targets.extend(batch[1].detach().cpu().numpy())
        else:
            if hasattr(self.hparams, "lr_annealing"):
                self.hparams.lr_annealing.on_batch_end(self.optimizer)
        return loss

    def on_fit_start(self,):
        """Gets called at the beginning of ``fit()``"""
        self.init_model(self.hparams.model)
        self.init_optimizers()
        in_shape = (
            (1,)
            + tuple(np.floor(self.hparams.input_shape[1:-1]).astype(int))
            + (1,)
        )
        model_summary = summary(self.hparams.model, input_size=in_shape)
        with open(
            os.path.join(self.hparams.exp_dir, "model.txt"), "w"
        ) as text_file:
            text_file.write(str(model_summary))

    def on_stage_start(self, stage, epoch=None):
        "Gets called when a stage (either training, validation, test) starts."
        if stage != sb.Stage.TRAIN:
            self.preds = []
            self.targets = []

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of a epoch."""
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        else:
            preds = np.array(self.preds)
            y_pred = np.argmax(preds, axis=-1)
            y_true = self.targets
            self.last_eval_stats = {
                "loss": stage_loss,
            }
            for metric_key in self.hparams.metrics.keys():
                self.last_eval_stats[metric_key] = self.hparams.metrics[
                    metric_key
                ](y_true=y_true, y_pred=y_pred)
            if stage == sb.Stage.VALID:
                # Learning rate scheduler
                if hasattr(self.hparams, "lr_annealing"):
                    old_lr, new_lr = self.hparams.lr_annealing(epoch)
                    sb.nnet.schedulers.update_learning_rate(
                        self.optimizer, new_lr
                    )
                    self.hparams.train_logger.log_stats(
                        stats_meta={"epoch": epoch, "lr": old_lr},
                        train_stats={"loss": self.train_loss},
                        valid_stats=self.last_eval_stats,
                    )
                else:
                    self.hparams.train_logger.log_stats(
                        stats_meta={"epoch": epoch},
                        train_stats={"loss": self.train_loss},
                        valid_stats=self.last_eval_stats,
                    )

                if epoch == 1:
                    self.best_eval_stats = self.last_eval_stats

                # The current model is saved if it is the best or the last
                is_best = self.check_if_best(
                    self.last_eval_stats,
                    self.best_eval_stats,
                    keys=[self.hparams.test_key],
                )
                is_last = (
                    epoch
                    > self.hparams.number_of_epochs - self.hparams.avg_models
                )

                # Check if we have to save the model
                if self.hparams.test_with == "last" and is_last:
                    save_ckpt = True
                elif self.hparams.test_with == "best" and is_best:
                    save_ckpt = True
                else:
                    save_ckpt = False

                # Saving the checkpoint
                if save_ckpt:
                    min_keys, max_keys = [], []
                    if self.hparams.test_key == "loss":
                        min_keys = [self.hparams.test_key]
                    else:
                        max_keys = [self.hparams.test_key]
                    meta = {}
                    for eval_key in self.last_eval_stats.keys():
                        if eval_key != "cm":
                            meta[str(eval_key)] = float(
                                self.last_eval_stats[eval_key]
                            )
                    self.checkpointer.save_and_keep_only(
                        meta=meta,
                        num_to_keep=self.hparams.avg_models,
                        min_keys=min_keys,
                        max_keys=max_keys,
                    )

            elif stage == sb.Stage.TEST:
                self.hparams.train_logger.log_stats(
                    stats_meta={
                        "epoch loaded": self.hparams.epoch_counter.current
                    },
                    test_stats=self.last_eval_stats,
                )
                # save the averaged checkpoint at the end of the evaluation stage
                # delete the rest of the intermediate checkpoints
                # ACC is set to 1.1 so checkpointer only keeps the averaged checkpoint
                if self.hparams.avg_models > 1:
                    min_keys, max_keys = [], []
                    if self.hparams.test_key == "loss":
                        min_keys = [self.hparams.test_key]
                        fake_meta = {self.hparams.test_key: 0.0, "epoch": epoch}
                    else:
                        max_keys = [self.hparams.test_key]
                        fake_meta = {self.hparams.test_key: 1.1, "epoch": epoch}
                    self.checkpointer.save_and_keep_only(
                        meta=fake_meta,
                        min_keys=min_keys,
                        max_keys=max_keys,
                        num_to_keep=1,
                    )

    def on_evaluate_start(self, max_key=None, min_key=None):
        """Perform checkpoint average if needed"""
        super().on_evaluate_start()

        ckpts = self.checkpointer.find_checkpoints(
            max_key=max_key, min_key=min_key
        )
        ckpt = sb.utils.checkpoints.average_checkpoints(
            ckpts, recoverable_name="model", device=self.device
        )

        self.hparams.model.load_state_dict(ckpt, strict=True)
        self.hparams.model.eval()

    def check_if_best(
        self, last_eval_stats, best_eval_stats, keys,
    ):
        """Checks if the current model is the best according at least to
        one of the monitored metrics. """
        is_best = False
        for key in keys:
            if key == "loss":
                if last_eval_stats[key] < best_eval_stats[key]:
                    is_best = True
                    best_eval_stats[key] = last_eval_stats[key]
                    break
            else:
                if last_eval_stats[key] > best_eval_stats[key]:
                    is_best = True
                    best_eval_stats[key] = last_eval_stats[key]
                    break
        return is_best


def run_experiment(hparams, run_opts, datasets):
    """This function performs a single training (e.g., single cross-validation fold)"""
    idx_examples = np.arange(datasets["train"].dataset.tensors[0].shape[0])
    n_examples_perclass = [
        idx_examples[
            np.where(datasets["train"].dataset.tensors[1] == c)[0]
        ].shape[0]
        for c in range(hparams["n_classes"])
    ]
    n_examples_perclass = np.array(n_examples_perclass)
    class_weights = n_examples_perclass.max() / n_examples_perclass
    hparams["class_weights"] = class_weights

    checkpointer = sb.utils.checkpoints.Checkpointer(
        checkpoints_dir=os.path.join(hparams["exp_dir"], "save"),
        recoverables={
            "model": hparams["model"],
            "counter": hparams["epoch_counter"],
        },
    )
    hparams["train_logger"] = sb.utils.train_logger.FileTrainLogger(
        save_file=os.path.join(hparams["exp_dir"], "train_log.txt")
    )
    logger = logging.getLogger(__name__)
    logger.info("Experiment directory: {0}".format(hparams["exp_dir"]))
    logger.info(
        "Input shape: {0}".format(
            datasets["train"].dataset.tensors[0].shape[1:]
        )
    )
    logger.info(
        "Training set avg value: {0}".format(
            datasets["train"].dataset.tensors[0].mean()
        )
    )
    datasets_summary = "Number of examples: {0} (training), {1} (validation), {2} (test)".format(
        datasets["train"].dataset.tensors[0].shape[0],
        datasets["valid"].dataset.tensors[0].shape[0],
        datasets["test"].dataset.tensors[0].shape[0],
    )
    logger.info(datasets_summary)

    brain = MOABBBrain(
        modules={"model": hparams["model"]},
        opt_class=hparams["optimizer"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=checkpointer,
    )
    # training
    brain.fit(
        epoch_counter=hparams["epoch_counter"],
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        progressbar=False,
    )
    # evaluation after loading model using specific key
    perform_evaluation(brain, hparams, datasets, dataset_key="test")
    # After the first evaluation only 1 checkpoint (best overall or averaged) is stored.
    # Setting avg_models to 1 to prevent deleting the checkpoint in subsequent calls of the evaluation stage.
    brain.hparams.avg_models = 1
    perform_evaluation(brain, hparams, datasets, dataset_key="valid")


def perform_evaluation(brain, hparams, datasets, dataset_key="test"):
    """This function perform the evaluation stage on a dataset and save the performance metrics in a pickle file"""
    min_key, max_key = None, None
    if hparams["test_key"] == "loss":
        min_key = hparams["test_key"]
    else:
        max_key = hparams["test_key"]
    # perform evaluation
    brain.evaluate(
        datasets[dataset_key],
        progressbar=False,
        min_key=min_key,
        max_key=max_key,
    )
    # saving metrics on the desired dataset in a pickle file
    metrics_fpath = os.path.join(
        hparams["exp_dir"], "{0}_metrics.pkl".format(dataset_key)
    )
    with open(metrics_fpath, "wb") as handle:
        pickle.dump(
            brain.last_eval_stats, handle, protocol=pickle.HIGHEST_PROTOCOL
        )


if __name__ == "__main__":
    argv = sys.argv[1:]
    # loading hparams to prepare the dataset and the data iterators
    hparams_file, run_opts, overrides = sb.core.parse_arguments(argv)
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # defining data iterator to use
    print("Prepare dataset iterators...")
    data_iterator = None

    if hparams["data_iterator_name"] == "leave-one-session-out":
        data_iterator = LeaveOneSessionOut(
            seed=hparams["seed"]
        )  # within-subject and cross-session
    elif hparams["data_iterator_name"] == "leave-one-subject-out":
        data_iterator = LeaveOneSubjectOut(
            seed=hparams["seed"]
        )  # cross-subject and cross-session

    if data_iterator is not None:
        tail_path, datasets = data_iterator.prepare(
            data_folder=hparams["data_folder"],
            dataset=hparams["dataset"],
            cached_data_folder=hparams["cached_data_folder"],
            batch_size=hparams["batch_size"],
            valid_ratio=hparams["valid_ratio"],
            target_subject_idx=hparams["target_subject_idx"],
            target_session_idx=hparams["target_session_idx"],
            events_to_load=hparams["events_to_load"],
            original_sample_rate=hparams["original_sample_rate"],
            sample_rate=hparams["sample_rate"],
            fmin=hparams["fmin"],
            fmax=hparams["fmax"],
            tmin=hparams["tmin"],
            tmax=hparams["tmax"],
            save_prepared_dataset=hparams["save_prepared_dataset"],
            n_steps_channel_selection=hparams["n_steps_channel_selection"],
        )

        # override C and T, to be sure that network input shape matches the dataset (e.g., after time cropping or channel sampling)
        argv += [
            "--T",
            str(datasets["train"].dataset.tensors[0].shape[1]),
            "--C",
            str(datasets["train"].dataset.tensors[0].shape[-2]),
            "--n_train_examples",
            str(datasets["train"].dataset.tensors[0].shape[0]),
        ]

        # loading hparams for the each training and evaluation processes
        hparams_file, run_opts, overrides = sb.core.parse_arguments(argv)
        with open(hparams_file) as fin:
            hparams = load_hyperpyyaml(fin, overrides)
        hparams["exp_dir"] = os.path.join(hparams["output_folder"], tail_path)

        # creating experiment directory
        sb.create_experiment_directory(
            experiment_directory=hparams["exp_dir"],
            hyperparams_to_save=hparams_file,
            overrides=overrides,
        )

        # Run training
        run_experiment(hparams, run_opts, datasets)
