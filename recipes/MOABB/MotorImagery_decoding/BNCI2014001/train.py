#!/usr/bin/python
"""
Recipe for training a compact CNN to decode motor imagery from single EEG trials.
The CNN is based on EEGNet and the dataset is BNCI2014001.
Reference to EEGNet: V. J. Lawhern et al., J Neural Eng 2018 (https://doi.org/10.1088/1741-2552/aace8c).
Reference to BNCI2014001:  (https://doi.org/10.1016/j.neuroimage.2020.117465).

To run this recipe:

    > python3 train.py train.yaml --data_folder '/path/to/MOABB_BNCI2014001'

Author
------
Davide Borra, 2021
"""

import pickle
import os
import sys
import torch
from hyperpyyaml import load_hyperpyyaml
from torch.nn import init
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
import speechbrain as sb
from MOABB_dataio_iterators import (
    WithinSession,
    CrossSession,
    LeaveOneSubjectOut,
)
from moabb.paradigms import MotorImagery
from moabb.datasets import BNCI2014001


class BNCI2014001Brain(sb.Brain):
    def compute_forward(self, batch, stage):
        "Given an input batch it computes the model output."
        inputs = batch[0].to(self.device)
        return self.modules.model(inputs)

    def compute_objectives(self, predictions, batch, stage):
        "Given the network predictions and targets computes the loss."
        targets = batch[1].to(self.device)
        loss = self.hparams.loss(
            predictions,
            targets,
            weight=torch.FloatTensor(self.hparams.class_weight).to(self.device),
        )
        if stage != sb.Stage.TRAIN:
            tmp_preds = torch.exp(predictions)
            self.preds.extend(tmp_preds.detach().cpu().numpy())
            self.targets.extend(batch[1].detach().cpu().numpy())
        return loss

    def on_fit_start(self,):
        """Gets called at the beginning of ``fit()``"""
        initialize_module(self.hparams.model)
        self.init_optimizers()
        self.metrics = {}
        self.metrics["loss"] = []
        self.metrics["f1"] = []
        self.metrics["auc"] = []
        self.metrics["cm"] = []

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
            f1 = f1_score(y_true=y_true, y_pred=y_pred, average="macro")
            auc = roc_auc_score(
                y_true=y_true, y_score=preds, multi_class="ovo", average="macro"
            )
            cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
            self.last_eval_loss = stage_loss
            self.last_eval_f1 = float(f1)
            self.last_eval_auc = float(auc)
            self.last_eval_cm = cm
        if stage == sb.Stage.VALID:
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch},
                train_stats={"loss": self.train_loss},
                valid_stats={
                    "loss": self.last_eval_loss,
                    "f1": self.last_eval_f1,
                    "auc": self.last_eval_auc,
                    "cm": self.last_eval_cm,
                },
            )
            # track valid metric history
            self.metrics["loss"].append(self.last_eval_loss)
            self.metrics["f1"].append(self.last_eval_f1)
            self.metrics["auc"].append(self.last_eval_auc)
            self.metrics["cm"].append(self.last_eval_cm)
            min_key, max_key = None, None
            if self.hparams.direction == "max":
                min_key = None
                max_key = self.hparams.target_valid_metric
            elif self.hparams.direction == "min":
                min_key = self.hparams.target_valid_metric
                max_key = None

            self.checkpointer.save_and_keep_only(
                meta={
                    "loss": self.metrics["loss"][-1],
                    "f1": self.metrics["f1"][-1],
                    "auc": self.metrics["auc"][-1],
                },
                min_keys=[min_key],
                max_keys=[max_key],
            )

            # early stopping
            current_metric = self.metrics[self.hparams.target_valid_metric][-1]
            if self.hparams.epoch_counter.should_stop(
                current=epoch, current_metric=current_metric,
            ):
                self.hparams.epoch_counter.current = (
                    self.hparams.epoch_counter.limit
                )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch loaded": self.hparams.epoch_counter.current},
                test_stats={
                    "loss": self.last_eval_loss,
                    "f1": self.last_eval_f1,
                    "auc": self.last_eval_auc,
                    "cm": self.last_eval_cm,
                },
            )


def run_single_fold(hparams, run_opts, datasets):
    """This function performs a single cross-validation fold."""
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
    brain = BNCI2014001Brain(
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
    # evaluation
    min_key, max_key = None, None
    if hparams["direction"] == "max":
        min_key = None
        max_key = hparams["target_valid_metric"]
    elif hparams["direction"] == "min":
        min_key = hparams["target_valid_metric"]
        max_key = None

    brain.evaluate(
        datasets["test"], progressbar=False, min_key=min_key, max_key=max_key
    )
    test_loss, test_f1, test_auc, test_cm = (
        brain.last_eval_loss,
        brain.last_eval_f1,
        brain.last_eval_auc,
        brain.last_eval_cm,
    )

    tmp_metrics_dict = {
        "loss": test_loss,
        "f1": test_f1,
        "auc": test_auc,
        "cm": test_cm,
    }
    return tmp_metrics_dict


def initialize_module(module):
    """Function to initialize neural network modules"""
    for mod in module.modules():
        if hasattr(mod, "weight"):
            if not ("BatchNorm" in mod.__class__.__name__):
                init.xavier_uniform_(mod.weight, gain=1)
            else:
                init.constant_(mod.weight, 1)
        if hasattr(mod, "bias"):
            if mod.bias is not None:
                init.constant_(mod.bias, 0)


if __name__ == "__main__":
    argv = sys.argv[1:]
    # Temporary switching off deprecation warning from mne
    import warnings  # noqa

    warnings.filterwarnings("ignore")
    hparams_file, run_opts, overrides = sb.core.parse_arguments(argv)
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    moabb_dataset = BNCI2014001()
    # to run on a subset of subjects: moabb_dataset.subject_list = [1, 2, 3, 4]
    moabb_dataset.download(path=hparams["data_folder"])
    moabb_paradigm = MotorImagery(
        n_classes=4,
        fmin=hparams["fmin"],
        fmax=hparams["fmax"],
        tmin=hparams["tmin"],
        tmax=hparams["tmax"],
        resample=hparams["sf"],
    )
    # defining data iterators to use
    data_its = [
        WithinSession(moabb_paradigm, hparams),
        CrossSession(moabb_paradigm, hparams),
        LeaveOneSubjectOut(moabb_paradigm, hparams),
    ]
    for data_it in data_its:
        print("Running {0} iterations".format(data_it.iterator_tag))
        for i, (exp_dir, datasets) in enumerate(data_it.prepare(moabb_dataset)):
            print("Running experiment %i" % (i))
            hparams["exp_dir"] = exp_dir
            # creating experiment directory
            sb.create_experiment_directory(
                experiment_directory=hparams["exp_dir"],
                hyperparams_to_save=hparams_file,
                overrides=overrides,
            )
            tmp_metrics_dict = run_single_fold(hparams, run_opts, datasets)
            # saving metrics on the test set in a pickle file
            metrics_fpath = os.path.join(hparams["exp_dir"], "metrics.pkl")
            with open(metrics_fpath, "wb",) as handle:
                pickle.dump(
                    tmp_metrics_dict, handle, protocol=pickle.HIGHEST_PROTOCOL
                )
            # restoring hparams for the next training and evaluation processes
            hparams_file, run_opts, overrides = sb.core.parse_arguments(argv)
            with open(hparams_file) as fin:
                hparams = load_hyperpyyaml(fin, overrides)
