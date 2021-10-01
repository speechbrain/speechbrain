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
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    roc_auc_score,
    balanced_accuracy_score,
)
import multiprocessing as mp

# sys.path.append("/home/davide/Documents/codes/shared_speechbrain-1")
import speechbrain as sb

# from torchsummary import summary
mp.set_start_method("spawn", force=True)


class MOABBBrain(sb.Brain):
    def init_model(self, model):
        """Function to initialize neural network modules"""
        for mod in model.modules():
            if hasattr(mod, "weight"):
                if not ("BatchNorm" in mod.__class__.__name__):
                    init.xavier_uniform_(mod.weight, gain=1)
                else:
                    init.constant_(mod.weight, 1)
            if hasattr(mod, "bias"):
                if mod.bias is not None:
                    init.constant_(mod.bias, 0)

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
            weight=torch.FloatTensor(self.hparams.class_weights).to(
                self.device
            ),
        )
        if stage != sb.Stage.TRAIN:
            tmp_preds = torch.exp(predictions)
            self.preds.extend(tmp_preds.detach().cpu().numpy())
            self.targets.extend(batch[1].detach().cpu().numpy())
        return loss

    def on_fit_start(self,):
        """Gets called at the beginning of ``fit()``"""
        self.init_model(self.hparams.model)
        self.init_optimizers()

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
            acc = balanced_accuracy_score(y_true=y_true, y_pred=y_pred)
            cm = confusion_matrix(y_true=y_true, y_pred=y_pred)

            self.last_eval_loss = stage_loss
            self.last_eval_f1 = float(f1)
            self.last_eval_auc = float(auc)
            self.last_eval_acc = float(acc)
            self.last_eval_cm = cm
            last_eval_stats = {
                "loss": self.last_eval_loss,
                "f1": self.last_eval_f1,
                "auc": self.last_eval_auc,
                "acc": self.last_eval_acc,
                "cm": self.last_eval_cm,
            }

            if stage == sb.Stage.VALID:
                self.hparams.train_logger.log_stats(
                    stats_meta={"epoch": epoch},
                    train_stats={"loss": self.train_loss},
                    valid_stats=last_eval_stats,
                )
                self.checkpointer.save_and_keep_only(
                    meta={
                        "loss": self.last_eval_loss,
                        "f1": self.last_eval_f1,
                        "auc": self.last_eval_auc,
                        "acc": self.last_eval_acc,
                    },
                    min_keys=["loss"],
                    max_keys=["f1", "auc", "acc"],
                )

            elif stage == sb.Stage.TEST:
                self.hparams.train_logger.log_stats(
                    stats_meta={
                        "epoch loaded": self.hparams.epoch_counter.current
                    },
                    test_stats=last_eval_stats,
                )


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
    # evaluation
    min_key, max_key = None, None
    if "min_key" in hparams.keys():
        min_key = hparams["min_key"]
    if "max_key" in hparams.keys():
        max_key = hparams["max_key"]
    if min_key is not None or max_key is not None:
        # perform evaluation only if min_key and max_key were specified

        brain.evaluate(
            datasets["test"],
            progressbar=False,
            min_key=min_key,
            max_key=max_key,
        )
        test_loss, test_f1, test_acc, test_auc, test_cm = (
            brain.last_eval_loss,
            brain.last_eval_f1,
            brain.last_eval_acc,
            brain.last_eval_auc,
            brain.last_eval_cm,
        )

        # saving metrics on the test set in a pickle file
        metrics_fpath = os.path.join(hparams["exp_dir"], "metrics.pkl")
        with open(metrics_fpath, "wb",) as handle:
            pickle.dump(
                {
                    "loss": test_loss,
                    "f1": test_f1,
                    "acc": test_acc,
                    "auc": test_auc,
                    "cm": test_cm,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )


def run_single_process(argv, tail_path, datasets):
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
    run_experiment(hparams, run_opts, datasets)


if __name__ == "__main__":
    argv = sys.argv[1:]
    # argv = ['train_BNCI2014001.yaml', '--data_folder', '/home/davide/Documents/data']
    # loading hparams to prepare the dataset and the data iterators
    hparams_file, run_opts, overrides = sb.core.parse_arguments(argv)
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    moabb_dataset = hparams["dataset"]
    moabb_dataset.subject_list = hparams["subject_list"]
    moabb_dataset.download(path=hparams["data_folder"])
    # defining data iterators to use
    data_its = hparams["data_iterators"]

    # summary(hparams['model'], hparams['input_shape'][1:])
    # defining the job list
    jobs = []
    for data_it in data_its:
        for i, (tail_path, datasets) in enumerate(
            data_it.prepare(
                moabb_dataset, hparams["batch_size"], hparams["valid_ratio"]
            )
        ):
            args = (argv, tail_path, datasets)
            p = mp.Process(target=run_single_process, args=args)
            jobs.append(p)
    # starting a fixed number of parallel processes at a time
    processes_start_idx = np.arange(len(jobs))[
        :: hparams["num_parallel_processes"]
    ]
    for start_idx in processes_start_idx:
        stop_idx = start_idx + hparams["num_parallel_processes"]
        stop_idx = min(stop_idx, len(jobs))

        for job_idx in np.arange(start_idx, stop_idx):
            jobs[job_idx].start()
        for job_idx in np.arange(start_idx, stop_idx):
            jobs[job_idx].join()
