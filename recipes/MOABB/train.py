#!/usr/bin/python
"""
Recipe for training neural networks to decode single EEG trials with different paradigms on MOABB datasets.
See the supported datasets and paradigms at http://moabb.neurotechx.com/docs/api.html.

To run this recipe (e.g., architecture: EEGNet; dataset: BNCI2014001):

    > python3 train.py hparams/EEGNet_BNCI2014001.yaml --data_folder '/path/to/BNCI2014001'

The dataset will be automatically downloaded in the specified folder.

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
import logging
import multiprocessing as mp
import speechbrain as sb

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
        print(inputs.shape)
        torch.save(inputs, "example.pt")
        import sys

        sys.exit(0)
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
            # From log to linear predictions
            tmp_preds = torch.exp(predictions)
            self.preds.extend(tmp_preds.detach().cpu().numpy())
            self.targets.extend(batch[1].detach().cpu().numpy())
        else:
            self.hparams.lr_annealing.on_batch_end(self.optimizer)
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
                # Learning rate scheduler
                old_lr, new_lr = self.hparams.lr_annealing(epoch)
                sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
                self.hparams.train_logger.log_stats(
                    stats_meta={"epoch": epoch, "lr": old_lr},
                    train_stats={"loss": self.train_loss},
                    valid_stats=last_eval_stats,
                )
                if epoch == 1:
                    self.best_eval_stats = last_eval_stats

                # The current model is saved if it is the best or the last
                is_best = self.check_if_best(
                    last_eval_stats,
                    self.best_eval_stats,
                    keys=self.hparams.test_keys,
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
                    self.checkpointer.save_and_keep_only(
                        meta={
                            "loss": self.last_eval_loss,
                            "f1": self.last_eval_f1,
                            "auc": self.last_eval_auc,
                            "acc": self.last_eval_acc,
                        },
                        num_to_keep=self.hparams.avg_models,
                    )

            elif stage == sb.Stage.TEST:
                self.hparams.train_logger.log_stats(
                    stats_meta={
                        "epoch loaded": self.hparams.epoch_counter.current
                    },
                    test_stats=last_eval_stats,
                )
                # save the averaged checkpoint at the end of the evaluation stage
                # delete the rest of the intermediate checkpoints
                # ACC is set to 1.1 so checkpointer only keeps the averaged checkpoint
                if self.hparams.avg_models > 1:
                    self.checkpointer.save_and_keep_only(
                        meta={"acc": 1.1, "epoch": epoch},
                        max_keys=["acc"],
                        num_to_keep=1,
                    )

    def on_evaluate_start(self, max_key=None, min_key=None):
        """perform checkpoint averge if needed"""
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
        self,
        last_eval_stats,
        best_eval_stats,
        keys=["loss", "f1", "auc", "acc"],
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
    # evaluation after loading model using different keys
    results = {}
    keys = hparams["test_keys"]
    for key in keys:
        results[key] = {}
        min_key, max_key = None, None
        if key == "loss":
            min_key = key
        else:
            max_key = key
        # perform evaluation
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
        results[key]["loss"] = test_loss
        results[key]["f1"] = test_f1
        results[key]["acc"] = test_acc
        results[key]["auc"] = test_auc
        results[key]["cm"] = test_cm
    # saving metrics on the test set in a pickle file
    metrics_fpath = os.path.join(hparams["exp_dir"], "metrics.pkl")
    with open(metrics_fpath, "wb") as handle:
        pickle.dump(results[key], handle, protocol=pickle.HIGHEST_PROTOCOL)


def run_single_process(argv, tail_path, datasets):
    """This function wraps up a single process (e.g., the training of a single cross-validation fold
    with a specific hparams file and experiment directory)"""
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
    # loading hparams to prepare the dataset and the data iterators
    hparams_file, run_opts, overrides = sb.core.parse_arguments(argv)
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    moabb_dataset = hparams["dataset"]
    moabb_dataset.subject_list = hparams["subject_list"]
    moabb_dataset.download(path=hparams["data_folder"])
    # defining data iterators to use
    data_its = hparams["data_iterators"]

    # defining the job list
    jobs = []
    print("Prepare dataset iterators...")
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

    print("Training experiments (in parallel different iterators)...")
    for start_idx in processes_start_idx:
        stop_idx = start_idx + hparams["num_parallel_processes"]
        stop_idx = min(stop_idx, len(jobs))

        for job_idx in np.arange(start_idx, stop_idx):
            jobs[job_idx].start()
        for job_idx in np.arange(start_idx, stop_idx):
            jobs[job_idx].join()
