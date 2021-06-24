#!/usr/bin/python3
"""
Recipe for training a compact CNN to decode the P300 event from single EEG trials.
The CNN is based on EEGNet and the dataset is ERPCore-P3.
Reference to EEGNet: V. J. Lawhern et al., J Neural Eng 2018 (https://doi.org/10.1088/1741-2552/aace8c).
Reference to ERPCore: E. S. Kappenman et al., Neuroimage 2021 (https://doi.org/10.1016/j.neuroimage.2020.117465).

To run this recipe, at first you should download the dataset:
> python3 download_required_data.py --data_folder /path/to/ERPCore_P3

Finally, you can train a subject-specific decoder (e.g., for subject 4) to decode the P3 event from single EEG trials:
> python3 train.py p3_decoding.yaml --sbj_id 'sub-004' --data_folder '/path/to/ERPCore_P3' --output_folder '/path/to/ERPCore_P3_results'

Author
------
Davide Borra, 2021
"""

import pickle
import os
import torch as th
import logging
from hyperpyyaml import load_hyperpyyaml
from torch.nn import init
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from itertools import islice
from sklearn.metrics import f1_score, roc_auc_score
from prepare import load_and_preprocess_p3_erp_core
import sys
import speechbrain as sb


def nth(iterable, n, default=None):
    return next(islice(iterable, n, None), default)


def standardize(x, m, s):
    demeaned = x - m.reshape((1, m.shape[0], 1))
    standardized = demeaned / (1e-14 + s.reshape((1, s.shape[0], 1)))
    return standardized


def dataio_prepare(hparams):
    # loading subject-specific dataset
    x, y = load_and_preprocess_p3_erp_core(hparams)
    np.random.seed(hparams["seed"])
    skf = StratifiedKFold(n_splits=hparams["nfolds"])

    # obtaining indices for train and test sets
    idx_train, idx_test = nth(
        skf.split(np.arange(y.shape[0]), y), hparams["sel_fold"]
    )

    # extraction of the validation set (equal proportion for 0: non-P300 and 1: P300 classes)
    to_select_class0 = idx_train[np.where(y[idx_train] == 0)[0]]
    to_select_class1 = idx_train[np.where(y[idx_train] == 1)[0]]

    tmp_idx_valid0 = np.random.choice(
        to_select_class0, round(0.2 * to_select_class0.shape[0]), replace=False
    )
    tmp_idx_valid1 = np.random.choice(
        to_select_class1, round(0.2 * to_select_class1.shape[0]), replace=False
    )

    idx_valid = np.concatenate((tmp_idx_valid0, tmp_idx_valid1))
    idx_train = np.setdiff1d(idx_train, idx_valid)

    x_train = x[idx_train, ...]
    y_train = y[idx_train]
    x_valid = x[idx_valid, ...]
    y_valid = y[idx_valid]
    x_test = x[idx_test, ...]
    y_test = y[idx_test]

    # computing statistics for standardization on the training set
    m = np.mean(x_train, axis=(0, -1))
    s = np.std(x_train, axis=(0, -1))

    # standardization
    x_train = standardize(x_train, m, s)
    x_valid = standardize(x_valid, m, s)
    x_test = standardize(x_test, m, s)

    # dataloaders
    inps = th.Tensor(
        x_train.reshape(
            (x_train.shape[0], 1, x_train.shape[1], x_train.shape[2])
        )
    )
    tgts = th.tensor(y_train, dtype=th.long)
    dataset = TensorDataset(inps, tgts)
    train_loader = DataLoader(
        dataset, batch_size=hparams["batch_size"], pin_memory=True
    )

    inps = th.Tensor(
        x_valid.reshape(
            (x_valid.shape[0], 1, x_valid.shape[1], x_valid.shape[2])
        )
    )
    tgts = th.tensor(y_valid, dtype=th.long)
    dataset = TensorDataset(inps, tgts)
    valid_loader = DataLoader(
        dataset, batch_size=hparams["batch_size"], pin_memory=True
    )

    inps = th.Tensor(
        x_test.reshape((x_test.shape[0], 1, x_test.shape[1], x_test.shape[2]))
    )
    tgts = th.tensor(y_test, dtype=th.long)
    dataset = TensorDataset(inps, tgts)
    test_loader = DataLoader(
        dataset, batch_size=hparams["batch_size"], pin_memory=True
    )
    datasets = {}
    datasets["train"] = train_loader
    datasets["valid"] = valid_loader
    datasets["test"] = test_loader
    return datasets


def initialize_module(module):
    for mod in module.modules():
        if hasattr(mod, "weight"):
            if not ("BatchNorm" in mod.__class__.__name__):
                init.xavier_uniform_(mod.weight, gain=1)
            else:
                init.constant_(mod.weight, 1)
        if hasattr(mod, "bias"):
            if mod.bias is not None:
                init.constant_(mod.bias, 0)


class P3Brain(sb.Brain):
    def compute_forward(self, batch, stage):
        return self.modules.model(batch[0].cuda())

    def compute_objectives(self, predictions, batch, stage):
        loss = self.hparams.loss(
            predictions,
            batch[1].cuda(),
            weight=th.Tensor(self.hparams.class_weight).cuda(),
        )
        if stage != sb.Stage.TRAIN:
            tmp_preds = th.exp(predictions)
            self.preds.extend(tmp_preds.detach().cpu().numpy())
            self.targets.extend(batch[1].detach().cpu().numpy())
        return loss

    def on_fit_start(self,):
        initialize_module(self.hparams.model)
        self.init_optimizers()
        self.metrics = {}
        self.metrics["loss"] = []
        self.metrics["f1"] = []
        self.metrics["auc"] = []

    def on_stage_start(self, stage, epoch=None):
        if stage != sb.Stage.TRAIN:
            self.preds = []
            self.targets = []

    def on_stage_end(self, stage, stage_loss, epoch=None):
        logger = logging.getLogger(__name__)
        if stage == sb.Stage.TRAIN:
            logger.info("Train loss:%.4f" % (stage_loss))
        else:
            preds = np.array(self.preds)
            y_pred = np.argmax(preds, axis=-1)
            y_true = self.targets
            f1 = f1_score(y_true=y_true, y_pred=y_pred)
            auc = roc_auc_score(y_true=y_true, y_score=preds[:, 1])
            self.last_eval_loss = stage_loss
            self.last_eval_f1 = float(f1)
            self.last_eval_auc = float(auc)
            set_info = "Valid " if epoch is not None else ""
            logger.info(
                set_info
                + "loss:%.4f; f-score:%.4f; auc: %.4f"
                % (self.last_eval_loss, self.last_eval_f1, self.last_eval_auc)
            )

            if epoch is not None:
                # track valid metric history
                self.metrics["loss"].append(self.last_eval_loss)
                self.metrics["f1"].append(self.last_eval_f1)
                self.metrics["auc"].append(self.last_eval_auc)
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
                if self.hparams.stopper.should_stop(
                    current=epoch,
                    current_metric=self.metrics[
                        self.hparams.target_valid_metric
                    ][-1],
                ):
                    self.hparams.epoch_counter.current = (
                        self.hparams.epoch_counter.limit
                    )


def run_single_fold(hparams, run_opts):
    # preparing dataset (loading all sets in RAM)
    datasets = dataio_prepare(hparams)
    checkpointer = sb.utils.checkpoints.Checkpointer(
        checkpoints_dir=hparams["exp_dir"],
        recoverables={
            "model": hparams["model"],
            "counter": hparams["epoch_counter"],
        },
    )
    brain = P3Brain(
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
        datasets["train"], progressbar=False, min_key=min_key, max_key=max_key
    )
    train_loss, train_f1, train_auc = (
        brain.last_eval_loss,
        brain.last_eval_f1,
        brain.last_eval_auc,
    )

    brain.evaluate(
        datasets["valid"], progressbar=False, min_key=min_key, max_key=max_key
    )
    valid_loss, valid_f1, valid_auc = (
        brain.last_eval_loss,
        brain.last_eval_f1,
        brain.last_eval_auc,
    )

    brain.evaluate(
        datasets["test"], progressbar=False, min_key=min_key, max_key=max_key
    )
    test_loss, test_f1, test_auc = (
        brain.last_eval_loss,
        brain.last_eval_f1,
        brain.last_eval_auc,
    )

    return [
        train_loss,
        train_f1,
        train_auc,
        valid_loss,
        valid_f1,
        valid_auc,
        test_loss,
        test_f1,
        test_auc,
    ]


if __name__ == "__main__":
    argv = sys.argv[1:]
    NFOLDS = 5

    fold_metrics = []
    # Within-subject training: cross-validation loop
    for sel_fold in range(NFOLDS):
        # Load hyperparameters file with command-line overrides
        hparams_file, run_opts, overrides = sb.core.parse_arguments(argv)
        with open(hparams_file) as fin:
            hparams = load_hyperpyyaml(fin, overrides)
        hparams["sel_fold"] = sel_fold
        hparams["nfolds"] = NFOLDS
        hparams["exp_dir"] = os.path.join(
            hparams["output_folder"],
            hparams["sbj_id"],
            "fold" + str(hparams["sel_fold"]).zfill(2),
        )
        # creating experiment directory (specific subject and fold)
        sb.create_experiment_directory(
            experiment_directory=hparams["exp_dir"],
            hyperparams_to_save=hparams_file,
            overrides=overrides,
        )
        tmp_metrics = run_single_fold(hparams, run_opts)
        fold_metrics.append(tmp_metrics)
    fold_metrics = np.array(fold_metrics)
    # saving cv results
    with open(
        os.path.join(
            hparams["output_folder"], hparams["sbj_id"], "metrics.pkl"
        ),
        "wb",
    ) as handle:
        pickle.dump(
            {"metrics": fold_metrics}, handle, protocol=pickle.HIGHEST_PROTOCOL
        )
