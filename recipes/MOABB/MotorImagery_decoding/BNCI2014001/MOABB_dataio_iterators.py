import numpy as np
from moabb.datasets.base import BaseDataset
from moabb.paradigms.base import BaseParadigm
from sklearn.model_selection import StratifiedKFold
from itertools import islice
import torch
from torch.utils.data import TensorDataset, DataLoader
import os


def standardize(x, m, s):
    """This function standardizes input EEG signals (x) using mean value (m)
    and standard deviation (s)."""
    demeaned = x - m.reshape((1, m.shape[0], 1))
    standardized = demeaned / (1e-14 + s.reshape((1, s.shape[0], 1)))
    return standardized


def nth(iterable, n, default=None):
    """This function returns cross-validation train and test indices of the i-th fold."""
    return next(islice(iterable, n, None), default)


class WithinSession(object):
    """Within Session """

    def __init__(self, paradigm, hparams):
        if not isinstance(paradigm, BaseParadigm):
            raise (ValueError("paradigm must be an Paradigm instance"))
        self.paradigm = paradigm
        self.hparams = hparams
        np.random.seed(hparams["seed"])
        self.skf = StratifiedKFold(n_splits=hparams["nfolds"])
        self.iterator_tag = "within-session"

    def prepare(self, dataset):
        if not (isinstance(dataset, BaseDataset)):
            raise (
                ValueError("datasets must only contains dataset " "instance")
            )

        # iterate over subjects
        for subject in dataset.subject_list:
            # loading subject-specific data
            x, y, metadata = self.paradigm.get_data(dataset, [subject], True)
            x = x.get_data()[..., :-1]
            y = [dataset.event_id[yy] for yy in y]
            y = np.array(y)
            y -= 1
            # iterate over sessions
            for session in np.unique(metadata.session):
                # obtaining indices for the current session
                idx = np.where(metadata.session == session)[0]
                for sel_fold in range(self.hparams["nfolds"]):
                    # obtaining indices for train and test sets
                    idx_train_, idx_test_ = nth(
                        self.skf.split(np.arange(idx.shape[0]), y[idx]),
                        sel_fold,
                    )
                    idx_train = idx[idx_train_]
                    idx_test = idx[idx_test_]
                    # validation set definition (equal proportion btw classes)
                    nclasses = y.max() + 1
                    idx_valid = []
                    for c in range(nclasses):
                        to_select_c = idx_train[np.where(y[idx_train] == c)[0]]
                        tmp_idx_valid_c = np.random.choice(
                            to_select_c,
                            round(
                                self.hparams["valid_ratio"]
                                * to_select_c.shape[0]
                            ),
                            replace=False,
                        )
                        idx_valid.extend(tmp_idx_valid_c)
                    idx_valid = np.array(idx_valid)
                    idx_train = np.setdiff1d(idx_train, idx_valid)
                    # computing statistics for standardization on the training set
                    m = np.mean(x[idx_train, ...], axis=(0, -1))
                    s = np.std(x[idx_train, ...], axis=(0, -1))

                    x_train = x[idx_train, ...]
                    y_train = y[idx_train]
                    x_valid = x[idx_valid, ...]
                    y_valid = y[idx_valid]
                    x_test = x[idx_test, ...]
                    y_test = y[idx_test]
                    # standardization
                    x_train = standardize(x_train, m, s)
                    x_valid = standardize(x_valid, m, s)
                    x_test = standardize(x_test, m, s)

                    # dataloaders
                    inps = torch.Tensor(
                        x_train.reshape(
                            (
                                x_train.shape[0],
                                x_train.shape[1],
                                x_train.shape[2],
                                1,
                            )
                        )
                    )
                    tgts = torch.tensor(y_train, dtype=torch.long)
                    ds = TensorDataset(inps, tgts)
                    train_loader = DataLoader(
                        ds,
                        batch_size=self.hparams["batch_size"],
                        pin_memory=True,
                    )

                    inps = torch.Tensor(
                        x_valid.reshape(
                            (
                                x_valid.shape[0],
                                x_valid.shape[1],
                                x_valid.shape[2],
                                1,
                            )
                        )
                    )
                    tgts = torch.tensor(y_valid, dtype=torch.long)
                    ds = TensorDataset(inps, tgts)
                    valid_loader = DataLoader(
                        ds,
                        batch_size=self.hparams["batch_size"],
                        pin_memory=True,
                    )

                    inps = torch.Tensor(
                        x_test.reshape(
                            (
                                x_test.shape[0],
                                x_test.shape[1],
                                x_test.shape[2],
                                1,
                            )
                        )
                    )
                    tgts = torch.tensor(y_test, dtype=torch.long)
                    ds = TensorDataset(inps, tgts)
                    test_loader = DataLoader(
                        ds,
                        batch_size=self.hparams["batch_size"],
                        pin_memory=True,
                    )
                    datasets = {}
                    datasets["train"] = train_loader
                    datasets["valid"] = valid_loader
                    datasets["test"] = test_loader

                    exp_dir = os.path.join(
                        self.hparams["output_folder"],
                        self.iterator_tag,
                        str(subject).zfill(3),
                        str(session).zfill(2),
                        str(sel_fold).zfill(2),
                    )
                    yield (exp_dir, datasets)


class CrossSession(object):
    """Cross Session """

    def __init__(self, paradigm, hparams):
        if not isinstance(paradigm, BaseParadigm):
            raise (ValueError("paradigm must be an Paradigm instance"))
        self.paradigm = paradigm
        self.hparams = hparams
        np.random.seed(hparams["seed"])
        self.skf = StratifiedKFold(n_splits=hparams["nfolds"])
        self.iterator_tag = "cross-session"

    def prepare(self, dataset):
        if not (isinstance(dataset, BaseDataset)):
            raise (
                ValueError("datasets must only contains dataset " "instance")
            )

        # iterate over subjects
        for subject in dataset.subject_list:
            # loading subject-specific data
            x, y, metadata = self.paradigm.get_data(dataset, [subject], True)
            x = x.get_data()[..., :-1]
            y = [dataset.event_id[yy] for yy in y]
            y = np.array(y)
            y -= 1
            # iterate over folds
            for sel_fold in range(self.hparams["nfolds"]):

                idx_train = []
                idx_valid = []
                idx_test = []
                # iterate over sessions to accumulate session examples in a balanced way across sessions
                for session in np.unique(metadata.session):
                    # obtaining indices for the current session
                    idx = np.where(metadata.session == session)[0]
                    # obtaining indices for train and test sets
                    idx_train_, idx_test_ = nth(
                        self.skf.split(np.arange(idx.shape[0]), y[idx]),
                        sel_fold,
                    )
                    tmp_idx_train = idx[idx_train_]
                    tmp_idx_test = idx[idx_test_]
                    # validation set definition (equal proportion btw classes)
                    nclasses = y.max() + 1
                    tmp_idx_valid = []
                    for c in range(nclasses):
                        to_select_c = tmp_idx_train[
                            np.where(y[tmp_idx_train] == c)[0]
                        ]
                        tmp_idx_valid_c = np.random.choice(
                            to_select_c,
                            round(
                                self.hparams["valid_ratio"]
                                * to_select_c.shape[0]
                            ),
                            replace=False,
                        )
                        tmp_idx_valid.extend(tmp_idx_valid_c)
                    tmp_idx_valid = np.array(tmp_idx_valid)
                    tmp_idx_train = np.setdiff1d(tmp_idx_train, tmp_idx_valid)
                    idx_train.extend(tmp_idx_train)
                    idx_valid.extend(tmp_idx_valid)
                    idx_test.extend(tmp_idx_test)
                idx_train = np.array(idx_train)
                idx_valid = np.array(idx_valid)
                idx_test = np.array(idx_test)

                # computing statistics for standardization on the training set
                m = np.mean(x[idx_train, ...], axis=(0, -1))
                s = np.std(x[idx_train, ...], axis=(0, -1))

                x_train = x[idx_train, ...]
                y_train = y[idx_train]
                x_valid = x[idx_valid, ...]
                y_valid = y[idx_valid]
                x_test = x[idx_test, ...]
                y_test = y[idx_test]
                # standardization
                x_train = standardize(x_train, m, s)
                x_valid = standardize(x_valid, m, s)
                x_test = standardize(x_test, m, s)

                # dataloaders
                inps = torch.Tensor(
                    x_train.reshape(
                        (
                            x_train.shape[0],
                            x_train.shape[1],
                            x_train.shape[2],
                            1,
                        )
                    )
                )
                tgts = torch.tensor(y_train, dtype=torch.long)
                ds = TensorDataset(inps, tgts)
                train_loader = DataLoader(
                    ds, batch_size=self.hparams["batch_size"], pin_memory=True
                )

                inps = torch.Tensor(
                    x_valid.reshape(
                        (
                            x_valid.shape[0],
                            x_valid.shape[1],
                            x_valid.shape[2],
                            1,
                        )
                    )
                )
                tgts = torch.tensor(y_valid, dtype=torch.long)
                ds = TensorDataset(inps, tgts)
                valid_loader = DataLoader(
                    ds, batch_size=self.hparams["batch_size"], pin_memory=True
                )

                inps = torch.Tensor(
                    x_test.reshape(
                        (x_test.shape[0], x_test.shape[1], x_test.shape[2], 1,)
                    )
                )
                tgts = torch.tensor(y_test, dtype=torch.long)
                ds = TensorDataset(inps, tgts)
                test_loader = DataLoader(
                    ds, batch_size=self.hparams["batch_size"], pin_memory=True
                )
                datasets = {}
                datasets["train"] = train_loader
                datasets["valid"] = valid_loader
                datasets["test"] = test_loader
                exp_dir = os.path.join(
                    self.hparams["output_folder"],
                    self.iterator_tag,
                    str(subject).zfill(3),
                    str(sel_fold).zfill(2),
                )
                yield (exp_dir, datasets)


class LeaveOneSubjectOut(object):
    """Leave one subject out"""

    def __init__(self, paradigm, hparams):
        if not isinstance(paradigm, BaseParadigm):
            raise (ValueError("paradigm must be an Paradigm instance"))
        self.paradigm = paradigm
        self.hparams = hparams
        np.random.seed(hparams["seed"])
        self.skf = StratifiedKFold(n_splits=hparams["nfolds"])
        self.iterator_tag = "leave-one-subject-out"

    def prepare(self, dataset):
        if not (isinstance(dataset, BaseDataset)):
            raise (
                ValueError("datasets must only contains dataset " "instance")
            )

        # iterate over subjects
        for subject in dataset.subject_list:
            # identification of training, test and validation splits
            idx_train = np.setdiff1d(dataset.subject_list, [subject])
            idx_valid = np.random.choice(
                idx_train,
                round(self.hparams["valid_ratio"] * idx_train.shape[0]),
                replace=False,
            )
            idx_train = np.setdiff1d(idx_train, idx_valid)
            idx_train = list(idx_train)
            idx_valid = list(idx_valid)
            idx_test = [subject]

            # loading training set
            x_train, y_train, _ = self.paradigm.get_data(
                dataset, idx_train, True
            )
            x_train = x_train.get_data()[..., :-1]
            y_train = [dataset.event_id[yy] for yy in y_train]
            y_train = np.array(y_train)
            y_train -= 1

            # loading test set
            x_test, y_test, _ = self.paradigm.get_data(dataset, idx_test, True)
            x_test = x_test.get_data()[..., :-1]
            y_test = [dataset.event_id[yy] for yy in y_test]
            y_test = np.array(y_test)
            y_test -= 1

            # loading validation set
            x_valid, y_valid, _ = self.paradigm.get_data(
                dataset, idx_valid, True
            )
            x_valid = x_valid.get_data()[..., :-1]
            y_valid = [dataset.event_id[yy] for yy in y_valid]
            y_valid = np.array(y_valid)
            y_valid -= 1

            # computing statistics for standardization on the training set
            m = np.mean(x_train, axis=(0, -1))
            s = np.std(x_train, axis=(0, -1))

            # standardization
            x_train = standardize(x_train, m, s)
            x_valid = standardize(x_valid, m, s)
            x_test = standardize(x_test, m, s)

            # dataloaders
            inps = torch.Tensor(
                x_train.reshape(
                    (x_train.shape[0], x_train.shape[1], x_train.shape[2], 1,)
                )
            )
            tgts = torch.tensor(y_train, dtype=torch.long)
            ds = TensorDataset(inps, tgts)
            train_loader = DataLoader(
                ds, batch_size=self.hparams["batch_size"], pin_memory=True
            )

            inps = torch.Tensor(
                x_valid.reshape(
                    (x_valid.shape[0], x_valid.shape[1], x_valid.shape[2], 1,)
                )
            )
            tgts = torch.tensor(y_valid, dtype=torch.long)
            ds = TensorDataset(inps, tgts)
            valid_loader = DataLoader(
                ds, batch_size=self.hparams["batch_size"], pin_memory=True
            )

            inps = torch.Tensor(
                x_test.reshape(
                    (x_test.shape[0], x_test.shape[1], x_test.shape[2], 1,)
                )
            )
            tgts = torch.tensor(y_test, dtype=torch.long)
            ds = TensorDataset(inps, tgts)
            test_loader = DataLoader(
                ds, batch_size=self.hparams["batch_size"], pin_memory=True
            )
            datasets = {}
            datasets["train"] = train_loader
            datasets["valid"] = valid_loader
            datasets["test"] = test_loader
            exp_dir = os.path.join(
                self.hparams["output_folder"],
                self.iterator_tag,
                str(subject).zfill(3),
            )
            yield (exp_dir, datasets)
