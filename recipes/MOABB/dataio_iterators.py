"""
Data iterators defined for MOABB datasets and paradigms.
Different training strategies are implemented as different data iterators:
* Within-session;
* Leave-one-session-out;
* Cross-session;
* Leave-one-subject-out.

Author
------
Davide Borra, 2021
"""

import numpy as np
from moabb.datasets.base import BaseDataset
from moabb.paradigms.base import BaseParadigm
from sklearn.model_selection import StratifiedKFold
from itertools import islice
import torch
from torch.utils.data import TensorDataset, DataLoader
import os


def standardize(x, m=None, s=None):
    """This function standardizes input EEG signals (x) using mean value (m)
    and standard deviation (s)."""
    if m is None and s is None:
        m = np.mean(x, axis=(0, -1))
        s = np.std(x, axis=(0, -1))
    demeaned = x - m.reshape((1, m.shape[0], 1))
    standardized = demeaned / (1e-14 + s.reshape((1, s.shape[0], 1)))
    return standardized


def nth(iterable, n, default=None):
    """This function returns cross-validation train and test indices of the i-th fold."""
    return next(islice(iterable, n, None), default)


def get_idx_train_valid_classbalanced(idx_train, valid_ratio, y):
    """This function returns train and valid indices balanced across classes."""
    idx_train = np.array(idx_train)

    nclasses = y[idx_train].max() + 1
    idx_valid = []
    for c in range(nclasses):
        to_select_c = idx_train[np.where(y[idx_train] == c)[0]]
        tmp_idx_valid_c = np.random.choice(
            to_select_c,
            round(valid_ratio * to_select_c.shape[0]),
            replace=False,
        )
        idx_valid.extend(tmp_idx_valid_c)
    idx_valid = np.array(idx_valid)
    idx_train = np.setdiff1d(idx_train, idx_valid)
    return idx_train, idx_valid


def get_xy_and_meta(paradigm, dataset, idx):
    """This function returns EEG signals and the corresponding labels. In addition metadata are provided too."""
    x, y, metadata = paradigm.get_data(dataset, idx, True)
    x = x.get_data()[..., :-1]
    y = [dataset.event_id[yy] for yy in y]
    y = np.array(y)
    y -= 1
    return x, y, metadata


class WithinSession(object):
    """Within session iterator for MOABB datasets.
    Designing a within-subject and within-session iterations on the dataset of a specific paradigm.
    For each subject and for each session, the training and test sets are defined using a stratified cross-validation partitioning.
    The validation set is sampled from the training set.
    All sets are extracted balanced across subjects, sessions and classes.

    Arguments
    ---------
    paradigm : moabb.paradigms.??
        Target MOABB paradigm to use.
    nfolds: int
        Number of cross-validation folds.
    seed: int
        Seed for random number generators.
    """

    def __init__(self, paradigm, nfolds, seed):
        if not isinstance(paradigm, BaseParadigm):
            raise (ValueError("paradigm must be an Paradigm instance"))
        self.paradigm = paradigm
        self.nfolds = nfolds
        self.skf = StratifiedKFold(n_splits=self.nfolds)
        self.iterator_tag = "within-session"
        np.random.seed(seed)

    def prepare(self, dataset, batch_size, valid_ratio):
        """This function returns the pre-processed datasets (training, validation and test sets)
        Arguments
        ---------
        dataset : moabb.datasets.??
            Target MOABB dataset to use.
        batch_size: int
            Batch size.
        valid_ratio: float
            Ratio of the training set to use as validation data.
        """

        if not (isinstance(dataset, BaseDataset)):
            raise (
                ValueError("datasets must only contains dataset " "instance")
            )

        # iterate over subjects
        for subject in dataset.subject_list:
            # loading subject-specific data
            x, y, metadata = get_xy_and_meta(self.paradigm, dataset, [subject])
            # iterate over sessions
            for session in np.unique(metadata.session):
                # obtaining indices for the current session
                idx = np.where(metadata.session == session)[0]
                for sel_fold in range(self.nfolds):
                    # obtaining indices for train and test sets
                    idx_train_, idx_test_ = nth(
                        self.skf.split(np.arange(idx.shape[0]), y[idx]),
                        sel_fold,
                    )
                    idx_train = idx[idx_train_]
                    idx_test = idx[idx_test_]

                    # validation set definition (equal proportion btw classes)
                    idx_train, idx_valid = get_idx_train_valid_classbalanced(
                        idx_train, valid_ratio, y
                    )

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

                    # swap axes: from (N_examples, C, T) to (N_examples, T, C)
                    x_train = np.swapaxes(x_train, -1, -2)
                    x_valid = np.swapaxes(x_valid, -1, -2)
                    x_test = np.swapaxes(x_test, -1, -2)

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
                        ds, batch_size=batch_size, pin_memory=True,
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
                        ds, batch_size=batch_size, pin_memory=True,
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
                        ds, batch_size=batch_size, pin_memory=True,
                    )
                    datasets = {}
                    datasets["train"] = train_loader
                    datasets["valid"] = valid_loader
                    datasets["test"] = test_loader

                    tail_path = os.path.join(
                        self.iterator_tag,
                        str(subject).zfill(3),
                        str(session).zfill(2),
                        str(sel_fold).zfill(2),
                    )
                    yield (tail_path, datasets)


class LeaveOneSessionOut(object):
    """Leave one session out iterator for MOABB datasets.
    Designing within-subject, cross-session and session-agnostic iterations on the dataset for a specific paradigm.
    For each subject, one session is held back as test set and the remaining ones are used to train neural networks.
    The validation set can be sampled from a separate (and held back) session if enough sessions are available; otherwise, the validation set is sampled from the training set.
    All sets are extracted balanced across subjects, sessions and classes.

    Arguments
    ---------
    paradigm : moabb.paradigms.??
        Target MOABB paradigm to use.
    seed: int
        Seed for random number generators.
    """

    def __init__(self, paradigm, seed):
        if not isinstance(paradigm, BaseParadigm):
            raise (ValueError("paradigm must be an Paradigm instance"))
        self.paradigm = paradigm
        self.iterator_tag = "leave-one-session-out"
        np.random.seed(seed)

    def prepare(
        self,
        dataset,
        batch_size,
        valid_ratio,
        sample_valid_from_train_examples=True,
    ):
        """This function returns the pre-processed datasets (training, validation and test sets)
        Arguments
        ---------
        dataset : moabb.datasets.??
            Target MOABB dataset to use.
        batch_size: int
            Batch size.
        valid_ratio: float
            Ratio of the training set to use as validation data.
        sample_valid_from_train_examples: bool
            Force the extraction of the validation examples from the training examples.
        """
        if not (isinstance(dataset, BaseDataset)):
            raise (
                ValueError("datasets must only contains dataset " "instance")
            )

        # iterate over subjects
        for subject in dataset.subject_list:
            # loading subject-specific data
            x, y, metadata = get_xy_and_meta(self.paradigm, dataset, [subject])

            if np.unique(metadata.session).shape[0] < 2:
                raise (
                    ValueError(
                        "The number of sessions in the dataset must be >= 2 for leave-one-session-out iterations"
                    )
                )
            for session in np.unique(metadata.session):
                sess_id_test = [session]
                sess_id_train = np.setdiff1d(
                    np.unique(metadata.session), sess_id_test
                )
                sess_id_valid = np.random.choice(
                    sess_id_train,
                    round(valid_ratio * sess_id_train.shape[0]),
                    replace=False,
                )
                sess_id_train = list(sess_id_train)
                sess_id_valid = list(sess_id_valid)
                # Check whether and if it is possible re-define the validation set from a separate held back session
                if (
                    not sample_valid_from_train_examples
                    and len(sess_id_valid) != 0
                ):
                    sess_id_train = np.setdiff1d(sess_id_train, sess_id_valid)
                    sess_id_train = list(sess_id_train)

                    idx_train = []
                    for s in sess_id_train:
                        # obtaining indices for the current session
                        idx = np.where(metadata.session == s)[0]
                        idx_train.extend(idx)
                    idx_valid = []
                    for s in sess_id_valid:
                        # obtaining indices for the current session
                        idx = np.where(metadata.session == s)[0]
                        idx_valid.extend(idx)
                else:
                    # iterate over sessions to accumulate session train and valid examples in a balanced way across sessions
                    idx_train, idx_valid = [], []
                    for s in sess_id_train:
                        # obtaining indices for the current session
                        idx = np.where(metadata.session == s)[0]
                        # validation set definition (equal proportion btw classes)
                        (
                            tmp_idx_train,
                            tmp_idx_valid,
                        ) = get_idx_train_valid_classbalanced(
                            idx, valid_ratio, y
                        )
                        idx_train.extend(tmp_idx_train)
                        idx_valid.extend(tmp_idx_valid)

                idx_test = []
                for s in sess_id_test:
                    # obtaining indices for the current session
                    idx = np.where(metadata.session == s)[0]
                    idx_test.extend(idx)

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

                # swap axes: from (N_examples, C, T) to (N_examples, T, C)
                x_train = np.swapaxes(x_train, -1, -2)
                x_valid = np.swapaxes(x_valid, -1, -2)
                x_test = np.swapaxes(x_test, -1, -2)

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
                    ds, batch_size=batch_size, pin_memory=True
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
                    ds, batch_size=batch_size, pin_memory=True
                )

                inps = torch.Tensor(
                    x_test.reshape(
                        (x_test.shape[0], x_test.shape[1], x_test.shape[2], 1,)
                    )
                )
                tgts = torch.tensor(y_test, dtype=torch.long)
                ds = TensorDataset(inps, tgts)
                test_loader = DataLoader(
                    ds, batch_size=batch_size, pin_memory=True
                )
                datasets = {}
                datasets["train"] = train_loader
                datasets["valid"] = valid_loader
                datasets["test"] = test_loader
                tail_path = os.path.join(
                    self.iterator_tag,
                    str(subject).zfill(3),
                    str(session).zfill(2),
                )
                yield (tail_path, datasets)


class CrossSession(object):
    """Cross session iterator for MOABB datasets.
    Designing a within-subject and cross-session iterations on the dataset for a specific paradigm.
    For each subject, all session' signals are merged together.
    Training and test sets are defined using a stratified cross-validation partitioning.
    The validation set are sampled from the training set.
    All sets are extracted balanced across subjects, sessions and classes.

    Arguments
    ---------
    paradigm : moabb.paradigms.??
        Target MOABB paradigm to use.
    nfolds: int
        Number of cross-validation folds.
    seed: int
        Seed for random number generators.
    """

    def __init__(self, paradigm, nfolds, seed):
        if not isinstance(paradigm, BaseParadigm):
            raise (ValueError("paradigm must be an Paradigm instance"))
        self.paradigm = paradigm
        self.nfolds = nfolds
        self.skf = StratifiedKFold(n_splits=self.nfolds)
        self.iterator_tag = "cross-session"
        np.random.seed(seed)

    def prepare(self, dataset, batch_size, valid_ratio):
        """This function returns the pre-processed datasets (training, validation and test sets)
        Arguments
        ---------
        dataset : moabb.datasets.??
            Target MOABB dataset to use.
        batch_size: int
            Batch size.
        valid_ratio: float
            Ratio of the training set to use as validation data.
        """
        if not (isinstance(dataset, BaseDataset)):
            raise (
                ValueError("datasets must only contains dataset " "instance")
            )

        # iterate over subjects
        for subject in dataset.subject_list:
            # loading subject-specific data
            x, y, metadata = get_xy_and_meta(self.paradigm, dataset, [subject])
            # iterate over folds
            for sel_fold in range(self.nfolds):
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
                    (
                        tmp_idx_train,
                        tmp_idx_valid,
                    ) = get_idx_train_valid_classbalanced(
                        tmp_idx_train, valid_ratio, y
                    )

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

                # swap axes: from (N_examples, C, T) to (N_examples, T, C)
                x_train = np.swapaxes(x_train, -1, -2)
                x_valid = np.swapaxes(x_valid, -1, -2)
                x_test = np.swapaxes(x_test, -1, -2)

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
                    ds, batch_size=batch_size, pin_memory=True
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
                    ds, batch_size=batch_size, pin_memory=True
                )

                inps = torch.Tensor(
                    x_test.reshape(
                        (x_test.shape[0], x_test.shape[1], x_test.shape[2], 1,)
                    )
                )
                tgts = torch.tensor(y_test, dtype=torch.long)
                ds = TensorDataset(inps, tgts)
                test_loader = DataLoader(
                    ds, batch_size=batch_size, pin_memory=True
                )
                datasets = {}
                datasets["train"] = train_loader
                datasets["valid"] = valid_loader
                datasets["test"] = test_loader
                tail_path = os.path.join(
                    self.iterator_tag,
                    str(subject).zfill(3),
                    str(sel_fold).zfill(2),
                )
                yield (tail_path, datasets)


class LeaveOneSubjectOut(object):
    """Leave one subject out iterator for MOABB datasets.
    Designing cross-subject, cross-session and subject-agnostic iterations on the dataset for a specific paradigm.
    One subject is held back as test set and the remaining ones are used to train neural networks.
    The validation set can be sampled from a separate (and held back) subject if enough subjects are available; otherwise, the validation set is sampled from the training set.
    All sets are extracted balanced across subjects, sessions and classes.

    Arguments
    ---------
    paradigm : moabb.paradigms.??
        Target MOABB paradigm to use.
    seed: int
        Seed for random number generators.
    """

    def __init__(self, paradigm, seed):
        if not isinstance(paradigm, BaseParadigm):
            raise (ValueError("paradigm must be an Paradigm instance"))
        self.paradigm = paradigm
        self.iterator_tag = "leave-one-subject-out"
        np.random.seed(seed)

    def prepare(
        self,
        dataset,
        batch_size,
        valid_ratio,
        sample_valid_from_train_examples=True,
    ):
        """This function returns the pre-processed datasets (training, validation and test sets)
        Arguments
        ---------
        dataset : moabb.datasets.??
            Target MOABB dataset to use.
        batch_size: int
            Batch size.
        valid_ratio: float
            Ratio of the training set to use as validation data.
        sample_valid_from_train_examples: bool
            Force the extraction of the validation examples from the training examples.
        """
        if not (isinstance(dataset, BaseDataset)):
            raise (
                ValueError("datasets must only contains dataset " "instance")
            )
        if len(dataset.subject_list) < 2:
            raise (
                ValueError(
                    "The number of subjects in the dataset must be >= 2 for leave-one-subject-out iterations"
                )
            )
        # iterate over subjects
        for subject in dataset.subject_list:
            # identification of training, test and validation splits
            sbj_id_test = [subject]
            sbj_id_train = np.setdiff1d(dataset.subject_list, sbj_id_test)
            sbj_id_valid = np.random.choice(
                sbj_id_train,
                round(valid_ratio * sbj_id_train.shape[0]),
                replace=False,
            )
            sbj_id_valid = list(sbj_id_valid)

            # loading test set
            x_test, y_test, _ = get_xy_and_meta(
                self.paradigm, dataset, sbj_id_test
            )
            if not sample_valid_from_train_examples and len(sbj_id_valid) != 0:
                sbj_id_train = np.setdiff1d(sbj_id_train, sbj_id_valid)
                sbj_id_train = list(sbj_id_train)
                # loading training set
                x_train, y_train, _ = get_xy_and_meta(
                    self.paradigm, dataset, sbj_id_train
                )
                # loading validation set
                x_valid, y_valid, _ = get_xy_and_meta(
                    self.paradigm, dataset, sbj_id_valid
                )
            else:
                sbj_id_train = list(sbj_id_train)
                x_train, y_train, x_valid, y_valid = [], [], [], []
                for tmp_subject in sbj_id_train:
                    # loading training set
                    tmp_x_train, tmp_y_train, tmp_metadata = get_xy_and_meta(
                        self.paradigm, dataset, [tmp_subject]
                    )
                    # defining training and validation indices from subjects and sessions in a balanced way
                    idx_train, idx_valid = [], []
                    for session in np.unique(tmp_metadata.session):
                        idx = np.where(tmp_metadata.session == session)[0]
                        # validation set definition (equal proportion btw classes)
                        (
                            tmp_idx_train,
                            tmp_idx_valid,
                        ) = get_idx_train_valid_classbalanced(
                            idx, valid_ratio, tmp_y_train
                        )
                        idx_train.extend(tmp_idx_train)
                        idx_valid.extend(tmp_idx_valid)
                    idx_train = np.array(idx_train)
                    idx_valid = np.array(idx_valid)

                    tmp_x_valid = tmp_x_train[idx_train, ...]
                    tmp_y_valid = tmp_y_train[idx_train]
                    tmp_x_train = tmp_x_train[idx_valid, ...]
                    tmp_y_train = tmp_y_train[idx_valid]

                    x_train.extend(tmp_x_valid)
                    y_train.extend(tmp_y_valid)
                    x_valid.extend(tmp_x_train)
                    y_valid.extend(tmp_y_train)
                x_train = np.array(x_train)
                y_train = np.array(y_train)
                x_valid = np.array(x_valid)
                y_valid = np.array(y_valid)

            # computing statistics for standardization on the training set
            m = np.mean(x_train, axis=(0, -1))
            s = np.std(x_train, axis=(0, -1))

            # standardization
            x_train = standardize(x_train, m, s)
            x_valid = standardize(x_valid, m, s)
            x_test = standardize(x_test, m, s)

            # swap axes: from (N_examples, C, T) to (N_examples, T, C)
            x_train = np.swapaxes(x_train, -1, -2)
            x_valid = np.swapaxes(x_valid, -1, -2)
            x_test = np.swapaxes(x_test, -1, -2)

            # dataloaders
            inps = torch.Tensor(
                x_train.reshape(
                    (x_train.shape[0], x_train.shape[1], x_train.shape[2], 1,)
                )
            )
            tgts = torch.tensor(y_train, dtype=torch.long)
            ds = TensorDataset(inps, tgts)
            train_loader = DataLoader(
                ds, batch_size=batch_size, pin_memory=True
            )

            inps = torch.Tensor(
                x_valid.reshape(
                    (x_valid.shape[0], x_valid.shape[1], x_valid.shape[2], 1,)
                )
            )
            tgts = torch.tensor(y_valid, dtype=torch.long)
            ds = TensorDataset(inps, tgts)
            valid_loader = DataLoader(
                ds, batch_size=batch_size, pin_memory=True
            )

            inps = torch.Tensor(
                x_test.reshape(
                    (x_test.shape[0], x_test.shape[1], x_test.shape[2], 1,)
                )
            )
            tgts = torch.tensor(y_test, dtype=torch.long)
            ds = TensorDataset(inps, tgts)
            test_loader = DataLoader(ds, batch_size=batch_size, pin_memory=True)
            datasets = {}
            datasets["train"] = train_loader
            datasets["valid"] = valid_loader
            datasets["test"] = test_loader
            tail_path = os.path.join(self.iterator_tag, str(subject).zfill(3),)
            yield (tail_path, datasets)
