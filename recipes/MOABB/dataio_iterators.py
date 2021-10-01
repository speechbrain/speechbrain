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


class WithinSession(object):
    """Within session iterator for MOABB datasets.
    Designing a within-subject and within-session iterations on the dataset of a specific paradigm.
    For each subject, it performs a stratified K-fold cross-validation on each session.

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
            x, y, metadata = self.paradigm.get_data(dataset, [subject], True)
            x = x.get_data()[..., :-1]
            y = [dataset.event_id[yy] for yy in y]
            y = np.array(y)
            y -= 1
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


class CrossSession(object):
    """Cross session iterator for MOABB datasets.
    Designing a within-subject and cross-session iterations on the dataset of a specific paradigm.
    For each subject, it performs a stratified K-fold cross-validation merging together all available sessions.

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
            x, y, metadata = self.paradigm.get_data(dataset, [subject], True)
            x = x.get_data()[..., :-1]
            y = [dataset.event_id[yy] for yy in y]
            y = np.array(y)
            y -= 1

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


class LeaveOneSessionOut(object):
    """Leave one session out iterator for MOABB datasets.
    Designing within-subject iterations on the dataset of a specific paradigm.
    In these iterations, each session is held back as test set and the other sessions as training and validation sets.
    Training and validation sets are obtained by merging together multiple sessions.
    If only 2 sessions are available (impossible to sample separate validation sessions), the validation set is sampled from the training set.

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
            x, y, metadata = self.paradigm.get_data(dataset, [subject], True)
            x = x.get_data()[..., :-1]
            y = [dataset.event_id[yy] for yy in y]
            y = np.array(y)
            y -= 1

            if np.unique(metadata.session).shape[0] < 2:
                raise (
                    ValueError(
                        "The number of sessions in the dataset must be >= 2 for leave-one-session-out iterations"
                    )
                )
            for session in np.unique(metadata.session):
                tmp_idx_test = [session]
                tmp_idx_train = np.setdiff1d(
                    np.unique(metadata.session), tmp_idx_test
                )

                tmp_idx_valid = np.random.choice(
                    tmp_idx_train,
                    round(valid_ratio * tmp_idx_train.shape[0]),
                    replace=False,
                )

                tmp_idx_train = np.setdiff1d(tmp_idx_train, tmp_idx_valid)
                tmp_idx_train = list(tmp_idx_train)
                tmp_idx_valid = list(tmp_idx_valid)

                idx_train = []
                for s in tmp_idx_train:
                    # obtaining indices for the current session
                    idx = np.where(metadata.session == s)[0]
                    idx_train.extend(idx)

                if len(tmp_idx_valid) != 0:
                    idx_valid = []
                    for s in tmp_idx_valid:
                        # obtaining indices for the current session
                        idx = np.where(metadata.session == s)[0]
                        idx_valid.extend(idx)
                else:
                    # if only 2 sessions are available extract validation examples from the single training session
                    idx_train, idx_valid = get_idx_train_valid_classbalanced(
                        idx_train, valid_ratio, y
                    )
                idx_test = []
                for s in tmp_idx_test:
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


class LeaveOneSubjectOut(object):
    """Leave one subject out iterator for MOABB datasets.
    Designing cross-subject iterations on the dataset of a specific paradigm.
    In these iterations, each subject is held back as test set and the other subjects as training and validation sets.
    Training and validation sets are obtained by merging together multiple subjects.
    If only 2 subjects are available (impossible to sample separate validation subject), the validation set is sampled from the training set.

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
        if len(dataset.subject_list) < 2:
            raise (
                ValueError(
                    "The number of subjects in the dataset must be >= 2 for leave-one-subject-out iterations"
                )
            )
        # iterate over subjects
        for subject in dataset.subject_list:
            # identification of training, test and validation splits
            idx_test = [subject]
            idx_train = np.setdiff1d(dataset.subject_list, idx_test)
            idx_valid = np.random.choice(
                idx_train,
                round(valid_ratio * idx_train.shape[0]),
                replace=False,
            )
            idx_train = np.setdiff1d(idx_train, idx_valid)
            idx_train = list(idx_train)
            idx_valid = list(idx_valid)

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

            if len(idx_valid) != 0:
                # loading validation set
                x_valid, y_valid, _ = self.paradigm.get_data(
                    dataset, idx_valid, True
                )
                x_valid = x_valid.get_data()[..., :-1]
                y_valid = [dataset.event_id[yy] for yy in y_valid]
                y_valid = np.array(y_valid)
                y_valid -= 1
            else:
                # if only 2 subjects are available extract validation examples from the single training subject
                (
                    tmp_idx_train,
                    tmp_idx_valid,
                ) = get_idx_train_valid_classbalanced(
                    np.arange(x_train.shape[0]), valid_ratio, y_train
                )

                x_valid = x_train[tmp_idx_valid, ...]
                y_valid = y_train[tmp_idx_valid]
                x_train = x_train[tmp_idx_train, ...]
                y_train = y_train[tmp_idx_train]

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
