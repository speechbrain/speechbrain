"""
Data iterators defined for MOABB datasets and paradigms.
Different training strategies are implemented as different data iterators:
* Leave-one-session-out;
* Leave-one-subject-out.

Author
------
Davide Borra, 2022
"""

import numpy as np
import pickle
from itertools import islice
import torch
from torch.utils.data import TensorDataset, DataLoader
import os


def standardize(x, m=None, s=None):
    """This function standardizes each input EEG channel (x) using mean value (m)
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


def get_dataloader(batch_size,
                   xy_train,
                   xy_valid,
                   xy_test):
    x_train, y_train = xy_train[0], xy_train[1]
    x_valid, y_valid = xy_valid[0], xy_valid[1]
    x_test, y_test = xy_test[0], xy_test[1]

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

    return train_loader, valid_loader, test_loader


def load_data(paradigm, dataset, idx):
    """This function load EEG data and useful variables (labels, metadata channel names and sampling rate)."""
    x, labels, metadata = paradigm.get_data(dataset, idx, True)
    ch_names = x.info.ch_names
    srate = x.info.sfreq

    x = x.get_data()
    # forcing time steps to be = (tmax-tmin)*srate after mne resampling procedure
    if paradigm.resample is not None:
        xx = np.zeros(
            x.shape[:-1]
            + (int(paradigm.resample * (paradigm.tmax - paradigm.tmin)),)
        )
        stop = x.shape[-1]
        if x.shape[-1] > xx.shape[-1]:
            stop = xx.shape[-1]
        xx = x[..., :stop]
    else:
        xx = x.copy()[..., :-1]
    y = [dataset.event_id[yy] for yy in labels]
    y = np.array(y)
    y -= y.min()

    return xx, y, labels, metadata, ch_names, srate


def crop_signals(x, srate, interval_in, interval_out):
    time = np.arange(interval_in[0], interval_in[1], 1 / srate)
    idx_start = np.argmin(np.abs(time - interval_out[0]))
    idx_stop = np.argmin(np.abs(time - interval_out[1]))
    return x[..., idx_start:idx_stop]


class LeaveOneSessionOut(object):
    """Leave one session out iterator for MOABB datasets.
    Designing within-subject, cross-session and session-agnostic iterations on the dataset for a specific paradigm.
    For each subject, one session is held back as test set and the remaining ones are used to train neural networks.
    The validation set can be sampled from a separate (and held back) session if enough sessions are available; otherwise, the validation set is sampled from the training set.
    All sets are extracted balanced across subjects, sessions and classes.

    Arguments
    ---------
    seed: int
        Seed for random number generators.
    data_folder : str
        Folder where the dataset is stored.
    """

    def __init__(self, seed, data_folder):
        self.iterator_tag = "leave-one-session-out"
        self.data_folder = data_folder
        np.random.seed(seed)

    def prepare(
            self,
            dataset,
            batch_size,
            cached_dataset_tag,
            interval=None,
            valid_ratio=0.2,
            target_subject_idx=None,
            target_session_idx=None,
            apply_standardization=False,
    ):
        """This function returns the pre-processed datasets (training, validation and test sets)
        Arguments
        ---------
        dataset : moabb.datasets.??
            Target MOABB dataset to use.
        batch_size: int
            Batch size.
        cached_dataset_tag: str
            Tag of the prepared data.
        interval: list
            Interval to crop EEG signals (s).
        valid_ratio: float
            Ratio of the training set to use as validation data.
        target_subject_idx: int
            Index of the subject signals to load.
        target_session_idx: int
            Index of the session signals to load. None if leave-one-subject-out.
        apply_standardization: bool
            Apply standardization after data loading.
        """
        dataset_code = dataset.code

        tail = os.path.join(self.data_folder,
                            'MOABB_pickled',
                            dataset_code,
                            cached_dataset_tag)
        fpath = os.path.join(tail,
                             'sub-{0}.pkl'.format(str(dataset.subject_list[target_subject_idx]).zfill(3)))
        with open(fpath, 'rb') as f:
            data_dict = pickle.load(f)

        x = data_dict['x']
        y = data_dict['y']
        srate = data_dict['srate']
        original_interval = data_dict['interval']
        metadata = data_dict['metadata']
        if np.unique(metadata.session).shape[0] < 2:
            raise (
                ValueError(
                    "The number of sessions in the dataset must be >= 2 for leave-one-session-out iterations"
                )
            )
        sessions = np.unique(metadata.session)
        sess_id_test = [sessions[target_session_idx]]
        sess_id_train = np.setdiff1d(
            sessions, sess_id_test
        )
        sess_id_train = list(sess_id_train)
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

        x_train = x[idx_train, ...]
        y_train = y[idx_train]
        x_valid = x[idx_valid, ...]
        y_valid = y[idx_valid]
        x_test = x[idx_test, ...]
        y_test = y[idx_test]

        # standardization
        if apply_standardization:
            # computing statistics (across trials and time steps) for standardization on the training set
            m = np.mean(x[idx_train, ...], axis=(0, -1))
            s = np.std(x[idx_train, ...], axis=(0, -1))

            x_train = standardize(x_train, m, s)
            x_valid = standardize(x_valid, m, s)
            x_test = standardize(x_test, m, s)
        # cropping
        if interval != original_interval:
            x_train = crop_signals(x=x_train, srate=srate, interval_in=original_interval, interval_out=interval)
            x_valid = crop_signals(x=x_valid, srate=srate, interval_in=original_interval, interval_out=interval)
            x_test = crop_signals(x=x_test, srate=srate, interval_in=original_interval, interval_out=interval)

        # swap axes: from (N_examples, C, T) to (N_examples, T, C)
        x_train = np.swapaxes(x_train, -1, -2)
        x_valid = np.swapaxes(x_valid, -1, -2)
        x_test = np.swapaxes(x_test, -1, -2)
        # dataloaders
        train_loader, valid_loader, test_loader = get_dataloader(batch_size,
                                                                 (x_train, y_train),
                                                                 (x_valid, y_valid),
                                                                 (x_test, y_test))
        datasets = {}
        datasets["train"] = train_loader
        datasets["valid"] = valid_loader
        datasets["test"] = test_loader
        tail_path = os.path.join(
            self.iterator_tag,
            'sub-{0}'.format(str(dataset.subject_list[target_subject_idx]).zfill(3)),
            sessions[target_session_idx],
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
    seed: int
        Seed for random number generators.
    data_folder : str
        Folder where the dataset is stored.
    """

    def __init__(self, seed, data_folder):
        self.iterator_tag = "leave-one-subject-out"
        self.data_folder = data_folder
        np.random.seed(seed)

    def prepare(
            self,
            dataset,
            batch_size,
            cached_dataset_tag,
            interval=None,
            valid_ratio=0.2,
            target_subject_idx=None,
            target_session_idx=None,
            apply_standardization=False,
    ):
        """This function returns the pre-processed datasets (training, validation and test sets)
        Arguments
        ---------
        dataset : moabb.datasets.??
            Target MOABB dataset to use.
        batch_size: int
            Batch size.
        cached_dataset_tag: str
            Tag of the prepared data.
        interval: list
            Interval to crop EEG signals (s).
        valid_ratio: float
            Ratio of the training set to use as validation data.
        target_subject_idx: int
            Index of the subject signals to load.
        target_session_idx: int
            Index of the session signals to load. None if leave-one-subject-out.
        apply_standardization: bool
            Apply standardization after data loading.
        """

        dataset_code = dataset.code
        tail = os.path.join(self.data_folder,
                            'MOABB_pickled',
                            dataset_code,
                            cached_dataset_tag)
        test_fpath = os.path.join(tail,
                                  'sub-{0}.pkl'.format(str(dataset.subject_list[target_subject_idx]).zfill(3)))

        with open(test_fpath, 'rb') as f:
            data_dict = pickle.load(f)

        x_test = data_dict['x']
        y_test = data_dict['y']
        original_interval = data_dict['interval']
        srate = data_dict['srate']

        train_fpaths = [os.path.join(tail, 'sub-{0}.pkl'.format(str(dataset.subject_list[i]).zfill(3))) \
                        for i in np.arange(len(dataset.subject_list)) if i != target_subject_idx]

        x_train, y_train, x_valid, y_valid = [], [], [], []
        for train_fpath in train_fpaths:
            # loading training set
            with open(train_fpath, 'rb') as f:
                data_dict = pickle.load(f)
            tmp_x_train = data_dict['x']
            tmp_y_train = data_dict['y']
            tmp_metadata = data_dict['metadata']

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

        # standardization
        if apply_standardization:
            # computing statistics (across trials and time steps) for standardization on the training set
            m = np.mean(x_train, axis=(0, -1))
            s = np.std(x_train, axis=(0, -1))

            x_train = standardize(x_train, m, s)
            x_valid = standardize(x_valid, m, s)
            x_test = standardize(x_test, m, s)

        # cropping
        if interval != original_interval:
            x_train = crop_signals(x=x_train, srate=srate, interval_in=original_interval, interval_out=interval)
            x_valid = crop_signals(x=x_valid, srate=srate, interval_in=original_interval, interval_out=interval)
            x_test = crop_signals(x=x_test, srate=srate, interval_in=original_interval, interval_out=interval)

        # swap axes: from (N_examples, C, T) to (N_examples, T, C)
        x_train = np.swapaxes(x_train, -1, -2)
        x_valid = np.swapaxes(x_valid, -1, -2)
        x_test = np.swapaxes(x_test, -1, -2)
        print(x_train.shape)
        # dataloaders
        train_loader, valid_loader, test_loader = get_dataloader(batch_size,
                                                                 (x_train, y_train),
                                                                 (x_valid, y_valid),
                                                                 (x_test, y_test))
        datasets = {}
        datasets["train"] = train_loader
        datasets["valid"] = valid_loader
        datasets["test"] = test_loader
        tail_path = os.path.join(self.iterator_tag, 'sub-{0}'.format(str(dataset.subject_list[target_subject_idx]).zfill(3)) )
        yield (tail_path, datasets)
