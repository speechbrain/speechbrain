#!/usr/bin/python3
"""
Prepare MOABB datasets.

Author
------
Davide Borra, 2022
"""
import mne
import numpy as np
from moabb.datasets import (
    BNCI2014001,
    BNCI2014004,
    BNCI2015001,
    BNCI2015004,
    Lee2019_MI,
    Shin2017A,
    Zhou2016,
)
from moabb.datasets import BNCI2014009, EPFLP300, Lee2019_ERP, bi2015a
from moabb.datasets import Lee2019_SSVEP
from moabb.paradigms import MotorImagery, P300, SSVEP
from mne.utils.config import set_config, get_config
import os
import pickle
import argparse
from mne.channels import find_ch_adjacency
import scipy
import warnings


# Set mne verbosity
mne.set_log_level(verbose="error")


def get_output_dict(
    dataset, subject, events_to_load, srate_in, srate_out, fmin, fmax, verbose=0
):
    """This function returns the dictionary with subject-specific data."""
    output_dict = {}
    output_dict["code"] = dataset.code
    output_dict["subject_list"] = dataset.subject_list
    output_dict["paradigm"] = dataset.paradigm
    output_dict["n_sessions"] = dataset.n_sessions
    output_dict["fmin"] = fmin
    output_dict["fmax"] = fmax
    output_dict["ival"] = dataset.interval
    output_dict["interval"] = list(
        np.array(dataset.interval) - np.min(dataset.interval)
    )
    output_dict["reference"] = dataset.doi

    event_id = dataset.event_id
    output_dict["event_id"] = event_id
    event_keys = list(event_id.keys())
    idx_sorting = []
    for e in event_keys:
        idx_sorting.append(int(event_id[e]) - 1)

    events = [event_keys[i] for i in np.argsort(idx_sorting)]
    if events_to_load is not None:
        events = [e for e in events if e in events_to_load]
    output_dict["events"] = events

    output_dict["original_srate"] = srate_in
    output_dict["srate"] = srate_out if srate_out is not None else srate_in

    paradigm = None
    if dataset.paradigm == "imagery":
        paradigm = MotorImagery(
            events=events_to_load,  # selecting all events or specific events
            n_classes=len(output_dict["events"]),  # setting number of classes
            fmin=fmin,  # band-pass filtering
            fmax=fmax,
            channels=None,  # all channels
            resample=srate_out,  # downsample
        )
    elif dataset.paradigm == "p300":
        paradigm = P300(
            fmin=fmin,  # band-pass filtering
            fmax=fmax,
            channels=None,  # all channels
            resample=srate_out,  # downsample
        )
    elif dataset.paradigm == "ssvep":
        paradigm = SSVEP(
            events=events_to_load,  # selecting all events or specific events
            n_classes=len(output_dict["events"]),  # setting number of classes
            fmin=fmin,  # band-pass filtering
            fmax=fmax,
            channels=None,  # all channels
            resample=srate_out,  # downsample
        )

    x, y, labels, metadata, channels, adjacency_mtx, srate = load_data(
        paradigm, dataset, [subject]
    )

    if verbose == 1:
        for l in np.unique(labels):
            print(
                print(
                    "Number of label {0} examples: {1}".format(
                        l, np.where(labels == l)[0].shape[0]
                    )
                )
            )

    if dataset.paradigm == "p300":
        if output_dict["events"] == ["Target", "NonTarget"]:
            y = 1 - y  # swap NonTarget to Target
            output_dict["events"] = ["NonTarget", "Target"]
    if verbose == 1:
        for c in np.unique(y):
            print(
                "Number of class {0} examples: {1}".format(
                    c, np.where(y == c)[0].shape[0]
                )
            )

    output_dict["channels"] = channels
    output_dict["adjacency_mtx"] = adjacency_mtx
    output_dict["x"] = x
    output_dict["y"] = y
    output_dict["labels"] = labels
    output_dict["metadata"] = metadata
    output_dict["subject"] = subject

    if verbose == 1:
        print(output_dict)
    return output_dict


def load_data(paradigm, dataset, idx):
    """This function returns EEG signals and the corresponding labels using MOABB methods
    In addition metadata, channel names and the sampling rate are provided too."""
    x, labels, metadata = paradigm.get_data(dataset, idx, True)
    ch_names = x.info.ch_names
    adjacency, _ = find_ch_adjacency(x.info, ch_type="eeg")
    adjacency_mtx = scipy.sparse.csr_matrix.toarray(
        adjacency
    )  # from sparse mtx to ndarray

    srate = x.info["sfreq"]
    x = x.get_data()
    y = [dataset.event_id[yy] for yy in labels]
    y = np.array(y)
    y -= y.min()
    return x, y, labels, metadata, ch_names, adjacency_mtx, srate


def download_data(data_folder, dataset):
    """This function download a specific MOABB dataset in a directory."""
    # changing default download directory
    for a in get_config().keys():
        set_config(a, data_folder)
    dataset.download()


def prepare_data(
    data_folder,
    dataset,
    events_to_load,
    srate_in,
    srate_out,
    fmin,
    fmax,
    cached_data_folder=None,
    idx_subject_to_prepare=-1,
    save_prepared_dataset=True,
    verbose=0,
):
    """This function prepare all datasets and save them in a separate pickle for each subject."""

    # Crete the data folder (if needed)
    if not os.path.exists(data_folder):
        print(data_folder)
        os.makedirs(data_folder)

    # changing default download directory
    for a in get_config().keys():
        set_config(a, data_folder)
    if cached_data_folder is None:
        cached_data_folder = data_folder
    tmp_output_dir = os.path.join(
        os.path.join(
            cached_data_folder,
            "MOABB_pickled",
            dataset.code,
            "{0}_{1}-{2}".format(
                str(
                    int(srate_out if srate_out is not None else srate_in)
                ).zfill(4),
                str(int(fmin)).zfill(3),
                str(int(fmax)).zfill(3),
            ),
        )
    )
    if not os.path.isdir(tmp_output_dir):
        os.makedirs(tmp_output_dir)

    if idx_subject_to_prepare < 0:
        subject_to_prepare = dataset.subject_list
    else:
        subject_to_prepare = [dataset.subject_list[idx_subject_to_prepare]]

    for kk, subject in enumerate(subject_to_prepare):
        fname = "sub-{0}.pkl".format(str(subject).zfill(3))
        output_dict_fpath = os.path.join(tmp_output_dir, fname)

        # Prepare dataset only if not already prepared
        output_dict = {}
        if os.path.isfile(output_dict_fpath):
            print("Using cached dataset at: {0}".format(output_dict_fpath))
            with open(output_dict_fpath, "rb") as handle:
                output_dict = pickle.load(handle)
        else:
            output_dict = get_output_dict(
                dataset,
                subject,
                events_to_load,
                srate_in,
                srate_out,
                fmin=fmin,
                fmax=fmax,
                verbose=verbose,
            )

        if save_prepared_dataset:
            if os.path.isfile(output_dict_fpath):
                print(
                    "Skipping data saving, a cached dataset was found at {0}".format(
                        output_dict_fpath
                    )
                )
            else:
                print("Saving the dataset at {0}".format(output_dict_fpath))
                with open(output_dict_fpath, "wb") as handle:
                    pickle.dump(
                        output_dict, handle, protocol=pickle.HIGHEST_PROTOCOL
                    )
        if (
            idx_subject_to_prepare > -1
        ):  # iterating over only 1 subject, return its dictionary
            return output_dict


if __name__ == "__main__":
    """Downloading and preparing multi-session MOABB datasets."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="",
        help="Dataset name to download and prepare (empty string for selecting all supported datasets)",
    )
    parser.add_argument(
        "--data_folder",
        type=str,
        default="/path/to/MOABB_datasets",
        help="Folder where dataset will be downloaded",
    )
    parser.add_argument(
        "--cached_data_folder",
        type=str,
        default="/path/to/pickled/MOABB_datasets",
        help="Folder where dataset will be prepared and saved as pkl",
    )
    parser.add_argument(
        "--to_download", type=int, default=0, help="Download flag"
    )
    parser.add_argument(
        "--to_prepare", type=int, default=0, help="Prepare flag"
    )
    parser.add_argument(
        "--fmin",
        type=float,
        default=1.0,
        help="Lower cut-off frequency for band-pass filtering",
    )
    parser.add_argument(
        "--fmax",
        type=float,
        default=40.0,
        help="Higher cut-off frequency for band-pass filtering",
    )
    FLAGS, unparsed = parser.parse_known_args()

    # SETTING UP MOTOR IMAGERY DATASETS
    mi_ds_names = [
        "BNCI2014001",
        "BNCI2014004",
        "BNCI2015001",
        "BNCI2015004",
        "Lee2019_MI",
        "Shin2017A",
        "Zhou2016",
    ]
    mi_datasets = [
        BNCI2014001(),
        BNCI2014004(),
        BNCI2015001(),
        BNCI2015004(),
        Lee2019_MI(),
        Shin2017A(accept=True),
        Zhou2016(),
    ]
    mi_srate_in_list = [250, 250, 512, 256, 1000, 1000, 250]
    mi_srate_out_list = [125, 125, 128, 128, 125, 125, 125]
    mi_events_to_load = [
        None,
        None,
        None,
        ["right_hand", "feet"],
        None,
        None,
        None,
    ]
    # SETTING UP P300 DATASETS
    p300_ds_names = ["BNCI2014009", "EPFLP300", "Lee2019_ERP", "bi2015a"]
    p300_datasets = [
        BNCI2014009(),
        EPFLP300(),
        Lee2019_ERP(),
        bi2015a(),
    ]
    p300_srate_in_list = [256, 512, 1000, 512]
    p300_srate_out_list = [128, 128, 125, 128]
    p300_events_to_load = [
        None,
        None,
        None,
        None,
    ]
    # SETTING UP SSVEP DATASETS
    ssvep_ds_names = ["Lee2019_SSVEP"]
    ssvep_datasets = [Lee2019_SSVEP()]
    ssvep_srate_in_list = [1000]
    ssvep_srate_out_list = [125]
    ssvep_events_to_load = [None]

    ds_names = mi_ds_names + p300_ds_names + ssvep_ds_names
    datasets = mi_datasets + p300_datasets + ssvep_datasets
    srate_in_list = mi_srate_in_list + p300_srate_in_list + ssvep_srate_in_list
    srate_out_list = (
        mi_srate_out_list + p300_srate_out_list + ssvep_srate_out_list
    )
    # no resampling if srate_out=srate_in
    for k in np.where(np.array(srate_out_list) == np.array(srate_in_list))[0]:
        srate_out_list[k] = None
    events_to_load_list = (
        mi_events_to_load + p300_events_to_load + ssvep_events_to_load
    )
    if FLAGS.dataset_name == "":
        print("The selected datasets were: ")
        for dataset in datasets:
            print("-{0}".format(dataset.code))
    else:
        idx_dataset_name = [
            i for i, s in enumerate(ds_names) if s == FLAGS.dataset_name
        ]
        assert len(idx_dataset_name) == 1, "Wrong dataset name"
        idx_dataset_name = idx_dataset_name[-1]
        print("The selected datasets was: ")
        print("-{0}".format(datasets[idx_dataset_name].code))
        datasets = datasets[idx_dataset_name : idx_dataset_name + 1]
        srate_in_list = srate_in_list[idx_dataset_name : idx_dataset_name + 1]
        srate_out_list = srate_out_list[idx_dataset_name : idx_dataset_name + 1]
        events_to_load_list = events_to_load_list[
            idx_dataset_name : idx_dataset_name + 1
        ]

    if FLAGS.to_download:
        print("Start downloading datasets, it might take a while...")
        for dataset in datasets:
            download_data(FLAGS.data_folder, dataset)

    if FLAGS.to_prepare:
        print("Start preparing the datasets...")
        for dataset, events_to_load, srate_in, srate_out in zip(
            datasets, events_to_load_list, srate_in_list, srate_out_list
        ):
            prepare_data(
                FLAGS.data_folder,
                dataset,
                events_to_load,
                srate_in,
                srate_out,
                fmin=FLAGS.fmin,
                fmax=FLAGS.fmax,
                cached_data_folder=FLAGS.cached_data_folder,
                verbose=0,
            )
            print("Ended: {0}".format(dataset.code))
