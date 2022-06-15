#!/usr/bin/python3
"""
Prepare MOABB multi-session datasets.

Author
------
Davide Borra, 2022
"""

import numpy as np
from moabb.datasets import BNCI2014001, BNCI2014004, BNCI2015001, BNCI2015004, Lee2019_MI, Shin2017A, Zhou2016
from moabb.datasets import BNCI2014009, EPFLP300, Lee2019_ERP, bi2015a
from moabb.datasets import Lee2019_SSVEP
from moabb.paradigms import MotorImagery, P300, SSVEP
from mne.utils.config import set_config, get_config
import os
import pickle


def get_output_dict(dataset,k,subject,events_list, srate_in_list, srate_out_list, fmin, fmax):
    output_dict = {}
    output_dict['code'] = dataset.code
    output_dict['paradigm'] = dataset.paradigm
    output_dict['n_sessions'] = dataset.n_sessions
    output_dict['fmin'] = fmin
    output_dict['fmax'] = fmax
    output_dict['ival'] = dataset.interval
    output_dict['interval'] = list(np.array(dataset.interval) - np.min(dataset.interval))
    output_dict['reference'] = dataset.doi
    output_dict['event_id'] = dataset.event_id

    event_keys = list(dataset.event_id.keys())
    idx_sorting = []
    for e in event_keys:
        idx_sorting.append(int(dataset.event_id[e]) - 1)
    events = list(np.array(event_keys)[idx_sorting])

    output_dict['events'] = events
    output_dict['original_srate'] = srate_in_list[k]
    output_dict['srate'] = srate_out_list[k] if srate_out_list[k] is not None else srate_in_list[k]
    if dataset.paradigm=='imagery':
        paradigm = MotorImagery(
            events=events_list[k],  # selecting all events or specific events
            n_classes=len(output_dict['events']),  # setting number of classes
            fmin=fmin,  # band-pass filtering
            fmax=fmax,
            channels=None,  # all channels
            resample=srate_out_list[k]  # downsample
        )
    elif dataset.paradigm=='p300':
        paradigm = P300(
            fmin=fmin,  # band-pass filtering
            fmax=fmax,
            channels=None,  # all channels
            resample=srate_out_list[k]  # downsample
        )
    elif dataset.paradigm=='ssvep':
        paradigm = SSVEP(
            events=events_list[k],  # selecting all events or specific events
            n_classes=len(output_dict['events']),  # setting number of classes
            fmin=fmin,  # band-pass filtering
            fmax=fmax,
            channels=None,  # all channels
            resample=srate_out_list[k]  # downsample
        )

    x, y, labels, metadata, channels, srate = load_data(paradigm, dataset, [subject])
    for l in np.unique(labels):
        print(print("Number of label {0} examples: {1}".format(l, np.where(labels==l)[0].shape[0])))
    if dataset.paradigm=='p300':
        if output_dict['events'] == ['Target', 'NonTarget']:
            y = 1 - y # swap NonTarget to Target
            output_dict['events'] = [ 'NonTarget', 'Target']
    for c in np.unique(y):
        print("Number of class {0} examples: {1}".format(c, np.where(y==c)[0].shape[0]))
    output_dict['channels'] = channels

    output_dict['x'] = x
    output_dict['y'] = y
    output_dict['labels'] = labels
    output_dict['metadata'] = metadata
    output_dict['subject'] = subject

    print(output_dict)
    return output_dict


def load_data(paradigm, dataset, idx):
    """This function returns EEG signals and the corresponding labels. In addition metadata are provided too."""
    x, labels, metadata = paradigm.get_data(dataset, idx, True)
    ch_names = x.info.ch_names
    srate = x.info['sfreq']
    x = x.get_data()
    y = [dataset.event_id[yy] for yy in labels]
    y = np.array(y)
    y -= y.min()
    return x, y, labels, metadata, ch_names, srate


data_folder = '/path/to/MOABB_datasets'
to_download = False
for a in get_config().keys():
    set_config(a, data_folder)

mi_datasets = [BNCI2014001(),
               BNCI2014004(),
               BNCI2015001(),
               BNCI2015004(),
               Lee2019_MI(),
               Shin2017A(accept=True),
               Zhou2016(),
               ]
p300_datasets = [BNCI2014009(),
                 EPFLP300(),
                 Lee2019_ERP(),
                 bi2015a(), ]
ssvep_datasets = [Lee2019_SSVEP()]

if to_download:
    for k, dataset in enumerate(mi_datasets+p300_datasets+ssvep_datasets):
        dataset.download()

# PREPARING SSVEP DATASET
events_list = [None]
srate_in_list = [1000]
srate_out_list0 = [125]
srate_out_list1 = [250]

srate_out_lists = [srate_out_list0, srate_out_list1]

for srate_out_list in srate_out_lists[:1]:
    for k in np.where(np.array(srate_out_list)==np.array(srate_in_list))[0]:
        srate_out_list[k] = None
    for k, dataset in enumerate(ssvep_datasets):
        for kk, subject in enumerate(dataset.subject_list):

            output_dict=get_output_dict(dataset, k, subject, events_list, srate_in_list, srate_out_list,
                                        fmin=1, fmax=40)
            tmp_output_dir = os.path.join(os.path.join(data_folder,
                                                       'MOABB_pickled',
                                                       output_dict['code'],
                                                       '{0}'.format(str(int(output_dict['srate'])).zfill(3)+'Hz')))
            if not os.path.isdir(tmp_output_dir):
                os.makedirs(tmp_output_dir)
            output_dict_fpath = os.path.join(tmp_output_dir, 'sub-{0}.pkl'.format(str(kk).zfill(3)))
            with open(output_dict_fpath, "wb") as handle:
                pickle.dump(
                    output_dict, handle, protocol=pickle.HIGHEST_PROTOCOL
                )

# PREPARING P300 DATASETS
events_list = [None,
               None,
               None,
               None,
]

srate_in_list = [256, 512, 1000, 512]
srate_out_list0 = [128, 128, 125, 128]
srate_out_list1 = [256, 256, 250, 256]

srate_out_lists = [srate_out_list0, srate_out_list1]


for srate_out_list in srate_out_lists[:1]:
    for k in np.where(np.array(srate_out_list)==np.array(srate_in_list))[0]:
        srate_out_list[k] = None

    for k, dataset in enumerate(p300_datasets):
        for kk, subject in enumerate(dataset.subject_list):

            output_dict=get_output_dict(dataset, k, subject, events_list, srate_in_list, srate_out_list,
                                        fmin=1, fmax=40)
            tmp_output_dir = os.path.join(os.path.join(data_folder,
                                                       'MOABB_pickled',
                                                       output_dict['code'],
                                                       '{0}'.format(str(int(output_dict['srate'])).zfill(3)+'Hz')))
            if not os.path.isdir(tmp_output_dir):
                os.makedirs(tmp_output_dir)
            output_dict_fpath = os.path.join(tmp_output_dir, 'sub-{0}.pkl'.format(str(kk).zfill(3)))
            with open(output_dict_fpath, "wb") as handle:
                pickle.dump(
                    output_dict, handle, protocol=pickle.HIGHEST_PROTOCOL
                )

# PREPARING MOTOR IMAGERY DATASETS
events_list = [None,
               None,
               None,
               ['right_hand', 'feet'],
               None,
               None,
               None
]

srate_in_list = [250, 250, 512, 256, 1000, 1000, 250]
srate_out_list0 = [125, 125, 128, 128, 125, 125, 125]
srate_out_list1 = [250, 250, 256, 256, 250, 250, 250]

srate_out_lists = [srate_out_list0, srate_out_list1]

for srate_out_list in srate_out_lists[:1]:
    for k in np.where(np.array(srate_out_list)==np.array(srate_in_list))[0]:
        srate_out_list[k] = None

    for k, dataset in enumerate(mi_datasets):
        for kk, subject in enumerate(dataset.subject_list):

            output_dict=get_output_dict(dataset, k, subject, events_list, srate_in_list, srate_out_list,
                                        fmin=1, fmax=40)
            tmp_output_dir = os.path.join(os.path.join(data_folder,
                                                       'MOABB_pickled',
                                                       output_dict['code'],
                                                       '{0}'.format(str(int(output_dict['srate'])).zfill(3) + 'Hz')))
            if not os.path.isdir(tmp_output_dir):
                os.makedirs(tmp_output_dir)
            output_dict_fpath = os.path.join(tmp_output_dir, 'sub-{0}.pkl'.format(str(kk).zfill(3)))
            with open(output_dict_fpath, "wb") as handle:
                pickle.dump(
                    output_dict, handle, protocol=pickle.HIGHEST_PROTOCOL
                )