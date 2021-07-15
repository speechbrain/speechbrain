"""
Data preparation and pre-processing.
Python-based implementation of the pre-processing adopted by
E. S. Kappenman et al., Neuroimage 2021 (https://doi.org/10.1016/j.neuroimage.2020.117465).

Author
------
Davide Borra, 2021
"""

import os
import numpy as np
import logging
import glob
import mne


logger = logging.getLogger(__name__)


def load_and_preprocess_p3_erp_core(hparams):
    """This function performs the data loading and pre-processing of
    each subject-specific EEG dataset."""
    # definition of the event IDs
    event_id = {
        "11.0": 1,
        "12.0": 2,
        "13.0": 3,
        "14.0": 4,
        "15.0": 5,
        "21.0": 6,
        "22.0": 7,
        "23.0": 8,
        "24.0": 9,
        "25.0": 10,
        "31.0": 11,
        "32.0": 12,
        "33.0": 13,
        "34.0": 14,
        "35.0": 15,
        "41.0": 16,
        "42.0": 17,
        "43.0": 18,
        "44.0": 19,
        "45.0": 20,
        "51.0": 21,
        "52.0": 22,
        "53.0": 23,
        "54.0": 24,
        "55.0": 25,
    }
    data_fpath = glob.glob(
        os.path.join(hparams["data_folder"], hparams["sbj_id"], "eeg", "*.set")
    )[0]
    # loading EEGLab dataset
    raw_eeglab = mne.io.read_raw_eeglab(
        data_fpath, verbose="INFO", preload=True
    )
    # checking if pick specific set of EEG channels
    if hparams["ch_names"] != []:
        raw_eeglab.pick_channels(hparams["ch_names"])
    # filtering between [2,20] Hz
    raw = raw_eeglab.copy().filter(l_freq=2, h_freq=20)
    # downsampling to reduce time samples to process
    raw.resample(hparams["sf"])
    # re-referencing to the average of P9 and P10
    raw.set_eeg_reference(ref_channels=["P9", "P10"])
    # obtaining events and event_dict from annotations, essential to define EEG epochs
    events_from_annot, event_dict = mne.events_from_annotations(
        raw, event_id=event_id, verbose="INFO"
    )
    baseline_tmin = -0.2
    baseline_tmax = 0.0
    tmin = hparams["tmin"]
    tmax = hparams["tmax"]
    # epoching EEG signals between tmin and tmax (included) and applying baseline
    # correction from -0.2 to 0. respect to stimulus onset
    evoked = mne.Epochs(
        raw,
        events=events_from_annot,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=(baseline_tmin, baseline_tmax),
        preload=True,
    )
    # shift in time to account for the LCD monitor delay, as performed in the
    # reference paper
    evoked = evoked.shift_time(0.026, relative=True)

    # getting EEG epochs
    x = evoked.get_data()

    # event IDs associated to deviant stimuli
    deviant_ids = [
        evoked.event_id[key] for key in ["11.0", "22.0", "33.0", "44.0", "55.0"]
    ]

    idx_deviant = []
    for id in deviant_ids:
        tmp_idx = np.where(evoked.events[..., -1] == id)[0]
        idx_deviant.extend(tmp_idx)
    idx_deviant = np.array(idx_deviant)
    # getting labels (0: non P300 class, associated to standard stimuli;
    # 1: P300 class, associated to deviant stimuli)
    y = np.zeros(x.shape[0]).astype(np.int)
    y[idx_deviant] = 1
    # 32*5=160 (standard) vs. 8*5=40 (deviant) --> 200 trials for each subject
    return x[..., :-1], y
