"""
Data reading and writing

Authors
 * Mirco Ravanelli 2020
 * Aku Rouhe 2020
 * Ju-Chieh Chou 2020
 * Samuele Cornell 2020
"""

import torch
import torchaudio
import pickle


def read_wav(waveforms_obj):
    files = waveforms_obj["files"]
    if not isinstance(files, list):
        files = [files]

    waveforms = []
    for f in files:
        if (
            "start" not in waveforms_obj.keys()
            or "stop" not in waveforms_obj.keys()
        ):
            tmp, fs = torchaudio.load(f)
            waveforms.append(tmp)
        else:
            num_frames = waveforms_obj["stop"] - waveforms_obj["start"]
            offset = waveforms_obj["start"]
            tmp, fs = torchaudio.load(f, num_frames=num_frames, offset=offset)
            waveforms.append(tmp)

    return torch.cat(waveforms, 0)


def load_pickle(pickle_path):
    with open(pickle_path, "r") as f:
        out = pickle.load(f)
    return out
