import torchaudio
import warnings
import pyloudnorm
import random
import torch
import pandas as pd
import os
import numpy as np
from torchaudio.transforms import Resample
from tqdm import tqdm

MAX_AMP = 0.9
MIN_LOUDNESS = -33
MAX_LOUDNESS = -25
minlen = 80000
maxlen = 80000
resample = Resample(44100, 16000)

def normalize(signal, is_noise=False):
    """
    This function normalizes the audio signals for loudness
    """
    meter = pyloudnorm.Meter(16000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c_loudness = meter.integrated_loudness(signal)
        if is_noise:
            target_loudness = random.uniform(
                MIN_LOUDNESS - 5, MAX_LOUDNESS - 5
            )
        else:
            target_loudness = random.uniform(MIN_LOUDNESS, MAX_LOUDNESS)
        signal = pyloudnorm.normalize.loudness(
            signal, c_loudness, target_loudness
        )

        # check for clipping
        if np.max(np.abs(signal)) >= 1:
            signal = signal * MAX_AMP / np.max(np.abs(signal))

    return torch.from_numpy(signal)

def combine_waveform(sources, n_overlap=0.75, prefix="/data1/ESC-50/audio"):
    if len(sources) == 1:
        wav, fs_read = torchaudio.load(os.path.join(prefix, sources[0]))
        wav = resample(wav)

        return normalize(wav.numpy()[0])[None]

    ss = []

    s1 = sources[0]
    s2 = sources[1:]
    if len(s2) == 2:
        ss.append(combine_waveform(s2))
    else:
        s2 = s2[0]
    
    for s in [s1, s2] if len(ss) == 0 else [s1]:
        s = os.path.join(prefix, s)
        tmp, fs_read = torchaudio.load(
            s,
            frame_offset=0,
        )
        tmp = resample(tmp)
        tmp = tmp[0].numpy()
        tmp = normalize(tmp)
        beginning_pad = int((1 - n_overlap) * maxlen)
        
        if random.uniform(0, 1) >= 0.5:
            tmp = torch.concat([torch.zeros(beginning_pad), tmp])[:-beginning_pad]
        else:
            tmp = torch.concat([tmp[beginning_pad:], torch.zeros(beginning_pad)])

        ss.append(tmp[None])
    
    ss = torch.stack(ss)
    mixture = torch.sum(ss, 0)
    
    return mixture

def merge_filenames(filenames):
    names = [f.split(".")[0] for f in filenames]

    return "_".join(names) + ".wav"

DATA_FOLDER = "/data1/ESC-50"
META = "esc10_ml.csv"
OUT_FOLDER = "ESC10-MultiLabel"

meta = pd.read_csv(META)
meta["filename"] = meta["filename"].apply(eval)

df = []
for row in tqdm(range(len(meta))):
    temp = {}
    prefix = os.path.join(OUT_FOLDER, "fold_" + str(meta.iloc[row].fold))
    os.makedirs(prefix, exist_ok=True)

    new_name = merge_filenames(meta.iloc[row].filename)
    out_path = os.path.join(prefix, new_name)
    mixture = combine_waveform(meta.iloc[row].filename, n_overlap=random.uniform(0.6, 0.8), prefix=os.path.join(DATA_FOLDER, "audio"))

    temp["filename"] = new_name
    temp["fold"] = str(meta.iloc[row].fold)
    temp["target"] = meta.iloc[row].target

    df.append(temp)
    torchaudio.save(out_path, mixture, sample_rate=16000)
        
    #    print('skipping data item')

df = pd.DataFrame(df)
df.to_csv(os.path.join(OUT_FOLDER, "meta.csv"))
