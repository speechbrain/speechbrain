import json
import itertools
import random
import pathlib as pl

import tqdm
import torch
import numpy as np
import pandas as pd
import speechbrain as sb


def np_array(tensor):
    tensor = tensor.squeeze(0)
    tensor = tensor.detach().cpu()
    return np.array(tensor)

def get_splits(data_folder, splits):
    ds_splits = []
    data_folder = pl.Path(data_folder)
    for split in splits:
        ds_splits.append(data_folder / f"{split}.json")
    return ds_splits

def fetch_data(splits, sample_pct, seed=1234):
    ds_splits = {}
    for split in splits:
        key = f"{split.parent}_{split.stem}"
        ds_splits[key] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=split,
            output_keys=["id", "wav"],
        )
        
    data = list(itertools.chain(*ds_splits.values()))
    random.seed(seed)
    if sample_pct < 1.0:
        data = random.sample(
            data, int(sample_pct * len(data))
        )
    return iter(data), len(data)

def extract_features(model, layer, splits, sample_pct, flatten, device='cpu'):
    data, num_files = fetch_data(splits, sample_pct)
    features_list = []
    id_list = []

    for item in tqdm.tqdm(data, total=num_files):
        wav = item['wav']
        with torch.no_grad():
            audio = sb.dataio.dataio.read_audio(wav)
            audio = audio.unsqueeze(0).to(device)
            feats = model.extract_features(audio)
            feats = feats[layer]
            feats = np_array(feats)
        features_list.append(feats)
        id_list.append(item['id'])

    if flatten:
        return np.concatenate(features_list), id_list

    return features_list, id_list
