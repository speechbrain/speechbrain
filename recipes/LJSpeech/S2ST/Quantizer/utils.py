import json
import itertools
import random
import pathlib as pl

import tqdm
import kaldiio
import numpy as np
import pandas as pd
import speechbrain as sb


def get_splits(data_folder, ds_list, splits):
    ds_splits = []
    data_folder = pl.Path(data_folder)
    for ds in ds_list:
        for split in splits:
            ds_splits.append(data_folder / f"{ds}/{split}.json")
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

def extract_features(feats_folder, splits, sample_pct, flatten, device="cpu"):
    data, num_files = fetch_data(splits, sample_pct)
    features_list = []
    id_list = []

    feats_scp =f"{feats_folder}/feats.scp"
    feats_df = pd.read_csv(feats_scp, delimiter=' ', header=None)
    feats_dict = {k:v for k, v in zip(feats_df[0], feats_df[1])}

    for item in tqdm.tqdm(data, total=num_files):
        if not item['id'] in feats_dict:
            print(f"Missing: {item['id']}")
            continue
        feats = kaldiio.load_mat(feats_dict[item['id']])
        features_list.append(feats)
        id_list.append(item['id'])

    if flatten:
        return np.concatenate(features_list), id_list

    return features_list, id_list
