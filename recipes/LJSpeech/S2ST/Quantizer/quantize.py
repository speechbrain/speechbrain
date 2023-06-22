"""
Script to quantize using Kmeans clustering over acoustic features.

Authors
 * Duret Jarod 2021
"""

# Adapted from https://github.com/facebookresearch/fairseq
# MIT License
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import json
import pathlib as pl

import kaldiio
import joblib
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

def setup_logger():
    log_format = "[%(asctime)s] [%(levelname)s]: %(message)s"
    logging.basicConfig(format=log_format, level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger

def get_device(use_cuda):
    use_cuda = use_cuda and torch.cuda.is_available()
    print('\n' + '=' * 30)
    print('USE_CUDA SET TO: {}'.format(use_cuda))
    print('CUDA AVAILABLE?: {}'.format(torch.cuda.is_available()))
    print('=' * 30 + '\n')
    return torch.device("cuda" if use_cuda else "cpu")

def fetch_config():
    parser = argparse.ArgumentParser(
        description="Extract discrete speech units"
    )

    parser.add_argument('--feats_dir', '-i', help="features directory", required=True)
    parser.add_argument('--save_dir', '-o', help="save directory", required=True)
    parser.add_argument('--model', '-m', help="Kmeans model", required=True)
    parser.add_argument('--start_idx', '-s', help="starting index", default=0, type=int)
    parser.add_argument('--end_idx', '-e', help="ending index", default=-1, type=int)

    # Leftovers
    parser.add_argument("--no_cuda", default=False, type=bool)
    return parser


if __name__ == "__main__":
    parser = fetch_config()
    args = parser.parse_args()
    logger = setup_logger()
    logger.info(args)
    
    feats_dir = pl.Path(args.feats_dir)
    save_dir = pl.Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    kmeans = pl.Path(args.model)

    # Fetch device
    device = get_device(not args.no_cuda)

    # Features loading/extraction for K-means
    logger.info(f"Extracting acoustic features from {feats_dir.as_posix()} ...")

    # K-means model
    logger.info(f"Loading K-means model from {kmeans} ...")
    kmeans_model = joblib.load(open(kmeans, "rb"))
    kmeans_model.verbose = False
    
    feats_scp = feats_dir / f"feats.scp"
    feats_df = pd.read_csv(feats_scp, delimiter=' ', header=None)
    feats_dict = {k:v for k, v in zip(feats_df[0], feats_df[1])}
    feats_keys = list(feats_dict.keys())

    unique_count = 0
    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx != -1 else len(feats_dict)
    base = f"feats_{start_idx}:{end_idx}"
    wspecifier = f"ark,scp,t:{save_dir.as_posix()}/{base}.ark,{save_dir.as_posix()}/{base}.scp"
    with kaldiio.WriteHelper(wspecifier, write_function="numpy") as writer:
        for i in tqdm(range(start_idx, end_idx)):
            key = feats_keys[i]
            feats = kaldiio.load_mat(feats_dict[key])
            pred = kmeans_model.predict(feats)
            writer(key, pred)
            unique_count += len(set(pred.tolist()))
    print(
        f"average of unique unit per utterance = {unique_count / (end_idx - start_idx)}"
    )