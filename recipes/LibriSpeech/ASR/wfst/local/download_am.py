#!/usr/bin/env python3
# Copyright    2021  Xiaomi Corp.       (authors: Mingshuang Luo)

"""
This file downloads the following trained LibriSpeech ASR files
based on speechbrain:

    - asr.ckpt
    - hyperparams.yaml
    - lm.ckpt
    - normalizer.ckpt
    - tokenizer.ckpt

Here, we mainly use the outputs of the encoder.

Usage:
    ./local/download_am.py --out-dir ./download/am

"""
import logging
import argparse

from speechbrain.pretrained.fetching import fetch
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, help="Output directory.")

    args = parser.parse_args()
    return args

def main(out_dir: str):
    hparams_file = 'hyperparams.yaml'
    source = 'speechbrain/asr-transformer-transformerlm-librispeech'
    savedir = out_dir
    use_auth_token = False
    overrides = {}

    hparams_local_path = fetch(hparams_file, source, savedir, use_auth_token)

    with open(hparams_local_path) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    pretrainer = ''
    pretrainer = hparams['pretrainer']

    pretrainer.set_collect_in(savedir)

    run_on_main(pretrainer.collect_files, kwargs={"default_source": source})

    pretrainer.load_collected(device='cpu')
    
    logging.info(f"Download AM files successful!")


if __name__ == "__main__":
    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"        
    )

    logging.basicConfig(format=formatter, level=logging.INFO)

    args = get_args()
    logging.info(f"out_dir: {args.out_dir}")

    main(out_dir=args.out_dir)
