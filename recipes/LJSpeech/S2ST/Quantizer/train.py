"""
Script to train K-means clustering model

Authors
 * Duret Jarod 2023
"""

# Adapted from https://github.com/facebookresearch/fairseq
# MIT License
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
import logging
import time

import joblib
import torch
from sklearn.cluster import MiniBatchKMeans
from hyperpyyaml import load_hyperpyyaml
import speechbrain as sb
from speechbrain.lobes.models.huggingface_wav2vec import HuggingFaceWav2Vec2

from utils import extract_features, get_splits


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


def fetch_kmeans_model(
    n_clusters,
    init,
    max_iter,
    batch_size,
    tol,
    max_no_improvement,
    n_init,
    reassignment_ratio,
    random_state,
):
    return MiniBatchKMeans(
        n_clusters=n_clusters,
        init=init,
        max_iter=max_iter,
        batch_size=batch_size,
        tol=tol,
        max_no_improvement=max_no_improvement,
        n_init=n_init,
        reassignment_ratio=reassignment_ratio,
        random_state=random_state,
        verbose=1,
        compute_labels=True,
        init_size=None,
    )


def train_kmeans(kmeans_model, features_batch):
    start_time = time.time()
    kmeans_model.fit(features_batch)
    time_taken = round((time.time() - start_time) // 60, 2)
    return kmeans_model, time_taken


if __name__ == "__main__":
    logger = setup_logger()

    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    sys.path.append("../../")
    from ljspeech_prepare import prepare_ljspeech

    sb.utils.distributed.run_on_main(
        prepare_ljspeech,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["save_folder"],
            "splits": hparams["splits"],
            "seed": hparams["seed"],
            "skip_prep": hparams["skip_prep"],
        },
    )

    # Fetch device
    device = get_device(not hparams["no_cuda"])

    logger.info(f"Loading encoder model from HF hub: {hparams['encoder_hub']}")    
    save_path = f"{hparams['output_folder']}/pretrained_models"
    encoder = HuggingFaceWav2Vec2(hparams["encoder_hub"], save_path, output_all_hiddens=True, output_norm=False, freeze_feature_extractor=True, freeze=True).to(device)
    
    splits = get_splits(hparams["save_folder"], hparams["splits"])

    # Features loading/extraction for K-means
    logger.info(f"Extracting acoustic features ...")

    (features_batch, idx) = extract_features(
        model=encoder,
        layer=hparams["layer"],
        splits=splits,
        sample_pct=hparams["sample_pct"],
        flatten=True,
        device=device,
    )

    logger.info(f"Features shape = {features_batch.shape}\n")

    # Train and save Kmeans model
    kmeans_model = fetch_kmeans_model(
        n_clusters=hparams["num_clusters"],
        init=hparams["init"],
        max_iter=hparams["max_iter"],
        batch_size=hparams["batch_size"],
        tol=hparams["tol"],
        max_no_improvement=hparams["max_no_improvement"],
        n_init=hparams["n_init"],
        reassignment_ratio=hparams["reassignment_ratio"],
        random_state=hparams["seed"],
    )

    logger.info("Starting k-means training...")
    kmeans_model, time_taken = train_kmeans(
        kmeans_model=kmeans_model, features_batch=features_batch
    )
    logger.info(f"k-means model trained in {time_taken} minutes")
    inertia = -kmeans_model.score(features_batch) / len(features_batch)
    logger.info(f"Total intertia: {round(inertia, 2)}\n")

    logger.info(f"Saving k-means model to {hparams['out_kmeans_model_path']}")
    joblib.dump(kmeans_model, open(hparams["out_kmeans_model_path"], "wb"))
