"""
Script to train K-means clustering model on self-supervised representations.

To run this recipe, do the following:
> python train.py hparams/kmeans.yaml --data_folder=/path/to/LJspeech

Authors
 * Jarod Duret 2023
"""


import sys
import logging
import time
import random
import itertools
import pathlib as pl

import joblib
import torch
import torchaudio
import tqdm
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from hyperpyyaml import load_hyperpyyaml
import speechbrain as sb
from ljspeech_prepare import prepare_ljspeech
from speechbrain.lobes.models.huggingface_wav2vec import HuggingFaceWav2Vec2


def setup_logger():
    """Set up a logger with a log format and logging level."""
    log_format = "[%(asctime)s] [%(levelname)s]: %(message)s"
    logging.basicConfig(format=log_format, level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger


def get_device(use_cuda):
    """Determine and return the appropriate device for computation."""
    use_cuda = use_cuda and torch.cuda.is_available()
    print("\n" + "=" * 30)
    print("USE_CUDA SET TO: {}".format(use_cuda))
    print("CUDA AVAILABLE?: {}".format(torch.cuda.is_available()))
    print("=" * 30 + "\n")
    return torch.device("cuda" if use_cuda else "cpu")


def np_array(tensor):
    """Convert a Pytorch tensor to a Numpy array."""
    tensor = tensor.squeeze(0)
    tensor = tensor.detach().cpu()
    return tensor.numpy()


def fetch_data(splits, sample_pct, seed=1234):
    """Fetch data from specified splits for k-means training."""
    ds_splits = {}
    for split in splits:
        key = f"{split.parent}_{split.stem}"
        ds_splits[key] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=split, output_keys=["id", "wav"],
        )

    data = list(itertools.chain(*ds_splits.values()))
    random.seed(seed)
    if sample_pct < 1.0:
        data = random.sample(data, int(sample_pct * len(data)))
    return iter(data), len(data)


def extract_features(
    model, layer, splits, sample_pct, flatten, device="cpu", sample_rate=16000
):
    """Extract features from audio using a pre-trained model."""
    data, num_files = fetch_data(splits, sample_pct)
    features_list = []
    id_list = []

    for item in tqdm.tqdm(data, total=num_files):
        wav = item["wav"]
        with torch.no_grad():
            info = torchaudio.info(wav)
            audio = sb.dataio.dataio.read_audio(wav)
            audio = torchaudio.transforms.Resample(
                info.sample_rate, sample_rate,
            )(audio)
            audio = audio.unsqueeze(0).to(device)
            feats = model.extract_features(audio)
            feats = feats[layer]
            feats = np_array(feats)
        features_list.append(feats)
        id_list.append(item["id"])

    if flatten:
        return np.concatenate(features_list), id_list

    return features_list, id_list


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
    """Return a k-means clustering model with specified parameters."""
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
    """Train a k-means clustering model using the provided features."""
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
    encoder = HuggingFaceWav2Vec2(
        hparams["encoder_hub"],
        hparams["encoder_folder"],
        output_all_hiddens=True,
        output_norm=False,
        freeze_feature_extractor=True,
        freeze=True,
    ).to(device)

    splits = []
    data_folder = pl.Path(hparams["save_folder"])
    for split in hparams["splits"]:
        splits.append(data_folder / f"{split}.json")

    # Features loading/extraction for K-means
    logger.info("Extracting acoustic features ...")

    (features_batch, idx) = extract_features(
        model=encoder,
        layer=hparams["layer"],
        splits=splits,
        sample_pct=hparams["sample_pct"],
        flatten=True,
        device=device,
        sample_rate=hparams["sample_rate"],
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
