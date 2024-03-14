"""
Utilities for training kmeans model.

Author
 * Pooneh Mousavi 2023
"""

import os
import logging
from tqdm.contrib import tqdm

try:
    from sklearn.cluster import MiniBatchKMeans
except ImportError:
    err_msg = "The optional dependency sklearn is needed to use this module\n"
    err_msg += "Cannot import sklearn.cluster.MiniBatchKMeans to use KMeans/\n"
    err_msg += "Please follow the instructions below\n"
    err_msg += "=============================\n"
    err_msg += "pip install -U scikit-learn\n"
    raise ImportError(err_msg)
import joblib

logger = logging.getLogger(__name__)


def accumulate_and_extract_features(
    batch, features_list, ssl_model, ssl_layer_num, device
):
    """ Extract features (output of SSL model) and acculamte them on cpu to be used for clustering.

    Arguments
    ---------
        batch: tensor
            Single batch of data.
        features_list : list
            accumulate features list.
        ssl_model
            SSL-model used to  extract features used for clustering.
        ssl_layer_num: int
            specify output of which layer of the ssl_model should be used.
        device
            CPU or  GPU.
    """
    batch = batch.to(device)
    wavs, wav_lens = batch.sig
    wavs, wav_lens = (
        wavs.to(device),
        wav_lens.to(device),
    )
    feats = ssl_model(wavs, wav_lens)[ssl_layer_num].flatten(end_dim=-2)
    features_list.extend(feats.to("cpu").detach().numpy())


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
    checkpoint_path,
):
    """Return a k-means clustering model with specified parameters.

    Arguments
    ---------
        n_clusters : MiniBatchKMeans
            The number of clusters to form as well as the number of centroids to generate.
        init : int
            Method for initialization: {'k-means++'', ''random''}
        max_iter : int
            Maximum number of iterations over the complete dataset before stopping independently of any early stopping criterion heuristics.
        batch_size : int
            Size of the mini batches.
        tol : float
            Control early stopping based on the relative center changes as measured by a smoothed, variance-normalized of the mean center squared position changes.
        max_no_improvement :int
            Control early stopping based on the consecutive number of mini batches that does not yield an improvement on the smoothed inertia.
        n_init : int
            Number of random initializations that are tried
        reassignment_ratio : float
            Control the fraction of the maximum number of counts for a center to be reassigned.
        random_state :int
            Determines random number generation for centroid initialization and random reassignment.
        compute_labels : bool
            Compute label assignment and inertia for the complete dataset once the minibatch optimization has converged in fit.
        init_size : int
            Number of samples to randomly sample for speeding up the initialization.
        checkpoint_path : str
            Path to saved model.

    Returns
    ---------
        MiniBatchKMeans
            a k-means clustering model with specified parameters.
    """
    if os.path.exists(checkpoint_path):
        logger.info(f"The checkpoint is loaded from {checkpoint_path}.")
        return joblib.load(checkpoint_path)

    logger.info(
        f"No checkpoint is found at {checkpoint_path}. New model is initialized for training."
    )
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


def train(
    model,
    train_set,
    ssl_model,
    ssl_layer_num,
    kmeans_batch_size=1000,
    device="cpu",
):
    """Train a  Kmeans model .

    Arguments
    ---------
        model : MiniBatchKMeans
            The initial kmeans model for training.
        train_set : Dataloader
            Batches of tarining data.
        ssl_model
            SSL-model used to  extract features used for clustering.
        ssl_layer_num : int
            Specify output of which layer of the ssl_model should be used.
        device
            CPU or  GPU.
        kmeans_batch_size : int
            Size of the mini batches.
    """
    logger.info("Start training kmeans model.")
    features_list = []
    with tqdm(train_set, dynamic_ncols=True,) as t:
        for batch in t:
            # train a kmeans model on a single batch if  features_list reaches the kmeans_batch_size.
            if len(features_list) >= kmeans_batch_size:
                model = model.fit(features_list)
                features_list = []
            # extract features from the SSL model
            accumulate_and_extract_features(
                batch, features_list, ssl_model, ssl_layer_num, device
            )


def save_model(model, checkpoint_path):
    """Save a  Kmeans model .

    Arguments
    ---------
        model : MiniBatchKMeans
            The  kmeans model to be saved.
        checkpoint_path : str)
            Path to save the model..
    """
    joblib.dump(model, open(checkpoint_path, "wb"))
