"""
Recipe  to train K-means clustering model on self-supervised representations.

To run this recipe, do the following:
> python train.py hparams/train_with_[SSL-model].yaml --data_folder=/path/to/LibriSPeech
Author
 * Pooneh Mousavi 2023
"""

import os
import sys
import logging
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main
from hyperpyyaml import load_hyperpyyaml
from tqdm.contrib import tqdm
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader
from speechbrain.dataio.dataloader import LoopedLoader
import joblib

logger = logging.getLogger(__name__)


def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"], replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    datasets = [train_data]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "sig"],
    )
    return train_data


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
    Args:
        n_clusters (int): The number of clusters to form as well as the number of centroids to generate.
        init (int):    Method for initialization: {'k-means++'', ''random''}
        max_iter (int): Maximum number of iterations over the complete dataset before stopping independently of any early stopping criterion heuristics.
        batch_size (int) : Size of the mini batches.
        tol (float): Control early stopping based on the relative center changes as measured by a smoothed, variance-normalized of the mean center squared position changes.
        max_no_improvement (int): Control early stopping based on the consecutive number of mini batches that does not yield an improvement on the smoothed inertia.
        n_init (int): Number of random initializations that are tried
        reassignment_ratio (float): Control the fraction of the maximum number of counts for a center to be reassigned.
        random_state (int): Determines random number generation for centroid initialization and random reassignment.
        compute_labels (bool): Compute label assignment and inertia for the complete dataset once the minibatch optimization has converged in fit.
        init_size (int): Number of samples to randomly sample for speeding up the initialization.
        checkpoint_path (str) : Path to saved model.
    Returns:
        MiniBatchKMeans: a k-means clustering model with specified parameters.
    """
    if os.path.exists(checkpoint_path):
        return joblib.load(checkpoint_path)
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

def train(model, train_set):
        # Train and save Kmeans model
    i =2
    with tqdm(train_set, dynamic_ncols=True,) as t:
        for batch in t:
            batch = batch.to(run_opts["device"])
            wavs, wav_lens = batch.sig
            wavs, wav_lens = (
                wavs.to(run_opts["device"]),
                wav_lens.to(run_opts["device"]),
            )
            feats = hparams["ssl_model"](wavs, wav_lens)[
                hparams["ssl_layer_num"]
            ].flatten(end_dim=-2)
            model = model.partial_fit(feats)



if __name__ == "__main__":
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

    # Dataset prep (parsing Librispeech)
    from librispeech_prepare import prepare_librispeech  # noqa

    # multi-gpu (ddp) save data preparation
    run_on_main(
        prepare_librispeech,
        kwargs={
            "data_folder": hparams["data_folder"],
            "tr_splits": hparams["train_splits"],
            "dev_splits": hparams["dev_splits"],
            "te_splits": hparams["test_splits"],
            "save_folder": hparams["output_folder"],
            "merge_lst": hparams["train_splits"],
            "merge_name": "train.csv",
            "skip_prep": hparams["skip_prep"],
        },
    )

    # Load SSL model
    hparams["ssl_model"] = hparams["ssl_model"].to(run_opts["device"])

    # Make training Dataloader
    train_set = dataio_prepare(hparams)
    if not (
        isinstance(train_set, DataLoader) or isinstance(train_set, LoopedLoader)
    ):
        train_set = sb.dataio.dataloader.make_dataloader(
            train_set, **hparams["train_dataloader_opts"]
        )

    # Load pretrained KMeans model if it exists. Otherwise,  create new one.
    checkpoint_path = os.path.join(
        hparams["save_folder"], f"kmeans_{hparams['num_clusters']}.pt"
    )

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
        checkpoint_path=checkpoint_path,
    )

    train(kmeans_model,train_set)

    joblib.dump(kmeans_model, open(checkpoint_path, "wb"))
