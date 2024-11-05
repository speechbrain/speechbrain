"""
Recipe  to train K-means clustering model on self-supervised representations.

To run this recipe, do the following:
> python train.py hparams/train_with_[SSL-model].yaml --data_folder=/path/to/LJSpeech
Author
 * Pooneh Mousavi 2023
"""

import os
import sys

import torchaudio
from hyperpyyaml import load_hyperpyyaml
from torch.utils.data import DataLoader

import speechbrain as sb
from speechbrain.dataio.dataloader import LoopedLoader
from speechbrain.utils.distributed import run_on_main
from speechbrain.utils.kmeans import fetch_kmeans_model, save_model, train
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


def dataio_prepare(hparams):

    # Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        info = torchaudio.info(wav)
        resampled = torchaudio.transforms.Resample(
            info.sample_rate,
            hparams["sample_rate"],
        )(sig)
        return resampled

    datasets = {}
    data_info = {
        "train": hparams["train_json"],
    }
    for dataset in hparams["splits"]:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=data_info[dataset],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline],
            output_keys=["id", "sig"],
        )

    return datasets

    return datasets


if __name__ == "__main__":
    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Dataset prep (parsing Librispeech)
    from ljspeech_prepare import prepare_ljspeech  # noqa

    # multi-gpu (ddp) save data preparation
    run_on_main(
        prepare_ljspeech,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["save_folder"],
            "splits": hparams["splits"],
            "split_ratio": hparams["split_ratio"],
            "seed": hparams["seed"],
            "skip_prep": hparams["skip_prep"],
        },
    )

    # Load SSL model
    hparams["ssl_model"] = hparams["ssl_model"].to(run_opts["device"])

    # Make training Dataloader
    train_set = dataio_prepare(hparams)["train"]
    if not (
        isinstance(train_set, DataLoader) or isinstance(train_set, LoopedLoader)
    ):
        train_set = sb.dataio.dataloader.make_dataloader(
            train_set, **hparams["train_dataloader_opts"]
        )
    os.makedirs(hparams["save_folder"], exist_ok=True)
    # If you use dataloader checkpoints, make sure to keep all the settings as in the previous run and keep the dataset ordering the same.
    dataloader_path = os.path.join(
        hparams["save_folder"], "dataloader-TRAIN.ckpt"
    )
    if os.path.exists(dataloader_path):
        logger.info(
            f"The dataloader checkpoint is loaded from {dataloader_path}."
        )
        train_set._speechbrain_load(dataloader_path, False)

    # Load pretrained KMeans model if it exists. Otherwise,  create new one.
    checkpoint_path = os.path.join(
        hparams["save_folder"],
        f"kmeans-cluster-{hparams['num_clusters']}-layer-{hparams['ssl_layer_num']}.pt",
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

    # Train and save Kmeans model
    train(
        kmeans_model,
        train_set,
        hparams["ssl_model"],
        hparams["save_folder"],
        hparams["ssl_layer_num"],
        kmeans_batch_size=hparams["kmeans_batch_size"],
        device=run_opts["device"],
        checkpoint_interval=hparams["checkpoint_interval"],
    )

    logger.info(f"Saving kmeans model at {checkpoint_path}.")
    save_model(kmeans_model, checkpoint_path)
    train_set._speechbrain_save(dataloader_path)
