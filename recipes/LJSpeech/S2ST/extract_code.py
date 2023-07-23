"""
Apply K-means clustering over acoustic features.

Authors
 * Jarod Duret 2023
"""

import logging
import json
import pathlib as pl

import joblib
import torch
import numpy as np
from tqdm import tqdm
import speechbrain as sb
from speechbrain.dataio.dataio import (
    load_pkl,
    save_pkl,
)
from speechbrain.lobes.models.huggingface_wav2vec import HuggingFaceWav2Vec2

OPT_FILE = "opt_ljspeech_extract.pkl"
TRAIN_JSON = "train.json"
VALID_JSON = "valid.json"
TEST_JSON = "test.json"


def setup_logger():
    log_format = "[%(asctime)s] [%(levelname)s]: %(message)s"
    logging.basicConfig(format=log_format, level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger


def get_device(use_cuda):
    use_cuda = use_cuda and torch.cuda.is_available()
    print("\n" + "=" * 30)
    print("USE_CUDA SET TO: {}".format(use_cuda))
    print("CUDA AVAILABLE?: {}".format(torch.cuda.is_available()))
    print("=" * 30 + "\n")
    return torch.device("cuda" if use_cuda else "cpu")


def np_array(tensor):
    tensor = tensor.squeeze(0)
    tensor = tensor.detach().cpu()
    return tensor.numpy()


def skip(splits, save_folder, conf):
    """
    Detects if the ljspeech data_extraction has been already done.
    If the extraction has been done, we can skip it.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """
    # Checking json files
    skip = True

    split_files = {
        "train": TRAIN_JSON,
        "valid": VALID_JSON,
        "test": TEST_JSON,
    }

    for split in splits:
        if not (save_folder / split_files[split]).exists():
            skip = False

    #  Checking saved options
    save_opt = save_folder / OPT_FILE
    if skip is True:
        if save_opt.is_file():
            opts_old = load_pkl(save_opt.as_posix())
            if opts_old == conf:
                skip = True
            else:
                skip = False
        else:
            skip = False
    return skip


def extract_ljspeech(
    data_folder,
    splits,
    kmeans_folder,
    encoder,
    layer,
    save_folder,
    skip_extract=False,
):
    logger = setup_logger()

    if skip_extract:
        return
    # Create configuration for easily skipping data_preparation stage
    conf = {
        "data_folder": data_folder,
        "splits": splits,
        "save_folder": save_folder,
        "kmeans_folder": kmeans_folder,
        "encoder": encoder,
        "layer": layer,
    }

    save_folder = pl.Path(save_folder)
    # Check if this phase is already done (if so, skip it)
    if skip(splits, save_folder, conf):
        logger.info("Skipping code extraction, completed in previous run.")
        return

    # Fetch device
    device = get_device(use_cuda=True)

    save_opt = save_folder / OPT_FILE
    data_folder = pl.Path(data_folder)
    kmeans_folder = pl.Path(kmeans_folder)
    kmeans_ckpt = kmeans_folder / "kmeans.cpt"
    encoder_save_path = kmeans_folder / "pretrained_models"
    code_folder = save_folder / "codes"
    code_folder.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading encoder: {encoder} ...")
    encoder = HuggingFaceWav2Vec2(
        encoder,
        encoder_save_path.as_posix(),
        output_all_hiddens=True,
        output_norm=False,
        freeze_feature_extractor=True,
        freeze=True,
    ).to(device)

    # K-means model
    logger.info(f"Loading K-means model from {kmeans_ckpt} ...")
    kmeans_model = joblib.load(open(kmeans_ckpt, "rb"))
    kmeans_model.verbose = False

    for split in splits:
        dataset_path = data_folder / f"{split}.json"
        logger.info(f"Reading dataset from {dataset_path} ...")
        meta_json = json.load(open(dataset_path))
        for key in tqdm(meta_json.keys()):
            item = meta_json[key]
            wav = item["wav"]
            with torch.no_grad():
                audio = sb.dataio.dataio.read_audio(wav)
                audio = audio.unsqueeze(0).to(device)
                feats = encoder.extract_features(audio)
                feats = feats[layer]
                feats = np_array(feats)
            pred = kmeans_model.predict(feats)
            np.save(code_folder / f"{key}.npy", pred)

    logger.info(f"Extraction completed.")
    save_pkl(conf, save_opt)
