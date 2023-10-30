"""
Apply K-means clustering over acoustic features to extract speech units for training the speech-to-unit translation model.

Authors
 * Jarod Duret 2023
"""

import logging
import json
import pathlib as pl

import joblib
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
import speechbrain as sb
from speechbrain.dataio.dataio import (
    load_pkl,
    save_pkl,
)
from speechbrain.lobes.models.huggingface_wav2vec import HuggingFaceWav2Vec2
from huggingface_hub import hf_hub_download

OPT_FILE = "opt_cvss_extract.pkl"
TRAIN_JSON = "train.json"
VALID_JSON = "valid.json"
VALID_SMALL = "valid_small.json"
TEST_JSON = "test.json"


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
        "valid_small": VALID_SMALL,
        "test": TEST_JSON,
    }

    for split in splits:
        if not (save_folder / split_files[split]).exists():
            skip = False

    code_folder = save_folder / "codes"
    if not code_folder.exists():
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


def extract_cvss(
    data_folder,
    splits,
    kmeans_folder,
    encoder,
    layer,
    save_folder,
    sample_rate=16000,
    skip_extract=False,
):
    """
    Extract speech units for HiFi-GAN training on the CVSS datasets.

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original CVSS dataset is stored.
    splits : list
        List of splits to prepare.
    kmeans_folder: str
        Path to the folder where the k-means model checkpoint is stored.
    encoder: str
        Url to the model used as feature extractor.
    layer: int
        Layer from which features are extracted.
    save_folder: str
        Path to the folder where the speech units are stored.
    sample_rate: int
        CVSS dataset sample rate
    skip_extract: Bool
        If True, skip extraction.

    Example
    -------
    >>> from recipes.CVSS.S2ST.extract_code import extract_cvss
    >>> data_folder = 'data/CVSS/'
    >>> splits = ['train', 'valid']
    >>> kmeans_folder = ./Quantization/results/kmeans/4321/save
    >>> encoder = facebook/hubert-base-ls960
    >>> layer = 6
    >>> save_folder = 'save/'
    >>> extract_cvss(data_folder, splits, kmeans_folder, encoder, layer, save_folder)
    """
    logger = setup_logger()

    if skip_extract:
        return
    # Create configuration for easily skipping code extraction stage
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

    # Fetch K-means model
    kmeans_folder = pl.Path(kmeans_folder)
    kmeans_ckpt = kmeans_folder / "kmeans.ckpt"
    if not kmeans_ckpt.exists():
        logger.info("K-means checkpoint not found, downloading it from HF.")
        kmeans_download_path = save_folder / "pretrained_models/quantization"
        kmeans_download_path.mkdir(exist_ok=True, parents=True)
        hf_hub_download(
            repo_id=kmeans_folder.as_posix(),
            filename="kmeans.ckpt",
            local_dir=kmeans_download_path,
        )
        kmeans_ckpt = kmeans_download_path / "kmeans.ckpt"

    encoder_save_path = save_folder / "pretrained_models"
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
            wav = item["tgt_audio"]
            with torch.no_grad():
                info = torchaudio.info(wav)
                audio = sb.dataio.dataio.read_audio(wav)
                audio = torchaudio.transforms.Resample(
                    info.sample_rate, sample_rate,
                )(audio)
                audio = audio.unsqueeze(0).to(device)
                feats = encoder.extract_features(audio)
                feats = feats[layer]
                feats = np_array(feats)
            pred = kmeans_model.predict(feats)
            np.save(code_folder / f"{key}_tgt.npy", pred)

    logger.info("Extraction completed.")
    save_pkl(conf, save_opt)
