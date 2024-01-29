"""
Apply K-means clustering over acoustic features to extract speech units for HiFi-GAN training.

Authors
 * Jarod Duret 2023
"""

import logging
import json
import pathlib as pl

import torch
import torchaudio
import numpy as np
from tqdm import tqdm
import speechbrain as sb
from speechbrain.dataio.dataio import (
    load_pkl,
    save_pkl,
)
from speechbrain.lobes.models.huggingface_transformers import (
    discrete_hubert,
    discrete_wav2vec2,
    discrete_wavlm,
)

OPT_FILE = "opt_libritts_extract.pkl"
TRAIN_JSON = "train.json"
VALID_JSON = "valid.json"
TEST_JSON = "test.json"

ENCODER_CLASSES = {
    "HuBERT": discrete_hubert.DiscreteHuBERT,
    "Wav2Vec2": discrete_wav2vec2.DiscreteWav2Vec2,
    "WavLM": discrete_wavlm.DiscreteWavLM,
}


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
    Detects if the libritts data_extraction has been already done.
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
        if (
            split in split_files
            and not (save_folder / split_files[split]).exists()
        ):
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


def extract_libritts(
    data_folder,
    splits,
    kmeans_folder,
    kmeans_filename,
    encoder_type,
    encoder_source,
    layer,
    save_folder,
    sample_rate=16000,
    skip_extract=False,
):
    """
    Extract speech units for HiFi-GAN training on the Libri-TTS datasets.

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original Libri-TTS dataset is stored.
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
        Libri-TTS dataset sample rate
    skip_extract: Bool
        If True, skip extraction.

    Example
    -------
    >>> from recipes.LibriTTS.vocoder.hifigan_unit.extract_code import extract_libritts
    >>> data_folder = 'data/libritts/'
    >>> splits = ['train', 'valid']
    >>> kmeans_folder = ./Quantization/results/kmeans/4321/save
    >>> encoder = facebook/hubert-base-ls960
    >>> layer = 6
    >>> save_folder = 'save/'
    >>> extract_libritts(data_folder, splits, kmeans_folder, encoder, layer, save_folder)
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
        "encoder_type": encoder_type,
        "encoder_source": encoder_source,
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
    save_path = save_folder / "savedir"
    code_folder = save_folder / "codes"
    code_folder.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading encoder: {encoder_source} ...")
    if encoder_type not in ENCODER_CLASSES:
        raise TypeError("Not a supported Encoder")

    encoder_class = ENCODER_CLASSES[encoder_type]
    encoder = encoder_class(
        source=encoder_source,
        save_path=save_path.as_posix(),
        kmeans_filename=kmeans_filename,
        kmeans_cache_dir=save_path.as_posix(),
        kmeans_repo_id=kmeans_folder,
        output_norm=False,
        freeze=True,
        freeze_feature_extractor=True,
        apply_spec_augment=False,
        output_all_hiddens=True,
        ssl_layer_num=layer,
    ).to(device)

    for split in splits:
        dataset_path = save_folder / f"{split}.json"
        logger.info(f"Reading dataset from {dataset_path} ...")
        meta_json = json.load(open(dataset_path))
        for key in tqdm(meta_json.keys()):
            item = meta_json[key]
            wav = item["wav"]
            wav = wav.replace("{data_root}", data_folder.as_posix())
            with torch.no_grad():
                info = torchaudio.info(wav)
                audio = sb.dataio.dataio.read_audio(wav)
                audio = torchaudio.transforms.Resample(
                    info.sample_rate, sample_rate,
                )(audio)
                audio = audio.unsqueeze(0).to(device)
                _, tokens = encoder(audio)
                tokens = np_array(tokens)
            np.save(code_folder / f"{key}.npy", tokens)

    logger.info("Extraction completed.")
    save_pkl(conf, save_opt)
