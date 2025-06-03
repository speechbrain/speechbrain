"""
Apply speaker recognition model to extract speaker embeddings for HiFi-GAN training.

Authors
 * Jarod Duret 2023
"""

import json
import logging
import pathlib as pl

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

import speechbrain as sb
from speechbrain.dataio.dataio import load_pkl, save_pkl
from speechbrain.inference.encoders import MelSpectrogramEncoder
from speechbrain.utils.logger import get_logger

OPT_FILE = "opt_libritts_extract_speaker.pkl"
TRAIN_JSON = "train.json"
VALID_JSON = "valid.json"
TEST_JSON = "test.json"


def setup_logger():
    """Set up a logger with a log format and logging level."""
    log_format = "[%(asctime)s] [%(levelname)s]: %(message)s"
    logging.basicConfig(format=log_format, level=logging.INFO)
    logger = get_logger(__name__)
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

    Arguments
    ---------
    splits : list
        List of splits to check.
    save_folder : str
        Path to the folder where the speech units are stored.
    conf : dict
        Loaded configuration options

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


def extract_libritts_embeddings(
    data_folder,
    splits,
    encoder_source,
    save_folder,
    sample_rate=16000,
    skip_extract=False,
):
    """
    Extract speaker embeddings for HiFi-GAN training on the LibriTTS datasets.

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original LibriTTS dataset is stored.
    splits : list
        List of splits to prepare.
    encoder_source: str
        Url to the model used as embedding extractor.
    save_folder: str
        Path to the folder where the speech units are stored.
    sample_rate: int
        LibriTTS dataset sample rate
    skip_extract: Bool
        If True, skip extraction.

    Returns
    -------
    None

    Example
    -------
    >>> from recipes.LibriTTS.vocoder.hifigan_unit.extract_speaker_embeddings import extract_libritts_embeddings
    >>> data_folder = 'data/libritts/'
    >>> splits = ['train', 'valid']
    >>> encoder_source = facebook/hubert-base-ls960
    >>> save_folder = 'save/'
    >>> extract_libritts_embeddings(data_folder, splits, encoder_source, save_folder)
    """
    logger = setup_logger()

    if skip_extract:
        return
    # Create configuration for easily skipping extraction stage
    conf = {
        "data_folder": data_folder,
        "splits": splits,
        "save_folder": save_folder,
        "encoder_source": encoder_source,
    }

    save_folder = pl.Path(save_folder)
    # Check if this phase is already done (if so, skip it)
    if skip(splits, save_folder, conf):
        logger.info(
            "Skipping speaker embeddings extraction, completed in previous run."
        )
        return

    # Fetch device
    device = get_device(use_cuda=True)

    save_opt = save_folder / OPT_FILE
    data_folder = pl.Path(data_folder)
    save_path = save_folder / "savedir/melspec_encoder"
    speaker_folder = save_folder / "speaker_embeddings"
    speaker_folder.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading encoder: {encoder_source} ...")
    encoder = MelSpectrogramEncoder.from_hparams(
        source=encoder_source,
        run_opts={"device": str(device)},
        savedir=save_path,
    )

    for split in splits:
        dataset_path = save_folder / f"{split}.json"
        logger.info(f"Reading dataset from {dataset_path} ...")
        meta_json = json.load(open(dataset_path, encoding="utf-8"))
        for key in tqdm(meta_json.keys()):
            item = meta_json[key]
            wav = item["wav"]
            with torch.no_grad():
                info = torchaudio.info(wav)
                audio = sb.dataio.dataio.read_audio(wav)
                audio = torchaudio.transforms.Resample(
                    info.sample_rate,
                    sample_rate,
                )(audio)
                audio = audio.to(device)
                feats = encoder.encode_waveform(audio)
                feats = np_array(feats.squeeze(0))
            np.save(speaker_folder / f"{key}.npy", feats)

    logger.info("Extraction completed.")
    save_pkl(conf, save_opt)
