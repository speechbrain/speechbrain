"""
Note for reviewer: this is a temporary script. It may be removed in the future.
Note2: for EU/US users, using this script might be VERY slow. It is instead
recommended to use the HuggingFace script.

Download script for GigaSpeech dataset.

Download instructions: https://github.com/SpeechColab/GigaSpeech
Reference: https://arxiv.org/abs/2106.06909

Author
-------
 * Adel Moumen, 2024
"""

import logging
from typing import Optional, Sequence, Union
import argparse

logger = logging.getLogger(__name__)


def download_gigaspeech(
    password: str,
    target_dir: str = ".",
    dataset_parts: Optional[Union[str, Sequence[str]]] = "auto",
    host: Optional[str] = "tsinghua",
) -> None:
    """Download GigaSpeech dataset.

    Parameters
    ----------
    password : str
        The password to access the GigaSpeech dataset.
    target_dir : str, optional
        The path to the directory where the dataset will be downloaded.
    dataset_parts : Union[str, Sequence[str]], optional
        The parts of the dataset to be downloaded.
        If "auto", all parts will be downloaded.
        If a string, it should be a comma-separated list of parts to be downloaded.
        If a list, it should be a list of parts to be downloaded.
    host : str, optional
        The host to be used for downloading the dataset.
        The available hosts are described in https://github.com/SpeechColab/GigaSpeech.
    """
    try:
        from speechcolab.datasets.gigaspeech import GigaSpeech
    except ImportError:
        raise ImportError(
            "Please install the speechcolab package to download the GigaSpeech dataset."
        )
    gigaspeech = GigaSpeech(target_dir)

    if dataset_parts == ["auto"]:
        dataset_parts = ["XL", "DEV", "TEST"]

    for part in dataset_parts:
        logging.info(f"Downloading GigaSpeech part: {part}")
        gigaspeech.download(password, "{" + part + "}", host=host)

    logger.info(f"GigaSpeech dataset finished downloading to {target_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download GigaSpeech dataset.")
    parser.add_argument(
        "--password",
        type=str,
        required=True,
        help="The password to access the GigaSpeech dataset.",
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        default=".",
        help="The path to the directory where the dataset will be downloaded.",
    )
    parser.add_argument(
        "--dataset_parts",
        type=str,
        nargs="+",  # '+' means one or more values will be collected into a list
        default=["auto"],
        help="The parts of the dataset to be downloaded.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="tsinghua",
        help="The host to be used for downloading the dataset.",
    )
    args = parser.parse_args()

    download_gigaspeech(
        args.password, args.target_dir, args.dataset_parts, args.host
    )
