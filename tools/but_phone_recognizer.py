import os

from enum import Enum
from typing import List

from helper.HTK import HTKFile

import docker
import torch
import numpy as np


class RecognizeSystem(Enum):
    """
    The Systems of BUT
    ref: https://speech.fit.vutbr.cz/software/phoneme-recognizer-based-long-temporal-context

    PHN_CZ_SPDAT_LCRC_N1500 - 8kHz, 2 block STC, trained on Czech SpeechDat-E
    PHN_HU_SPDAT_LCRC_N1500 - 8kHz, 2 block STC, trained on Hungarian SpeechDat-E
    PHN_RU_SPDAT_LCRC_N1500 - 8kHz, 2 block STC, trained on Russian SpeechDat-E
    PHN_EN_TIMIT_LCRC_N500 - 16kHz, 2 block STC, trained on TIMIT, 15 banks
    """

    CZECH = "PHN_CZ_SPDAT_LCRC_N1500"
    HUNGARIAN = "PHN_HU_SPDAT_LCRC_N1500"
    RUSSIAN = "PHN_RU_SPDAT_LCRC_N1500"
    TIMIT = "PHN_EN_TIMIT_LCRC_N500"

    def __str__(self) -> str:
        return str(self.value)


def read_HTK_file(file_path: str) -> torch.Tensor:
    """Read HTK file and return"""
    htk_reader = HTKFile()
    htk_reader.load(file_path)

    result = np.array(htk_reader.data)

    return torch.from_numpy(result)


def read_phone_label(file_path: str) -> List[str]:
    """Read the given file to get phone labels"""
    phones = []
    with open(file_path, "r", encoding="utf-8") as phone_file:
        phone_lines = phone_file.readlines()

        for phone_line in phone_lines:
            phone_line = phone_line.split(" ")
            phones.append(phone_line[2])

    return phones


def recognize_phone_label(
    mount_path: str, wav_path: str, system: RecognizeSystem
) -> List[str]:
    """Recognize the given file and return the phone labels"""
    client = docker.from_env()
    audio_name = wav_path.split(".")[0]
    feature_file = f"{audio_name}.fea"

    command = (
        f"./PhnRec/phnrec -v -c ./PhnRec/{system} "
        f"-i /usr/src/results/{wav_path} "
        f"-o /usr/src/results/{feature_file}"
    )
    client.containers.run(
        "phnrec",
        volumes={mount_path: {"bind": "/usr/src/results", "mode": "rw"}},
        command=command,
    )

    features = read_phone_label(feature_file)

    # clean up
    os.remove(feature_file)

    return features


def recognize_phone_posteriors(
    mount_path: str, wav_path: str, system: RecognizeSystem,
):
    """Recognize the given wav, and produce the result based on the recognize type"""
    client = docker.from_env()
    output_format = "-t post "
    audio_name = wav_path.split(".")[0]
    feature_file = f"{audio_name}.fea"

    command = (
        f"./PhnRec/phnrec -v -c ./PhnRec/{system} {output_format} "
        f"-i /usr/src/results/{wav_path} "
        f"-o /usr/src/results/{feature_file}"
    )
    client.containers.run(
        "phnrec",
        volumes={mount_path: {"bind": "/usr/src/results", "mode": "rw"}},
        command=command,
    )
