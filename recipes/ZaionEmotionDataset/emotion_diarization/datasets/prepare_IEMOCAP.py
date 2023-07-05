"""
Data preparation for IEMOCAP.

Dataset link: https://sail.usc.edu/iemocap/iemocap_release.htm

extra dependencies: pathlib, pydub, webrtcvad

Author
------
Yingzhi Wang 2023
"""

import numpy as np
import re
import os
import random
from pydub import AudioSegment
import json
from datasets.vad import write_audio
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
combinations = ["neu_emo", "emo_neu", "neu_emo_neu", "emo_emo"]
probabilities = np.array([0.25, 0.25, 0.25, 0.25])


def prepare_iemocap(
    data_folder, save_json, seed=12,
):
    """
    Prepares the json files for the IEMOCAP dataset.
    Arguments
    ---------
    data_original : str
        Path to the folder where the original IEMOCAP dataset is stored.
    save_json : str
        Path where the data specification file will be saved.
    seed : int
        Seed for reproducibility
    """
    random.seed(seed)

    # Check if this phase is already done (if so, skip it)
    if skip(save_json):
        logger.info("Preparation completed in previous run, skipping.")
        return

    logger.info("Applying VAD ...")
    dict = {}
    emotion_wavs = []
    neutral_wavs = []
    annotations = Path(data_folder).rglob("*/dialog/EmoEvaluation/*.txt")
    for i in annotations:
        lines = load_utterInfo(i)
        for line in lines:
            id = line[2]
            if line[3] == "hap":
                dict[id] = "happy"
                append_path_after_vad(data_folder, id, emotion_wavs)

            if line[3] == "exc":
                dict[id] = "happy"
                append_path_after_vad(data_folder, id, emotion_wavs)

            if line[3] == "sad":
                dict[id] = "sad"
                append_path_after_vad(data_folder, id, emotion_wavs)

            if line[3] == "ang":
                dict[id] = "angry"
                append_path_after_vad(data_folder, id, emotion_wavs)

            if line[3] == "neu":
                dict[id] = "neutral"
                append_path_after_vad(data_folder, id, neutral_wavs)

    logger.info("VAD finished")
    logger.info("Start IEMOCAP concatenation ...")
    data_json = concat_wavs(
        data_folder, save_json, emotion_wavs, neutral_wavs, dict
    )
    logger.info("IEMOCAP concatenation finished ...")
    return data_json


def append_path_after_vad(data_folder, id, list):
    """do vad and append the new path into the list

    Args:
        data_folder (str): the path to IEMOCAP
        id (str): id d'utterance
        list (list): which list to be put into

    Returns:
        list: new list after adding an element
    """
    file = get_path(data_folder, id)
    destin_folder = os.path.join(data_folder, "processed", id[:5] + id[-4])
    if not os.path.exists(destin_folder):
        os.makedirs(destin_folder)
    if not os.path.exists(os.path.join(destin_folder, f"{id}.wav")):
        write_audio(file, os.path.join(destin_folder, f"{id}.wav"))
    if os.path.exists(os.path.join(destin_folder, f"{id}.wav")):
        list.append(os.path.join(destin_folder, f"{id}.wav"))


def load_utterInfo(inputFile):
    """
    Load utterInfo from original IEMOCAP database
    """
    # this regx allow to create a list with:
    # [START_TIME - END_TIME] TURN_NAME EMOTION [V, A, D]
    # [V, A, D] means [Valence, Arousal, Dominance]
    pattern = re.compile(
        "[\[]*[0-9]*[.][0-9]*[ -]*[0-9]*[.][0-9]*[\]][\t][a-z0-9_]*[\t][a-z]{3}[\t][\[][0-9]*[.][0-9]*[, ]+[0-9]*[.][0-9]*[, ]+[0-9]*[.][0-9]*[\]]",
        re.IGNORECASE,
    )  # noqa
    with open(inputFile, "r") as myfile:
        data = myfile.read().replace("\n", " ")
    result = pattern.findall(data)
    out = []
    for i in result:
        a = i.replace("[", "")
        b = a.replace(" - ", "\t")
        c = b.replace("]", "")
        x = c.replace(", ", "\t")
        out.append(x.split("\t"))
    return out


def resampling_for_folder(in_folder, out_folder):
    """
    Resamples all the audios from a certain folder to 16kHz
    """
    files = os.listdir(in_folder)
    for file_name in files:
        try:
            sound = AudioSegment.from_file(
                os.path.join(in_folder, file_name), format="wav"
            )
            sound = sound.set_frame_rate(16000)
            sound.export(os.path.join(out_folder, file_name), format="wav")
        except Exception as e:
            logger.info(e)


def get_path(datafolder, id):
    """
    Get the filepath with ID
    """
    if "Ses01" in id:
        return os.path.join(
            datafolder, "Session1/sentences/wav", id[:-5], id + ".wav"
        )
    if "Ses02" in id:
        return os.path.join(
            datafolder, "Session2/sentences/wav", id[:-5], id + ".wav"
        )
    if "Ses03" in id:
        return os.path.join(
            datafolder, "Session3/sentences/wav", id[:-5], id + ".wav"
        )
    if "Ses04" in id:
        return os.path.join(
            datafolder, "Session4/sentences/wav", id[:-5], id + ".wav"
        )
    if "Ses05" in id:
        return os.path.join(
            datafolder, "Session5/sentences/wav", id[:-5], id + ".wav"
        )


def get_emotion(wav_path, annotations):
    """
    Get the emotion of an audio from its filepath
    """
    id = wav_path.split("/")[-1][:-4]
    return annotations[id]


def concat_wavs(data_folder, save_json, emo_wavs, neu_wavs, annotations):
    """
    Concatenate audios from the same speaker
    The concatenation is produced with a probability for each modality
    The amplitude of the sub-sentences are made equal during concatenation
    """
    repos = [
        "Ses01F",
        "Ses01M",
        "Ses02F",
        "Ses02M",
        "Ses03F",
        "Ses03M",
        "Ses04F",
        "Ses04M",
        "Ses05F",
        "Ses05M",
    ]
    data_json = {}
    for repo in repos:
        emotion_wavs = [
            i for i in emo_wavs if repo in i and f"_{repo[-1]}" in i
        ]
        neutral_wavs = [
            i for i in neu_wavs if repo in i and f"_{repo[-1]}" in i
        ]

        random.shuffle(emotion_wavs)
        random.shuffle(neutral_wavs)
        neutral_wavs = neutral_wavs * 10

        combine_path = os.path.join(data_folder, "combined", repo)
        if not os.path.exists(combine_path):
            os.makedirs(combine_path)

        while len(emotion_wavs) > 0:
            combination = np.random.choice(
                combinations, p=probabilities.ravel()
            )

            if combination == "neu_emo":
                neutral_sample = neutral_wavs[0]
                emo_sample = emotion_wavs[0]

                neutral_input = AudioSegment.from_wav(neutral_sample)
                emotion_input = AudioSegment.from_wav(emo_sample)

                emotion_input += neutral_input.dBFS - emotion_input.dBFS
                combined_input = neutral_input + emotion_input

                out_name = os.path.join(
                    combine_path,
                    neutral_sample.split("/")[-1][:-4]
                    + "_"
                    + emo_sample.split("/")[-1],
                )
                combined_input.export(out_name, format="wav")

                id = out_name.split("/")[-1][:-4]
                data_json[id] = {
                    "wav": out_name,
                    "duration": len(combined_input) / 1000,
                    "emotion": [
                        {
                            "emo": get_emotion(emo_sample, annotations),
                            "start": len(neutral_input) / 1000,
                            "end": len(combined_input) / 1000,
                        }
                    ],
                }

                neutral_wavs = neutral_wavs[1:]
                emotion_wavs = emotion_wavs[1:]

            elif combination == "emo_neu":
                neutral_sample = neutral_wavs[0]
                emo_sample = emotion_wavs[0]

                neutral_input = AudioSegment.from_wav(neutral_sample)
                emotion_input = AudioSegment.from_wav(emo_sample)

                neutral_input += emotion_input.dBFS - neutral_input.dBFS
                combined_input = emotion_input + neutral_input

                out_name = os.path.join(
                    combine_path,
                    emo_sample.split("/")[-1][:-4]
                    + "_"
                    + neutral_sample.split("/")[-1],
                )
                combined_input.export(out_name, format="wav")

                id = out_name.split("/")[-1][:-4]
                data_json[id] = {
                    "wav": out_name,
                    "duration": len(combined_input) / 1000,
                    "emotion": [
                        {
                            "emo": get_emotion(emo_sample, annotations),
                            "start": 0,
                            "end": len(emotion_input) / 1000,
                        }
                    ],
                }

                emotion_wavs = emotion_wavs[1:]
                neutral_wavs = neutral_wavs[1:]

            elif combination == "neu_emo_neu":
                neutral_sample_1 = neutral_wavs[0]
                neutral_sample_2 = neutral_wavs[1]
                emo_sample = emotion_wavs[0]

                neutral_input_1 = AudioSegment.from_wav(neutral_sample_1)
                neutral_input_2 = AudioSegment.from_wav(neutral_sample_2)
                emotion_input = AudioSegment.from_wav(emo_sample)

                emotion_input += neutral_input_1.dBFS - emotion_input.dBFS
                neutral_input_2 += neutral_input_1.dBFS - neutral_input_2.dBFS
                combined_input = (
                    neutral_input_1 + emotion_input + neutral_input_2
                )

                out_name = os.path.join(
                    combine_path,
                    neutral_sample_1.split("/")[-1][:-4]
                    + "_"
                    + emo_sample.split("/")[-1][:-4]
                    + "_"
                    + neutral_sample_2.split("/")[-1],
                )
                combined_input.export(out_name, format="wav")

                id = out_name.split("/")[-1][:-4]
                data_json[id] = {
                    "wav": out_name,
                    "duration": len(combined_input) / 1000,
                    "emotion": [
                        {
                            "emo": get_emotion(emo_sample, annotations),
                            "start": len(neutral_input_1) / 1000,
                            "end": len(neutral_input_1) / 1000
                            + len(emotion_input) / 1000,
                        }
                    ],
                }

                emotion_wavs = emotion_wavs[1:]
                neutral_wavs = neutral_wavs[2:]

            else:
                emo_sample_1 = emotion_wavs[0]

                emotion_input_1 = AudioSegment.from_wav(emo_sample_1)

                out_name = os.path.join(
                    combine_path, emo_sample_1.split("/")[-1]
                )
                emotion_input_1.export(out_name, format="wav")

                id = out_name.split("/")[-1][:-4]
                data_json[id] = {
                    "wav": out_name,
                    "duration": len(emotion_input_1) / 1000,
                    "emotion": [
                        {
                            "emo": get_emotion(emo_sample_1, annotations),
                            "start": 0,
                            "end": len(emotion_input_1) / 1000,
                        }
                    ],
                }

                emotion_wavs = emotion_wavs[1:]

    with open(save_json, "w") as outfile:
        json.dump(data_json, outfile)
    return data_json


def skip(save_json):
    """
    Detects if the data preparation has been already done.
    If the preparation has been done, we can skip it.
    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """
    if not os.path.isfile(save_json):
        return False
    return True
