"""
Data preparation for Emotion Speech Dataset (ESD).

Dataset link: https://github.com/HLTSingapore/Emotional-Speech-Data

extra dependencies: pydub, webrtcvad

Author
------
Yingzhi Wang 2023
"""

import numpy as np
import os
import random
from pydub import AudioSegment
import json
from datasets.vad import vad_for_folder
import logging

logger = logging.getLogger(__name__)
# we choose here only english utterances
repos = [
    "0011",
    "0012",
    "0013",
    "0014",
    "0015",
    "0016",
    "0017",
    "0018",
    "0019",
    "0020",
]
sub_folders = ["Angry", "Happy", "Neutral", "Sad"]
sub_sub_folders = ["train", "evaluation", "test"]
combinations = ["neu_emo", "emo_neu", "neu_emo_neu", "emo_emo"]
probabilities = np.array([0.25, 0.25, 0.25, 0.25])


def prepare_esd(
    data_folder, save_json, seed=12,
):
    """
    Prepares the json files for the ESD dataset.
    Arguments
    ---------
    data_original : str
        Path to the folder where the original ESD dataset is stored.
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

    logger.info("Applying VAD and resampling ...")
    for repo in repos:
        for sub_folder in sub_folders:
            for sub_sub_folder in sub_sub_folders:
                source_folder = os.path.join(
                    data_folder, repo, sub_folder, sub_sub_folder
                )
                destin_folder = os.path.join(
                    data_folder, "processed", repo, sub_folder
                )
                if not os.path.exists(destin_folder):
                    os.makedirs(destin_folder)
                vad_for_folder(source_folder, destin_folder)
    logger.info("vad and resampling finished")
    logger.info("Start ESD concatenation ...")
    data_json = concat_wavs(data_folder, save_json)
    logger.info("ESD concatenation finished ...")
    return data_json


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


def get_emotion(wav_path):
    """
    Get the emotion of an audio from its filepath
    """
    num = wav_path.split("_")[-1][:-4]
    num = int(num) - 1
    if num // 350 == 1:
        return "angry"
    elif num // 350 == 2:
        return "happy"
    elif num // 350 == 3:
        return "sad"


def concat_wavs(data_folder, save_json):
    """
    Concatenate audios from the same speaker
    The concatenation is produced with a probability for each modality
    The amplitude of the sub-sentences are made equal during concatenation
    """
    data_json = {}
    for repo in repos:
        emotion_wavs = []
        neutral_wavs = []

        angry_files = os.listdir(
            os.path.join(data_folder, "processed", repo, "Angry")
        )
        happy_files = os.listdir(
            os.path.join(data_folder, "processed", repo, "Happy")
        )
        sad_files = os.listdir(
            os.path.join(data_folder, "processed", repo, "Sad")
        )
        neutral_files = os.listdir(
            os.path.join(data_folder, "processed", repo, "Neutral")
        )

        for file in angry_files:
            emotion_wavs.append(
                os.path.join(data_folder, "processed", repo, "Angry", file)
            )
        for file in happy_files:
            emotion_wavs.append(
                os.path.join(data_folder, "processed", repo, "Happy", file)
            )
        for file in sad_files:
            emotion_wavs.append(
                os.path.join(data_folder, "processed", repo, "Sad", file)
            )
        for file in neutral_files:
            neutral_wavs.append(
                os.path.join(data_folder, "processed", repo, "Neutral", file)
            )

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
                    + emo_sample.split("_")[-1],
                )
                combined_input.export(out_name, format="wav")

                id = out_name.split("/")[-1][:-4]
                data_json[id] = {
                    "wav": out_name,
                    "duration": len(combined_input) / 1000,
                    "emotion": [
                        {
                            "emo": get_emotion(emo_sample),
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
                    + neutral_sample.split("_")[-1],
                )
                combined_input.export(out_name, format="wav")

                id = out_name.split("/")[-1][:-4]
                data_json[id] = {
                    "wav": out_name,
                    "duration": len(combined_input) / 1000,
                    "emotion": [
                        {
                            "emo": get_emotion(emo_sample),
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
                    + emo_sample.split("/")[-1][4:-4]
                    + "_"
                    + neutral_sample_2.split("_")[-1],
                )
                combined_input.export(out_name, format="wav")

                id = out_name.split("/")[-1][:-4]
                data_json[id] = {
                    "wav": out_name,
                    "duration": len(combined_input) / 1000,
                    "emotion": [
                        {
                            "emo": get_emotion(emo_sample),
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
                            "emo": get_emotion(emo_sample_1),
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
