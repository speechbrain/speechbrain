"""
Data preparation

Author
-----
YAO-FEI, CHENG 2021
"""

import os
import json
import logging
import subprocess

from typing import List
from dataclasses import dataclass, field

import torch
import torchaudio

from splits.split_train import TrainOption
from pre_processing import (
    clean_transcription,
    clean_translation,
    normalize_punctuation,
    remove_punctuation,
)

from tqdm import tqdm
from sacremoses import MosesPunctNormalizer, MosesTokenizer
from speechbrain.utils.data_utils import get_all_files
from speechbrain.utils.torch_audio_backend import get_torchaudio_backend
from speechbrain.processing.speech_augmentation import Resample

logger = logging.getLogger(__name__)
torchaudio_backend = get_torchaudio_backend()
torchaudio.set_audio_backend(torchaudio_backend)

es_normalizer = MosesPunctNormalizer(lang="es")
en_normalizer = MosesPunctNormalizer(lang="en")

en_tokenizer = MosesTokenizer(lang="en")

SAMPLE_RATE = 16000


@dataclass
class TDF:
    """
    channel: int
        channel of utterance
    start: int
        start time of utterance
    end: int
        end time of utterance
    transcript: str
        transcript of utteranc
    """
    channel: int
    start: int
    end: int
    transcript: str


@dataclass
class Data:
    """
    each data contains a transcription and a translation for train set
    four translations for dev, dev2, test set
    """
    uid: str = ""
    wav: str = ""
    transcription: str = ""
    duration: float = 0
    translations: List[str] = field(default_factory=lambda: [])


def prepare_fisher_callhome_spanish(
    data_folder: str, save_folder: str, device: str = "cpu",
):

    """
    Prepares the json files for the Mini Fisher-Callhome-Spanish dataset.
    Arguments
    ---------
    data_folder : str
        Path to the folder where the Fisher-Callhome-Spanish dataset is stored.
    save_folder: str:
        Path of train/valid/test specification file will be saved.
    Example
    -------
    >>> data_folder = '/path/to/fisher-callhome'
    >>> save_foler = 'data'
    >>> prepare_fisher_callhome_spanish(data_folder, save_folder)
    """

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # If the dataset doesn't exist yet, terminate the whole program
    speech_folder = os.path.join(f"{data_folder}/LDC2010S01/data/speech")
    transcription_folder = os.path.join(f"{data_folder}/LDC2010T04/fisher_spa_tr/data/transcripts")

    if check_folders(speech_folder, transcription_folder) is not True:
        logger.error(
            "Speech or transcription directories are missing or not properly organised within the speech data dir"
            "Typical format is LDC2010S01/data/speech and LDC2010T04/fisher_spa_tr/data/transcripts"
        )
        return

    datasets = ["dev", "dev2", "test", "train"]

    corpus_path = f"{save_folder}/fisher-callhome-corpus"
    download_translations(path=corpus_path)

    for dataset in datasets:
        if not os.path.exists(f"{save_folder}/{dataset}/wav"):
            os.makedirs(f"{save_folder}/{dataset}/wav")

        if skip(save_folder, dataset):
            logger.info(f"Skipping preparation of {dataset}, completed in previous run.")
            continue

        # get file lists
        transcription_files = get_transcription_files_by_dataset(dataset, transcription_folder=transcription_folder)

        # extract all transcriptions from files
        extracted_transcriptions = {}
        for transcription_file in transcription_files:
            filename = transcription_file.split("/")[-1].split(".")[0]
            extracted_transcriptions[filename] = extract_transcription(transcription_file)

        # concate short utterance via mapping file
        concated_data = concate_transcriptions_by_mapping_file(
            speech_folder=speech_folder,
            mapping_file_path=f"{corpus_path}/mapping/fisher_{dataset}",
            extracted_transcriptions=extracted_transcriptions,
        )

        # get translation through fisher-callhome-corpus
        if dataset != "train":
            # dev, dev2, test got four translations
            for number in range(4):
                translation_path = f"{corpus_path}/corpus/ldc/fisher_{dataset}.en.{number}"
                translations = get_translations_from_path(translation_path)

                concated_data = insert_translation_into_existing_dataset(data=concated_data, translations=translations)
        else:
            translation_path = f"{corpus_path}/corpus/ldc/fisher_{dataset}.en"
            translations = get_translations_from_path(translation_path)
            concated_data = insert_translation_into_existing_dataset(data=concated_data, translations=translations)

        # filter out empty or long transcription/translation
        concated_data = list(filter(lambda data: 0 < len(data.transcription) < 400, concated_data))

        if dataset != "train":
            for number in range(4):
                concated_data = list(filter(lambda data: 0 < len(data.translations[number]) < 400, concated_data))
        else:
            concated_data = list(filter(lambda data: 0 < len(data.translations[0]) < 400, concated_data))

        # ignore empty or long utterances
        concated_data = list(filter(lambda data: 0 < data.duration < 30, concated_data))

        # sort by utterance id
        concated_data = sorted(concated_data, key=lambda data: data.uid)

        # store transcription/translation/wav files
        data_dict = {}
        for data in tqdm(concated_data, desc=f"pre-processing [{dataset}]"):
            wav_save_path = f"{save_folder}/{dataset}/wav/{data.uid}.wav"
            # prepare audio files
            wav_information = data.wav.split(" ")
            segment_audio(
                audio_path=wav_information[0],
                channel=int(wav_information[1]),
                start=int(wav_information[2]),
                end=int(wav_information[3]),
                save_path=wav_save_path,
                sample_rate=SAMPLE_RATE,
                device=device,
            )

            # prepare json file
            if dataset != "train":
                data_dict[data.uid] = {
                    "wav": "{data_root}" + f"/{dataset}/wav/{data.uid}.wav",
                    "duration": data.duration,
                    "transcription": data.transcription,
                }

                for number in range(4):
                    translation_dict = {f"translation_{number}": data.translations[number]}
                    data_dict[data.uid].update(translation_dict)
            else:
                data_dict[data.uid] = {
                    "wav": "{data_root}" + f"/{dataset}/wav/{data.uid}.wav",
                    "duration": data.duration,
                    "transcription": data.transcription,
                    "translation_0": data.translations[0],
                }

        # save json
        json_path = f"{save_folder}/{dataset}/data.json"
        with open(json_path, "w", encoding="utf-8") as data_json:
            json.dump(data_dict, data_json, indent=2, ensure_ascii=False)

        logger.info(f"{json_path} successfully created!")


def skip(save_folder: str, dataset: str) -> bool:
    """Detect when fisher-callhome data preparation can be skipped"""
    is_skip = True

    if not os.path.isfile(f"{save_folder}/{dataset}/data.json"):
        is_skip = False

    return is_skip


def check_folders(*folders) -> bool:
    """Returns False if any passed folder does not exist."""
    for folder in folders:
        if not os.path.exists(folder):
            return False
    return True


def get_train_split_list(train_option: TrainOption) -> str:
    if train_option == TrainOption.ALL:
        return get_data_list("splits/train_all")
    elif train_option == TrainOption.MID:
        return get_data_list("splits/train_mid")
    elif train_option == TrainOption.LOW:
        return get_data_list("splits/train_low")
    else:
        logger.warning("please checkout the split list")


def get_data_list(path: str) -> str:
    with open(path, "r", encoding="utf-8") as data_file:
        return data_file.readlines()


def extract_transcription(transcription_path: str) -> List[TDF]:
    """Extract transcriptions from given file"""
    extracted_transcriptions = []

    with open(transcription_path) as transcription_file:
        # get rid of the first three useless headers
        transcriptions = transcription_file.readlines()[3:]

        for transcription in transcriptions:
            transcription_fields = transcription.split("\t")

            channel = int(transcription_fields[1])
            start = float(transcription_fields[2]) * 100
            end = float(transcription_fields[3]) * 100
            start = int(start)
            end = int(end)

            transcript = transcription_fields[7]

            cleaned_transcript = clean_transcription(transcript)

            # cleaned_transcript = transcript
            extracted_transcriptions.append(
                TDF(
                    channel=channel,
                    start=start,
                    end=end,
                    transcript=cleaned_transcript,
                )
            )

    return extracted_transcriptions


def concate_transcriptions_by_mapping_file(
    speech_folder: str,
    mapping_file_path: str,
    extracted_transcriptions: List[TDF],
) -> List[Data]:
    """return concated transcriptions from the given mapping file"""

    with open(mapping_file_path, "r", encoding="utf-8") as fisher_mapping_file:
        fisher_mapping = fisher_mapping_file.readlines()
        utterances = []

        for fisher_mapping_line in fisher_mapping:
            fisher_mapping_line = fisher_mapping_line.strip()
            fisher_mapping_line = fisher_mapping_line.split(" ")
            uid = fisher_mapping_line[0]
            need_to_be_concate_lines = fisher_mapping_line[1].split("_")
            need_to_be_concate_lines = list(map(int, need_to_be_concate_lines))

            selected_transcription = extracted_transcriptions[uid]

            # concate multiple transcripts
            if len(need_to_be_concate_lines) > 1:
                # index shift one is because id is count from 1 in file however, list start from 0
                concated_transcripts = selected_transcription[need_to_be_concate_lines[0] - 1: need_to_be_concate_lines[-1]]
                concated_transcripts = list(map(lambda tdf: tdf.transcript, concated_transcripts))
                concated_transcripts = " ".join(concated_transcripts)

                start = selected_transcription[need_to_be_concate_lines[0] - 1].start
                end = selected_transcription[need_to_be_concate_lines[-1] - 1].end
            else:
                concated_transcripts = selected_transcription[need_to_be_concate_lines[-1] - 1].transcript
                start = selected_transcription[need_to_be_concate_lines[-1] - 1].start
                end = selected_transcription[need_to_be_concate_lines[-1] - 1].end

            # clean up
            concated_transcripts = normalize_punctuation(concated_transcripts)
            concated_transcripts = es_normalizer.normalize(concated_transcripts)

            channel = selected_transcription[need_to_be_concate_lines[0] - 1].channel
            channel_symbol = "B" if channel == 1 else "A"
            uttrance_id = f"{uid}-{channel_symbol}-{start:06d}-{end:06d}"

            utterances.append(
                Data(
                    uid=uttrance_id,
                    transcription=concated_transcripts,
                    wav=f"{speech_folder}/{uid}.sph {channel} {start} {end}",
                    duration=(end - start) / 100,
                )
            )

        return utterances


def segment_audio(
    audio_path: str,
    channel: int,
    start: int,
    end: int,
    save_path: str,
    sample_rate: int = 16000,
    device: str = "cpu"
):
    """segment and resample audio"""

    start = int(start / 100 * 8000)
    end = int(end / 100 * 8000)
    num_frames = end - start

    data, _ = torchaudio.load(audio_path, frame_offset=start, num_frames=num_frames)

    resampler = Resample(orig_freq=8000, new_freq=sample_rate).to(device=device)

    data = resampler(data)
    data = torch.unsqueeze(data[channel], 0)

    torchaudio.save(save_path, src=data, sample_rate=sample_rate)


def get_transcription_files_by_dataset(dataset: str, transcription_folder: str) -> List[str]:
    """return paths of transcriptions from the given data set and the path of all of transcriptions"""
    train_set = get_data_list(f"splits/{dataset}")
    transcription_train_set = list(map(lambda path: path.split(".")[0], train_set))
    transcription_train_set = list(map(lambda path: f"{path}.tdf", transcription_train_set))

    transcription_files = get_all_files(transcription_folder, match_or=transcription_train_set)

    return transcription_files


def get_translations_from_path(translation_path: str) -> List[str]:
    """"return translations from the given path"""
    extracted_translations = []
    with open(translation_path, "rb") as translations_file:
        original_translations = translations_file.readlines()

        for translation in original_translations:
            translation = translation.replace(b"\r", b"")
            translation = translation.decode("utf-8")

            translation = clean_translation(translation)
            translation = normalize_punctuation(translation)
            translation = en_normalizer.normalize(translation)
            translation = remove_punctuation(translation)
            translation = en_tokenizer.tokenize(translation)

            translation = " ".join(translation)
            extracted_translations.append(translation)

    return extracted_translations


def insert_translation_into_existing_dataset(data: List[Data], translations: List[str]) -> List[Data]:
    """insert corresponding translation to given data"""

    for index in range(len(data)):
        corresponding_translation = translations[index]
        data[index].translations.append(corresponding_translation)

    return data


def download_translations(path: str):
    repo = "https://github.com/joshua-decoder/fisher-callhome-corpus.git"

    if not os.path.isdir(path):
        logger.info(
            f"Translation file not found. Downloading from {repo}."
        )
        subprocess.run(["git", "clone", repo])
        subprocess.run(["mv", "fisher-callhome-corpus", f"{path}"])


if __name__ == "__main__":
    data_folder = "PATH_TO_YOUR_DATA"
    save_folder = "data"
    device = "cuda:0"

    prepare_fisher_callhome_spanish(data_folder, save_folder, device=device)
