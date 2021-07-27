import logging
import os
from dataclasses import dataclass
from typing import Dict, List

import re
import json

logger = logging.getLogger(__name__)


@dataclass
class Transcription:
    id: str
    text: str


@dataclass
class SegmentInfo:
    wav: str
    start: float
    end: float


@dataclass
class Data:
    wav: str
    start: float
    duration: float
    transcription: str


def prepare_matbn(
    dataset_folder: str, save_folder: str, skip_prep: bool = False
):
    if skip_prep:
        return

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    wav_folder = os.path.join(dataset_folder, "wav")
    data_folder = os.path.join(dataset_folder, "data")

    if check_folders_exist(wav_folder, data_folder) is not True:
        logger.error(
            "the folder wav or data does not exist (it is expected in the "
            "MATBN dataset)"
        )

    splits = ["dev", "eval", "test", "train"]

    for split in splits:
        split_data_folder = os.path.join(data_folder, split)
        split_wav_folder = os.path.join(wav_folder, split)
        if split == "eval":
            split_wav_folder = os.path.join(wav_folder, "test")
        transcriptions_path = os.path.join(split_data_folder, "text")
        segments_path = os.path.join(split_data_folder, "segments")

        segments_info = extract_segments_info(segments_path)
        transcriptions = extract_transcriptions(transcriptions_path)

        useful_transcriptions = remove_useless_transcripts(transcriptions)

        concanated_data = concat_segments_info_and_transcriptions(
            segments_info, useful_transcriptions
        )

        for key, data in concanated_data.items():
            concanated_data[key].wav = os.path.join(
                split_wav_folder, f"{data.wav}.wav"
            )

        save_path = os.path.join(save_folder, f"{split}.json")

        with open(save_path, "w", encoding="utf-8") as save_file:
            json.dump(concanated_data, save_file, indent=2)


def check_folders_exist(*folders) -> bool:
    for folder in folders:
        if not os.path.exists(folder):
            return False
    return True


def extract_segments_info(segments_path: str) -> Dict[str, SegmentInfo]:
    segments_info: Dict[str, SegmentInfo] = {}
    with open(segments_path, "r", encoding="utf-8") as segments_file:
        segments_file_lines = segments_file.readlines()
        for segments_file_line in segments_file_lines:
            id, wav, start, end = segments_file_line.split()
            segments_info[id] = SegmentInfo(wav, float(start), float(end))
    return segments_info


def extract_transcriptions(transcriptions_path: str) -> List[Transcription]:
    transcriptions: List[Transcription] = []
    with open(
        transcriptions_path, "r", encoding="utf-8"
    ) as transcriptions_file:
        transcriptions_file_lines = transcriptions_file.readlines()
        for transcriptions_file_line in transcriptions_file_lines:
            split_line = transcriptions_file_line.split()
            transcriptions.append(
                Transcription(id=split_line[0], text=" ".join(split_line[1:]))
            )
    return transcriptions


def remove_useless_transcripts(
    transcriptions: List[Transcription], keep_unk=False
) -> List[Transcription]:
    useful_transcripts = []

    check_useability_regex = r"\w+\b(?<!\bUNK)"
    if keep_unk:
        check_useability_regex = r"\w"

    for transcription in transcriptions:
        useless = bool(re.search(check_useability_regex, transcription))
        if not useless:
            useful_transcripts.append(transcription)

    return useful_transcripts


def concat_segments_info_and_transcriptions(
    segments_info: Dict[str, SegmentInfo], transcriptions: List[Transcription]
) -> Dict[str, Data]:
    concatenate_data: Dict[str, Data] = {}

    for transcription in transcriptions:
        segment_info = segments_info[transcription.id]
        concatenate_data[transcription.id] = Data(
            segment_info.wav,
            segment_info.start,
            segment_info.end - segment_info.start,
            transcription.text,
        )

    return concatenate_data
